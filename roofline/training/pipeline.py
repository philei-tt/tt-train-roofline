# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Pipeline-parallel schedule model for roofline estimation.

Interleaved 1F1B (Narayanan et al., 2021, "Efficient Large-Scale Language Model Training on GPU
Clusters Using Megatron-LM"): every device holds v model chunks; device d holds chunks
d, P + d, 2P + d, ... With v = 1 this is plain 1F1B.

The schedule is simulated op by op in Megatron's order (warm-up, steady one-forward-one-backward,
cool-down; micro-batches taken in groups of P) to get the peak number of micro-batches whose
activations a device holds. The bubble has the closed form (P - 1) / (v * m + P - 1).
"""

from __future__ import annotations
from typing import List, Tuple


def pipeline_bubble(num_stages: int, num_microbatches: int, interleave: int = 1) -> float:
    """Idle fraction of a synchronous (interleaved) 1F1B step: (P - 1) / (v * m + P - 1)."""
    if num_stages <= 1:
        return 0.0
    return (num_stages - 1) / (interleave * num_microbatches + num_stages - 1)


def interleaved_1f1b_order(
    num_stages: int, num_microbatches: int, interleave: int
) -> List[List[Tuple[str, int, int]]]:
    """Per-device op order of Megatron's interleaved 1F1B: lists of ("F"|"B", microbatch, chunk)."""
    P, m, v = num_stages, num_microbatches, interleave
    if m % P != 0:
        raise ValueError(f"interleaved 1F1B needs num_microbatches ({m}) to be a multiple of num_stages ({P})")
    forwards = [(g * P + j, c) for g in range(m // P) for c in range(v) for j in range(P)]
    backwards = [(j, v - 1 - c) for (j, c) in forwards]
    orders = []
    for d in range(P):
        warmup = min(P - d - 1, m) if v == 1 else min((P - d - 1) * 2 + (v - 1) * P, m * v)
        seq = [("F",) + f for f in forwards[:warmup]]
        fi, bi = warmup, 0
        while fi < m * v:
            seq.append(("F",) + forwards[fi]); fi += 1
            seq.append(("B",) + backwards[bi]); bi += 1
        while bi < m * v:
            seq.append(("B",) + backwards[bi]); bi += 1
        orders.append(seq)
    return orders


def simulate_pipeline(
    num_stages: int,
    num_microbatches: int,
    interleave: int = 1,
    t_forward: float = 1.0,
    t_backward: float = 2.0,
) -> Tuple[float, List[float]]:
    """Simulate the interleaved 1F1B schedule.

    Args:
        num_stages: P, pipeline devices
        num_microbatches: m, micro-batches per optimizer step (multiple of P)
        interleave: v, model chunks per device (1 = plain 1F1B)
        t_forward, t_backward: time of one full-stage forward / backward (any unit)

    Returns:
        (bubble fraction, peak activations held per device, in full-stage micro-batches)
    """
    P, m, v = num_stages, num_microbatches, interleave
    if P <= 1:
        return 0.0, [1.0]
    orders = interleaved_1f1b_order(P, m, v)
    V = P * v
    done = {}
    pos = [0] * P
    free = [0.0] * P
    live = [0.0] * P
    peak = [0.0] * P
    progress = True
    while progress:
        progress = False
        for d in range(P):
            while pos[d] < len(orders[d]):
                kind, mb, chunk = orders[d][pos[d]]
                stage = chunk * P + d
                if kind == "F":
                    dep = ("F", mb, stage - 1) if stage > 0 else None
                else:
                    dep = ("B", mb, stage + 1) if stage < V - 1 else ("F", mb, stage)
                if dep is not None and dep not in done:
                    break
                start = max(free[d], done[dep] if dep is not None else 0.0)
                free[d] = start + (t_forward if kind == "F" else t_backward) / v
                done[(kind, mb, stage)] = free[d]
                live[d] += (1.0 if kind == "F" else -1.0) / v
                peak[d] = max(peak[d], live[d])
                pos[d] += 1
                progress = True
    if any(pos[d] != len(orders[d]) for d in range(P)):
        raise RuntimeError("pipeline schedule deadlocked")
    step = max(free)
    return 1.0 - m * (t_forward + t_backward) / step, peak
