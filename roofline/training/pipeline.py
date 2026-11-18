# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Pipeline-parallel schedules for roofline estimation.

A schedule fixes, for each of the P pipeline devices, the order of forward ("F") and backward ("B")
passes of the m micro-batches (and of the v model chunks per device for interleaved schedules).
With T = one micro-batch's forward + backward through one full stage, every schedule here keeps
each device busy for m * T and differs in how long the step lasts and how many micro-batches'
activations a device holds at once:

    schedule                         step / T            bubble = idle / step       held (device 0)
    naive (one micro-batch at a time)  m * P               (P - 1) / P                1
    GPipe (Huang et al., 2019)         m + P - 1           (P - 1) / (m + P - 1)      m
    1F1B (Narayanan et al., 2021)      m + P - 1           (P - 1) / (m + P - 1)      P
    interleaved 1F1B (same paper)      m + (P - 1) / v     (P - 1) / (v m + P - 1)    simulated

`simulate_schedule` plays any order out op by op on a common clock and checks these.
"""

from __future__ import annotations
from typing import List, Tuple

Op = Tuple[str, int, int]  # ("F" | "B", micro-batch, chunk)


class PipelineSchedule:
    """Base class: per-device op order, step length and bubble of one synchronous step."""

    name = "base"
    interleave = 1  # model chunks per device

    def order(self, num_stages: int, num_microbatches: int) -> List[List[Op]]:
        """Ordered ops of each device."""
        raise NotImplementedError

    def slots(self, num_stages: int, num_microbatches: int) -> float:
        """Step length in units of T (one micro-batch's forward + backward through one stage)."""
        raise NotImplementedError

    def bubble(self, num_stages: int, num_microbatches: int) -> float:
        """Idle fraction of the step: (slots - m) / slots."""
        s = self.slots(num_stages, num_microbatches)
        return (s - num_microbatches) / s if s > 0 else 0.0

    def peak_activations(self, num_stages: int, num_microbatches: int) -> List[float]:
        """Peak micro-batches (of a full stage) whose activations each device holds."""
        return simulate_schedule(self, num_stages, num_microbatches)[1]

    def validate(self, num_stages: int, num_microbatches: int, blocks_per_stage: int) -> str:
        """Empty string if the schedule applies, else the reason it does not."""
        return ""

    def __repr__(self) -> str:
        return self.name


class NaiveSchedule(PipelineSchedule):
    """One micro-batch at a time: forward through all stages, backward through all stages, next."""

    name = "naive"

    def order(self, num_stages, num_microbatches):
        return [[op for j in range(num_microbatches) for op in (("F", j, 0), ("B", j, 0))]
                for _ in range(num_stages)]

    def slots(self, num_stages, num_microbatches):
        return num_microbatches * num_stages

    def peak_activations(self, num_stages, num_microbatches):
        return [1.0] * num_stages


class GPipeSchedule(PipelineSchedule):
    """All forwards, then all backwards (Huang et al., 2019, GPipe)."""

    name = "gpipe"

    def order(self, num_stages, num_microbatches):
        m = num_microbatches
        return [[("F", j, 0) for j in range(m)] + [("B", j, 0) for j in range(m)] for _ in range(num_stages)]

    def slots(self, num_stages, num_microbatches):
        return num_microbatches + num_stages - 1

    def peak_activations(self, num_stages, num_microbatches):
        return [float(num_microbatches)] * num_stages


class OneFOneBSchedule(PipelineSchedule):
    """1F1B: P - d - 1 warm-up forwards on device d, then one forward and one backward alternately."""

    name = "1f1b"

    def order(self, num_stages, num_microbatches):
        P, m = num_stages, num_microbatches
        orders = []
        for d in range(P):
            warmup = min(P - d - 1, m)
            seq = [("F", j, 0) for j in range(warmup)]
            for j in range(m - warmup):
                seq += [("F", warmup + j, 0), ("B", j, 0)]
            seq += [("B", j, 0) for j in range(m - warmup, m)]
            orders.append(seq)
        return orders

    def slots(self, num_stages, num_microbatches):
        return num_microbatches + num_stages - 1

    def peak_activations(self, num_stages, num_microbatches):
        return [float(min(num_stages - d, num_microbatches)) for d in range(num_stages)]


class InterleavedOneFOneBSchedule(PipelineSchedule):
    """Interleaved 1F1B (Narayanan et al., 2021, Megatron-LM): device d holds chunks d, P + d, ...

    Megatron's order: warm-up, then one forward and one backward alternately, then cool-down;
    micro-batches are taken in groups of P, so m must be a multiple of P.
    """

    name = "interleaved"

    def __init__(self, interleave: int = 2):
        self.interleave = interleave

    def order(self, num_stages, num_microbatches):
        P, m, v = num_stages, num_microbatches, self.interleave
        forwards = [(g * P + j, c) for g in range(m // P) for c in range(v) for j in range(P)]
        backwards = [(j, v - 1 - c) for (j, c) in forwards]
        orders = []
        for d in range(P):
            warmup = min((P - d - 1) * 2 + (v - 1) * P, m * v)
            seq = [("F",) + f for f in forwards[:warmup]]
            for k in range(m * v - warmup):
                seq += [("F",) + forwards[warmup + k], ("B",) + backwards[k]]
            seq += [("B",) + b for b in backwards[m * v - warmup:]]
            orders.append(seq)
        return orders

    def slots(self, num_stages, num_microbatches):
        return num_microbatches + (num_stages - 1) / self.interleave

    def peak_activations(self, num_stages, num_microbatches):
        # set by the warm-up: the same for any m >= 4P, so simulate at most 4P micro-batches
        m = min(num_microbatches, 4 * num_stages)
        return simulate_schedule(self, num_stages, m)[1]

    def validate(self, num_stages, num_microbatches, blocks_per_stage):
        if num_microbatches % num_stages:
            return f"interleaved 1F1B needs num_microbatches ({num_microbatches}) to be a multiple of pp ({num_stages})"
        if blocks_per_stage % self.interleave:
            return f"blocks per stage ({blocks_per_stage}) not divisible by interleave ({self.interleave})"
        return ""

    def __repr__(self):
        return f"interleaved(v={self.interleave})"


def make_schedule(name: str, interleave: int = 1) -> PipelineSchedule:
    """Schedule by name: naive, gpipe, 1f1b, interleaved (with interleave = v chunks per device)."""
    if name == "interleaved" or (name == "1f1b" and interleave > 1):
        if interleave < 2:
            raise ValueError("interleaved 1F1B needs interleave >= 2 (interleave = 1 is plain 1f1b)")
        return InterleavedOneFOneBSchedule(interleave)
    schedules = {"naive": NaiveSchedule, "gpipe": GPipeSchedule, "1f1b": OneFOneBSchedule}
    if name not in schedules:
        raise ValueError(f"Unknown schedule: {name}. Available: naive, gpipe, 1f1b, interleaved")
    return schedules[name]()


def simulate_schedule(
    schedule: PipelineSchedule,
    num_stages: int,
    num_microbatches: int,
    t_forward: float = 1.0,
    t_backward: float = 2.0,
) -> Tuple[float, List[float]]:
    """Play a schedule out op by op on a common clock.

    Each device runs its ops in order; an op starts when the device is free and its input is ready
    (forward: the previous virtual stage's forward of the same micro-batch; backward: the next
    virtual stage's backward, or its own forward on the last stage). Hand-offs take no time.

    Args:
        schedule: the schedule
        num_stages: P
        num_microbatches: m
        t_forward, t_backward: forward / backward time of one full stage (any unit)

    Returns:
        (bubble fraction, peak activations held per device in full-stage micro-batches)
    """
    P, m, v = num_stages, num_microbatches, schedule.interleave
    if P <= 1:
        return 0.0, [1.0]
    orders = schedule.order(P, m)
    last = P * v - 1
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
                    dep = ("B", mb, stage + 1) if stage < last else ("F", mb, stage)
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
        raise RuntimeError(f"{schedule} deadlocked")
    return 1.0 - m * (t_forward + t_backward) / max(free), peak
