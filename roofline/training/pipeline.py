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

With `make_schedule(..., wgrad_deferral=True)` the backward is split into the input gradient (B)
and the weight gradient (W), and W runs later, in gaps it fits in (Zero Bubble Pipeline
Parallelism, Qi et al., 2024). Between B and W a micro-batch keeps only the weight gradients'
operands. No device may hold more activations than device 0 does without deferral (ZB-H1's
limit): later devices, which hold fewer in 1F1B, hold up to that many; on 1F1B this gives ZB-H1's
bubble, (P - 1)(F + B - W). The bubble and the activations held then come from
`simulate_schedule` with each pass's resource times; a pass takes the largest of its FLOP, DRAM
and fabric time.
"""

from __future__ import annotations
import copy
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

Op = Tuple[str, int, int]  # ("F" | "B", micro-batch, chunk)


@dataclass(frozen=True)
class PassCost:
    """Time of one pass of one micro-batch through one full stage, per resource."""

    flop: float = 0.0
    dram: float = 0.0
    fabric: float = 0.0

    def __add__(self, o: "PassCost") -> "PassCost":
        return PassCost(self.flop + o.flop, self.dram + o.dram, self.fabric + o.fabric)

    def __sub__(self, o: "PassCost") -> "PassCost":
        return self + o.scale(-1.0)

    def scale(self, k: float) -> "PassCost":
        return PassCost(self.flop * k, self.dram * k, self.fabric * k)

    def time(self) -> float:
        """Roofline time: the largest resource."""
        return max(self.flop, self.dram, self.fabric)


@dataclass(frozen=True)
class StageCosts:
    """Passes of one micro-batch through one full stage.

    backward includes wgrad; wgrad_held is what a micro-batch keeps from B to W, as a fraction
    of the activation it keeps from F to B.
    """

    forward: PassCost
    backward: PassCost
    wgrad: PassCost = PassCost()
    wgrad_held: float = 0.0


# F : B = 1 : 2, all compute; the bubble of a schedule without options does not depend on it
UNIT_COSTS = StageCosts(PassCost(flop=1.0), PassCost(flop=2.0), PassCost(flop=1.0), 0.0)


class SimResult(NamedTuple):
    bubble: float  # idle fraction of the step, averaged over devices
    peak: List[float]  # peak activations held per device, in full-stage micro-batches
    step: float  # step length, in the costs' time unit


class PipelineSchedule:
    """Base class: per-device op order, step length and bubble of one synchronous step."""

    name = "base"
    interleave = 1  # model chunks per device
    wgrad_deferral = False  # split the backward into B and W, run W in idle time

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

    def _options(self) -> str:
        return " + wgrad deferral" if self.wgrad_deferral else ""

    def __repr__(self) -> str:
        return self.name + self._options()


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
        return f"interleaved(v={self.interleave})" + self._options()


def make_schedule(name: str, interleave: int = 1, wgrad_deferral: bool = False) -> PipelineSchedule:
    """Schedule by name: naive, gpipe, 1f1b, interleaved (with interleave = v chunks per device).

    wgrad_deferral: see the module docstring.
    """
    if name == "interleaved":
        if interleave < 2:
            raise ValueError("interleaved 1F1B needs interleave >= 2 (interleave = 1 is plain 1f1b)")
        schedule = InterleavedOneFOneBSchedule(interleave)
    else:
        schedules = {"naive": NaiveSchedule, "gpipe": GPipeSchedule, "1f1b": OneFOneBSchedule}
        if name not in schedules:
            raise ValueError(f"Unknown schedule: {name}. Available: naive, gpipe, 1f1b, interleaved")
        schedule = schedules[name]()
    schedule.wgrad_deferral = wgrad_deferral
    return schedule


def simulate_schedule(
    schedule: PipelineSchedule, num_stages: int, num_microbatches: int,
    costs: StageCosts = UNIT_COSTS, wgrad_cap: Optional[Sequence[float]] = None,
) -> SimResult:
    """Play a schedule out op by op on a common clock.

    Each device runs its ops in order; an op starts when the device is free and its input is ready
    (forward: the previous virtual stage's forward of the same micro-batch; backward: the next
    virtual stage's backward, or its own forward on the last stage) and takes PassCost.time().
    Hand-offs take no time beyond the fabric time in `costs`. With wgrad deferral, a device runs
    its oldest pending W when the W fits before its next op's input is ready, or when a forward
    would hold more than wgrad_cap[device] micro-batches (default on every device: the pipeline's peak without deferral, device 0's);
    a backward that would itself go above the cap runs its W right after it. The rest run at
    the end.

    Args:
        schedule: the schedule
        num_stages: P
        num_microbatches: m
        costs: pass costs of one full stage, the same on every device (default: F = 1, B = 2)
        wgrad_cap: per-device limit on activations held, with wgrad deferral

    Returns:
        SimResult(bubble, peak activations per device in full-stage micro-batches, step)
    """
    P, m = num_stages, num_microbatches
    v = schedule.interleave
    split = schedule.wgrad_deferral
    f = costs.forward.scale(1.0 / v)
    w = costs.wgrad.scale(1.0 / v) if split else PassCost()
    b = costs.backward.scale(1.0 / v) - w
    held = costs.wgrad_held / v if split else 0.0
    if split and wgrad_cap is None:
        base = copy.copy(schedule)
        base.wgrad_deferral = False
        wgrad_cap = [max(simulate_schedule(base, P, m, costs).peak)] * P

    orders = schedule.order(P, m)
    last = P * v - 1
    done: Dict[Tuple[str, int, int], float] = {}
    pos = [0] * P
    free = [0.0] * P
    busy = [0.0] * P
    live = [0.0] * P
    peak = [0.0] * P
    pending = [deque() for _ in range(P)]

    def ready_time(d: int, op: Op) -> float:
        kind, mb, chunk = op
        stage = chunk * P + d
        if kind == "F":
            dep = ("F", mb, stage - 1) if stage > 0 else None
        else:
            dep = ("B", mb, stage + 1) if stage < last else ("F", mb, stage)
        if dep is None:
            return free[d]
        return max(free[d], done[dep]) if dep in done else float("inf")

    def run_w(d: int) -> None:
        pending[d].popleft()
        free[d] += w.time()
        busy[d] += w.time()
        live[d] -= held

    # A device's actions change only its own timeline and the ready times of the ops it runs,
    # so each device can be advanced as soon as the input of its next op is known.
    progress = True
    while progress:
        progress = False
        for d in range(P):
            while pos[d] < len(orders[d]):
                kind, mb, chunk = orders[d][pos[d]]
                start = ready_time(d, orders[d][pos[d]])
                over = split and kind == "F" and live[d] + 1.0 / v > wgrad_cap[d] + 1e-9
                if pending[d] and (over or free[d] + w.time() <= start + 1e-12 < float("inf")):
                    run_w(d)
                    progress = True
                    continue
                if start == float("inf"):
                    break  # wait until the input's time is known
                dt = (f if kind == "F" else b).time()
                free[d] = start + dt
                busy[d] += dt
                done[(kind, mb, chunk * P + d)] = free[d]
                if kind == "F":
                    live[d] += 1.0 / v
                else:
                    live[d] += held - 1.0 / v
                    if split:
                        pending[d].append((mb, chunk))
                    while split and pending[d] and live[d] > wgrad_cap[d] + 1e-9:
                        run_w(d)  # no room to defer: the W runs as part of the backward
                peak[d] = max(peak[d], live[d])
                pos[d] += 1
                progress = True
    for d in range(P):
        while pending[d]:
            run_w(d)
    if any(pos[d] != len(orders[d]) for d in range(P)):
        raise RuntimeError(f"{schedule} deadlocked")
    step = max(free)
    return SimResult(1.0 - sum(busy) / (P * step), peak, step)
