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

Two options apply on top of a schedule (`make_schedule(..., pairing=, wgrad_deferral=)`); with
either one the step time comes from `simulate_schedule` with each pass's resource times:

- pairing: each steady-state forward and backward (two different micro-batches) run as one
  unit, so one micro-batch's TP collectives run under the other's compute (DeepSeek-V3's
  DualPipe overlaps "a pair of individual forward and backward chunks", DeepSeek-AI, 2024).
  An op run alone waits for its own collectives. 1F1B and interleaved 1F1B only. A pair hands both outputs on when it ends,
  so a micro-batch crosses one device per pair in each direction; to keep every device busy,
  the warm-up has to shrink by two forwards per device and the last device needs one, as
  interleaved 1F1B's already does (+1 forward). For 1F1B that is 2 (P - d - 1) + 1 warm-up
  forwards instead of P - d - 1: nearly twice the activations on device 0.
- wgrad_deferral: the backward is split into the input gradient (B) and the weight gradient
  (W), and W runs later, in gaps it fits in (Zero Bubble Pipeline Parallelism, Qi et al., 2024).
  Between B and W a micro-batch keeps only the weight gradients' operands. No device may hold
  more activations than device 0 does without deferral (ZB-H1's limit): later devices, which
  hold fewer in 1F1B, hold up to that many; on 1F1B this gives ZB-H1's bubble, (P - 1)(F + B - W).
"""

from __future__ import annotations
import copy
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

Op = Tuple[str, int, int]  # ("F" | "B", micro-batch, chunk)


@dataclass(frozen=True)
class PassCost:
    """Device time of one pass of one micro-batch through one full stage, per resource."""

    flop: float = 0.0
    dram: float = 0.0
    fabric: float = 0.0  # overlaps the pass's own compute (ring attention shifts, stage hand-offs)
    exposed_fabric: float = 0.0  # collectives the pass waits for (TP), unless paired

    def __add__(self, o: "PassCost") -> "PassCost":
        return PassCost(self.flop + o.flop, self.dram + o.dram, self.fabric + o.fabric,
                        self.exposed_fabric + o.exposed_fabric)

    def __sub__(self, o: "PassCost") -> "PassCost":
        return self + o.scale(-1.0)

    def scale(self, k: float) -> "PassCost":
        return PassCost(self.flop * k, self.dram * k, self.fabric * k, self.exposed_fabric * k)

    def alone(self) -> float:
        """Time when run alone: its collectives are on its critical path."""
        return max(self.flop, self.dram, self.fabric) + self.exposed_fabric

    def overlapped(self) -> float:
        """Time when its collectives run under other work: the largest resource."""
        return max(self.flop, self.dram, self.fabric + self.exposed_fabric)


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
    pairing = False  # run adjacent forward / backward ops of two micro-batches as one unit
    wgrad_deferral = False  # split the backward into B and W, run W in idle time

    def order(self, num_stages: int, num_microbatches: int) -> List[List[Op]]:
        """Ordered ops of each device."""
        return [[op for unit in u for op in unit] for u in self.units(num_stages, num_microbatches)]

    def units(self, num_stages: int, num_microbatches: int) -> List[List[Tuple[Op, ...]]]:
        """Ordered run units of each device: one op, or a forward / backward pair with pairing."""
        return [[(op,) for op in o] for o in self.order(num_stages, num_microbatches)]

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
        opts = [o for o, on in (("pairing", self.pairing), ("wgrad deferral", self.wgrad_deferral)) if on]
        return f" + {', '.join(opts)}" if opts else ""

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

    def units(self, num_stages, num_microbatches):
        P, m = num_stages, num_microbatches
        orders = []
        for d in range(P):
            warmup = min(2 * (P - d - 1) + 1 if self.pairing else P - d - 1, m)
            seq = [(("F", j, 0),) for j in range(warmup)]
            for j in range(m - warmup):
                f, b = ("F", warmup + j, 0), ("B", j, 0)
                seq += [(f, b)] if self.pairing else [(f,), (b,)]
            seq += [(("B", j, 0),) for j in range(m - warmup, m)]
            orders.append(seq)
        return orders

    def slots(self, num_stages, num_microbatches):
        return num_microbatches + num_stages - 1

    def peak_activations(self, num_stages, num_microbatches):
        if self.pairing:
            return simulate_schedule(self, num_stages, num_microbatches)[1]
        return [float(min(num_stages - d, num_microbatches)) for d in range(num_stages)]


class InterleavedOneFOneBSchedule(PipelineSchedule):
    """Interleaved 1F1B (Narayanan et al., 2021, Megatron-LM): device d holds chunks d, P + d, ...

    Megatron's order: warm-up, then one forward and one backward alternately, then cool-down;
    micro-batches are taken in groups of P, so m must be a multiple of P.
    """

    name = "interleaved"

    def __init__(self, interleave: int = 2):
        self.interleave = interleave

    def units(self, num_stages, num_microbatches):
        P, m, v = num_stages, num_microbatches, self.interleave
        forwards = [(g * P + j, c) for g in range(m // P) for c in range(v) for j in range(P)]
        backwards = [(j, v - 1 - c) for (j, c) in forwards]
        orders = []
        for d in range(P):
            warmup = min((P - d - 1) * 2 + (v - 1) * P + self.pairing, m * v)
            seq = [(("F",) + f,) for f in forwards[:warmup]]
            for k in range(m * v - warmup):
                f, b = ("F",) + forwards[warmup + k], ("B",) + backwards[k]
                seq += [(f, b)] if self.pairing else [(f,), (b,)]
            seq += [(("B",) + b,) for b in backwards[m * v - warmup:]]
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


def make_schedule(name: str, interleave: int = 1, pairing: bool = False,
                  wgrad_deferral: bool = False) -> PipelineSchedule:
    """Schedule by name: naive, gpipe, 1f1b, interleaved (with interleave = v chunks per device).

    pairing, wgrad_deferral: see the module docstring.
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
    if pairing and name not in ("1f1b", "interleaved"):
        raise ValueError("pairing needs a steady state of alternating forwards and backwards: 1f1b or interleaved")
    schedule.pairing, schedule.wgrad_deferral = pairing, wgrad_deferral
    return schedule


def simulate_schedule(
    schedule: PipelineSchedule, num_stages: int, num_microbatches: int,
    costs: StageCosts = UNIT_COSTS, wgrad_cap: Optional[Sequence[float]] = None,
) -> SimResult:
    """Play a schedule out unit by unit on a common clock.

    Each device runs its units in order; a unit starts when the device is free and the inputs of
    its ops are ready (forward: the previous virtual stage's forward of the same micro-batch;
    backward: the next virtual stage's backward, or its own forward on the last stage); its
    outputs are ready when it ends. Hand-offs take no time beyond the fabric time in `costs`.
    An op alone takes PassCost.alone(), a pair PassCost.overlapped() of the two summed. With
    wgrad deferral, a device runs its oldest pending W when the W fits before its next unit's
    inputs are ready, or when the unit's forward would hold more than wgrad_cap[device]
    micro-batches (default on every device: the pipeline's peak without deferral, device 0's);
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

    def unit_time(unit) -> float:
        passes = [f if op[0] == "F" else b for op in unit]
        return passes[0].alone() if len(passes) == 1 else (passes[0] + passes[1]).overlapped()

    units = schedule.units(P, m)
    last = P * v - 1
    done: Dict[Tuple[str, int, int], float] = {}
    pos = [0] * P
    free = [0.0] * P
    busy = [0.0] * P
    live = [0.0] * P
    peak = [0.0] * P
    pending = [deque() for _ in range(P)]

    def ready_time(d: int, unit) -> float:
        t = free[d]
        for kind, mb, chunk in unit:
            stage = chunk * P + d
            if kind == "F":
                dep = ("F", mb, stage - 1) if stage > 0 else None
            else:
                dep = ("B", mb, stage + 1) if stage < last else ("F", mb, stage)
            if dep is not None:
                if dep not in done:
                    return float("inf")
                t = max(t, done[dep])
        return t

    def run_w(d: int) -> None:
        pending[d].popleft()
        free[d] += w.alone()
        busy[d] += w.alone()
        live[d] -= held

    # A device's actions change only its own timeline and the ready times of the ops it runs,
    # so each device can be advanced as soon as the inputs of its next unit are known.
    progress = True
    while progress:
        progress = False
        for d in range(P):
            while pos[d] < len(units[d]):
                unit = units[d][pos[d]]
                start = ready_time(d, unit)
                n_fwd = sum(op[0] == "F" for op in unit)
                over = split and live[d] + n_fwd / v > wgrad_cap[d] + 1e-9
                if pending[d] and (over or free[d] + w.alone() <= start + 1e-12 < float("inf")):
                    run_w(d)
                    progress = True
                    continue
                if start == float("inf"):
                    break  # wait until the inputs' time is known
                dt = unit_time(unit)
                free[d] = start + dt
                busy[d] += dt
                peak[d] = max(peak[d], live[d] + n_fwd / v)
                for kind, mb, chunk in unit:
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
    if any(pos[d] != len(units[d]) for d in range(P)):
        raise RuntimeError(f"{schedule} deadlocked")
    step = max(free)
    return SimResult(1.0 - sum(busy) / (P * step), peak, step)
