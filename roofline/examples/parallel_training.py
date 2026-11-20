# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Roofline analysis for Llama training with pipeline, tensor, sequence and context parallelism.

Combines, on `chips = pp * tp * cp` devices:
  - PP: a pipeline schedule (training/pipeline.py), optional vocabulary parallelism
  - TP: tensor parallelism inside a stage, optionally with sequence parallelism (--sp)
  - CP: context parallelism inside a stage (ring attention, load-balanced split)
  - activation recompute (--recompute, the memory-efficient runner)

Every distinct pipeline stage (first, middle, last, and each block count when the layer split is
uneven) is run for one micro-batch in its own RooflineContext. A stage's time per micro-batch is a
resource-level roofline (FLOP, DRAM and fabric time each summed, the largest sets the time); the
schedule is then played out with each device's own forward and backward time; see the README.

Examples (from the repository root):
    python3 -m roofline.examples.parallel_training --model llama-70b -s 8192 -m 2048 \\
        --pp 16 --tp 2 --cp 4 --recompute --attn-bwd-kv-block 512
    python3 -m roofline.examples.parallel_training --model llama-70b -s 8192 -m 2048 \\
        --sweep 128 --sp --attn-bwd-kv-block 512
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from roofline import (
    HARDWARE_PRESETS,
    DataType,
    MathFidelity,
    MockAdamW,
    MockCrossEntropyLossOp,
    MockDistributedLlama,
    MockTensor,
    RooflineContext,
    TensorLabel,
    mock_clip_grad_norm,
    send_recv_roofline,
)
from roofline.examples.training import MODEL_PRESETS, ModelType, distributed_llama_config
from roofline.training.distributed import synchronize_gradients
from roofline.training.pipeline import make_schedule, simulate_schedule

MICROBATCH_PHASES = ("forward", "backward")
OPTIMIZER_PHASES = ("optimizer", "grad_clip")
SYNC_PHASES = ("ccl",)  # gradient all-reduce across the CP group


@dataclass
class StageCost:
    """Roofline cost of one pipeline stage (times in ms, memory in bytes)."""

    mb_flop_ms: float  # one micro-batch, forward + backward
    mb_dram_ms: float
    mb_fabric_ms: float
    mb_serial_ms: float  # sum over ops of max(compute, DRAM, network)
    mb_forward_ms: float  # forward alone: max(FLOP, DRAM, fabric) of the forward phase
    mb_backward_ms: float
    mb_forward_serial_ms: float
    mb_backward_serial_ms: float
    opt_flop_ms: float  # optimizer + gradient clipping, once per step
    opt_dram_ms: float
    opt_serial_ms: float
    sync_fabric_ms: float  # CP gradient all-reduce, once per step
    sync_serial_ms: float
    activation_bytes: int  # held from forward to backward, one micro-batch
    state_bytes: int  # parameters + gradients + optimizer state
    peak_bytes: int  # peak of the whole one-micro-batch run (incl. backward transients)
    flops_per_mb: int  # one micro-batch on one device, forward + backward
    embedding_flops_per_mb: int
    output_bytes: int  # activation sent to the next stage, per device


def run_stage(
    hw, preset: dict, seq_len: int, tp: int, cp: int, pp: int, stage_index: int, stage_blocks: int,
    sequence_parallel: bool, vocab_parallel: bool, recompute: bool, attn_bwd_kv_block: Optional[int],
) -> StageCost:
    """Run one micro-batch (one sequence) through one pipeline stage of `stage_blocks` blocks,
    then the optimizer step."""
    # Objects from earlier runs never take part in this run's cycles: freeze them so the mock
    # autograd's frequent gc.collect() calls only scan this run's objects.
    gc.collect()
    gc.freeze()
    try:
        return _run_stage(hw, preset, seq_len, tp, cp, pp, stage_index, stage_blocks, sequence_parallel,
                          vocab_parallel, recompute, attn_bwd_kv_block)
    finally:
        gc.unfreeze()


def _run_stage(hw, preset, seq_len, tp, cp, pp, stage_index, stage_blocks, sequence_parallel, vocab_parallel,
               recompute, attn_bwd_kv_block) -> StageCost:
    ctx = RooflineContext(hw, attn_bwd_kv_block=attn_bwd_kv_block)
    config = distributed_llama_config(
        preset, tp_size=tp, cp_size=cp, sequence_parallel=sequence_parallel,
        num_stages=pp, stage_index=stage_index, stage_num_blocks=stage_blocks, vocab_parallel=vocab_parallel,
    )
    stage = MockDistributedLlama(config, runner="mem_eff" if recompute else "default")
    params = stage.parameters()
    optimizer = MockAdamW(params)
    local_seq = seq_len // cp
    mem_before, _ = ctx.memory_tracker.current_memory()

    if stage.is_first:
        x = MockTensor((1, 1, 1, local_seq), dtype=DataType.BFLOAT16, requires_grad=False,
                       label=TensorLabel.ACTIVATION, name="indices")
    else:
        rows = local_seq // (tp if sequence_parallel else 1)
        x = MockTensor((1, 1, rows, config.embedding_dim), dtype=DataType.BFLOAT16, requires_grad=True,
                       label=TensorLabel.ACTIVATION, name="received_activation")
    out, logits = stage.forward_stage(ctx, x)
    roots: List[MockTensor] = []
    if logits is not None:
        targets = MockTensor((1, 1, 1, local_seq), dtype=DataType.BFLOAT16, requires_grad=False,
                             label=TensorLabel.ACTIVATION, name="targets")
        roots.append(MockCrossEntropyLossOp.apply(ctx, logits, targets))
    if not stage.is_last:
        roots.append(out)  # the gradient arrives from the next stage
    mem_after_forward, _ = ctx.memory_tracker.current_memory()
    output_bytes = out.bytes()

    for root in roots:
        root.backward(ctx, retain_graph=False)
    synchronize_gradients(ctx, params, cp)  # weights are replicated across the CP group
    optimizer.step(ctx)
    mock_clip_grad_norm(ctx, params, max_norm=1.0)
    del x, out, logits, roots  # release the micro-batch's tensors before reading the state
    gc.collect()
    state_bytes, _ = ctx.memory_tracker.current_memory()
    peak_bytes, _ = ctx.memory_tracker.peak_memory()
    ctx.disable_memory_tracking()

    def serial_ms(phases):
        return sum(e.theoretical_time_ns for e in ctx.estimates if e.phase in phases) / 1e6

    def overlapped_ms(phases):
        return ctx.overlapped_time_ns(phases) / 1e6

    return StageCost(
        mb_flop_ms=ctx.compute_time_ns(MICROBATCH_PHASES) / 1e6,
        mb_dram_ms=ctx.dram_time_ns(MICROBATCH_PHASES) / 1e6,
        mb_fabric_ms=ctx.network_time_ns(MICROBATCH_PHASES) / 1e6,
        mb_serial_ms=serial_ms(MICROBATCH_PHASES),
        mb_forward_ms=overlapped_ms(("forward",)),
        mb_backward_ms=overlapped_ms(("backward",)),
        mb_forward_serial_ms=serial_ms(("forward",)),
        mb_backward_serial_ms=serial_ms(("backward",)),
        opt_flop_ms=ctx.compute_time_ns(OPTIMIZER_PHASES) / 1e6,
        opt_dram_ms=ctx.dram_time_ns(OPTIMIZER_PHASES + SYNC_PHASES) / 1e6,
        opt_serial_ms=serial_ms(OPTIMIZER_PHASES),
        sync_fabric_ms=ctx.network_time_ns(SYNC_PHASES) / 1e6,
        sync_serial_ms=serial_ms(SYNC_PHASES),
        activation_bytes=mem_after_forward - mem_before,
        state_bytes=state_bytes,
        peak_bytes=peak_bytes,
        flops_per_mb=ctx.forward_flops() + ctx.backward_flops(),
        embedding_flops_per_mb=sum(e.total_flops for e in ctx.estimates if e.operation.startswith("Embedding.")),
        output_bytes=output_bytes,
    )


def run_parallel_roofline(
    model_name: str = "llama-70b",
    seq_len: int = 8192,
    num_microbatches: int = 64,
    hardware: str = "bh_quad",
    pp: int = 1,
    tp: int = 1,
    cp: int = 1,
    schedule: str = "1f1b",
    interleave: int = 1,
    sequence_parallel: bool = False,
    vocab_parallel: bool = False,
    recompute: bool = False,
    cut_links: int = 8,
    attn_bwd_kv_block: Optional[int] = None,
    num_links: Optional[int] = None,
    topology: Optional[str] = None,
    blocks_per_stage: Optional[Sequence[int]] = None,
    stage_cache: Optional[Dict[Any, StageCost]] = None,
) -> Dict[str, Any]:
    """Roofline of one optimizer step of `num_microbatches` sequences on pp * tp * cp chips.

    Args:
        model_name: Llama preset from training.MODEL_PRESETS
        seq_len: Tokens per sequence (one sequence per micro-batch)
        num_microbatches: m, sequences per optimizer step
        hardware: name in HARDWARE_PRESETS (default: bh_quad)
        pp, tp, cp: pipeline, tensor and context parallel degrees
        schedule: pipeline schedule: "naive", "gpipe", "1f1b" or "interleaved"
        interleave: v, model chunks per pipeline device (interleaved schedule only)
        sequence_parallel: Megatron sequence parallelism inside TP
        vocab_parallel: split the LM head across all pipeline stages
        recompute: recompute each block's forward in backward (memory-efficient runner)
        cut_links: Ethernet links (per direction) across the cut between two neighbouring stages.
            Quad (32 x 4 torus), stages laid along the 32-chip axis: one row of 4 chip pairs,
            2 links each = 8.
        attn_bwd_kv_block: see fused_attention_roofline(bwd_kv_block)
        num_links, topology: override the hardware's links per neighbour and topology
        blocks_per_stage: transformer blocks on each pipeline device (as tt-train's
            PipelineParallelConfig.blocks_per_rank); default an even split, the remainder on the
            first devices (the last one holds the LM head)
        stage_cache: optional dict reused across calls (stage runs do not depend on m or the schedule)

    Returns:
        Result dict, or {"error": str}
    """
    preset = MODEL_PRESETS.get(model_name)
    if preset is None or preset["model_type"] != ModelType.LLAMA:
        return {"error": f"Unknown Llama preset: {model_name}"}
    if hardware not in HARDWARE_PRESETS:
        return {"error": f"Unknown hardware: {hardware}. Available: {', '.join(HARDWARE_PRESETS)}"}
    hw = HARDWARE_PRESETS[hardware]
    overrides = {k: v for k, v in (("num_links", num_links), ("topology", topology)) if v is not None}
    if overrides:
        hw = dataclasses.replace(hw, **overrides)
    chips = pp * tp * cp
    L = preset["num_blocks"]
    try:
        sched = make_schedule(schedule, interleave)
    except ValueError as exc:
        return {"error": str(exc)}
    blocks = list(blocks_per_stage) if blocks_per_stage is not None else even_split(L, pp)
    schedule_error = sched.validate(pp, num_microbatches, blocks) if pp > 1 else ""
    checks = [
        (chips <= hw.chips_per_galaxy, f"pp*tp*cp = {chips} > {hw.chips_per_galaxy} chips of {hw.name}"),
        (len(blocks) == pp, f"blocks_per_stage has {len(blocks)} entries, pp is {pp}"),
        (sum(blocks) == L and min(blocks) >= 1,
         f"blocks_per_stage {blocks} must be >= 1 each and sum to num_blocks ({L})"),
        (not schedule_error, schedule_error),
        (preset["num_groups"] % tp == 0, f"num_groups ({preset['num_groups']}) not divisible by tp ({tp})"),
        (seq_len % (2 * cp * (tp if sequence_parallel else 1)) == 0, "seq_len not divisible by 2 * cp (* tp with SP)"),
    ]
    for ok, msg in checks:
        if not ok:
            return {"error": msg}

    cache = stage_cache if stage_cache is not None else {}

    def role(d: int) -> int:
        # first, middle or last: what a stage holds besides its blocks
        return 0 if d == 0 else (pp - 1 if d == pp - 1 else 1)

    def stage_cost(d: int, with_recompute: bool) -> StageCost:
        key = (model_name, seq_len, hardware, hw.num_links, hw.topology, pp, tp, cp, role(d), blocks[d],
               sequence_parallel, vocab_parallel, with_recompute, attn_bwd_kv_block)
        if key not in cache:
            cache[key] = run_stage(hw, preset, seq_len, tp, cp, pp, role(d), blocks[d], sequence_parallel,
                                   vocab_parallel, with_recompute, attn_bwd_kv_block)
        return cache[key]

    costs = [stage_cost(d, recompute) for d in range(pp)]

    # Model FLOPs for MFU: FLOPs the mock run counts for one micro-batch through every stage,
    # without recompute (as tt-train's tools/profiling/results_json/add_roofline.py does).
    # Counts are per device: x tp * cp.
    model_flops_per_mb = sum(stage_cost(d, False).flops_per_mb for d in range(pp)) * tp * cp

    # Stage hand-off: every device of the stage sends its part of the activation (forward) or its
    # gradient (backward) across the cut; without SP every TP chip holds the full activation.
    # Per micro-batch a device sends v activations one way and v gradients the other way; the two
    # directions of the full-duplex links run concurrently, so v hand-offs count.
    m, v = num_microbatches, sched.interleave
    handoff_bytes = costs[0].output_bytes * tp * cp
    hand = send_recv_roofline(hw, handoff_bytes, cut_links, operation="PipelineHandoff").ideal_network_ns / 1e6
    hand_per_mb = v * hand if pp > 1 else 0.0

    def device_time(c: StageCost) -> Dict[str, float]:
        flop, dram, fabric = c.mb_flop_ms, c.mb_dram_ms, c.mb_fabric_ms + hand_per_mb
        return {"flop_ms": flop, "dram_ms": dram, "fabric_ms": fabric, "overlapped_ms": max(flop, dram, fabric),
                "serial_ms": c.mb_serial_ms + hand_per_mb}

    per_stage = [device_time(c) for c in costs]
    D = max(per_stage, key=lambda p: p["overlapped_ms"])

    # Play the schedule out with each device's own time per micro-batch, split into forward and
    # backward in the proportion of that device's forward and backward phases. With equal devices
    # this is slots(P, m) x D; uneven stages set the pace with the slowest and fill/drain with the sum.
    def split(total: float, fwd: float, bwd: float):
        r = fwd / (fwd + bwd) if fwd + bwd > 0 else 1 / 3
        return total * r, total * (1 - r)

    over = [split(p["overlapped_ms"], c.mb_forward_ms, c.mb_backward_ms) for p, c in zip(per_stage, costs)]
    ser = [split(p["serial_ms"], c.mb_forward_serial_ms, c.mb_backward_serial_ms) for p, c in zip(per_stage, costs)]
    bubble, held, pipeline_ms = simulate_schedule(sched, pp, m, [f for f, _ in over], [b for _, b in over])
    _, _, pipeline_serial_ms = simulate_schedule(sched, pp, m, [f for f, _ in ser], [b for _, b in ser])

    # The optimizer runs after the last backward; the step waits for the slowest stage's update
    opt = max(max(c.opt_flop_ms, c.opt_dram_ms) for c in costs)
    link_total = max(m * p["fabric_ms"] + c.sync_fabric_ms for p, c in zip(per_stage, costs))
    step_overlapped = max(pipeline_ms + opt, link_total)
    step_serial = pipeline_serial_ms + max(c.opt_serial_ms + c.sync_serial_ms for c in costs)

    # Memory per device: the one-micro-batch run's peak (state, its activations, backward
    # transients) plus the other micro-batches the schedule makes it hold at the same time
    device_memory = [c.peak_bytes + (held[d] - 1) * c.activation_bytes for d, c in enumerate(costs)]
    worst = max(range(pp), key=lambda d: device_memory[d])
    memory = device_memory[worst]

    tokens = m * seq_len
    peak = hw.tflops_per_chip(MathFidelity.HiFi4) * 1e12
    model_flops = model_flops_per_mb * m

    def throughput(step_ms: float) -> Dict[str, float]:
        return {"step_ms": step_ms, "tokens_per_second": tokens / (step_ms / 1e3),
                "mfu": model_flops / (step_ms / 1e3) / (chips * peak)}

    bound = max(("flop", "dram", "fabric"), key=lambda k: D[f"{k}_ms"])
    return {
        "config": {"model": model_name, "seq_len": seq_len, "num_microbatches": m, "hardware": hw.name,
                   "num_links": hw.num_links, "topology": hw.topology, "chips": chips, "pp": pp, "tp": tp,
                   "cp": cp, "schedule": str(sched), "interleave": v, "sequence_parallel": sequence_parallel,
                   "vocab_parallel": vocab_parallel, "recompute": recompute, "cut_links": cut_links,
                   "attn_bwd_kv_block": attn_bwd_kv_block, "blocks_per_stage": blocks},
        "per_microbatch": {**D, "bound": bound, "handoff_ms": hand},
        "bubble": bubble,
        "overlapped": throughput(step_overlapped),
        "serial": throughput(step_serial),
        "memory": {"peak_gb": memory / 1e9, "fits": memory <= hw.dram_gb_per_chip * 1e9, "device": worst,
                   "activation_gb_per_mb": costs[worst].activation_bytes / 1e9,
                   "peak_activations": held[worst], "state_gb": costs[worst].state_bytes / 1e9},
    }


def even_split(num_blocks: int, pp: int) -> List[int]:
    """Blocks per pipeline device: as even as possible, the remainder on the first devices."""
    base, extra = divmod(num_blocks, pp)
    return [base + (1 if d < extra else 0) for d in range(pp)]


SWEEP_SCHEDULES = (("1f1b", 1), ("interleaved", 2), ("interleaved", 5))


def sweep(chips: int, schedules=SWEEP_SCHEDULES, **kwargs) -> List[Dict[str, Any]]:
    """Every split of `chips` into pp * tp * cp, schedule and recompute choice that is valid."""
    cache: Dict[Any, StageCost] = {}
    results = []
    for tp in (1, 2, 4, 8, 16, 32):
        for cp in (1, 2, 4, 8, 16, 32):
            if chips % (tp * cp):
                continue
            pp = chips // (tp * cp)
            for schedule, v in schedules:
                if pp == 1 and schedule != "1f1b":
                    continue
                for recompute in (False, True):
                    r = run_parallel_roofline(pp=pp, tp=tp, cp=cp, schedule=schedule, interleave=v,
                                              recompute=recompute, stage_cache=cache, **kwargs)
                    if "error" not in r:
                        results.append(r)
    return results


def _describe(r: Dict[str, Any]) -> str:
    c = r["config"]
    sched = "" if c["pp"] == 1 else f", {c['schedule']}"
    return f"PP{c['pp']} x TP{c['tp']} x CP{c['cp']}{sched}{', recompute' if c['recompute'] else ''}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="llama-70b", help="Llama preset (default: llama-70b)")
    parser.add_argument("-s", "--seq", type=int, default=8192, help="Sequence length (default: 8192)")
    parser.add_argument("-m", "--microbatches", type=int, default=64, help="Sequences per optimizer step")
    parser.add_argument("--hardware", default="bh_quad", choices=list(HARDWARE_PRESETS))
    parser.add_argument("--pp", type=int, default=1, help="Pipeline stages")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel degree")
    parser.add_argument("--cp", type=int, default=1, help="Context parallel degree (ring attention)")
    parser.add_argument("--schedule", default="1f1b", choices=("naive", "gpipe", "1f1b", "interleaved"),
                        help="Pipeline schedule (default: 1f1b)")
    parser.add_argument("--interleave", type=int, default=2,
                        help="Chunks per pipeline device for --schedule interleaved (default: 2)")
    parser.add_argument("--sp", action="store_true", help="Sequence parallelism inside TP")
    parser.add_argument("--vocab-parallel", action="store_true", help="Split the LM head across stages")
    parser.add_argument("--recompute", action="store_true", help="Recompute block forward in backward")
    parser.add_argument("--cut-links", type=int, default=8, help="Links across a stage boundary (default: 8)")
    parser.add_argument("--attn-bwd-kv-block", type=int, default=None,
                        help="K/V tokens per pass in the attention backward (default: one 32-token tile)")
    parser.add_argument("--blocks-per-stage", default=None, metavar="N,N,...",
                        help="Transformer blocks on each pipeline device (default: even split, remainder first)")
    parser.add_argument("--links", type=int, default=None, help="Override links per neighbouring chip pair")
    parser.add_argument("--topology", choices=("ring", "linear"), default=None, help="Override topology")
    parser.add_argument("--sweep", type=int, default=None, metavar="CHIPS",
                        help="Evaluate every PP x TP x CP split of CHIPS chips and print the best that fit")
    parser.add_argument("--top", type=int, default=10, help="Rows to print with --sweep")
    parser.add_argument("--output", default=None, help="Write the result (or sweep) as JSON")
    args = parser.parse_args()

    common = dict(model_name=args.model, seq_len=args.seq, num_microbatches=args.microbatches,
                  hardware=args.hardware, sequence_parallel=args.sp, vocab_parallel=args.vocab_parallel,
                  cut_links=args.cut_links, attn_bwd_kv_block=args.attn_bwd_kv_block,
                  num_links=args.links, topology=args.topology)
    if args.sweep:
        results = sweep(args.sweep, **common)
        fits = sorted((r for r in results if r["memory"]["fits"]), key=lambda r: -r["overlapped"]["tokens_per_second"])
        print(f"{len(fits)} of {len(results)} configurations fit in memory")
        print(f"{'configuration':40s} {'tokens/s':>10s} {'MFU':>7s} {'bound':>7s} {'bubble':>7s} {'mem GB':>7s}")
        for r in fits[: args.top]:
            print(f"{_describe(r):40s} {r['overlapped']['tokens_per_second']:10,.0f} {r['overlapped']['mfu']:7.1%} "
                  f"{r['per_microbatch']['bound']:>7s} {r['bubble']:7.1%} {r['memory']['peak_gb']:7.1f}")
        out = results
    else:
        blocks = [int(b) for b in args.blocks_per_stage.split(",")] if args.blocks_per_stage else None
        out = run_parallel_roofline(pp=args.pp, tp=args.tp, cp=args.cp, schedule=args.schedule,
                                    interleave=args.interleave if args.schedule == "interleaved" else 1,
                                    recompute=args.recompute, blocks_per_stage=blocks, **common)
        if "error" in out:
            raise SystemExit(out["error"])
        pm, o, s, mem = out["per_microbatch"], out["overlapped"], out["serial"], out["memory"]
        print(f"{_describe(out)} on {out['config']['chips']} chips ({out['config']['hardware']}), "
              f"{args.microbatches} x {args.seq} tokens per step")
        print(f"  per micro-batch, slowest stage: FLOP {pm['flop_ms']:.2f} ms, DRAM {pm['dram_ms']:.2f} ms, "
              f"fabric {pm['fabric_ms']:.2f} ms (hand-off {pm['handoff_ms']:.3f} ms) -> {pm['bound']}-bound")
        if len(set(out["config"]["blocks_per_stage"])) > 1:
            print(f"  blocks per stage: {out['config']['blocks_per_stage']}")
        print(f"  bubble {out['bubble']:.2%}")
        print(f"  overlapped roofline: {o['tokens_per_second']:,.0f} tokens/s, MFU {o['mfu']:.1%}, step {o['step_ms']/1e3:.2f} s")
        print(f"  serial (per-op sum): {s['tokens_per_second']:,.0f} tokens/s, MFU {s['mfu']:.1%}")
        print(f"  memory per chip: {mem['peak_gb']:.1f} GB on device {mem['device']} "
              f"({'fits' if mem['fits'] else 'does not fit'}; state {mem['state_gb']:.1f} GB + "
              f"{mem['peak_activations']:.1f} x {mem['activation_gb_per_mb']:.2f} GB held)")
    if args.output:
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2, default=str)


if __name__ == "__main__":
    main()
