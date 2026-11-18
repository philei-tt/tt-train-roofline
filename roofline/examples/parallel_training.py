# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Roofline analysis for Llama training with pipeline, tensor, sequence and context parallelism.

Combines, on `chips = pp * tp * cp` devices:
  - PP: a pipeline schedule (naive, GPipe, 1F1B, interleaved 1F1B with v chunks per device),
    optional vocabulary parallelism
  - TP: tensor parallelism inside a stage, optionally with sequence parallelism (--sp)
  - CP: context parallelism inside a stage (ring attention, load-balanced split)
  - activation recompute (--recompute, the memory-efficient runner)

Each distinct pipeline stage (the first and the last) is run for one micro-batch in its own
RooflineContext. The step time is a resource-level roofline: FLOP time, DRAM time and fabric
(Ethernet) time are each summed and the largest one sets the time, i.e. all three overlap:

    D       = max(FLOP, DRAM, fabric) of one micro-batch on the slowest stage, fabric incl. hand-offs
    T_step  = max( slots * D + max(optimizer FLOP, optimizer DRAM),  m * fabric + CP gradient sync )
    slots   = step length in micro-batch times, from the schedule (training/pipeline.py):
              naive m * P, GPipe and 1F1B m + P - 1, interleaved 1F1B m + (P - 1) / v
    bubble  = (slots - m) / slots

The per-op serial time (sum over ops of max(compute, DRAM, network), as in training.py) is
reported next to it.

Examples (from the repository root):
    python3 -m roofline.examples.parallel_training --model llama-70b -s 8192 -m 2048 \\
        --pp 16 --tp 2 --cp 4 --recompute --attn-bwd-kv-block 512
    python3 -m roofline.examples.parallel_training --model llama-70b -s 8192 -m 2048 \\
        --sweep 128 --sp --attn-bwd-kv-block 512
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from roofline import (
    BLACKHOLE_GALAXY,
    BLACKHOLE_P150,
    BLACKHOLE_QUAD,
    DataType,
    MathFidelity,
    MockAdamW,
    MockCrossEntropyLossOp,
    MockLlamaPipelineStage,
    MockLlamaPipelineStageConfig,
    MockTensor,
    RooflineContext,
    TensorLabel,
    mock_clip_grad_norm,
    send_recv_roofline,
)
from roofline.examples.training import MODEL_PRESETS, ModelType
from roofline.training.distributed import synchronize_gradients
from roofline.training.pipeline import make_schedule

HARDWARE = {"p150": BLACKHOLE_P150, "bh_glx": BLACKHOLE_GALAXY, "bh_quad": BLACKHOLE_QUAD}


@dataclass
class StageCost:
    """Roofline cost of one pipeline stage (times in ms, memory in bytes)."""

    mb_flop_ms: float  # one micro-batch, forward + backward
    mb_dram_ms: float
    mb_fabric_ms: float
    mb_serial_ms: float  # sum over ops of max(compute, DRAM, network)
    opt_flop_ms: float  # optimizer + gradient clipping, once per step
    opt_dram_ms: float
    opt_serial_ms: float
    sync_fabric_ms: float  # CP gradient all-reduce, once per step
    sync_serial_ms: float
    activation_bytes: int  # held from forward to backward, one micro-batch
    state_bytes: int  # parameters + gradients + optimizer state
    flops_per_mb: int


def llama_flops_per_token(preset: dict, seq_len: int) -> float:
    """Model FLOPs per token, as tt-train computes MFU: 6 N + 12 L H Q S (PaLM)."""
    e, h, g, L = preset["embedding_dim"], preset["num_heads"], preset["num_groups"], preset["num_blocks"]
    ff = preset["intermediate_dim"]
    kv = g * (e // h)
    n = L * (2 * e * e + 2 * e * kv + e + 3 * e * ff + e) + preset["vocab_size"] * e + e
    return 6 * n + 12 * L * h * (e // h) * seq_len


def run_stage(
    hw,
    preset: dict,
    seq_len: int,
    tp: int,
    cp: int,
    pp: int,
    stage_index: int,
    sequence_parallel: bool,
    vocab_parallel: bool,
    recompute: bool,
    attn_bwd_kv_block: Optional[int],
) -> StageCost:
    """Run one micro-batch (one sequence) through one pipeline stage, then the optimizer step."""
    ctx = RooflineContext(hw)
    ctx.attn_bwd_kv_block = attn_bwd_kv_block
    config = MockLlamaPipelineStageConfig(
        vocab_size=preset["vocab_size"],
        max_sequence_length=preset["max_sequence_length"],
        embedding_dim=preset["embedding_dim"],
        intermediate_dim=preset.get("intermediate_dim"),
        num_heads=preset["num_heads"],
        num_groups=preset["num_groups"],
        dropout_prob=preset["dropout"],
        num_blocks=preset["num_blocks"],
        theta=preset["theta"],
        tp_size=tp,
        cp_size=cp,
        sequence_parallel=sequence_parallel,
        num_stages=pp,
        stage_index=stage_index,
        vocab_parallel=vocab_parallel,
    )
    stage = MockLlamaPipelineStage(config, runner="mem_eff" if recompute else "default")
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
    out, logits = stage(ctx, x)
    roots: List[MockTensor] = []
    if logits is not None:
        targets = MockTensor((1, 1, 1, local_seq), dtype=DataType.BFLOAT16, requires_grad=False,
                             label=TensorLabel.ACTIVATION, name="targets")
        roots.append(MockCrossEntropyLossOp.apply(ctx, logits, targets))
    if not stage.is_last:
        roots.append(out)  # the gradient arrives from the next stage
    mem_after_forward, _ = ctx.memory_tracker.current_memory()

    for root in roots:
        root.backward(ctx, retain_graph=False)
    n_mb = len(ctx.estimates)
    synchronize_gradients(ctx, params, cp)  # weights are replicated across the CP group
    n_sync = len(ctx.estimates)
    optimizer.step(ctx)
    mock_clip_grad_norm(ctx, params, max_norm=1.0)
    state_bytes, _ = ctx.memory_tracker.current_memory()
    ctx.disable_memory_tracking()

    def sums(estimates):
        return (
            sum(e.ideal_compute_ns for e in estimates) / 1e6,
            sum(e.ideal_memory_ns for e in estimates) / 1e6,
            sum(e.ideal_network_ns for e in estimates) / 1e6,
            sum(e.theoretical_time_ns for e in estimates) / 1e6,
        )

    mb, sync, opt = ctx.estimates[:n_mb], ctx.estimates[n_mb:n_sync], ctx.estimates[n_sync:]
    mb_f, mb_d, mb_n, mb_s = sums(mb)
    _, sy_d, sy_n, sy_s = sums(sync)
    op_f, op_d, _, op_s = sums(opt)
    return StageCost(
        mb_flop_ms=mb_f, mb_dram_ms=mb_d, mb_fabric_ms=mb_n, mb_serial_ms=mb_s,
        opt_flop_ms=op_f, opt_dram_ms=op_d + sy_d, opt_serial_ms=op_s,
        sync_fabric_ms=sy_n, sync_serial_ms=sy_s,
        activation_bytes=mem_after_forward - mem_before,
        state_bytes=state_bytes,
        flops_per_mb=sum(e.total_flops for e in mb),
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
    stage_cache: Optional[Dict[Any, StageCost]] = None,
) -> Dict[str, Any]:
    """Roofline of one optimizer step of `num_microbatches` sequences on pp * tp * cp chips.

    Args:
        model_name: Llama preset from training.MODEL_PRESETS
        seq_len: Tokens per sequence (one sequence per micro-batch)
        num_microbatches: m, sequences per optimizer step
        hardware: "bh_quad", "bh_glx" or "p150"
        pp, tp, cp: pipeline, tensor and context parallel degrees
        schedule: pipeline schedule: "naive", "gpipe", "1f1b" or "interleaved"
        interleave: v, model chunks per pipeline device (interleaved schedule only)
        sequence_parallel: Megatron sequence parallelism inside TP
        vocab_parallel: split the LM head across all pipeline stages
        recompute: recompute each block's forward in backward (memory-efficient runner)
        cut_links: Ethernet links (per direction) across the cut between two neighbouring stages.
            Quad (32 x 4 torus), stages laid along the 32-chip axis: one row of 4 chip pairs,
            2 links each = 8.
        attn_bwd_kv_block: K/V tokens per pass in the fused attention backward (None = 32)
        num_links, topology: override the hardware's links per neighbour and topology
        stage_cache: optional dict reused across calls (stage runs do not depend on m or v)

    Returns:
        Result dict, or {"error": str}
    """
    preset = MODEL_PRESETS.get(model_name)
    if preset is None or preset["model_type"] != ModelType.LLAMA:
        return {"error": f"Unknown Llama preset: {model_name}"}
    if hardware not in HARDWARE:
        return {"error": f"Unknown hardware: {hardware}. Available: {', '.join(HARDWARE)}"}
    hw = HARDWARE[hardware]
    overrides = {k: v for k, v in (("num_links", num_links), ("topology", topology)) if v is not None}
    if overrides:
        hw = dataclasses.replace(hw, **overrides)
    chips = pp * tp * cp
    L = preset["num_blocks"]
    try:
        sched = make_schedule(schedule, interleave)
    except ValueError as exc:
        return {"error": str(exc)}
    checks = [
        (chips <= hw.chips_per_galaxy, f"pp*tp*cp = {chips} > {hw.chips_per_galaxy} chips of {hw.name}"),
        (L % pp == 0, f"num_blocks ({L}) not divisible by pp ({pp})"),
        (pp == 1 or not sched.validate(pp, num_microbatches, L // pp), sched.validate(pp, num_microbatches, max(L // pp, 1))),
        (preset["num_groups"] % tp == 0, f"num_groups ({preset['num_groups']}) not divisible by tp ({tp})"),
        (seq_len % (2 * cp * (tp if sequence_parallel else 1)) == 0, "seq_len not divisible by 2 * cp (* tp with SP)"),
    ]
    for ok, msg in checks:
        if not ok:
            return {"error": msg}

    cache = stage_cache if stage_cache is not None else {}
    stage_ids = sorted({0, pp - 1})
    costs = []
    for sid in stage_ids:
        key = (model_name, seq_len, hardware, hw.num_links, hw.topology, pp, tp, cp, sid,
               sequence_parallel, vocab_parallel, recompute, attn_bwd_kv_block)
        if key not in cache:
            cache[key] = run_stage(hw, preset, seq_len, tp, cp, pp, sid, sequence_parallel,
                                   vocab_parallel, recompute, attn_bwd_kv_block)
        costs.append(cache[key])

    # Stage hand-off: the activation (forward) or its gradient (backward) of the whole sequence
    # crosses the cut; without SP every TP chip of the next stage needs the full activation.
    m, v = num_microbatches, sched.interleave
    act_bytes = seq_len * preset["embedding_dim"] * 2 * (1 if sequence_parallel else tp)
    hand = send_recv_roofline(hw, act_bytes, cut_links, operation="PipelineHandoff").ideal_network_ns / 1e6
    hand_per_mb = 2 * v * hand if pp > 1 else 0.0

    def device_time(c: StageCost) -> Dict[str, float]:
        flop, dram, fabric = c.mb_flop_ms, c.mb_dram_ms, c.mb_fabric_ms + hand_per_mb
        return {"flop_ms": flop, "dram_ms": dram, "fabric_ms": fabric, "overlapped_ms": max(flop, dram, fabric),
                "serial_ms": c.mb_serial_ms + hand_per_mb}

    per_stage = [device_time(c) for c in costs]
    slow = max(range(len(costs)), key=lambda i: per_stage[i]["overlapped_ms"])
    D, c = per_stage[slow], costs[slow]
    slots = sched.slots(pp, m) if pp > 1 else m
    step_overlapped = max(slots * D["overlapped_ms"] + max(c.opt_flop_ms, c.opt_dram_ms),
                          m * D["fabric_ms"] + c.sync_fabric_ms)
    slow_serial = max(range(len(costs)), key=lambda i: per_stage[i]["serial_ms"])
    step_serial = (slots * per_stage[slow_serial]["serial_ms"] + costs[slow_serial].opt_serial_ms
                   + costs[slow_serial].sync_serial_ms)

    # Memory: the device that holds the most micro-batches (stage 0) and the last stage
    stash = sched.peak_activations(pp, m) if pp > 1 else [1.0]
    stash_by_stage = {0: stash[0], pp - 1: stash[-1]}
    memory = max(cs.state_bytes + stash_by_stage[sid] * cs.activation_bytes for sid, cs in zip(stage_ids, costs))

    tokens = m * seq_len
    peak = hw.tflops_per_chip(MathFidelity.HiFi4) * 1e12
    model_flops = llama_flops_per_token(preset, seq_len) * tokens

    def throughput(step_ms: float) -> Dict[str, float]:
        return {"step_ms": step_ms, "tokens_per_second": tokens / (step_ms / 1e3),
                "mfu": model_flops / (step_ms / 1e3) / (chips * peak)}

    bound = max(("flop", "dram", "fabric"), key=lambda k: D[f"{k}_ms"])
    return {
        "config": {"model": model_name, "seq_len": seq_len, "num_microbatches": m, "hardware": hw.name,
                   "num_links": hw.num_links, "topology": hw.topology, "chips": chips, "pp": pp, "tp": tp,
                   "cp": cp, "schedule": str(sched), "interleave": v, "sequence_parallel": sequence_parallel,
                   "vocab_parallel": vocab_parallel, "recompute": recompute, "cut_links": cut_links,
                   "attn_bwd_kv_block": attn_bwd_kv_block},
        "per_microbatch": {**D, "bound": bound, "handoff_ms": hand},
        "bubble": sched.bubble(pp, m) if pp > 1 else 0.0,
        "overlapped": throughput(step_overlapped),
        "serial": throughput(step_serial),
        "memory": {"peak_gb": memory / 1e9, "fits": memory <= hw.dram_gb_per_chip * 1e9,
                   "activation_gb_per_mb": costs[0].activation_bytes / 1e9,
                   "stash_microbatches": stash[0], "state_gb": costs[0].state_bytes / 1e9},
    }


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
    return (f"PP{c['pp']} x TP{c['tp']} x CP{c['cp']}{sched}"
            f"{', recompute' if c['recompute'] else ''}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="llama-70b", help="Llama preset (default: llama-70b)")
    parser.add_argument("-s", "--seq", type=int, default=8192, help="Sequence length (default: 8192)")
    parser.add_argument("-m", "--microbatches", type=int, default=64, help="Sequences per optimizer step")
    parser.add_argument("--hardware", default="bh_quad", choices=sorted(HARDWARE))
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
                        help="K/V tokens per pass in the attention backward (default: 32, one tile)")
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
        out = run_parallel_roofline(pp=args.pp, tp=args.tp, cp=args.cp, schedule=args.schedule,
                                    interleave=args.interleave if args.schedule == "interleaved" else 1,
                                    recompute=args.recompute, **common)
        if "error" in out:
            raise SystemExit(out["error"])
        pm, o, s, mem = out["per_microbatch"], out["overlapped"], out["serial"], out["memory"]
        print(f"{_describe(out)} on {out['config']['chips']} chips ({out['config']['hardware']}), "
              f"{args.microbatches} x {args.seq} tokens per step")
        print(f"  per micro-batch, slowest stage: FLOP {pm['flop_ms']:.2f} ms, DRAM {pm['dram_ms']:.2f} ms, "
              f"fabric {pm['fabric_ms']:.2f} ms (hand-off {pm['handoff_ms']:.3f} ms) -> {pm['bound']}-bound")
        print(f"  bubble {out['bubble']:.2%}")
        print(f"  overlapped roofline: {o['tokens_per_second']:,.0f} tokens/s, MFU {o['mfu']:.1%}, step {o['step_ms']/1e3:.2f} s")
        print(f"  serial (per-op sum): {s['tokens_per_second']:,.0f} tokens/s, MFU {s['mfu']:.1%}")
        print(f"  memory per chip: {mem['peak_gb']:.1f} GB ({'fits' if mem['fits'] else 'does not fit'}; "
              f"state {mem['state_gb']:.1f} GB + {mem['stash_microbatches']:.1f} x {mem['activation_gb_per_mb']:.2f} GB)")
    if args.output:
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2, default=str)


if __name__ == "__main__":
    main()
