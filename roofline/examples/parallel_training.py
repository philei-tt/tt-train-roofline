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
uneven) is run for one micro-batch in its own RooflineContext. A stage's time per micro-batch is the
sum over its ops of max(FLOP, DRAM, fabric) time, a matmul fused with its collective counting as one
op; the schedule is then played out with each device's own forward and backward time; see the README.
TP and CP are placed on whole axes of the stage's part of the mesh, as tt-train requires; each group's
collectives run as a ring only where the group fills a whole wrapped mesh axis.

Examples (from the repository root):
    python3 -m roofline.examples.parallel_training --model llama-70b -s 8192 -m 2048 \\
        --pp 16 --tp 2 --cp 4 --recompute --attn-block 512
    python3 -m roofline.examples.parallel_training --model llama-70b -s 8192 -m 2048 \\
        --sweep 128 --sp --attn-block 512
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
    MockVocabParallelCrossEntropyLossOp,
    MockDistributedLlama,
    MockTensor,
    RooflineContext,
    TensorLabel,
    mock_clip_grad_norm,
    send_recv_roofline,
)
from roofline.operations.operation import create_grad_tensor
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
    mb_forward_ms: float  # roofline: sum over ops of max(FLOP, DRAM, fabric), fused ops as one
    mb_backward_ms: float
    mb_forward_bound_ms: float  # upper bound: max(sum FLOP, sum DRAM, sum fabric) of the phase
    mb_backward_bound_ms: float
    opt_ms: float  # optimizer + gradient clipping, once per step (per-op roofline)
    opt_flop_ms: float
    opt_dram_ms: float
    sync_ms: float  # CP gradient all-reduce, once per step (per-op roofline)
    sync_fabric_ms: float
    activation_bytes: int  # held from forward to backward by the blocks, one micro-batch (excl. LM head)
    state_bytes: int  # parameters + gradients + optimizer state
    peak_bytes: int  # peak of the whole one-micro-batch run (incl. backward transients)
    flops_per_mb: int  # one micro-batch on one device, forward + backward
    embedding_flops_per_mb: int
    output_bytes: int  # activation sent to the next stage, per device


def run_stage(
    hw, preset: dict, seq_len: int, tp: int, cp: int, pp: int, stage_index: int, stage_blocks: int,
    sequence_parallel: bool, vocab_parallel: bool, recompute: bool, attn_block: Optional[int],
    group_topology: Optional[Dict[str, str]] = None,
) -> StageCost:
    """Run one micro-batch (one sequence) through one pipeline stage of `stage_blocks` blocks,
    then the optimizer step."""
    # Objects from earlier runs never take part in this run's cycles: freeze them so the mock
    # autograd's frequent gc.collect() calls only scan this run's objects.
    gc.collect()
    gc.freeze()
    try:
        return _run_stage(hw, preset, seq_len, tp, cp, pp, stage_index, stage_blocks, sequence_parallel,
                          vocab_parallel, recompute, attn_block, group_topology or {})
    finally:
        gc.unfreeze()


def _run_stage(hw, preset, seq_len, tp, cp, pp, stage_index, stage_blocks, sequence_parallel, vocab_parallel,
               recompute, attn_block, group_topology) -> StageCost:
    ctx = RooflineContext(hw, attn_block=attn_block, group_topology=dict(group_topology))
    config = distributed_llama_config(
        preset, tp_size=tp, cp_size=cp, sequence_parallel=sequence_parallel,
        num_stages=pp, stage_index=stage_index, stage_num_blocks=stage_blocks, vocab_parallel=vocab_parallel,
    )
    stage = MockDistributedLlama(config, runner="mem_eff" if recompute else "default")
    params = stage.parameters()
    optimizer = MockAdamW(params)
    local_seq = seq_len // cp
    # Steady state: every micro-batch after the first finds all gradients already allocated, adds
    # into them out of place (tt-train's Tensor::add_grad) and holds them next to its activations
    for p in params.values():
        if p.requires_grad:
            p.add_grad(create_grad_tensor(p.shape, p.dtype, p.layout, name="accumulated_grad",
                                          num_shards=p.num_shards))
    mem_before, _ = ctx.memory_tracker.current_memory()

    if stage.is_first:
        x = MockTensor((1, 1, 1, local_seq), dtype=DataType.BFLOAT16, requires_grad=False,
                       label=TensorLabel.ACTIVATION, name="indices")
    else:
        rows = local_seq // (tp if sequence_parallel else 1)
        x = MockTensor((1, 1, rows, config.embedding_dim), dtype=DataType.BFLOAT16, requires_grad=True,
                       label=TensorLabel.ACTIVATION, name="received_activation")
    out = stage.forward_blocks(ctx, x)
    # The micro-batches held at once by the schedule hold only the blocks' activations: the LM head
    # and loss run and free theirs within one micro-batch (they are in peak_bytes)
    mem_after_blocks, _ = ctx.memory_tracker.current_memory()
    logits = stage.forward_head(ctx, out)
    roots: List[MockTensor] = []
    if logits is not None:
        targets = MockTensor((1, 1, 1, local_seq), dtype=DataType.BFLOAT16, requires_grad=False,
                             label=TensorLabel.ACTIVATION, name="targets")
        roots.append(MockVocabParallelCrossEntropyLossOp.apply(ctx, logits, targets, tp))
    if not stage.is_last:
        roots.append(out)  # the gradient arrives from the next stage
    output_bytes = out.bytes()

    for root in roots:
        root.backward(ctx, retain_graph=False)
    synchronize_gradients(ctx, params, cp, group="cp")  # weights are replicated across the CP group
    optimizer.step(ctx)
    mock_clip_grad_norm(ctx, params, max_norm=1.0)
    del x, out, logits, roots  # release the micro-batch's tensors before reading the state
    gc.collect()
    state_bytes, _ = ctx.memory_tracker.current_memory()
    peak_bytes, _ = ctx.memory_tracker.peak_memory()
    ctx.disable_memory_tracking()

    def op_ms(phases):
        return ctx.op_time_ns(phases) / 1e6

    def bound_ms(phases):
        return ctx.overlapped_time_ns(phases) / 1e6

    return StageCost(
        mb_flop_ms=ctx.compute_time_ns(MICROBATCH_PHASES) / 1e6,
        mb_dram_ms=ctx.dram_time_ns(MICROBATCH_PHASES) / 1e6,
        mb_fabric_ms=ctx.network_time_ns(MICROBATCH_PHASES) / 1e6,
        mb_forward_ms=op_ms(("forward",)),
        mb_backward_ms=op_ms(("backward",)),
        mb_forward_bound_ms=bound_ms(("forward",)),
        mb_backward_bound_ms=bound_ms(("backward",)),
        opt_ms=op_ms(OPTIMIZER_PHASES),
        opt_flop_ms=ctx.compute_time_ns(OPTIMIZER_PHASES) / 1e6,
        opt_dram_ms=ctx.dram_time_ns(OPTIMIZER_PHASES + SYNC_PHASES) / 1e6,
        sync_ms=op_ms(SYNC_PHASES),
        sync_fabric_ms=ctx.network_time_ns(SYNC_PHASES) / 1e6,
        activation_bytes=mem_after_blocks - mem_before,
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
    cut_links: Optional[int] = None,
    attn_block: Optional[int] = None,
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
        cut_links: Ethernet links (per direction) across the cut between two neighbouring stages;
            default from the stage layout (place_stages), e.g. a quad (32 x 4 torus) with stages
            of whole rows of 4 chips: 4 chip pairs x 2 links = 8
        attn_block: tokens per attention block, see fused_attention_roofline(block)
        num_links, topology: override the hardware's links per neighbour; topology forces one ring or
            line topology on every group instead of deriving it from the placement
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
    placement = place_stages(hw, pp, tp, cp, interleaved=sched.interleave > 1)
    if "error" in placement:
        return placement
    group_topology = placement["topology"] if topology is None else {"tp": topology, "cp": topology}
    if cut_links is None:
        cut_links = placement["cut_links"]

    cache = stage_cache if stage_cache is not None else {}

    def role(d: int) -> int:
        # first, middle or last: what a stage holds besides its blocks
        return 0 if d == 0 else (pp - 1 if d == pp - 1 else 1)

    def stage_cost(d: int, with_recompute: bool) -> StageCost:
        key = (model_name, seq_len, hardware, hw.num_links, tuple(sorted(group_topology.items())), pp, tp, cp,
               role(d), blocks[d], sequence_parallel, vocab_parallel, with_recompute, attn_block)
        if key not in cache:
            cache[key] = run_stage(hw, preset, seq_len, tp, cp, pp, role(d), blocks[d], sequence_parallel,
                                   vocab_parallel, with_recompute, attn_block, group_topology)
        return cache[key]

    costs = [stage_cost(d, recompute) for d in range(pp)]

    # Model FLOPs for MFU: FLOPs the mock run counts for one micro-batch through every stage,
    # without recompute (as tt-train's tools/profiling/results_json/add_roofline.py does).
    # Counts are per device: x tp * cp.
    model_flops_per_mb = sum(stage_cost(d, False).flops_per_mb for d in range(pp)) * tp * cp

    # Stage hand-off: every device of the stage sends its part of the activation (forward) or its
    # gradient (backward) across the cut; without SP every TP chip holds the full activation.
    # Per micro-batch a device sends v activations and v gradients. In the roofline each send is its
    # own op on the device (2v ops); in the upper bound the two directions of the full-duplex links
    # overlap, so v hand-offs count against the fabric.
    m, v = num_microbatches, sched.interleave
    handoff_bytes = costs[0].output_bytes * tp * cp
    hand = (send_recv_roofline(hw, handoff_bytes, cut_links, operation="PipelineHandoff").ideal_network_ns / 1e6
            if pp > 1 else 0.0)

    def device_time(c: StageCost) -> Dict[str, float]:
        flop, dram, fabric = c.mb_flop_ms, c.mb_dram_ms, c.mb_fabric_ms + v * hand
        return {"flop_ms": flop, "dram_ms": dram, "fabric_ms": fabric,
                "roofline_ms": c.mb_forward_ms + c.mb_backward_ms + 2 * v * hand,
                "upper_bound_ms": max(flop, dram, fabric)}

    per_stage = [device_time(c) for c in costs]
    D = max(per_stage, key=lambda p: p["roofline_ms"])

    # Play the schedule out with each device's own forward and backward time per micro-batch.
    # With equal devices this is slots(P, m) x D; uneven stages set the pace with the slowest and
    # fill/drain with the sum.
    fwd = [c.mb_forward_ms + v * hand for c in costs]
    bwd = [c.mb_backward_ms + v * hand for c in costs]
    bubble, held, pipeline_ms = simulate_schedule(sched, pp, m, fwd, bwd)

    def split(total: float, f: float, b: float):
        r = f / (f + b) if f + b > 0 else 1 / 3
        return total * r, total * (1 - r)

    ub = [split(p["upper_bound_ms"], c.mb_forward_bound_ms, c.mb_backward_bound_ms) for p, c in zip(per_stage, costs)]
    _, _, pipeline_bound_ms = simulate_schedule(sched, pp, m, [f for f, _ in ub], [b for _, b in ub])

    # The CP gradient all-reduce and the optimizer run after the last backward, one after the other;
    # the step waits for the slowest stage
    step_roofline = pipeline_ms + max(c.sync_ms + c.opt_ms for c in costs)
    link_total = max(m * p["fabric_ms"] + c.sync_fabric_ms for p, c in zip(per_stage, costs))
    step_bound = max(pipeline_bound_ms + max(max(c.opt_flop_ms, c.opt_dram_ms) for c in costs), link_total)

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
                   "num_links": hw.num_links, "topology": group_topology, "stage_mesh": placement["stage_mesh"],
                   "chips": chips, "pp": pp, "tp": tp,
                   "cp": cp, "schedule": str(sched), "interleave": v, "sequence_parallel": sequence_parallel,
                   "vocab_parallel": vocab_parallel, "recompute": recompute, "cut_links": cut_links,
                   "attn_block": attn_block, "blocks_per_stage": blocks},
        "per_microbatch": {**D, "bound": bound, "handoff_ms": hand},
        "bubble": bubble,
        "roofline": throughput(step_roofline),
        "upper_bound": throughput(step_bound),
        "memory": {"peak_gb": memory / 1e9, "fits": memory <= hw.dram_gb_per_chip * 1e9, "device": worst,
                   "activation_gb_per_mb": costs[worst].activation_bytes / 1e9,
                   "peak_activations": held[worst], "state_gb": costs[worst].state_bytes / 1e9},
    }


def place_stages(hw, pp: int, tp: int, cp: int, interleaved: bool = False) -> Dict[str, Any]:
    """Lay pipeline stages along the first mesh axis and TP / CP on whole axes of each stage's block.

    The hardware's mesh (hw.mesh_shape, rows x columns, hw.mesh_wrap) is cut into pp equal blocks of
    tp * cp chips: whole rows when a stage spans at least one row, else parts of a row. tt-train's
    collectives run over whole mesh axes (ttnn_fixed/distributed/ttnn_ops.cpp), so TP and CP must each
    be one axis of the block (one of them may be 1 on a one-row block); TP takes the column axis when
    it fits. A group's collectives are a ring when it fills a whole wrapped axis of the mesh and has
    more than 2 chips (tt-train's get_topology), else a line.

    Returns {"topology": {"tp": ..., "cp": ...}, "cut_links", "stage_mesh"} or {"error": str}.
    """
    R, C = hw.mesh_shape
    wrap_r, wrap_c = hw.mesh_wrap
    n = tp * cp
    if R * C == 1:  # single chip / unspecified mesh: keep the hardware's own topology
        return {"topology": {}, "cut_links": hw.num_links, "stage_mesh": (1, n)}
    if pp * n > R * C:
        return {"error": f"pp*tp*cp = {pp * n} > {R * C} chips of {hw.name}"}
    if n % C == 0:
        block = (n // C, C)
    elif C % n == 0:
        block = (1, n)
    else:
        return {"error": f"a stage of {n} chips is neither whole rows nor part of a row of {C}"}
    rows, cols = block

    def topo(size: int, full: bool, wrap: bool) -> str:
        return "ring" if size > 2 and full and wrap else "linear"

    row_axis = (rows, rows == R, wrap_r)
    col_axis = (cols, cols == C, wrap_c)
    if tp > 1 and cp > 1:
        if (tp, cp) == (cols, rows):
            tp_axis, cp_axis = col_axis, row_axis
        elif (tp, cp) == (rows, cols):
            tp_axis, cp_axis = row_axis, col_axis
        else:
            return {"error": f"TP{tp} x CP{cp} is not one mesh axis each on a {rows} x {cols} stage "
                             "(tt-train collectives run over whole mesh axes)"}
    elif max(tp, cp) > 1:
        if rows != 1:
            return {"error": f"TP{tp} x CP{cp} on a {rows} x {cols} stage: one group cannot cover two mesh axes"}
        tp_axis = cp_axis = col_axis
    else:
        tp_axis = cp_axis = col_axis
    if interleaved and not (wrap_r and pp * rows == R):
        return {"error": "interleaved needs the last stage to neighbour the first (stages filling a wrapped axis)"}
    pairs = cols if rows >= 1 and cols == C else 1  # chip pairs across the cut between two stages
    return {"topology": {"tp": topo(tp, *tp_axis[1:]) if tp > 1 else "linear",
                         "cp": topo(cp, *cp_axis[1:]) if cp > 1 else "linear"},
            "cut_links": pairs * hw.num_links, "stage_mesh": block}


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
    parser.add_argument("--cut-links", type=int, default=None,
                        help="Links across a stage boundary (default: from the stage layout)")
    parser.add_argument("--attn-block", type=int, default=None,
                        help="Tokens kept on chip per attention block, forward and backward "
                             "(default: one 32-token tile, today's kernels)")
    parser.add_argument("--blocks-per-stage", default=None, metavar="N,N,...",
                        help="Transformer blocks on each pipeline device (default: even split, remainder first)")
    parser.add_argument("--links", type=int, default=None, help="Override links per neighbouring chip pair")
    parser.add_argument("--topology", choices=("ring", "linear"), default=None,
                        help="Force one topology on every group (default: derived from the placement)")
    parser.add_argument("--sweep", type=int, default=None, metavar="CHIPS",
                        help="Evaluate every PP x TP x CP split of CHIPS chips and print the best that fit")
    parser.add_argument("--top", type=int, default=10, help="Rows to print with --sweep")
    parser.add_argument("--output", default=None, help="Write the result (or sweep) as JSON")
    args = parser.parse_args()

    common = dict(model_name=args.model, seq_len=args.seq, num_microbatches=args.microbatches,
                  hardware=args.hardware, sequence_parallel=args.sp, vocab_parallel=args.vocab_parallel,
                  cut_links=args.cut_links, attn_block=args.attn_block,
                  num_links=args.links, topology=args.topology)
    if args.sweep:
        results = sweep(args.sweep, **common)
        fits = sorted((r for r in results if r["memory"]["fits"]), key=lambda r: -r["roofline"]["tokens_per_second"])
        print(f"{len(fits)} of {len(results)} configurations fit in memory")
        print(f"{'configuration':40s} {'tokens/s':>10s} {'MFU':>7s} {'bound':>7s} {'bubble':>7s} {'mem GB':>7s}")
        for r in fits[: args.top]:
            print(f"{_describe(r):40s} {r['roofline']['tokens_per_second']:10,.0f} {r['roofline']['mfu']:7.1%} "
                  f"{r['per_microbatch']['bound']:>7s} {r['bubble']:7.1%} {r['memory']['peak_gb']:7.1f}")
        out = results
    else:
        blocks = [int(b) for b in args.blocks_per_stage.split(",")] if args.blocks_per_stage else None
        out = run_parallel_roofline(pp=args.pp, tp=args.tp, cp=args.cp, schedule=args.schedule,
                                    interleave=args.interleave if args.schedule == "interleaved" else 1,
                                    recompute=args.recompute, blocks_per_stage=blocks, **common)
        if "error" in out:
            raise SystemExit(out["error"])
        pm, o, ub, mem = out["per_microbatch"], out["roofline"], out["upper_bound"], out["memory"]
        print(f"{_describe(out)} on {out['config']['chips']} chips ({out['config']['hardware']}), "
              f"{args.microbatches} x {args.seq} tokens per step")
        cfg = out["config"]
        print(f"  stage mesh {cfg['stage_mesh'][0]} x {cfg['stage_mesh'][1]}, TP {cfg['topology'].get('tp', '-')}, "
              f"CP {cfg['topology'].get('cp', '-')}, {cfg['cut_links']} links across a stage cut")
        print(f"  per micro-batch, slowest stage: {pm['roofline_ms']:.2f} ms (sum of FLOP {pm['flop_ms']:.2f}, "
              f"DRAM {pm['dram_ms']:.2f}, fabric {pm['fabric_ms']:.2f} ms; hand-off {pm['handoff_ms']:.3f} ms) "
              f"-> mostly {pm['bound']}")
        if len(set(out["config"]["blocks_per_stage"])) > 1:
            print(f"  blocks per stage: {out['config']['blocks_per_stage']}")
        print(f"  bubble {out['bubble']:.2%}")
        print(f"  roofline (sum of per-op max): {o['tokens_per_second']:,.0f} tokens/s, MFU {o['mfu']:.1%}, "
              f"step {o['step_ms']/1e3:.2f} s")
        print(f"  upper bound (max of summed resources): {ub['tokens_per_second']:,.0f} tokens/s, MFU {ub['mfu']:.1%}")
        print(f"  memory per chip: {mem['peak_gb']:.1f} GB on device {mem['device']} "
              f"({'fits' if mem['fits'] else 'does not fit'}; state {mem['state_gb']:.1f} GB + "
              f"{mem['peak_activations']:.1f} x {mem['activation_gb_per_mb']:.2f} GB held)")
    if args.output:
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2, default=str)


if __name__ == "__main__":
    main()
