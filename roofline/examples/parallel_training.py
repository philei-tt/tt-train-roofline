# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Roofline analysis for Llama training with pipeline, tensor, sequence, context and data parallelism.

Combines, on `chips = dp * pp * tp * cp` devices:
  - PP: a pipeline schedule (training/pipeline.py)
  - TP: tensor parallelism inside a stage, optionally with sequence parallelism (--sp)
  - CP: context parallelism inside a stage (ring attention)
  - DP: dp copies of the whole pipeline, each running m / dp micro-batches; optional ZeRO-1
  - activation recompute: none, selective (norms, activations, attention) or full (whole blocks)
  - precision: bf16 or fp32 master weights, bf16 or fp32 gradient accumulation

--mode picks what the kernels and features are: "current" is tt-train today (tile-wide attention
blocks, contiguous ring attention with unfused merges and socket shifts); "target" is the planned
implementation (512-token attention blocks, load-balanced ring attention with fused merges and
overlapped shifts) and is the only mode that allows the features tt-train does not have yet:
ZeRO-1 and selective recompute. A single run reports both modes.

Every distinct pipeline stage (first, middle, last, and each block count when the layer split is
uneven) is run for one micro-batch in its own RooflineContext. A stage's time per micro-batch is the
sum over its ops of max(FLOP, DRAM, fabric) time, a matmul fused with its collective counting as one
op; the schedule is then played out with each device's own forward and backward time; see the README.
TP and CP are placed on whole axes of the stage's part of the mesh, as tt-train requires; each group's
collectives run as a ring only where the group fills a whole wrapped mesh axis.

Examples (from the repository root):
    python3 -m roofline.examples.parallel_training --model llama-70b -s 8192 -m 2048 \\
        --pp 16 --tp 4 --cp 2 --sp --recompute selective
    python3 -m roofline.examples.parallel_training --model llama-70b -s 8192 -m 2048 \\
        --sweep 128 --sp --mode current
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import itertools
import math
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from roofline import (
    HARDWARE_PRESETS,
    DataType,
    MathFidelity,
    MockAdamW,
    MockCrossEntropyLossOp,
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
from roofline.training.distributed import synchronize_replica_gradients
from roofline.training.pipeline import make_schedule, simulate_schedule

MICROBATCH_PHASES = ("forward", "backward")
OPTIMIZER_PHASES = ("optimizer", "grad_clip")
SYNC_PHASES = ("ccl",)  # gradient sync across the CP and DP groups
GATHER_PHASES = ("param_gather",)  # ZeRO-1: all-gather of the updated parameters, after the optimizer

# What --mode sets (see the module docstring); features only the target mode allows
ATTN_BLOCK = {"current": None, "target": 512}  # per mode; the mode is also the ring-attention model
TARGET_ONLY = {"zero1": "ZeRO-1", "selective": "selective recompute", "sequence_parallel": "sequence parallelism",
               "grad_fp32": "fp32 gradient accumulation"}  # tt-train runners today: default, memory-efficient
RECOMPUTE = ("none", "selective", "full")
DTYPES = {"bf16": DataType.BFLOAT16, "fp32": DataType.FLOAT32}


@dataclass(frozen=True)
class Options:
    """How a stage runs, apart from its parallel degrees (hashable: part of the stage cache key)."""

    sequence_parallel: bool = False
    recompute: str = "none"  # none | selective | full
    attn_block: Optional[int] = None  # tokens per attention block; None = one tile
    cp_mode: str = "target"  # ring attention: current | target
    master_dtype: str = "bf16"  # stored weights, m and v (tt-train precision.master_dtype)
    grad_dtype: str = "bf16"  # gradient accumulation across micro-batches
    zero1: bool = False  # optimizer state sharded over the CP x DP replicas (target only)
    accumulate: bool = True  # more than one micro-batch per step: gradients accumulate across them
    explicit_mask: bool = False  # today's pipeline trainer passes an S x S mask (full-square attention)
    plain_loss: bool = False  # tt-train's cross_entropy_loss (TP1, or today's pipeline trainer)
    clip: bool = True  # gradient-norm clipping


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
    sync_ms: float  # gradient sync over CP and DP, once per step (per-op roofline)
    sync_fabric_ms: float
    gather_ms: float  # ZeRO-1 parameter all-gather after the optimizer (per-op roofline)
    sync_dram_ms: float  # DRAM time of the layers' gradient sync (competes with the backward it overlaps)
    tail_sync_ms: float  # sync of gradients final only at the very end (embedding table)
    grad_bytes: int  # one block's gradients
    activation_bytes: int  # held from forward to backward by the blocks, one micro-batch (excl. LM head)
    state_bytes: int  # parameters + gradients + optimizer state
    peak_bytes: int  # peak of the whole one-micro-batch run (incl. backward transients)
    output_bytes: int  # activation sent to the next stage, per device


def run_stage(
    hw, preset: dict, seq_len: int, tp: int, cp: int, dp: int, pp: int, stage_index: int, stage_blocks: int,
    opts: Options, group_topology: Optional[Dict[str, str]] = None, link_share: Optional[Dict[str, int]] = None,
) -> StageCost:
    """Run one micro-batch (one sequence) through one pipeline stage of `stage_blocks` blocks,
    then the gradient sync and the optimizer step."""
    # Objects from earlier runs never take part in this run's cycles: freeze them so the mock
    # autograd's frequent gc.collect() calls only scan this run's objects.
    gc.collect()
    gc.freeze()
    try:
        return _run_stage(hw, preset, seq_len, tp, cp, dp, pp, stage_index, stage_blocks, opts,
                          group_topology or {}, link_share or {})
    finally:
        gc.unfreeze()


def _context(hw, opts: Options, group_topology, link_share) -> RooflineContext:
    return RooflineContext(hw, attn_block=opts.attn_block, cp_mode=opts.cp_mode,
                           selective_recompute=opts.recompute == "selective",
                           lean_captures=opts.cp_mode == "target", explicit_mask=opts.explicit_mask,
                           fuse_collectives=opts.cp_mode == "target",
                           single_kernel_attention_bw=opts.cp_mode == "target",
                           group_topology=dict(group_topology), group_link_share=dict(link_share))


def _run_stage(hw, preset, seq_len, tp, cp, dp, pp, stage_index, stage_blocks, opts, group_topology,
               link_share) -> StageCost:
    ctx = _context(hw, opts, group_topology, link_share)
    config = distributed_llama_config(
        preset, tp_size=tp, cp_size=cp, sequence_parallel=opts.sequence_parallel,
        num_stages=pp, stage_index=stage_index, stage_num_blocks=stage_blocks,
    )
    stage = MockDistributedLlama(config, runner="mem_eff" if opts.recompute == "full" else "default")
    params = stage.parameters()
    master = DTYPES[opts.master_dtype]
    replica_groups = [("cp", cp), ("dp", dp)]  # weights are replicated across CP and the DP copies
    optimizer = MockAdamW(params, master_dtype=master if master != DataType.BFLOAT16 else None,
                          num_shards=math.prod(n for _, n in replica_groups) if opts.zero1 else 1)
    local_seq = seq_len // cp
    # Steady state: every micro-batch after the first finds all gradients already allocated, adds
    # into them out of place (tt-train's Tensor::add_grad) and holds them next to its activations.
    # With one micro-batch per step there is nothing to add into (zero_grad frees the gradients).
    grad_dtype = DTYPES[opts.grad_dtype]
    for p in params.values():
        if p.requires_grad and opts.accumulate:
            p.add_grad(create_grad_tensor(p.shape, grad_dtype if p.dtype == DataType.BFLOAT16 else p.dtype,
                                          p.layout, name="accumulated_grad", num_shards=p.num_shards))
    mask = (MockTensor((1, 1, seq_len, seq_len), dtype=DataType.BFLOAT16, requires_grad=False,
                       label=TensorLabel.ACTIVATION, name="attention_mask") if opts.explicit_mask else None)
    mem_before, _ = ctx.memory_tracker.current_memory()

    sp_split = opts.sequence_parallel and tp > 1
    if stage.is_first:
        # with SP each TP device embeds only its own S / tp tokens
        x = MockTensor((1, 1, 1, local_seq // (tp if sp_split else 1)), dtype=DataType.BFLOAT16,
                       requires_grad=False, label=TensorLabel.ACTIVATION, name="indices")
    else:
        rows = local_seq // (tp if opts.sequence_parallel else 1)
        x = MockTensor((1, 1, rows, config.embedding_dim), dtype=DataType.BFLOAT16, requires_grad=True,
                       label=TensorLabel.ACTIVATION, name="received_activation")
    out = stage.forward_blocks(ctx, x)
    if ctx.lean_captures:
        # Target: a tensor no backward reads is freed once its last reader is done; only an op's
        # inputs list (the autograd edges) still points at it. The stage output is freed after the
        # send (its gradient comes back from the next stage).
        gc.collect()
        for t in gc.get_objects():
            if (isinstance(t, MockTensor) and t._tracked and t.label == TensorLabel.ACTIVATION
                    and t not in ctx.saved_ids and t is not x and not (t is out and stage.is_last)):
                t.untrack()
    # the peak is measured from here on: before, the timeline still holds what the scan freed
    peak_from = len(ctx.memory_tracker._events) if ctx.lean_captures else 0
    # The micro-batches held at once by the schedule hold only the blocks' activations: the LM head
    # and loss run and free theirs within one micro-batch (they are in peak_bytes)
    mem_after_blocks, _ = ctx.memory_tracker.current_memory()
    logits = stage.forward_head(ctx, out)
    roots: List[MockTensor] = []
    if logits is not None:
        roots.append(_loss(ctx, logits, local_seq, tp, opts.plain_loss))
    if not stage.is_last:
        roots.append(out)  # the gradient arrives from the next stage
    output_bytes = out.bytes()

    for root in roots:
        root.backward(ctx, retain_graph=False)
    del root
    # one block's gradient: the in-flight sync buffer when the sync overlaps the last backward
    grad_bytes = sum(p.get_grad().bytes() for n, p in params.items()
                     if n.startswith("blocks.") and p.is_grad_initialized()) // max(stage_blocks, 1)
    first_sync = len(ctx.estimates)
    if stage.is_first and sp_split:
        # each TP device saw only its own tokens: sum the replicated table's partial gradients over TP
        synchronize_replica_gradients(ctx, {"tok_emb.weight": params["tok_emb.weight"]}, [("tp", tp)])
    synchronize_replica_gradients(ctx, params, replica_groups, zero1=opts.zero1)
    # The embedding table's gradient is final only after the embedding backward, the stage's last op
    tail = [e for e in ctx.estimates[first_sync:] if "tok_emb" in e.operation and e.phase == "ccl"]
    tail_sync_ms = sum(max(e.ideal_compute_ns, e.ideal_memory_ns, e.ideal_network_ns) for e in tail) / 1e6
    tail_dram_ms = sum(e.ideal_memory_ns for e in tail) / 1e6
    optimizer.step(ctx)
    if opts.clip:
        mock_clip_grad_norm(ctx, params, max_norm=1.0, num_shards=optimizer.num_shards)
    del x, out, logits, roots  # release the micro-batch's tensors before reading the state
    gc.collect()
    state_bytes, _ = ctx.memory_tracker.current_memory()
    timeline = ctx.memory_tracker.get_memory_timeline()[max(peak_from - 1, 0):]
    peak_bytes = max(s.total_bytes for s in timeline)
    del mask  # resident for the whole step: in the state and the peak
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
        opt_dram_ms=ctx.dram_time_ns(OPTIMIZER_PHASES + SYNC_PHASES + GATHER_PHASES) / 1e6,
        sync_ms=op_ms(SYNC_PHASES),
        sync_fabric_ms=ctx.network_time_ns(SYNC_PHASES + GATHER_PHASES) / 1e6,
        gather_ms=op_ms(GATHER_PHASES),
        sync_dram_ms=ctx.dram_time_ns(SYNC_PHASES) / 1e6 - tail_dram_ms,
        tail_sync_ms=tail_sync_ms,
        grad_bytes=grad_bytes,
        activation_bytes=mem_after_blocks - mem_before,
        state_bytes=state_bytes,
        peak_bytes=peak_bytes,
        output_bytes=output_bytes,
    )


def _loss(ctx: RooflineContext, logits: MockTensor, local_seq: int, tp: int, plain: bool) -> MockTensor:
    """tt-train's loss: vocab-parallel cross-entropy over TP (losses.cpp), or its fused
    cross_entropy_loss on the local logits (TP1, and today's pipeline trainer, which calls it on the
    vocabulary-sharded logits without any collective)."""
    targets = MockTensor((1, 1, 1, local_seq), dtype=DataType.BFLOAT16, requires_grad=False,
                         label=TensorLabel.ACTIVATION, name="targets")
    if plain or tp == 1:
        return MockCrossEntropyLossOp.apply(ctx, logits, targets)
    return MockVocabParallelCrossEntropyLossOp.apply(ctx, logits, targets, tp)


def run_parallel_roofline(
    model_name: str = "llama-70b",
    seq_len: int = 8192,
    num_microbatches: int = 64,
    hardware: str = "bh_quad",
    pp: int = 1,
    tp: int = 1,
    cp: int = 1,
    dp: int = 1,
    schedule: str = "1f1b",
    interleave: int = 1,
    mode: str = "target",
    sequence_parallel: bool = False,
    recompute: str = "none",
    master_dtype: str = "bf16",
    grad_dtype: str = "bf16",
    zero1: bool = False,
    clip: bool = True,
    cut_links: Optional[int] = None,
    attn_block: Optional[int] = None,
    num_links: Optional[int] = None,
    topology: Optional[str] = None,
    blocks_per_stage: Optional[Sequence[int]] = None,
    stage_cache: Optional[Dict[Any, StageCost]] = None,
) -> Dict[str, Any]:
    """Roofline of one optimizer step of `num_microbatches` sequences on dp * pp * tp * cp chips.

    Args:
        model_name: Llama preset from training.MODEL_PRESETS
        seq_len: Tokens per sequence (one sequence per micro-batch)
        num_microbatches: m, sequences per optimizer step (m / dp per pipeline copy)
        hardware: name in HARDWARE_PRESETS (default: bh_quad)
        pp, tp, cp, dp: pipeline, tensor, context and data parallel degrees
        schedule: pipeline schedule: "naive", "gpipe", "1f1b" or "interleaved"
        interleave: v, model chunks per pipeline device (interleaved schedule only)
        mode: "current" (tt-train today) or "target" (planned kernels and features); see ATTN_BLOCK
        sequence_parallel: Megatron sequence parallelism inside TP
        recompute: "none", "selective" (norms, activations and attention recomputed in backward;
            matmul outputs held) or "full" (each block's forward, the memory-efficient runner)
        master_dtype, grad_dtype: "bf16" or "fp32": stored weights with m and v; gradient accumulation
        zero1: shard the optimizer state over the CP x DP replicas (target only)
        cut_links: Ethernet links (per direction) across the cut between two neighbouring stages;
            default from the stage layout (place_stages), e.g. a quad (32 x 4 torus) with stages
            of whole rows of 4 chips: 4 chip pairs x 2 links = 8
        attn_block: tokens per attention block, see fused_attention_roofline(block); default from mode
        num_links, topology: override the hardware's links per neighbour; topology forces one ring or
            line topology on every group instead of deriving it from the placement
        blocks_per_stage: transformer blocks on each pipeline device (as tt-train's
            PipelineParallelConfig.blocks_per_rank); default an even split, the remainder on the
            first devices (the last one holds the LM head)
        stage_cache: optional dict reused across calls (stage runs do not depend on m or the schedule)

    Returns:
        Result dict, or {"error": str}
    """
    if min(dp, pp, tp, cp, interleave, num_microbatches) < 1:
        return {"error": "dp, pp, tp, cp, interleave and the number of micro-batches must be >= 1"}
    preset = MODEL_PRESETS.get(model_name)
    if preset is None or preset["model_type"] != ModelType.LLAMA:
        return {"error": f"Unknown Llama preset: {model_name}"}
    if hardware not in HARDWARE_PRESETS:
        return {"error": f"Unknown hardware: {hardware}. Available: {', '.join(HARDWARE_PRESETS)}"}
    if mode not in ATTN_BLOCK:
        return {"error": f"mode must be one of {', '.join(ATTN_BLOCK)}"}
    hw = HARDWARE_PRESETS[hardware]
    overrides = {k: v for k, v in (("num_links", num_links), ("topology", topology)) if v is not None}
    if overrides:
        hw = dataclasses.replace(hw, **overrides)
    chips = dp * pp * tp * cp
    L = preset["num_blocks"]
    try:
        sched = make_schedule(schedule, interleave)
    except ValueError as exc:
        return {"error": str(exc)}
    blocks = list(blocks_per_stage) if blocks_per_stage is not None else even_split(L, pp)
    m = num_microbatches // dp if num_microbatches % dp == 0 else 0
    schedule_error = sched.validate(pp, m, blocks) if pp > 1 and m else ""
    target_only = [TARGET_ONLY[k] for k, on in (("zero1", zero1), ("selective", recompute == "selective"),
                                                 ("sequence_parallel", sequence_parallel),
                                                 ("grad_fp32", grad_dtype == "fp32")) if on]
    current = mode == "current"
    checks = [
        (not (current and target_only),
         f"{' and '.join(target_only)}: not in tt-train today, only with mode target"),
        # tt-train today: the pipeline runs one micro-batch at a time; DP exists only inside a mesh;
        # ring_shift needs an even ring (ttnn_fixed/distributed/ttnn_ops.cpp)
        (not (current and pp > 1 and schedule != "naive"),
         "tt-train's pipeline runs one micro-batch at a time today: mode current needs --schedule naive"),
        (not (current and pp > 1 and dp > 1), "tt-train has no data parallelism across pipelines today"),
        (not (current and pp == 1 and sum(n > 1 for n in (dp, cp, tp)) > 2),
         "tt-train's ParallelismContext takes at most two of DP, CP, TP on a 2D mesh"),
        (not (current and cp > 1 and cp % 2), f"tt-train's ring attention needs an even CP ring, got CP{cp}"),
        (recompute in RECOMPUTE, f"recompute must be one of {', '.join(RECOMPUTE)}"),
        (master_dtype in DTYPES and grad_dtype in DTYPES, f"dtypes must be one of {', '.join(DTYPES)}"),
        (chips <= hw.chips_per_galaxy, f"dp*pp*tp*cp = {chips} > {hw.chips_per_galaxy} chips of {hw.name}"),
        (m > 0, f"m = {num_microbatches} is not divisible by dp = {dp}"),
        (len(blocks) == pp, f"blocks_per_stage has {len(blocks)} entries, pp is {pp}"),
        (sum(blocks) == L and min(blocks) >= 1,
         f"blocks_per_stage {blocks} must be >= 1 each and sum to num_blocks ({L})"),
        (not schedule_error, schedule_error),
        (preset["num_groups"] % tp == 0, f"num_groups ({preset['num_groups']}) not divisible by tp ({tp})"),
        # target's load-balanced ring attention gives each device two chunks; current is contiguous
        (seq_len % ((2 if mode == "target" else 1) * cp * (tp if sequence_parallel else 1)) == 0,
         f"seq_len not divisible by {'2 * ' if mode == 'target' else ''}cp (* tp with SP)"),
    ]
    for ok, msg in checks:
        if not ok:
            return {"error": msg}
    placement = place_stages(hw, pp, tp, cp, dp, interleaved=sched.interleave > 1)
    if "error" in placement:
        return placement
    if current and placement.get("tp_on_rows"):
        return {"error": "tt-train puts CP on the first mesh axis (rows) and TP on the second"}
    group_topology = placement["topology"] if topology is None else {g: topology for g in ("tp", "cp", "dp")}
    if current and pp > 1 and topology is None:
        # tt-train's multi-host pipeline descriptors declare LINE axes (ring only on a TORUS axis);
        # single-galaxy descriptors include torus ones, so with PP1 the placement's topology holds
        group_topology = {g: "linear" for g in group_topology}
    if cut_links is None:
        # today's hand-off: one socket per chip pair across the cut, on one link (socket_manager.cpp)
        cut_links = placement["cut_links"] // (hw.num_links if current else 1)
    opts = Options(sequence_parallel=sequence_parallel,
                   recompute=recompute, cp_mode=mode,
                   attn_block=attn_block if attn_block is not None else ATTN_BLOCK[mode],
                   master_dtype=master_dtype, grad_dtype=grad_dtype, zero1=zero1, accumulate=m > 1,
                   explicit_mask=current and pp > 1 and cp == 1, plain_loss=current and pp > 1, clip=clip)

    cache = stage_cache if stage_cache is not None else {}

    def role(d: int) -> int:
        # first, middle or last: what a stage holds besides its blocks
        return 0 if d == 0 else (pp - 1 if d == pp - 1 else 1)

    def stage_cost(d: int) -> StageCost:
        key = (model_name, seq_len, hardware, hw.num_links, tuple(sorted(group_topology.items())),
               tuple(sorted(placement["link_share"].items())), pp, tp, cp, dp, role(d), blocks[d], opts)
        if key not in cache:
            cache[key] = run_stage(hw, preset, seq_len, tp, cp, dp, pp, role(d), blocks[d], opts,
                                   group_topology, placement["link_share"])
        return cache[key]

    costs = [stage_cost(d) for d in range(pp)]

    # Stage hand-off: every device of the stage sends its part of the activation (forward) or its
    # gradient (backward) across the cut; without SP every TP chip holds the full activation.
    # Per micro-batch a device sends v activations and v gradients. In the roofline each send is its
    # own op on the device (2v ops); in the upper bound the two directions of the full-duplex links
    # overlap, so v hand-offs count against the fabric.
    v = sched.interleave
    handoff_bytes = costs[0].output_bytes * tp * cp
    hand = (send_recv_roofline(hw, handoff_bytes, cut_links, operation="PipelineHandoff").ideal_network_ns / 1e6
            if pp > 1 else 0.0)

    # hand-offs a device sends per micro-batch: v forward (v - 1 on the last device), v backward
    # (v - 1 on the first)
    sends_f = [v - (d == pp - 1) for d in range(pp)]
    sends_b = [v - (d == 0) for d in range(pp)]

    def device_time(d: int, c: StageCost) -> Dict[str, float]:
        flop, dram, fabric = c.mb_flop_ms, c.mb_dram_ms, c.mb_fabric_ms + max(sends_f[d], sends_b[d]) * hand
        return {"flop_ms": flop, "dram_ms": dram, "fabric_ms": fabric,
                "roofline_ms": c.mb_forward_ms + c.mb_backward_ms + (sends_f[d] + sends_b[d]) * hand,
                "upper_bound_ms": max(flop, dram, fabric)}

    per_stage = [device_time(d, c) for d, c in enumerate(costs)]
    D = max(per_stage, key=lambda p: p["roofline_ms"])

    # Play the schedule out with each device's own forward and backward time per micro-batch.
    # With equal devices this is slots(P, m) x D; uneven stages set the pace with the slowest and
    # fill/drain with the sum.
    fwd = [c.mb_forward_ms + sends_f[d] * hand for d, c in enumerate(costs)]
    bwd = [c.mb_backward_ms + sends_b[d] * hand for d, c in enumerate(costs)]
    bubble, held, pipeline_ms = simulate_schedule(sched, pp, m, fwd, bwd)
    # the same schedule with every device at the average: what fill and drain alone cost
    fill_bubble = simulate_schedule(sched, pp, m, [sum(fwd) / pp] * pp, [sum(bwd) / pp] * pp)[0] if pp > 1 else 0.0

    def split(total: float, f: float, b: float):
        r = f / (f + b) if f + b > 0 else 1 / 3
        return total * r, total * (1 - r)

    ub = [split(p["upper_bound_ms"], c.mb_forward_bound_ms, c.mb_backward_bound_ms) for p, c in zip(per_stage, costs)]
    _, _, pipeline_bound_ms = simulate_schedule(sched, pp, m, [f for f, _ in ub], [b for _, b in ub])

    # Each layer's gradient sync starts as soon as its gradient is final (after the last micro-batch's
    # backward through it) and the syncs queue on the fabric; what is left after the last backward is
    # exposed, then the optimizer and (ZeRO-1) the parameter all-gather; the step waits for the slowest
    # stage
    overlap = not current  # tt-train syncs after the last backward today
    step_roofline = pipeline_ms + max(exposed_sync(c, blocks[d], overlap) + c.opt_ms + c.gather_ms
                                      for d, c in enumerate(costs))
    link_total = max(m * p["fabric_ms"] + c.sync_fabric_ms for p, c in zip(per_stage, costs))
    step_bound = max(pipeline_bound_ms + max(max(c.opt_flop_ms, c.opt_dram_ms) for c in costs), link_total)

    # Memory per device: the one-micro-batch run's peak (state, its activations, backward
    # transients) plus the other micro-batches the schedule makes it hold at the same time
    # With interleaving a chunk's received input is the tensor the whole-stage run keeps at that block
    # boundary, and its output is freed after the send: no extra. With the sync overlapped, one
    # layer's gradient is in flight, only during the last micro-batch, when the device holds one.
    # Today's sockets keep an 80 MB receive FIFO per peer (socket_manager.cpp): the pipeline
    # neighbours (two, the first and last stages linked for the targets; one with PP2) and, with CP,
    # ring_shift's sockets in each direction.
    fifos = 80e6 * ((min(pp - 1, 2) if pp > 1 else 0) + (2 if cp > 1 else 0)) if current else 0
    device_memory = [max(c.peak_bytes + (held[d] - 1) * c.activation_bytes,
                         c.peak_bytes + (c.grad_bytes if overlap and c.sync_ms > 0 else 0)) + fifos
                     for d, c in enumerate(costs)]
    worst = max(range(pp), key=lambda d: device_memory[d])
    memory = device_memory[worst]

    tokens = num_microbatches * seq_len
    # MFU = ideal time / step time: the ideal time runs the model's FLOPs (closed form, causal
    # attention) on every chip at peak, the fidelity the roofline times matmuls at (HiFi4)
    peak = hw.tflops_per_chip(MathFidelity.HiFi4) * 1e12
    model_flops = model_flops_per_token(preset, seq_len) * tokens
    ideal_ms = model_flops / (chips * peak) * 1e3

    def throughput(step_ms: float) -> Dict[str, float]:
        return {"step_ms": step_ms, "tokens_per_second": tokens / (step_ms / 1e3), "mfu": ideal_ms / step_ms}

    bound = max(("flop", "dram", "fabric"), key=lambda k: D[f"{k}_ms"])
    return {
        "config": {"model": model_name, "seq_len": seq_len, "num_microbatches": num_microbatches,
                   "hardware": hw.name, "num_links": hw.num_links, "topology": group_topology,
                   "stage_mesh": placement["stage_mesh"], "chips": chips, "dp": dp, "pp": pp, "tp": tp,
                   "cp": cp, "schedule": str(sched), "interleave": v, "mode": mode, "cut_links": cut_links,
                   **dataclasses.asdict(opts), "blocks_per_stage": blocks},
        "per_microbatch": {**D, "bound": bound, "handoff_ms": hand},
        "pipeline_ms": pipeline_ms,
        "step_end": {"grad_sync_ms": max(c.sync_ms for c in costs),
                     "grad_sync_exposed_ms": max(exposed_sync(c, blocks[d], overlap) for d, c in enumerate(costs)),
                     "optimizer_ms": max(c.opt_ms for c in costs),
                     "param_gather_ms": max(c.gather_ms for c in costs)},
        "bubble": bubble,
        "bubble_fill": fill_bubble,
        "ideal_ms": ideal_ms,
        "peak": {"tflops_per_chip": peak / 1e12, "fidelity": "HiFi4", "cores": hw.tensix_cores_per_chip,
                 "clock_ghz": hw.clock_ghz},
        "roofline": throughput(step_roofline),
        "upper_bound": throughput(step_bound),
        "memory": {"peak_gb": memory / 1e9, "fits": memory <= hw.dram_gb_per_chip * 1e9, "device": worst,
                   "activation_gb_per_mb": costs[worst].activation_bytes / 1e9,
                   "peak_activations": held[worst], "state_gb": costs[worst].state_bytes / 1e9},
    }


def exposed_sync(c: StageCost, num_blocks: int, overlap: bool = True) -> float:
    """Gradient sync left after the stage's last backward, with each of its k layers' syncs started as
    soon as that layer's last backward ends and the syncs run one after another: with b the last
    micro-batch's backward, t = b / k per layer, S the sync and s = S / k per layer, layer i's sync
    ends at F_i = max(F_(i-1), i t) + s, so the exposed part is F_k - b = max(s, S - b + t). Only the
    last layer's sync is exposed when s <= t; a link-bound sync (s > t) stays mostly exposed."""
    S = c.sync_ms
    if S <= 0 or not overlap:
        return S
    # the embedding table's sync (first stage) can only start after the last op: fully exposed;
    # the layers' syncs read and write DRAM, which the backward they overlap also uses
    T = min(c.tail_sync_ms, S)
    S -= T
    k = max(num_blocks, 1)
    b = max(c.mb_backward_ms - c.sync_dram_ms, 0.0)
    return T + (max(S / k, S - b + b / k) if S > 0 else 0.0)


def model_flops_per_token(preset: dict, seq_len: int) -> float:
    """Training FLOPs per token in closed form: 6 * N + 6 * L * E * S.

    N counts the matmul weights as tt-train's ttml/models/llama/flops.py does (per block q, k, v, o,
    gate, up, down and the norms; plus the LM head; the embedding lookup does no matmul). Attention
    (QK^T and AV, forward and backward) is counted over the causal half, 6 * L * H * d * S with
    H * d = E; tt-train's flops.py counts the full square, 12 * L * H * d * S.
    """
    E, L, V = preset["embedding_dim"], preset["num_blocks"], preset["vocab_size"]
    head_dim = E // preset["num_heads"]
    kv = preset["num_groups"] * head_dim
    ff = preset.get("intermediate_dim")
    if ff is None:  # as tt-train's LlamaMLP: 8E/3 rounded up to a multiple of 256
        ff = ((4 * E * 2) // 3 + 255) // 256 * 256
    n = L * (2 * E * E + 2 * E * kv + E + 3 * E * ff + E) + V * E + E
    return 6 * n + 6 * L * E * seq_len


def place_stages(hw, pp: int, tp: int, cp: int, dp: int = 1, interleaved: bool = False) -> Dict[str, Any]:
    """Lay pipeline stages along the first mesh axis and TP / CP on whole axes of each stage's block.

    The hardware's mesh (hw.mesh_shape, rows x columns, hw.mesh_wrap) is cut into dp * pp equal blocks
    of tp * cp chips: whole rows when a stage spans at least one row, else parts of a row. The dp copies
    of the pipeline sit one after the other along the rows, so hand-offs stay between neighbours and a
    DP group's members are a whole pipeline (pp blocks) apart. tt-train's collectives run over whole
    mesh axes (ttnn_fixed/distributed/ttnn_ops.cpp), so TP and CP must each be one axis of the block
    (one of them may be 1 on a one-row block); TP takes the column axis when it fits. A group's
    collectives are a ring when it fills a whole wrapped axis of the mesh and has more than 2 chips
    (tt-train's get_topology), else a line.

    DP groups: every chip position of a pipeline forms one, and all of them run their all-reduce
    along the same column links at once, so each gets 1 / (pp * rows per stage) of a link; neighbouring
    members are that many hops apart. When the copies fill the wrapped axis the group is a ring, even
    with two copies (half of each transfer goes each way round), else a line.

    Returns {"topology": {"tp", "cp", "dp"}, "link_share": {"dp": ...}, "cut_links", "stage_mesh"}
    or {"error": str}.
    """
    R, C = hw.mesh_shape
    wrap_r, wrap_c = hw.mesh_wrap
    n = tp * cp
    if R * C == 1:  # single chip / unspecified mesh: keep the hardware's own topology
        return {"topology": {}, "link_share": {}, "cut_links": hw.num_links, "stage_mesh": (1, n)}
    if dp * pp * n > R * C:
        return {"error": f"dp*pp*tp*cp = {dp * pp * n} > {R * C} chips of {hw.name}"}
    if n % C == 0:
        block = (n // C, C)
    elif C % n == 0 and dp == 1:
        block = (1, n)
    else:
        return {"error": f"a stage of {n} chips is neither whole rows nor part of a row of {C}"
                         + (" (DP needs stages of whole rows)" if dp > 1 else "")}
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
        # one group over the whole block; on more than one row it is a line folded over the block
        # (tt-train's mesh graph descriptor declares e.g. a 4 x 2 block as a [1, 8] mesh)
        tp_axis = cp_axis = col_axis if rows == 1 else (rows * cols, False, False)
    else:
        tp_axis = cp_axis = col_axis
    if interleaved and cols != C:
        return {"error": "interleaved needs stages of whole rows (the last stage must neighbour the first)"}
    if interleaved and not (wrap_r and pp * rows == R):
        return {"error": "interleaved needs the last stage to neighbour the first (one pipeline filling a wrapped axis)"}
    pairs = cols if rows >= 1 and cols == C else 1  # chip pairs across the cut between two stages
    return {"tp_on_rows": tp > 1 and cp > 1 and tp_axis is row_axis,
            "topology": {"tp": topo(tp, *tp_axis[1:]) if tp > 1 else "linear",
                         "cp": topo(cp, *cp_axis[1:]) if cp > 1 else "linear",
                         # copies filling the wrapped axis form a ring, even two of them: the halves of
                         # each transfer go both ways round (opposite arcs)
                         "dp": "ring" if dp > 1 and dp * pp * rows == R and wrap_r else "linear"},
            "link_share": {"dp": pp * rows} if dp > 1 else {},
            "cut_links": pairs * hw.num_links, "stage_mesh": block}


def even_split(num_blocks: int, pp: int, lighten_ends: int = 0) -> List[int]:
    """Blocks per pipeline device: as even as possible, the remainder on the middle devices (the first
    holds the embedding, the last the LM head). lighten_ends > 0 moves that many more blocks off the
    first and the last device each, onto the middle ones."""
    base, extra = divmod(num_blocks, pp)
    blocks = [base] * pp
    ends = [0, pp - 1] if pp > 1 else []
    for _ in range(lighten_ends):
        for d in ends:
            if blocks[d] > 1:
                blocks[d] -= 1
                extra += 1
    middle = list(range(1, pp - 1)) or list(range(pp))
    i = 0
    while extra > 0:
        blocks[middle[i % len(middle)]] += 1
        extra -= 1
        i += 1
    return blocks


SWEEP_SCHEDULES = (("1f1b", 1), ("interleaved", 2), ("interleaved", 4), ("interleaved", 5), ("interleaved", 10))
DEGREES = (1, 2, 4, 8, 16, 32)


def sweep(chips: int, schedules=SWEEP_SCHEDULES, **kwargs) -> List[Dict[str, Any]]:
    """Every split of `chips` into dp * pp * tp * cp, schedule and recompute level that is valid; for
    each, the even layer split and the one with a block moved off the first and the last device."""
    cache: Dict[Any, StageCost] = {}
    results = []
    num_blocks = MODEL_PRESETS[kwargs.get("model_name", "llama-70b")]["num_blocks"]
    for tp, cp, dp in itertools.product(DEGREES, DEGREES, DEGREES):
        if chips % (tp * cp * dp):
            continue
        pp = chips // (tp * cp * dp)
        current = kwargs.get("mode") == "current"
        levels = ("none", "full") if current else RECOMPUTE
        for (schedule, v), recompute in itertools.product((("naive", 1),) if current else schedules, levels):
            if pp == 1 and schedule not in ("1f1b", "naive"):
                continue
            splits = [even_split(num_blocks, pp)]
            lighter = even_split(num_blocks, pp, lighten_ends=1)
            if pp > 2 and lighter != splits[0]:
                splits.append(lighter)
            for blocks in splits:
                r = run_parallel_roofline(pp=pp, tp=tp, cp=cp, dp=dp, schedule=schedule, interleave=v,
                                          recompute=recompute, blocks_per_stage=blocks, stage_cache=cache, **kwargs)
                if "error" not in r:
                    results.append(r)
    return results


def _describe(r: Dict[str, Any]) -> str:
    c = r["config"]
    sched = "" if c["pp"] == 1 else f", {c['schedule']}"
    dp = f"DP{c['dp']} x " if c["dp"] > 1 else ""
    rc = "" if c["recompute"] == "none" else f", {c['recompute']} recompute"
    b = c["blocks_per_stage"]
    split = f", blocks {b[0]}/{b[1] if len(b) > 2 else ''}../{b[-1]}" if len(set(b)) > 1 else ""
    return f"{dp}PP{c['pp']} x TP{c['tp']} x CP{c['cp']}{sched}{rc}{split}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="llama-70b", help="Llama preset (default: llama-70b)")
    parser.add_argument("-s", "--seq", type=int, default=8192, help="Sequence length (default: 8192)")
    parser.add_argument("-m", "--microbatches", type=int, default=64, help="Sequences per optimizer step")
    parser.add_argument("--hardware", default="bh_quad", choices=list(HARDWARE_PRESETS))
    parser.add_argument("--pp", type=int, default=1, help="Pipeline stages")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel degree")
    parser.add_argument("--cp", type=int, default=1, help="Context parallel degree (ring attention)")
    parser.add_argument("--dp", type=int, default=1, help="Data parallel copies of the pipeline")
    parser.add_argument("--mode", default="target", choices=list(ATTN_BLOCK),
                        help="current: tt-train today; target: planned kernels and features (default)")
    parser.add_argument("--schedule", default="1f1b", choices=("naive", "gpipe", "1f1b", "interleaved"),
                        help="Pipeline schedule (default: 1f1b)")
    parser.add_argument("--interleave", type=int, default=2,
                        help="Chunks per pipeline device for --schedule interleaved (default: 2)")
    parser.add_argument("--sp", action="store_true", help="Sequence parallelism inside TP")
    parser.add_argument("--recompute", default="none", choices=RECOMPUTE,
                        help="selective (target only): norms, activations, attention; full: whole blocks "
                             "(default: none)")
    parser.add_argument("--master-dtype", default="bf16", choices=list(DTYPES),
                        help="Stored weights, m and v (tt-train precision.master_dtype; default bf16)")
    parser.add_argument("--grad-dtype", default="bf16", choices=list(DTYPES),
                        help="Gradient accumulation across micro-batches (default bf16)")
    parser.add_argument("--zero1", action="store_true", help="Shard optimizer state over CP x DP (target only)")
    parser.add_argument("--no-clip", action="store_true", help="No gradient-norm clipping (tt-train's use_clip_grad_norm: false)")
    parser.add_argument("--cut-links", type=int, default=None,
                        help="Links across a stage boundary (default: from the stage layout)")
    parser.add_argument("--attn-block", type=int, default=None,
                        help="Tokens kept on chip per attention block (default: 32 current, 512 target)")
    parser.add_argument("--blocks-per-stage", default=None, metavar="N,N,...",
                        help="Transformer blocks on each pipeline device (default: even split, remainder first)")
    parser.add_argument("--links", type=int, default=None, help="Override links per neighbouring chip pair")
    parser.add_argument("--topology", choices=("ring", "linear"), default=None,
                        help="Force one topology on every group (default: derived from the placement)")
    parser.add_argument("--sweep", type=int, default=None, metavar="CHIPS",
                        help="Evaluate every DP x PP x TP x CP split of CHIPS chips and print the best that fit")
    parser.add_argument("--top", type=int, default=10, help="Rows to print with --sweep")
    parser.add_argument("--output", default=None,
                        help="Write the result (with the other mode's under 'other_mode'), or the sweep, as JSON")
    args = parser.parse_args()
    if args.sweep:
        ignored = [f for f, on in (("--pp/--tp/--cp/--dp", (args.pp, args.tp, args.cp, args.dp) != (1, 1, 1, 1)),
                                   ("--schedule/--interleave", args.schedule != "1f1b" or args.interleave != 2),
                                   ("--recompute", args.recompute != "none"),
                                   ("--blocks-per-stage", args.blocks_per_stage is not None)) if on]
        if ignored:
            print(f"note: --sweep tries every split, schedule and recompute level; ignoring {', '.join(ignored)}")

    common = dict(model_name=args.model, seq_len=args.seq, num_microbatches=args.microbatches,
                  hardware=args.hardware, sequence_parallel=args.sp,
                  master_dtype=args.master_dtype, grad_dtype=args.grad_dtype, zero1=args.zero1, mode=args.mode,
                  clip=not args.no_clip,
                  cut_links=args.cut_links, attn_block=args.attn_block, num_links=args.links, topology=args.topology)
    if args.sweep:
        target_only = [TARGET_ONLY[k] for k, on in (("zero1", args.zero1), ("sequence_parallel", args.sp),
                                                     ("grad_fp32", args.grad_dtype == "fp32")) if on]
        if args.mode == "current" and target_only:
            raise SystemExit(f"{' and '.join(target_only)}: not in tt-train today, only with mode target")
        results = sweep(args.sweep, **common)
        fits = sorted((r for r in results if r["memory"]["fits"]), key=lambda r: -r["roofline"]["tokens_per_second"])
        print(f"mode {args.mode}: {len(fits)} of {len(results)} configurations fit in memory")
        print(f"{'configuration':52s} {'tokens/s':>10s} {'MFU':>7s} {'bound':>7s} {'bubble':>7s} {'mem GB':>7s}")
        for r in fits[: args.top]:
            print(f"{_describe(r):52s} {r['roofline']['tokens_per_second']:10,.0f} {r['roofline']['mfu']:7.1%} "
                  f"{r['per_microbatch']['bound']:>7s} {r['bubble']:7.1%} {r['memory']['peak_gb']:7.1f}")
        out = results
    else:
        blocks = [int(b) for b in args.blocks_per_stage.split(",")] if args.blocks_per_stage else None
        single = dict(pp=args.pp, tp=args.tp, cp=args.cp, dp=args.dp, schedule=args.schedule,
                      interleave=args.interleave if args.schedule == "interleaved" else 1,
                      recompute=args.recompute, blocks_per_stage=blocks, **common)
        out = run_parallel_roofline(**single)
        if "error" in out:
            raise SystemExit(out["error"])
        _print(out, args)
        # The other mode, for comparison, with that mode's own attention block (a given
        # --attn-block applies to the chosen mode only); current drops the target-only features
        other = "current" if args.mode == "target" else "target"
        cfg = out["config"]
        overrides, dropped = {"attn_block": None}, []
        if other == "current":
            for key, value, on in (("zero1", False, cfg["zero1"]), ("sequence_parallel", False, cfg["sequence_parallel"]),
                                   ("grad_dtype", "bf16", cfg["grad_dtype"] == "fp32"),
                                   ("recompute", "full", cfg["recompute"] == "selective")):
                if on:
                    overrides[key] = value
                    dropped.append({"grad_dtype": "fp32 gradient accumulation", "recompute": "selective recompute (full)",
                                    "zero1": "ZeRO-1", "sequence_parallel": "sequence parallelism"}[key])
            if cfg["pp"] > 1 and cfg["schedule"] != "naive":
                overrides.update(schedule="naive", interleave=1)
                dropped.append(f"{cfg['schedule']} (naive schedule)")
        alt = run_parallel_roofline(**{**single, "mode": other, **overrides})
        if "error" in alt:
            print(f"  {other}: {alt['error']}")
        else:
            a = alt["roofline"]
            print(f"  {other} (attention block {alt['config']['attn_block'] or 32})"
                  f"{' without ' + ' and '.join(dropped) if dropped else ''}: "
                  f"{a['tokens_per_second']:,.0f} tokens/s, MFU {a['mfu']:.1%}, "
                  f"memory {alt['memory']['peak_gb']:.1f} GB{'' if alt['memory']['fits'] else ' (does not fit)'}")
        out["other_mode"] = alt  # the comparison run, same shape
    if args.output:
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2, default=str)


def _print(out: Dict[str, Any], args) -> None:
    cfg, pm, o, ub, mem = out["config"], out["per_microbatch"], out["roofline"], out["upper_bound"], out["memory"]
    print(f"{_describe(out)} on {cfg['chips']} chips ({cfg['hardware']}), {args.microbatches} x {args.seq} "
          f"tokens per step, mode {cfg['mode']}")
    parts = [f"{g.upper()} {t}" for g, t in cfg["topology"].items() if cfg[g] > 1]
    print(f"  stage mesh {cfg['stage_mesh'][0]} x {cfg['stage_mesh'][1]}, "
          + ", ".join(parts + [f"{cfg['cut_links']} links across a stage cut"]))
    print(f"  precision: {cfg['master_dtype']} master, {cfg['grad_dtype']} gradient accumulation"
          f"{', ZeRO-1' if cfg['zero1'] else ''}; attention block {cfg['attn_block'] or 32}, ring attention {cfg['cp_mode']}")
    print(f"  per micro-batch, slowest stage: {pm['roofline_ms']:.2f} ms (sum of FLOP {pm['flop_ms']:.2f}, "
          f"DRAM {pm['dram_ms']:.2f}, fabric {pm['fabric_ms']:.2f} ms; hand-off {pm['handoff_ms']:.3f} ms) "
          f"-> mostly {pm['bound']}")
    if len(set(cfg["blocks_per_stage"])) > 1:
        print(f"  blocks per stage: {cfg['blocks_per_stage']}")
    print(f"  bubble {out['bubble']:.2%} (fill and drain {out['bubble_fill']:.2%}, "
          f"stage imbalance {out['bubble'] - out['bubble_fill']:.2%})")
    se = out["step_end"]
    overlapped = cfg["mode"] == "target"
    print(f"  after the pipeline: gradient sync {se['grad_sync_ms']:.1f} ms, {se['grad_sync_exposed_ms']:.1f} ms "
          f"exposed{' (per-layer syncs start during the last backward)' if overlapped else ''}, "
          f"optimizer {se['optimizer_ms']:.1f} ms"
          + (f", ZeRO-1 parameter all-gather {se['param_gather_ms']:.1f} ms" if se['param_gather_ms'] else ""))
    pk = out["peak"]
    print(f"  roofline (sum of per-op max): {o['tokens_per_second']:,.0f} tokens/s, step {o['step_ms']/1e3:.2f} s")
    print(f"  MFU {o['mfu']:.1%} = ideal {out['ideal_ms']/1e3:.2f} s / roofline {o['step_ms']/1e3:.2f} s; ideal: "
          f"6N + causal attention at {pk['tflops_per_chip']:.1f} TFLOP/s per chip "
          f"({pk['fidelity']}, {pk['cores']} cores, {pk['clock_ghz']} GHz) x {cfg['chips']} chips")
    print(f"  upper bound (max of summed resources): {ub['tokens_per_second']:,.0f} tokens/s, MFU {ub['mfu']:.1%}")
    print(f"  memory per chip: {mem['peak_gb']:.1f} GB on device {mem['device']} "
          f"({'fits' if mem['fits'] else 'does not fit'}; state {mem['state_gb']:.1f} GB + "
          f"{mem['peak_activations']:.1f} x {mem['activation_gb_per_mb']:.2f} GB held)")


if __name__ == "__main__":
    main()
