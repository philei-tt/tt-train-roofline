# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Roofline analysis for Llama training with pipeline, tensor, sequence, context and data parallelism.

Combines, on `chips = dp * pp * tp * cp` devices:
  - PP: a pipeline schedule (training/pipeline.py), optional vocabulary parallelism across stages
  - TP: tensor parallelism inside a stage, optionally with sequence parallelism (--sp)
  - CP: context parallelism inside a stage (ring attention)
  - DP: dp copies of the whole pipeline, each running m / dp micro-batches; optional ZeRO-1
  - activation recompute: none, selective (norms, activations, attention) or full (whole blocks)
  - precision: bf16 or fp32 master weights, bf16 or fp32 gradient accumulation

--mode picks what the kernels and features are: "current" is tt-train today (tile-wide attention
blocks, contiguous ring attention with unfused merges and socket shifts); "target" is the planned
implementation (512-token attention blocks, load-balanced ring attention with fused merges and
overlapped shifts) and is the only mode that allows the features tt-train does not have yet:
vocabulary parallelism across stages and ZeRO-1. A single run reports both modes.

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
    MockVocabParallelCrossEntropyLossOp,
    MockDistributedLlama,
    MockTensor,
    RooflineContext,
    TensorLabel,
    mock_clip_grad_norm,
    all_reduce_roofline,
    send_recv_roofline,
)
from roofline.operations.operation import create_grad_tensor
from roofline.examples.training import MODEL_PRESETS, ModelType, distributed_llama_config
from roofline.training.distributed import synchronize_replica_gradients
from roofline.training.pipeline import make_schedule, simulate_schedule

MICROBATCH_PHASES = ("forward", "backward")
OPTIMIZER_PHASES = ("optimizer", "grad_clip")
SYNC_PHASES = ("ccl",)  # gradient sync across the CP and DP groups

# What --mode sets (see the module docstring); features only the target mode allows
ATTN_BLOCK = {"current": None, "target": 512}  # per mode; the mode is also the ring-attention model
TARGET_ONLY = {"vocab_parallel": "vocabulary parallelism across pipeline stages", "zero1": "ZeRO-1"}
RECOMPUTE = ("none", "selective", "full")
DTYPES = {"bf16": DataType.BFLOAT16, "fp32": DataType.FLOAT32}


@dataclass(frozen=True)
class Options:
    """How a stage runs, apart from its parallel degrees (hashable: part of the stage cache key)."""

    sequence_parallel: bool = False
    vocab_parallel: bool = False  # LM head split across all pipeline stages (target only)
    recompute: str = "none"  # none | selective | full
    attn_block: Optional[int] = None  # tokens per attention block; None = one tile
    cp_mode: str = "target"  # ring attention: current | target
    master_dtype: str = "bf16"  # stored weights, m and v (tt-train precision.master_dtype)
    grad_dtype: str = "bf16"  # gradient accumulation across micro-batches
    zero1: bool = False  # optimizer state sharded over the CP x DP replicas (target only)


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
    output_bytes: int  # activation sent to the next stage, per device
    head_forward_ms: float = 0.0  # LM-head shard + loss, vocabulary parallelism only (part of mb_*)
    head_backward_ms: float = 0.0


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
                           group_topology=dict(group_topology), group_link_share=dict(link_share))


def _run_stage(hw, preset, seq_len, tp, cp, dp, pp, stage_index, stage_blocks, opts, group_topology,
               link_share) -> StageCost:
    ctx = _context(hw, opts, group_topology, link_share)
    config = distributed_llama_config(
        preset, tp_size=tp, cp_size=cp, sequence_parallel=opts.sequence_parallel,
        num_stages=pp, stage_index=stage_index, stage_num_blocks=stage_blocks, vocab_parallel=opts.vocab_parallel,
    )
    stage = MockDistributedLlama(config, runner="mem_eff" if opts.recompute == "full" else "default")
    params = stage.parameters()
    master = DTYPES[opts.master_dtype]
    replica_groups = [("cp", cp), ("dp", dp)]  # weights are replicated across CP and the DP copies
    optimizer = MockAdamW(params, master_dtype=master if master != DataType.BFLOAT16 else None,
                          num_shards=math.prod(n for _, n in replica_groups) if opts.zero1 else 1)
    local_seq = seq_len // cp
    # Steady state: every micro-batch after the first finds all gradients already allocated, adds
    # into them out of place (tt-train's Tensor::add_grad) and holds them next to its activations
    grad_dtype = DTYPES[opts.grad_dtype]
    for p in params.values():
        if p.requires_grad:
            p.add_grad(create_grad_tensor(p.shape, grad_dtype if p.dtype == DataType.BFLOAT16 else p.dtype,
                                          p.layout, name="accumulated_grad", num_shards=p.num_shards))
    mem_before, _ = ctx.memory_tracker.current_memory()

    if stage.is_first:
        x = MockTensor((1, 1, 1, local_seq), dtype=DataType.BFLOAT16, requires_grad=False,
                       label=TensorLabel.ACTIVATION, name="indices")
    else:
        rows = local_seq // (tp if opts.sequence_parallel else 1)
        x = MockTensor((1, 1, rows, config.embedding_dim), dtype=DataType.BFLOAT16, requires_grad=True,
                       label=TensorLabel.ACTIVATION, name="received_activation")
    out = stage.forward_blocks(ctx, x)
    # The micro-batches held at once by the schedule hold only the blocks' activations: the LM head
    # and loss run and free theirs within one micro-batch (they are in peak_bytes)
    mem_after_blocks, _ = ctx.memory_tracker.current_memory()
    logits = stage.forward_head(ctx, out)
    roots: List[MockTensor] = []
    if logits is not None:
        roots.append(_loss(ctx, logits, local_seq, tp))
    if not stage.is_last:
        roots.append(out)  # the gradient arrives from the next stage
    output_bytes = out.bytes()
    out_shape = out.shape

    for root in roots:
        root.backward(ctx, retain_graph=False)
    synchronize_replica_gradients(ctx, params, replica_groups, zero1=opts.zero1)
    optimizer.step(ctx)
    mock_clip_grad_norm(ctx, params, max_norm=1.0, num_shards=optimizer.num_shards)
    del x, out, logits, roots  # release the micro-batch's tensors before reading the state
    gc.collect()
    state_bytes, _ = ctx.memory_tracker.current_memory()
    peak_bytes, _ = ctx.memory_tracker.peak_memory()
    ctx.disable_memory_tracking()

    head_forward_ms = head_backward_ms = 0.0
    if opts.vocab_parallel:
        # The LM-head shard and loss alone, for the schedule's separate vocabulary pass
        head = _context(hw, opts, group_topology, link_share)
        head.disable_memory_tracking()
        hidden = MockTensor(out_shape, dtype=DataType.BFLOAT16, requires_grad=True,
                            label=TensorLabel.ACTIVATION, name="final_hidden")
        _loss(head, stage.forward_head(head, hidden), local_seq, tp).backward(head, retain_graph=False)
        head_forward_ms = head.op_time_ns(("forward",)) / 1e6
        head_backward_ms = head.op_time_ns(("backward",)) / 1e6

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
        output_bytes=output_bytes,
        head_forward_ms=head_forward_ms,
        head_backward_ms=head_backward_ms,
    )


def _loss(ctx: RooflineContext, logits: MockTensor, local_seq: int, tp: int) -> MockTensor:
    targets = MockTensor((1, 1, 1, local_seq), dtype=DataType.BFLOAT16, requires_grad=False,
                         label=TensorLabel.ACTIVATION, name="targets")
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
    vocab_parallel: bool = False,
    recompute: str = "none",
    master_dtype: str = "bf16",
    grad_dtype: str = "bf16",
    zero1: bool = False,
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
        vocab_parallel: split the LM head across all pipeline stages (target only)
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
    vocab_parallel = vocab_parallel and pp > 1
    target_only = [TARGET_ONLY[k] for k, on in (("vocab_parallel", vocab_parallel), ("zero1", zero1)) if on]
    checks = [
        (not (mode == "current" and target_only),
         f"{' and '.join(target_only)}: not in tt-train today, only with mode target"),
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
    group_topology = placement["topology"] if topology is None else {g: topology for g in ("tp", "cp", "dp")}
    if cut_links is None:
        cut_links = placement["cut_links"]
    opts = Options(sequence_parallel=sequence_parallel, vocab_parallel=vocab_parallel,
                   recompute=recompute, cp_mode=mode,
                   attn_block=attn_block if attn_block is not None else ATTN_BLOCK[mode],
                   master_dtype=master_dtype, grad_dtype=grad_dtype, zero1=zero1)

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

    # Vocabulary parallelism across stages (target): every device runs its LM-head shard as a separate
    # pass per micro-batch. It receives the final hidden state broadcast from the last stage, takes part
    # in the cross-entropy all-reduces over the stages (max, sum of exponentials, target logit; one fp32
    # tile column per 32 rows) and sends its part of the hidden state's gradient back (a reduce): both
    # cost about one hand-off. The schedule plays out the pass's dependencies (simulate_schedule).
    vocab_comm = 0.0
    if opts.vocab_parallel:
        stats_bytes = seq_len // cp * 32 * 4
        line = dataclasses.replace(hw, topology="linear")
        vocab_comm = 2 * hand + 3 * all_reduce_roofline(line, stats_bytes, pp).ideal_network_ns / 1e6

    def device_time(c: StageCost) -> Dict[str, float]:
        flop, dram, fabric = c.mb_flop_ms, c.mb_dram_ms, c.mb_fabric_ms + v * hand + vocab_comm
        return {"flop_ms": flop, "dram_ms": dram, "fabric_ms": fabric,
                "roofline_ms": c.mb_forward_ms + c.mb_backward_ms + 2 * v * hand + vocab_comm,
                "upper_bound_ms": max(flop, dram, fabric)}

    per_stage = [device_time(c) for c in costs]
    D = max(per_stage, key=lambda p: p["roofline_ms"])

    # Play the schedule out with each device's own forward and backward time per micro-batch.
    # With equal devices this is slots(P, m) x D; uneven stages set the pace with the slowest and
    # fill/drain with the sum.
    head = [c.head_forward_ms + c.head_backward_ms for c in costs]
    fwd = [c.mb_forward_ms - c.head_forward_ms + v * hand for c in costs]
    bwd = [c.mb_backward_ms - c.head_backward_ms + v * hand for c in costs]
    vocab = [h + vocab_comm for h in head] if opts.vocab_parallel else None
    bubble, held, pipeline_ms = simulate_schedule(sched, pp, m, fwd, bwd, vocab)

    def split(total: float, f: float, b: float):
        r = f / (f + b) if f + b > 0 else 1 / 3
        return total * r, total * (1 - r)

    ub_f, ub_b, ub_v = [], [], []
    for p, h in zip(per_stage, head):
        # the vocabulary pass's share of the device's roofline time (its communication is in fabric_ms)
        share = (h + vocab_comm) / p["roofline_ms"] if opts.vocab_parallel else 0.0
        c = costs[len(ub_f)]
        f, b = split(p["upper_bound_ms"] * (1 - share), c.mb_forward_bound_ms, c.mb_backward_bound_ms)
        ub_f.append(f)
        ub_b.append(b)
        ub_v.append(p["upper_bound_ms"] * share)
    _, _, pipeline_bound_ms = simulate_schedule(sched, pp, m, ub_f, ub_b, ub_v if opts.vocab_parallel else None)

    # The gradient sync (CP, DP) and the optimizer run after the last backward, one after the other;
    # the step waits for the slowest stage
    step_roofline = pipeline_ms + max(c.sync_ms + c.opt_ms for c in costs)
    link_total = max(m * p["fabric_ms"] + c.sync_fabric_ms for p, c in zip(per_stage, costs))
    step_bound = max(pipeline_bound_ms + max(max(c.opt_flop_ms, c.opt_dram_ms) for c in costs), link_total)

    # Memory per device: the one-micro-batch run's peak (state, its activations, backward
    # transients) plus the other micro-batches the schedule makes it hold at the same time
    device_memory = [c.peak_bytes + (held[d] - 1) * c.activation_bytes for d, c in enumerate(costs)]
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
        "per_microbatch": {**D, "bound": bound, "handoff_ms": hand, "vocab_pass_ms": max(vocab or [0.0])},
        "bubble": bubble,
        "ideal_ms": ideal_ms,
        "peak": {"tflops_per_chip": peak / 1e12, "fidelity": "HiFi4", "cores": hw.tensix_cores_per_chip,
                 "clock_ghz": hw.clock_ghz},
        "roofline": throughput(step_roofline),
        "upper_bound": throughput(step_bound),
        "memory": {"peak_gb": memory / 1e9, "fits": memory <= hw.dram_gb_per_chip * 1e9, "device": worst,
                   "activation_gb_per_mb": costs[worst].activation_bytes / 1e9,
                   "peak_activations": held[worst], "state_gb": costs[worst].state_bytes / 1e9},
    }


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
    along the same column links at once, so each gets 1 / (pp * rows per stage) of a link.

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
        if rows != 1:
            return {"error": f"TP{tp} x CP{cp} on a {rows} x {cols} stage: one group cannot cover two mesh axes"}
        tp_axis = cp_axis = col_axis
    else:
        tp_axis = cp_axis = col_axis
    if interleaved and not (wrap_r and pp * rows == R):
        return {"error": "interleaved needs the last stage to neighbour the first (one pipeline filling a wrapped axis)"}
    pairs = cols if rows >= 1 and cols == C else 1  # chip pairs across the cut between two stages
    return {"topology": {"tp": topo(tp, *tp_axis[1:]) if tp > 1 else "linear",
                         "cp": topo(cp, *cp_axis[1:]) if cp > 1 else "linear",
                         "dp": topo(dp, dp * pp * rows == R, wrap_r) if dp > 1 else "linear"},
            "link_share": {"dp": pp * rows} if dp > 1 else {},
            "cut_links": pairs * hw.num_links, "stage_mesh": block}


def even_split(num_blocks: int, pp: int) -> List[int]:
    """Blocks per pipeline device: as even as possible, the remainder on the first devices."""
    base, extra = divmod(num_blocks, pp)
    return [base + (1 if d < extra else 0) for d in range(pp)]


SWEEP_SCHEDULES = (("1f1b", 1), ("interleaved", 2), ("interleaved", 5))
DEGREES = (1, 2, 4, 8, 16, 32)


def sweep(chips: int, schedules=SWEEP_SCHEDULES, **kwargs) -> List[Dict[str, Any]]:
    """Every split of `chips` into dp * pp * tp * cp, schedule and recompute level that is valid."""
    cache: Dict[Any, StageCost] = {}
    results = []
    for tp, cp, dp in itertools.product(DEGREES, DEGREES, DEGREES):
        if chips % (tp * cp * dp):
            continue
        pp = chips // (tp * cp * dp)
        for (schedule, v), recompute in itertools.product(schedules, RECOMPUTE):
            if pp == 1 and schedule != "1f1b":
                continue
            r = run_parallel_roofline(pp=pp, tp=tp, cp=cp, dp=dp, schedule=schedule, interleave=v,
                                      recompute=recompute, stage_cache=cache, **kwargs)
            if "error" not in r:
                results.append(r)
    return results


def _describe(r: Dict[str, Any]) -> str:
    c = r["config"]
    sched = "" if c["pp"] == 1 else f", {c['schedule']}"
    dp = f"DP{c['dp']} x " if c["dp"] > 1 else ""
    rc = "" if c["recompute"] == "none" else f", {c['recompute']} recompute"
    return f"{dp}PP{c['pp']} x TP{c['tp']} x CP{c['cp']}{sched}{rc}"


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
    parser.add_argument("--vocab-parallel", action="store_true",
                        help="Split the LM head across stages (target only)")
    parser.add_argument("--recompute", default="none", choices=RECOMPUTE,
                        help="selective: norms, activations, attention; full: whole blocks (default: none)")
    parser.add_argument("--master-dtype", default="bf16", choices=list(DTYPES),
                        help="Stored weights, m and v (tt-train precision.master_dtype; default bf16)")
    parser.add_argument("--grad-dtype", default="bf16", choices=list(DTYPES),
                        help="Gradient accumulation across micro-batches (default bf16)")
    parser.add_argument("--zero1", action="store_true", help="Shard optimizer state over CP x DP (target only)")
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

    common = dict(model_name=args.model, seq_len=args.seq, num_microbatches=args.microbatches,
                  hardware=args.hardware, sequence_parallel=args.sp, vocab_parallel=args.vocab_parallel,
                  master_dtype=args.master_dtype, grad_dtype=args.grad_dtype, zero1=args.zero1, mode=args.mode,
                  cut_links=args.cut_links, attn_block=args.attn_block, num_links=args.links, topology=args.topology)
    if args.sweep:
        target_only = [TARGET_ONLY[k] for k in TARGET_ONLY if common[k]]
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
        overrides = {"attn_block": None, **({k: False for k in TARGET_ONLY} if other == "current" else {})}
        dropped = [TARGET_ONLY[k] for k in TARGET_ONLY if k in overrides and out["config"][k]]
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
    if cfg["vocab_parallel"]:
        print(f"  vocabulary pass per micro-batch: {pm['vocab_pass_ms']:.2f} ms (LM-head shard, loss, broadcast, reduce)")
    if len(set(cfg["blocks_per_stage"])) > 1:
        print(f"  blocks per stage: {cfg['blocks_per_stage']}")
    print(f"  bubble {out['bubble']:.2%}")
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
