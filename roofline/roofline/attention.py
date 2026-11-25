# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Roofline performance model for attention operations."""

from __future__ import annotations
from typing import Optional, List, TYPE_CHECKING
import math

from ..hardware import HardwareSpec, DataType, MathFidelity
from .matmul import matmul_roofline
from .softmax import softmax_roofline

if TYPE_CHECKING:
    from .roofline import RooflineEstimate


def attention_roofline(
    hw: HardwareSpec,
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    dtype: DataType = DataType.BFLOAT16,
    fidelity: MathFidelity = MathFidelity.HiFi4,
    num_cores: Optional[int] = None,
    operation: str = "ScaledDotProductAttention",
    phase: str = "forward",
) -> List["RooflineEstimate"]:
    """
    Performance model for scaled dot-product attention.

    SDPA computes: softmax(Q @ K^T / sqrt(d_k)) @ V

    Forward:
        1. Q @ K^T: matmul [B*H, S, d] x [B*H, d, S] -> [B*H, S, S]
        2. Scale by 1/sqrt(d_k): elementwise
        3. Softmax over last dim
        4. Attn @ V: matmul [B*H, S, S] x [B*H, S, d] -> [B*H, S, d]

    Backward:
        1. grad_V = Attn^T @ grad_output
        2. grad_Attn = grad_output @ V^T
        3. grad_softmax -> grad_scores (softmax backward)
        4. grad_Q = grad_scores @ K
        5. grad_K = grad_scores^T @ Q

    Args:
        hw: Hardware specification
        batch_size: Batch size
        num_heads: Number of attention heads
        seq_len: Sequence length
        head_dim: Dimension per head
        dtype: Data type for operands
        fidelity: Math fidelity level
        num_cores: Number of cores to use (default: all)
        operation: Base name for the operation
        phase: "forward" or "backward"

    Returns:
        List of RooflineEstimate for each sub-operation
    """
    from .roofline import RooflineEstimate
    from .elementwise import elementwise_roofline

    estimates = []
    B = batch_size
    H = num_heads
    S = seq_len
    d = head_dim

    if phase == "forward":
        # Q @ K^T: [B*H, S, d] x [B*H, d, S] -> [B*H, S, S]
        estimates.append(
            matmul_roofline(
                hw,
                B * H * S,
                d,
                S,
                dtype=dtype,
                fidelity=fidelity,
                num_cores=num_cores,
                operation=f"{operation}.QK",
                phase="forward",
            )
        )

        # Scale by 1/sqrt(d_k) - elementwise
        estimates.append(
            elementwise_roofline(
                hw,
                B * H * S * S,
                num_inputs=1,
                sfpu_ops_per_element=1.0,
                fpu_ops_per_element=0.0,
                dtype=dtype,
                fidelity=fidelity,
                num_cores=num_cores,
                operation=f"{operation}.scale",
                phase="forward",
            )
        )

        # Softmax
        estimates.append(
            softmax_roofline(
                hw,
                B * H * S,
                S,
                dtype=dtype,
                fidelity=fidelity,
                num_cores=num_cores,
                operation=f"{operation}.softmax",
                phase="forward",
            )
        )

        # Attn @ V: [B*H, S, S] x [B*H, S, d] -> [B*H, S, d]
        estimates.append(
            matmul_roofline(
                hw,
                B * H * S,
                S,
                d,
                dtype=dtype,
                fidelity=fidelity,
                num_cores=num_cores,
                operation=f"{operation}.AV",
                phase="forward",
            )
        )
    else:
        # grad_V = Attn^T @ grad_output: [B*H, S, S]^T x [B*H, S, d] -> [B*H, S, d]
        estimates.append(
            matmul_roofline(
                hw,
                B * H * S,
                S,
                d,
                dtype=dtype,
                fidelity=fidelity,
                num_cores=num_cores,
                operation=f"{operation}.backward.grad_V",
                phase="backward",
            )
        )

        # grad_Attn = grad_output @ V^T: [B*H, S, d] x [B*H, d, S] -> [B*H, S, S]
        estimates.append(
            matmul_roofline(
                hw,
                B * H * S,
                d,
                S,
                dtype=dtype,
                fidelity=fidelity,
                num_cores=num_cores,
                operation=f"{operation}.backward.grad_Attn",
                phase="backward",
            )
        )

        # Softmax backward
        estimates.append(
            softmax_roofline(
                hw,
                B * H * S,
                S,
                dtype=dtype,
                fidelity=fidelity,
                num_cores=num_cores,
                operation=f"{operation}.backward.softmax",
                phase="backward",
            )
        )

        # Scale backward - elementwise
        # note: mul uses fpu
        estimates.append(
            elementwise_roofline(
                hw,
                B * H * S * S,
                num_inputs=1,
                sfpu_ops_per_element=0.0,
                fpu_ops_per_element=1.0,
                dtype=dtype,
                fidelity=fidelity,
                num_cores=num_cores,
                operation=f"{operation}.backward.scale",
                phase="backward",
            )
        )

        # grad_Q = grad_scores @ K: [B*H, S, S] x [B*H, S, d] -> [B*H, S, d]
        estimates.append(
            matmul_roofline(
                hw,
                B * H * S,
                S,
                d,
                dtype=dtype,
                fidelity=fidelity,
                num_cores=num_cores,
                operation=f"{operation}.backward.grad_Q",
                phase="backward",
            )
        )

        # grad_K = grad_scores^T @ Q: [B*H, S, S]^T x [B*H, S, d] -> [B*H, S, d]
        estimates.append(
            matmul_roofline(
                hw,
                B * H * S,
                S,
                d,
                dtype=dtype,
                fidelity=fidelity,
                num_cores=num_cores,
                operation=f"{operation}.backward.grad_K",
                phase="backward",
            )
        )

    return estimates


def heads_creation_roofline(
    hw: HardwareSpec,
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    num_tensors: int = 3,  # Q, K, V
    dtype: DataType = DataType.BFLOAT16,
    fidelity: MathFidelity = MathFidelity.HiFi4,
    num_cores: Optional[int] = None,
    operation: str = "HeadsCreation",
    phase: str = "forward",
) -> "RooflineEstimate":
    """
    Performance model for heads creation (split QKV into heads).

    This is a reshape/transpose operation - pure memory movement.
    Input: [B, 1, S, 3*H*d] -> Output: 3x [B, H, S, d]

    Args:
        hw: Hardware specification
        batch_size: Batch size
        seq_len: Sequence length
        num_heads: Number of attention heads
        head_dim: Dimension per head
        num_tensors: Number of output tensors (3 for QKV)
        dtype: Data type
        fidelity: Math fidelity level
        num_cores: Number of cores to use
        operation: Name for the operation
        phase: "forward" or "backward"

    Returns:
        RooflineEstimate with performance metrics
    """
    from .roofline import RooflineEstimate

    if num_cores is None:
        num_cores = hw.tensix_cores_per_chip

    bytes_per_elem = dtype.value

    # Total elements: B * S * num_tensors * H * d
    num_elements = batch_size * seq_len * num_tensors * num_heads * head_dim

    # Pure memory movement: read input, write output
    total_bytes = int(2 * num_elements * bytes_per_elem)
    total_flops = 0  # No compute, just reshape/transpose

    ideal_compute_ns = 0
    ideal_memory_ns = total_bytes / hw.dram_bw_gb_s

    return RooflineEstimate(
        operation=operation,
        phase=phase,
        total_flops=total_flops,
        total_bytes=total_bytes,
        ideal_compute_ns=ideal_compute_ns,
        ideal_memory_ns=ideal_memory_ns,
        hw=hw,
    )


def grouped_heads_creation_roofline(
    hw: HardwareSpec,
    batch_size: int,
    seq_len: int,
    num_heads: int,
    num_groups: int,
    head_dim: int,
    dtype: DataType = DataType.BFLOAT16,
    fidelity: MathFidelity = MathFidelity.HiFi4,
    num_cores: Optional[int] = None,
    operation: str = "GroupedHeadsCreation",
    phase: str = "forward",
) -> "RooflineEstimate":
    """
    Performance model for grouped heads creation (split Q and KV into heads for GQA).

    This is a reshape/transpose operation - pure memory movement.
    For Grouped Query Attention:
    - Q: [B, 1, S, E] -> [B, num_heads, S, head_dim]
    - KV: [B, 1, S, 2*num_groups*head_dim] -> K [B, num_groups, S, head_dim], V [B, num_groups, S, head_dim]

    Args:
        hw: Hardware specification
        batch_size: Batch size
        seq_len: Sequence length
        num_heads: Number of attention heads for Q
        num_groups: Number of KV groups (num_heads for MHA, fewer for GQA)
        head_dim: Dimension per head
        dtype: Data type
        fidelity: Math fidelity level
        num_cores: Number of cores to use
        operation: Name for the operation
        phase: "forward" or "backward"

    Returns:
        RooflineEstimate with performance metrics
    """
    from .roofline import RooflineEstimate

    if num_cores is None:
        num_cores = hw.tensix_cores_per_chip

    bytes_per_elem = dtype.value

    # Total elements: Q has num_heads, K and V have num_groups each
    q_elements = batch_size * seq_len * num_heads * head_dim
    kv_elements = batch_size * seq_len * 2 * num_groups * head_dim
    num_elements = q_elements + kv_elements

    # Pure memory movement: read input, write output; the backward also concatenates the K and V
    # gradients (ttnn::concat, multi_head_utils.cpp): one more pass over them
    total_bytes = int(2 * num_elements * bytes_per_elem)
    if phase == "backward":
        total_bytes += int(2 * kv_elements * bytes_per_elem)
    total_flops = 0  # No compute, just reshape/transpose

    ideal_compute_ns = 0
    ideal_memory_ns = total_bytes / hw.dram_bw_gb_s

    return RooflineEstimate(
        operation=operation,
        phase=phase,
        total_flops=total_flops,
        total_bytes=total_bytes,
        ideal_compute_ns=ideal_compute_ns,
        ideal_memory_ns=ideal_memory_ns,
        hw=hw,
    )


def fused_attention_roofline(
    hw: HardwareSpec,
    batch_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    seq_len: int,
    head_dim: int,
    dtype: DataType = DataType.BFLOAT16,
    fidelity: MathFidelity = MathFidelity.HiFi4,
    num_cores: Optional[int] = None,
    operation: str = "FusedSDPA",
    phase: str = "forward",
    tile_height: int = 32,
    tile_width: int = 32,
    block: Optional[int] = None,
    causal: bool = True,
    single_kernel_bw: bool = False,
) -> "RooflineEstimate":
    """
    Performance model for fused scaled dot-product attention (Flash Attention style), following
    tt-train's kernels (tt-train/sources/ttml/metal/ops/sdpa_fw, sdpa_bw).

    Fused SDPA computes: softmax(Q @ K^T / sqrt(d_k)) @ V in a single kernel
    using online softmax and tiling to avoid materializing the full attention matrix.

    Key differences from unfused SDPA:
    - Memory: O(S) instead of O(S*S) - no need to store full attention matrix
    - Compute: Same total FLOPs, but better memory access patterns
    - Backward: Uses recomputation instead of reading saved attention weights

    Causal attention; tiles above the diagonal are skipped, so the visited volume per head is
    M = S * (S + tile) / 2 (diagonal tiles are computed in full; the forward rounds the diagonal up
    to its 4-tile K/V chunk, S * (S + 4 tile) / 2). With a block of Bt tokens kept on chip, the
    opposite operands are re-read once per block: about S / Bt times, causal-halved.

    SFPU work is counted in instructions per element (32 lanes per instruction, so 32 per core per
    cycle): exp is 15 on Blackhole (non-approximate, as tt-train runs it), the backward softmax
    recompute and gradient 26 per score and kernel; it runs after the matmuls (summed cycles).

    Forward (one kernel, Q-block outer, per Q head):
        - FLOPs: QK^T and PV, 4 * B * Hq * M * d, plus exp per score and a rescale per row and chunk
        - DRAM: Q read once; K and V re-read per Q block per Q head; O written; the log-sum-exp
          statistic written as one fp32 32-wide tile per 32 rows
    Backward (two kernels, run one after the other):
        - Q kernel (Q-block outer, per Q head): Q, O, dO and the statistic read once, u = rowsum(dO * O)
          computed in it and written as an fp32 tile; K and V re-read per Q block; dQ accumulated on
          chip and written once. FLOPs: QK^T, dO V^T, dS K = 6 * B * Hq * M * d
        - KV kernel (K/V-block outer, per KV head, over the g Q heads of the group): K and V read once;
          Q, dO and the two fp32 statistic tiles re-read per K/V block per Q head; dK and dV accumulated
          on chip and written once. FLOPs: QK^T, dO V^T, P^T dO, dS^T Q = 8 * B * Hq * M * d
        - Total matmul FLOPs 14 * B * Hq * M * d (FlashAttention-2 recomputes less: 10)

    Args:
        hw: Hardware specification
        batch_size: Batch size
        num_q_heads, num_kv_heads: query and key/value heads
        seq_len: Sequence length
        head_dim: Dimension per head
        dtype: Data type for operands
        fidelity: Math fidelity level
        num_cores: Number of cores to use (default: all)
        operation: Base name for the operation
        phase: "forward" or "backward"
        block: tokens kept on chip per block (forward and Q-kernel Q block, KV-kernel K/V block).
            None (default) = one tile (tile_height tokens), today's kernels.
        causal: False for a full S x S block (ring attention: a K/V shard from earlier tokens)
        single_kernel_bw: backward as one FlashAttention-2-style kernel (the target): K/V-block outer,
            10 * B * Hq * M * d matmul FLOPs, the softmax recomputed once; Q, dO and the statistics
            re-read per K/V block, dQ accumulated in fp32 in DRAM (read + write per K/V block), after a
            pass computing u = rowsum(dO * O). False: tt-train's two kernels today (below)

    Returns:
        Single RooflineEstimate for the fused operation
    """
    from .roofline import (
        RooflineEstimate,
        fpu_mm_flops_per_core_per_cycle,
        sfpu_flops_per_core_per_cycle,
    )

    if num_cores is None:
        num_cores = hw.tensix_cores_per_chip

    B = batch_size
    Hq = num_q_heads
    Hkv = num_kv_heads
    S = seq_len
    d = head_dim
    bpe = dtype.value
    Bt = block if block else tile_height
    # SFPU work is counted in instructions per element: one instruction handles 32 lanes, so
    # sfpu_flops_per_core_per_cycle (32) turns it into cycles. Blackhole, tt-train's kernels with
    # math_approx_mode=false: exp is 15 instructions per 32 elements (_sfpu_exp_21f_bf16_tti_,
    # ckernel_sfpu_exp.h BODY_LEN); the backward per score and kernel is sub (4) + exp (15) +
    # mul (4) + mul (3) = 26 (sdpa_bw_compute_utils.hpp)
    exp_instr, bw_instr_per_kernel = 15, 26
    # sdpa_fw walks K/V in chunks of Sk_chunk_t = 4 tiles; a larger Q block visits whole blocks
    chunk = max(4 * tile_height, Bt)
    if causal:
        M = S * (S + tile_height) / 2  # causal volume visited per head, tile granularity (backward)
        # forward: the diagonal is rounded up to a whole K/V chunk
        M_fw = S * (S + chunk) / 2
        # Opposite operand re-read once per block: sum over blocks i of (i + 1) * Bt rows
        rows_reread = (S / Bt) * (S + Bt) / 2
        rows_reread_fw = (S / Bt) * (S + chunk) / 2
        chunks_per_row = (S / chunk + 1) / 2
    else:
        M = M_fw = S * S
        rows_reread = rows_reread_fw = (S / Bt) * S
        chunks_per_row = S / chunk
    stat_bytes_per_row = 32 * 4  # one fp32 tile (32 wide) per 32 rows
    row = d * bpe  # bytes of one row of Q, K, V, O, dO, dQ

    if phase == "forward":
        matmul_flops = 4 * B * Hq * M_fw * d
        # exp per score (max and subtract run on the FPU); per row and K/V chunk the running max,
        # sum and output rescale
        sfpu_flops = exp_instr * B * Hq * M_fw + (exp_instr + d + 2) * B * Hq * S * chunks_per_row

        reads = B * Hq * S * row + 2 * B * Hq * rows_reread_fw * row  # Q once; K and V per Q block
        writes = B * Hq * S * row + B * Hq * S * stat_bytes_per_row  # O, log-sum-exp
        total_bytes = int(reads + writes)

    elif single_kernel_bw:
        matmul_flops = 10 * B * Hq * M * d
        sfpu_flops = B * Hq * (M * bw_instr_per_kernel + S * (2 * d - 1))
        u_pass = B * Hq * S * (2 * row + stat_bytes_per_row)  # read dO and O, write u
        kv_reads = 2 * B * Hkv * S * row + B * Hq * rows_reread * (2 * row + 2 * stat_bytes_per_row)
        dq_rmw = B * Hq * rows_reread * 2 * d * 4  # fp32 dQ read and written per K/V block
        kv_writes = 2 * B * Hkv * S * row  # dK, dV
        total_bytes = int(u_pass + kv_reads + dq_rmw + kv_writes)

    else:  # backward
        matmul_flops = 14 * B * Hq * M * d
        sfpu_flops = 2 * B * Hq * (M * bw_instr_per_kernel + S * (2 * d - 1))  # softmax recomputed in both kernels

        # Q kernel
        q_reads = 3 * B * Hq * S * row + B * Hq * S * stat_bytes_per_row + 2 * B * Hq * rows_reread * row
        q_writes = B * Hq * S * row + B * Hq * S * stat_bytes_per_row  # dQ, u
        # KV kernel
        kv_reads = 2 * B * Hkv * S * row + B * Hq * rows_reread * (2 * row + 2 * stat_bytes_per_row)
        kv_writes = 2 * B * Hkv * S * row  # dK, dV
        total_bytes = int(q_reads + q_writes + kv_reads + kv_writes)

    total_flops = int(matmul_flops + sfpu_flops)

    # Compute time: split between FPU (matmul) and SFPU (scale, softmax)
    fpu_mm_ops_per_cycle = fpu_mm_flops_per_core_per_cycle(fidelity)
    sfpu_ops_per_cycle = sfpu_flops_per_core_per_cycle(fidelity)

    matmul_cycles = math.ceil(matmul_flops / (num_cores * fpu_mm_ops_per_cycle))
    sfpu_cycles = math.ceil(sfpu_flops / (num_cores * sfpu_ops_per_cycle))

    # In fused kernel, operations are pipelined but still sequential in data dependency
    # QK^T -> scale -> softmax -> AV, so we sum the times
    ideal_compute_cycles = matmul_cycles + sfpu_cycles
    ideal_compute_ns = ideal_compute_cycles / hw.clock_ghz  # cycles / (cycles/ns)

    # Memory time
    ideal_memory_ns = total_bytes / hw.dram_bw_gb_s  # bytes / (bytes/ns)

    return RooflineEstimate(
        operation=operation,
        phase=phase,
        total_flops=total_flops,
        total_bytes=total_bytes,
        ideal_compute_ns=ideal_compute_ns,
        ideal_memory_ns=ideal_memory_ns,
        hw=hw,
    )


def heads_fusion_roofline(
    hw: HardwareSpec,
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    dtype: DataType = DataType.BFLOAT16,
    fidelity: MathFidelity = MathFidelity.HiFi4,
    num_cores: Optional[int] = None,
    operation: str = "HeadsFusion",
    phase: str = "forward",
) -> "RooflineEstimate":
    """
    Performance model for heads fusion (merge heads back).

    This is a reshape/transpose operation - pure memory movement.
    Input: [B, H, S, d] -> Output: [B, 1, S, H*d]

    Args:
        hw: Hardware specification
        batch_size: Batch size
        seq_len: Sequence length
        num_heads: Number of attention heads
        head_dim: Dimension per head
        dtype: Data type
        fidelity: Math fidelity level
        num_cores: Number of cores to use
        operation: Name for the operation
        phase: "forward" or "backward"

    Returns:
        RooflineEstimate with performance metrics
    """
    from .roofline import RooflineEstimate

    if num_cores is None:
        num_cores = hw.tensix_cores_per_chip

    bytes_per_elem = dtype.value

    # Total elements: B * H * S * d
    num_elements = batch_size * num_heads * seq_len * head_dim

    # Pure memory movement: read input, write output
    # backward (tt-train heads_fusion, multi_head_utils.cpp): two transposes, each a full pass
    passes = 2 if phase == "backward" else 1
    total_bytes = int(2 * passes * num_elements * bytes_per_elem)
    total_flops = 0  # No compute, just reshape/transpose

    ideal_compute_ns = 0
    ideal_memory_ns = total_bytes / hw.dram_bw_gb_s

    return RooflineEstimate(
        operation=operation,
        phase=phase,
        total_flops=total_flops,
        total_bytes=total_bytes,
        ideal_compute_ns=ideal_compute_ns,
        ideal_memory_ns=ideal_memory_ns,
        hw=hw,
    )
