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
    M = S * (S + tile) / 2 (diagonal tiles are computed in full). With a block of Bt tokens kept on
    chip, the opposite operands are re-read once per block: about S / Bt times, causal-halved.

    Forward (one kernel, Q-block outer, per Q head):
        - FLOPs: QK^T and PV, 4 * B * Hq * M * d, plus softmax on the SFPU
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
    flops_per_exp = 5
    M = S * (S + tile_height) / 2  # causal volume visited per head, tile granularity
    # Opposite operand re-read once per block: sum over blocks i of (i + 1) * Bt rows
    rows_reread = (S / Bt) * (S + Bt) / 2
    stat_bytes_per_row = 32 * 4  # one fp32 tile (32 wide) per 32 rows
    row = d * bpe  # bytes of one row of Q, K, V, O, dO, dQ

    if phase == "forward":
        matmul_flops = 4 * B * Hq * M * d
        sfpu_flops = (flops_per_exp + 5) * B * Hq * M + (flops_per_exp + d + 2) * B * Hq * S * (S / Bt)

        reads = B * Hq * S * row + 2 * B * Hq * rows_reread * row  # Q once; K and V per Q block
        writes = B * Hq * S * row + B * Hq * S * stat_bytes_per_row  # O, log-sum-exp
        total_bytes = int(reads + writes)

    else:  # backward
        matmul_flops = 14 * B * Hq * M * d
        sfpu_flops = 2 * B * Hq * (M * (flops_per_exp + 4) + S * (2 * d - 1))  # softmax recomputed in both kernels

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
