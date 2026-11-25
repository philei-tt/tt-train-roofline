# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Roofline performance model for embedding operations."""

from __future__ import annotations
from typing import Optional, TYPE_CHECKING

from ..hardware import HardwareSpec, DataType, MathFidelity

if TYPE_CHECKING:
    from .roofline import RooflineEstimate


def embedding_roofline(
    hw: HardwareSpec,
    batch_seq: int,
    embedding_dim: int,
    vocab_size: int,
    dtype: DataType = DataType.BFLOAT16,
    fidelity: MathFidelity = MathFidelity.HiFi4,
    num_cores: Optional[int] = None,
    operation: str = "Embedding",
    phase: str = "forward",
) -> "RooflineEstimate":
    """
    Performance model for embedding lookup operation.

    Forward: Pure memory-bound gather operation.
    Backward: Scatter-add gradients to weight table.

    Args:
        hw: Hardware specification
        batch_seq: Batch size * sequence length (number of tokens)
        embedding_dim: Dimension of embeddings
        vocab_size: Size of vocabulary
        dtype: Data type for operands
        fidelity: Math fidelity level
        num_cores: Number of cores to use (default: all)
        operation: Name for the operation
        phase: "forward" or "backward"

    Returns:
        RooflineEstimate with performance metrics
    """
    # Local import to avoid circular dependency
    from .roofline import RooflineEstimate, sfpu_flops_per_core_per_cycle

    if num_cores is None:
        num_cores = hw.tensix_cores_per_chip

    bytes_per_elem = dtype.value

    if phase == "forward":
        # Forward (tt-train embedding_op): untilize the whole [V, E] table (read + write it), then
        # gather batch_seq rows from the untilized copy
        table_bytes = vocab_size * embedding_dim * bytes_per_elem
        rows = batch_seq * embedding_dim * bytes_per_elem
        total_bytes = int(2 * table_bytes + 2 * rows)  # untilize; gather reads and writes the rows
        total_flops = 0  # Pure gather, no compute
    else:
        # Backward (ttnn::embedding_bw, reader_embedding_backward.cpp): the dense [V, E] gradient is
        # zero-filled; then for each 32-token group, every 32-row vocabulary chunk its tokens hit
        # (up to 32 distinct ones) is read, accumulated and written back in full E width; grad_output
        # is read once. Repeated tokens within a group hit fewer chunks (this counts them distinct).
        table_bytes = vocab_size * embedding_dim * bytes_per_elem
        groups = -(-batch_seq // 32)
        chunks = min(32, -(-vocab_size // 32))
        rmw = groups * chunks * 2 * 32 * embedding_dim * bytes_per_elem
        total_bytes = int(table_bytes + rmw + batch_seq * embedding_dim * bytes_per_elem)
        total_flops = batch_seq * embedding_dim  # Addition for accumulation

    # Compute time (minimal for forward, some for backward accumulation)
    ops_per_cycle = sfpu_flops_per_core_per_cycle(fidelity)
    ideal_compute_ns = (
        total_flops / (num_cores * ops_per_cycle) / hw.clock_ghz
        if total_flops > 0
        else 0
    )

    # Memory time
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
