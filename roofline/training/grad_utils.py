# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Gradient utility functions for roofline modeling.

This module provides mock implementations of gradient utilities
for roofline estimation.
"""

from __future__ import annotations
from typing import Dict, TYPE_CHECKING
import math

from ..mock_tensor import MockTensor
from ..roofline import elementwise_roofline, reduction_roofline, RooflineEstimate

if TYPE_CHECKING:
    from ..roofline import RooflineContext


def mock_clip_grad_norm(
    ctx: "RooflineContext",
    parameters: Dict[str, MockTensor],
    max_norm: float = 1.0,
    num_shards: int = 1,
) -> None:
    """Estimate cost of gradient norm clipping.

    This function estimates the roofline cost of:
    1. Computing the total gradient norm (sum of squared norms per param)
    2. Conditionally scaling all gradients

    Args:
        ctx: Roofline context for estimates
        parameters: Dictionary of parameter names to MockTensors
        max_norm: Maximum gradient norm
        num_shards: ZeRO-1: each replica clips its 1/num_shards of every gradient (the partial norms'
            scalar all-reduce is not modelled)
    """
    # Gradient elements per dtype (gradients may be accumulated in fp32)
    elements: Dict = {}
    for p in parameters.values():
        dtype = p.get_grad().dtype if p.is_grad_initialized() else p.dtype
        elements[dtype] = elements.get(dtype, 0) + p.logical_volume() // num_shards

    for dtype, total_elements in elements.items():
        # Phase 1: sum of squares (elementwise square + reduction), then sqrt
        ctx.add_perf_result(elementwise_roofline(
            ctx.hw, total_elements, num_inputs=1, sfpu_ops_per_element=1.0, fpu_ops_per_element=0.0,
            dtype=dtype, operation="GradNorm.square", phase="grad_clip"))
        ctx.add_perf_result(reduction_roofline(
            ctx.hw, total_elements, total_elements, dtype=dtype, operation="GradNorm.reduce", phase="grad_clip"))
        # Phase 2: scale gradients (conditional, but we model worst case)
        ctx.add_perf_result(elementwise_roofline(
            ctx.hw, total_elements, num_inputs=1, sfpu_ops_per_element=0.0, fpu_ops_per_element=1.0,
            dtype=dtype, operation="GradNorm.scale", phase="grad_clip"))


def mock_zero_grad(
    ctx: "RooflineContext",
    parameters: Dict[str, MockTensor],
) -> None:
    """Estimate cost of zeroing gradients.

    Args:
        ctx: Roofline context for estimates
        parameters: Dictionary of parameter names to MockTensors
    """
    total_elements = sum(p.logical_volume() for p in parameters.values())
    dtype = next(iter(parameters.values())).dtype

    # Just write zeros to all gradients
    estimate = elementwise_roofline(
        ctx.hw,
        total_elements,
        num_inputs=0,  # No reads needed for zero
        sfpu_ops_per_element=0.0,
        fpu_ops_per_element=0.0,  # No compute
        dtype=dtype,
        operation="ZeroGrad",
        phase="grad_zero",
    )

    # Adjust to only count writes
    estimate = RooflineEstimate(
        operation=estimate.operation,
        phase=estimate.phase,
        total_flops=0,
        total_bytes=int(total_elements * dtype.value),
        ideal_compute_ns=0,
        ideal_memory_ns=total_elements * dtype.value / ctx.hw.dram_bw_gb_s,
        hw=ctx.hw,
    )
    ctx.add_perf_result(estimate)
