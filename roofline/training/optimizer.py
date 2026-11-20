# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Optimizer roofline estimation for training.

This module provides MockAdamW for roofline estimation of
optimizer weight update operations.
"""

from __future__ import annotations
from typing import Dict, List, Optional, TYPE_CHECKING

from ..hardware import DataType
from ..mock_tensor import MockTensor, TensorLabel
from ..roofline import RooflineEstimate, elementwise_roofline

if TYPE_CHECKING:
    from ..roofline import RooflineContext
    from ..modules import MockModule


class MockAdamW:
    """Mock AdamW optimizer for roofline estimation.

    Estimates the cost of optimizer weight updates:
    - Read: weight, grad, m (momentum), v (variance) - 4 tensors per param
    - Write: weight, m, v - 3 tensors per param
    - Compute: ~10 FLOPs per element (sqrt, div, mul, add, etc.)

    Example:
        >>> model = MockNanoGPT(config)
        >>> optimizer = MockAdamW(model.parameters())
        >>> ctx = RooflineContext(WORMHOLE_N150)
        >>> # ... run forward/backward ...
        >>> optimizer.step(ctx)
    """

    def __init__(
        self,
        parameters: Dict[str, MockTensor],
        lr: float = 1e-4,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        master_dtype: Optional[DataType] = None,
        num_shards: int = 1,
    ):
        """Initialize AdamW optimizer.

        Args:
            parameters: Dictionary of parameter names to MockTensors
            lr: Learning rate
            betas: Coefficients for computing running averages
            eps: Term for numerical stability
            weight_decay: Weight decay coefficient
            master_dtype: dtype the parameters are stored and updated in (tt-train's
                precision.master_dtype). None: the parameter's own dtype. FLOAT32 with bf16
                parameters: an fp32 master copy, m and v in fp32, and the bf16 copy the model
                computes with rewritten every step (as tt-train's PreferredPrecision cache)
            num_shards: ZeRO-1: master copy, m and v (and the update) split over this many replicas
        """
        self.parameters = parameters
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.num_shards = num_shards

        # Calculate total parameter count
        self.total_params = sum(p.logical_volume() for p in parameters.values())

        # Optimizer state tensors: m and v (and the master copy) for each parameter, this replica's shard
        self._state_tensors: List[MockTensor] = []
        self._state_dtype: Dict[str, DataType] = {}
        for name, param in parameters.items():
            dtype = master_dtype or param.dtype
            master = dtype != param.dtype
            self._state_dtype[name] = dtype
            shape = (param.logical_volume() // num_shards,)
            for suffix in ("m", "v") + (("master",) if master else ()):
                self._state_tensors.append(MockTensor(
                    shape, dtype, param.layout, requires_grad=False,
                    label=TensorLabel.OPTIMIZER_STATE, name=f"{name}.{suffix}",
                ))

    def step(self, ctx: "RooflineContext") -> None:
        """Estimate optimizer step cost.

        Per element of this replica's shard: read the gradient; read and write m, v and the stored
        weight; with an fp32 master copy also write the bf16 copy. About 10 SFPU FLOPs.

        Args:
            ctx: Roofline context for estimates
        """
        for name, param in self.parameters.items():
            num_elements = param.logical_volume() // self.num_shards
            state = self._state_dtype[name].value  # m, v and the stored weight
            master = self._state_dtype[name] != param.dtype
            grad = param.get_grad()
            grad_bytes = (grad.dtype if grad is not None else param.dtype).value
            per_element = grad_bytes + 6 * state + (param.dtype.value if master else 0)

            estimate = elementwise_roofline(
                ctx.hw,
                num_elements,
                num_inputs=4,  # weight, grad, m, v
                sfpu_ops_per_element=10.0,  # sqrt, div, mul, add operations
                fpu_ops_per_element=0.0,
                dtype=param.dtype,
                operation=f"AdamW.{name}",
                phase="optimizer",
            )
            total_bytes = int(per_element * num_elements)
            ctx.add_perf_result(RooflineEstimate(
                operation=estimate.operation,
                phase=estimate.phase,
                total_flops=estimate.total_flops,
                total_bytes=total_bytes,
                ideal_compute_ns=estimate.ideal_compute_ns,
                ideal_memory_ns=total_bytes / ctx.hw.dram_bw_gb_s,
                hw=ctx.hw,
            ))

    def estimate_memory(self) -> int:
        """Optimizer state memory of this replica in bytes (m, v and any master copy)."""
        return sum(t.bytes() for t in self._state_tensors)

    def __repr__(self) -> str:
        return (
            f"MockAdamW(params={len(self.parameters)}, "
            f"total_elements={self.total_params:,}, "
            f"lr={self.lr}, weight_decay={self.weight_decay})"
        )
