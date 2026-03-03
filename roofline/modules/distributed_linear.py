# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Column-parallel and row-parallel linear modules for tensor parallelism.

Mirrors ttml/modules/distributed/linear.{hpp,cpp}. These modules shard
weight matrices across TP devices and insert the appropriate CCL
communication operations (broadcast, all-reduce, scatter, all-gather).

Typical TP usage pattern (as in ttml DistributedGPTMLP):
    fc1 = MockColumnParallelLinear(emb, emb*4, gather_output=False, tp_size=N)
    fc2 = MockRowParallelLinear(emb*4, emb, input_is_parallel=True, tp_size=N)
    # fc1 shards output along last dim -> fc2 takes sharded input -> all-reduces
"""

from __future__ import annotations
from typing import TYPE_CHECKING

from ..mock_tensor import MockTensor
from ..hardware import DataType
from ..operations import MockLinearOp
from ..operations.comm_ops import (
    MockAllReduceOp,
    MockAllGatherOp,
    MockBroadcastOp,
    MockScatterOp,
)
from ..operations.elementwise import MockAddOp
from .module import MockModule, MockParameter

if TYPE_CHECKING:
    from ..roofline import RooflineContext


class MockColumnParallelLinear(MockModule):
    """Column-parallel linear layer: weight sharded along output dimension.

    Each TP device holds weight [1, 1, out_features/tp_size, in_features].
    Forward: broadcast input -> per-device linear -> optionally all-gather.

    Mirrors ttml::modules::distributed::ColumnParallelLinear.

    Args:
        in_features: Full input feature size
        out_features: Full output feature size (must be divisible by tp_size)
        has_bias: Whether to include bias
        gather_output: If True, all-gather output to full size
        tp_size: Tensor parallelism degree (number of TP devices)
        dtype: Data type for parameters
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        has_bias: bool = True,
        gather_output: bool = False,
        tp_size: int = 1,
        dtype: DataType = DataType.BFLOAT16,
    ):
        super().__init__()

        assert out_features % tp_size == 0, (
            f"ColumnParallelLinear: out_features ({out_features}) must be divisible by tp_size ({tp_size})"
        )

        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = has_bias
        self.gather_output = gather_output
        self.tp_size = tp_size

        local_out_features = out_features // tp_size

        # Weight sharded along output dim (rank-2 in [1,1,out,in])
        self.weight = MockParameter(
            (1, 1, local_out_features, in_features), dtype=dtype, name="weight",
        )

        if has_bias:
            # Bias also sharded along output dim
            self.bias = MockParameter(
                (1, 1, 1, local_out_features), dtype=dtype, name="bias",
            )
        else:
            self.bias = None

    def forward(self, ctx: "RooflineContext", x: MockTensor) -> MockTensor:
        # Broadcast: noop in forward, all-reduce in backward
        x = MockBroadcastOp.apply(ctx, x, self.tp_size)

        bias_tensor = self.bias.tensor if self.bias is not None else None
        x = MockLinearOp.apply(ctx, x, self.weight.tensor, bias_tensor)

        if self.gather_output and self.tp_size > 1:
            x = MockAllGatherOp.apply(ctx, x, x.shape.__len__() - 1, self.tp_size)

        return x

    def __repr__(self) -> str:
        return (
            f"MockColumnParallelLinear(in={self.in_features}, out={self.out_features}, "
            f"bias={self.has_bias}, gather_output={self.gather_output}, tp_size={self.tp_size})"
        )


class MockRowParallelLinear(MockModule):
    """Row-parallel linear layer: weight sharded along input dimension.

    Each TP device holds weight [1, 1, out_features, in_features/tp_size].
    Forward: optionally scatter input -> per-device linear -> all-reduce.

    Mirrors ttml::modules::distributed::RowParallelLinear.

    Args:
        in_features: Full input feature size (must be divisible by tp_size)
        out_features: Full output feature size
        has_bias: Whether to include bias
        input_is_parallel: If True, input is already sharded (skip scatter)
        tp_size: Tensor parallelism degree (number of TP devices)
        dtype: Data type for parameters
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        has_bias: bool = True,
        input_is_parallel: bool = False,
        tp_size: int = 1,
        dtype: DataType = DataType.BFLOAT16,
    ):
        super().__init__()

        assert in_features % tp_size == 0, (
            f"RowParallelLinear: in_features ({in_features}) must be divisible by tp_size ({tp_size})"
        )

        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = has_bias
        self.input_is_parallel = input_is_parallel
        self.tp_size = tp_size

        local_in_features = in_features // tp_size

        # Weight sharded along input dim (rank-1 in [1,1,out,in])
        self.weight = MockParameter(
            (1, 1, out_features, local_in_features), dtype=dtype, name="weight",
        )

        if has_bias:
            # Bias is NOT sharded for row-parallel (applied after all-reduce)
            self.bias = MockParameter(
                (1, 1, 1, out_features), dtype=dtype, name="bias",
            )
        else:
            self.bias = None

    def forward(self, ctx: "RooflineContext", x: MockTensor) -> MockTensor:
        if not self.input_is_parallel and self.tp_size > 1:
            x = MockScatterOp.apply(ctx, x, len(x.shape) - 1, self.tp_size)

        # Linear without bias (bias added after all-reduce)
        x = MockLinearOp.apply(ctx, x, self.weight.tensor, None)

        # All-reduce to sum partial results across TP devices.
        # noop_backward when input_is_parallel to avoid double all-reduce
        # (broadcast in the paired ColumnParallelLinear already handles it).
        x = MockAllReduceOp.apply(
            ctx, x, self.input_is_parallel, self.tp_size,
        )

        if self.bias is not None:
            x = MockAddOp.apply(ctx, x, self.bias.tensor)

        return x

    def __repr__(self) -> str:
        return (
            f"MockRowParallelLinear(in={self.in_features}, out={self.out_features}, "
            f"bias={self.has_bias}, input_is_parallel={self.input_is_parallel}, tp_size={self.tp_size})"
        )
