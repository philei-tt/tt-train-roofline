# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Distributed (tensor-parallel) Llama MLP for roofline modeling.

Mirrors ttml::modules::distributed::DistributedLlamaMLP.
SwiGLU MLP with column-parallel w1/w3 and row-parallel w2.
"""

from __future__ import annotations
from typing import Optional, TYPE_CHECKING

from ..mock_tensor import MockTensor
from ..hardware import DataType
from ..operations import MockMulOp, MockDropoutOp
from ..operations.silu import MockSiLUOp
from .module import MockModule
from .distributed_linear import MockColumnParallelLinear, MockRowParallelLinear

if TYPE_CHECKING:
    from ..roofline import RooflineContext


class MockDistributedLlamaMLP(MockModule):
    """Distributed Llama MLP (SwiGLU) with TP.

    Mirrors ttml::modules::distributed::DistributedLlamaMLP.
    - w1, w3: ColumnParallelLinear (gather_output=False)
    - w2: RowParallelLinear (input_is_parallel=True)
    """

    def __init__(
        self,
        embedding_size: int,
        dropout: float = 0.0,
        intermediate_dim: Optional[int] = None,
        tp_size: int = 1,
        dtype: DataType = DataType.BFLOAT16,
    ):
        super().__init__()

        multiple_of = 256
        if intermediate_dim is not None:
            hidden_size = intermediate_dim
        else:
            unrounded = int(4 * embedding_size * (2.0 / 3.0))
            hidden_size = ((unrounded + multiple_of - 1) // multiple_of) * multiple_of

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.dropout_prob = dropout
        self.tp_size = tp_size

        self.w1 = MockColumnParallelLinear(
            embedding_size,
            hidden_size,
            has_bias=False,
            gather_output=False,
            tp_size=tp_size,
            dtype=dtype,
        )
        self.w3 = MockColumnParallelLinear(
            embedding_size,
            hidden_size,
            has_bias=False,
            gather_output=False,
            tp_size=tp_size,
            dtype=dtype,
        )
        self.w2 = MockRowParallelLinear(
            hidden_size,
            embedding_size,
            has_bias=False,
            input_is_parallel=True,
            tp_size=tp_size,
            dtype=dtype,
        )

    def forward(self, ctx: "RooflineContext", x: MockTensor) -> MockTensor:
        swished = MockSiLUOp.apply(ctx, self.w1(ctx, x))
        gate = self.w3(ctx, x)
        gated = MockMulOp.apply(ctx, swished, gate)
        out = self.w2(ctx, gated)
        if self.dropout_prob > 0:
            out = MockDropoutOp.apply(ctx, out, self.dropout_prob)
        return out

    def __repr__(self) -> str:
        return (
            f"MockDistributedLlamaMLP(embedding_size={self.embedding_size}, "
            f"hidden_size={self.hidden_size}, tp_size={self.tp_size})"
        )
