# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Distributed (tensor-parallel) Llama transformer block for roofline modeling.

Mirrors ttml::modules::distributed::DistributedLlamaBlock.
Same as MockLlamaBlock but uses MockDistributedLlamaMLP and
MockDistributedGroupedQueryAttention.
"""

from __future__ import annotations
from typing import Optional, TYPE_CHECKING

from ..mock_tensor import MockTensor
from ..hardware import DataType
from ..operations import MockAddOp
from .module import MockModule
from .rmsnorm import MockRMSNormLayer
from .grouped_query_attention import RoPEParams
from .distributed_grouped_query_attention import MockDistributedGroupedQueryAttention
from .distributed_llama_mlp import MockDistributedLlamaMLP

if TYPE_CHECKING:
    from ..roofline import RooflineContext


class MockDistributedLlamaBlock(MockModule):
    """Distributed Llama transformer block with TP.

    Mirrors ttml::modules::distributed::DistributedLlamaBlock.
    Pre-norm + attention (distributed GQA) + residual; pre-norm + MLP (distributed) + residual.
    """

    def __init__(
        self,
        embedding_size: int,
        num_heads: int,
        num_groups: int,
        rope_params: Optional[RoPEParams] = None,
        dropout: float = 0.0,
        intermediate_dim: Optional[int] = None,
        tp_size: int = 1,
        dtype: DataType = DataType.BFLOAT16,
    ):
        super().__init__()

        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.tp_size = tp_size

        self.attention_norm = MockRMSNormLayer(embedding_size, dtype=dtype)
        self.attention = MockDistributedGroupedQueryAttention(
            embedding_dim=embedding_size,
            num_heads=num_heads,
            num_groups=num_groups,
            dropout=dropout,
            rope_params=rope_params,
            tp_size=tp_size,
            dtype=dtype,
        )
        self.mlp_norm = MockRMSNormLayer(embedding_size, dtype=dtype)
        self.mlp = MockDistributedLlamaMLP(
            embedding_size=embedding_size,
            dropout=dropout,
            intermediate_dim=intermediate_dim,
            tp_size=tp_size,
            dtype=dtype,
        )

    def forward(
        self,
        ctx: "RooflineContext",
        x: MockTensor,
        mask: Optional[MockTensor] = None,
    ) -> MockTensor:
        residual = x
        h = self.attention_norm(ctx, x)
        h = self.attention(ctx, h, mask)
        h = MockAddOp.apply(ctx, h, residual)

        residual = h
        out = self.mlp_norm(ctx, h)
        out = self.mlp(ctx, out)
        out = MockAddOp.apply(ctx, out, residual)

        return out

    def __repr__(self) -> str:
        return (
            f"MockDistributedLlamaBlock(embedding_size={self.embedding_size}, "
            f"num_heads={self.num_heads}, num_groups={self.num_groups}, "
            f"tp_size={self.tp_size})"
        )
