# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Distributed (tensor-parallel) Grouped Query Attention for roofline modeling.

Mirrors ttml::modules::distributed::DistributedGroupedQueryAttention.
Q/KV use column-parallel linears; output uses row-parallel linear.
Heads are sharded across TP: each device has num_local_heads and num_local_groups.
"""

from __future__ import annotations
from typing import Optional, TYPE_CHECKING

from ..mock_tensor import MockTensor
from ..hardware import DataType
from ..operations import (
    MockGroupedHeadsCreationOp,
    MockHeadsFusionOp,
    MockScaledDotProductAttentionOp,
    MockDropoutOp,
)
from .module import MockModule
from .distributed_linear import MockColumnParallelLinear, MockRowParallelLinear
from .rope import MockRotaryEmbedding
from .grouped_query_attention import RoPEParams

if TYPE_CHECKING:
    from ..roofline import RooflineContext


class MockDistributedGroupedQueryAttention(MockModule):
    """Distributed GQA with tensor parallelism.

    Mirrors ttml::modules::distributed::DistributedGroupedQueryAttention.
    - q_linear: ColumnParallelLinear (gather_output=False)
    - kv_linear: ColumnParallelLinear (gather_output=False)
    - out_linear: RowParallelLinear (input_is_parallel=True)
    Uses num_local_heads and num_local_groups for head creation (sharded across TP).
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        num_groups: int,
        dropout: float = 0.0,
        rope_params: Optional[RoPEParams] = None,
        tp_size: int = 1,
        dtype: DataType = DataType.BFLOAT16,
    ):
        super().__init__()

        if num_heads % tp_size != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible by tp_size ({tp_size})"
            )
        if num_groups % tp_size != 0:
            raise ValueError(
                f"num_groups ({num_groups}) must be divisible by tp_size ({tp_size})"
            )

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.num_local_heads = num_heads // tp_size
        self.num_local_groups = num_groups // tp_size
        self.head_dim = embedding_dim // num_heads
        self.dropout_prob = dropout
        self.tp_size = tp_size

        self.q_linear = MockColumnParallelLinear(
            embedding_dim,
            embedding_dim,
            has_bias=False,
            gather_output=False,
            tp_size=tp_size,
            dtype=dtype,
        )
        concat_kv_dim = 2 * num_groups * self.head_dim
        self.kv_linear = MockColumnParallelLinear(
            embedding_dim,
            concat_kv_dim,
            has_bias=False,
            gather_output=False,
            tp_size=tp_size,
            dtype=dtype,
        )
        self.out_linear = MockRowParallelLinear(
            embedding_dim,
            embedding_dim,
            has_bias=False,
            input_is_parallel=True,
            tp_size=tp_size,
            dtype=dtype,
        )

        if rope_params is not None:
            self.rope = MockRotaryEmbedding(
                head_dim=rope_params.head_dim,
                max_seq_len=rope_params.max_seq_len,
                theta=rope_params.theta,
                dtype=dtype,
            )
        else:
            self.rope = None

    def forward(
        self,
        ctx: "RooflineContext",
        x: MockTensor,
        mask: Optional[MockTensor] = None,
    ) -> MockTensor:
        q = self.q_linear(ctx, x)
        kv = self.kv_linear(ctx, x)

        query, key, value = MockGroupedHeadsCreationOp.apply(
            ctx, q, kv, self.num_local_heads, self.num_local_groups
        )

        if self.rope is not None:
            query = self.rope(ctx, query)
            key = self.rope(ctx, key)

        attn_out = MockScaledDotProductAttentionOp.apply(ctx, query, key, value, mask)
        merged = MockHeadsFusionOp.apply(ctx, attn_out)
        out = self.out_linear(ctx, merged)

        if self.dropout_prob > 0:
            out = MockDropoutOp.apply(ctx, out, self.dropout_prob)

        return out

    def __repr__(self) -> str:
        return (
            f"MockDistributedGroupedQueryAttention(embedding_dim={self.embedding_dim}, "
            f"num_heads={self.num_heads}, num_groups={self.num_groups}, "
            f"tp_size={self.tp_size})"
        )
