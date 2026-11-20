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
    MockScaledDotProductAttentionFusedOp,
    MockRingAttentionOp,
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
    sequence_parallel: input/output sharded along the sequence across TP devices.
    cp_size > 1: the input holds S/cp_size tokens and attention runs as ring attention.
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
        sequence_parallel: bool = False,
        cp_size: int = 1,
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
        self.cp_size = cp_size

        self.q_linear = MockColumnParallelLinear(
            embedding_dim,
            embedding_dim,
            has_bias=False,
            gather_output=False,
            tp_size=tp_size,
            dtype=dtype,
            sequence_parallel=sequence_parallel,
        )
        concat_kv_dim = 2 * num_groups * self.head_dim
        self.kv_linear = MockColumnParallelLinear(
            embedding_dim,
            concat_kv_dim,
            has_bias=False,
            gather_output=False,
            tp_size=tp_size,
            dtype=dtype,
            sequence_parallel=sequence_parallel,
        )
        self.out_linear = MockRowParallelLinear(
            embedding_dim,
            embedding_dim,
            has_bias=False,
            input_is_parallel=True,
            tp_size=tp_size,
            dtype=dtype,
            sequence_parallel=sequence_parallel,
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
        if self.q_linear.sequence_parallel:
            # Sequence parallelism: gather the norm output once for both projections (Megatron)
            x = self.q_linear.gather_input(ctx, x)
            q = self.q_linear(ctx, x, input_gathered=True)
            kv = self.kv_linear(ctx, x, input_gathered=True)
        else:
            q = self.q_linear(ctx, x)
            kv = self.kv_linear(ctx, x)

        query, key, value = MockGroupedHeadsCreationOp.apply(
            ctx, q, kv, self.num_local_heads, self.num_local_groups
        )

        if self.rope is not None:
            query = self.rope(ctx, query)
            key = self.rope(ctx, key)

        if self.cp_size > 1:
            attn_out = MockRingAttentionOp.apply(ctx, query, key, value, self.cp_size)
        else:
            attn_out = MockScaledDotProductAttentionFusedOp.apply(ctx, query, key, value, mask)
        merged = MockHeadsFusionOp.apply(ctx, attn_out)
        out = self.out_linear(ctx, merged)

        if self.dropout_prob > 0:
            out = MockDropoutOp.apply(ctx, out, self.dropout_prob)

        return out

    def __repr__(self) -> str:
        return (
            f"MockDistributedGroupedQueryAttention(embedding_dim={self.embedding_dim}, "
            f"num_heads={self.num_heads}, num_groups={self.num_groups}, "
            f"tp_size={self.tp_size}, cp_size={self.cp_size})"
        )
