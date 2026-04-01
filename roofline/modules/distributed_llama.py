# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Distributed (tensor-parallel) Llama model for roofline modeling.

Mirrors ttml::models::distributed::llama::DistributedLlama.
Same as MockLlama but uses ColumnParallelLinear for the output projection (fc)
and MockDistributedLlamaBlock for each transformer block.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

from ..mock_tensor import MockTensor, memory_efficient_runner
from ..hardware import DataType
from .module import MockModule, MockModuleList
from .embedding import MockEmbedding
from .rmsnorm import MockRMSNormLayer
from .distributed_linear import MockColumnParallelLinear
from .distributed_llama_block import MockDistributedLlamaBlock
from .grouped_query_attention import RoPEParams

if TYPE_CHECKING:
    from ..roofline import RooflineContext


@dataclass
class MockDistributedLlamaConfig:
    """Configuration for MockDistributedLlama (TP)."""

    vocab_size: int = 32000
    max_sequence_length: int = 2048
    embedding_dim: int = 4096
    intermediate_dim: Optional[int] = None
    num_heads: int = 32
    num_groups: int = 8
    dropout_prob: float = 0.0
    num_blocks: int = 32
    theta: float = 10000.0
    tp_size: int = 1


class MockDistributedLlama(MockModule):
    """Distributed Llama model with tensor parallelism.

    Mirrors ttml::models::distributed::llama::DistributedLlama.
    - tok_emb: same as Llama (replicated)
    - blocks: MockDistributedLlamaBlock (col/row linear inside)
    - ln_fc: RMSNorm (replicated)
    - fc: ColumnParallelLinear(embedding_dim, vocab_size, gather_output=True)
    """

    def __init__(
        self,
        config: MockDistributedLlamaConfig,
        dtype: DataType = DataType.BFLOAT16,
        runner: str = "default",
    ):
        super().__init__()

        self.config = config
        self.dtype = dtype
        self.runner = runner
        tp_size = config.tp_size

        vocab_size_padded = (config.vocab_size + 31) // 32 * 32
        head_dim = config.embedding_dim // config.num_heads
        rope_params = RoPEParams(
            head_dim=head_dim,
            max_seq_len=config.max_sequence_length,
            theta=config.theta,
        )

        self.tok_emb = MockEmbedding(
            vocab_size_padded, config.embedding_dim, dtype=dtype
        )
        self.blocks = MockModuleList(
            [
                MockDistributedLlamaBlock(
                    embedding_size=config.embedding_dim,
                    num_heads=config.num_heads,
                    num_groups=config.num_groups,
                    rope_params=rope_params,
                    dropout=config.dropout_prob,
                    intermediate_dim=config.intermediate_dim,
                    tp_size=tp_size,
                    dtype=dtype,
                )
                for _ in range(config.num_blocks)
            ]
        )
        self.ln_fc = MockRMSNormLayer(config.embedding_dim, dtype=dtype)
        self.fc = MockColumnParallelLinear(
            config.embedding_dim,
            vocab_size_padded,
            has_bias=False,
            gather_output=True,
            tp_size=tp_size,
            dtype=dtype,
        )

    def forward(
        self,
        ctx: "RooflineContext",
        indices: MockTensor,
        mask: Optional[MockTensor] = None,
    ) -> MockTensor:
        out = self.tok_emb(ctx, indices)
        for block in self.blocks:
            if self.runner == "mem_eff":
                out = memory_efficient_runner(block, ctx, out, mask)
            else:
                out = block(ctx, out, mask)
        out = self.ln_fc(ctx, out)
        logits = self.fc(ctx, out)
        return logits

    def __repr__(self) -> str:
        return (
            f"MockDistributedLlama(\n"
            f"  vocab_size={self.config.vocab_size},\n"
            f"  embedding_dim={self.config.embedding_dim},\n"
            f"  num_blocks={self.config.num_blocks},\n"
            f"  tp_size={self.config.tp_size}\n"
            f")"
        )


def create_mock_distributed_llama(config: MockDistributedLlamaConfig) -> MockDistributedLlama:
    """Factory to create MockDistributedLlama."""
    return MockDistributedLlama(config)
