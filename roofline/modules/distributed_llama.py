# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Distributed (tensor-parallel) Llama model for roofline modeling.

Mirrors ttml::models::distributed::llama::DistributedLlama.
Same as MockLlama but uses ColumnParallelLinear for the output projection (fc)
and MockDistributedLlamaBlock for each transformer block.

Optionally the model is one stage of a pipeline (num_stages > 1): it holds blocks
[stage_index * L/P, (stage_index + 1) * L/P), the first stage the token embedding,
the last stage the final norm and LM head. With vocab_parallel (Yeung et al., 2024,
"Balancing Pipeline Parallelism with Vocabulary Parallelism") every stage holds 1/P of
the LM head instead.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, TYPE_CHECKING

from ..mock_tensor import MockTensor, memory_efficient_runner
from ..hardware import DataType
from ..operations.comm_ops import MockScatterOp
from .module import MockModule, MockModuleList
from .embedding import MockEmbedding
from .rmsnorm import MockRMSNormLayer
from .distributed_linear import MockColumnParallelLinear, SEQ_DIM
from .distributed_llama_block import MockDistributedLlamaBlock
from .grouped_query_attention import RoPEParams

if TYPE_CHECKING:
    from ..roofline import RooflineContext


@dataclass
class MockDistributedLlamaConfig:
    """Configuration for MockDistributedLlama (TP, optionally SP, CP and one pipeline stage)."""

    vocab_size: int = 32000
    max_sequence_length: int = 2048
    embedding_dim: int = 4096
    intermediate_dim: Optional[int] = None
    num_heads: int = 32
    num_groups: int = 8
    dropout_prob: float = 0.0
    num_blocks: int = 32  # blocks of the whole model
    theta: float = 10000.0
    tp_size: int = 1
    sequence_parallel: bool = False  # norms and residual adds hold S / tp_size tokens
    cp_size: int = 1  # context parallelism: S / cp_size tokens per device, ring attention
    num_stages: int = 1  # pipeline stages
    stage_index: int = 0
    vocab_parallel: bool = False  # LM head split across all pipeline stages


class MockDistributedLlama(MockModule):
    """Distributed Llama model with tensor parallelism.

    Mirrors ttml::models::distributed::llama::DistributedLlama.
    - tok_emb: same as Llama (replicated); first pipeline stage only
    - blocks: MockDistributedLlamaBlock (col/row linear inside)
    - ln_fc: RMSNorm (replicated); stages holding the LM head only
    - fc: ColumnParallelLinear(embedding_dim, vocab_size, gather_output=True)
    """

    def __init__(
        self,
        config: MockDistributedLlamaConfig,
        dtype: DataType = DataType.BFLOAT16,
        runner: str = "default",
    ):
        super().__init__()

        if config.num_blocks % config.num_stages != 0:
            raise ValueError(
                f"num_blocks ({config.num_blocks}) must be divisible by num_stages ({config.num_stages})"
            )
        self.config = config
        self.dtype = dtype
        self.runner = runner
        tp_size = config.tp_size
        self.is_first = config.stage_index == 0
        self.is_last = config.stage_index == config.num_stages - 1
        self.has_head = self.is_last or config.vocab_parallel

        vocab_size_padded = (config.vocab_size + 31) // 32 * 32
        head_dim = config.embedding_dim // config.num_heads
        rope_params = RoPEParams(
            head_dim=head_dim,
            max_seq_len=config.max_sequence_length,
            theta=config.theta,
        )

        self.tok_emb = (
            MockEmbedding(vocab_size_padded, config.embedding_dim, dtype=dtype) if self.is_first else None
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
                    sequence_parallel=config.sequence_parallel,
                    cp_size=config.cp_size,
                )
                for _ in range(config.num_blocks // config.num_stages)
            ]
        )
        self.ln_fc = None
        self.fc = None
        if self.has_head:
            head_vocab = vocab_size_padded
            if config.vocab_parallel and config.num_stages > 1:
                quantum = 32 * tp_size
                head_vocab = (vocab_size_padded // config.num_stages + quantum - 1) // quantum * quantum
            self.ln_fc = MockRMSNormLayer(config.embedding_dim, dtype=dtype)
            self.fc = MockColumnParallelLinear(
                config.embedding_dim,
                head_vocab,
                has_bias=False,
                gather_output=True,
                tp_size=tp_size,
                dtype=dtype,
                sequence_parallel=config.sequence_parallel,
            )

    def forward_stage(
        self,
        ctx: "RooflineContext",
        x: MockTensor,
        mask: Optional[MockTensor] = None,
    ) -> Tuple[MockTensor, Optional[MockTensor]]:
        """Run this stage: token indices in on the first stage, the received activation otherwise.

        Returns (activation sent to the next stage, logits or None).
        """
        out = x
        if self.tok_emb is not None:
            out = self.tok_emb(ctx, x)
            if self.config.sequence_parallel and self.config.tp_size > 1:
                out = MockScatterOp.apply(ctx, out, SEQ_DIM, self.config.tp_size)
        for block in self.blocks:
            if self.runner == "mem_eff":
                out = memory_efficient_runner(block, ctx, out, mask)
            else:
                out = block(ctx, out, mask)
        if not self.has_head:
            return out, None
        head_in = out
        if not self.is_last:
            # Vocabulary parallelism: the last stage's final hidden state, broadcast to every
            # stage (the broadcast is not modelled)
            head_in = MockTensor(
                out.shape, out.dtype, out.layout, requires_grad=True,
                label=out.label, name="final_hidden", num_shards=out.num_shards,
            )
        return out, self.fc(ctx, self.ln_fc(ctx, head_in))

    def forward(
        self,
        ctx: "RooflineContext",
        indices: MockTensor,
        mask: Optional[MockTensor] = None,
    ) -> MockTensor:
        _, logits = self.forward_stage(ctx, indices, mask)
        return logits

    def __repr__(self) -> str:
        c = self.config
        stage = f",\n  stage={c.stage_index}/{c.num_stages}" if c.num_stages > 1 else ""
        return (
            f"MockDistributedLlama(\n"
            f"  vocab_size={c.vocab_size},\n"
            f"  embedding_dim={c.embedding_dim},\n"
            f"  num_blocks={c.num_blocks},\n"
            f"  tp_size={c.tp_size}, cp_size={c.cp_size}, sequence_parallel={c.sequence_parallel}{stage}\n"
            f")"
        )


def create_mock_distributed_llama(config: MockDistributedLlamaConfig) -> MockDistributedLlama:
    """Factory to create MockDistributedLlama."""
    return MockDistributedLlama(config)
