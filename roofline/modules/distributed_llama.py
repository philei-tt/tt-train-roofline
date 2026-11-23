# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Distributed (tensor-parallel) Llama model for roofline modeling.

Mirrors ttml::models::distributed::llama::DistributedLlama.
Same as MockLlama but uses ColumnParallelLinear for the output projection (fc)
and MockDistributedLlamaBlock for each transformer block.

Optionally the model is one stage of a pipeline (num_stages > 1): it holds stage_num_blocks
blocks (L/P by default; splits may be uneven), the first stage the token embedding,
the last stage the final norm and LM head.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, TYPE_CHECKING

from ..mock_tensor import MockTensor, memory_efficient_runner
from ..hardware import DataType
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
    stage_num_blocks: Optional[int] = None  # blocks on this stage; default num_blocks // num_stages


class MockDistributedLlama(MockModule):
    """Distributed Llama model with tensor parallelism.

    Mirrors ttml::models::distributed::llama::DistributedLlama.
    - tok_emb: same as Llama (replicated); first pipeline stage only
    - blocks: MockDistributedLlamaBlock (col/row linear inside)
    - ln_fc: RMSNorm (replicated); stages holding the LM head only
    - fc: ColumnParallelLinear(embedding_dim, vocab_size, gather_output=False): logits stay sharded
      over the vocabulary for MockVocabParallelCrossEntropyLossOp (as tt-train)
    """

    def __init__(
        self,
        config: MockDistributedLlamaConfig,
        dtype: DataType = DataType.BFLOAT16,
        runner: str = "default",
    ):
        super().__init__()

        if config.stage_num_blocks is None and config.num_blocks % config.num_stages != 0:
            raise ValueError(
                f"num_blocks ({config.num_blocks}) must be divisible by num_stages ({config.num_stages}) "
                "unless stage_num_blocks is set"
            )
        stage_blocks = (config.stage_num_blocks if config.stage_num_blocks is not None
                        else config.num_blocks // config.num_stages)
        self.config = config
        self.dtype = dtype
        self.runner = runner
        tp_size = config.tp_size
        self.is_first = config.stage_index == 0
        self.is_last = config.stage_index == config.num_stages - 1
        self.has_head = self.is_last

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
                for _ in range(stage_blocks)
            ]
        )
        self.ln_fc = None
        self.fc = None
        if self.has_head:
            self.ln_fc = MockRMSNormLayer(config.embedding_dim, dtype=dtype)
            self.fc = MockColumnParallelLinear(
                config.embedding_dim,
                vocab_size_padded,
                has_bias=False,
                gather_output=False,
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
        out = self.forward_blocks(ctx, x, mask)
        return out, self.forward_head(ctx, out)

    def forward_blocks(
        self,
        ctx: "RooflineContext",
        x: MockTensor,
        mask: Optional[MockTensor] = None,
    ) -> MockTensor:
        """Embedding (first stage) and this stage's blocks: the activation sent to the next stage.

        With sequence parallelism the first stage receives only its own S / tp token indices: each TP
        device embeds its own tokens (no communication) and accumulates a partial table gradient,
        summed over TP once per step with the replica sync."""
        out = x
        if self.tok_emb is not None:
            out = self.tok_emb(ctx, x)
        for block in self.blocks:
            if self.runner == "mem_eff":
                out = memory_efficient_runner(block, ctx, out, mask)
            else:
                out = block(ctx, out, mask)
        return out

    def forward_head(self, ctx: "RooflineContext", out: MockTensor) -> Optional[MockTensor]:
        """Final norm and LM head on the last stage: the logits (sharded over the vocabulary
        across TP), else None."""
        if not self.has_head:
            return None
        return self.fc(ctx, self.ln_fc(ctx, out))

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
