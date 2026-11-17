# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""One pipeline stage of a distributed Llama for roofline modeling.

A stage holds a contiguous run of transformer blocks (the blocks of all its interleaved chunks
together, since a device runs all of them for every micro-batch). The first stage also holds the
token embedding; the last stage holds the final norm and the LM head. With vocabulary parallelism
(Yeung et al., 2024, "Balancing Pipeline Parallelism with Vocabulary Parallelism") every stage
holds 1/num_stages of the LM head instead.

Inside a stage: tensor parallelism (tp_size), optionally with sequence parallelism, and context
parallelism (cp_size, ring attention). The stage sees S / cp_size tokens of each sequence.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

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
class MockLlamaPipelineStageConfig:
    """Configuration for one pipeline stage."""

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
    cp_size: int = 1
    sequence_parallel: bool = False
    num_stages: int = 1
    stage_index: int = 0
    vocab_parallel: bool = False


class MockLlamaPipelineStage(MockModule):
    """Blocks [stage_index * L/P, (stage_index + 1) * L/P) of a distributed Llama.

    Forward input: token indices on the first stage, the received activation otherwise.
    Forward output: logits on stages holding (part of) the LM head, the activation otherwise.
    """

    def __init__(
        self,
        config: MockLlamaPipelineStageConfig,
        dtype: DataType = DataType.BFLOAT16,
        runner: str = "default",
    ):
        super().__init__()
        if config.num_blocks % config.num_stages != 0:
            raise ValueError(
                f"num_blocks ({config.num_blocks}) must be divisible by num_stages ({config.num_stages})"
            )
        self.config = config
        self.runner = runner
        self.is_first = config.stage_index == 0
        self.is_last = config.stage_index == config.num_stages - 1
        self.has_head = self.is_last or config.vocab_parallel
        tp_size = config.tp_size

        head_dim = config.embedding_dim // config.num_heads
        rope_params = RoPEParams(head_dim=head_dim, max_seq_len=config.max_sequence_length, theta=config.theta)
        vocab_size_padded = (config.vocab_size + 31) // 32 * 32

        self.tok_emb = MockEmbedding(vocab_size_padded, config.embedding_dim, dtype=dtype) if self.is_first else None
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
        else:
            self.ln_fc = None
            self.fc = None

    def forward(
        self,
        ctx: "RooflineContext",
        x: MockTensor,
        mask: Optional[MockTensor] = None,
    ) -> MockTensor:
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
        if self.has_head:
            head_in = out
            if self.config.vocab_parallel and not self.is_last:
                # The last stage's final hidden state, broadcast to every stage (not modelled)
                head_in = MockTensor(
                    out.shape, out.dtype, out.layout, requires_grad=True,
                    label=out.label, name="final_hidden", num_shards=out.num_shards,
                )
            logits = self.fc(ctx, self.ln_fc(ctx, head_in))
            return out, logits
        return out, None

    def __repr__(self) -> str:
        c = self.config
        return (
            f"MockLlamaPipelineStage(stage={c.stage_index}/{c.num_stages}, "
            f"blocks={len(self.blocks)}, tp={c.tp_size}, cp={c.cp_size}, "
            f"sp={c.sequence_parallel}, head={self.has_head})"
        )
