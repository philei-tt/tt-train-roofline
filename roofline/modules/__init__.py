# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Modules for roofline modeling.

This package contains module implementations for roofline estimation.
"""

from .module import MockParameter, MockModule, MockModuleList, MockModuleDict
from .linear import MockLinearLayer
from .embedding import MockEmbedding, MockTrainablePositionalEmbedding
from .layernorm import MockLayerNorm
from .rmsnorm import MockRMSNormLayer
from .dropout import MockDropout
from .attention import MockMultiHeadAttention
from .rope import MockRotaryEmbedding
from .grouped_query_attention import MockGroupedQueryAttention, RoPEParams
from .mlp import MockGPTMLP
from .llama_mlp import MockLlamaMLP
from .llama_mlp_fused import MockLlamaMLPFused
from .gpt_block import MockGPTBlock
from .llama_block import MockLlamaBlock
from .nanogpt import MockNanoGPT, MockNanoGPTConfig, create_mock_nanogpt
from .llama import MockLlama, MockLlamaConfig, create_mock_llama
from .distributed_linear import MockColumnParallelLinear, MockRowParallelLinear
from .distributed_llama import (
    MockDistributedLlama,
    MockDistributedLlamaConfig,
    create_mock_distributed_llama,
)
from .distributed_llama_block import MockDistributedLlamaBlock
from .distributed_llama_mlp import MockDistributedLlamaMLP
from .distributed_grouped_query_attention import MockDistributedGroupedQueryAttention

__all__ = [
    # Base classes
    "MockParameter",
    "MockModule",
    "MockModuleList",
    "MockModuleDict",
    # Linear
    "MockLinearLayer",
    # Distributed Linear (TP)
    "MockColumnParallelLinear",
    "MockRowParallelLinear",
    # Embedding
    "MockEmbedding",
    "MockTrainablePositionalEmbedding",
    # Normalization
    "MockLayerNorm",
    "MockRMSNormLayer",
    # Dropout
    "MockDropout",
    # Attention
    "MockMultiHeadAttention",
    "MockGroupedQueryAttention",
    "RoPEParams",
    # Position Embedding
    "MockRotaryEmbedding",
    # MLP
    "MockGPTMLP",
    "MockLlamaMLP",
    "MockLlamaMLPFused",
    # Transformer Blocks
    "MockGPTBlock",
    "MockLlamaBlock",
    # NanoGPT
    "MockNanoGPT",
    "MockNanoGPTConfig",
    "create_mock_nanogpt",
    # Llama
    "MockLlama",
    "MockLlamaConfig",
    "create_mock_llama",
    # Distributed Llama (TP)
    "MockDistributedLlama",
    "MockDistributedLlamaConfig",
    "create_mock_distributed_llama",
    "MockDistributedLlamaBlock",
    "MockDistributedLlamaMLP",
    "MockDistributedGroupedQueryAttention",
]
