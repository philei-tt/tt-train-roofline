# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Training utilities for roofline modeling.

This package contains training-related utilities for roofline estimation:
- MockAdamW: Optimizer roofline estimation
- mock_clip_grad_norm: Gradient clipping roofline estimation
- shard_batch: Shard tensors for data parallelism
- synchronize_gradients: All-reduce + average gradients for DDP
"""

from .optimizer import MockAdamW
from .grad_utils import mock_clip_grad_norm
from .distributed import shard_batch, synchronize_gradients

__all__ = [
    "MockAdamW",
    "mock_clip_grad_norm",
    "shard_batch",
    "synchronize_gradients",
]
