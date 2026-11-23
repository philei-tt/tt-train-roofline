# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Cross-entropy loss operation for roofline modeling.

This module provides MockCrossEntropyLossOp for roofline estimation of
cross-entropy loss computation.
"""

from __future__ import annotations
import dataclasses
from typing import Tuple, TYPE_CHECKING

from ..hardware import DataType
from ..mock_tensor import MockTensor, TensorLabel
from ..roofline import cross_entropy_roofline
from .operation import (
    RooflineFunctionContext,
    RooflineFunction,
    create_grad_tensor,
    create_activation_tensor,
)

if TYPE_CHECKING:
    from ..roofline import RooflineContext


class MockCrossEntropyLossOp(RooflineFunction):
    """Roofline estimation for cross-entropy loss.

    CrossEntropyLoss = -log(softmax(logits)[target])

    Forward: softmax + log + gather + reduce
    Backward: grad_logits = softmax - one_hot(target)
    """

    @staticmethod
    def forward(
        ctx: RooflineFunctionContext,
        roofline_ctx: "RooflineContext",
        logits: MockTensor,
        targets: MockTensor,
    ) -> MockTensor:
        """Forward pass: compute cross-entropy loss.

        Args:
            ctx: Function context
            roofline_ctx: Roofline context for estimates
            logits: Logits tensor [B, 1, S, vocab_size]
            targets: Target indices tensor [B, 1, 1, S]

        Returns:
            Loss tensor (scalar or per-token depending on reduction)
        """
        ctx.save_for_backward(logits, targets)

        # Get dimensions
        batch_seq = targets.logical_volume()
        vocab_size = logits.shape[-1]

        estimate = cross_entropy_roofline(
            roofline_ctx.hw,
            batch_seq,
            vocab_size,
            dtype=logits.dtype,
            operation="CrossEntropyLoss.forward",
            phase="forward",
        )
        roofline_ctx.add_perf_result(estimate)

        # Output is scalar loss (or per-sample losses)
        output_shape = (1,)  # Scalar after reduction
        return create_activation_tensor(output_shape, logits.dtype, logits.layout)

    @staticmethod
    def backward(
        ctx: RooflineFunctionContext,
        roofline_ctx: "RooflineContext",
        grad_output: MockTensor,
    ) -> Tuple[MockTensor, None]:
        """Backward pass: compute gradient for logits.

        Args:
            ctx: Function context with saved tensors
            roofline_ctx: Roofline context for estimates
            grad_output: Gradient from upstream (scalar)

        Returns:
            (grad_logits, None) tuple - None for targets (not differentiable)
        """
        logits, targets = ctx.saved_tensors

        batch_seq = targets.logical_volume()
        vocab_size = logits.shape[-1]

        estimate = cross_entropy_roofline(
            roofline_ctx.hw,
            batch_seq,
            vocab_size,
            dtype=logits.dtype,
            operation="CrossEntropyLoss.backward",
            phase="backward",
        )
        roofline_ctx.add_perf_result(estimate)

        # Gradient has same shape as logits
        grad_logits = create_grad_tensor(
            logits.shape, logits.dtype, logits.layout, name="grad_logits"
        )

        return grad_logits, None


class MockVocabParallelCrossEntropyLossOp(RooflineFunction):
    """Cross-entropy on logits sharded over the vocabulary across tp_size devices.

    Mirrors tt-train's ttml::ops::distributed::vocab_parallel_cross_entropy_loss
    (ops/distributed/losses.cpp), pass by pass over the [S, V/tp] logits shard:
    - forward: max (read bf16), fused subtract + exp to an fp32 buffer (read 2, write 4), sum over the
      fp32 buffer (read 4): 12 bytes per logit; the per-token max is all-gathered, the sum and the
      target logit all-reduced over TP, each a [S, 1] tensor padded to one 32-wide tile column
    - backward: subtract + exp to fp32 again (read 2, write 4), multiply to bf16 (read 4, write 2),
      subtract at the target (read 2, write 2), multiply by the incoming gradient in place (read 2,
      write 2): 18 bytes per logit
    The fp32 exp buffer (and in backward the bf16 softmax) exist only during the pass.
    """

    @staticmethod
    def _passes(roofline_ctx, logits, passes, phase):
        from ..roofline import elementwise_roofline

        n = logits.logical_volume()
        hw = roofline_ctx.hw
        for name, read, write, sfpu in passes:
            est = elementwise_roofline(hw, n, num_inputs=1, sfpu_ops_per_element=sfpu, dtype=logits.dtype,
                                       operation=f"VocabParallelCrossEntropy.{name}", phase=phase)
            nbytes = n * (read + write)
            roofline_ctx.add_perf_result(dataclasses.replace(
                est, total_bytes=nbytes, ideal_memory_ns=nbytes / hw.dram_bw_gb_s))

    @staticmethod
    def forward(
        ctx: RooflineFunctionContext,
        roofline_ctx: "RooflineContext",
        logits: MockTensor,
        targets: MockTensor,
        tp_size: int,
    ) -> MockTensor:
        from ..roofline import all_gather_roofline, all_reduce_roofline

        ctx.save_for_backward(logits, targets)
        rows = targets.logical_volume()
        exp_buffer = MockTensor(logits.shape, DataType.FLOAT32, logits.layout, requires_grad=False,
                                label=TensorLabel.ACTIVATION, name="ce_local_exp")
        MockVocabParallelCrossEntropyLossOp._passes(roofline_ctx, logits, [
            ("max", 2, 0, 0.0), ("sub_exp", 2, 4, 16.0), ("sum", 4, 0, 1.0),
        ], "forward")
        del exp_buffer
        if tp_size > 1:
            stat = rows * 32  # [S, 1] padded to a 32-wide tile column
            roofline_ctx.add_perf_result(all_gather_roofline(
                roofline_ctx.hw_for("tp"), stat * logits.dtype.value, tp_size,
                operation="VocabParallelCrossEntropy.max_gather", phase="forward",
            ))
            for name in ("sum_all_reduce", "target_all_reduce"):
                roofline_ctx.add_perf_result(all_reduce_roofline(
                    roofline_ctx.hw_for("tp"), stat * 4, tp_size,
                    operation=f"VocabParallelCrossEntropy.{name}", phase="forward",
                ))
        return create_activation_tensor((1,), logits.dtype, logits.layout)

    @staticmethod
    def backward(
        ctx: RooflineFunctionContext,
        roofline_ctx: "RooflineContext",
        grad_output: MockTensor,
    ) -> Tuple[MockTensor, None, None]:
        logits, targets = ctx.saved_tensors
        # the bf16 softmax buffer becomes the gradient (scaled in place); the fp32 exp buffer is freed
        grad = create_grad_tensor(logits.shape, logits.dtype, logits.layout, name="grad_logits")
        exp_buffer = MockTensor(logits.shape, DataType.FLOAT32, logits.layout, requires_grad=False,
                                label=TensorLabel.ACTIVATION, name="ce_local_exp")
        MockVocabParallelCrossEntropyLossOp._passes(roofline_ctx, logits, [
            ("sub_exp", 2, 4, 16.0), ("scale", 4, 2, 1.0), ("subtract_at_target", 2, 2, 0.0),
            ("multiply_grad", 2, 2, 1.0),
        ], "backward")
        del exp_buffer
        return grad, None, None
