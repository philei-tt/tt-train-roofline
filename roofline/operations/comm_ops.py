# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""CCL (Collective Communication Library) operations for roofline modeling.

Mirrors ttml/ops/distributed/comm_ops.cpp with autograd-aware forward/backward
and roofline cost estimation for ring-based collectives.
"""

from __future__ import annotations
from typing import Optional, Tuple, TYPE_CHECKING

from ..mock_tensor import MockTensor
from ..hardware import DataType
from ..roofline import (
    all_reduce_roofline,
    reduce_scatter_roofline,
    all_gather_roofline,
    elementwise_roofline,
)
from .operation import (
    RooflineFunctionContext,
    RooflineFunction,
    create_grad_tensor,
    create_activation_tensor,
)

if TYPE_CHECKING:
    from ..roofline import RooflineContext


class MockAllReduceOp(RooflineFunction):
    """All-reduce across devices.

    Forward: sum tensor across all devices (ring all-reduce cost).
    Backward: noop (pass grad through) or all-reduce of gradients,
              controlled by noop_backward.

    Mirrors ttml::ops::distributed::all_reduce.
    """

    @staticmethod
    def forward(
        ctx: RooflineFunctionContext,
        roofline_ctx: "RooflineContext",
        tensor: MockTensor,
        noop_backward: bool = False,
        num_devices: int = 1,
    ) -> MockTensor:
        ctx.save_for_backward(tensor)
        ctx._noop_backward = noop_backward
        ctx._num_devices = num_devices

        if num_devices > 1:
            estimate = all_reduce_roofline(
                roofline_ctx.hw,
                tensor.bytes(),
                num_devices,
                operation="AllReduce.forward",
                phase="forward",
            )
            roofline_ctx.add_perf_result(estimate)

        return create_activation_tensor(
            tensor.shape, tensor.dtype, tensor.layout,
            num_shards=tensor.num_shards,
        )

    @staticmethod
    def backward(
        ctx: RooflineFunctionContext,
        roofline_ctx: "RooflineContext",
        grad_output: MockTensor,
    ) -> Tuple[MockTensor]:
        (tensor,) = ctx.saved_tensors
        num_devices = ctx._num_devices

        if not ctx._noop_backward and num_devices > 1:
            estimate = all_reduce_roofline(
                roofline_ctx.hw,
                grad_output.bytes(),
                num_devices,
                operation="AllReduce.backward",
                phase="backward",
            )
            roofline_ctx.add_perf_result(estimate)

        grad_input = create_grad_tensor(
            tensor.shape, tensor.dtype, tensor.layout, name="grad_allreduce",
            num_shards=tensor.num_shards,
        )
        return (grad_input,)


class MockReduceScatterOp(RooflineFunction):
    """Reduce-scatter across devices.

    Forward: reduce + scatter along dim. Output shape has dim divided by N.
    Backward: all-gather (inverse of reduce-scatter).

    Mirrors ttml::ops::distributed::reduce_scatter.
    """

    @staticmethod
    def forward(
        ctx: RooflineFunctionContext,
        roofline_ctx: "RooflineContext",
        tensor: MockTensor,
        dim: int,
        num_devices: int = 1,
    ) -> MockTensor:
        ctx.save_for_backward(tensor)
        ctx._dim = dim
        ctx._num_devices = num_devices

        if num_devices > 1:
            assert tensor.shape[dim] % num_devices == 0, (
                f"ReduceScatter: shape[{dim}]={tensor.shape[dim]} not divisible by {num_devices}"
            )
            estimate = reduce_scatter_roofline(
                roofline_ctx.hw,
                tensor.bytes(),
                num_devices,
                operation="ReduceScatter.forward",
                phase="forward",
            )
            roofline_ctx.add_perf_result(estimate)

        out_shape = list(tensor.shape)
        ndim = len(tensor.shape)
        if num_devices > 1:
            out_shape[dim] //= num_devices
            new_num_shards = tuple(
                num_devices if i == dim else tensor.num_shards[i] for i in range(ndim)
            )
        else:
            new_num_shards = tensor.num_shards
        return create_activation_tensor(
            tuple(out_shape), tensor.dtype, tensor.layout,
            num_shards=new_num_shards,
        )

    @staticmethod
    def backward(
        ctx: RooflineFunctionContext,
        roofline_ctx: "RooflineContext",
        grad_output: MockTensor,
    ) -> Tuple[MockTensor]:
        (tensor,) = ctx.saved_tensors
        dim = ctx._dim
        num_devices = ctx._num_devices

        if num_devices > 1:
            estimate = all_gather_roofline(
                roofline_ctx.hw,
                grad_output.bytes(),
                num_devices,
                operation="ReduceScatter.backward(AllGather)",
                phase="backward",
            )
            roofline_ctx.add_perf_result(estimate)

        grad_input = create_grad_tensor(
            tensor.shape, tensor.dtype, tensor.layout, name="grad_reduce_scatter",
            num_shards=tensor.num_shards,
        )
        return (grad_input,)


class MockAllGatherOp(RooflineFunction):
    """All-gather across devices.

    Forward: gather shards along dim. Output shape has dim multiplied by N.
    Backward: reduce-scatter (inverse of all-gather).

    Mirrors ttml::ops::distributed::all_gather.
    """

    @staticmethod
    def forward(
        ctx: RooflineFunctionContext,
        roofline_ctx: "RooflineContext",
        tensor: MockTensor,
        dim: int,
        num_devices: int = 1,
    ) -> MockTensor:
        ctx.save_for_backward(tensor)
        ctx._dim = dim
        ctx._num_devices = num_devices

        if num_devices > 1:
            estimate = all_gather_roofline(
                roofline_ctx.hw,
                tensor.bytes(),
                num_devices,
                operation="AllGather.forward",
                phase="forward",
            )
            roofline_ctx.add_perf_result(estimate)

        out_shape = list(tensor.shape)
        ndim = len(tensor.shape)
        if num_devices > 1:
            out_shape[dim] *= num_devices
            new_num_shards = tuple(
                1 if i == dim else tensor.num_shards[i] for i in range(ndim)
            )
        else:
            new_num_shards = tensor.num_shards
        return create_activation_tensor(
            tuple(out_shape), tensor.dtype, tensor.layout,
            num_shards=new_num_shards,
        )

    @staticmethod
    def backward(
        ctx: RooflineFunctionContext,
        roofline_ctx: "RooflineContext",
        grad_output: MockTensor,
    ) -> Tuple[MockTensor]:
        (tensor,) = ctx.saved_tensors
        dim = ctx._dim
        num_devices = ctx._num_devices

        if num_devices > 1:
            estimate = reduce_scatter_roofline(
                roofline_ctx.hw,
                grad_output.bytes(),
                num_devices,
                operation="AllGather.backward(ReduceScatter)",
                phase="backward",
            )
            roofline_ctx.add_perf_result(estimate)

        grad_input = create_grad_tensor(
            tensor.shape, tensor.dtype, tensor.layout, name="grad_all_gather",
            num_shards=tensor.num_shards,
        )
        return (grad_input,)


class MockBroadcastOp(RooflineFunction):
    """Broadcast (identity in forward, all-reduce in backward).

    Forward: noop -- data is already replicated, zero communication cost.
    Backward: all-reduce of gradients (to sum contributions from all devices).

    Used in ColumnParallelLinear to ensure correct backward gradient flow.
    Mirrors ttml::ops::distributed::broadcast.
    """

    @staticmethod
    def forward(
        ctx: RooflineFunctionContext,
        roofline_ctx: "RooflineContext",
        tensor: MockTensor,
        num_devices: int = 1,
    ) -> MockTensor:
        ctx.save_for_backward(tensor)
        ctx._num_devices = num_devices

        return create_activation_tensor(
            tensor.shape, tensor.dtype, tensor.layout,
            num_shards=tensor.num_shards,
        )

    @staticmethod
    def backward(
        ctx: RooflineFunctionContext,
        roofline_ctx: "RooflineContext",
        grad_output: MockTensor,
    ) -> Tuple[MockTensor]:
        (tensor,) = ctx.saved_tensors
        num_devices = ctx._num_devices

        if num_devices > 1:
            estimate = all_reduce_roofline(
                roofline_ctx.hw,
                grad_output.bytes(),
                num_devices,
                operation="Broadcast.backward(AllReduce)",
                phase="backward",
            )
            roofline_ctx.add_perf_result(estimate)

        grad_input = create_grad_tensor(
            tensor.shape, tensor.dtype, tensor.layout, name="grad_broadcast",
            num_shards=tensor.num_shards,
        )
        return (grad_input,)


class MockScatterOp(RooflineFunction):
    """Scatter replicated input across devices.

    Forward: reduce-scatter + divide by N (scatter replicated data).
    Backward: all-gather (to reconstruct full gradient).

    Used in RowParallelLinear when input_is_parallel=False.
    Mirrors ttml::ops::distributed::scatter.
    """

    @staticmethod
    def forward(
        ctx: RooflineFunctionContext,
        roofline_ctx: "RooflineContext",
        tensor: MockTensor,
        dim: int,
        num_devices: int = 1,
    ) -> MockTensor:
        ctx.save_for_backward(tensor)
        ctx._dim = dim
        ctx._num_devices = num_devices

        if num_devices > 1:
            assert tensor.shape[dim] % num_devices == 0, (
                f"Scatter: shape[{dim}]={tensor.shape[dim]} not divisible by {num_devices}"
            )

            # reduce-scatter cost
            estimate = reduce_scatter_roofline(
                roofline_ctx.hw,
                tensor.bytes(),
                num_devices,
                operation="Scatter.forward(ReduceScatter)",
                phase="forward",
            )
            roofline_ctx.add_perf_result(estimate)

            # divide by N (elementwise scale, negligible but modeled)
            out_elements = tensor.logical_volume() // num_devices
            estimate = elementwise_roofline(
                roofline_ctx.hw,
                out_elements,
                num_inputs=1,
                sfpu_ops_per_element=0.0,
                fpu_ops_per_element=1.0,
                dtype=tensor.dtype,
                operation="Scatter.forward(Scale)",
                phase="forward",
            )
            roofline_ctx.add_perf_result(estimate)

        out_shape = list(tensor.shape)
        ndim = len(tensor.shape)
        if num_devices > 1:
            out_shape[dim] //= num_devices
            new_num_shards = tuple(
                num_devices if i == dim else tensor.num_shards[i] for i in range(ndim)
            )
        else:
            new_num_shards = tensor.num_shards
        return create_activation_tensor(
            tuple(out_shape), tensor.dtype, tensor.layout,
            num_shards=new_num_shards,
        )

    @staticmethod
    def backward(
        ctx: RooflineFunctionContext,
        roofline_ctx: "RooflineContext",
        grad_output: MockTensor,
    ) -> Tuple[MockTensor]:
        (tensor,) = ctx.saved_tensors
        dim = ctx._dim
        num_devices = ctx._num_devices

        if num_devices > 1:
            estimate = all_gather_roofline(
                roofline_ctx.hw,
                grad_output.bytes(),
                num_devices,
                operation="Scatter.backward(AllGather)",
                phase="backward",
            )
            roofline_ctx.add_perf_result(estimate)

        grad_input = create_grad_tensor(
            tensor.shape, tensor.dtype, tensor.layout, name="grad_scatter",
            num_shards=tensor.num_shards,
        )
        return (grad_input,)
