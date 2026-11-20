# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Distributed training utilities for roofline modeling.

Provides helpers for DDP (Data-Distributed Parallelism):
- shard_batch: split a tensor along a dimension for data parallelism
- synchronize_gradients: all-reduce + average parameter gradients

Mirrors ttml::core::distributed (distributed.cpp).
"""

from __future__ import annotations
from typing import Dict, Sequence, Tuple, TYPE_CHECKING

from ..mock_tensor import MockTensor, TensorLabel
from ..roofline import all_gather_roofline, all_reduce_roofline, elementwise_roofline, reduce_scatter_roofline

if TYPE_CHECKING:
    from ..roofline import RooflineContext


def shard_batch(
    tensor: MockTensor,
    dim: int,
    num_devices: int,
) -> MockTensor:
    """Shard a tensor along a dimension for data parallelism.

    Returns a new MockTensor whose shape along `dim` is divided by
    `num_devices`, with sharding metadata set.

    Args:
        tensor: Input tensor (global batch)
        dim: Dimension to shard along (typically 0 for batch)
        num_devices: Number of DDP devices

    Returns:
        MockTensor with per-device shape and sharding metadata
    """
    if num_devices <= 1:
        return tensor

    assert tensor.shape[dim] % num_devices == 0, (
        f"shard_batch: shape[{dim}]={tensor.shape[dim]} not divisible by {num_devices}"
    )

    new_shape = list(tensor.shape)
    new_shape[dim] //= num_devices
    ndim = len(tensor.shape)
    new_num_shards = tuple(
        num_devices if i == dim else tensor.num_shards[i] for i in range(ndim)
    )

    return MockTensor(
        tuple(new_shape),
        dtype=tensor.dtype,
        layout=tensor.layout,
        requires_grad=tensor.requires_grad,
        label=tensor.label,
        name=tensor.name,
        num_shards=new_num_shards,
    )


def synchronize_gradients(
    ctx: "RooflineContext",
    parameters: Dict[str, MockTensor],
    num_devices: int,
    group: str = "dp",
) -> None:
    """Synchronize (all-reduce + average) parameter gradients across DDP devices.

    Mirrors ttml::core::distributed::synchronize_gradients:
      1. all_reduce each parameter gradient
      2. multiply by 1/N to average

    Args:
        ctx: Roofline context for accumulating estimates
        parameters: Dict of parameter name -> MockTensor
        num_devices: Number of DDP devices
        group: parallel group the replicas form ("dp", or "cp" for context-parallel replicas);
            picks its ring or line topology (RooflineContext.hw_for)
    """
    if num_devices <= 1:
        return

    for name, param in parameters.items():
        if not param.is_grad_initialized():
            continue

        grad = param.get_grad()
        tensor_bytes = grad.bytes() if grad is not None else param.bytes()

        # all-reduce
        estimate = all_reduce_roofline(
            ctx.hw_for(group),
            tensor_bytes,
            num_devices,
            operation=f"SyncGrad.AllReduce.{name}",
            phase="ccl",
        )
        ctx.add_perf_result(estimate)

        # divide by N (elementwise multiply by 1/N)
        num_elements = grad.logical_volume() if grad is not None else param.logical_volume()
        estimate = elementwise_roofline(
            ctx.hw,
            num_elements,
            num_inputs=1,
            sfpu_ops_per_element=0.0,
            fpu_ops_per_element=1.0,
            dtype=param.dtype,
            operation=f"SyncGrad.Scale.{name}",
            phase="ccl",
        )
        ctx.add_perf_result(estimate)


def synchronize_replica_gradients(
    ctx: "RooflineContext",
    parameters: Dict[str, MockTensor],
    groups: Sequence[Tuple[str, int]],
    zero1: bool = False,
) -> None:
    """Gradient sync across the replicas of a stage: CP and DP groups (phase "ccl").

    Without ZeRO-1, as tt-train's synchronize_gradients_along: one all-reduce of the full gradient per
    group, one group after the other. With ZeRO-1 (target), every replica keeps the optimizer state of
    1/N of each parameter (N = product of the group sizes): the gradient is reduce-scattered group by
    group, the optimizer updates the shard, and the updated parameter is all-gathered back.

    Args:
        groups: (group name, size) in order, e.g. [("cp", 4), ("dp", 2)]; hw_for(name) gives the
            group's topology and link share
    """
    groups = [(g, n) for g, n in groups if n > 1]
    if not groups:
        return
    for name, param in parameters.items():
        if not param.is_grad_initialized():
            continue
        grad = param.get_grad()
        replicas = 1
        for _, n in groups:
            replicas *= n
        # average: multiply by 1/N, on the full gradient or this replica's shard
        ctx.add_perf_result(elementwise_roofline(
            ctx.hw, grad.logical_volume() // (replicas if zero1 else 1), num_inputs=1, sfpu_ops_per_element=0.0,
            fpu_ops_per_element=1.0, dtype=grad.dtype, operation=f"SyncGrad.Scale.{name}", phase="ccl"))
        if not zero1:
            for g, n in groups:
                ctx.add_perf_result(all_reduce_roofline(
                    ctx.hw_for(g), grad.bytes(), n, operation=f"SyncGrad.AllReduce.{g}.{name}", phase="ccl"))
            continue
        size = grad.bytes()
        for g, n in groups:
            ctx.add_perf_result(reduce_scatter_roofline(
                ctx.hw_for(g), size, n, operation=f"SyncGrad.ReduceScatter.{g}.{name}", phase="ccl"))
            size //= n
        size = param.bytes()
        for g, n in groups:
            size //= n
        for g, n in reversed(groups):
            ctx.add_perf_result(all_gather_roofline(
                ctx.hw_for(g), size, n, operation=f"ZeRO1.AllGather.{g}.{name}", phase="ccl"))
            size *= n
