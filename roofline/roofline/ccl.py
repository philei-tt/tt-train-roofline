# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Roofline performance models for CCL (Collective Communication Library) operations.

Models ring/linear topology collective communication costs using per-link
Ethernet bandwidth. Ring topology has 2x better effective bandwidth than linear.
"""

from __future__ import annotations
from typing import Optional, TYPE_CHECKING

from ..hardware import HardwareSpec, DataType

if TYPE_CHECKING:
    from .roofline import RooflineEstimate


def _ccl_bytes_and_time(
    hw: HardwareSpec,
    tensor_bytes: int,
    num_devices: int,
    factor: float,
) -> tuple[int, float]:
    """Compute effective bytes moved and ideal time (ns) for a CCL phase.

    Formula: total_bytes_moved = factor * tensor_bytes * ((n-1)/n) / num_links / (2 if ring else 1)
    Then time_ns = total_bytes_moved / (eth_bw_gb_s_per_link * num_links)
    (with bytes and GB/s: time_ns = total_bytes_moved / (bw_gb_s * 1e9) * 1e9 = total_bytes_moved / bw_gb_s)

    factor: 2 for all-reduce (RS+AG), 1 for reduce-scatter or all-gather single phase.
    """
    if num_devices <= 1:
        return 0, 0.0
    n = num_devices
    topology_divisor = 2 if hw.topology == "ring" else 1
    total_bytes_moved = (
        factor
        * tensor_bytes
        * ((n - 1) / n)
        / hw.num_links
        / topology_divisor
    )
    total_bytes_moved = int(total_bytes_moved)
    # total_bytes_moved is already per-link (we divided by num_links above), so use per-link BW
    # time_ns = bytes / (GB/s) with bytes and GB/s gives 1e-9*s, so *1e9 => time_ns = bytes / bw_gb_s
    ideal_network_ns = total_bytes_moved / hw.eth_bw_gb_s_per_link if hw.eth_bw_gb_s_per_link > 0 else 0.0
    return total_bytes_moved, ideal_network_ns


def all_reduce_roofline(
    hw: HardwareSpec,
    tensor_bytes: int,
    num_devices: int,
    operation: str = "AllReduce",
    phase: str = "ccl",
) -> "RooflineEstimate":
    """Ring/linear all-reduce performance model.

    All-reduce = reduce-scatter + all-gather, so factor 2 in bytes moved.
    Uses topology (ring vs linear) and num_links; linear is 2x worse than ring.

    Args:
        hw: Hardware specification (uses eth_bw_gb_s_per_link, topology, num_links)
        tensor_bytes: Size of the tensor in bytes (per device)
        num_devices: Number of devices participating
        operation: Name for the operation
        phase: Phase label (default: "ccl")

    Returns:
        RooflineEstimate for the all-reduce
    """
    from .roofline import RooflineEstimate

    if num_devices <= 1:
        return RooflineEstimate(
            operation=operation, phase=phase,
            total_flops=0, total_bytes=0,
            ideal_compute_ns=0.0, ideal_memory_ns=0.0, hw=hw,
        )

    bytes_per_device, ideal_network_ns = _ccl_bytes_and_time(
        hw, tensor_bytes, num_devices, factor=2.0
    )

    return RooflineEstimate(
        operation=operation,
        phase=phase,
        total_flops=0,
        total_bytes=bytes_per_device,
        ideal_compute_ns=0.0,
        ideal_memory_ns=0.0,
        ideal_network_ns=ideal_network_ns,
        hw=hw,
    )


def reduce_scatter_roofline(
    hw: HardwareSpec,
    tensor_bytes: int,
    num_devices: int,
    operation: str = "ReduceScatter",
    phase: str = "ccl",
) -> "RooflineEstimate":
    """Ring/linear reduce-scatter performance model.

    Each device sends (N-1)/N of the tensor data through the ring/linear.
    tensor_bytes is the size of the full input tensor (per device, before scatter).

    Args:
        hw: Hardware specification
        tensor_bytes: Size of the input tensor in bytes (per device, before scatter)
        num_devices: Number of devices participating
        operation: Name for the operation
        phase: Phase label

    Returns:
        RooflineEstimate for the reduce-scatter
    """
    from .roofline import RooflineEstimate

    if num_devices <= 1:
        return RooflineEstimate(
            operation=operation, phase=phase,
            total_flops=0, total_bytes=0,
            ideal_compute_ns=0.0, ideal_memory_ns=0.0, hw=hw,
        )

    bytes_per_device, ideal_network_ns = _ccl_bytes_and_time(
        hw, tensor_bytes, num_devices, factor=1.0
    )

    return RooflineEstimate(
        operation=operation,
        phase=phase,
        total_flops=0,
        total_bytes=bytes_per_device,
        ideal_compute_ns=0.0,
        ideal_memory_ns=0.0,
        ideal_network_ns=ideal_network_ns,
        hw=hw,
    )


def all_gather_roofline(
    hw: HardwareSpec,
    tensor_bytes: int,
    num_devices: int,
    operation: str = "AllGather",
    phase: str = "ccl",
) -> "RooflineEstimate":
    """Ring/linear all-gather performance model.

    tensor_bytes is the per-device INPUT (shard) size; total output size is
    tensor_bytes * num_devices. Each device receives (N-1)/N of the total
    output, so we use total_output_bytes = tensor_bytes * n in the formula.

    Args:
        hw: Hardware specification
        tensor_bytes: Size of the per-device input (shard) in bytes
        num_devices: Number of devices participating
        operation: Name for the operation
        phase: Phase label

    Returns:
        RooflineEstimate for the all-gather
    """
    from .roofline import RooflineEstimate

    if num_devices <= 1:
        return RooflineEstimate(
            operation=operation, phase=phase,
            total_flops=0, total_bytes=0,
            ideal_compute_ns=0.0, ideal_memory_ns=0.0, hw=hw,
        )

    n = num_devices
    total_output_bytes = tensor_bytes * n
    bytes_per_device, ideal_network_ns = _ccl_bytes_and_time(
        hw, total_output_bytes, num_devices, factor=1.0
    )

    return RooflineEstimate(
        operation=operation,
        phase=phase,
        total_flops=0,
        total_bytes=bytes_per_device,
        ideal_compute_ns=0.0,
        ideal_memory_ns=0.0,
        ideal_network_ns=ideal_network_ns,
        hw=hw,
    )


def ring_shift_roofline(
    hw: HardwareSpec,
    tensor_bytes: int,
    num_devices: int,
    num_steps: Optional[int] = None,
    operation: str = "RingShift",
    phase: str = "ccl",
) -> "RooflineEstimate":
    """Ring shift: every device passes a tensor to its ring neighbour, num_steps times.

    Used by ring attention (context parallelism): the K/V shards travel around the CP ring.
    All devices send at once. On a ring, half of every tensor goes clockwise and half
    counter-clockwise, so both directions of the full-duplex links carry data (topology
    divisor 2, as for the other collectives).

    Formula: time = num_steps * tensor_bytes / (num_links * topology_divisor * link bandwidth)

    Args:
        hw: Hardware specification
        tensor_bytes: Bytes each device sends per step
        num_devices: Ring size
        num_steps: Number of shifts (default: num_devices - 1, a full rotation)
        operation: Name for the operation
        phase: Phase label

    Returns:
        RooflineEstimate with the time in ideal_network_ns
    """
    from .roofline import RooflineEstimate

    if num_devices <= 1:
        return RooflineEstimate(
            operation=operation, phase=phase,
            total_flops=0, total_bytes=0,
            ideal_compute_ns=0.0, ideal_memory_ns=0.0, hw=hw,
        )
    steps = num_devices - 1 if num_steps is None else num_steps
    topology_divisor = 2 if hw.topology == "ring" else 1
    total_bytes_moved = int(steps * tensor_bytes)
    ideal_network_ns = (
        total_bytes_moved / (hw.num_links * topology_divisor) / hw.eth_bw_gb_s_per_link
        if hw.eth_bw_gb_s_per_link > 0 else 0.0
    )
    return RooflineEstimate(
        operation=operation,
        phase=phase,
        total_flops=0,
        total_bytes=total_bytes_moved,
        ideal_compute_ns=0.0,
        ideal_memory_ns=0.0,
        hw=hw,
        ideal_network_ns=ideal_network_ns,
    )


def send_recv_roofline(
    hw: HardwareSpec,
    tensor_bytes: int,
    num_links: int,
    operation: str = "SendRecv",
    phase: str = "ccl",
) -> "RooflineEstimate":
    """Point-to-point transfer over a fixed number of links (pipeline stage hand-off).

    A pipeline hand-off moves the activation (forward) or its gradient (backward) across the
    cut between two neighbouring stages. Every chip pair on the cut carries part of it, so the
    transfer uses all links across the cut. Activations and gradients travel in opposite
    directions of the full-duplex links and do not share bandwidth.

    Formula: time = tensor_bytes / (num_links * link bandwidth)

    Args:
        hw: Hardware specification
        tensor_bytes: Bytes crossing the cut
        num_links: Links across the cut (per direction)
        operation: Name for the operation
        phase: Phase label

    Returns:
        RooflineEstimate with the time in ideal_network_ns
    """
    from .roofline import RooflineEstimate

    ideal_network_ns = (
        tensor_bytes / num_links / hw.eth_bw_gb_s_per_link
        if hw.eth_bw_gb_s_per_link > 0 and num_links > 0 else 0.0
    )
    return RooflineEstimate(
        operation=operation,
        phase=phase,
        total_flops=0,
        total_bytes=int(tensor_bytes),
        ideal_compute_ns=0.0,
        ideal_memory_ns=0.0,
        hw=hw,
        ideal_network_ns=ideal_network_ns,
    )
