# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Roofline performance models for CCL (Collective Communication Library) operations.

Models ring/linear topology collective communication costs using per-link
Ethernet bandwidth. Ring topology has 2x better effective bandwidth than linear.

Each collective also reads its input from DRAM and writes its output there (DRAM time), and pays a
fixed per-call latency on the fabric, fitted to measured ttnn collectives in tt-metal's
tech_reports/CCLs/CCL_bandwidth.md ("Fitted models"): line a + b * (N - 1), ring a + b * N / 2.
"""

from __future__ import annotations
from typing import Optional, TYPE_CHECKING

from ..hardware import HardwareSpec, DataType

if TYPE_CHECKING:
    from .roofline import RooflineEstimate


# Per-call latency (us): arch -> collective -> topology -> (a, b). An all-reduce is a
# reduce-scatter followed by an all-gather.
CCL_LATENCY_US = {
    "wormhole": {
        "all_gather": {"linear": (4.5, 1.9), "ring": (4.5, 2.5)},
        "reduce_scatter": {"linear": (3.2, 4.0), "ring": (3.2, 3.1)},
        "all_reduce": {"linear": (9.1, 5.8), "ring": (9.1, 5.3)},
    },
    "blackhole": {
        "all_gather": {"linear": (3.7, 1.3), "ring": (3.7, 1.6)},
        "reduce_scatter": {"linear": (1.3, 3.0), "ring": (1.3, 2.1)},
        "all_reduce": {"linear": (4.4, 4.4), "ring": (4.4, 3.9)},
    },
}


def ccl_latency_ns(hw: HardwareSpec, collective: str, num_devices: int) -> float:
    """Fixed per-call latency of one collective on num_devices devices (0 for one device)."""
    if num_devices <= 1:
        return 0.0
    a, b = CCL_LATENCY_US[hw.arch][collective][hw.topology]
    hops = num_devices / 2 if hw.topology == "ring" else num_devices - 1
    return (a + b * hops) * 1e3


def _ccl_estimate(
    hw: HardwareSpec, operation: str, phase: str, link_bytes: int, network_ns: float,
    dram_bytes: float, collective: str, num_devices: int,
) -> "RooflineEstimate":
    """Collective estimate: link time plus per-call latency on the fabric, DRAM read + write."""
    from .roofline import RooflineEstimate

    return RooflineEstimate(
        operation=operation,
        phase=phase,
        total_flops=0,
        total_bytes=link_bytes,
        ideal_compute_ns=0.0,
        ideal_memory_ns=dram_bytes / hw.dram_bw_gb_s,
        ideal_network_ns=network_ns + ccl_latency_ns(hw, collective, num_devices),
        hw=hw,
    )


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
    # reduce-scatter: read the input, write the shard; all-gather: read the shard, write the output
    n = num_devices
    dram_bytes = 2 * tensor_bytes * (1 + 1 / n)
    return _ccl_estimate(hw, operation, phase, bytes_per_device, ideal_network_ns, dram_bytes, "all_reduce", n)


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
    # read the input, write the shard
    dram_bytes = tensor_bytes * (1 + 1 / num_devices)
    return _ccl_estimate(
        hw, operation, phase, bytes_per_device, ideal_network_ns, dram_bytes, "reduce_scatter", num_devices
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
    # read the shard, write the gathered output
    dram_bytes = tensor_bytes + total_output_bytes
    return _ccl_estimate(hw, operation, phase, bytes_per_device, ideal_network_ns, dram_bytes, "all_gather", n)



def _link_estimate(
    hw: HardwareSpec, operation: str, phase: str, total_bytes: int, num_links: int, directions: int
) -> "RooflineEstimate":
    """Fabric-only estimate: total_bytes over num_links links, using `directions` link directions."""
    from .roofline import RooflineEstimate

    bandwidth = hw.eth_bw_gb_s_per_link * num_links * directions
    return RooflineEstimate(
        operation=operation,
        phase=phase,
        total_flops=0,
        total_bytes=int(total_bytes),
        ideal_compute_ns=0.0,
        ideal_memory_ns=0.0,
        hw=hw,
        ideal_network_ns=total_bytes / bandwidth if bandwidth > 0 else 0.0,
    )


def ring_shift_roofline(
    hw: HardwareSpec,
    tensor_bytes: int,
    num_devices: int,
    operation: str = "RingShift",
    phase: str = "ccl",
    num_steps: Optional[int] = None,
) -> "RooflineEstimate":
    """Ring shifts: every device passes tensor_bytes to its neighbour num_steps times
    (default num_devices - 1: every shard visits every device).

    Used by ring attention (context parallelism): the K/V shards travel around the CP ring.
    All devices send at once; on a ring half of every tensor goes each way, so both directions
    of the full-duplex links carry data (topology divisor 2, as for the other collectives).

    Formula: time = num_steps * tensor_bytes / (num_links * topology_divisor * link bandwidth)
             + num_steps * per-call latency of a two-device transfer
    DRAM: every step reads the outgoing shard and writes the incoming one.
    """
    import dataclasses

    directions = 2 if hw.topology == "ring" else 1
    steps = max(num_devices - 1, 0) if num_steps is None else (num_steps if num_devices > 1 else 0)
    est = _link_estimate(hw, operation, phase, steps * tensor_bytes, hw.num_links, directions)
    return dataclasses.replace(
        est,
        ideal_memory_ns=2 * steps * tensor_bytes / hw.dram_bw_gb_s,
        ideal_network_ns=est.ideal_network_ns + steps * ccl_latency_ns(hw, "all_gather", 2),
    )


def socket_ring_shift_roofline(
    hw: HardwareSpec,
    tensor_bytes: int,
    operation: str = "RingShift",
    phase: str = "ccl",
) -> "RooflineEstimate":
    """One ring shift as tt-train does it today (ttnn_fixed::distributed::ring_shift): every device
    sends tensor_bytes to its neighbour through a fabric socket on one worker core, in two phases
    (even ranks send, then odd ranks), so one link in one direction carries each tensor, twice in a row.
    The host synchronizes the device before and after; that wait is not part of a roofline.

    Formula: time = 2 * tensor_bytes / link bandwidth + 2 * per-call latency of a two-device transfer
    DRAM: every device reads its outgoing shard and writes the incoming one.
    """
    import dataclasses

    est = _link_estimate(hw, operation, phase, 2 * tensor_bytes, 1, 1)
    return dataclasses.replace(
        est,
        ideal_memory_ns=2 * tensor_bytes / hw.dram_bw_gb_s,
        ideal_network_ns=est.ideal_network_ns + 2 * ccl_latency_ns(hw, "all_gather", 2),
    )


def send_recv_roofline(
    hw: HardwareSpec,
    tensor_bytes: int,
    num_links: int,
    operation: str = "SendRecv",
    phase: str = "ccl",
) -> "RooflineEstimate":
    """Point-to-point transfer over a fixed number of links (pipeline stage hand-off).

    A hand-off moves the activation (forward) or its gradient (backward) across the cut between
    two neighbouring stages, using every link across the cut. Activations and gradients use
    opposite directions of the full-duplex links and do not share bandwidth.

    Formula: time = tensor_bytes / (num_links * link bandwidth) + per-call latency of a two-device
    transfer
    """
    import dataclasses

    est = _link_estimate(hw, operation, phase, tensor_bytes, num_links, 1)
    return dataclasses.replace(est, ideal_network_ns=est.ideal_network_ns + ccl_latency_ns(hw, "all_gather", 2))
