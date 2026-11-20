# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Attention operations for roofline modeling.

This module provides MockHeadsCreationOp, MockHeadsFusionOp, and
MockScaledDotProductAttentionOp for roofline estimation of attention operations.
"""

from __future__ import annotations
import dataclasses
from typing import Optional, Tuple, TYPE_CHECKING

from ..mock_tensor import MockTensor
from ..hardware import DataType
from ..roofline import (
    elementwise_roofline,
    ring_shift_roofline,
    socket_ring_shift_roofline,
    attention_roofline,
    fused_attention_roofline,
    heads_creation_roofline,
    heads_fusion_roofline,
    grouped_heads_creation_roofline,
)
from .operation import (
    RooflineFunctionContext,
    RooflineFunction,
    create_grad_tensor,
    create_activation_tensor,
)

if TYPE_CHECKING:
    from ..roofline import RooflineContext


class MockHeadsCreationOp(RooflineFunction):
    """Roofline estimation for heads creation (split QKV).

    Splits combined QKV tensor into separate Q, K, V tensors with head dimension.
    Input: [B, 1, S, 3*H*d] -> Output: 3x [B, H, S, d]
    """

    @staticmethod
    def forward(
        ctx: RooflineFunctionContext,
        roofline_ctx: "RooflineContext",
        qkv: MockTensor,
        num_heads: int,
    ) -> Tuple[MockTensor, MockTensor, MockTensor]:
        """Forward pass: split QKV into separate tensors.

        Args:
            ctx: Function context
            roofline_ctx: Roofline context for estimates
            qkv: Combined QKV tensor [B, 1, S, 3*H*d]
            num_heads: Number of attention heads

        Returns:
            Tuple of (query, key, value) tensors, each [B, H, S, d]
        """
        ctx.save_for_backward(qkv, num_heads)

        # Parse dimensions from qkv shape [B, 1, S, 3*H*d]
        batch_size = qkv.shape[0]
        seq_len = qkv.shape[2]
        total_dim = qkv.shape[3]
        head_dim = total_dim // (3 * num_heads)

        estimate = heads_creation_roofline(
            roofline_ctx.hw,
            batch_size,
            seq_len,
            num_heads,
            head_dim,
            num_tensors=3,
            dtype=qkv.dtype,
            operation="HeadsCreation.forward",
            phase="forward",
        )
        roofline_ctx.add_perf_result(estimate)

        # Output shapes: [B, H, S, d]
        output_shape = (batch_size, num_heads, seq_len, head_dim)
        query = create_activation_tensor(output_shape, qkv.dtype, qkv.layout)
        key = create_activation_tensor(output_shape, qkv.dtype, qkv.layout)
        value = create_activation_tensor(output_shape, qkv.dtype, qkv.layout)

        return query, key, value

    @staticmethod
    def backward(
        ctx: RooflineFunctionContext,
        roofline_ctx: "RooflineContext",
        grad_query: MockTensor,
        grad_key: MockTensor,
        grad_value: MockTensor,
    ) -> Tuple[MockTensor]:
        """Backward pass: concatenate gradients back.

        Args:
            ctx: Function context with saved tensors
            roofline_ctx: Roofline context for estimates
            grad_query: Gradient for query
            grad_key: Gradient for key
            grad_value: Gradient for value

        Returns:
            (grad_qkv,) tuple
        """
        qkv, num_heads = ctx.saved_tensors

        batch_size = qkv.shape[0]
        seq_len = qkv.shape[2]
        total_dim = qkv.shape[3]
        head_dim = total_dim // (3 * num_heads)

        estimate = heads_creation_roofline(
            roofline_ctx.hw,
            batch_size,
            seq_len,
            num_heads,
            head_dim,
            num_tensors=3,
            dtype=qkv.dtype,
            operation="HeadsCreation.backward",
            phase="backward",
        )
        roofline_ctx.add_perf_result(estimate)

        grad_qkv = create_grad_tensor(qkv.shape, qkv.dtype, qkv.layout, name="grad_qkv")

        return (grad_qkv,)


class MockGroupedHeadsCreationOp(RooflineFunction):
    """Roofline estimation for grouped heads creation (split Q and KV for GQA).

    For Grouped Query Attention:
    - Q: [B, 1, S, E] -> [B, num_heads, S, head_dim]
    - KV: [B, 1, S, 2*num_groups*head_dim] -> K [B, num_groups, S, head_dim], V [B, num_groups, S, head_dim]
    """

    recompute = True

    @staticmethod
    def forward(
        ctx: RooflineFunctionContext,
        roofline_ctx: "RooflineContext",
        q: MockTensor,
        kv: MockTensor,
        num_heads: int,
        num_groups: int,
    ) -> Tuple[MockTensor, MockTensor, MockTensor]:
        """Forward pass: split Q and KV into separate head tensors.

        Args:
            ctx: Function context
            roofline_ctx: Roofline context for estimates
            q: Query tensor [B, 1, S, E] where E = num_heads * head_dim
            kv: Combined KV tensor [B, 1, S, 2*num_groups*head_dim]
            num_heads: Number of attention heads for Q
            num_groups: Number of KV groups

        Returns:
            Tuple of (query, key, value) tensors
            - query: [B, num_heads, S, head_dim]
            - key: [B, num_groups, S, head_dim]
            - value: [B, num_groups, S, head_dim]
        """
        ctx.save_for_backward(q, kv, num_heads, num_groups)

        # Parse dimensions from q shape [B, 1, S, E]
        batch_size = q.shape[0]
        seq_len = q.shape[2]
        embedding_dim = q.shape[3]
        head_dim = embedding_dim // num_heads

        estimate = grouped_heads_creation_roofline(
            roofline_ctx.hw,
            batch_size,
            seq_len,
            num_heads,
            num_groups,
            head_dim,
            dtype=q.dtype,
            operation="GroupedHeadsCreation.forward",
            phase="forward",
        )
        roofline_ctx.add_perf_result(estimate)

        # Output shapes
        q_shape = (batch_size, num_heads, seq_len, head_dim)
        kv_shape = (batch_size, num_groups, seq_len, head_dim)

        query = create_activation_tensor(q_shape, q.dtype, q.layout)
        key = create_activation_tensor(kv_shape, kv.dtype, kv.layout)
        value = create_activation_tensor(kv_shape, kv.dtype, kv.layout)

        return query, key, value

    @staticmethod
    def backward(
        ctx: RooflineFunctionContext,
        roofline_ctx: "RooflineContext",
        grad_query: MockTensor,
        grad_key: MockTensor,
        grad_value: MockTensor,
    ) -> Tuple[MockTensor, MockTensor, None, None]:
        """Backward pass: concatenate gradients back.

        Args:
            ctx: Function context with saved tensors
            roofline_ctx: Roofline context for estimates
            grad_query: Gradient for query
            grad_key: Gradient for key
            grad_value: Gradient for value

        Returns:
            (grad_q, grad_kv, None, None) tuple
        """
        q, kv, num_heads, num_groups = ctx.saved_tensors

        batch_size = q.shape[0]
        seq_len = q.shape[2]
        embedding_dim = q.shape[3]
        head_dim = embedding_dim // num_heads

        estimate = grouped_heads_creation_roofline(
            roofline_ctx.hw,
            batch_size,
            seq_len,
            num_heads,
            num_groups,
            head_dim,
            dtype=q.dtype,
            operation="GroupedHeadsCreation.backward",
            phase="backward",
        )
        roofline_ctx.add_perf_result(estimate)

        grad_q = create_grad_tensor(q.shape, q.dtype, q.layout, name="grad_q")
        grad_kv = create_grad_tensor(kv.shape, kv.dtype, kv.layout, name="grad_kv")

        return grad_q, grad_kv, None, None


class MockHeadsFusionOp(RooflineFunction):
    """Roofline estimation for heads fusion (merge heads back).

    Merges attention output heads back into single tensor.
    Input: [B, H, S, d] -> Output: [B, 1, S, H*d]
    """

    recompute = True

    @staticmethod
    def forward(
        ctx: RooflineFunctionContext,
        roofline_ctx: "RooflineContext",
        attention_out: MockTensor,
    ) -> MockTensor:
        """Forward pass: merge heads back.

        Args:
            ctx: Function context
            roofline_ctx: Roofline context for estimates
            attention_out: Attention output [B, H, S, d]

        Returns:
            Merged tensor [B, 1, S, H*d]
        """
        ctx.save_for_backward(attention_out)

        # Parse dimensions [B, H, S, d]
        batch_size = attention_out.shape[0]
        num_heads = attention_out.shape[1]
        seq_len = attention_out.shape[2]
        head_dim = attention_out.shape[3]

        estimate = heads_fusion_roofline(
            roofline_ctx.hw,
            batch_size,
            seq_len,
            num_heads,
            head_dim,
            dtype=attention_out.dtype,
            operation="HeadsFusion.forward",
            phase="forward",
        )
        roofline_ctx.add_perf_result(estimate)

        # Output shape: [B, 1, S, H*d]
        output_shape = (batch_size, 1, seq_len, num_heads * head_dim)
        return create_activation_tensor(
            output_shape, attention_out.dtype, attention_out.layout
        )

    @staticmethod
    def backward(
        ctx: RooflineFunctionContext,
        roofline_ctx: "RooflineContext",
        grad_output: MockTensor,
    ) -> Tuple[MockTensor]:
        """Backward pass: split gradient back into heads.

        Args:
            ctx: Function context with saved tensors
            roofline_ctx: Roofline context for estimates
            grad_output: Gradient tensor [B, 1, S, H*d]

        Returns:
            (grad_attention_out,) tuple with shape [B, H, S, d]
        """
        (attention_out,) = ctx.saved_tensors

        batch_size = attention_out.shape[0]
        num_heads = attention_out.shape[1]
        seq_len = attention_out.shape[2]
        head_dim = attention_out.shape[3]

        estimate = heads_fusion_roofline(
            roofline_ctx.hw,
            batch_size,
            seq_len,
            num_heads,
            head_dim,
            dtype=attention_out.dtype,
            operation="HeadsFusion.backward",
            phase="backward",
        )
        roofline_ctx.add_perf_result(estimate)

        grad_attention_out = create_grad_tensor(
            attention_out.shape,
            attention_out.dtype,
            attention_out.layout,
            name="grad_attention_out",
        )

        return (grad_attention_out,)


class MockScaledDotProductAttentionOp(RooflineFunction):
    """Roofline estimation for scaled dot-product attention.

    SDPA: softmax(Q @ K^T / sqrt(d_k)) @ V

    This operation combines:
    1. Q @ K^T matmul
    2. Scale by 1/sqrt(d_k)
    3. Softmax
    4. Attn @ V matmul
    """

    @staticmethod
    def forward(
        ctx: RooflineFunctionContext,
        roofline_ctx: "RooflineContext",
        query: MockTensor,
        key: MockTensor,
        value: MockTensor,
        mask: Optional[MockTensor] = None,
    ) -> MockTensor:
        """Forward pass: compute scaled dot-product attention.

        Args:
            ctx: Function context
            roofline_ctx: Roofline context for estimates
            query: Query tensor [B, H, S, d]
            key: Key tensor [B, H, S, d] or [B, G, S, d] for GQA
            value: Value tensor [B, H, S, d] or [B, G, S, d] for GQA
            mask: Optional attention mask

        Returns:
            Attention output tensor [B, H, S, d]
        """
        # Parse dimensions [B, H, S, d]
        batch_size = query.shape[0]
        num_heads = query.shape[1]
        seq_len = query.shape[2]
        head_dim = query.shape[3]

        # Create attention_weights tensor (B, H, S, S) - this is the softmax output
        # that must be saved for backward pass (see scaled_dot_product_attention.cpp)
        # This is a significant memory allocation, especially for long sequences!
        attention_weights_shape = (batch_size, num_heads, seq_len, seq_len)
        attention_weights = create_activation_tensor(
            attention_weights_shape,
            query.dtype,
            query.layout,
            name="attention_weights",
        )

        # Save tensors for backward - must include attention_weights
        ctx.save_for_backward(query, key, value, mask, attention_weights)

        estimates = attention_roofline(
            roofline_ctx.hw,
            batch_size,
            num_heads,
            seq_len,
            head_dim,
            dtype=query.dtype,
            operation="SDPA",
            phase="forward",
        )
        for estimate in estimates:
            roofline_ctx.add_perf_result(estimate)

        # Output shape same as input: [B, H, S, d]
        return create_activation_tensor(query.shape, query.dtype, query.layout)

    @staticmethod
    def backward(
        ctx: RooflineFunctionContext,
        roofline_ctx: "RooflineContext",
        grad_output: MockTensor,
    ) -> Tuple[MockTensor, MockTensor, MockTensor, None]:
        """Backward pass: compute gradients for Q, K, V.

        Args:
            ctx: Function context with saved tensors
            roofline_ctx: Roofline context for estimates
            grad_output: Gradient tensor [B, H, S, d]

        Returns:
            (grad_query, grad_key, grad_value, None) tuple
            (None for mask which doesn't need gradients)

        Note:
            attention_weights is saved for backward but is not an input,
            so we don't return a gradient for it.
        """
        query, key, value, mask, attention_weights = ctx.saved_tensors

        batch_size = query.shape[0]
        num_heads = query.shape[1]
        seq_len = query.shape[2]
        head_dim = query.shape[3]

        estimates = attention_roofline(
            roofline_ctx.hw,
            batch_size,
            num_heads,
            seq_len,
            head_dim,
            dtype=query.dtype,
            operation="SDPA",
            phase="backward",
        )
        for estimate in estimates:
            roofline_ctx.add_perf_result(estimate)

        grad_query = create_grad_tensor(
            query.shape, query.dtype, query.layout, name="grad_query"
        )
        grad_key = create_grad_tensor(key.shape, key.dtype, key.layout, name="grad_key")
        grad_value = create_grad_tensor(
            value.shape, value.dtype, value.layout, name="grad_value"
        )

        # No gradient for mask (it's optional and doesn't require grad)
        return grad_query, grad_key, grad_value, None


class MockScaledDotProductAttentionFusedOp(RooflineFunction):
    """Roofline estimation for fused scaled dot-product attention (Flash Attention style).

    Fused SDPA: softmax(Q @ K^T / sqrt(d_k)) @ V computed in a single kernel.

    Key benefits over unfused SDPA:
    1. Memory: O(S) instead of O(S²) - no need to materialize full attention matrix
    2. Speed: Single fused kernel with better memory access patterns
    3. Backward: Uses recomputation instead of storing attention weights

    This models hardware-optimized attention like Flash Attention, where the
    attention matrix is computed and consumed in tiles without full materialization.
    """

    recompute = True

    @staticmethod
    def forward(
        ctx: RooflineFunctionContext,
        roofline_ctx: "RooflineContext",
        query: MockTensor,
        key: MockTensor,
        value: MockTensor,
        mask: Optional[MockTensor] = None,
    ) -> MockTensor:
        """Forward pass: compute fused scaled dot-product attention.

        Args:
            ctx: Function context
            roofline_ctx: Roofline context for estimates
            query: Query tensor [B, H, S, d]
            key: Key tensor [B, H, S, d] or [B, G, S, d] for GQA
            value: Value tensor [B, H, S, d] or [B, G, S, d] for GQA
            mask: Optional attention mask (not used in fused kernel)

        Returns:
            Attention output tensor [B, H, S, d]
        """
        # Parse dimensions [B, H, S, d]
        batch_size = query.shape[0]
        num_q_heads = query.shape[1]
        num_kv_heads = key.shape[1]
        seq_len = query.shape[2]
        head_dim = query.shape[3]

        #   - max values per attention row: [B, H, S]
        #   - sum(exp-max) per row: [B, H, S]
        max_values = create_activation_tensor(
            (batch_size, num_q_heads, seq_len),
            query.dtype,
            query.layout,
            name="attention_max_values",
        )
        sum_values = create_activation_tensor(
            (batch_size, num_q_heads, seq_len),
            query.dtype,
            query.layout,
            name="attention_sum_values",
        )

        # Save for backward: inputs + intermediate values
        # Backward will recompute attention on-the-fly using these saved values
        ctx.save_for_backward(query, key, value, mask, max_values, sum_values)

        estimate = fused_attention_roofline(
            roofline_ctx.hw,
            batch_size,
            num_q_heads,
            num_kv_heads,
            seq_len,
            head_dim,
            dtype=query.dtype,
            operation="FusedSDPA",
            phase="forward",
            block=roofline_ctx.attn_block,
        )
        roofline_ctx.add_perf_result(estimate)

        # Output shape same as input: [B, H, S, d]
        return create_activation_tensor(query.shape, query.dtype, query.layout)

    @staticmethod
    def backward(
        ctx: RooflineFunctionContext,
        roofline_ctx: "RooflineContext",
        grad_output: MockTensor,
    ) -> Tuple[MockTensor, MockTensor, MockTensor, None, None, None]:
        """Backward pass: compute gradients for Q, K, V using recomputation.

        Fused backward recomputes attention weights on-the-fly instead of
        reading them from memory. This trades compute for memory bandwidth.
        Uses saved max and sum values to efficiently recompute softmax.

        Args:
            ctx: Function context with saved tensors
            roofline_ctx: Roofline context for estimates
            grad_output: Gradient tensor [B, H, S, d]

        Returns:
            (grad_query, grad_key, grad_value, None, None, None) tuple
            (None for mask, max_values, and sum_values which don't need gradients)
        """
        query, key, value, mask, max_values, sum_values = ctx.saved_tensors

        batch_size = query.shape[0]
        num_q_heads = query.shape[1]
        num_kv_heads = key.shape[1]
        seq_len = query.shape[2]
        head_dim = query.shape[3]

        estimate = fused_attention_roofline(
            roofline_ctx.hw,
            batch_size,
            num_q_heads,
            num_kv_heads,
            seq_len,
            head_dim,
            dtype=query.dtype,
            operation="FusedSDPA",
            phase="backward",
            block=roofline_ctx.attn_block,
        )
        roofline_ctx.add_perf_result(estimate)

        grad_query = create_grad_tensor(
            query.shape, query.dtype, query.layout, name="grad_query"
        )
        grad_key = create_grad_tensor(key.shape, key.dtype, key.layout, name="grad_key")
        grad_value = create_grad_tensor(
            value.shape, value.dtype, value.layout, name="grad_value"
        )

        # No gradients for mask, max_values, or sum_values (intermediate values)
        return grad_query, grad_key, grad_value, None, None, None



def _ring_attention_estimates(
    roofline_ctx: "RooflineContext",
    query: MockTensor,
    key: MockTensor,
    cp_size: int,
    phase: str,
    accumulator_elements: int,
    kv_bytes: int,
) -> None:
    """Add one ring-attention pass to the context: the target design, or tt-train's current one
    (roofline_ctx.cp_mode).

    Target, three parts:
    - attention: fused attention over the full sequence, divided by cp_size (balanced split)
    - merge: one read + one write of the fp32 accumulators per ring step (fused into the kernel)
    - shift: the K/V shards visit every device (cp - 1 hops); backward also moves the fp32
      dK/dV accumulators, which need cp hops to return to the device that owns K/V

    The three are one op (their own fusion): the shifts and merges run under the attention.
    """
    if roofline_ctx.cp_mode == "current":
        _current_ring_attention(roofline_ctx, query, key, cp_size, phase)
        return
    with roofline_ctx.fused(roofline_ctx.new_fusion("RingAttention")):
        _ring_attention_parts(roofline_ctx, query, key, cp_size, phase, accumulator_elements, kv_bytes)


def _current_ring_attention(roofline_ctx, query, key, cp_size, phase) -> None:
    """tt-train's ring attention today (ttml/ops/distributed/ring_attention_sdpa.cpp), one op per
    ttnn call, on the device that sets the pace.

    - Contiguous split: device d holds tokens [d, d + 1) * S / cp. At step s it attends to the K/V
      shard of device (d + s) mod cp: the diagonal shard causally, earlier shards in full, later
      ones not at all (ring_sdpa_utils.hpp). The last device works at every step: half a block, then
      cp - 1 full blocks, against cp / 2 blocks per device with a balanced split.
    - Forward, per step: reset the statistics tile, ring_sdpa_fw, slice the log-sum-exp, 12 fp32
      ops on it (log-add-exp and the two weights), cast the step output to fp32, rescale and add
      into the fp32 accumulator (3 passes); then shift K, then V.
    - Backward (ring_sdpa_bw, steps in reverse): zero the three step buffers, the Q and KV kernels,
      cast each step gradient to fp32 and add it into its accumulator; then shift K, V (bf16) and
      the dK, dV accumulators (fp32).
    - Shifts: socket_ring_shift_roofline (one link, one direction, two phases).
    - Each ttnn call is its own op: nothing overlaps. Identical calls (every step but the diagonal
      one, the three gradients' casts and adds) are added once, scaled by their count.
    """
    hw = roofline_ctx.hw
    batch_size, num_q_heads, local_seq, head_dim = query.shape
    num_kv_heads = key.shape[1]
    f32 = DataType.FLOAT32
    stat = batch_size * num_q_heads * local_seq * 32  # fp32 statistics, one 32-wide tile per 32 rows
    q_elems = batch_size * num_q_heads * local_seq * head_dim
    kv_elems = batch_size * num_kv_heads * local_seq * head_dim
    k_bytes = kv_elems * query.dtype.value  # one of K, V
    steps, shifts = cp_size, cp_size - 1

    def op(estimate, count=1) -> None:
        """`count` identical ops: their sum of per-op max is count times one."""
        if count == 0:
            return
        if count > 1:
            estimate = dataclasses.replace(
                estimate, total_flops=estimate.total_flops * count, total_bytes=estimate.total_bytes * count,
                ideal_compute_ns=estimate.ideal_compute_ns * count, ideal_memory_ns=estimate.ideal_memory_ns * count,
                ideal_network_ns=estimate.ideal_network_ns * count)
        estimate.fusion = roofline_ctx.new_fusion(estimate.operation)
        roofline_ctx.add_perf_result(estimate)

    def eltwise(name, elements, inputs, count=1, sfpu=0.0, dtype=f32, out_dtype=None) -> None:
        op(elementwise_roofline(hw, elements, num_inputs=inputs, sfpu_ops_per_element=sfpu, dtype=dtype,
                                out_dtype=out_dtype, operation=f"RingAttention.{name}", phase=phase), count)

    def attention(causal, count) -> None:
        op(fused_attention_roofline(
            hw, batch_size, num_q_heads, num_kv_heads, local_seq, head_dim, dtype=query.dtype,
            operation="RingAttention.sdpa", phase=phase, block=roofline_ctx.attn_block, causal=causal,
        ), count)

    def shift(name, nbytes) -> None:
        op(socket_ring_shift_roofline(roofline_ctx.hw_for("cp"), nbytes, operation=f"RingAttention.{name}",
                                      phase=phase), shifts)

    attention(causal=True, count=1)  # the diagonal shard
    attention(causal=False, count=steps - 1)
    shift("shift_k", k_bytes)
    shift("shift_v", k_bytes)
    if phase == "forward":
        eltwise("init_accumulator", q_elems, 0)
        eltwise("reset_stats", stat, 1, steps)  # ttnn::copy from a pre-filled tensor: read + write
        eltwise("slice_lse", stat, 1, steps)
        eltwise("merge_lse_binary", stat, 2, 7 * steps, sfpu=1)  # max, 4 sub, 2 add
        eltwise("merge_lse_unary", stat, 1, 5 * steps, sfpu=1)  # 4 exp, log
        eltwise("cast_out", q_elems, 1, steps, dtype=query.dtype, out_dtype=f32)
        eltwise("rescale", q_elems, 1, 2 * steps, sfpu=1)  # accumulator and step output
        eltwise("add_out", q_elems, 2, steps, sfpu=1)
        eltwise("cast_result", q_elems, 1, out_dtype=query.dtype)
        return
    shift("shift_dk", 2 * k_bytes)  # fp32 accumulators: twice the bytes of K in bf16
    shift("shift_dv", 2 * k_bytes)
    eltwise("init_accumulators", q_elems + 2 * kv_elems, 0)
    eltwise("zero_step_buffers", q_elems + 2 * kv_elems, 1, steps, dtype=query.dtype)  # ttnn::copy of zeros
    eltwise("cast_grads", q_elems + 2 * kv_elems, 1, steps, dtype=query.dtype, out_dtype=f32)
    eltwise("add_grads", q_elems + 2 * kv_elems, 2, steps, sfpu=1)
    eltwise("cast_grad_results", q_elems + 2 * kv_elems, 1, out_dtype=query.dtype)


def _ring_attention_parts(roofline_ctx, query, key, cp_size, phase, accumulator_elements, kv_bytes) -> None:
    batch_size, num_q_heads, local_seq, head_dim = query.shape
    full = fused_attention_roofline(
        roofline_ctx.hw, batch_size, num_q_heads, key.shape[1], local_seq * cp_size, head_dim,
        dtype=query.dtype, operation="RingAttention", phase=phase,
        block=roofline_ctx.attn_block,
    )
    roofline_ctx.add_perf_result(dataclasses.replace(
        full,
        total_flops=full.total_flops // cp_size,
        total_bytes=full.total_bytes // cp_size,
        ideal_compute_ns=full.ideal_compute_ns / cp_size,
        ideal_memory_ns=full.ideal_memory_ns / cp_size,
    ))
    roofline_ctx.add_perf_result(elementwise_roofline(
        roofline_ctx.hw, cp_size * accumulator_elements, num_inputs=1, sfpu_ops_per_element=0.0,
        dtype=DataType.FLOAT32, operation="RingAttention.merge", phase=phase,
    ))
    roofline_ctx.add_perf_result(ring_shift_roofline(
        roofline_ctx.hw_for("cp"), kv_bytes, cp_size, operation="RingAttention.shift_kv", phase=phase,
    ))
    if phase == "backward":
        roofline_ctx.add_perf_result(ring_shift_roofline(
            roofline_ctx.hw_for("cp"), 2 * kv_bytes, cp_size, operation="RingAttention.shift_dkv", phase=phase,
            num_steps=cp_size,  # dK, dV accumulators in fp32: twice the bytes of K, V in bf16
        ))


class MockRingAttentionOp(RooflineFunction):
    """Roofline estimation for ring attention (context parallelism, causal).

    The sequence is split across cp_size devices; each device holds the Q, K, V of its
    local tokens and K/V shards travel around the CP ring, one step per shard.
    roofline_ctx.cp_mode picks the model: "current" is tt-train's ring_attention_sdpa today
    (_current_ring_attention); "target", the default, is the planned implementation:

    - Load-balanced split: device i holds sequence chunks i and 2*cp_size - 1 - i, so every
      device does the same causal work: attention FLOPs and DRAM traffic are those of fused
      attention over the full sequence divided by cp_size.
    - Ring shifts: forward moves K, V (bf16); backward moves K, V (bf16) and the dK, dV
      accumulators (fp32). Half of each tensor goes each way around the ring.
    - The merge of per-step results is fused into the attention kernel: one read and one
      write of the fp32 accumulator per step (output forward; dQ, dK, dV backward).
    """

    recompute = True

    @staticmethod
    def forward(
        ctx: RooflineFunctionContext,
        roofline_ctx: "RooflineContext",
        query: MockTensor,
        key: MockTensor,
        value: MockTensor,
        cp_size: int,
    ) -> MockTensor:
        """Forward pass.

        Args:
            query: Local query shard [B, H, S/cp_size, d]
            key, value: Local key/value shards [B, G, S/cp_size, d]
            cp_size: Context-parallel degree

        Returns:
            Attention output for the local queries [B, H, S/cp_size, d]
        """
        stats = create_activation_tensor(
            query.shape[:3], query.dtype, query.layout, name="ring_attention_lse",
        )
        ctx.save_for_backward(query, key, value, stats)
        ctx._cp_size = cp_size
        _ring_attention_estimates(
            roofline_ctx, query, key, cp_size, "forward",
            accumulator_elements=query.logical_volume(),
            kv_bytes=key.bytes() + value.bytes(),
        )
        return create_activation_tensor(query.shape, query.dtype, query.layout)

    @staticmethod
    def backward(
        ctx: RooflineFunctionContext,
        roofline_ctx: "RooflineContext",
        grad_output: MockTensor,
    ) -> Tuple[MockTensor, MockTensor, MockTensor, None]:
        """Backward pass: dQ, dK, dV with the K/V shards and dK, dV accumulators on the ring."""
        query, key, value, stats = ctx.saved_tensors
        _ring_attention_estimates(
            roofline_ctx, query, key, ctx._cp_size, "backward",
            accumulator_elements=query.logical_volume() + key.logical_volume() + value.logical_volume(),
            kv_bytes=key.bytes() + value.bytes(),
        )
        grad_query = create_grad_tensor(query.shape, query.dtype, query.layout, name="grad_query")
        grad_key = create_grad_tensor(key.shape, key.dtype, key.layout, name="grad_key")
        grad_value = create_grad_tensor(value.shape, value.dtype, value.layout, name="grad_value")
        return grad_query, grad_key, grad_value, None
