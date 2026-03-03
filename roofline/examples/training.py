#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Transformer model roofline analysis example.

This script demonstrates roofline estimation for transformer models (GPT and Llama),
providing performance analysis for forward pass, backward pass,
and full training iteration including optimizer step.

Supported Models:
    GPT Models:  nanogpt-char, nanogpt-bpe, gpt2-small, gpt2-medium, gpt2-large
    Llama Models: nanollama, tinyllama, llama-1b, llama-8b, llama-70b, llama-405b

Run from tt-train directory:
    # GPT models
    python3 -m roofline.examples.nanogpt                             # Default: nanogpt-char
    python3 -m roofline.examples.nanogpt --model nanogpt-char -b 64 -s 256
    python3 -m roofline.examples.nanogpt --model gpt2-small -b 4 -s 1024

    # Llama models
    python3 -m roofline.examples.nanogpt --model tinyllama -b 1 -s 2048
    python3 -m roofline.examples.nanogpt --model nanollama -b 64 -s 256
    python3 -m roofline.examples.nanogpt --model llama-1b -b 1 -s 4096
    python3 -m roofline.examples.nanogpt --model llama-8b -b 1 -s 8192
    python3 -m roofline.examples.nanogpt --model llama-70b -b 1 -s 2048
    python3 -m roofline.examples.nanogpt --model llama-405b -b 1 -s 1024

    # Load from tt-train training config (infers model arch + training hyperparams)
    python3 -m roofline.examples.nanogpt --config path/to/training_config.yaml
    python3 -m roofline.examples.nanogpt --config path/to/training_config.yaml -b 32 -s 128

    # List all available models
    python3 -m roofline.examples.nanogpt --list
"""

import os
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict


class ModelType(Enum):
    """Type of transformer model."""

    GPT = "gpt"
    LLAMA = "llama"


# Preset model configurations
MODEL_PRESETS: Dict[str, dict] = {
    # ==================== GPT Models ====================
    # Char tokenizer models (smaller vocab)
    "nanogpt-char": {
        "model_type": ModelType.GPT,
        "vocab_size": 96,  # ~65 chars + special tokens, rounded up to multiple of 32
        "block_size": 256,
        "n_embd": 384,
        "n_layer": 6,
        "n_head": 6,
        "dropout": 0.2,
        # bias: uses MockNanoGPTConfig default (True)
        "description": "NanoGPT with char tokenizer (Shakespeare)",
    },
    # BPE tokenizer models (GPT-2 vocab)
    "nanogpt-bpe": {
        "model_type": ModelType.GPT,
        "vocab_size": 50257,  # GPT-2 BPE vocab size
        "block_size": 256,
        "n_embd": 384,
        "n_layer": 6,
        "n_head": 6,
        "dropout": 0.2,
        # bias: uses MockNanoGPTConfig default (True)
        "description": "NanoGPT with BPE tokenizer",
    },
    "gpt2-small": {
        "model_type": ModelType.GPT,
        "vocab_size": 50257,
        "block_size": 1024,
        "n_embd": 768,
        "n_layer": 12,
        "n_head": 12,
        "dropout": 0.2,
        # bias: uses MockNanoGPTConfig default (True)
        "description": "GPT-2 Small (124M params)",
    },
    "gpt2-medium": {
        "model_type": ModelType.GPT,
        "vocab_size": 50257,
        "block_size": 1024,
        "n_embd": 1024,
        "n_layer": 24,
        "n_head": 16,
        "dropout": 0.1,
        # bias: uses MockNanoGPTConfig default (True)
        "description": "GPT-2 Medium (355M params)",
    },
    "gpt2-large": {
        "model_type": ModelType.GPT,
        "vocab_size": 50257,
        "block_size": 1024,
        "n_embd": 1280,
        "n_layer": 36,
        "n_head": 20,
        "dropout": 0.1,
        # bias: uses MockNanoGPTConfig default (True)
        "description": "GPT-2 Large (774M params)",
    },
    # ==================== Llama Models ====================
    "nanollama": {
        "model_type": ModelType.LLAMA,
        "vocab_size": 32000,
        "max_sequence_length": 256,
        "embedding_dim": 384,
        "num_heads": 6,
        "num_groups": 3,
        "num_blocks": 6,
        "dropout": 0.0,
        "theta": 500000.0,
        "weight_tying": False,
        "description": "Nano Llama (Shakespeare, ~10M params)",
    },
    "nanollama-char": {
        "model_type": ModelType.LLAMA,
        "vocab_size": 96,
        "max_sequence_length": 256,
        "embedding_dim": 384,
        "num_heads": 6,
        "num_groups": 3,
        "num_blocks": 6,
        "dropout": 0.0,
        "theta": 500000.0,
        "weight_tying": False,
        "description": "Nano Llama (char tokenizer, ~10M params)",
    },
    "tinyllama": {
        "model_type": ModelType.LLAMA,
        "vocab_size": 32000,
        # "max_sequence_length": 2048,
        "max_sequence_length": 131072,
        "embedding_dim": 2048,
        "intermediate_dim": 5632,
        "num_heads": 32,
        "num_groups": 4,
        "num_blocks": 22,
        "dropout": 0.0,
        "theta": 10000.0,
        "weight_tying": False,
        "description": "TinyLlama 1.1B (1.1B params)",
    },
    "tinyllama-char": {
        "model_type": ModelType.LLAMA,
        "vocab_size": 96,
        "max_sequence_length": 2048,
        "embedding_dim": 2048,
        "num_heads": 32,
        "num_groups": 4,
        "num_blocks": 22,
        "dropout": 0.0,
        "theta": 10000.0,
        "weight_tying": False,
        "description": "TinyLlama 1.1B (char tokenizer, 0.96B params)",
    },
    "llama-1b": {
        "model_type": ModelType.LLAMA,
        "vocab_size": 128256,
        "max_sequence_length": 131072,
        "embedding_dim": 2048,
        "intermediate_dim": 8192,
        "num_heads": 32,
        "num_groups": 8,  # GQA with 8 KV heads (32/8 = 4 queries per KV)
        "num_blocks": 16,
        "dropout": 0.0,
        "theta": 500000.0,
        "weight_tying": False,
        "description": "Llama 3.2 1B (128K context)",
    },
    "llama-8b": {
        "model_type": ModelType.LLAMA,
        "vocab_size": 128256,
        "max_sequence_length": 131072,
        "embedding_dim": 4096,
        "intermediate_dim": 14336,
        "num_heads": 32,
        "num_groups": 8,  # GQA with 8 KV heads (32/8 = 4 queries per KV)
        "num_blocks": 32,
        "dropout": 0.0,
        "theta": 500000.0,
        "weight_tying": False,
        "description": "Llama 3.1 8B (128K context)",
    },
    "llama-70b": {
        "model_type": ModelType.LLAMA,
        "vocab_size": 128256,
        "max_sequence_length": 131072,
        "embedding_dim": 8192,
        "intermediate_dim": 28672,
        "num_heads": 64,
        "num_groups": 8,  # GQA with 8 KV heads (64/8 = 8 queries per KV)
        "num_blocks": 80,
        "dropout": 0.0,
        "theta": 500000.0,
        "weight_tying": False,
        "description": "Llama 3.1 70B (128K context)",
    },
    "llama-405b": {
        "model_type": ModelType.LLAMA,
        "vocab_size": 128256,
        "max_sequence_length": 131072,
        "embedding_dim": 16384,
        "intermediate_dim": 53248,
        "num_heads": 128,
        "num_groups": 8,  # GQA with 16 KV heads (128/16 = 8 queries per KV, using 8 groups for consistency)
        "num_blocks": 126,
        "dropout": 0.0,
        "theta": 500000.0,
        "weight_tying": False,
        "description": "Llama 3.1 405B (128K context)",
    },
}


def _resolve_model_config_path(model_config_rel: str, training_config_path: str) -> str:
    """Resolve a model config path that is relative to the tt-train project root.

    tt-train config files store the model_config path relative to the project root
    (e.g. "configs/model_configs/nanollama3.yaml").  We walk up from the training
    config file until we find a directory that contains the referenced path.
    """
    if os.path.isabs(model_config_rel) and os.path.exists(model_config_rel):
        return model_config_rel

    candidate = os.path.dirname(os.path.abspath(training_config_path))
    for _ in range(6):
        path = os.path.normpath(os.path.join(candidate, model_config_rel))
        if os.path.exists(path):
            return path
        parent = os.path.dirname(candidate)
        if parent == candidate:
            break
        candidate = parent

    raise FileNotFoundError(
        f"Could not resolve model config path '{model_config_rel}' "
        f"relative to '{training_config_path}'. "
        "Make sure the file exists and the path is correct."
    )


def load_training_config(config_path: str) -> dict:
    """Load model and training parameters from a tt-train training config YAML.

    Reads both the training config and the model config it references, returning
    a preset dict compatible with MODEL_PRESETS plus a ``_training`` sub-dict
    with training hyperparameters.

    Args:
        config_path: Path to a tt-train training config YAML file.

    Returns:
        A preset dict with keys matching MODEL_PRESETS entries, plus::

            preset["_training"] = {
                "batch_size": ...,
                "learning_rate": ...,
                "weight_decay": ...,
                "use_clip_grad_norm": ...,
                "clip_grad_norm_max_norm": ...,
            }
    """
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required to load training configs. Install it with: pip install pyyaml"
        ) from exc

    config_path = os.path.abspath(config_path)
    with open(config_path) as f:
        training_cfg = yaml.safe_load(f)

    tc = training_cfg.get("training_config", {})

    model_config_rel = tc.get("model_config")
    if model_config_rel is None:
        raise ValueError(
            f"'model_config' key not found in training_config section of: {config_path}"
        )

    model_config_path = _resolve_model_config_path(model_config_rel, config_path)
    with open(model_config_path) as f:
        model_cfg = yaml.safe_load(f)

    mc = model_cfg.get("transformer_config", {})
    model_type_str = mc.get("model_type", "llama").lower()
    config_label = os.path.basename(config_path)

    if model_type_str == "llama":
        preset: dict = {
            "model_type": ModelType.LLAMA,
            "vocab_size": mc["vocab_size"],
            "max_sequence_length": mc["max_sequence_length"],
            "embedding_dim": mc["embedding_dim"],
            "num_heads": mc["num_heads"],
            "num_groups": mc.get("num_groups", mc["num_heads"]),
            "num_blocks": mc["num_blocks"],
            "dropout": mc.get("dropout_prob", 0.0),
            "theta": mc.get("theta", 500000.0),
            "weight_tying": mc.get("weight_tying", False),
            "description": f"Loaded from {config_label} ({model_config_rel})",
        }
        if "intermediate_dim" in mc:
            preset["intermediate_dim"] = mc["intermediate_dim"]
    elif model_type_str in ("gpt", "nanogpt"):
        preset = {
            "model_type": ModelType.GPT,
            "vocab_size": mc["vocab_size"],
            "block_size": mc.get("block_size", mc.get("max_sequence_length", 256)),
            "n_embd": mc.get("n_embd", mc.get("embedding_dim", 384)),
            "n_layer": mc.get("n_layer", mc.get("num_blocks", 6)),
            "n_head": mc.get("n_head", mc.get("num_heads", 6)),
            "dropout": mc.get("dropout", mc.get("dropout_prob", 0.0)),
            "description": f"Loaded from {config_label} ({model_config_rel})",
        }
    else:
        raise ValueError(
            f"Unknown model_type '{model_type_str}' in {model_config_path}. "
            "Expected 'llama' or 'gpt'."
        )

    preset["_training"] = {
        "batch_size": tc.get("batch_size"),
        "learning_rate": tc.get("learning_rate", 1e-4),
        "weight_decay": tc.get("weight_decay", 0.1),
        "use_clip_grad_norm": tc.get("use_clip_grad_norm", False),
        "clip_grad_norm_max_norm": tc.get("clip_grad_norm_max_norm", 1.0),
    }

    return preset


def run_model_roofline(
    model_name: str = "nanogpt-char",
    batch_size: int = 64,
    seq_len: int = 256,
    hardware: str = "n150",
    plot_memory: bool = True,
    detailed: bool = False,
    preset: Optional[dict] = None,
    lr: float = 1e-4,
    weight_decay: float = 0.1,
    use_clip_grad_norm: bool = True,
    clip_grad_norm_max_norm: float = 1.0,
    tp_size: int = 1,
    ddp_size: int = 1,
):
    """Run transformer model roofline analysis (supports both GPT and Llama models).

    Args:
        model_name: Name of the model preset to use (ignored when ``preset`` is given).
        batch_size: Global  size for the analysis.
        seq_len: Sequence length for the analysis.
        hardware: Hardware configuration to use.
        plot_memory: Whether to generate memory usage plot.
        detailed: Print full per-op summary at the end.
        preset: Optional preset dict (same format as MODEL_PRESETS values).
            When provided, ``model_name`` is used only as a display label.
        lr: AdamW learning rate.
        weight_decay: AdamW weight-decay coefficient.
        use_clip_grad_norm: Whether to run gradient-norm clipping.
        clip_grad_norm_max_norm: Max norm value for gradient clipping.
        tp_size: Tensor parallelism size (1 = no TP). For Llama, tp_size > 1 uses MockDistributedLlama.
        ddp_size: Data-parallel replicas (1 = no DDP). Total devices = tp_size * ddp_size.
    """
    from roofline import (
        MockTensor,
        MockNanoGPT,
        MockNanoGPTConfig,
        MockLlama,
        MockLlamaConfig,
        MockDistributedLlama,
        MockDistributedLlamaConfig,
        RooflineContext,
        WORMHOLE_N150,
        WORMHOLE_N300,
        BLACKHOLE_P100,
        BLACKHOLE_P150,
        BLACKHOLE_GALAXY,
        DataType,
        MathFidelity,
        MockAdamW,
        mock_clip_grad_norm,
        MockCrossEntropyLossOp,
        TensorLabel,
    )
    from roofline.training.distributed import shard_batch, synchronize_gradients

    # Hardware mapping
    hardware_map = {
        "n150": WORMHOLE_N150,
        "n300": WORMHOLE_N300,
        "p100": BLACKHOLE_P100,
        "p150": BLACKHOLE_P150,
        "bh_glx": BLACKHOLE_GALAXY,
    }

    if hardware not in hardware_map:
        print(f"Unknown hardware: {hardware}")
        print(f"Available hardware: {', '.join(hardware_map.keys())}")
        return

    hw_spec = hardware_map[hardware]
    num_devices = tp_size * ddp_size
    assert (
        num_devices <= hw_spec.chips_per_galaxy
    ), f"num_devices (tp={tp_size} * ddp={ddp_size} = {num_devices}) must be <= chips_per_galaxy ({hw_spec.chips_per_galaxy}) for {hw_spec.name}"
    ddp_enabled = ddp_size > 1

    if ddp_enabled and batch_size % ddp_size != 0:
        print(f"Error: batch_size ({batch_size}) must be divisible by ddp_size ({ddp_size})")
        return

    print("=" * 70)
    print("TRANSFORMER MODEL ROOFLINE ANALYSIS")
    print("=" * 70)
    print()
    print(f"Hardware: {hw_spec.name}")
    print(f"  Cores:      {hw_spec.tensix_cores_per_chip}")
    print(f"  Clock:      {hw_spec.clock_ghz} GHz")
    print(f"  DRAM BW:    {hw_spec.dram_bw_gb_s} GB/s")
    print(f"  Eth BW/link: {hw_spec.eth_bw_gb_s_per_link} GB/s ({hw_spec.num_links} links, {hw_spec.topology})")
    print(f"  Peak (HiFi4): {hw_spec.tflops_per_chip(MathFidelity.HiFi4):.1f} TFLOPs")
    if ddp_enabled:
        print(f"  DDP replicas: {ddp_size}")
    if tp_size > 1:
        print(f"  TP Size: {tp_size} (tensor-parallel Llama)")
    print()

    # Resolve preset: use the directly-supplied preset or look up by name
    if preset is None:
        if model_name not in MODEL_PRESETS:
            print(f"Unknown model: {model_name}")
            print(f"Available presets: {', '.join(MODEL_PRESETS.keys())}")
            return
        preset = MODEL_PRESETS[model_name]

    model_type = preset["model_type"]

    # Create roofline context FIRST so that parameter tensors are tracked
    # (Memory tracking is enabled when context is created)
    ctx = RooflineContext(hw_spec)

    # Create model based on type (parameters will now be tracked)
    if model_type == ModelType.GPT:
        config = MockNanoGPTConfig(
            vocab_size=preset["vocab_size"],
            block_size=preset["block_size"],
            n_embd=preset["n_embd"],
            n_layer=preset["n_layer"],
            n_head=preset["n_head"],
            dropout=preset["dropout"],
        )
        model = MockNanoGPT(config)
        max_seq_len = config.block_size

        # Print GPT-specific config
        print(f"Model: {model_name} (GPT)")
        print(f"Description: {preset['description']}")
        print()
        print(f"Model Configuration:")
        print(f"  vocab_size:  {config.vocab_size:,}")
        print(f"  block_size:  {config.block_size}")
        print(f"  n_embd:      {config.n_embd}")
        print(f"  n_layer:     {config.n_layer}")
        print(f"  n_head:      {config.n_head}")
        print(f"  dropout:     {config.dropout}")
        print()

    elif model_type == ModelType.LLAMA:
        use_distributed_llama = tp_size > 1
        if use_distributed_llama:
            num_heads = preset["num_heads"]
            num_groups = preset["num_groups"]
            if num_heads % tp_size != 0 or num_groups % tp_size != 0:
                print(
                    f"Error: for TP size {tp_size}, Llama requires num_heads ({num_heads}) "
                    f"and num_groups ({num_groups}) divisible by tp_size. "
                    f"Choose a model preset where both are divisible by {tp_size}, or use --tp 1."
                )
                return
        if use_distributed_llama:
            config = MockDistributedLlamaConfig(
                vocab_size=preset["vocab_size"],
                max_sequence_length=preset["max_sequence_length"],
                embedding_dim=preset["embedding_dim"],
                intermediate_dim=preset.get("intermediate_dim"),
                num_heads=preset["num_heads"],
                num_groups=preset["num_groups"],
                num_blocks=preset["num_blocks"],
                dropout_prob=preset["dropout"],
                theta=preset["theta"],
                tp_size=tp_size,
            )
            model = MockDistributedLlama(config)
        else:
            config = MockLlamaConfig(
                vocab_size=preset["vocab_size"],
                max_sequence_length=preset["max_sequence_length"],
                embedding_dim=preset["embedding_dim"],
                intermediate_dim=preset.get("intermediate_dim"),
                num_heads=preset["num_heads"],
                num_groups=preset["num_groups"],
                num_blocks=preset["num_blocks"],
                dropout_prob=preset["dropout"],
                theta=preset["theta"],
                weight_tying=preset["weight_tying"],
            )
            model = MockLlama(config)
        max_seq_len = config.max_sequence_length

        # Print Llama-specific config
        print(f"Model: {model_name} (Llama{' [TP]' if use_distributed_llama else ''})")
        print(f"Description: {preset['description']}")
        print()
        print(f"Model Configuration:")
        print(f"  vocab_size:         {config.vocab_size:,}")
        print(f"  max_sequence_length: {config.max_sequence_length}")
        print(f"  embedding_dim:      {config.embedding_dim}")
        print(f"  num_blocks:         {config.num_blocks}")
        print(f"  num_heads:          {config.num_heads}")
        print(f"  num_groups:         {config.num_groups}")
        print(f"  dropout:            {config.dropout_prob}")
        print(f"  theta:              {config.theta}")
        if use_distributed_llama:
            print(f"  tp_size:            {config.tp_size}")
        print()

    else:
        print(f"Unknown model type: {model_type}")
        return

    # Clamp seq_len to max sequence length
    if seq_len > max_seq_len:
        print(
            f"Warning: seq_len ({seq_len}) > max_seq_len ({max_seq_len}), clamping to max_seq_len"
        )
        seq_len = max_seq_len

    per_device_batch = batch_size // ddp_size if ddp_enabled else batch_size

    print(f"Batch Configuration:")
    print(f"  batch_size (global): {batch_size}")
    if ddp_enabled:
        print(f"  batch_size (per device): {per_device_batch}")
    print(f"  seq_len:     {seq_len}")
    print()

    # Count parameters
    params = model.parameters()
    total_params = sum(p.logical_volume() for p in params.values())
    param_memory = sum(p.bytes() for p in params.values())

    print(f"Model Statistics:")
    print(f"  Parameters:  {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"  Param Memory: {param_memory/1e9:.3f} GB (BF16)")
    print()

    print("-" * 70)
    print(f"Running Analysis: B={batch_size}, S={seq_len}")
    print("-" * 70)

    # Helper to print memory snapshot
    def print_memory_snapshot(label: str):
        if ctx.memory_tracker is not None:
            current_bytes, breakdown = ctx.memory_tracker.current_memory()
            print(f"--- {label} ---")
            print(f"  Current Memory: {current_bytes / 1e6:.2f} MB")

    # Snapshot after model creation
    print_memory_snapshot("AFTER_MODEL_CREATION")

    # Create optimizer right after model (like ttml)
    # This allocates optimizer state (m and v tensors for AdamW)
    optimizer = MockAdamW(params, lr=lr, weight_decay=weight_decay)

    # Snapshot after optimizer creation
    print_memory_snapshot("AFTER_OPTIMIZER_CREATION")

    # Create input tensors (not tracked as training tensors)
    # Note: indices shape is [batch, 1, 1, seq_len] for ttml
    indices = MockTensor(
        (batch_size, 1, 1, seq_len),
        dtype=DataType.BFLOAT16,  # Will be cast to int internally
        requires_grad=False,
        label=TensorLabel.ACTIVATION,
        name="indices",
    )

    # Target for loss (same shape as indices)
    targets = MockTensor(
        (batch_size, 1, 1, seq_len),
        dtype=DataType.BFLOAT16,
        requires_grad=False,
        label=TensorLabel.ACTIVATION,
        name="targets",
    )

    # DDP: shard batch across devices (each device gets batch_size/N)
    if ddp_enabled:
        indices = shard_batch(indices, dim=0, num_devices=ddp_size)
        targets = shard_batch(targets, dim=0, num_devices=ddp_size)

    # Forward pass
    logits = model(ctx, indices)

    # Compute loss
    loss = MockCrossEntropyLossOp.apply(ctx, logits, targets)

    # Snapshot after forward pass
    print_memory_snapshot("AFTER_FORWARD_PASS")

    forward_time_ms = ctx.total_time_ms()
    forward_flops = ctx.total_flops()

    # Backward pass (retain_graph=False to deallocate activations/gradients early)
    loss.backward(ctx, retain_graph=False)

    # Snapshot after backward pass
    print_memory_snapshot("AFTER_BACKWARD_PASS")

    backward_time_ms = ctx.total_time_ms() - forward_time_ms
    backward_flops = ctx.total_flops() - forward_flops
    after_backward_time_ms = ctx.total_time_ms()
    after_backward_flops = ctx.total_flops()

    # DDP: synchronize gradients (all-reduce + average) before optimizer
    ccl_time_ms = 0.0
    ccl_flops = 0
    if ddp_enabled:
        synchronize_gradients(ctx, params, ddp_size)
        print_memory_snapshot("AFTER_GRAD_SYNC")
        ccl_time_ms = ctx.total_time_ms() - after_backward_time_ms
        ccl_flops = ctx.total_flops() - after_backward_flops

    after_ccl_time_ms = ctx.total_time_ms()
    after_ccl_flops = ctx.total_flops()

    # Optimizer step (optimizer already created above)
    optimizer.step(ctx)

    # Snapshot after optimizer step
    print_memory_snapshot("AFTER_OPTIMIZER_STEP")

    optimizer_time_ms = ctx.total_time_ms() - after_ccl_time_ms
    optimizer_flops = ctx.total_flops() - after_ccl_flops

    # Gradient clipping (optional)
    after_optimizer_time_ms = ctx.total_time_ms()
    after_optimizer_flops = ctx.total_flops()
    if use_clip_grad_norm:
        mock_clip_grad_norm(ctx, params, max_norm=clip_grad_norm_max_norm)

    # Snapshot after iteration complete
    print_memory_snapshot("ITERATION_COMPLETE")
    ctx.print_peak_memory()

    grad_clip_time_ms = ctx.total_time_ms() - after_optimizer_time_ms
    grad_clip_flops = ctx.total_flops() - after_optimizer_flops

    # Final metrics
    iteration_time_ms = ctx.total_time_ms()
    iteration_flops = ctx.total_flops()
    total_tokens = batch_size * seq_len  # global batch tokens
    tokens_per_second = total_tokens / (iteration_time_ms / 1000)

    print()
    print("Timing Breakdown (per device):")
    print(f"  Forward:     {forward_time_ms:.4f} ms ({forward_flops/1e12:.4f} TFLOPs)")
    print(
        f"  Backward:    {backward_time_ms:.4f} ms ({backward_flops/1e12:.4f} TFLOPs)"
    )
    if ddp_enabled:
        print(
            f"  CCL Sync:    {ccl_time_ms:.4f} ms"
        )
    print(
        f"  Optimizer:   {optimizer_time_ms:.4f} ms ({optimizer_flops/1e12:.4f} TFLOPs)"
    )
    print(
        f"  Grad Clip:   {grad_clip_time_ms:.4f} ms ({grad_clip_flops/1e12:.4f} TFLOPs)"
    )
    print(
        f"  Total:       {iteration_time_ms:.4f} ms ({iteration_flops/1e12:.4f} TFLOPs)"
    )
    print()
    print("Throughput:")
    print(f"  Tokens/iter: {total_tokens:,} (global batch)")
    if ddp_enabled:
        per_device_tokens = per_device_batch * seq_len
        print(f"  Tokens/device: {per_device_tokens:,}")
    print(f"  Tokens/sec:  {tokens_per_second:,.0f} (cluster total)")
    print(f"  TFLOPs/dev:  {ctx.achieved_tflops():.2f}")
    if ddp_enabled:
        compute_only_time_ms = forward_time_ms + backward_time_ms + optimizer_time_ms + grad_clip_time_ms
        ccl_overhead_pct = ccl_time_ms / iteration_time_ms * 100 if iteration_time_ms > 0 else 0
        print()
        print(f"DDP Summary ({ddp_size} replicas):")
        print(f"  Compute time:   {compute_only_time_ms:.4f} ms")
        print(f"  CCL overhead:   {ccl_time_ms:.4f} ms ({ccl_overhead_pct:.1f}%)")
        print(f"  Scaling eff:    {compute_only_time_ms / iteration_time_ms * 100:.1f}%")
    print()

    # Bottleneck analysis
    breakdown = ctx.bottleneck_breakdown()
    print("Bottleneck Analysis:")
    for btype, count in sorted(breakdown.items(), key=lambda x: -x[1]):
        print(f"  {btype.value}: {count} ops")

    # Peak memory analysis from tracking
    print()
    print("-" * 70)
    print("Memory Tracking Analysis")
    print("-" * 70)
    ctx.print_peak_memory()

    # Generate memory usage plots
    if plot_memory:
        # Stacked area plot (overview)
        plot_filename = f"memory_usage_{model_name}_b{batch_size}_s{seq_len}.png"
        ctx.plot_memory_usage(
            filename=plot_filename,
            title=f"Memory Usage: {model_name} (B={batch_size}, S={seq_len})",
            stacked=True,
        )
        # Detailed per-category plots (shows individual fluctuations)
        detail_filename = f"memory_detail_{model_name}_b{batch_size}_s{seq_len}.png"
        ctx.plot_memory_usage(
            filename=detail_filename,
            title=f"Memory Detail: {model_name} (B={batch_size}, S={seq_len})",
            stacked=False,
        )
    
    if detailed:
        print(ctx.summary())

    print()
    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    # Disable memory tracking when done
    ctx.disable_memory_tracking()


def list_models():
    """List available model presets."""
    print("Available model presets:")
    print("-" * 70)

    # Group by model type
    gpt_models = {
        k: v for k, v in MODEL_PRESETS.items() if v["model_type"] == ModelType.GPT
    }
    llama_models = {
        k: v for k, v in MODEL_PRESETS.items() if v["model_type"] == ModelType.LLAMA
    }

    if gpt_models:
        print("GPT Models:")
        for name, preset in gpt_models.items():
            print(f"  {name:<20} {preset['description']}")
        print()

    if llama_models:
        print("Llama Models:")
        for name, preset in llama_models.items():
            print(f"  {name:<20} {preset['description']}")
        print()

    print("-" * 70)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Transformer Model Roofline Analysis (GPT and Llama)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # GPT models
  python3 -m roofline.examples.nanogpt                            # Default: nanogpt-char, B=64, S=256, n150
  python3 -m roofline.examples.nanogpt -b 16 -s 512               # Custom batch/seq
  python3 -m roofline.examples.nanogpt --model gpt2-small -b 4 -s 1024

  # Llama models
  python3 -m roofline.examples.nanogpt --model nanollama -b 64 -s 256
  python3 -m roofline.examples.nanogpt --model tinyllama -b 1 -s 2048

  # Load from tt-train training config (infers model arch + training hyperparams)
  python3 -m roofline.examples.nanogpt --config path/to/training_config.yaml
  python3 -m roofline.examples.nanogpt --config path/to/training_config.yaml -b 32

  # Hardware configurations
  python3 -m roofline.examples.nanogpt --hardware n300            # Wormhole n300
  python3 -m roofline.examples.nanogpt --hardware p100            # Blackhole P100
  python3 -m roofline.examples.nanogpt --hardware p150            # Blackhole P150

  # DDP (data-distributed parallelism). Total devices = tp * ddp.
  python3 -m roofline.examples.nanogpt --model gpt2-small -b 32 --ddp 8  # 8-device DDP
  python3 -m roofline.examples.nanogpt --model nanogpt-char -b 64 --ddp 4

  # Tensor parallelism (Llama only; uses MockDistributedLlama when tp > 1)
  python3 -m roofline.examples.nanogpt --model tinyllama -b 1 -s 2048 --tp 2
  python3 -m roofline.examples.nanogpt --model nanollama --tp 4

  # TP + DDP combined
  python3 -m roofline.examples.nanogpt --model tinyllama --tp 2 --ddp 4 -b 8

  # Utilities
  python3 -m roofline.examples.nanogpt --list                     # List available presets
  python3 -m roofline.examples.nanogpt --detailed                 # Single-block analysis
""",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        metavar="TRAINING_CONFIG",
        help=(
            "Path to a tt-train training config YAML file. "
            "Model architecture and training hyperparameters are inferred from the config "
            "and the model config it references. "
            "Individual flags (--batch, --seq, --hardware) still override config values."
        ),
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=None,
        help="Model preset name (see --list for available presets). Ignored when --config is set.",
    )
    parser.add_argument(
        "--batch",
        "-b",
        type=int,
        default=None,
        help=(
            "Batch size. When --config is provided defaults to the config value; "
            "otherwise defaults to 64."
        ),
    )
    parser.add_argument(
        "--seq",
        "-s",
        type=int,
        default=None,
        help=(
            "Sequence length. When --config is provided defaults to the model's "
            "max_sequence_length; otherwise defaults to 256."
        ),
    )
    parser.add_argument(
        "--hardware",
        "-hw",
        type=str,
        choices=["n150", "n300", "p100", "p150", "bh_glx"],
        default="n150",
        help="Hardware configuration (default: n150)",
    )
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List available model presets",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Run detailed single-block analysis (GPT only)",
    )
    parser.add_argument(
        "--ddp",
        type=int,
        default=1,
        metavar="N",
        help="Number of data-parallel replicas (default: 1). Total devices = --tp * --ddp.",
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=1,
        metavar="N",
        help="Tensor parallelism size for Llama (default: 1). If > 1, uses MockDistributedLlama.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable memory usage plot generation",
    )
    args = parser.parse_args()

    if args.list:
        list_models()
        return

    # ------------------------------------------------------------------ #
    # Resolve model preset and training hyperparameters                    #
    # ------------------------------------------------------------------ #
    preset = None
    lr = 1e-4
    weight_decay = 0.1
    use_clip_grad_norm = True
    clip_grad_norm_max_norm = 1.0

    if args.config:
        try:
            preset = load_training_config(args.config)
        except (FileNotFoundError, ValueError, KeyError) as exc:
            print(f"Error loading training config: {exc}")
            sys.exit(1)

        training = preset.get("_training", {})
        model_name = os.path.splitext(os.path.basename(args.config))[0]

        lr = training.get("learning_rate", lr)
        weight_decay = training.get("weight_decay", weight_decay)
        use_clip_grad_norm = training.get("use_clip_grad_norm", use_clip_grad_norm)
        clip_grad_norm_max_norm = training.get("clip_grad_norm_max_norm", clip_grad_norm_max_norm)

        # Defaults sourced from config; CLI flags override
        mt = preset["model_type"]
        max_seq_from_config = (
            preset.get("max_sequence_length")
            if mt == ModelType.LLAMA
            else preset.get("block_size")
        )
        batch_size = args.batch if args.batch is not None else (training.get("batch_size") or 64)
        seq_len = args.seq if args.seq is not None else (max_seq_from_config or 256)

        print(f"Loaded training config: {args.config}")
    else:
        model_name = args.model or "nanogpt-char"
        batch_size = args.batch if args.batch is not None else 64
        seq_len = args.seq if args.seq is not None else 256

    run_model_roofline(
        model_name,
        batch_size=batch_size,
        seq_len=seq_len,
        hardware=args.hardware,
        plot_memory=not args.no_plot,
        detailed=args.detailed,
        preset=preset,
        lr=lr,
        weight_decay=weight_decay,
        use_clip_grad_norm=use_clip_grad_norm,
        clip_grad_norm_max_norm=clip_grad_norm_max_norm,
        tp_size=args.tp,
        ddp_size=args.ddp,
    )

if __name__ == "__main__":
    main()
