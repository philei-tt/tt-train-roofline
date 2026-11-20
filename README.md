# tt-train-roofline

Roofline performance analysis tool for training transformer models on Tenstorrent hardware. It simulates a full training loop — forward pass, backward pass, optimizer step — using metadata-only mock tensors, estimating ideal execution time, memory usage, and compute/memory bottlenecks without requiring actual hardware.

The module interface mirrors [ttml](https://github.com/tenstorrent/tt-metal/tree/main/tt-train) (Tenstorrent's training framework), which itself is similar to PyTorch: models are built from composable modules, operations produce and consume tensors, and gradients flow through an autograd-style backward graph.

## How It Works

Each operation (matmul, layernorm, softmax, etc.) is implemented as a `RooflineFunction` that:

1. Computes output tensor **shapes** (no actual data).
2. Calculates the **FLOPs** and **bytes transferred** for both forward and backward passes.
3. Creates a `RooflineEstimate` and registers it with the `RooflineContext`.

The `RooflineContext` accumulates all estimates across the training iteration and classifies each operation as **compute-bound** or **DRAM-bound** based on the hardware's peak TFLOPS and memory bandwidth.

A `MemoryTracker` runs alongside, logging every allocation and deallocation by category (parameters, activations, gradients, optimizer state), enabling peak memory analysis and memory usage plots.

## Architecture

```
roofline/
├── roofline/        # Core roofline math per op (FLOPs, bytes, ideal time)
├── operations/      # RooflineFunction wrappers that build the backward graph
├── modules/         # High-level modules (Linear, Attention, GPTBlock, Llama, ...)
├── training/        # AdamW optimizer, gradient clipping
├── examples/        # Ready-to-run analysis scripts
├── hardware.py      # Hardware specs (Wormhole N150/N300, Blackhole P100/P150)
├── mock_tensor.py   # Metadata-only tensor with autograd graph
└── memory_tracker.py# Allocation tracking and plotting
```

**Supported hardware:** Wormhole N150, Wormhole N300, Blackhole P100, Blackhole P150

**Supported models:** NanoGPT (char/BPE), GPT-2 (Small/Medium/Large), NanoLlama, TinyLlama 1.1B, Llama 3.x (1B, 8B, 70B, 405B)

**Supported operations:** MatMul, Linear, Embedding, LayerNorm, RMSNorm, Softmax, Scaled Dot-Product Attention (unfused & fused), RoPE, Dropout, SiLU, GELU, SwiGLU (fused), Cross-Entropy Loss

## Examples

All examples are run from the repository root.

### Full training analysis

Analyze a complete training iteration (forward + backward + optimizer) for any supported model:

```bash
# NanoGPT on Shakespeare (default)
python3 -m roofline.examples.training

# TinyLlama 1.1B, batch=1, seq=2048
python3 -m roofline.examples.training --model tinyllama -b 1 -s 2048

# Llama 8B on Blackhole P150
python3 -m roofline.examples.training --model llama-8b -b 1 -s 8192 --hardware p150

# List all available model presets
python3 -m roofline.examples.training --list
```

Outputs timing breakdown (forward/backward/optimizer), throughput (tokens/sec), bottleneck classification per op, peak memory, and memory usage plots.

Example of output:
```
======================================================================
TRANSFORMER MODEL ROOFLINE ANALYSIS
======================================================================

Hardware: Blackhole
  Cores:      120
  Clock:      1.35 GHz
  DRAM BW:    480 GB/s
  Peak (HiFi4): 165.9 TFLOPs

Model: tinyllama-char (Llama)
Description: TinyLlama 1.1B (char tokenizer, 0.96B params)

Model Configuration:
  vocab_size:         96
  max_sequence_length: 2048
  embedding_dim:      2048
  num_blocks:         22
  num_heads:          32
  num_groups:         4
  dropout:            0.0
  theta:              10000.0

Batch Configuration:
  batch_size:  1
  seq_len:     2048

Model Statistics:
  Parameters:  969,369,600 (969.4M)
  Param Memory: 1.939 GB (BF16)

----------------------------------------------------------------------
Running Analysis: B=1, S=2048
----------------------------------------------------------------------
--- AFTER_MODEL_CREATION ---
  Current Memory: 1938.74 MB
--- AFTER_OPTIMIZER_CREATION ---
  Current Memory: 5816.22 MB
--- AFTER_FORWARD_PASS ---
  Current Memory: 15914.41 MB
--- AFTER_BACKWARD_PASS ---
  Current Memory: 7754.96 MB
--- AFTER_OPTIMIZER_STEP ---
  Current Memory: 7754.96 MB
--- ITERATION_COMPLETE ---
  Current Memory: 7754.96 MB
Peak Memory Usage: 15.990 GB (15,990,300,674 bytes)
Peak Memory Breakdown by category:
  parameter      : 1.939 GB (12.1%)
  optimizer_state: 3.877 GB (24.2%)
  activation     : 10.073 GB (63.0%)
  gradient       : 0.101 GB (0.6%)

Timing Breakdown:
  Forward:     110.3905 ms (4.7669 TFLOPs)
  Backward:    194.7631 ms (9.4815 TFLOPs)
  Optimizer:   28.2733 ms (0.0010 TFLOPs)
  Grad Clip:   20.1952 ms (0.0029 TFLOPs)
  Total:       353.6221 ms (14.2523 TFLOPs)

Throughput:
  Tokens/iter: 2,048
  Tokens/sec:  5,791
  TFLOPs:      40.30

Bottleneck Analysis:
  DRAM: 961 ops
  COMPUTE: 330 ops
  BOTH: 66 ops

----------------------------------------------------------------------
Memory Tracking Analysis
----------------------------------------------------------------------
Peak Memory Usage: 15.990 GB (15,990,300,674 bytes)
Peak Memory Breakdown by category:
  parameter      : 1.939 GB (12.1%)
  optimizer_state: 3.877 GB (24.2%)
  activation     : 10.073 GB (63.0%)
  gradient       : 0.101 GB (0.6%)
Memory usage plot saved to: memory_usage_tinyllama-char_b1_s2048.png
Memory usage plot saved to: memory_detail_tinyllama-char_b1_s2048.png

======================================================================
ANALYSIS COMPLETE
======================================================================
```

Memory plot example:
![](./media/memory_usage_tinyllama-char_b1_s2048.png)

### Fused vs unfused attention

Compare standard multi-head attention against fused (Flash Attention-style) SDPA across multiple model sizes:

```bash
python3 -m roofline.examples.attention_comparison
```

### SwiGLU comparison

Compare different SwiGLU MLP implementations:

```bash
python3 -m roofline.examples.swiglu_comparison
```

### Matmul benchmarking

Roofline analysis for matrix multiplications across a range of shapes:

```bash
python3 -m roofline.examples.benchmark_matmuls
```

### Simple matmul pipeline

Minimal example — a 5-layer linear model showing per-layer roofline breakdown:

```bash
python3 -m roofline.examples.matmuls
```

### Parallel training: pipeline, tensor, sequence and context parallelism

`roofline.examples.parallel_training` models one optimizer step of a Llama on `pp * tp * cp` chips:

- **PP** with a choice of schedule (`--schedule naive | gpipe | 1f1b | interleaved`, `--interleave v` chunks per device for interleaved), optional vocabulary parallelism (`--vocab-parallel`, LM head split across stages)
- **TP** inside a stage, optionally with Megatron sequence parallelism (`--sp`)
- **CP** inside a stage: ring attention with a load-balanced split and K/V shifts in both ring directions
- activation recompute (`--recompute`, the memory-efficient runner)

```bash
# One configuration: Llama 70B, 8K sequences, 2048 sequences per step, on a quad galaxy (128 chips)
python3 -m roofline.examples.parallel_training --model llama-70b -s 8192 -m 2048 \
    --pp 16 --tp 2 --cp 4 --recompute --vocab-parallel --attn-block 512

# Every PP x TP x CP split of 128 chips, best ones that fit in memory
python3 -m roofline.examples.parallel_training --model llama-70b -s 8192 -m 2048 \
    --sweep 128 --sp --vocab-parallel --attn-block 512

# Uneven layer split (as tt-train's PipelineParallelConfig.blocks_per_rank): fewer blocks on the LM-head stage
python3 -m roofline.examples.parallel_training --model llama-70b -s 8192 -m 2048 \
    --pp 8 --tp 4 --cp 4 --recompute --blocks-per-stage 11,11,10,10,10,10,10,8
```

Layers are split evenly across pipeline devices by default, the remainder going to the first devices; `--blocks-per-stage` sets any split. Every distinct stage (first, middle, last, each block count) is run for one micro-batch in its own `RooflineContext`. A stage's time per micro-batch is a resource-level roofline: compute, DRAM and fabric (Ethernet) time are summed separately and the largest one sets the time, so the three overlap:

```
D_d     = max(FLOP, DRAM, fabric) for one micro-batch on device d
          (fabric includes v stage hand-offs per micro-batch)
T_pipe  = the schedule played out op by op (simulate_schedule), device d taking D_d per micro-batch,
          split into forward and backward in the proportion of its own forward and backward phases
        = slots * D when all devices are equal
T_step  = max( T_pipe + max(optimizer FLOP, optimizer DRAM),  m * fabric + CP gradient sync )
bubble  = idle share of the step over all devices = (slots - m) / slots when all devices are equal
MFU     = model FLOPs / T_step / (chips * peak)
          model FLOPs: counted by the mock run over all stages, forward + backward without recompute
          (as tt-train's tools/profiling/results_json/add_roofline.py)
```

`slots` is the step length in micro-batch times and comes from the schedule (`roofline/training/pipeline.py`). Each schedule also gives the micro-batches whose activations a device holds at once, which sets memory:

| Schedule | slots | bubble | held on device 0 |
|---|---|---|---|
| `naive`: one micro-batch at a time | m P | (P - 1) / P | 1 |
| `gpipe`: all forwards, then all backwards (Huang et al., 2019) | m + P - 1 | (P - 1) / (m + P - 1) | m |
| `1f1b` (Narayanan et al., 2021) | m + P - 1 | (P - 1) / (m + P - 1) | P |
| `interleaved`, v chunks per device (same paper) | m + (P - 1) / v | (P - 1) / (v m + P - 1) | simulated |

`simulate_schedule` plays any schedule out op by op on a common clock; with equal devices every closed form above matches it. With uneven devices the slowest one sets the pace and the fill and drain take the sum of the devices' times; the held counts it reports set each device's memory.

The per-op serial time (sum of `max(compute, DRAM, network)` over ops, as in `training.py`) is reported next to it.

Supporting changes:

| Where | What |
|---|---|
| `RooflineEstimate.ideal_network_ns`, `roofline/roofline/ccl.py` | Collectives put their link time and a fixed per-call latency (fitted to measured ttnn collectives, tt-metal `tech_reports/CCLs/CCL_bandwidth.md`) on the fabric, and their input read and output write on DRAM; `theoretical_time_ns = max(compute, memory, network)` |
| `RooflineContext.compute_time_ns / dram_time_ns / network_time_ns / overlapped_time_ns` | Per-resource sums and their max |
| `fused_attention_roofline(..., block)`, `RooflineContext.attn_block` (`--attn-block`) | Follows tt-train's `sdpa_fw` and split `sdpa_bw` (Q and KV kernels): causal, K/V re-read per Q block in the forward and the Q kernel, Q, dO and the fp32 statistics re-read per K/V block in the KV kernel; backward matmul FLOPs 14 · B · Hq · M · d. Default block: one 32-token tile, today's kernels; 512 makes attention compute-bound |
| `ring_shift_roofline`, `send_recv_roofline` | CP ring shifts (both ring directions) and stage hand-offs over the links across a stage cut |
| `MockRingAttentionOp`, `cp_size` on the distributed block | Ring attention: attention of the full sequence divided by `cp`, K/V (bf16) shifts forward, K/V + dK/dV (fp32) backward, fp32 accumulator read + write per ring step |
| `sequence_parallel` on the parallel linears, attention, MLP, block and `MockDistributedLlamaConfig` | All-gather / reduce-scatter along the sequence instead of broadcast / all-reduce; norms and residual adds hold `S / tp` tokens |
| `num_stages`, `stage_index`, `stage_num_blocks`, `vocab_parallel` on `MockDistributedLlamaConfig`; `training/pipeline.py` | The model as one pipeline stage (`forward_stage`); schedule classes (`NaiveSchedule`, `GPipeSchedule`, `OneFOneBSchedule`, `InterleavedOneFOneBSchedule`) and a schedule simulator |
| Embedding, LM head, gradients | Embedding untilizes its whole table each forward and returns a dense table gradient (tt-train `embedding_op`); LM head keeps logits sharded over the vocabulary with `MockVocabParallelCrossEntropyLossOp` (tt-train `vocab_parallel_cross_entropy_loss`); each stage run starts with every gradient allocated, as after the first micro-batch, so the out-of-place gradient add and the resident gradients are in the step time and the memory peak; the micro-batches held at once count only the blocks' activations |
| `BLACKHOLE_QUAD`, `HARDWARE_PRESETS` | 4 galaxies, 128 chips, 32 x 4 torus, 2 links per neighbouring chip pair; one name-to-spec map used by both examples (`--hardware bh_quad`) |

Limitations: TP must divide the number of KV heads (as in tt-train). With `--sp` the attention and MLP gather the norm output once and keep the gathered copy for backward (Megatron keeps the shard and re-gathers), so memory is higher than Megatron's. The communication of vocabulary parallelism is not modelled. `--cut-links` (default 8) assumes stages laid along the 32-chip axis of a quad.

## Dependencies

Python 3.10+ with only standard-library dependencies. `matplotlib` is optional (used for memory usage plots).
