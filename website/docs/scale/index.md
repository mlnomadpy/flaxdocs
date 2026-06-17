---
sidebar_position: 0
title: Distributed Training with JAX & Flax
description: Learn distributed training with Flax NNX and JAX. Master data parallelism, model parallelism, and sharding strategies for training at scale across multiple devices.
keywords: [Flax distributed training, JAX parallelism, multi-GPU training, TPU training, data parallelism, model parallelism, sharding]
---

# Distributed Training Overview

Scale your Flax NNX models across multiple devices with JAX's powerful parallelism primitives. This guide covers everything from simple data parallelism to advanced sharding strategies for training models at any scale.

:::note Prerequisites
Distributed training builds on the single-device workflow. Make sure you're comfortable with [Your First Training Loop](/basics/workflows/simple-training) and [Data Loading](/basics/workflows/data-loading-simple) first.
:::

:::tip What you'll learn
- How to pick a parallelism strategy from a model-size-vs-device-memory decision guide
- The trade-offs between data parallelism, SPMD, pipeline, and FSDP (memory, throughput, communication)
- How to profile a model's parameter count and step time before choosing a strategy
- Why training needs ~4× model size in memory (params + gradients + optimizer state)
- How to combine strategies (FSDP + data, pipeline + tensor, 3D parallelism) for the largest models
:::

## Quick Decision Guide

**Choose your parallelism strategy:**

```
┌─────────────────────────────────────────┐
│ Does your model fit on a single device? │
└──────────────┬──────────────────────────┘
               │
        ┌──────┴──────┐
        │             │
       YES            NO
        │             │
        ▼             ▼
    ┌───────┐    ┌──────────────────┐
    │ Data  │    │ Need sequential  │
    │Parallel│   │ architecture?    │
    └───────┘    └─────────┬────────┘
                           │
                    ┌──────┴──────┐
                    │             │
                   YES            NO
                    │             │
                    ▼             ▼
            ┌──────────┐    ┌──────────┐
            │Pipeline  │    │FSDP or   │
            │Parallel  │    │Tensor    │
            └──────────┘    │Parallel  │
                            └──────────┘
```

## Parallelism Strategies

### 1. Data Parallelism

**Replicate model, split data**

- ✅ Simplest to implement
- ✅ Perfect scaling for throughput
- ✅ No model changes needed
- ❌ Model must fit on single device

```python
@jax.pmap
def train_step(state, batch):
    # Each device processes different data
    grads = compute_gradients(state, batch)
    grads = jax.lax.pmean(grads, 'devices')  # Sync
    return state.apply_gradients(grads)
```

**📚 [Learn Data Parallelism →](./data-parallelism.md)**

### 2. SPMD / Automatic Sharding

**Flexible sharding with compiler optimization**

- ✅ Very flexible (any sharding pattern)
- ✅ Modern JAX best practice
- ✅ Automatic optimization
- ⚠️ Requires understanding sharding

```python
mesh = Mesh(devices, axis_names=('data', 'model'))
sharding = NamedSharding(mesh, P('data', 'model'))

@jax.jit
def train_step(state, batch):
    # Compiler handles communication automatically
    return update(state, batch)
```

**📚 [Learn SPMD Sharding →](./spmd-sharding.md)**

### 3. Pipeline Parallelism

**Split model into stages**

- ✅ Train very large models
- ✅ Works with sequential architectures
- ❌ Pipeline bubbles (70-90% efficiency)
- ❌ Complex implementation

```python
# Stage 1 on Device 0
# Stage 2 on Device 1
# Stage 3 on Device 2
# Stage 4 on Device 3

# Microbatches flow through pipeline
```

**📚 [Learn Pipeline Parallelism →](./pipeline-parallelism.md)**

### 4. FSDP (Fully Sharded Data Parallel)

**Shard everything to save memory**

- ✅ Massive memory savings (N× reduction)
- ✅ Train N× larger models
- ❌ More communication overhead
- ⚠️ Needs fast interconnect

```python
# Shard parameters across all devices
mesh = Mesh(devices, axis_names=('fsdp',))
params = shard_fsdp(params, mesh)

# Automatic all-gather and reduce-scatter
```

**📚 [Learn FSDP →](./fsdp-fully-sharded.md)**

## Strategy Comparison

| Strategy | Memory/Device | Throughput | Communication | Best For |
|----------|---------------|------------|---------------|----------|
| **Data Parallel** | Full model (P) | Excellent | O(P) once/step | Standard training |
| **SPMD** | Configurable | Excellent | Optimized | Flexible needs |
| **Pipeline** | P/N | Good (70-90%) | O(activations) | Very large models |
| **FSDP** | P/N | Good | O(2P) per layer | Memory constrained |

**P** = Model size, **N** = Number of devices

## Real-World Examples

### Example 1: ResNet-50 Training (25M params)

**Model fits easily on single GPU**

```python
# ✅ Best: Data Parallelism
# - Simple implementation
# - Perfect scaling
# - No memory concerns

# Configuration:
# - 8× A100 GPUs
# - Batch size: 32/device = 256 total
# - Training time: 100% efficient
```

### Example 2: GPT-2 Medium (355M params)

**Model fits but tight on memory**

```python
# ✅ Best: SPMD with data parallelism
# - Flexible for future growth
# - Modern approach
# - Can add model parallelism if needed

# Configuration:
# - 8× A100-40GB
# - Pure data parallel: P('data', None)
# - Or light tensor parallel: P('data', 'model') with mesh (4, 2)
```

### Example 3: GPT-3 Scale (175B params)

**Model way too large for single device**

```python
# ✅ Best: Combination strategy
# - FSDP for memory: 1024 devices
# - Pipeline: 8 stages
# - Tensor parallel: 4-way per stage

# Configuration:
# - 1024× A100-80GB
# - Mesh: (8, 4, 32) = (pipeline, tensor, fsdp)
# - Per device: ~600MB
```

### Example 4: Vision Transformer (1B params)

**Sequential architecture, moderately large**

```python
# ✅ Good options:
# Option A: FSDP (if 16+ GPUs)
# - Memory: 1GB per device (16 GPUs)
# - Clean implementation

# Option B: Pipeline (if 4-8 GPUs)
# - 4 stages, 8 microbatches
# - Efficiency: 73%

# Configuration:
# - 8× A100-40GB
# - Choose based on interconnect speed
```

## Combining Strategies

Many large-scale training runs combine multiple strategies:

### FSDP + Data Parallelism

```python
# Shard model parameters (FSDP)
# Each shard replica uses data parallelism

mesh = Mesh(devices, axis_names=('fsdp', 'data'))
# Shape: (8, 16) = 128 devices total
# 8-way FSDP, 16-way data parallel per shard
```

### Pipeline + Tensor Parallelism

```python
# Split model into pipeline stages
# Each stage uses tensor parallelism

# Stage 1: Devices 0-7 (8-way tensor parallel)
# Stage 2: Devices 8-15 (8-way tensor parallel)
# Stage 3: Devices 16-23 (8-way tensor parallel)
# Stage 4: Devices 24-31 (8-way tensor parallel)
```

### 3D Parallelism (Pipeline + Tensor + Data)

```python
# The ultimate combination for massive models

mesh = Mesh(devices, axis_names=('pipeline', 'tensor', 'data'))
# Shape: (8, 8, 16) = 1024 devices
# - 8 pipeline stages
# - 8-way tensor parallel per stage
# - 16-way data parallel

# Used by: GPT-3, PaLM, LLaMA-2
```

## Getting Started

### Step 1: Profile Your Model

```python
import jax
import jax.numpy as jnp
from flax import nnx

# Initialize model
model = YourModel(...)

# Check size
graphdef, params = nnx.split(model)
total_params = sum(p.size for p in jax.tree.leaves(params))
model_size_gb = total_params * 4 / 1e9  # float32

print(f"Model: {total_params/1e9:.2f}B parameters ({model_size_gb:.2f} GB)")

# Profile one training step
@jax.jit
def profile_step(state, batch):
    # Your training step
    pass

# Run once to compile
state, metrics = profile_step(state, batch)

# Time it
import time
start = time.time()
for _ in range(10):
    state, metrics = profile_step(state, batch)
elapsed = (time.time() - start) / 10

print(f"Step time: {elapsed*1000:.1f}ms")
```

### Step 2: Choose Strategy

Use the decision guide above based on:
- Model size vs device memory
- Number of available devices
- Interconnect speed
- Architecture (sequential or not)

### Step 3: Implement

Follow the detailed guides for your chosen strategy:

- 🚀 **Starting out?** → [Data Parallelism](./data-parallelism.md)
- 🎯 **Want flexibility?** → [SPMD Sharding](./spmd-sharding.md)
- 💾 **Need memory?** → [FSDP](./fsdp-fully-sharded.md)
- 🏗️ **Very large model?** → [Pipeline Parallelism](./pipeline-parallelism.md)

### Step 4: Optimize

After basic implementation works:

1. **Profile** with `jax.profiler.trace()`
2. **Check device utilization** (should be >80%)
3. **Adjust batch size** (larger = better efficiency)
4. **Enable mixed precision** (bfloat16)
5. **Tune communication** (see Best Practices)

## Common Pitfalls

### ❌ Wrong: Using pmap for large models

```python
# Model: 10B parameters, won't fit on single GPU!
@jax.pmap
def train_step(state, batch):
    # Each device needs full 10B model = OOM!
    pass
```

**✅ Use FSDP or Pipeline Parallelism instead**

### ❌ Wrong: Too few microbatches with pipeline

```python
# 4 pipeline stages, only 2 microbatches
# Efficiency: 2/(2+4-1) = 40% (terrible!)
```

**✅ Use 4× stages microbatches minimum (16 for 4 stages)**

### ❌ Wrong: FSDP with slow interconnect

```python
# Using FSDP over 10Gb Ethernet
# Communication time > compute time!
```

**✅ FSDP needs NVLink or InfiniBand (100+ GB/s)**

### ❌ Wrong: Not accounting for optimizer state

```python
# Model: 10GB
# Fits on A100-40GB?
# NO! Adam needs: 10GB params + 10GB grads + 20GB optimizer = 40GB
```

**✅ Budget 4× model size for training (params + grads + optimizer)**

## Scaling Laws

### Data Parallelism Scaling

```
Throughput = single_device_throughput × num_devices × efficiency

# Efficiency typically:
# - 2-4 devices: 95-98%
# - 8 devices: 90-95%
# - 16+ devices: 85-92%
# (Depends on model size and interconnect)
```

### FSDP Scaling

```
Max_model_size = device_memory × num_devices / 4

# Examples (A100-40GB):
# - 8 devices: 80GB model (20B params)
# - 64 devices: 640GB model (160B params)
# - 1024 devices: 10TB model (2.5T params)
```

### Pipeline Efficiency

```
Efficiency = M / (M + S - 1)
where M = microbatches, S = stages

# To reach 90% efficiency:
M ≥ 9 × S - 9

# Examples:
# 4 stages: Need 27+ microbatches for 90%
# 8 stages: Need 63+ microbatches for 90%
```

## Hardware Considerations

### Interconnect Speed

| Interconnect | Bandwidth | Good For |
|--------------|-----------|----------|
| **NVLink (V100)** | 300 GB/s | All strategies ✅ |
| **NVLink (A100)** | 600 GB/s | All strategies ✅✅ |
| **NVLink (H100)** | 900 GB/s | All strategies ✅✅✅ |
| **PCIe 4.0** | 64 GB/s | Data parallel only |
| **10Gb Ethernet** | 1.25 GB/s | Single device only |
| **InfiniBand** | 200 GB/s | All strategies ✅✅ |

### Memory Hierarchy

```
Device Memory (fast, small):
  ├─ L2 Cache: ~40-80MB (fastest)
  ├─ HBM: 40-80GB (fast)
  └─ When full → OOM!

Host Memory (slow, large):
  └─ RAM: 100s of GB (for data loading)

Storage (slowest, largest):
  └─ Disk: TBs (for dataset)
```

**Key insight:** Training happens in device memory. Must fit model + gradients + optimizer + activations.

## Monitoring Training

### Essential Metrics

```python
# 1. Loss / Accuracy (correctness)
# 2. Step time (efficiency)
# 3. Device utilization (>80% ideal)
# 4. Memory usage (should be high but not OOM)
# 5. Communication time (should be <30% of step time)

# Log these every N steps:
if step % 100 == 0:
    metrics = {
        'loss': float(loss),
        'accuracy': float(accuracy),
        'step_time_ms': step_time * 1000,
        'device_utilization': utilization,
    }
    print(metrics)
```

### Profiling

```python
# Profile to find bottlenecks
from jax import profiler

with profiler.trace("/tmp/jax-trace"):
    for _ in range(10):
        state = train_step(state, batch)

# View in TensorBoard:
# tensorboard --logdir=/tmp/jax-trace

# Look for:
# - Computation time (should be high)
# - Communication time (minimize)
# - Idle time (minimize)
```

## Next steps

- **Start simple:** [Data Parallelism](/scale/data-parallelism) — replicate the model, split the batch with `pmap`
- **Go modern:** [SPMD Sharding](/scale/spmd-sharding) — declarative sharding with `jax.jit` and device meshes
- **Save memory:** [FSDP](/scale/fsdp-fully-sharded) — shard params, gradients, and optimizer state
- **Very large models:** [Pipeline Parallelism](/scale/pipeline-parallelism) — split the model into stages

## Example Code

Check out our complete, runnable examples:

- `examples/distributed/data_parallel_pmap.py` - Data parallelism with pmap
- `examples/distributed/sharding_spmd.py` - SPMD automatic sharding
- `examples/distributed/pipeline_parallel.py` - Pipeline parallelism
- `examples/distributed/fsdp_sharding.py` - FSDP fully sharded training

Each example is self-contained and includes detailed comments explaining what's happening under the hood.

## Further Reading

- [JAX Documentation on Parallelism](https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html)
- [Google Cloud TPU Guide](https://cloud.google.com/tpu/docs/jax-pods)
- [Megatron-LM Paper](https://arxiv.org/abs/1909.08053) (Pipeline + Tensor Parallelism)
- [ZeRO Paper](https://arxiv.org/abs/1910.02054) (FSDP inspiration)
- [GPipe Paper](https://arxiv.org/abs/1811.06965) (Pipeline Parallelism)
