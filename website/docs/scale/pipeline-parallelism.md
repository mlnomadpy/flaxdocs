---
sidebar_position: 3
---

# Pipeline Parallelism

Learn how to split large models across multiple devices using pipeline parallelism with Flax NNX.

## Overview

**Pipeline parallelism** splits a model into sequential stages, with each stage running on a different device. Multiple microbatches flow through the pipeline to maximize device utilization.

```
Device 0: [Stage 1: Embedding + Early Layers]
          ↓
Device 1: [Stage 2: Middle Layers]
          ↓
Device 2: [Stage 3: More Middle Layers]
          ↓
Device 3: [Stage 4: Final Layers + Head]
```

## When to Use Pipeline Parallelism

✅ **Use pipeline parallelism when:**
- Your model is too large for a single device
- Model has a sequential structure (Transformers, ResNets)
- You have multiple devices available
- Data parallelism alone isn't sufficient

❌ **Don't use pipeline parallelism when:**
- Model fits on one device → Use data parallelism
- Non-sequential architecture (complex DAGs)
- Very few devices (inefficient with <4 devices)

## How It Works

### The Pipeline Schedule

Consider a 4-stage pipeline with 4 microbatches:

```
Time →
T0:  [S0→MB0] [----] [----] [----]
T1:  [S0→MB1] [S1→MB0] [----] [----]
T2:  [S0→MB2] [S1→MB1] [S2→MB0] [----]
T3:  [S0→MB3] [S1→MB2] [S2→MB1] [S3→MB0]  ← Pipeline full
T4:  [----] [S1→MB3] [S2→MB2] [S3→MB1]
T5:  [----] [----] [S2→MB3] [S3→MB2]
T6:  [----] [----] [----] [S3→MB3]

S = Stage, MB = Microbatch
```

### Pipeline Efficiency

```python
efficiency = num_microbatches / (num_microbatches + num_stages - 1)

# Examples:
# 4 stages, 4 microbatches: 4 / (4 + 4 - 1) = 57%
# 4 stages, 8 microbatches: 8 / (8 + 4 - 1) = 73%
# 4 stages, 16 microbatches: 16 / (16 + 4 - 1) = 84%

# More microbatches = higher efficiency (but more memory)
```

### Pipeline Bubbles

"Bubbles" are idle time at the start and end of the pipeline:

```
[Busy] = Device doing work
[Idle] = Device idle (pipeline bubble)

      Stage 0    Stage 1    Stage 2    Stage 3
T0:   [Busy]     [Idle]     [Idle]     [Idle]  ← 3 idle
T1:   [Busy]     [Busy]     [Idle]     [Idle]  ← 2 idle
T2:   [Busy]     [Busy]     [Busy]     [Idle]  ← 1 idle
T3:   [Busy]     [Busy]     [Busy]     [Busy]  ← All busy!
T4:   [Idle]     [Busy]     [Busy]     [Busy]  ← 1 idle
T5:   [Idle]     [Idle]     [Busy]     [Busy]  ← 2 idle
T6:   [Idle]     [Idle]     [Idle]     [Busy]  ← 3 idle
```

## Implementation Guide

### Step 1: Define Model Stages

Split your model into sequential stages:

```python
from flax import nnx

class Stage1(nnx.Module):
    """Early layers: Embedding + Initial processing."""
    
    def __init__(self, vocab_size: int, d_model: int, rngs: nnx.Rngs = None):
        self.embedding = nnx.Embed(vocab_size, d_model, rngs=rngs)
        self.conv1 = nnx.Conv(d_model, d_model, kernel_size=(3,), rngs=rngs)
        self.conv2 = nnx.Conv(d_model, d_model, kernel_size=(3,), rngs=rngs)
        self.ln = nnx.LayerNorm(d_model, rngs=rngs)
    
    def __call__(self, token_ids):
        x = self.embedding(token_ids)
        x = self.conv1(x)
        x = nnx.relu(x)
        x = self.conv2(x)
        x = self.ln(x)
        return x


class Stage2(nnx.Module):
    """Middle layers: Transformer block."""
    
    def __init__(self, d_model: int, num_heads: int, rngs: nnx.Rngs = None):
        self.attention = TransformerBlock(d_model, num_heads, rngs=rngs)
    
    def __call__(self, x):
        return self.attention(x)


class Stage3(nnx.Module):
    """More middle layers: Another transformer block."""
    
    def __init__(self, d_model: int, num_heads: int, rngs: nnx.Rngs = None):
        self.attention = TransformerBlock(d_model, num_heads, rngs=rngs)
    
    def __call__(self, x):
        return self.attention(x)


class Stage4(nnx.Module):
    """Final layers: Pooling + Classification."""
    
    def __init__(self, d_model: int, num_classes: int, rngs: nnx.Rngs = None):
        self.ln = nnx.LayerNorm(d_model, rngs=rngs)
        self.classifier = nnx.Linear(d_model, num_classes, rngs=rngs)
    
    def __call__(self, x):
        x = self.ln(x)
        x = jnp.mean(x, axis=1)  # Global average pooling
        return self.classifier(x)
```

### Step 2: Place Stages on Devices

```python
import jax

num_devices = jax.device_count()
devices = jax.devices()

# Assign each stage to a device
stage_devices = {
    1: devices[0],
    2: devices[1] if num_devices > 1 else devices[0],
    3: devices[2] if num_devices > 2 else devices[0],
    4: devices[3] if num_devices > 3 else devices[0],
}

# Initialize stages
rngs = nnx.Rngs(0)
stages = {
    1: Stage1(vocab_size=10000, d_model=256, rngs=rngs),
    2: Stage2(d_model=256, num_heads=8, rngs=rngs),
    3: Stage3(d_model=256, num_heads=8, rngs=rngs),
    4: Stage4(d_model=256, num_classes=10, rngs=rngs),
}

# Place stage parameters on devices
for stage_id, stage in stages.items():
    graphdef, params = nnx.split(stage)
    
    # Move params to specific device
    device = stage_devices[stage_id]
    params_on_device = jax.tree.map(
        lambda x: jax.device_put(x, device),
        params
    )
    
    stages[stage_id] = nnx.merge(graphdef, params_on_device)
```

### Step 3: Microbatching

```python
def split_into_microbatches(batch: Dict, num_microbatches: int) -> List[Dict]:
    """
    Split batch into microbatches.
    
    Why? To keep all pipeline stages busy simultaneously.
    While Stage 1 processes microbatch N, Stage 2 processes microbatch N-1.
    """
    batch_size = batch['input_ids'].shape[0]
    microbatch_size = batch_size // num_microbatches
    
    microbatches = []
    for i in range(num_microbatches):
        start = i * microbatch_size
        end = (i + 1) * microbatch_size
        
        microbatch = {
            'input_ids': batch['input_ids'][start:end],
            'label': batch['label'][start:end]
        }
        microbatches.append(microbatch)
    
    return microbatches
```

### Step 4: Pipeline Execution

```python
def pipeline_forward(stages, microbatches, stage_devices):
    """
    Execute forward pass through pipeline.
    
    Returns activations for each stage and microbatch.
    """
    num_microbatches = len(microbatches)
    num_stages = len(stages)
    
    # Store intermediate activations
    # activations[stage_id][mb_id] = activation tensor
    activations = {stage: {} for stage in range(1, num_stages + 1)}
    
    # Process each microbatch through stages sequentially
    for mb_idx, microbatch in enumerate(microbatches):
        # Stage 1
        x = stages[1](microbatch['input_ids'])
        x = jax.device_put(x, stage_devices[2])  # Transfer to next stage
        activations[1][mb_idx] = x
        
        # Stage 2
        x = stages[2](x)
        x = jax.device_put(x, stage_devices[3])
        activations[2][mb_idx] = x
        
        # Stage 3
        x = stages[3](x)
        x = jax.device_put(x, stage_devices[4])
        activations[3][mb_idx] = x
        
        # Stage 4
        logits = stages[4](x)
        activations[4][mb_idx] = logits
    
    return activations


def pipeline_backward(stages, activations, microbatches):
    """
    Execute backward pass through pipeline.
    
    Compute gradients for each stage, averaged across microbatches.
    """
    num_microbatches = len(microbatches)
    
    # Accumulate gradients for each stage
    accumulated_grads = {stage: None for stage in stages.keys()}
    
    for mb_idx, microbatch in enumerate(microbatches):
        # Stage 4: compute loss and gradients
        def loss_fn_stage4(stage):
            logits = activations[4][mb_idx]
            labels_onehot = jax.nn.one_hot(microbatch['label'], num_classes=10)
            return optax.softmax_cross_entropy(logits, labels_onehot).mean()
        
        grads4 = nnx.grad(loss_fn_stage4)(stages[4])
        
        # Stage 3: gradient w.r.t inputs (simplified)
        def loss_fn_stage3(stage):
            x = activations[2][mb_idx]
            return jnp.sum(stage(x) ** 2)
        
        grads3 = nnx.grad(loss_fn_stage3)(stages[3])
        
        # Stage 2
        def loss_fn_stage2(stage):
            x = activations[1][mb_idx]
            return jnp.sum(stage(x) ** 2)
        
        grads2 = nnx.grad(loss_fn_stage2)(stages[2])
        
        # Stage 1
        def loss_fn_stage1(stage):
            x = microbatch['input_ids']
            return jnp.sum(stage(x) ** 2)
        
        grads1 = nnx.grad(loss_fn_stage1)(stages[1])
        
        # Accumulate gradients
        for stage_id, grads in [(1, grads1), (2, grads2), (3, grads3), (4, grads4)]:
            if accumulated_grads[stage_id] is None:
                accumulated_grads[stage_id] = grads
            else:
                accumulated_grads[stage_id] = jax.tree.map(
                    lambda a, g: a + g,
                    accumulated_grads[stage_id], grads
                )
    
    # Average gradients across microbatches
    averaged_grads = {}
    for stage_id, grads in accumulated_grads.items():
        averaged_grads[stage_id] = jax.tree.map(
            lambda g: g / num_microbatches,
            grads
        )
    
    return averaged_grads
```

### Step 5: Training Loop

```python
# Create optimizers for each stage
optimizers = {}
for stage_id, stage in stages.items():
    optimizer = optax.adam(learning_rate=1e-3)
    optimizers[stage_id] = nnx.Optimizer(stage, optimizer)

# Training
num_microbatches = 8  # More microbatches = higher efficiency

for epoch in range(num_epochs):
    for batch in data_loader:
        # Split into microbatches
        microbatches = split_into_microbatches(batch, num_microbatches)
        
        # Forward pass through pipeline
        activations = pipeline_forward(stages, microbatches, stage_devices)
        
        # Backward pass
        gradients = pipeline_backward(stages, activations, microbatches)
        
        # Update each stage
        for stage_id in stages.keys():
            optimizers[stage_id].update(gradients[stage_id])
```

## Advanced: GPipe Schedule

GPipe is a more sophisticated pipeline schedule that overlaps forward and backward passes:

```
Forward pass: F1, F2, F3, F4
Backward pass: B1, B2, B3, B4

Time 0:  [F1] [--] [--] [--]
Time 1:  [F1] [F2] [--] [--]
Time 2:  [F1] [F2] [F3] [--]
Time 3:  [F1] [F2] [F3] [F4]
Time 4:  [F1] [B2] [F3] [F4]  ← Forward and backward overlap!
Time 5:  [B1] [B2] [B3] [F4]
Time 6:  [B1] [B2] [B3] [B4]
Time 7:  [B1] [B2] [B3] [B4]
```

This reduces pipeline bubbles significantly.

## Memory Considerations

### Activation Checkpointing

Pipeline parallelism stores activations for the backward pass, which can use significant memory:

```python
# Memory per microbatch:
# - Store activation at each stage boundary
# - N microbatches × (N-1) boundaries × activation_size

# With activation checkpointing:
# - Only store inputs to each stage
# - Recompute activations during backward
# - Trades compute for memory

def checkpoint_stage(stage, x):
    """Recompute activations during backward."""
    return nnx.remat(stage)(x)
```

### Microbatch Size Selection

```python
# Total batch size = microbatch_size × num_microbatches

# Trade-offs:
# - More microbatches → Better pipeline efficiency
# - More microbatches → More memory (storing activations)
# - Larger microbatches → Less communication overhead

# Rule of thumb:
num_microbatches = 4 * num_stages  # Good starting point
```

## Limitations and Challenges

### 1. Pipeline Bubbles

Wasted computation at start/end:

```python
# Efficiency formula:
efficiency = num_microbatches / (num_microbatches + num_stages - 1)

# With 4 stages:
# 4 microbatches: 57% efficiency (43% wasted)
# 8 microbatches: 73% efficiency (27% wasted)
# 16 microbatches: 84% efficiency (16% wasted)

# Solution: Use more microbatches (but increases memory)
```

### 2. Load Imbalance

Stages must take equal time:

```python
# If stage times are: 10ms, 20ms, 15ms, 5ms
# Total time is determined by slowest stage (20ms)
# Other stages idle waiting

# Solution: Profile and rebalance stages
```

### 3. Sequential Architecture Required

Pipeline parallelism only works for sequential models:

```python
# ✅ Works: Sequential (Transformers, ResNets)
output = stage4(stage3(stage2(stage1(input))))

# ❌ Doesn't work: Complex DAG
out1 = branch1(input)
out2 = branch2(input)
output = merge(out1, out2)
```

## Combining with Other Strategies

### Pipeline + Data Parallelism

```python
# Each stage can use data parallelism internally

# Stage 1 on Devices 0-3 (data parallel)
# Stage 2 on Devices 4-7 (data parallel)
# etc.

# Benefits:
# - Larger models (pipeline)
# - Higher throughput (data parallel per stage)
```

### Pipeline + Tensor Parallelism

```python
# Split model vertically (pipeline) and horizontally (tensor)

# Stage 1: Devices 0-1 (tensor parallel within stage)
# Stage 2: Devices 2-3 (tensor parallel within stage)

# Benefits:
# - Very large models
# - Large layers within each stage
```

## Example: Complete Script

See [`examples/18_pipeline_parallelism.py`](../../examples/18_pipeline_parallelism.py) for a complete implementation with:

- ✅ Model stage definition
- ✅ Device placement
- ✅ Microbatching
- ✅ Pipeline schedule
- ✅ Gradient accumulation

## Comparison with Alternatives

| Strategy | Memory/Device | Communication | Efficiency | Complexity |
|----------|---------------|---------------|------------|------------|
| **Data Parallel** | Full model | Gradients only | 100% | Low |
| **Pipeline** | 1/N of model | Activations | 70-90% | Medium |
| **Tensor Parallel** | 1/N of layers | Every layer | 100% | High |
| **FSDP** | 1/N of params | All-gather/reduce | 100% | Medium |

## Next Steps

- **Need more memory savings?** → Try [FSDP](./fsdp-fully-sharded.md)
- **Simple parallelism?** → Start with [Data Parallelism](./data-parallelism.md)
- **Flexible sharding?** → Learn [SPMD](./spmd-sharding.md)
- **Best practices?** → Read [Distributed Training Tips](./best-practices.md)
