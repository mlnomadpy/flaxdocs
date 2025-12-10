---
sidebar_position: 4
---

# FSDP: Fully Sharded Data Parallel

Learn how to train very large models using Fully Sharded Data Parallel (FSDP) to minimize memory usage per device.

## Overview

**FSDP (Fully Sharded Data Parallel)** shards model parameters, gradients, and optimizer states across devices, dramatically reducing memory usage per device. This allows training models that wouldn't fit with standard data parallelism.

**Key Idea:** Instead of replicating the full model on each device, shard it!

```python
# Standard Data Parallel (pmap):
# Device 0: [Full Model] [Full Optimizer State]
# Device 1: [Full Model] [Full Optimizer State]
# Device 2: [Full Model] [Full Optimizer State]
# Memory per device: 100% of model

# FSDP:
# Device 0: [Model Shard 0] [Optimizer Shard 0]
# Device 1: [Model Shard 1] [Optimizer Shard 1]
# Device 2: [Model Shard 2] [Optimizer Shard 2]
# Memory per device: ~33% of model (with 3 devices)
```

## When to Use FSDP

✅ **Use FSDP when:**
- Model too large for single device with data parallelism
- Training 10B+ parameter models
- Want to maximize model size given hardware
- Have 8+ devices with fast interconnect

❌ **Don't use FSDP when:**
- Model fits easily on single device → Use data parallelism
- Very small models → Communication overhead not worth it
- Slow interconnect → Communication bottleneck

## How FSDP Works

### The FSDP Cycle

For each layer during forward pass:

1. **All-Gather**: Temporarily gather full parameters from shards
2. **Compute**: Execute layer with full parameters
3. **Discard**: Free gathered parameters (optional, for memory)
4. **Repeat**: For next layer

During backward pass:

1. **All-Gather**: Gather parameters again (if discarded)
2. **Compute Gradients**: With full parameters
3. **Reduce-Scatter**: Average gradients, shard results
4. **Update**: Each device updates its shard

```
Forward Pass (example with 4 devices):
┌─────────────┐
│  Device 0   │  Has: Param shard 0
│  Device 1   │  Has: Param shard 1  
│  Device 2   │  Has: Param shard 2
│  Device 3   │  Has: Param shard 3
└─────────────┘
       ↓ All-Gather
┌─────────────┐
│  Device 0   │  Now has: Full parameters (shards 0+1+2+3)
│  Device 1   │  Now has: Full parameters (shards 0+1+2+3)
│  Device 2   │  Now has: Full parameters (shards 0+1+2+3)
│  Device 3   │  Now has: Full parameters (shards 0+1+2+3)
└─────────────┘
       ↓ Compute Forward
       ↓ Discard parameters
       
Backward Pass:
┌─────────────┐
│  Device 0   │  Has: Gradient shard for all params
│  Device 1   │  Has: Gradient shard for all params
│  Device 2   │  Has: Gradient shard for all params
│  Device 3   │  Has: Gradient shard for all params
└─────────────┘
       ↓ Reduce-Scatter
┌─────────────┐
│  Device 0   │  Has: Averaged gradient shard 0
│  Device 1   │  Has: Averaged gradient shard 1
│  Device 2   │  Has: Averaged gradient shard 2
│  Device 3   │  Has: Averaged gradient shard 3
└─────────────┘
       ↓ Update parameters
```

### Memory Breakdown

```python
# Without FSDP (standard data parallel):
# Per device:
#   Parameters: P
#   Gradients: P
#   Optimizer states (Adam): 2P
#   Total: 4P

# With FSDP and N devices:
# Per device:
#   Parameters: P/N (sharded)
#   Gradients: P/N (sharded)
#   Optimizer states: 2P/N (sharded)
#   Total: 4P/N

# Memory reduction: N× smaller per device!
```

## Implementation with JAX

### Step 1: Create FSDP Mesh

```python
import jax
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import mesh_utils

# Create 1D mesh for FSDP
num_devices = jax.device_count()
devices = mesh_utils.create_device_mesh((num_devices,))
mesh = Mesh(devices, axis_names=('fsdp',))

print(f"FSDP across {num_devices} devices")
```

### Step 2: Shard Parameters

```python
from flax import nnx

# Initialize model
rngs = nnx.Rngs(0)
model = LargeTransformer(
    vocab_size=50000,
    d_model=2048,
    num_layers=24,
    rngs=rngs
)

# Extract parameters
graphdef, params = nnx.split(model)

# Create shardings
fsdp_sharding = NamedSharding(mesh, P('fsdp'))
replicated_sharding = NamedSharding(mesh, P())

def shard_fsdp(params, threshold=1024):
    """
    Shard large parameters, replicate small ones.
    
    Args:
        params: Model parameters
        threshold: Minimum size to shard (in elements)
    """
    def shard_array(path, array):
        path_str = '/'.join(str(p) for p in path)
        
        # Shard large weight matrices
        if array.size >= threshold and 'kernel' in path_str:
            return jax.device_put(array, fsdp_sharding)
        else:
            # Replicate small parameters (biases, norms)
            return jax.device_put(array, replicated_sharding)
    
    return jax.tree_util.tree_map_with_path(shard_array, params)

# Apply FSDP sharding
with mesh:
    params_sharded = shard_fsdp(params)

# Reconstruct model
model = nnx.merge(graphdef, params_sharded)
```

### Step 3: Create FSDP Training Step

```python
from functools import partial
import optax

def create_fsdp_train_step(mesh):
    """
    Training step with FSDP.
    
    JAX automatically handles:
    - All-gather before forward pass
    - Reduce-scatter after backward pass
    """
    
    # Sharding for data
    data_sharding = NamedSharding(mesh, P('fsdp'))
    
    @partial(jax.jit, donate_argnums=(0,))
    def train_step(state: nnx.Optimizer, batch: Dict):
        """
        FSDP training step with automatic collective operations.
        
        What happens:
        1. Parameters start sharded: P('fsdp')
        2. Forward pass triggers all-gather (compiler inserts)
        3. Backward computes gradients
        4. Reduce-scatter averages and shards gradients (automatic)
        5. Update sharded parameters
        """
        
        def loss_fn(model):
            logits = model(batch['input_ids'])
            labels_onehot = jax.nn.one_hot(batch['label'], num_classes=10)
            loss = optax.softmax_cross_entropy(logits, labels_onehot).mean()
            
            predictions = jnp.argmax(logits, axis=-1)
            accuracy = jnp.mean(predictions == batch['label'])
            
            return loss, {'accuracy': accuracy}
        
        # Compute gradients
        # JAX automatically:
        # - All-gathers sharded parameters
        # - Computes gradients  
        # - Reduce-scatters gradients
        grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
        (loss, metrics), grads = grad_fn(state.model)
        
        # Update (parameters remain sharded)
        state.update(grads)
        
        return state, loss, metrics
    
    return train_step
```

### Step 4: Training Loop

```python
# Create optimizer
optimizer = optax.adam(learning_rate=1e-4)
state = nnx.Optimizer(model, optimizer)

# Create training step
train_step = create_fsdp_train_step(mesh)

# Sharding for input data
data_sharding = NamedSharding(mesh, P('fsdp'))

# Training loop
with mesh:
    for epoch in range(num_epochs):
        for batch in data_loader:
            # Shard input data
            batch_sharded = jax.tree.map(
                lambda x: jax.device_put(x, data_sharding),
                batch
            )
            
            # Training step (FSDP magic happens automatically)
            state, loss, metrics = train_step(state, batch_sharded)
            
            # Metrics are replicated, use directly
            if step % 100 == 0:
                print(f"Loss: {float(loss):.4f}")
```

## Memory Analysis

### Calculate Memory Savings

```python
def analyze_memory(params, num_devices):
    """Analyze memory usage with and without FSDP."""
    
    total_params = sum(p.size for p in jax.tree.leaves(params))
    bytes_per_param = 4  # float32
    
    # Without FSDP (replicated)
    model_memory = total_params * bytes_per_param
    gradient_memory = total_params * bytes_per_param
    optimizer_memory = total_params * bytes_per_param * 2  # Adam: 2× params
    
    total_per_device_replicated = model_memory + gradient_memory + optimizer_memory
    
    # With FSDP (sharded)
    total_per_device_fsdp = total_per_device_replicated / num_devices
    
    print(f"\nMemory Analysis:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Model size: {model_memory / 1e9:.2f} GB")
    print(f"\n  WITHOUT FSDP:")
    print(f"    Per device: {total_per_device_replicated / 1e9:.2f} GB")
    print(f"    Total: {total_per_device_replicated * num_devices / 1e9:.2f} GB")
    print(f"\n  WITH FSDP:")
    print(f"    Per device: {total_per_device_fsdp / 1e9:.2f} GB")
    print(f"    Savings: {(1 - total_per_device_fsdp / total_per_device_replicated) * 100:.1f}%")
    print(f"    Can train {total_per_device_replicated / total_per_device_fsdp:.1f}× larger model!")

# Example output:
# Memory Analysis:
#   Total parameters: 1,000,000,000
#   Model size: 4.00 GB
#
#   WITHOUT FSDP:
#     Per device: 16.00 GB  (4 + 4 + 8 GB)
#     Total: 128.00 GB (8 devices)
#
#   WITH FSDP:
#     Per device: 2.00 GB
#     Savings: 87.5%
#     Can train 8.0× larger model!
```

### Real-World Examples

```python
# GPT-3 Scale (175B parameters):
# - Without FSDP: ~700GB per device (won't fit on any GPU!)
# - With FSDP on 1024 devices: ~0.68GB per device (fits!)

# LLaMA-65B:
# - Without FSDP: ~260GB per device
# - With FSDP on 64 devices: ~4GB per device (fits on A100-40GB)

# Your model (10B parameters):
# - Without FSDP: ~40GB per device
# - With FSDP on 8 devices: ~5GB per device
```

## Advanced Techniques

### Activation Checkpointing with FSDP

Combine FSDP with activation checkpointing for even more memory savings:

```python
def fsdp_with_checkpointing(model_fn):
    """Combine FSDP with activation checkpointing."""
    
    @nnx.remat  # Recompute activations during backward
    def checkpointed_forward(x):
        return model_fn(x)
    
    return checkpointed_forward

# Memory:
# FSDP alone: 4P/N per device
# FSDP + checkpointing: ~2P/N per device (depends on model)
```

### Hybrid Sharding

Combine FSDP with tensor parallelism:

```python
# Create 2D mesh: FSDP × Tensor Parallel
devices = mesh_utils.create_device_mesh((8, 4))  # 32 devices
mesh = Mesh(devices, axis_names=('fsdp', 'tensor'))

# Shard parameters on both axes
hybrid_sharding = NamedSharding(mesh, P('fsdp', 'tensor'))

# Benefits:
# - FSDP reduces memory: P/8 per device
# - Tensor parallel: Split large matrices further by 4
# - Total: P/32 per device

# Use case: 100B+ parameter models
```

### Selective Sharding

Shard only the largest layers:

```python
def selective_fsdp(params, size_threshold=10_000_000):
    """
    Shard only very large parameters.
    
    Useful when:
    - Some layers are huge (embeddings)
    - Most layers fit comfortably on device
    - Want to minimize communication
    """
    def shard_if_large(path, array):
        if array.size >= size_threshold:
            print(f"Sharding: {path} ({array.size:,} elements)")
            return jax.device_put(array, fsdp_sharding)
        else:
            return jax.device_put(array, replicated_sharding)
    
    return jax.tree_util.tree_map_with_path(shard_if_large, params)

# Example: Shard embedding layer (100M+ params), replicate everything else
```

## Communication Patterns

### All-Gather

Reconstruct full parameters from shards:

```python
# Before: Each device has 1/N of parameters
# Device 0: [W0]
# Device 1: [W1]
# Device 2: [W2]
# Device 3: [W3]

# After all-gather: Each device has full parameters
# Device 0: [W0, W1, W2, W3]
# Device 1: [W0, W1, W2, W3]
# Device 2: [W0, W1, W2, W3]
# Device 3: [W0, W1, W2, W3]

# Communication: Each device sends/receives (N-1)/N of parameter size
# Total data transferred per device: ~P (for P parameters)
```

### Reduce-Scatter

Average gradients and shard result:

```python
# Before: Each device has full gradients
# Device 0: [G0, G1, G2, G3]
# Device 1: [G0', G1', G2', G3']
# Device 2: [G0'', G1'', G2'', G3'']
# Device 3: [G0''', G1''', G2''', G3''']

# After reduce-scatter: Each device has averaged shard
# Device 0: [avg(G0, G0', G0'', G0''')]
# Device 1: [avg(G1, G1', G1'', G1''')]
# Device 2: [avg(G2, G2', G2'', G2''')]
# Device 3: [avg(G3, G3', G3'', G3''')]

# Communication: Same as all-gather, ~P per device
```

### Total Communication Cost

```python
# Per training step:
# - All-gather: P bytes per layer
# - Reduce-scatter: P bytes per layer
# - Total: 2P per layer

# Compared to data parallelism:
# - Data parallel: P bytes (all-reduce)
# - FSDP: 2P bytes per layer

# FSDP has more communication, but enables much larger models
```

## Performance Optimization

### Batch Size Selection

```python
# Larger batches amortize communication:

# Small batch (32):
# - Compute time: 10ms
# - Communication: 15ms
# - Efficiency: 40% (10/(10+15))

# Large batch (256):
# - Compute time: 80ms
# - Communication: 15ms (same!)
# - Efficiency: 84% (80/(80+15))

# Rule: Use largest batch that fits in memory (after FSDP sharding)
```

### Overlapping Communication and Computation

JAX automatically overlaps when possible:

```python
# While layer N is computing forward pass,
# layer N+1 can be all-gathering its parameters

# This reduces effective communication time

# Achieved automatically with jax.jit compilation
```

### Fast Interconnect

FSDP performance heavily depends on device interconnect:

```python
# NVLink (900 GB/s): Excellent FSDP performance
# PCIe 4.0 (64 GB/s): Acceptable but slower
# Ethernet (10 Gb/s): Too slow, don't use FSDP

# Rule: FSDP needs ≥100 GB/s interconnect for efficiency
```

## Debugging FSDP

### Check Sharding

```python
# Verify parameters are actually sharded
graphdef, params = nnx.split(model)

def print_sharding(path, array):
    path_str = '/'.join(str(p) for p in path)
    if hasattr(array, 'sharding'):
        print(f"{path_str:50s} {str(array.shape):20s} {array.sharding.spec}")

jax.tree_util.tree_map_with_path(print_sharding, params)

# Look for P('fsdp') in output - these are sharded
```

### Monitor Memory Usage

```python
# Check actual memory usage
import psutil
import os

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1e9  # GB

before = get_memory_usage()
# ... training step ...
after = get_memory_usage()

print(f"Memory used: {after - before:.2f} GB")
```

### Profile Communication

```python
from jax import profiler

with profiler.trace("/tmp/jax-trace"):
    state, loss = train_step(state, batch)

# View in TensorBoard:
# Look for:
# - AllGather operations
# - ReduceScatter operations
# Check their duration vs compute time
```

## Example: Complete Script

See `examples/19_fsdp_sharding.py` in the repository for a complete implementation with:

- ✅ FSDP mesh creation
- ✅ Parameter sharding strategies
- ✅ Automatic collective operations
- ✅ Memory analysis and savings calculation
- ✅ Training loop with FSDP

## Comparison with Alternatives

| Strategy | Memory/Device | Model Size Limit | Communication | Complexity |
|----------|---------------|------------------|---------------|------------|
| **Data Parallel** | 4P | Single device | 1× (all-reduce) | Low |
| **FSDP** | 4P/N | N× larger | 2× (all-gather + reduce-scatter) | Medium |
| **Pipeline** | P/N | N× larger | Activations only | High |
| **Tensor Parallel** | P/N per layer | Per-layer limit | Per-layer | High |

## Real-World Usage

### Meta LLaMA

```python
# LLaMA-65B trained with FSDP on 2048 A100 GPUs
# - Parameters: 65B
# - Without FSDP: Impossible (needs 260GB per GPU)
# - With FSDP: ~130MB per GPU ✓

# Configuration:
# - FSDP across all devices
# - Mixed precision (bfloat16)
# - Gradient accumulation
# - Activation checkpointing
```

### Your Use Case

```python
# Model: 10B parameters
# Hardware: 8× A100-40GB

# Without FSDP:
# - Needs: 40GB per GPU (doesn't fit!)

# With FSDP:
# - Needs: 5GB per GPU (fits easily!)
# - Overhead: ~30% slower (due to communication)
# - Result: Can train 8× larger model ✓
```

## Complete Example

**FSDP (Fully Sharded Data Parallel) implementation:**
- [`examples/distributed/fsdp_sharding.py`](https://github.com/mlnomadpy/flaxdocs/tree/master/examples/distributed/fsdp_sharding.py) - Complete FSDP training with parameter sharding, all-gather/reduce-scatter, and memory optimization for large models

## Next Steps

- **Simpler approach?** → Start with [Data Parallelism](./data-parallelism.md)
- **Need even more memory?** → Combine with [Pipeline Parallelism](./pipeline-parallelism.md)
- **Flexible sharding?** → Learn [SPMD](./spmd-sharding.md)
