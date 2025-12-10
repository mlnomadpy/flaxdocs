---
sidebar_position: 1
---

# Data Parallelism with pmap

Learn how to scale training across multiple devices using JAX's `pmap` for data parallelism with Flax NNX.

## Overview

Data parallelism is the simplest and most common form of distributed training. It works by:

1. **Replicating** your model on each device
2. **Splitting** each batch across devices  
3. **Computing** forward/backward passes independently on each device
4. **Synchronizing** gradients across all devices
5. **Updating** parameters identically on all devices

This approach is ideal when your model fits comfortably on a single device and you want to process more data per second.

## When to Use Data Parallelism

✅ **Use data parallelism when:**
- Your model fits on a single device
- You want to increase training throughput
- You want to use larger batch sizes
- You have multiple GPUs/TPUs available
- Your model architecture is standard (no special requirements)

❌ **Don't use data parallelism when:**
- Your model is too large for a single device → Use FSDP or Pipeline Parallelism
- You need very specific sharding patterns → Use SPMD sharding
- Communication is a major bottleneck → Consider model parallelism

## How pmap Works

### The Execution Model

`jax.pmap` (parallel map) replicates a function across multiple devices:

```python
@jax.pmap
def parallel_function(x):
    # This function runs independently on each device
    # x has shape (per_device_data, ...)
    return x * 2

# Input shape: (num_devices, per_device_batch, ...)
# Each device gets one slice: x[device_id]
```

### Key Concepts

1. **Device Axis**: The first dimension of inputs/outputs is the device axis
2. **SPMD Execution**: Same Program, Multiple Data - same code runs on all devices
3. **Collective Operations**: Use `jax.lax.pmean`, `psum`, etc. for cross-device communication
4. **Automatic Compilation**: pmap compiles code once and replicates execution

## Implementation Guide

### Step 1: Check Available Devices

```python
import jax
import jax.numpy as jnp
from flax import nnx

# Check devices
num_devices = jax.local_device_count()
print(f"Available devices: {num_devices}")
print(f"Devices: {jax.local_devices()}")

# Device count should match your hardware:
# - 1 CPU
# - N GPUs if you have N GPUs
# - 8 TPU cores if on TPU
```

### Step 2: Define Your Model

```python
class CNNModel(nnx.Module):
    """Example CNN for data parallel training."""
    
    def __init__(self, num_classes: int = 10, rngs: nnx.Rngs = None):
        self.conv1 = nnx.Conv(3, 32, kernel_size=(3, 3), rngs=rngs)
        self.conv2 = nnx.Conv(32, 64, kernel_size=(3, 3), rngs=rngs)
        self.dense = nnx.Linear(64 * 8 * 8, num_classes, rngs=rngs)
    
    def __call__(self, x, train: bool = False):
        x = self.conv1(x)
        x = nnx.relu(x)
        x = nnx.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        x = self.conv2(x)
        x = nnx.relu(x)
        x = nnx.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        x = x.reshape(x.shape[0], -1)
        x = self.dense(x)
        return x
```

### Step 3: Replicate Model Across Devices

```python
# Initialize model on host
rngs = nnx.Rngs(0)
model = CNNModel(num_classes=10, rngs=rngs)

# Create optimizer
optimizer = optax.adam(learning_rate=1e-3)
state = nnx.Optimizer(model, optimizer)

# Replicate state across devices
graphdef, state_arrays = nnx.split(state)

# Add device dimension by broadcasting
replicated_state = jax.tree.map(
    lambda x: jnp.broadcast_to(x, (num_devices,) + x.shape),
    state_arrays
)

# Merge back
state = nnx.merge(graphdef, replicated_state)
```

### Step 4: Create Parallel Training Step

```python
from functools import partial

@partial(jax.pmap, axis_name='devices')
def train_step(state: nnx.Optimizer, batch: Dict):
    """
    Training step executed in parallel on all devices.
    
    Args:
        state: Optimizer state (replicated, shape: (num_devices, ...))
        batch: Data batch (sharded, shape: (num_devices, per_device_batch, ...))
    
    Returns:
        Updated state, loss, and metrics (all with device dimension)
    """
    
    def loss_fn(model):
        logits = model(batch['image'], train=True)
        labels_onehot = jax.nn.one_hot(batch['label'], num_classes=10)
        loss = optax.softmax_cross_entropy(logits, labels_onehot).mean()
        
        predictions = jnp.argmax(logits, axis=-1)
        accuracy = jnp.mean(predictions == batch['label'])
        
        return loss, {'accuracy': accuracy}
    
    # Compute gradients on local data shard
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, metrics), grads = grad_fn(state.model)
    
    # CRITICAL: Synchronize gradients across devices
    # pmean computes the mean across the 'devices' axis
    # This ensures all devices have the same gradient
    grads = jax.lax.pmean(grads, axis_name='devices')
    loss = jax.lax.pmean(loss, axis_name='devices')
    metrics = jax.tree.map(lambda x: jax.lax.pmean(x, axis_name='devices'), metrics)
    
    # Update parameters (identical on all devices after pmean)
    state.update(grads)
    
    return state, loss, metrics
```

### Step 5: Prepare Data for pmap

```python
def shard_batch(batch: Dict, num_devices: int) -> Dict:
    """
    Reshape batch to add device dimension.
    
    Input:  (total_batch_size, ...)
    Output: (num_devices, per_device_batch_size, ...)
    """
    def reshape_for_devices(x):
        batch_size = x.shape[0]
        assert batch_size % num_devices == 0, \
            f"Batch size {batch_size} must be divisible by num_devices"
        
        per_device_batch = batch_size // num_devices
        return x.reshape((num_devices, per_device_batch) + x.shape[1:])
    
    return jax.tree.map(reshape_for_devices, batch)


# Example usage
batch_size = 128  # Total batch size
per_device_batch = batch_size // num_devices  # 16 per device (if 8 devices)

# Get your batch (shape: (128, 32, 32, 3))
batch = {
    'image': images,  # (128, 32, 32, 3)
    'label': labels   # (128,)
}

# Shard for pmap (shape: (8, 16, 32, 32, 3))
batch_sharded = shard_batch(batch, num_devices)
```

### Step 6: Training Loop

```python
num_epochs = 10
total_batch_size = 128

for epoch in range(num_epochs):
    epoch_loss = 0.0
    epoch_acc = 0.0
    num_batches = 0
    
    for batch in data_loader:
        # Ensure batch size is correct
        if batch['image'].shape[0] != total_batch_size:
            continue
        
        # Shard batch across devices
        batch_sharded = shard_batch(batch, num_devices)
        
        # Parallel training step
        state, loss, metrics = train_step(state, batch_sharded)
        
        # Extract metrics (take first device since all are identical)
        epoch_loss += float(loss[0])
        epoch_acc += float(metrics['accuracy'][0])
        num_batches += 1
    
    # Log epoch metrics
    avg_loss = epoch_loss / num_batches
    avg_acc = epoch_acc / num_batches
    print(f"Epoch {epoch + 1}: Loss={avg_loss:.4f}, Acc={avg_acc:.4f}")
```

## Deep Dive: Gradient Synchronization

### What Happens During `pmean`

```python
# Before pmean: each device has different gradients
# Device 0: grad = [1.0, 2.0, 3.0]
# Device 1: grad = [1.5, 2.5, 3.5]
# Device 2: grad = [0.5, 1.5, 2.5]

grads = jax.lax.pmean(grads, axis_name='devices')

# After pmean: all devices have the same (averaged) gradient
# All devices: grad = [1.0, 2.0, 3.0]  # (1.0+1.5+0.5)/3, etc.
```

### Under the Hood: All-Reduce

`pmean` implements an **all-reduce** collective operation:

1. **All-Reduce**: Each device contributes its gradients, all devices receive averaged result
2. **Ring Algorithm**: For N devices, requires N-1 communication steps
3. **Bandwidth Optimal**: Transfers minimum data needed

```
Time 0: [D0] → [D1] → [D2] → [D3] → (back to D0)
Time 1: [D0] → [D1] → [D2] → [D3] → (back to D0)
...
Result: All devices have sum/average of all gradients
```

### Communication Cost

- **Data transferred per device**: O(model_size)
- **Time complexity**: O(model_size / bandwidth)
- **Scaling**: Communication cost is constant regardless of batch size!

This is why larger batches are more efficient - you amortize communication over more computation.

## Advanced Techniques

### 1. Gradient Accumulation with pmap

When you want even larger effective batch sizes:

```python
@partial(jax.pmap, axis_name='devices')
def train_step_with_accumulation(state, batches):
    """Accumulate gradients over multiple microbatches."""
    
    def compute_grads(params, batch):
        def loss_fn(params):
            logits = state.model.apply({'params': params}, batch['image'])
            return optax.softmax_cross_entropy_with_integer_labels(
                logits, batch['label']
            ).mean()
        return jax.grad(loss_fn)(params)
    
    # Accumulate gradients
    accumulated_grads = None
    
    for batch in batches:
        grads = compute_grads(state.params, batch)
        
        if accumulated_grads is None:
            accumulated_grads = grads
        else:
            accumulated_grads = jax.tree.map(
                lambda a, g: a + g,
                accumulated_grads, grads
            )
    
    # Average accumulated gradients
    accumulated_grads = jax.tree.map(
        lambda g: g / len(batches),
        accumulated_grads
    )
    
    # Synchronize across devices
    accumulated_grads = jax.lax.pmean(accumulated_grads, axis_name='devices')
    
    # Update
    state = state.apply_gradients(grads=accumulated_grads)
    return state
```

### 2. Mixed Precision Training

Reduce communication and computation with mixed precision:

```python
@partial(jax.pmap, axis_name='devices')
def train_step_mixed_precision(state, batch):
    """Training step with automatic mixed precision."""
    
    def loss_fn(model):
        # Cast inputs to bfloat16
        x = batch['image'].astype(jnp.bfloat16)
        
        # Forward pass in bfloat16
        logits = model(x, train=True)
        
        # Cast back to float32 for loss
        logits = logits.astype(jnp.float32)
        labels_onehot = jax.nn.one_hot(batch['label'], num_classes=10)
        
        loss = optax.softmax_cross_entropy(logits, labels_onehot).mean()
        return loss
    
    grad_fn = nnx.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.model)
    
    # Gradients computed in float32, sync in float32
    grads = jax.lax.pmean(grads, axis_name='devices')
    
    state.update(grads)
    return state, loss
```

### 3. Dynamic Batch Sizes

Handle variable batch sizes with masking:

```python
@partial(jax.pmap, axis_name='devices')
def train_step_dynamic(state, batch, mask):
    """Handle variable-length batches with masking."""
    
    def loss_fn(model):
        logits = model(batch['image'], train=True)
        labels_onehot = jax.nn.one_hot(batch['label'], num_classes=10)
        
        # Mask out padding
        loss = optax.softmax_cross_entropy(logits, labels_onehot)
        loss = (loss * mask).sum() / mask.sum()
        
        return loss
    
    grad_fn = nnx.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.model)
    
    # Synchronize
    grads = jax.lax.pmean(grads, axis_name='devices')
    loss = jax.lax.pmean(loss, axis_name='devices')
    
    state.update(grads)
    return state, loss
```

## Performance Optimization

### Batch Size Selection

```python
# Rule of thumb:
total_batch_size = per_device_batch_size * num_devices

# Example with 8 GPUs:
per_device_batch_size = 32  # What fits in GPU memory
total_batch_size = 32 * 8 = 256

# Adjust learning rate for larger batch:
base_lr = 1e-3
scaled_lr = base_lr * (total_batch_size / 32)  # Linear scaling
```

### Monitoring Device Utilization

```python
# Check if all devices are being used
from jax import profiler

with profiler.trace("/tmp/jax-trace"):
    state, loss, metrics = train_step(state, batch_sharded)

# View trace in TensorBoard or Chrome's chrome://tracing
```

### Avoiding Host-Device Transfers

```python
# ❌ BAD: Transfers data to host on every step
for step in range(num_steps):
    state, loss, metrics = train_step(state, batch)
    print(f"Step {step}: Loss={float(loss[0])}")  # Transfer!

# ✅ GOOD: Only transfer occasionally
for step in range(num_steps):
    state, loss, metrics = train_step(state, batch)
    
    if step % 100 == 0:  # Only log every 100 steps
        print(f"Step {step}: Loss={float(loss[0])}")
```

## Common Pitfalls

### 1. Batch Size Not Divisible by Device Count

```python
# ❌ WRONG: 100 is not divisible by 8
batch_size = 100
num_devices = 8
# This will crash!

# ✅ CORRECT: Use divisible batch size
batch_size = 96  # 96 / 8 = 12 per device
```

### 2. Forgetting pmean

```python
# ❌ WRONG: Gradients not synchronized
@jax.pmap
def train_step(state, batch):
    grads = compute_gradients(state, batch)
    # Missing: grads = jax.lax.pmean(grads, axis_name='devices')
    state = state.apply_gradients(grads=grads)
    return state
# Result: Devices diverge!

# ✅ CORRECT: Always pmean gradients
@jax.pmap
def train_step(state, batch):
    grads = compute_gradients(state, batch)
    grads = jax.lax.pmean(grads, axis_name='devices')  # Synchronize!
    state = state.apply_gradients(grads=grads)
    return state
```

### 3. Incorrect Data Sharding

```python
# ❌ WRONG: No device dimension
batch = {'image': images}  # Shape: (128, 32, 32, 3)
state = train_step(state, batch)  # Error!

# ✅ CORRECT: Add device dimension
batch = shard_batch(batch, num_devices)  # Shape: (8, 16, 32, 32, 3)
state = train_step(state, batch)  # Works!
```

## Complete Example

**Data parallelism with pmap:**
- [`examples/distributed/data_parallel_pmap.py`](https://github.com/mlnomadpy/flaxdocs/tree/master/examples/distributed/data_parallel_pmap.py) - Complete training script with model replication, data sharding, gradient synchronization, and multi-device evaluation

## Next Steps

- **Larger Models?** → Try [FSDP for memory-efficient training](./fsdp-fully-sharded.md)
- **Complex Sharding?** → Learn about [SPMD with automatic sharding](./spmd-sharding.md)
- **Sequential Models?** → Explore [Pipeline Parallelism](./pipeline-parallelism.md)
