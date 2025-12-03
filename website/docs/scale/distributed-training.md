---
sidebar_position: 1
---

# Distributed Training with Flax

Learn how to scale your Flax models across multiple devices and hosts for faster training.

## Overview

Distributed training allows you to:

- **Train Larger Models**: Split models across multiple devices
- **Faster Training**: Process more data in parallel
- **Better Resource Utilization**: Use all available GPUs/TPUs

## JAX Parallelism Basics

JAX provides powerful primitives for distributed computation:

- `jax.pmap`: Parallel map across devices
- `jax.vmap`: Vectorized map (within device)
- `jax.jit`: Just-in-time compilation

## Data Parallelism with pmap

The simplest form of distributed training: replicate your model across devices.

### Basic Setup

```python
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax

# Check available devices
print(f"Available devices: {jax.devices()}")
print(f"Device count: {jax.device_count()}")

# Simple model
class SimpleModel(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(10)(x)
        return x
```

### Replicate Model Across Devices

```python
from flax import jax_utils

# Initialize model on one device
rng = jax.random.PRNGKey(0)
model = SimpleModel()
params = model.init(rng, jnp.ones([1, 784]))['params']

# Create optimizer
tx = optax.adam(learning_rate=1e-3)

# Create training state
state = train_state.TrainState.create(
    apply_fn=model.apply,
    params=params,
    tx=tx
)

# Replicate across devices
state = jax_utils.replicate(state)
```

### Parallel Training Step

```python
@jax.pmap
def train_step(state, batch):
    """Training step parallelized across devices."""
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, batch['image'])
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=batch['label']
        ).mean()
        return loss, logits
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    
    # Average gradients across devices
    grads = jax.lax.pmean(grads, axis_name='batch')
    loss = jax.lax.pmean(loss, axis_name='batch')
    
    state = state.apply_gradients(grads=grads)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == batch['label'])
    
    return state, loss, accuracy
```

### Data Loading for Multiple Devices

```python
def prepare_batch_for_devices(batch, num_devices):
    """Reshape batch for pmap by adding device dimension."""
    batch_size = batch['image'].shape[0]
    assert batch_size % num_devices == 0, "Batch size must be divisible by device count"
    
    # Reshape to (num_devices, batch_per_device, ...)
    per_device_batch_size = batch_size // num_devices
    
    batch['image'] = batch['image'].reshape(
        (num_devices, per_device_batch_size) + batch['image'].shape[1:]
    )
    batch['label'] = batch['label'].reshape(
        (num_devices, per_device_batch_size) + batch['label'].shape[1:]
    )
    
    return batch

# Training loop
num_devices = jax.device_count()
batch_size = 128 * num_devices  # Total batch size

for step in range(num_steps):
    batch = next(data_iterator)
    batch = prepare_batch_for_devices(batch, num_devices)
    
    state, loss, accuracy = train_step(state, batch)
    
    # Unreplicate for logging (take first device's values)
    if step % 100 == 0:
        print(f'Step {step}, Loss: {loss[0]:.4f}, Accuracy: {accuracy[0]:.4f}')
```

## Multi-Host Training

Scale beyond a single machine by distributing across multiple hosts.

### Initialization

```python
import jax
from jax.experimental import multihost_utils

# Initialize JAX distributed
jax.distributed.initialize()

# Get host/device information
print(f"Process index: {jax.process_index()}")
print(f"Process count: {jax.process_count()}")
print(f"Local devices: {jax.local_devices()}")
print(f"All devices: {jax.devices()}")
```

### Multi-Host Training Setup

```python
from flax import jax_utils
import numpy as np

def create_multi_host_data_iterator(dataset, global_batch_size):
    """Create data iterator for multi-host training."""
    process_id = jax.process_index()
    process_count = jax.process_count()
    local_batch_size = global_batch_size // process_count
    
    # Each host loads its portion of data
    start_idx = process_id * local_batch_size
    
    def data_generator():
        while True:
            # Load data specific to this host
            batch = get_batch(dataset, start_idx, local_batch_size)
            yield batch
    
    return data_generator()

# Training with multi-host
@jax.pmap
def multi_host_train_step(state, batch):
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, batch['image'])
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=batch['label']
        ).mean()
        return loss
    
    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(state.params)
    
    # Synchronize gradients across ALL hosts and devices
    grads = jax.lax.pmean(grads, axis_name='batch')
    state = state.apply_gradients(grads=grads)
    
    return state
```

## Model Parallelism

Split large models across devices when they don't fit on a single device.

### Using jax.pjit for Model Parallelism

```python
from jax.experimental import mesh_utils
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.experimental.pjit import pjit

# Create device mesh (2D: data parallel x model parallel)
devices = mesh_utils.create_device_mesh((2, 4))  # 2 data, 4 model parallel
mesh = Mesh(devices, axis_names=('data', 'model'))

# Define sharding specifications
data_sharding = NamedSharding(mesh, PartitionSpec('data', None))
model_sharding = NamedSharding(mesh, PartitionSpec(None, 'model'))

# Create large model with sharding
class LargeModel(nn.Module):
    @nn.compact
    def __call__(self, x):
        # This layer is sharded across model parallel axis
        x = nn.Dense(
            features=8192,
            kernel_init=nn.initializers.normal(stddev=0.02)
        )(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)
        return x

# Use pjit for sharded training
@pjit
def train_step_sharded(state, batch):
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, batch['image'])
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=batch['label']
        ).mean()
        return loss
    
    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state
```

## Pipeline Parallelism

Split model into stages and pipeline execution across devices.

```python
from flax import linen as nn

class PipelineModel(nn.Module):
    """Model split into pipeline stages."""
    
    def stage1(self, x):
        x = nn.Dense(512)(x)
        x = nn.relu(x)
        return x
    
    def stage2(self, x):
        x = nn.Dense(512)(x)
        x = nn.relu(x)
        return x
    
    def stage3(self, x):
        x = nn.Dense(10)(x)
        return x
    
    @nn.compact
    def __call__(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return x

# Manual pipeline execution (simplified)
def pipeline_train_step(state, batch, num_microbatches=4):
    """Execute training with pipeline parallelism."""
    microbatch_size = batch['image'].shape[0] // num_microbatches
    
    for i in range(num_microbatches):
        start_idx = i * microbatch_size
        end_idx = (i + 1) * microbatch_size
        
        microbatch = {
            'image': batch['image'][start_idx:end_idx],
            'label': batch['label'][start_idx:end_idx],
        }
        
        # Process microbatch through pipeline
        # (In practice, you'd use a pipeline library)
        state = train_step(state, microbatch)
    
    return state
```

## Gradient Accumulation

Simulate larger batch sizes when memory is limited.

```python
def train_step_with_accumulation(state, batch, num_accumulation_steps):
    """Train with gradient accumulation."""
    
    def compute_grads(params, batch):
        def loss_fn(params):
            logits = state.apply_fn({'params': params}, batch['image'])
            loss = optax.softmax_cross_entropy_with_integer_labels(
                logits=logits, labels=batch['label']
            ).mean()
            return loss
        
        grad_fn = jax.grad(loss_fn)
        return grad_fn(params)
    
    # Accumulate gradients
    accumulated_grads = jax.tree_map(lambda x: jnp.zeros_like(x), state.params)
    
    for i in range(num_accumulation_steps):
        micro_batch = get_micro_batch(batch, i)
        grads = compute_grads(state.params, micro_batch)
        accumulated_grads = jax.tree_map(
            lambda acc, g: acc + g / num_accumulation_steps,
            accumulated_grads,
            grads
        )
    
    state = state.apply_gradients(grads=accumulated_grads)
    return state
```

## Performance Optimization

### 1. Efficient Data Loading

```python
import tensorflow_datasets as tfds

def create_efficient_dataset(batch_size, num_devices):
    """Create efficient data pipeline."""
    
    # Use TensorFlow datasets for efficient loading
    ds = tfds.load('imagenet2012', split='train')
    
    # Prefetch and batch
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    
    # Convert to numpy iterator
    return tfds.as_numpy(ds)
```

### 2. Mixed Precision Training

```python
# Enable mixed precision globally
from jax import config
config.update('jax_default_matmul_precision', 'tensorfloat32')

# Or use explicit casting in model
class MixedPrecisionModel(nn.Module):
    @nn.compact
    def __call__(self, x):
        # Use bfloat16 for computation
        x = x.astype(jnp.bfloat16)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(10)(x)
        # Cast back to float32 for loss
        return x.astype(jnp.float32)
```

## Best Practices

### 1. Start Simple
- Begin with data parallelism
- Add model parallelism only for large models
- Profile before optimizing

### 2. Monitor Device Utilization
```python
# Check if devices are being used
from jax.profiler import trace

with trace("/tmp/jax-trace"):
    state, loss = train_step(state, batch)
```

### 3. Batch Size Guidelines
- Total batch size = per_device_batch_size × num_devices
- Larger batches may need learning rate scaling
- Monitor validation metrics when changing batch size

### 4. Synchronization Points
- Minimize host-device transfers
- Use asynchronous checkpointing
- Reduce frequency of metric logging

## Troubleshooting

### Out of Memory
- Reduce batch size per device
- Use gradient accumulation
- Enable mixed precision training
- Use model parallelism for large models

### Slow Training
- Check device utilization
- Profile with JAX profiler
- Ensure data loading isn't a bottleneck
- Use `jax.jit` and `jax.pmap` appropriately

### Communication Overhead
- Increase computation per communication
- Use gradient accumulation
- Optimize network topology

## Next Steps

