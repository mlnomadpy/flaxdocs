---
sidebar_position: 2
---

# SPMD: Automatic Sharding with jax.jit

Learn how to use JAX's modern automatic sharding API for flexible, efficient distributed training with Flax NNX.

## Overview

**SPMD (Single Program Multiple Data)** is JAX's modern approach to parallelism that offers more flexibility than `pmap`. Instead of explicitly mapping functions across devices, you specify *how data should be sharded* and let the compiler figure out the rest.

Key advantages:
- 🎯 **Declarative**: Specify *what* to shard, not *how*
- 🚀 **Flexible**: Mix data and model parallelism freely
- ⚡ **Optimized**: Compiler generates efficient communication
- 🔧 **Composable**: Easy to experiment with different strategies

## When to Use SPMD Sharding

✅ **Use SPMD when:**
- You want more control than `pmap` provides
- Your model benefits from model parallelism (tensor/pipeline)
- You need complex sharding patterns
- You're writing new JAX code (modern best practice)
- You want the best performance

❌ **Use pmap instead when:**
- You only need simple data parallelism
- You have existing `pmap` code that works
- You're prototyping and want simplicity

## Core Concepts

### 1. Device Mesh

A **mesh** is a logical arrangement of devices with named axes:

```python
from jax.sharding import Mesh
from jax.experimental import mesh_utils

# Create a 2D mesh: 4 devices for data, 2 for model parallelism
devices = mesh_utils.create_device_mesh((4, 2))
mesh = Mesh(devices, axis_names=('data', 'model'))

# Shape: (4, 2) - 8 total devices arranged in 2D grid
# Axis 'data': 4 devices (rows)
# Axis 'model': 2 devices (columns)
```

**Mesh topologies:**

```python
# 1D mesh - pure data parallelism
mesh = Mesh(devices, axis_names=('data',))  # Shape: (8,)

# 2D mesh - hybrid parallelism
mesh = Mesh(devices, axis_names=('data', 'model'))  # Shape: (4, 2)

# 3D mesh - for very large scale
mesh = Mesh(devices, axis_names=('data', 'model', 'pipeline'))  # Shape: (2, 2, 2)
```

### 2. PartitionSpec

A **PartitionSpec** describes how each dimension of a tensor is sharded:

```python
from jax.sharding import PartitionSpec as P

# Examples:
P('data')           # Shard 1st dimension along 'data' axis
P(None, 'model')    # Replicate 1st dim, shard 2nd along 'model'
P('data', 'model')  # Shard both dimensions
P()                 # Replicate completely (no sharding)
```

**Visual example:**

```python
# Tensor shape: (128, 512)
# Mesh: (8, 1) with axis 'data'

P('data', None)
# Result: (128, 512) → 8 devices, each gets (16, 512)
# Split along batch dimension

P(None, 'data')
# Result: (128, 512) → 8 devices, each gets (128, 64)
# Split along feature dimension

P('data', 'model')  # With mesh (4, 2)
# Result: (128, 512) → each device gets (32, 256)
# Split both dimensions
```

### 3. NamedSharding

Combines a mesh and PartitionSpec:

```python
from jax.sharding import NamedSharding

mesh = Mesh(devices, axis_names=('data', 'model'))

# Define shardings
data_sharding = NamedSharding(mesh, P('data', None))
model_sharding = NamedSharding(mesh, P(None, 'model'))
replicated_sharding = NamedSharding(mesh, P())

# Apply to arrays
x_sharded = jax.device_put(x, data_sharding)
w_sharded = jax.device_put(w, model_sharding)
```

## Implementation Guide

### Step 1: Create Device Mesh

```python
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import mesh_utils

# Get available devices
num_devices = jax.local_device_count()
print(f"Devices: {num_devices}")

# Strategy 1: Pure data parallelism (like pmap)
devices = mesh_utils.create_device_mesh((num_devices,))
mesh = Mesh(devices, axis_names=('data',))

# Strategy 2: Hybrid (4 data, 2 model)
# Requires 8 devices
if num_devices == 8:
    devices = mesh_utils.create_device_mesh((4, 2))
    mesh = Mesh(devices, axis_names=('data', 'model'))

# Strategy 3: FSDP-style (all devices in one axis)
devices = mesh_utils.create_device_mesh((num_devices,))
mesh = Mesh(devices, axis_names=('fsdp',))
```

### Step 2: Define Sharding Strategies

```python
def create_sharding_specs(mesh):
    """Create common sharding patterns."""
    
    return {
        # Data parallelism: shard batch dimension
        'data_parallel': NamedSharding(mesh, P('data', None)),
        
        # Model parallelism: shard feature dimension
        'model_parallel': NamedSharding(mesh, P(None, 'model')),
        
        # 2D sharding: shard both dimensions
        'hybrid': NamedSharding(mesh, P('data', 'model')),
        
        # Replicated: no sharding
        'replicated': NamedSharding(mesh, P()),
    }

shardings = create_sharding_specs(mesh)
```

### Step 3: Shard Model Parameters

```python
from flax import nnx

# Initialize model
rngs = nnx.Rngs(0)
model = MyTransformer(d_model=512, num_layers=6, rngs=rngs)

# Extract parameters
graphdef, params = nnx.split(model)

# Shard parameters
def shard_params(params, shardings):
    """Apply sharding to model parameters."""
    
    def shard_array(path, array):
        path_str = '/'.join(str(p) for p in path)
        
        # Shard large weight matrices
        if 'kernel' in path_str and array.shape[0] > 256:
            return jax.device_put(array, shardings['model_parallel'])
        else:
            # Replicate small parameters (biases, norms)
            return jax.device_put(array, shardings['replicated'])
    
    return jax.tree_util.tree_map_with_path(shard_array, params)

# Apply sharding
with mesh:
    params_sharded = shard_params(params, shardings)

# Reconstruct model
model = nnx.merge(graphdef, params_sharded)
```

### Step 4: Create Sharded Training Step

```python
from functools import partial

def create_train_step(mesh, shardings):
    """Create training step with automatic sharding."""
    
    @partial(
        jax.jit,
        # Inputs are already sharded
        in_shardings=(shardings['replicated'], shardings['data_parallel']),
        # Outputs should be replicated for logging
        out_shardings=(shardings['replicated'], shardings['replicated'], shardings['replicated'])
    )
    def train_step(state: nnx.Optimizer, batch: Dict):
        """
        Training step with automatic sharding propagation.
        
        JAX compiler will:
        1. Analyze sharding of inputs
        2. Propagate shardings through computation
        3. Insert collectives (all-reduce, all-gather) as needed
        4. Optimize communication patterns
        """
        
        def loss_fn(model):
            logits = model(batch['input_ids'])
            labels_onehot = jax.nn.one_hot(batch['label'], num_classes=10)
            loss = optax.softmax_cross_entropy(logits, labels_onehot).mean()
            
            predictions = jnp.argmax(logits, axis=-1)
            accuracy = jnp.mean(predictions == batch['label'])
            
            return loss, {'accuracy': accuracy}
        
        # Compute gradients
        grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
        (loss, metrics), grads = grad_fn(state.model)
        
        # Update (compiler handles synchronization)
        state.update(grads)
        
        return state, loss, metrics
    
    return train_step
```

### Step 5: Training Loop

```python
# Create optimizer
optimizer = optax.adam(learning_rate=1e-3)
state = nnx.Optimizer(model, optimizer)

# Create training step
train_step = create_train_step(mesh, shardings)

# Training loop
with mesh:
    for epoch in range(num_epochs):
        for batch in data_loader:
            # Shard input data
            batch_sharded = jax.tree.map(
                lambda x: jax.device_put(x, shardings['data_parallel']),
                batch
            )
            
            # Training step (automatic sharding propagation)
            state, loss, metrics = train_step(state, batch_sharded)
            
            # Log (outputs are replicated, can use directly)
            print(f"Loss: {float(loss):.4f}, Acc: {float(metrics['accuracy']):.4f}")
```

## Sharding Strategies

### Strategy 1: Data Parallelism (SPMD-style)

Equivalent to `pmap`, but using modern sharding API:

```python
# Create 1D mesh
mesh = Mesh(devices, axis_names=('data',))

# Shard batch dimension only
data_sharding = NamedSharding(mesh, P('data'))

# All parameters replicated
param_sharding = NamedSharding(mesh, P())

# Apply sharding
batch_sharded = jax.device_put(batch, data_sharding)
params_replicated = jax.device_put(params, param_sharding)
```

**When to use:**
- Simple data parallelism
- Model fits on single device
- Want modern API instead of pmap

### Strategy 2: Tensor Parallelism

Shard large weight matrices across devices:

```python
# Create 1D mesh for model parallelism
mesh = Mesh(devices, axis_names=('model',))

# Shard weight matrices along second dimension
def shard_weights(params):
    def shard_array(path, array):
        if 'kernel' in str(path) and len(array.shape) >= 2:
            # Shard: (input_dim, output_dim) along output_dim
            return jax.device_put(
                array,
                NamedSharding(mesh, P(None, 'model'))
            )
        else:
            return jax.device_put(
                array,
                NamedSharding(mesh, P())
            )
    
    return jax.tree_util.tree_map_with_path(shard_array, params)
```

**Example with Transformer:**

```python
# Query projection: (d_model, d_model)
# Without sharding: Each device has (512, 512)
# With P(None, 'model') and 8 devices: Each device has (512, 64)

# This reduces memory and can speed up large matrix multiplications
```

**When to use:**
- Very large layers (e.g., 8K hidden size)
- Model doesn't fit on single device
- Have fast interconnect (NVLink, ICI)

### Strategy 3: 2D Parallelism

Combine data and model parallelism:

```python
# Create 2D mesh: 4 data parallel, 2 model parallel
devices = mesh_utils.create_device_mesh((4, 2))
mesh = Mesh(devices, axis_names=('data', 'model'))

# Data: shard batch along 'data'
data_sharding = NamedSharding(mesh, P('data'))

# Weights: shard along 'model'
weight_sharding = NamedSharding(mesh, P(None, 'model'))

# Activations: can be 2D sharded
activation_sharding = NamedSharding(mesh, P('data', 'model'))
```

**Communication pattern:**

```python
# Matrix multiplication: (batch, d_model) @ (d_model, d_ff)
# Input: P('data', 'model')  - sharded both dimensions
# Weight: P(None, 'model')   - sharded along output
# Output: P('data', 'model') - sharded both dimensions

# Compiler inserts all-reduce along 'model' axis for correctness
```

**When to use:**
- Very large scale (100B+ parameters)
- Both batch and model dimensions are huge
- Have many devices (64+)

### Strategy 4: FSDP-Style Sharding

Shard all parameters to reduce memory:

```python
# 1D mesh for FSDP
mesh = Mesh(devices, axis_names=('fsdp',))

# Shard parameters along first dimension
fsdp_sharding = NamedSharding(mesh, P('fsdp'))

def shard_fsdp(params):
    """Shard parameters FSDP-style."""
    def shard_array(path, array):
        # Shard large parameters
        if array.size > 1024:
            return jax.device_put(array, fsdp_sharding)
        else:
            # Replicate small (biases, layer norms)
            return jax.device_put(array, NamedSharding(mesh, P()))
    
    return jax.tree_util.tree_map_with_path(shard_array, params)
```

**Memory savings:**

```python
# Without FSDP: 400MB model × 8 devices = 3.2GB total
# With FSDP: 400MB / 8 per device = 50MB per device, 400MB total
```

See [FSDP guide](./fsdp-fully-sharded.md) for details.

## Advanced Topics

### Inspecting Sharding

```python
# Check how an array is sharded
x = jax.device_put(x, sharding)

print(f"Shape: {x.shape}")
print(f"Sharding: {x.sharding}")
print(f"Devices: {x.sharding.device_set}")

# Visualize sharding
jax.debug.visualize_array_sharding(x)
```

### Resharding Arrays

JAX automatically reshard arrays when needed:

```python
# Start with data-parallel sharding
x_data = jax.device_put(x, NamedSharding(mesh, P('data')))

# Function expects model-parallel sharding
@jax.jit
def process(x):
    # Automatic reshard!
    y = jax.device_put(x, NamedSharding(mesh, P('model')))
    return y * 2

# JAX inserts collective operations to reshard
result = process(x_data)
```

### Constraint Propagation

```python
@jax.jit
def compute(x, w):
    # JAX infers output sharding based on inputs
    # If x: P('data', None) and w: P(None, 'model')
    # Then output: P('data', 'model')
    return jnp.dot(x, w)

# No need to specify output sharding!
```

### Manual Sharding Constraints

Force specific sharding at any point:

```python
from jax.experimental.shard_map import shard_map

@jax.jit
def train_step(params, batch):
    # Ensure batch is data-parallel
    batch = jax.lax.with_sharding_constraint(
        batch,
        NamedSharding(mesh, P('data'))
    )
    
    # Computation...
    logits = model(params, batch)
    
    # Ensure logits are replicated before loss
    logits = jax.lax.with_sharding_constraint(
        logits,
        NamedSharding(mesh, P())
    )
    
    return compute_loss(logits, batch['label'])
```

## Performance Considerations

### Communication Overhead

Different sharding patterns have different communication costs:

```python
# Data parallelism: O(parameters) all-reduce per step
# Good for small models, large batches

# Tensor parallelism: O(activations) all-reduce per layer
# Good for large models, but more communication

# 2D parallelism: Reduced communication on both axes
# Best for very large scale

# FSDP: All-gather before each layer, reduce-scatter after
# Good memory/communication trade-off
```

### Choosing Mesh Shape

```python
# Rule of thumb for 2D mesh:
# - More data parallelism → better batch throughput
# - More model parallelism → larger models fit

# Example with 64 devices:
mesh_shapes = [
    (64, 1),   # Pure data parallel - fastest for small models
    (32, 2),   # Light model parallelism
    (16, 4),   # Balanced
    (8, 8),    # Balanced
    (1, 64),   # Pure model parallel - for huge layers
]

# Choose based on:
# - Model size (does it fit?)
# - Batch size requirements
# - Communication bandwidth
```

### Profiling Sharding

```python
# Profile to see communication patterns
with jax.profiler.trace("/tmp/jax-trace"):
    state, loss = train_step(state, batch)

# Look for:
# - CollectivePermute (all-to-all)
# - AllReduce (gradient sync)
# - AllGather (FSDP)
# - ReduceScatter (FSDP)
```

## Comparison: pmap vs SPMD

| Feature | pmap | SPMD (jax.jit + sharding) |
|---------|------|---------------------------|
| **Ease of use** | Simple | Moderate |
| **Flexibility** | Limited | Very flexible |
| **Performance** | Good | Better (optimized collectives) |
| **Data parallelism** | ✅ Native | ✅ Via P('data') |
| **Model parallelism** | ❌ Manual | ✅ Via P('model') |
| **FSDP-style** | ❌ Hard | ✅ Easy |
| **Mixed strategies** | ❌ | ✅ |
| **Modern JAX** | Legacy | ✅ Recommended |

## Example: Complete Script

See `examples/17_sharding_spmd.py` in the repository for a complete example with:

- ✅ Device mesh creation
- ✅ Parameter sharding strategies
- ✅ Automatic sharding propagation
- ✅ Training loop with sharding
- ✅ Memory analysis

## Next Steps

- **Save Memory?** → Try [FSDP for large models](./fsdp-fully-sharded.md)
- **Sequential Models?** → Learn [Pipeline Parallelism](./pipeline-parallelism.md)
- **Simple Approach?** → Start with [Data Parallelism](./data-parallelism.md)
