---
sidebar_position: 2
---

# Understanding State in NNX

Learn how Flax NNX manages different types of state in your neural networks - the key to proper training.

## The Three Types of State

Every NNX module can contain three kinds of state:

1. **Parameters** (`nnx.Param`) - Trainable weights updated by gradient descent
2. **Variables** (`nnx.Variable`) - Non-trainable state like running statistics
3. **RNG State** (`nnx.Rngs`) - Random number generators for dropout and initialization

Understanding this distinction is crucial for writing correct training code.

## Parameters: What Gets Trained

Parameters are the values your optimizer updates during training:

```python
from flax import nnx
import jax.numpy as jnp

class MyLayer(nnx.Module):
    def __init__(self, features: int, *, rngs: nnx.Rngs):
        # This is a parameter - it will be trained
        self.weight = nnx.Param(
            nnx.initializers.lecun_normal()(
                rngs.params(),
                (features, features)
            )
        )
        # This is also a parameter
        self.bias = nnx.Param(jnp.zeros((features,)))
    
    def __call__(self, x):
        return x @ self.weight.value + self.bias.value
```

**When to use**: Anything you want the optimizer to update - weights, biases, embeddings, etc.

## Variables: State That Doesn't Get Gradients

Variables store state that changes during training but isn't updated by gradients:

```python
class BatchNormExample(nnx.Module):
    def __init__(self, num_features: int, *, rngs: nnx.Rngs):
        # Parameters (trainable)
        self.scale = nnx.Param(jnp.ones((num_features,)))
        self.bias = nnx.Param(jnp.zeros((num_features,)))
        
        # Variables (not trainable, but updated manually)
        self.running_mean = nnx.Variable(jnp.zeros((num_features,)))
        self.running_var = nnx.Variable(jnp.ones((num_features,)))
    
    def __call__(self, x, *, train: bool = True):
        if train:
            # During training: use batch statistics
            mean = jnp.mean(x, axis=0)
            var = jnp.var(x, axis=0)
            
            # Update running statistics (manual update)
            momentum = 0.9
            self.running_mean.value = (
                momentum * self.running_mean.value + (1 - momentum) * mean
            )
            self.running_var.value = (
                momentum * self.running_var.value + (1 - momentum) * var
            )
        else:
            # During inference: use running statistics
            mean = self.running_mean.value
            var = self.running_var.value
        
        # Normalize
        x = (x - mean) / jnp.sqrt(var + 1e-5)
        return self.scale.value * x + self.bias.value
```

**When to use**: Running statistics, counters, cached values - things that need to persist but shouldn't receive gradients.

## RNG State: Randomness in Your Model

RNG state manages random number generation for reproducibility:

```python
class ModelWithDropout(nnx.Module):
    def __init__(self, features: int, *, rngs: nnx.Rngs):
        self.linear = nnx.Linear(features, features, rngs=rngs)
        self.dropout = nnx.Dropout(rate=0.5, rngs=rngs)
    
    def __call__(self, x, *, train: bool = True):
        x = self.linear(x)
        if train:
            x = self.dropout(x)
        return x

# Create model with separate RNG streams
model = ModelWithDropout(
    features=128,
    rngs=nnx.Rngs(
        params=0,   # For weight initialization
        dropout=1,  # For dropout masks
    )
)
```

**Why separate RNG streams?**
- **params**: Used once during initialization
- **dropout**: Used every forward pass during training
- Keeping them separate ensures reproducibility

## Extracting and Updating State

NNX provides utilities to extract and restore state:

```python
from flax import nnx

# Create a model
model = MyModel(rngs=nnx.Rngs(params=0))

# Extract all state
state = nnx.state(model)

# Extract only parameters
params = nnx.state(model, nnx.Param)

# Extract only variables
variables = nnx.state(model, nnx.Variable)

# Update model with new state
nnx.update(model, state)
```

This is essential for:
- **Checkpointing**: Save and load model state
- **Optimization**: Optimizers need to extract/update parameters
- **Distributed training**: Sync state across devices

## Practical Example: Training vs Inference

Here's how state management differs between training and inference:

```python
class CompleteModel(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        self.conv = nnx.Conv(1, 32, kernel_size=(3, 3), rngs=rngs)
        self.bn = nnx.BatchNorm(32, rngs=rngs)
        self.dropout = nnx.Dropout(0.5, rngs=rngs)
        self.linear = nnx.Linear(32 * 26 * 26, 10, rngs=rngs)
    
    def __call__(self, x, *, train: bool = True):
        x = self.conv(x)
        # BatchNorm behaves differently in train vs eval
        x = self.bn(x, use_running_average=not train)
        x = nnx.relu(x)
        # Dropout only active during training
        if train:
            x = self.dropout(x)
        x = x.reshape(x.shape[0], -1)
        return self.linear(x)

# Create model
model = CompleteModel(rngs=nnx.Rngs(params=0, dropout=1))

# Training mode (updates running stats, applies dropout)
logits_train = model(x, train=True)

# Inference mode (uses running stats, no dropout)
logits_eval = model(x, train=False)
```

## Common Pitfalls

❌ **Forgetting to pass `train=` flag**
```python
# Wrong - will use training mode during evaluation
predictions = model(test_data)

# Right - explicitly set eval mode
predictions = model(test_data, train=False)
```

❌ **Manually updating parameters (let optimizer do it)**
```python
# Wrong - don't manually update parameters
model.weight.value = model.weight.value - 0.01 * grads

# Right - use an optimizer
optimizer = nnx.Optimizer(model, optax.adam(0.001))
optimizer.update(grads)
```

❌ **Not extracting state for checkpointing**
```python
# Wrong - can't save the model object directly with JAX
jnp.save('model.npy', model)

# Right - extract state first
state = nnx.state(model)
# Then use proper checkpointing (see checkpointing guide)
```

## Key Takeaways

- **Parameters**: Trainable weights updated by optimizers via gradients
- **Variables**: Non-trainable state updated manually (running stats, counters)
- **RNG State**: Manages randomness for reproducibility
- Always pass `train=` flag to models that behave differently during training vs inference
- Use `nnx.state()` and `nnx.update()` to extract and restore state

## Next Steps

- [Simple Training Loop](../workflows/simple-training.md) - Put it all together
- [Checkpointing](../../basics/checkpointing.md) - Save and load models
