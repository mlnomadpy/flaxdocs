---
sidebar_position: 0
title: Flax NNX Fundamentals
description: Learn the core Flax NNX concepts for building neural networks in JAX, including modules, parameters, and state management that power every model type.
keywords: [Flax NNX, JAX neural networks, NNX modules, model fundamentals, parameters, state management, deep learning basics, nnx.Module]
---

# Fundamentals

Master the core concepts of Flax NNX - how to build and manage neural network models.

:::tip What you'll learn
- How `nnx.Module` works as a stateful Python class that holds parameters
- The role of `nnx.Param` and why parameters are wrapped explicitly
- How NNX manages trainable parameters, non-trainable variables, and RNG state
- Why `nnx.Rngs` makes initialization reproducible
- How these fundamentals carry over to vision, text, and advanced architectures
:::

## What You'll Learn

In this section, you'll learn the foundational concepts that apply to all Flax NNX models, regardless of whether you're building vision models, text models, or anything else.

### Core Concepts

**[Your First Model](./your-first-model.md)** - Start here!  
Build your first neural network from scratch. Learn about modules, parameters, and the basic structure of NNX models.

**[Understanding State](./understanding-state.md)**  
Learn how NNX manages different types of state: trainable parameters, non-trainable variables, and RNG state.

## Why Start Here?

These fundamentals are the building blocks for everything else:
- Every model type (vision, text, etc.) uses the same module system
- Understanding state is crucial for proper training and checkpointing
- These concepts translate directly to advanced architectures

## Quick Example

Here's what you'll be able to build after this section:

```python
from flax import nnx
import jax.numpy as jnp

# Define a model
class MyModel(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        self.layer1 = nnx.Linear(784, 256, rngs=rngs)
        self.layer2 = nnx.Linear(256, 10, rngs=rngs)
    
    def __call__(self, x):
        x = self.layer1(x)
        x = nnx.relu(x)
        return self.layer2(x)

# Create and use it
model = MyModel(rngs=nnx.Rngs(params=0))
x = jnp.ones((32, 784))
output = model(x)  # Shape: (32, 10)
```

Simple, explicit, and powerful!

## Next steps

- [Your First Model](/basics/fundamentals/your-first-model) - Build a linear layer and MLP from scratch
- [Understanding State](/basics/fundamentals/understanding-state) - Learn how NNX tracks parameters, variables, and RNGs
- [Training Workflows](/basics/workflows) - Train the models you build here
