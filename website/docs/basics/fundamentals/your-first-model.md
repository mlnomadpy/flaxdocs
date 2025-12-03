---
sidebar_position: 1
---

# Your First Model

Learn to build your first neural network in Flax NNX - a simple linear layer that you can understand completely.

## What is a Module?

In Flax NNX, a **module** is just a Python class that inherits from `nnx.Module`. Think of it as a building block:

- **Modules hold state**: Parameters like weights and biases live inside the module
- **Modules are mutable**: You can update them in-place during training
- **Modules are explicit**: No hidden magic - everything is visible

```python
import jax
import jax.numpy as jnp
from flax import nnx

class SimpleLinear(nnx.Module):
    """A single linear layer: y = Wx + b"""
    
    def __init__(self, in_features: int, out_features: int, *, rngs: nnx.Rngs):
        # Initialize the weight matrix
        self.weight = nnx.Param(
            nnx.initializers.lecun_normal()(
                rngs.params(), 
                (in_features, out_features)
            )
        )
        
        # Initialize the bias vector
        self.bias = nnx.Param(jnp.zeros((out_features,)))
    
    def __call__(self, x: jax.Array) -> jax.Array:
        # Simple matrix multiplication + bias
        return x @ self.weight.value + self.bias.value

# Create a model: 784 inputs -> 10 outputs (like MNIST)
model = SimpleLinear(
    in_features=784, 
    out_features=10, 
    rngs=nnx.Rngs(params=0)  # Seed for reproducibility
)

# Use it
x = jnp.ones((32, 784))  # Batch of 32 examples
logits = model(x)  # Shape: (32, 10)
print(f"Output shape: {logits.shape}")
```

## Understanding the Code

### Why `rngs` is keyword-only (`*,` syntax)
The `*` forces you to write `rngs=...` explicitly, which prevents bugs and makes code more readable.

### Why wrap in `nnx.Param`
This tells NNX "this is trainable" so optimizers know to update it. Without wrapping, it would be a static attribute that never changes.

### Why `.value` in the forward pass
`self.weight` is a `Param` object that wraps the actual array. Use `.value` to get the underlying JAX array for computations.

## Building a Multi-Layer Perceptron

Now let's add depth - multiple layers with activation functions:

```python
class SimpleMLP(nnx.Module):
    """Multi-layer perceptron with 2 hidden layers"""
    
    def __init__(
        self, 
        in_features: int,
        hidden_features: int,
        out_features: int,
        *,
        rngs: nnx.Rngs
    ):
        # Three layers: input -> hidden -> hidden -> output
        self.layer1 = nnx.Linear(in_features, hidden_features, rngs=rngs)
        self.layer2 = nnx.Linear(hidden_features, hidden_features, rngs=rngs)
        self.layer3 = nnx.Linear(hidden_features, out_features, rngs=rngs)
    
    def __call__(self, x: jax.Array) -> jax.Array:
        # Layer 1 with activation
        x = self.layer1(x)
        x = nnx.relu(x)
        
        # Layer 2 with activation
        x = self.layer2(x)
        x = nnx.relu(x)
        
        # Output layer (no activation - raw logits)
        x = self.layer3(x)
        return x

# Create a 3-layer MLP
model = SimpleMLP(
    in_features=784,
    hidden_features=256,
    out_features=10,
    rngs=nnx.Rngs(params=42)
)
```

### Why ReLU?
ReLU (Rectified Linear Unit) is the most common activation function:
- Fast to compute: `max(0, x)`
- Prevents vanishing gradients
- Introduces non-linearity (without it, stacking layers is pointless)

### Why no activation on the last layer?
The output layer produces "logits" (raw scores). We'll apply softmax later during loss computation for numerical stability.

## Common Mistakes to Avoid

❌ **Forgetting `.value`**
```python
# Wrong - this multiplies Param objects, not arrays
return x @ self.weight + self.bias

# Right - extract the arrays first
return x @ self.weight.value + self.bias.value
```

❌ **Creating RNGs inside modules**
```python
# Wrong - non-reproducible!
class BadModel(nnx.Module):
    def __init__(self):
        rngs = nnx.Rngs(params=0)
        self.layer = nnx.Linear(10, 10, rngs=rngs)

# Right - pass RNGs from outside
class GoodModel(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        self.layer = nnx.Linear(10, 10, rngs=rngs)
```

❌ **Using zeros for weight initialization**
```python
# Wrong - all neurons will compute the same thing!
self.weight = nnx.Param(jnp.zeros(shape))

# Right - use proper initialization
self.weight = nnx.Param(
    nnx.initializers.lecun_normal()(rngs.params(), shape)
)
```

## Next Steps

You now understand the basics of building models in Flax NNX! Next, learn:

- [Training Your First Model](../workflows/simple-training.md) - Write a complete training loop
- [Computer Vision Models](../vision/simple-cnn.md) - Build CNNs for image classification
- [Text Models](../text/simple-transformer.md) - Build transformers for language

## Complete Example

See the full runnable code in [`examples/01_basic_model_definition.py`](https://github.com/mlnomadpy/flaxdocs/tree/master/examples/01_basic_model_definition.py).
