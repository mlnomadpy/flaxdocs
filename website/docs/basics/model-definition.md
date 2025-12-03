---
sidebar_position: 1
---

# Building Models in Flax NNX

Learn how to define neural network models in Flax NNX from first principles. This guide teaches you the concepts behind NNX's module system, not just how to write code.

## Understanding the NNX Module System

### What is a Module?

In Flax NNX, a **module** is any Python class that inherits from `nnx.Module`. Unlike functional programming approaches, NNX uses object-oriented design where:

- **Modules hold state**: Parameters, batch norm statistics, RNG keys live inside the module
- **Modules are mutable**: You can update parameters in-place during training
- **Modules are explicit**: No hidden global state or magic - everything is visible

This design makes NNX feel like PyTorch but with JAX's performance benefits.

### The Three Types of State

Every NNX module manages three kinds of state:

1. **Parameters** (`nnx.Param`): Trainable weights that gradient descent updates
2. **Variables** (`nnx.Variable`): Non-trainable state like batch norm running means
3. **RNG State** (`nnx.Rngs`): Random number generators for dropout, initialization, etc.

Understanding this distinction is crucial for proper training loops.

## Your First Model: A Simple Linear Layer

Let's build the simplest possible model - a single linear layer - and understand every part:

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
```

### Key Concepts Explained

**Why `rngs` is keyword-only (`*,` syntax)**:
- Forces you to explicitly pass RNG state
- Prevents accidental bugs from positional arguments
- Makes code more readable: you always see `rngs=...`

**Why wrap in `nnx.Param`**:
- Tells NNX "this is trainable" so optimizers can find it
- Without wrapping, it would be a static attribute
- You can also use `nnx.Variable` for non-trainable state

**Why `.value` in the forward pass**:
- `self.weight` is a `Param` object that wraps the actual array
- `.value` extracts the underlying JAX array
- This indirection lets NNX track and manage state

## Building a Multi-Layer Perceptron (MLP)

Now let's build something more realistic - a multi-layer network with activation functions:

```python
class MLP(nnx.Module):
    """Multi-layer perceptron with configurable depth and width"""
    
    def __init__(
        self, 
        in_features: int,
        hidden_features: int,
        out_features: int,
        num_layers: int,
        *,
        rngs: nnx.Rngs
    ):
        # First layer: input -> hidden
        self.layers = [
            nnx.Linear(in_features, hidden_features, rngs=rngs)
        ]
        
        # Middle layers: hidden -> hidden
        for _ in range(num_layers - 2):
            self.layers.append(
                nnx.Linear(hidden_features, hidden_features, rngs=rngs)
            )
        
        # Final layer: hidden -> output
        self.layers.append(
            nnx.Linear(hidden_features, out_features, rngs=rngs)
        )
    
    def __call__(self, x: jax.Array) -> jax.Array:
        # Apply all layers with ReLU activation (except last)
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = nnx.relu(x)
        
        # Last layer with no activation (logits)
        return self.layers[-1](x)

# Create a 3-layer MLP
model = MLP(
    in_features=784,
    hidden_features=256,
    out_features=10,
    num_layers=3,
    rngs=nnx.Rngs(params=42)
)
```

### Why We Store Layers in a List

Notice we use `self.layers = [...]` instead of `self.layer1`, `self.layer2`, etc. This pattern:

- **Scales to any depth**: Works for 3 or 300 layers without code changes
- **Enables dynamic architectures**: Can change depth based on input
- **Simplifies forward pass**: Just loop through layers instead of manual chaining

NNX automatically tracks all modules in lists, tuples, and dicts!

## Convolutional Networks for Vision

For image data, we need spatial operations. Let's build a simple CNN:

```python
class SimpleCNN(nnx.Module):
    """Convolutional network for image classification"""
    
    def __init__(self, num_classes: int, *, rngs: nnx.Rngs):
        # Convolutional layers extract visual features
        self.conv1 = nnx.Conv(
            in_features=1,      # Grayscale input
            out_features=32,    # 32 feature maps
            kernel_size=(3, 3), # 3x3 filters
            rngs=rngs
        )
        
        self.conv2 = nnx.Conv(
            in_features=32,
            out_features=64,
            kernel_size=(3, 3),
            rngs=rngs
        )
        
        # Dense layers for classification
        self.dense1 = nnx.Linear(64 * 5 * 5, 128, rngs=rngs)
        self.dense2 = nnx.Linear(128, num_classes, rngs=rngs)
    
    def __call__(self, x: jax.Array) -> jax.Array:
        # x shape: (batch, height, width, channels)
        # For MNIST: (batch, 28, 28, 1)
        
        # First conv block: conv -> relu -> maxpool
        x = self.conv1(x)
        x = nnx.relu(x)
        x = nnx.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        # Shape: (batch, 14, 14, 32)
        
        # Second conv block
        x = self.conv2(x)
        x = nnx.relu(x)
        x = nnx.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        # Shape: (batch, 5, 5, 64)
        
        # Flatten spatial dimensions
        x = x.reshape(x.shape[0], -1)  # (batch, 64*5*5)
        
        # Classification head
        x = self.dense1(x)
        x = nnx.relu(x)
        x = self.dense2(x)
        
        return x  # Logits for each class

# Create model for MNIST (28x28 grayscale images, 10 classes)
model = SimpleCNN(num_classes=10, rngs=nnx.Rngs(params=0))

# Use it
images = jnp.ones((32, 28, 28, 1))  # Batch of 32 MNIST images
logits = model(images)  # Shape: (32, 10)
```

### Understanding Convolutions

**Why use convolutions for images?**
- **Parameter sharing**: Same filters scan entire image (fewer parameters)
- **Translation invariance**: Detects features regardless of position
- **Spatial hierarchy**: Early layers find edges, later layers find objects

**Shape tracking is critical**:
- Conv layers preserve spatial structure (height, width)
- Pooling reduces spatial dimensions (downsampling)
- Must flatten before dense layers
- Common mistake: wrong flatten size → shape errors

## Residual Networks (ResNets)

Modern architectures use skip connections to train very deep networks:

```python
class ResidualBlock(nnx.Module):
    """A single residual block: out = F(x) + x"""
    
    def __init__(self, features: int, *, rngs: nnx.Rngs):
        # Two conv layers with batch norm
        self.conv1 = nnx.Conv(
            in_features=features,
            out_features=features,
            kernel_size=(3, 3),
            padding='SAME',  # Keep spatial dimensions
            rngs=rngs
        )
        self.bn1 = nnx.BatchNorm(num_features=features, rngs=rngs)
        
        self.conv2 = nnx.Conv(
            in_features=features,
            out_features=features,
            kernel_size=(3, 3),
            padding='SAME',
            rngs=rngs
        )
        self.bn2 = nnx.BatchNorm(num_features=features, rngs=rngs)
    
    def __call__(self, x: jax.Array, *, train: bool = True) -> jax.Array:
        # Save input for skip connection
        residual = x
        
        # First conv block
        out = self.conv1(x)
        out = self.bn1(out, use_running_average=not train)
        out = nnx.relu(out)
        
        # Second conv block
        out = self.conv2(out)
        out = self.bn2(out, use_running_average=not train)
        
        # Add skip connection before final activation
        out = out + residual
        out = nnx.relu(out)
        
        return out

class ResNet(nnx.Module):
    """Complete ResNet architecture"""
    
    def __init__(
        self, 
        num_classes: int, 
        num_blocks: int = 3,
        *, 
        rngs: nnx.Rngs
    ):
        # Initial conv to increase channels
        self.conv1 = nnx.Conv(
            in_features=3,  # RGB input
            out_features=64,
            kernel_size=(7, 7),
            strides=(2, 2),
            padding='SAME',
            rngs=rngs
        )
        self.bn1 = nnx.BatchNorm(num_features=64, rngs=rngs)
        
        # Stack of residual blocks
        self.blocks = [
            ResidualBlock(64, rngs=rngs) 
            for _ in range(num_blocks)
        ]
        
        # Classification head
        self.fc = nnx.Linear(64, num_classes, rngs=rngs)
    
    def __call__(self, x: jax.Array, *, train: bool = True) -> jax.Array:
        # Initial processing
        x = self.conv1(x)
        x = self.bn1(x, use_running_average=not train)
        x = nnx.relu(x)
        x = nnx.max_pool(x, window_shape=(3, 3), strides=(2, 2))
        
        # Process through residual blocks
        for block in self.blocks:
            x = block(x, train=train)
        
        # Global average pooling
        x = jnp.mean(x, axis=(1, 2))  # Average over spatial dims
        
        # Classification
        return self.fc(x)
```

### Key ResNet Concepts

**Why skip connections matter**:
- **Gradient flow**: Gradients can bypass layers during backprop
- **Identity mapping**: Network can learn to skip bad layers
- **Enables depth**: Can train 100+ layer networks

**Batch normalization complexity**:
- Has two modes: training vs inference
- During training: normalize by batch statistics
- During inference: use running averages
- Must pass `train=` flag correctly!

## Transformer Architecture

The backbone of modern NLP:

```python
class MultiHeadAttention(nnx.Module):
    """Self-attention with multiple heads"""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        rngs: nnx.Rngs
    ):
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5  # Scaling factor
        
        # Linear projections for Q, K, V
        self.qkv = nnx.Linear(embed_dim, embed_dim * 3, rngs=rngs)
        self.proj = nnx.Linear(embed_dim, embed_dim, rngs=rngs)
    
    def __call__(self, x: jax.Array, mask: jax.Array | None = None) -> jax.Array:
        batch, seq_len, embed_dim = x.shape
        
        # Project to queries, keys, values
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch, seq_len, 3, self.num_heads, self.head_dim)
        qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))  # (3, batch, heads, seq, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention scores: Q @ K^T / sqrt(d_k)
        attn = (q @ jnp.swapaxes(k, -2, -1)) * self.scale
        
        # Apply causal mask if provided
        if mask is not None:
            attn = jnp.where(mask, attn, float('-inf'))
        
        # Softmax and weighted sum
        attn = jax.nn.softmax(attn, axis=-1)
        out = attn @ v
        
        # Reshape and project
        out = jnp.transpose(out, (0, 2, 1, 3))  # (batch, seq, heads, head_dim)
        out = out.reshape(batch, seq_len, embed_dim)
        return self.proj(out)

class TransformerBlock(nnx.Module):
    """A complete transformer block with attention and FFN"""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_dim: int,
        *,
        rngs: nnx.Rngs
    ):
        # Multi-head self-attention
        self.attention = MultiHeadAttention(embed_dim, num_heads, rngs=rngs)
        
        # Feed-forward network
        self.mlp = nnx.Sequential(
            nnx.Linear(embed_dim, mlp_dim, rngs=rngs),
            nnx.gelu,
            nnx.Linear(mlp_dim, embed_dim, rngs=rngs),
        )
        
        # Layer normalization
        self.norm1 = nnx.LayerNorm(embed_dim, rngs=rngs)
        self.norm2 = nnx.LayerNorm(embed_dim, rngs=rngs)
    
    def __call__(self, x: jax.Array, mask: jax.Array | None = None) -> jax.Array:
        # Attention with residual connection
        x = x + self.attention(self.norm1(x), mask)
        
        # FFN with residual connection
        x = x + self.mlp(self.norm2(x))
        
        return x
```

### Understanding Attention

**Why attention works**:
- **Dynamic routing**: Each token can attend to any other token
- **Context aggregation**: Combines relevant information from entire sequence
- **Position-agnostic**: Same mechanism for any position

**Key implementation details**:
- **Scaling by sqrt(d_k)**: Prevents softmax saturation
- **Multiple heads**: Different heads learn different patterns
- **Pre-norm vs post-norm**: Modern transformers use pre-norm (normalize before sublayer)

## Best Practices

### Initialization Matters

```python
# Good: Use proper initializers
self.weight = nnx.Param(
    nnx.initializers.lecun_normal()(rngs.params(), shape)
)

# Bad: Don't use zeros for weights
self.weight = nnx.Param(jnp.zeros(shape))  # All neurons compute same thing!
```

**Common initializers**:
- `lecun_normal`: Good default for ReLU/tanh
- `he_normal`: Better for ReLU specifically
- `xavier_uniform`: Good for linear/sigmoid
- `zeros`: Only for biases, never weights

### Module Composition

```python
# Good: Compose reusable modules
class MyModel(nnx.Module):
    def __init__(self, *, rngs):
        self.encoder = ResNet(rngs=rngs)
        self.decoder = TransformerBlock(rngs=rngs)

# Bad: Don't hardcode everything
class MyModel(nnx.Module):
    def __init__(self, *, rngs):
        # 50 lines of conv/linear definitions...
```

### RNG Management

```python
# Good: Thread RNGs through constructors
model = MyModel(rngs=nnx.Rngs(params=0, dropout=1))

# Bad: Don't create RNGs inside modules
class BadModel(nnx.Module):
    def __init__(self):
        rngs = nnx.Rngs(params=0)  # Non-reproducible!
```

## Common Pitfalls

1. **Forgetting `.value`**: `self.weight` is a `Param`, need `.value` for the array
2. **Wrong tensor shapes**: Always print shapes during debugging
3. **Missing RNG state**: Dropout/initialization need RNG keys
4. **Batch norm in eval**: Must pass `use_running_average=True` during inference
5. **Mutable vs immutable**: NNX modules are mutable, but JAX transformations expect immutability (solved by `nnx.jit`)

## Next Steps

Now that you understand model definition, learn how to:
- [Load and preprocess data efficiently](./data-loading)
- [Write training loops with optimization](./training-loops)
- [Save and load model checkpoints](./checkpointing)

## Reference Code

See complete runnable examples:
- [`01_basic_model_definition.py`](https://github.com/mlnomadpy/flaxdocs/tree/master/examples/01_basic_model_definition.py) - All architectures with training code
