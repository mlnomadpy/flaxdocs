---
sidebar_position: 2
---

# ResNet: Deep Networks with Skip Connections

Learn to build ResNet architectures that can train networks with 50+ layers using skip connections.

## The Problem with Deep Networks

When you stack many layers, two problems emerge:

1. **Vanishing gradients**: Gradients get smaller as they flow backward, making early layers train slowly
2. **Degradation**: Deeper networks sometimes perform worse than shallower ones

ResNets solve this with **skip connections** (also called residual connections).

## The ResNet Building Block

The key insight: learn the *residual* (difference) instead of the full transformation:

```python
import jax
import jax.numpy as jnp
from flax import nnx

class ResidualBlock(nnx.Module):
    """A single residual block: out = F(x) + x"""
    
    def __init__(self, features: int, *, rngs: nnx.Rngs):
        # Two conv layers with batch norm
        self.conv1 = nnx.Conv(
            in_features=features,
            out_features=features,
            kernel_size=(3, 3),
            padding='SAME',  # Keep spatial dimensions unchanged
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
        
        # Second conv block (no activation yet)
        out = self.conv2(out)
        out = self.bn2(out, use_running_average=not train)
        
        # Add skip connection, then activate
        out = out + residual
        out = nnx.relu(out)
        
        return out
```

## Why Skip Connections Work

**Mathematical intuition**: 
- Without skip: `out = F(x)` - must learn everything from scratch
- With skip: `out = F(x) + x` - only needs to learn the *difference*

**Benefits**:
1. **Gradient flow**: Gradients can flow directly through the skip connection
2. **Identity mapping**: If `F(x) = 0`, the block becomes identity (do nothing)
3. **Easier optimization**: Easier to learn small adjustments than full transformations

## Complete ResNet Architecture

Here's a full ResNet for image classification:

```python
class ResNet(nnx.Module):
    """Complete ResNet for image classification"""
    
    def __init__(
        self, 
        num_classes: int,
        num_blocks: int = 3,
        initial_features: int = 64,
        *, 
        rngs: nnx.Rngs
    ):
        # Initial conv: increase channels, reduce resolution
        self.conv1 = nnx.Conv(
            in_features=3,  # RGB input
            out_features=initial_features,
            kernel_size=(7, 7),
            strides=(2, 2),
            padding='SAME',
            rngs=rngs
        )
        self.bn1 = nnx.BatchNorm(num_features=initial_features, rngs=rngs)
        
        # Stack of residual blocks
        self.blocks = [
            ResidualBlock(initial_features, rngs=rngs) 
            for _ in range(num_blocks)
        ]
        
        # Classification head
        self.fc = nnx.Linear(initial_features, num_classes, rngs=rngs)
    
    def __call__(self, x: jax.Array, *, train: bool = True) -> jax.Array:
        # Initial processing
        x = self.conv1(x)
        x = self.bn1(x, use_running_average=not train)
        x = nnx.relu(x)
        x = nnx.max_pool(x, window_shape=(3, 3), strides=(2, 2))
        
        # Process through residual blocks
        for block in self.blocks:
            x = block(x, train=train)
        
        # Global average pooling (replaces flatten)
        x = jnp.mean(x, axis=(1, 2))  # Average over height and width
        
        # Classification
        return self.fc(x)

# Create ResNet
model = ResNet(
    num_classes=10,
    num_blocks=9,  # 9 residual blocks = ~20 layers total
    initial_features=64,
    rngs=nnx.Rngs(params=0)
)
```

## Understanding Global Average Pooling

Instead of flattening all spatial positions, we average them:

```python
# Traditional approach: flatten everything
x = x.reshape(x.shape[0], -1)  # (batch, height*width*channels)
# Problem: Too many parameters, position-specific

# ResNet approach: global average pooling
x = jnp.mean(x, axis=(1, 2))  # (batch, channels)
# Benefit: Position-invariant, fewer parameters
```

## Training a ResNet

```python
import optax
from flax import nnx

def train_resnet():
    # Create model
    model = ResNet(
        num_classes=10,
        num_blocks=9,
        rngs=nnx.Rngs(params=0)
    )
    
    # Create optimizer with weight decay
    optimizer = nnx.Optimizer(
        model,
        optax.adamw(learning_rate=1e-3, weight_decay=1e-4)
    )
    
    # Training loop
    for epoch in range(100):
        for batch in train_loader:
            images, labels = batch
            
            # Forward and backward
            def loss_fn(model):
                logits = model(images, train=True)
                loss = optax.softmax_cross_entropy(logits, labels).mean()
                return loss
            
            loss, grads = nnx.value_and_grad(loss_fn)(model)
            optimizer.update(grads)
        
        # Evaluate
        accuracy = evaluate(model)
        print(f"Epoch {epoch}: Accuracy = {accuracy:.2%}")

def evaluate(model):
    """Evaluate on test set"""
    correct = 0
    total = 0
    
    for batch in test_loader:
        images, labels = batch
        logits = model(images, train=False)  # Important: train=False!
        preds = jnp.argmax(logits, axis=-1)
        correct += jnp.sum(preds == jnp.argmax(labels, axis=-1))
        total += len(images)
    
    return correct / total
```

## Common Pitfalls

### Pitfall 1: Forgetting `train=` Flag

❌ **Wrong**: Batch norm will use wrong statistics
```python
logits = model(images)  # Uses training mode by default!
```

✅ **Right**: Always specify mode explicitly
```python
logits = model(images, train=True)   # During training
logits = model(images, train=False)  # During evaluation
```

### Pitfall 2: Shape Mismatch in Skip Connection

❌ **Wrong**: Skip connection requires matching shapes
```python
out = out + residual  # Error if shapes don't match!
```

✅ **Right**: Use projection when changing dimensions
```python
if out.shape != residual.shape:
    residual = self.projection(residual)  # 1x1 conv to match shapes
out = out + residual
```

### Pitfall 3: Wrong Activation Placement

❌ **Wrong**: Activating before adding skip connection
```python
out = self.conv2(out)
out = nnx.relu(out)  # Too early!
out = out + residual
```

✅ **Right**: Add first, then activate
```python
out = self.conv2(out)
out = out + residual
out = nnx.relu(out)  # After skip connection
```

## Variants and Extensions

### Bottleneck Blocks (ResNet-50+)

For deeper networks, use bottleneck blocks to reduce computation:

```python
class BottleneckBlock(nnx.Module):
    """Bottleneck: 1x1 -> 3x3 -> 1x1"""
    
    def __init__(self, features: int, bottleneck_features: int, *, rngs: nnx.Rngs):
        # 1x1 conv to reduce dimensions
        self.conv1 = nnx.Conv(features, bottleneck_features, (1, 1), rngs=rngs)
        self.bn1 = nnx.BatchNorm(bottleneck_features, rngs=rngs)
        
        # 3x3 conv (main computation)
        self.conv2 = nnx.Conv(
            bottleneck_features, bottleneck_features, (3, 3), 
            padding='SAME', rngs=rngs
        )
        self.bn2 = nnx.BatchNorm(bottleneck_features, rngs=rngs)
        
        # 1x1 conv to restore dimensions
        self.conv3 = nnx.Conv(bottleneck_features, features, (1, 1), rngs=rngs)
        self.bn3 = nnx.BatchNorm(features, rngs=rngs)
    
    def __call__(self, x, *, train: bool = True):
        residual = x
        
        # Reduce
        out = self.conv1(x)
        out = self.bn1(out, use_running_average=not train)
        out = nnx.relu(out)
        
        # Transform
        out = self.conv2(out)
        out = self.bn2(out, use_running_average=not train)
        out = nnx.relu(out)
        
        # Expand
        out = self.conv3(out)
        out = self.bn3(out, use_running_average=not train)
        
        # Skip connection
        out = out + residual
        return nnx.relu(out)
```

## Key Takeaways

- **Skip connections** enable training of very deep networks (50-200+ layers)
- Always add the skip connection **before** the final activation
- Remember to set `train=False` during evaluation for batch norm
- Global average pooling is more efficient than flatten + dense layers
- Use bottleneck blocks for ResNet-50 and deeper

## Next Steps

- [Training at Scale](../../scale/) - Train on multiple GPUs
- [Streaming Data](../workflows/streaming-data.md) - Handle large datasets

## Complete Examples

**Modular example with shared components:**
- [`examples/integrations/resnet_streaming.py`](https://github.com/mlnomadpy/flaxdocs/tree/master/examples/integrations/resnet_streaming.py) - ResNet training with streaming datasets from HuggingFace
- [`examples/shared/models.py`](https://github.com/mlnomadpy/flaxdocs/tree/master/examples/shared/models.py) - Reusable ResNetBlock implementation with skip connections
