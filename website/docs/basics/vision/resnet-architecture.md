---
sidebar_position: 2
title: Build ResNet in Flax NNX
description: Implement ResNet from scratch in Flax NNX with residual blocks and skip connections, plus bottleneck blocks and global average pooling for deep vision models.
keywords: [ResNet, skip connections, residual block, Flax NNX, deep networks, bottleneck block, global average pooling, image classification]
---

# ResNet: Deep Networks with Skip Connections

Learn to build ResNet architectures that can train networks with 50+ layers using skip connections.

> Looking for the theory? For *why* residual networks work (vanishing gradients, the math behind skip connections) and the full family of variants (ResNet-18/34/50, basic vs. bottleneck blocks), see the architecture explainer: [Residual Networks (ResNet) in JAX](/architectures/resnet). This page is the hands-on build.

:::note Prerequisites
This guide builds on [Simple CNN](/basics/vision/simple-cnn) and [Simple Training Loop](/basics/workflows/simple-training).
:::

:::tip What you'll learn
- Implement a `ResNetBlock` whose skip connection adds the input back: `out = F(x) + x`
- Add a 1x1 projection shortcut when stride or channel count changes the shape
- Track running BatchNorm statistics as NNX state and pass `train=` correctly
- Replace flatten with global average pooling via `jnp.mean` over spatial axes
- Swap the basic block for a bottleneck block (1x1 to 3x3 to 1x1) for ResNet-50+
:::

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

class ResNetBlock(nnx.Module):
    """A single residual block: out = F(x) + shortcut(x)

    Handles both same-shape blocks and downsampling blocks. When the
    stride is > 1 or the channel count changes, a 1x1 projection shortcut
    brings the skip connection to the right shape so the add still works.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        *,
        rngs: nnx.Rngs,
    ):
        # First conv applies the stride (this is where any downsampling happens)
        self.conv1 = nnx.Conv(
            in_channels,
            out_channels,
            kernel_size=(3, 3),
            strides=(stride, stride),
            padding='SAME',
            rngs=rngs
        )
        self.bn1 = nnx.BatchNorm(out_channels, rngs=rngs)

        self.conv2 = nnx.Conv(
            out_channels,
            out_channels,
            kernel_size=(3, 3),
            padding='SAME',  # Keep spatial dimensions unchanged
            rngs=rngs
        )
        self.bn2 = nnx.BatchNorm(out_channels, rngs=rngs)

        # Projection shortcut: only needed when shapes change
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nnx.Conv(
                in_channels,
                out_channels,
                kernel_size=(1, 1),
                strides=(stride, stride),
                rngs=rngs
            )
            self.bn_shortcut = nnx.BatchNorm(out_channels, rngs=rngs)
        else:
            self.shortcut = None

    def __call__(self, x, train: bool = False):
        residual = x

        # First conv block (may downsample via stride)
        x = self.conv1(x)
        x = self.bn1(x, use_running_average=not train)
        x = nnx.relu(x)

        # Second conv block (no activation yet)
        x = self.conv2(x)
        x = self.bn2(x, use_running_average=not train)

        # Project the skip connection only when shapes differ
        if self.shortcut is not None:
            residual = self.shortcut(residual)
            residual = self.bn_shortcut(residual, use_running_average=not train)

        # Add skip connection, then activate
        x = x + residual
        x = nnx.relu(x)

        return x
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
        
        # Stack of residual blocks. The channel count stays fixed here, so
        # every block is a same-shape block (stride=1). Bumping the stride or
        # the channel count on a block automatically enables its projection
        # shortcut — that's how the deeper variants downsample between stages.
        self.blocks = [
            ResNetBlock(initial_features, initial_features, stride=1, rngs=rngs)
            for _ in range(num_blocks)
        ]
        
        # Classification head
        self.fc = nnx.Linear(initial_features, num_classes, rngs=rngs)
    
    def __call__(self, x: jax.Array, train: bool = False) -> jax.Array:
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
        optax.adamw(learning_rate=1e-3, weight_decay=1e-4),
        wrt=nnx.Param
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
            optimizer.update(model, grads)
        
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

❌ **Wrong**: During training this silently runs in eval mode (`train=False` is
the default), so batch norm uses the wrong statistics
```python
logits = model(images)  # Defaults to train=False — eval mode!
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

✅ **Right**: Use a projection shortcut when changing dimensions — exactly what
the `self.shortcut` 1x1 conv in `ResNetBlock` above does:
```python
if self.shortcut is not None:        # stride != 1 or channels changed
    residual = self.shortcut(residual)  # 1x1 conv to match shapes
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

The block above is the *basic* residual block used in ResNet-18/34. Deeper
networks (ResNet-50/101/152) swap it for a *bottleneck* block, and full models
stack blocks into four downsampling stages. For the complete tour of the ResNet
family and when to use each block, see the explainer:
[Residual Networks (ResNet) in JAX](/architectures/resnet).

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
    
    def __call__(self, x, train: bool = False):
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

## Next steps

- [Residual Networks (ResNet) in JAX](/architectures/resnet) - The theory and full ResNet family
- [Contrastive Learning](/research/contrastive-learning) - Reuse the ResNetBlock as a feature encoder
- [Streaming Data](/basics/workflows/streaming-data) - Train ResNets on datasets larger than memory
- [Training at Scale](/scale) - Train on multiple GPUs

## Complete Examples

**Modular example with shared components:**
- [`examples/integrations/resnet_streaming.py`](https://github.com/mlnomadpy/flaxdocs/tree/master/examples/integrations/resnet_streaming.py) - ResNet training with streaming datasets from HuggingFace
- [`examples/shared/models.py`](https://github.com/mlnomadpy/flaxdocs/tree/master/examples/shared/models.py) - Reusable ResNetBlock implementation with skip connections
