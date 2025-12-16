---
sidebar_position: 1
---

# ResNet (Residual Networks)

ResNets uses residual connections to train very deep networks. They are a fundamental architecture for computer vision.

## Anatomy of a Residual Block

**The Core Insight**: Skip connections allow gradients to flow directly through the network, mitigating the vanishing gradient problem.

```
Residual: out = F(x) + x
```
- Only needs to learn the "residue" (difference)
- Can easily learn the identity function (by making F(x) = 0)

### Implementation in Flax NNX

```python
from flax import linen as nnx
import jax.numpy as jnp

class ResidualBlock(nnx.Module):
    """Basic residual block: out = F(x) + x"""
    
    def __init__(
        self,
        features: int,
        stride: int = 1,
        *,
        rngs: nnx.Rngs
    ):
        # Main path: two 3x3 convolutions
        self.conv1 = nnx.Conv(
            in_features=features,
            out_features=features,
            kernel_size=(3, 3),
            strides=(stride, stride),
            padding='SAME',
            use_bias=False,
            rngs=rngs
        )
        self.bn1 = nnx.BatchNorm(features, rngs=rngs)
        
        self.conv2 = nnx.Conv(
            in_features=features,
            out_features=features,
            kernel_size=(3, 3),
            padding='SAME',
            use_bias=False,
            rngs=rngs
        )
        self.bn2 = nnx.BatchNorm(features, rngs=rngs)
        
        # Shortcut path: identity or projection
        if stride != 1:
            # Need to downsample skip connection
            self.shortcut = nnx.Sequential(
                nnx.Conv(
                    in_features=features,
                    out_features=features,
                    kernel_size=(1, 1),
                    strides=(stride, stride),
                    use_bias=False,
                    rngs=rngs
                ),
                nnx.BatchNorm(features, rngs=rngs)
            )
        else:
            # Identity shortcut
            self.shortcut = lambda x, train: x
    
    def __call__(self, x, *, train: bool = True):
        # Main path
        out = self.conv1(x)
        out = self.bn1(out, use_running_average=not train)
        out = nnx.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out, use_running_average=not train)
        
        # Skip connection
        identity = self.shortcut(x, train=train) if callable(self.shortcut) else self.shortcut(x)
        
        # Add and activate
        out = out + identity
        out = nnx.relu(out)
        
        return out
```

## Complete ResNet Architecture

```python
class ResNet(nnx.Module):
    """ResNet architecture for ImageNet"""
    
    def __init__(
        self,
        num_classes: int = 1000,
        layers: list[int] = [2, 2, 2, 2],  # ResNet-18
        *,
        rngs: nnx.Rngs
    ):
        # Stem: Initial downsampling
        self.conv1 = nnx.Conv(
            in_features=3,  # RGB
            out_features=64,
            kernel_size=(7, 7),
            strides=(2, 2),
            padding='SAME',
            use_bias=False,
            rngs=rngs
        )
        self.bn1 = nnx.BatchNorm(64, rngs=rngs)
        
        # 4 stages with increasing channels
        self.layer1 = self._make_layer(64, 64, layers[0], stride=1, rngs=rngs)
        self.layer2 = self._make_layer(64, 128, layers[1], stride=2, rngs=rngs)
        self.layer3 = self._make_layer(128, 256, layers[2], stride=2, rngs=rngs)
        self.layer4 = self._make_layer(256, 512, layers[3], stride=2, rngs=rngs)
        
        # Classification head
        self.fc = nnx.Linear(512, num_classes, rngs=rngs)
    
    def _make_layer(self, in_features, out_features, num_blocks, stride, rngs):
        """Create a stack of residual blocks"""
        layers = []
        
        # First block may downsample
        layers.append(ResidualBlock(out_features, stride=stride, rngs=rngs))
        
        # Rest are identity stride using updated out_features
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_features, stride=1, rngs=rngs))
        
        return layers
    
    def __call__(self, x, *, train: bool = True):
        # Stem
        x = self.conv1(x)
        x = self.bn1(x, use_running_average=not train)
        x = nnx.relu(x)
        x = nnx.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding='SAME')
        
        # Stages
        for block in self.layer1:
            x = block(x, train=train)
        for block in self.layer2:
            x = block(x, train=train)
        for block in self.layer3:
            x = block(x, train=train)
        for block in self.layer4:
            x = block(x, train=train)
        
        # Global average pooling
        x = jnp.mean(x, axis=(1, 2))
        
        # Classification
        return self.fc(x)
```

## Common Variants

| Model | Layers Config | Total Layers |
|-------|---------------|--------------|
| **ResNet-18** | `[2, 2, 2, 2]` | 18 |
| **ResNet-34** | `[3, 4, 6, 3]` | 34 |
| **ResNet-50** | `[3, 4, 6, 3]`* | 50 (uses bottleneck blocks) |
