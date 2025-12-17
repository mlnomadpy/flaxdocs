---
sidebar_position: 1
---

# ResNet (Residual Networks)

ResNet, introduced in 2015, revolutionized deep learning by enabling the training of incredibly deep networks (up to 1000 layers). Before ResNet, networks rarely exceeded 20-30 layers due to the **vanishing gradient problem**.

## The Vanishing Gradient Problem

In deep networks, gradients are backpropagated through the chain rule. As they pass through many layers, they are repeatedly multiplied by small weights and activation derivatives. This causes the gradient signal to vanish (approach zero) before reaching the early layers, meaning the early layers stop learning.

### The Residual Solution

ResNet introduces **Skip Connections**. Instead of learning a function $H(x)$, we learn a residual function $F(x) = H(x) - x$.

$$
y = \underbrace{F(x)}_{\text{learned layers}} + \underbrace{x}_{\text{skip connection}}
$$

1.  **Forward Pass**: Information flows directly from input to output.
2.  **Backward Pass**: Gradients flow through the "identity highway" ($+x$) without being diminished.

## 1. Network Components

We define two types of blocks:
1.  **BasicBlock**: Standard 2-layer convolution block (Used in ResNet18/34).
2.  **Bottleneck**: 3-layer block with $1 \times 1$ compress/expand projections (Used in ResNet50/101).

```python
from flax import linen as nnx
import jax.numpy as jnp
from typing import Optional

class BasicBlock(nnx.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 downsample: Optional[nnx.Module] = None, rngs: nnx.Rngs = None):
        self.conv1 = nnx.Conv(in_channels, out_channels, kernel_size=(3, 3),
                              strides=stride, padding=1, use_bias=False, rngs=rngs)
        self.bn1 = nnx.BatchNorm(out_channels, rngs=rngs)
        self.conv2 = nnx.Conv(out_channels, out_channels, kernel_size=(3, 3),
                              strides=1, padding=1, use_bias=False, rngs=rngs)
        self.bn2 = nnx.BatchNorm(out_channels, rngs=rngs)
        self.downsample = downsample

    def __call__(self, x, training: bool = True):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out, use_running_average=not training)
        out = nnx.relu(out)
        out = self.conv2(out)
        out = self.bn2(out, use_running_average=not training)

        if self.downsample is not None:
            identity = self.downsample(x, training=training)

        out += identity
        out = nnx.relu(out)
        return out
```

### The Bottleneck Block

For deeper networks, we use a $1 \times 1$ convolution to reduce dimensions before the expensive $3 \times 3$, then expand it back.

```python
class Bottleneck(nnx.Module):
    expansion = 4

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 downsample: Optional[nnx.Module] = None, rngs: nnx.Rngs = None):
        self.conv1 = nnx.Conv(in_channels, out_channels, kernel_size=(1, 1),
                              use_bias=False, rngs=rngs)
        self.bn1 = nnx.BatchNorm(out_channels, rngs=rngs)
        self.conv2 = nnx.Conv(out_channels, out_channels, kernel_size=(3, 3),
                              strides=stride, padding=1, use_bias=False, rngs=rngs)
        self.bn2 = nnx.BatchNorm(out_channels, rngs=rngs)
        self.conv3 = nnx.Conv(out_channels, out_channels * self.expansion,
                              kernel_size=(1, 1), use_bias=False, rngs=rngs)
        self.bn3 = nnx.BatchNorm(out_channels * self.expansion, rngs=rngs)
        self.downsample = downsample

    def __call__(self, x, training: bool = True):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out, use_running_average=not training)
        out = nnx.relu(out)
        out = self.conv2(out)
        out = self.bn2(out, use_running_average=not training)
        out = nnx.relu(out)
        out = self.conv3(out)
        out = self.bn3(out, use_running_average=not training)

        if self.downsample is not None:
            identity = self.downsample(x, training=training)

        out += identity
        out = nnx.relu(out)
        return out
```

## 2. Dynamic Architecture

We construct the network dynamically using `_make_layer`. This allows us to easily switch between ResNet18, 50, etc.

```python
class ResNet(nnx.Module):
    def __init__(self, block_cls, layers: list[int], num_classes: int = 1000,
                 dtype=jnp.float32, rngs: nnx.Rngs = None):
        self.in_channels = 64
        self.dtype = dtype

        self.conv1 = nnx.Conv(3, 64, kernel_size=(7, 7), strides=2, padding=3,
                              use_bias=False, rngs=rngs)
        self.bn1 = nnx.BatchNorm(64, rngs=rngs)

        # Dynamic Layer Creation
        self.layer1 = self._make_layer(block_cls, 64, layers[0], rngs=rngs)
        self.layer2 = self._make_layer(block_cls, 128, layers[1], stride=2, rngs=rngs)
        self.layer3 = self._make_layer(block_cls, 256, layers[2], stride=2, rngs=rngs)
        self.layer4 = self._make_layer(block_cls, 512, layers[3], stride=2, rngs=rngs)

        self.feature_dim = 512 * block_cls.expansion
        self.head = nnx.Linear(self.feature_dim, num_classes, rngs=rngs)

    def _make_layer(self, block_cls, out_channels, blocks, stride=1, rngs=None):
        downsample = None
        # Create downsample layer if stride != 1 or dimensions change
        if stride != 1 or self.in_channels != out_channels * block_cls.expansion:
            downsample = DownsampleBlock(self.in_channels, out_channels * block_cls.expansion, stride, rngs)

        layers = []
        layers.append(block_cls(self.in_channels, out_channels, stride, downsample, rngs=rngs))
        self.in_channels = out_channels * block_cls.expansion
        for _ in range(1, blocks):
            layers.append(block_cls(self.in_channels, out_channels, rngs=rngs))
        
        return nnx.List(layers)

    def __call__(self, x, training: bool = True):
        x = x.astype(self.dtype)
        x = self.conv1(x)
        x = self.bn1(x, use_running_average=not training)
        x = nnx.relu(x)
        x = nnx.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding=((1, 1), (1, 1)))

        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for block in layer:
                x = block(x, training=training)

        x = jnp.mean(x, axis=(1, 2))
        return self.head(x)
        
class DownsampleBlock(nnx.Module):
    def __init__(self, in_channels, out_channels, stride, rngs):
        self.conv = nnx.Conv(in_channels, out_channels, kernel_size=(1, 1),
                             strides=stride, use_bias=False, rngs=rngs)
        self.bn = nnx.BatchNorm(out_channels, rngs=rngs)

    def __call__(self, x, training=True):
        x = self.conv(x)
        x = self.bn(x, use_running_average=not training)
        return x
```



## 4. Production Training with Sharding

For scale, we use JAX's `NamedSharding` to distribute data across devices.

### Data Sharding
We partition the batch dimension (`'data'`) across the available mesh of devices.

```python
    devices = jax.devices()
    mesh = Mesh(np.array(devices), ('data',))
    # Shard the '0th' dimension (batch) across 'data' axis
    data_sharding = NamedSharding(mesh, P('data', None, None, None)) 
    label_sharding = NamedSharding(mesh, P('data'))
    replicated_sharding = NamedSharding(mesh, P()) # Replicate weights
```

### The Training Step (JIT)
We use `nnx.jit` to compile the step. Arguments are standard JAX arrays.

```python
@nnx.jit
def train_step(model, optimizer, batch_images, batch_labels):
    
    def loss_fn(model):
        outputs = model(batch_images, training=True)
        loss = optax.softmax_cross_entropy_with_integer_labels(outputs, batch_labels).mean()
        
        acc = jnp.mean(jnp.argmax(outputs, axis=1) == batch_labels)
        return loss, acc

    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, acc), grads = grad_fn(model)
    optimizer.update(model, grads)
    return loss, acc
```

## 5. Deployment & Logging

We integrate `wandb` for experiment tracking and `orbax` for robust checkpointing.

```python
    # Checkpointing Setup
    ckpt_dir = os.path.abspath(args.save_dir)
    options = ocp.CheckpointManagerOptions(max_to_keep=args.checkpoint_keep, create=True)
    mngr = ocp.CheckpointManager(ckpt_dir, ocp.StandardCheckpointer(), options)

    # In Loop
    if val_acc > best_acc:
        best_acc = val_acc
        # Save state
        raw_state = nnx.state((model, optimizer))
        mngr.save(step=epoch, args=ocp.args.StandardSave(raw_state))
```

## Limitations & Evolution

While ResNet remains a strong baseline, it faces modern challenges:

1.  **Limited Receptive Field**: Convolution kernels ($3 \times 3$) only see local pixels. To understand the relationship between top-left and bottom-right pixels, the signal must pass through many layers.
    *   *Evolution*: **Vision Transformers (ViT)** use Self-Attention to give every pixel global awareness immediately.
2.  **Compute Inefficiency**: Dense convolutions process all pixels equally, even "background" sky.
    *   *Evolution*: **Sparse Networks** and **EfficientNet** optimize the width/depth/resolution balance.
3.  **The Saturation of Depth**: Beyond 1000 layers, even residual connections suffer from signal propagation issues.
    *   *Evolution*: **Normalization-Free Networks (NFNet)** and **Deep Equilibrium Models** explore alternatives to standard depth scaling.
