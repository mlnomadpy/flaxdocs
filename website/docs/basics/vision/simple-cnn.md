---
sidebar_position: 1
---

# Image Classification with CNNs

Learn to build convolutional neural networks (CNNs) for computer vision tasks, starting with the simplest approach and building up.

## Why CNNs for Vision?

Images have special structure that regular neural networks ignore:
- **Spatial relationships**: Nearby pixels are related
- **Translation invariance**: A cat is a cat whether it's on the left or right
- **Hierarchical patterns**: Edges → shapes → objects

CNNs exploit these properties through **convolutions** and **pooling**.

## Your First CNN

Let's build a simple CNN for MNIST (28x28 grayscale images, 10 digits):

```python
import jax
import jax.numpy as jnp
from flax import nnx

class SimpleCNN(nnx.Module):
    """Basic CNN: 2 conv layers + 2 dense layers"""
    
    def __init__(self, num_classes: int = 10, *, rngs: nnx.Rngs):
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
        x = x.reshape(x.shape[0], -1)  # (batch, 64*5*5 = 1600)
        
        # Classification head
        x = self.dense1(x)
        x = nnx.relu(x)
        x = self.dense2(x)
        
        return x  # Logits for each class

# Create model
model = SimpleCNN(num_classes=10, rngs=nnx.Rngs(params=0))

# Test with dummy data
images = jnp.ones((4, 28, 28, 1))  # Batch of 4 images
logits = model(images)  # Shape: (4, 10)
print(f"Output shape: {logits.shape}")
```

## Understanding Each Component

### Convolution Layers

```python
self.conv1 = nnx.Conv(
    in_features=1,       # Number of input channels
    out_features=32,     # Number of filters (output channels)
    kernel_size=(3, 3),  # Filter size
    rngs=rngs
)
```

**What it does**: Slides 3x3 filters across the image, detecting features like edges and corners.

**Why 32 filters?**: Each filter learns a different pattern. More filters = more capacity to learn complex features.

### Max Pooling

```python
x = nnx.max_pool(x, window_shape=(2, 2), strides=(2, 2))
```

**What it does**: Takes maximum value in each 2x2 region, downsampling the image.

**Why?**: 
- Reduces computation (smaller spatial dimensions)
- Adds translation invariance (small shifts don't matter)
- Builds hierarchical representations

### Shape Tracking (Critical!)

The most common CNN bug is shape mismatches. Track shapes carefully:

```python
# Input: (batch, 28, 28, 1)
x = self.conv1(x)  # (batch, 28, 28, 32) - same size with padding
x = nnx.max_pool(x, ...)  # (batch, 14, 14, 32) - halved by pooling
x = self.conv2(x)  # (batch, 14, 14, 64)
x = nnx.max_pool(x, ...)  # (batch, 5, 5, 64)
x = x.reshape(x.shape[0], -1)  # (batch, 1600)
```

Always print shapes during debugging!

## Complete Training Example

Here's a full training loop for MNIST:

```python
import optax
from flax import nnx

def create_dataloader():
    """Load MNIST data (simplified)"""
    import tensorflow_datasets as tfds
    
    ds = tfds.load('mnist', split='train', as_supervised=True)
    ds = ds.map(lambda img, label: (
        jnp.float32(img) / 255.0,  # Normalize to [0, 1]
        jax.nn.one_hot(label, 10)   # One-hot encode
    ))
    ds = ds.batch(128).prefetch(1)
    return tfds.as_numpy(ds)

def train_step(model, optimizer, batch):
    """Single training step"""
    images, labels = batch
    
    def loss_fn(model):
        logits = model(images)
        loss = optax.softmax_cross_entropy(logits, labels).mean()
        return loss
    
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(grads)
    return loss

# Setup
model = SimpleCNN(rngs=nnx.Rngs(params=0))
optimizer = nnx.Optimizer(model, optax.adam(1e-3))
train_loader = create_dataloader()

# Train
for epoch in range(5):
    epoch_loss = 0
    for i, batch in enumerate(train_loader):
        loss = train_step(model, optimizer, batch)
        epoch_loss += loss
    
    print(f"Epoch {epoch + 1}: Loss = {epoch_loss / (i + 1):.4f}")
```

## Data Augmentation

Real-world CNNs need data augmentation to generalize:

```python
def augment_image(image, *, rngs: nnx.Rngs):
    """Simple augmentation pipeline"""
    # Random horizontal flip
    if jax.random.uniform(rngs.augment()) > 0.5:
        image = jnp.fliplr(image)
    
    # Random crop
    h, w = image.shape[:2]
    top = jax.random.randint(rngs.augment(), (), 0, 4)
    left = jax.random.randint(rngs.augment(), (), 0, 4)
    image = jax.lax.dynamic_slice(image, (top, left, 0), (h-4, w-4, 1))
    
    # Random brightness
    brightness = jax.random.uniform(rngs.augment(), (), minval=0.8, maxval=1.2)
    image = jnp.clip(image * brightness, 0, 1)
    
    return image

# Apply during training
for batch in train_loader:
    images, labels = batch
    images = jax.vmap(lambda img: augment_image(img, rngs=rngs))(images)
    # ... train with augmented images
```

## Common Issues and Solutions

### Issue 1: Shape Mismatch in Flatten

❌ **Wrong**: Hardcoding flatten size
```python
x = x.reshape(x.shape[0], 1600)  # Breaks if image size changes
```

✅ **Right**: Dynamic reshaping
```python
x = x.reshape(x.shape[0], -1)  # Automatically calculates size
```

### Issue 2: Forgetting Channel Dimension

❌ **Wrong**: (batch, height, width)
```python
images = jnp.ones((4, 28, 28))  # Missing channel dimension!
```

✅ **Right**: (batch, height, width, channels)
```python
images = jnp.ones((4, 28, 28, 1))  # Grayscale has 1 channel
```

### Issue 3: Not Normalizing Inputs

❌ **Wrong**: Raw pixel values [0, 255]
```python
images = images  # Values too large, unstable training
```

✅ **Right**: Normalize to [0, 1] or [-1, 1]
```python
images = images / 255.0  # Scale to [0, 1]
# or
images = (images / 255.0) * 2 - 1  # Scale to [-1, 1]
```

## Performance Tips

1. **Use larger batch sizes**: CNNs benefit from batch sizes of 64-256
2. **Start with small learning rate**: 1e-3 or 1e-4 for Adam
3. **Monitor validation accuracy**: Stop when it plateaus
4. **Use data augmentation**: Crucial for small datasets

## Next Steps

- [ResNet Architecture](./resnet-architecture.md) - Build deeper networks with skip connections
- [Data Loading](../workflows/data-loading-simple.md) - Efficient data pipelines
- [Streaming Data](../workflows/streaming-data.md) - Handle large datasets

## Complete Examples

**Modular training with shared components:**
- [`examples/training/vision_mnist.py`](https://github.com/mlnomadpy/flaxdocs/tree/master/examples/training/vision_mnist.py) - Complete MNIST training using `shared.models.CNN` and `shared.training_utils`
- [`examples/shared/models.py`](https://github.com/mlnomadpy/flaxdocs/tree/master/examples/shared/models.py) - Reusable CNN architecture with batch normalization and dropout
