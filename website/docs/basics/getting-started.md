---
sidebar_position: 1
---

# Getting Started with Flax Training

This guide will help you set up your environment and create your first Flax training script.

## Prerequisites

Before you begin, make sure you have:

- Python 3.8 or higher
- Basic understanding of neural networks
- Familiarity with NumPy

## Installation

### Installing JAX

First, install JAX. The installation depends on your system and whether you want CPU or GPU support.

**For CPU:**

```bash
pip install --upgrade jax jaxlib
```

**For GPU (CUDA 12):**

```bash
pip install --upgrade "jax[cuda12]"
```

**For TPU:**

```bash
pip install --upgrade "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

### Installing Flax

Once JAX is installed, install Flax:

```bash
pip install flax
```

### Additional Dependencies

For a complete training setup, you'll also need:

```bash
pip install optax tensorflow-datasets matplotlib
```

- **Optax**: Gradient processing and optimization library
- **TensorFlow Datasets**: Easy access to common datasets
- **Matplotlib**: Visualization

## Your First Flax Model

Let's create a simple neural network for MNIST classification.

### 1. Import Required Libraries

```python
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
import tensorflow_datasets as tfds
```

### 2. Define the Model

```python
class SimpleCNN(nn.Module):
    """A simple CNN model."""
    
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # Flatten
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)
        return x
```

### 3. Create Training State

```python
def create_train_state(rng, learning_rate):
    """Creates initial training state."""
    model = SimpleCNN()
    params = model.init(rng, jnp.ones([1, 28, 28, 1]))['params']
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )
```

### 4. Define Training Step

```python
@jax.jit
def train_step(state, batch):
    """Train for a single step."""
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, batch['image'])
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=batch['label']
        ).mean()
        return loss, logits
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    
    accuracy = jnp.mean(jnp.argmax(logits, -1) == batch['label'])
    return state, loss, accuracy
```

### 5. Load Data and Train

```python
def get_datasets():
    """Load MNIST train and test datasets into memory."""
    ds_builder = tfds.builder('mnist')
    ds_builder.download_and_prepare()
    train_ds = tfds.as_numpy(ds_builder.as_dataset(split='train', batch_size=-1))
    test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))
    
    train_ds['image'] = jnp.float32(train_ds['image']) / 255.0
    test_ds['image'] = jnp.float32(test_ds['image']) / 255.0
    
    return train_ds, test_ds

# Initialize
rng = jax.random.PRNGKey(0)
state = create_train_state(rng, learning_rate=0.001)

# Load data
train_ds, test_ds = get_datasets()

# Training loop
num_epochs = 10
batch_size = 128

for epoch in range(num_epochs):
    # Simple batching
    num_batches = len(train_ds['image']) // batch_size
    
    for i in range(num_batches):
        batch = {
            'image': train_ds['image'][i*batch_size:(i+1)*batch_size],
            'label': train_ds['label'][i*batch_size:(i+1)*batch_size],
        }
        state, loss, accuracy = train_step(state, batch)
    
    # Evaluation (simplified)
    test_logits = state.apply_fn({'params': state.params}, test_ds['image'])
    test_accuracy = jnp.mean(jnp.argmax(test_logits, -1) == test_ds['label'])
    
    print(f'Epoch {epoch + 1}, Loss: {loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
```

## Next Steps

Now that you have a basic training script working, explore:

- [Training Best Practices](./training-best-practices) - Learn optimization techniques
- [Model Checkpointing](./checkpointing) - Save and restore models

## Common Issues

### Out of Memory

If you run out of memory:
- Reduce batch size
- Use gradient accumulation
- Enable mixed precision training

### Slow Training

To speed up training:
- Ensure you're using JIT compilation (`@jax.jit`)
- Use appropriate hardware (GPU/TPU)
- Profile your code to find bottlenecks

## Resources

- [Flax GitHub Examples](https://github.com/google/flax/tree/main/examples)
- [JAX Quickstart](https://jax.readthedocs.io/en/latest/quickstart.html)
- [Optax Documentation](https://optax.readthedocs.io/)
