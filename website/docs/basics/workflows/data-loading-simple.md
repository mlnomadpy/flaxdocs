---
sidebar_position: 2
title: Data Loading for Neural Networks - Simple Approaches
description: Learn to load and prepare data for neural network training with Flax. Master data loading strategies from simple to efficient streaming approaches.
keywords: [data loading, training data, batch loading, data pipeline, neural network data, dataset preparation, data preprocessing]
---

# Loading Data Simply

Learn to load and prepare data for training - starting with the simplest approach that actually works.

## The Two Questions

Before loading data, answer:
1. **Where is my data?** (files, datasets, database)
2. **How big is it?** (fits in RAM or needs streaming)

## Option 1: Data Fits in RAM

The simplest case - load everything at once:

```python
import jax.numpy as jnp
import numpy as np

def load_simple_dataset():
    """Load entire dataset into memory"""
    # Load data (example with MNIST)
    import tensorflow_datasets as tfds
    
    # Load as numpy arrays
    data = tfds.load('mnist', split='train', batch_size=-1, as_supervised=True)
    images, labels = tfds.as_numpy(data)
    
    # Normalize images
    images = images.astype('float32') / 255.0
    
    # One-hot encode labels
    labels = jax.nn.one_hot(labels, 10)
    
    return images, labels

# Load once
images, labels = load_simple_dataset()
print(f"Loaded {len(images)} images")
```

**Pros**: Simple, no complexity  
**Cons**: Only works for small datasets (< 1GB)

## Option 2: Mini-Batch Loading

For larger datasets, load in batches:

```python
def create_batches(images, labels, batch_size=32, shuffle=True):
    """Create mini-batches from data"""
    num_samples = len(images)
    indices = np.arange(num_samples)
    
    if shuffle:
        np.random.shuffle(indices)
    
    # Split into batches
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_indices = indices[start_idx:end_idx]
        
        yield images[batch_indices], labels[batch_indices]

# Usage
for batch_images, batch_labels in create_batches(images, labels, batch_size=128):
    # Train on this batch
    loss = train_step(model, optimizer, (batch_images, batch_labels))
```

## Option 3: TensorFlow Datasets (Recommended)

For standard datasets, use TFDS pipelines:

```python
import tensorflow_datasets as tfds
import tensorflow as tf

def create_tfds_pipeline(batch_size=32):
    """Efficient TFDS pipeline"""
    
    # Load dataset
    ds = tfds.load('mnist', split='train', shuffle_files=True)
    
    # Preprocessing function
    def preprocess(example):
        image = tf.cast(example['image'], tf.float32) / 255.0
        label = tf.one_hot(example['label'], 10)
        return image, label
    
    # Build pipeline
    ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(buffer_size=10_000)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    
    return tfds.as_numpy(ds)

# Usage
train_loader = create_tfds_pipeline(batch_size=128)

for epoch in range(10):
    for batch in train_loader:
        images, labels = batch
        loss = train_step(model, optimizer, (images, labels))
```

**Key optimizations**:
- `shuffle`: Randomize data order
- `batch`: Group into mini-batches
- `prefetch`: Load next batch while training current one
- `num_parallel_calls=AUTOTUNE`: Parallel preprocessing

## Understanding Data Shapes

Common confusion: what shape should my data be?

```python
# Images (vision)
images.shape  # (batch, height, width, channels)
# Examples:
# MNIST: (32, 28, 28, 1) - grayscale
# CIFAR: (32, 32, 32, 3) - RGB

# Labels (classification)
labels.shape  # (batch, num_classes) - one-hot
# Example: (32, 10) for 10 classes

# Sequences (text)
tokens.shape  # (batch, sequence_length)
# Example: (32, 128) for sequences of length 128
```

## Splitting Train/Val/Test

Always split your data:

```python
def load_split_data():
    """Load with proper train/val/test splits"""
    
    # Train: 80% of data
    train_ds = tfds.load('mnist', split='train[:80%]')
    
    # Validation: 20% of training data
    val_ds = tfds.load('mnist', split='train[80%:]')
    
    # Test: separate test set
    test_ds = tfds.load('mnist', split='test')
    
    return train_ds, val_ds, test_ds

train_ds, val_ds, test_ds = load_split_data()
```

**Why three splits?**
- **Train**: Optimize parameters
- **Validation**: Tune hyperparameters, early stopping
- **Test**: Final evaluation (use only once!)

## Data Augmentation (Vision)

Improve generalization with augmentation:

```python
def augment_image(image):
    """Apply random augmentations"""
    # Random horizontal flip
    image = tf.image.random_flip_left_right(image)
    
    # Random brightness
    image = tf.image.random_brightness(image, max_delta=0.1)
    
    # Random crop and resize
    image = tf.image.random_crop(image, size=[28, 28, 1])
    
    return image

def create_augmented_pipeline(batch_size=32):
    ds = tfds.load('mnist', split='train')
    
    def preprocess(example):
        image = tf.cast(example['image'], tf.float32) / 255.0
        image = augment_image(image)  # Apply augmentation
        label = tf.one_hot(example['label'], 10)
        return image, label
    
    ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(10_000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return tfds.as_numpy(ds)
```

## Working with HuggingFace Datasets

For NLP tasks, HuggingFace datasets are great:

```python
from datasets import load_dataset

def load_text_data():
    """Load text dataset from HuggingFace"""
    # Load dataset
    dataset = load_dataset('imdb', split='train')
    
    # Tokenize (simplified)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    def tokenize(examples):
        return tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=128
        )
    
    dataset = dataset.map(tokenize, batched=True)
    dataset = dataset.with_format('jax')  # Convert to JAX arrays
    
    return dataset

# Usage
dataset = load_text_data()
for batch in dataset.iter(batch_size=32):
    input_ids = batch['input_ids']
    labels = batch['label']
    # Train...
```

## Common Pitfalls

### Pitfall 1: Not Shuffling

❌ **Wrong**: Training on sorted data
```python
ds = ds.batch(32)  # Not shuffled - sees all 0s, then all 1s, etc.
```

✅ **Right**: Always shuffle
```python
ds = ds.shuffle(10_000).batch(32)  # Randomize before batching
```

### Pitfall 2: Shuffling After Batching

❌ **Wrong**: Shuffles batches, not samples
```python
ds = ds.batch(32).shuffle(100)  # Shuffles batches, not helpful
```

✅ **Right**: Shuffle before batching
```python
ds = ds.shuffle(10_000).batch(32)  # Shuffles individual samples
```

### Pitfall 3: No Prefetching

❌ **Wrong**: GPU waits for data
```python
ds = ds.shuffle(10_000).batch(32)  # CPU prepares next batch while GPU is idle
```

✅ **Right**: Prefetch next batch
```python
ds = ds.shuffle(10_000).batch(32).prefetch(tf.data.AUTOTUNE)
# CPU prepares next batch while GPU trains on current batch
```

### Pitfall 4: Wrong Normalization

❌ **Wrong**: Forgetting to normalize
```python
image = example['image']  # Values in [0, 255] - too large!
```

✅ **Right**: Normalize to [0, 1] or [-1, 1]
```python
image = tf.cast(example['image'], tf.float32) / 255.0  # [0, 1]
# or
image = (tf.cast(example['image'], tf.float32) / 127.5) - 1  # [-1, 1]
```

## Performance Checklist

✅ Use `shuffle()` before `batch()`  
✅ Use `prefetch(AUTOTUNE)` at the end  
✅ Use `num_parallel_calls=AUTOTUNE` for `map()`  
✅ Normalize inputs to [0, 1] or [-1, 1]  
✅ Use larger batch sizes if memory allows (64-256)  
✅ Cache small datasets in memory with `.cache()`

## Quick Reference

```python
# Efficient pipeline template
def create_pipeline(split='train', batch_size=32):
    ds = tfds.load('dataset_name', split=split, shuffle_files=True)
    
    ds = ds.map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.cache()  # If dataset fits in RAM
    ds = ds.shuffle(buffer_size=10_000)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    
    return tfds.as_numpy(ds)
```

## Next Steps

- [Streaming Large Datasets](./streaming-data.md) - Handle data larger than memory
- [Simple Training Loop](./simple-training.md) - Put data loading to use

## Complete Examples

**Organized modular examples:**
- [`examples/basics/data_loading_tfds.py`](https://github.com/mlnomadpy/flaxdocs/tree/master/examples/basics/data_loading_tfds.py) - TensorFlow Datasets examples (MNIST, CIFAR-10, ImageNet)
- [`examples/basics/data_loading_grain.py`](https://github.com/mlnomadpy/flaxdocs/tree/master/examples/basics/data_loading_grain.py) - Pure Python Grain data loading

**Additional examples:**
- [`examples/training/vision_mnist.py`](https://github.com/mlnomadpy/flaxdocs/tree/master/examples/training/vision_mnist.py) - Complete training with TFDS data loading
