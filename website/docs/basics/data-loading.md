---
sidebar_position: 2
---

# Data Loading Strategies

Learn how to efficiently load and preprocess data for training neural networks. This guide covers both TensorFlow Datasets (TFDS) and Grain, explaining when and why to use each.

## Why Data Loading Matters

Training modern neural networks requires:
- **High throughput**: Keep GPUs fed with data (avoid bottlenecks)
- **Memory efficiency**: Handle datasets larger than RAM
- **Reproducibility**: Same shuffle order across runs
- **Flexibility**: Easy preprocessing and augmentation

Poor data loading can make your GPU sit idle, wasting compute!

## Two Approaches: TFDS vs Grain

### TensorFlow Datasets (TFDS)

**Pros**:
- Huge catalog of ready-to-use datasets (1000+)
- Mature, well-tested infrastructure
- Great documentation and community support
- Seamless integration with JAX

**Cons**:
- Requires TensorFlow installation (large dependency)
- Can be complex for custom datasets
- TensorFlow-specific quirks

**When to use**: Standard benchmarks (MNIST, CIFAR, ImageNet), quick prototyping

### Grain

**Pros**:
- Pure Python, no TensorFlow dependency
- Designed for JAX from the ground up
- Better control over data pipelines
- More explicit and debuggable

**Cons**:
- Fewer built-in datasets
- Newer, less documentation
- More manual setup

**When to use**: Custom datasets, production pipelines, minimal dependencies

## TensorFlow Datasets: The Basics

### Loading a Simple Dataset

```python
import tensorflow_datasets as tfds
import jax.numpy as jnp

# Load MNIST
ds = tfds.load(
    'mnist',
    split='train',
    shuffle_files=True,
    as_supervised=True,  # Returns (image, label) tuples
)

# Iterate over examples
for image, label in ds.take(3):
    print(f"Image shape: {image.shape}, Label: {label}")
    # Image shape: (28, 28, 1), Label: 5
```

### Understanding Splits

Datasets have predefined splits:

```python
# Training data (60k examples)
train_ds = tfds.load('mnist', split='train')

# Test data (10k examples)
test_ds = tfds.load('mnist', split='test')

# Custom splits
train_ds = tfds.load('mnist', split='train[:80%]')  # First 80%
val_ds = tfds.load('mnist', split='train[80%:]')    # Last 20%

# Combine splits
all_ds = tfds.load('mnist', split='train+test')
```

**Why this matters**: Proper train/val/test splits prevent overfitting and ensure fair evaluation.

### Building an Efficient Pipeline

Raw datasets need preprocessing:

```python
def create_mnist_pipeline(split='train', batch_size=32):
    """Complete MNIST data pipeline with all optimizations"""
    
    # 1. Load dataset
    ds = tfds.load('mnist', split=split, shuffle_files=True)
    
    # 2. Shuffle: important for SGD
    if split == 'train':
        ds = ds.shuffle(10_000, seed=42)  # Shuffle buffer size
    
    # 3. Normalize and preprocess
    def preprocess(example):
        image = example['image']
        label = example['label']
        
        # Normalize to [0, 1]
        image = tf.cast(image, tf.float32) / 255.0
        
        # One-hot encode labels
        label = tf.one_hot(label, depth=10)
        
        return image, label
    
    ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    
    # 4. Batch
    ds = ds.batch(batch_size, drop_remainder=True)
    
    # 5. Prefetch: overlap data loading with training
    ds = ds.prefetch(tf.data.AUTOTUNE)
    
    # 6. Convert to NumPy iterator for JAX
    return tfds.as_numpy(ds)

# Use it
train_loader = create_mnist_pipeline('train', batch_size=128)

for images, labels in train_loader:
    # images: (128, 28, 28, 1)
    # labels: (128, 10)
    logits = model(images)  # Your model here
    loss = cross_entropy(logits, labels)
    # ... training code
```

### Key Pipeline Concepts

**Shuffle buffer size**:
- Too small: Poor randomization, bad SGD convergence
- Too large: High memory usage, slow startup
- Rule of thumb: 5-10x batch size, or 10k for large datasets

**`num_parallel_calls=AUTOTUNE`**:
- Runs preprocessing on multiple CPU threads
- Essential for GPU utilization
- Let TensorFlow optimize automatically

**`drop_remainder=True`**:
- Ensures all batches have same size
- Required for JIT compilation (JAX needs static shapes)
- Last partial batch is dropped

**`prefetch`**:
- Prepares next batch while GPU processes current batch
- Hides data loading latency
- Always use it!

## Data Augmentation for Images

Augmentation prevents overfitting by creating variations:

```python
import tensorflow as tf

def augment_image(image, label):
    """Apply random augmentations during training"""
    
    # Random horizontal flip (50% chance)
    image = tf.image.random_flip_left_right(image)
    
    # Random brightness adjustment
    image = tf.image.random_brightness(image, max_delta=0.1)
    
    # Random contrast
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    
    # Random crop and resize (zoom effect)
    image = tf.image.random_crop(image, size=[28, 28, 1])
    image = tf.image.resize(image, [32, 32])
    
    # Clip to valid range
    image = tf.clip_by_value(image, 0.0, 1.0)
    
    return image, label

def create_cifar10_pipeline(split='train', batch_size=32):
    ds = tfds.load('cifar10', split=split, shuffle_files=True)
    
    if split == 'train':
        ds = ds.shuffle(50_000)
        # Apply augmentation ONLY during training
        ds = ds.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    ds = ds.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    
    return tfds.as_numpy(ds)
```

### Augmentation Best Practices

1. **Only augment training data**: Never augment validation/test
2. **Match your domain**: Different augmentations for different tasks
3. **Don't overdo it**: Too much augmentation can hurt performance
4. **Preserve semantics**: Don't change what the image represents

## Advanced: Caching and Repeating

For small datasets that fit in memory:

```python
ds = tfds.load('mnist', split='train')

# Cache the entire dataset in RAM
ds = ds.cache()  # First epoch loads, rest are instant

# Preprocessing
ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)

# Repeat indefinitely (no epoch boundaries)
ds = ds.repeat()

# Shuffle, batch, prefetch
ds = ds.shuffle(10_000)
ds = ds.batch(32, drop_remainder=True)
ds = ds.prefetch(tf.data.AUTOTUNE)
```

**When to cache**:
- Dataset fits in RAM (< few GB)
- Expensive preprocessing (augmentation, etc.)
- Want faster epochs 2+

**When NOT to cache**:
- Large datasets (will cause OOM)
- Augmentation (caching defeats randomization)

## Grain: Pure Python Data Loading

Grain is JAX's native data loading library. Let's build from scratch:

### Basic Grain Pipeline

```python
import grain.python as grain
import numpy as np

# Step 1: Create a data source
class NumpyArraySource(grain.RandomAccessDataSource):
    """Load data from NumPy arrays"""
    
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return {
            'image': self.images[idx],
            'label': self.labels[idx]
        }

# Load your data (e.g., from disk)
images = np.random.rand(10_000, 28, 28, 1).astype(np.float32)
labels = np.random.randint(0, 10, size=10_000)

source = NumpyArraySource(images, labels)

# Step 2: Create transformations
def normalize_transform(example):
    example['image'] = example['image'] / 255.0
    return example

def one_hot_transform(example):
    label = example['label']
    one_hot = np.zeros(10, dtype=np.float32)
    one_hot[label] = 1.0
    example['label'] = one_hot
    return example

# Step 3: Build the pipeline
loader = grain.DataLoader(
    data_source=source,
    sampler=grain.IndexSampler(
        len(source),
        shuffle=True,
        seed=42,
    ),
    operations=[
        grain.MapTransform(normalize_transform),
        grain.MapTransform(one_hot_transform),
        grain.Batch(batch_size=32, drop_remainder=True),
    ],
    worker_count=4,  # Parallel workers
)

# Use it
for batch in loader:
    images = batch['image']  # (32, 28, 28, 1)
    labels = batch['label']  # (32, 10)
    # Training code here
```

### Understanding Grain Components

**`RandomAccessDataSource`**:
- Interface for accessing data by index
- Implement `__len__` and `__getitem__`
- Can be backed by files, databases, arrays, etc.

**`IndexSampler`**:
- Controls the order of data access
- Handles shuffling and sharding
- Ensures reproducibility with seeds

**`MapTransform`**:
- Applies a function to each example
- Runs in parallel workers
- Keep transformations pure (no side effects)

**`Batch`**:
- Groups examples into batches
- Stacks arrays along batch dimension
- `drop_remainder=True` for consistent shapes

## Streaming Large Datasets

When data doesn't fit in memory, stream from disk:

### HuggingFace Datasets Streaming

```python
from datasets import load_dataset

# Load dataset in streaming mode (doesn't download everything)
dataset = load_dataset(
    'HuggingFaceFW/fineweb-edu',
    name='sample-10BT',
    split='train',
    streaming=True  # KEY: Stream instead of download
)

# Shuffle with buffer
dataset = dataset.shuffle(seed=42, buffer_size=10_000)

# Tokenize text
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('gpt2')

def tokenize(example):
    return tokenizer(
        example['text'],
        truncation=True,
        max_length=512,
        return_tensors='np'
    )

dataset = dataset.map(tokenize, batched=True)

# Batch and iterate
for i, batch in enumerate(dataset.iter(batch_size=32)):
    input_ids = batch['input_ids']
    # Training code
    
    if i >= 1000:  # Train for 1000 steps
        break
```

### Grain Streaming from Files

```python
import grain.python as grain
import json

class JsonLineSource(grain.RandomAccessDataSource):
    """Stream from .jsonl files without loading all into memory"""
    
    def __init__(self, filepath):
        self.filepath = filepath
        
        # Count lines
        with open(filepath) as f:
            self._length = sum(1 for _ in f)
    
    def __len__(self):
        return self._length
    
    def __getitem__(self, idx):
        # Read only the specific line we need
        with open(self.filepath) as f:
            for i, line in enumerate(f):
                if i == idx:
                    return json.loads(line)
        raise IndexError(f"Index {idx} out of range")

# Use it
source = JsonLineSource('data.jsonl')
loader = grain.DataLoader(
    data_source=source,
    sampler=grain.IndexSampler(len(source), shuffle=True, seed=42),
    operations=[
        grain.MapTransform(your_preprocessing),
        grain.Batch(batch_size=32),
    ],
    worker_count=4,
)
```

**Warning**: This is slow! Better to use indexed formats (TFRecord, Parquet, etc.) for production.

## Custom Datasets

### Loading Images from Disk

```python
import glob
from PIL import Image

class ImageFolderSource(grain.RandomAccessDataSource):
    """Load images from directory structure: data/class_name/image.jpg"""
    
    def __init__(self, root_dir):
        self.files = []
        self.labels = []
        
        # Find all images
        for class_idx, class_dir in enumerate(sorted(glob.glob(f"{root_dir}/*"))):
            for img_path in glob.glob(f"{class_dir}/*.jpg"):
                self.files.append(img_path)
                self.labels.append(class_idx)
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert('RGB')
        img = np.array(img, dtype=np.float32) / 255.0
        
        return {
            'image': img,
            'label': self.labels[idx]
        }

# Use it
source = ImageFolderSource('/path/to/imagenet/train')
loader = grain.DataLoader(
    data_source=source,
    sampler=grain.IndexSampler(len(source), shuffle=True, seed=42),
    operations=[
        grain.MapTransform(augment_fn),
        grain.Batch(batch_size=256, drop_remainder=True),
    ],
    worker_count=8,
)
```

## Performance Optimization

### Bottleneck Diagnosis

```python
import time

# Time your data loading
start = time.time()
for i, batch in enumerate(train_loader):
    if i >= 100:
        break
end = time.time()

throughput = 100 * batch_size / (end - start)
print(f"Throughput: {throughput:.1f} examples/sec")

# If throughput < 1000 examples/sec, you have a bottleneck!
```

### Common Bottlenecks and Fixes

**Symptom**: Low GPU utilization (< 80%)
- **Cause**: Data loading too slow
- **Fix**: Increase `num_parallel_calls`, add `prefetch`, reduce preprocessing

**Symptom**: High memory usage
- **Cause**: Shuffle buffer or cache too large
- **Fix**: Reduce buffer size, don't cache large datasets

**Symptom**: First epoch very slow, rest fast
- **Cause**: Dataset initialization overhead
- **Fix**: Use caching, preprocess offline, use faster formats

### Best Practices Checklist

✅ Always use `prefetch` for GPU training  
✅ Set `drop_remainder=True` for JIT compilation  
✅ Normalize inputs (0-1 range or standardization)  
✅ Shuffle training data every epoch  
✅ Use parallel preprocessing (`num_parallel_calls`)  
✅ Profile throughput before scaling up  
✅ Keep preprocessing simple (complex ops → CPU bottleneck)  
✅ Use augmentation only on training split  

## Next Steps

Now that you can load data efficiently:
- [Write effective training loops](./training-loops)
- [Implement checkpointing for long runs](./checkpointing)
- [Scale to distributed training](../scale/distributed-training)

## Reference Code

Complete examples:
- [`03_data_loading_tfds.py`](https://github.com/mlnomadpy/flaxdocs/tree/master/examples/03_data_loading_tfds.py) - TFDS patterns
- [`04_data_loading_grain.py`](https://github.com/mlnomadpy/flaxdocs/tree/master/examples/04_data_loading_grain.py) - Grain pipelines
