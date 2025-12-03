"""
Flax NNX: Data Loading with TensorFlow Datasets (TFDS)
=======================================================
This guide shows how to load and preprocess data using TFDS.
Run: python 03_data_loading_tfds.py
"""

import jax
import jax.numpy as jnp
from flax import nnx
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
from typing import Iterator, Dict


# Disable GPU for TensorFlow to avoid conflicts with JAX
tf.config.set_visible_devices([], 'GPU')


# ============================================================================
# 1. BASIC DATA LOADING
# ============================================================================

def load_mnist_basic():
    """Load MNIST dataset - simplest approach."""
    print("\n" + "=" * 80)
    print("1. Basic MNIST Loading")
    print("=" * 80)
    
    # Load entire dataset into memory
    ds_builder = tfds.builder('mnist')
    ds_builder.download_and_prepare()
    
    train_ds = tfds.as_numpy(ds_builder.as_dataset(split='train', batch_size=-1))
    test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))
    
    # Normalize
    train_images = train_ds['image'].astype(np.float32) / 255.0
    train_labels = train_ds['label']
    test_images = test_ds['image'].astype(np.float32) / 255.0
    test_labels = test_ds['label']
    
    print(f"Train images shape: {train_images.shape}")
    print(f"Train labels shape: {train_labels.shape}")
    print(f"Test images shape: {test_images.shape}")
    print(f"Test labels shape: {test_labels.shape}")
    print(f"Image dtype: {train_images.dtype}")
    print(f"Label dtype: {train_labels.dtype}")
    
    return train_images, train_labels, test_images, test_labels


# ============================================================================
# 2. BATCHED DATA LOADING
# ============================================================================

def load_mnist_batched(batch_size: int = 32):
    """Load MNIST with batching."""
    print("\n" + "=" * 80)
    print("2. Batched MNIST Loading")
    print("=" * 80)
    
    def prepare_dataset(split: str):
        ds = tfds.load('mnist', split=split, shuffle_files=True)
        
        # Normalize and prepare
        def preprocess(example):
            image = tf.cast(example['image'], tf.float32) / 255.0
            label = example['label']
            return {'image': image, 'label': label}
        
        ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.cache()
        ds = ds.shuffle(10000)
        ds = ds.batch(batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        
        return ds
    
    train_ds = prepare_dataset('train')
    test_ds = prepare_dataset('test')
    
    # Convert to numpy iterator
    train_iter = iter(tfds.as_numpy(train_ds))
    
    # Get first batch
    batch = next(train_iter)
    print(f"Batch image shape: {batch['image'].shape}")
    print(f"Batch label shape: {batch['label'].shape}")
    
    return train_ds, test_ds


# ============================================================================
# 3. DATA AUGMENTATION
# ============================================================================

def load_mnist_augmented(batch_size: int = 32):
    """Load MNIST with data augmentation."""
    print("\n" + "=" * 80)
    print("3. MNIST with Data Augmentation")
    print("=" * 80)
    
    def augment(example):
        image = tf.cast(example['image'], tf.float32) / 255.0
        label = example['label']
        
        # Random crop and resize
        image = tf.image.resize_with_crop_or_pad(image, 32, 32)
        image = tf.image.random_crop(image, [28, 28, 1])
        
        # Random brightness
        image = tf.image.random_brightness(image, 0.2)
        
        # Random rotation (small angles)
        # Note: Using resize as a simple augmentation
        image = tf.image.resize(image, [30, 30])
        image = tf.image.resize(image, [28, 28])
        
        # Clip values
        image = tf.clip_by_value(image, 0.0, 1.0)
        
        return {'image': image, 'label': label}
    
    train_ds = tfds.load('mnist', split='train', shuffle_files=True)
    train_ds = train_ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.cache()
    train_ds = train_ds.shuffle(10000)
    train_ds = train_ds.batch(batch_size)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    
    # Show sample
    sample = next(iter(tfds.as_numpy(train_ds)))
    print(f"Augmented batch shape: {sample['image'].shape}")
    print(f"Image value range: [{sample['image'].min():.3f}, {sample['image'].max():.3f}]")
    
    return train_ds


# ============================================================================
# 4. CIFAR-10 LOADING
# ============================================================================

def load_cifar10(batch_size: int = 32):
    """Load CIFAR-10 dataset."""
    print("\n" + "=" * 80)
    print("4. CIFAR-10 Loading")
    print("=" * 80)
    
    def prepare_cifar(example):
        image = tf.cast(example['image'], tf.float32) / 255.0
        label = example['label']
        
        # Data augmentation for CIFAR-10
        image = tf.image.random_flip_left_right(image)
        image = tf.image.resize_with_crop_or_pad(image, 40, 40)
        image = tf.image.random_crop(image, [32, 32, 3])
        
        return {'image': image, 'label': label}
    
    train_ds = tfds.load('cifar10', split='train', shuffle_files=True)
    train_ds = train_ds.map(prepare_cifar, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.cache()
    train_ds = train_ds.shuffle(10000)
    train_ds = train_ds.batch(batch_size)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    
    test_ds = tfds.load('cifar10', split='test')
    test_ds = test_ds.map(
        lambda x: {
            'image': tf.cast(x['image'], tf.float32) / 255.0,
            'label': x['label']
        },
        num_parallel_calls=tf.data.AUTOTUNE
    )
    test_ds = test_ds.batch(batch_size)
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)
    
    sample = next(iter(tfds.as_numpy(train_ds)))
    print(f"CIFAR-10 batch shape: {sample['image'].shape}")
    print(f"Number of classes: {len(set(sample['label']))}")
    
    return train_ds, test_ds


# ============================================================================
# 5. IMAGENET SUBSET
# ============================================================================

def load_imagenet_resized(batch_size: int = 32):
    """Load ImageNet Resized (smaller version for quick demos)."""
    print("\n" + "=" * 80)
    print("5. ImageNet Resized Loading")
    print("=" * 80)
    
    try:
        def prepare_imagenet(example):
            image = tf.cast(example['image'], tf.float32) / 255.0
            label = example['label']
            
            # Resize to standard size
            image = tf.image.resize(image, [64, 64])
            
            # Augmentation
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_brightness(image, 0.2)
            
            return {'image': image, 'label': label}
        
        # Using imagenet_resized/64x64 which is a smaller version
        train_ds = tfds.load('imagenet_resized/64x64', split='train', shuffle_files=True)
        train_ds = train_ds.map(prepare_imagenet, num_parallel_calls=tf.data.AUTOTUNE)
        train_ds = train_ds.batch(batch_size)
        train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
        
        sample = next(iter(tfds.as_numpy(train_ds)))
        print(f"ImageNet batch shape: {sample['image'].shape}")
        
        return train_ds
    except Exception as e:
        print(f"Note: ImageNet Resized not available: {e}")
        print("This dataset requires manual download.")
        return None


# ============================================================================
# 6. CUSTOM DATA ITERATOR
# ============================================================================

class DataIterator:
    """Custom data iterator for training loops."""
    
    def __init__(self, dataset, epochs: int = 1):
        self.dataset = dataset
        self.epochs = epochs
        self.current_epoch = 0
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current_epoch >= self.epochs:
            raise StopIteration
        
        for batch in tfds.as_numpy(self.dataset):
            # Convert to JAX arrays
            batch = {
                'image': jnp.array(batch['image']),
                'label': jnp.array(batch['label'])
            }
            yield batch
        
        self.current_epoch += 1


# ============================================================================
# 7. MIXED PRECISION PREPROCESSING
# ============================================================================

def load_mnist_mixed_precision(batch_size: int = 32):
    """Load MNIST with mixed precision support."""
    print("\n" + "=" * 80)
    print("7. MNIST with Mixed Precision")
    print("=" * 80)
    
    def preprocess_mixed(example):
        # Images in float16 for memory efficiency
        image = tf.cast(example['image'], tf.float16) / 255.0
        # Labels stay as integers
        label = example['label']
        return {'image': image, 'label': label}
    
    train_ds = tfds.load('mnist', split='train', shuffle_files=True)
    train_ds = train_ds.map(preprocess_mixed, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.batch(batch_size)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    
    sample = next(iter(tfds.as_numpy(train_ds)))
    print(f"Image dtype: {sample['image'].dtype}")
    print(f"Image shape: {sample['image'].shape}")
    
    return train_ds


# ============================================================================
# DEMONSTRATION
# ============================================================================

def main():
    print("=" * 80)
    print("Flax NNX Data Loading with TFDS Examples")
    print("=" * 80)
    
    # Test 1: Basic loading
    train_images, train_labels, test_images, test_labels = load_mnist_basic()
    
    # Test 2: Batched loading
    train_ds_batched, test_ds_batched = load_mnist_batched(batch_size=64)
    
    # Count batches
    num_batches = sum(1 for _ in tfds.as_numpy(train_ds_batched))
    print(f"Number of training batches: {num_batches}")
    
    # Test 3: Augmented data
    train_ds_aug = load_mnist_augmented(batch_size=32)
    
    # Test 4: CIFAR-10
    cifar_train, cifar_test = load_cifar10(batch_size=128)
    
    # Test 5: ImageNet (if available)
    imagenet_ds = load_imagenet_resized(batch_size=32)
    
    # Test 6: Custom iterator
    print("\n" + "=" * 80)
    print("6. Custom Data Iterator")
    print("=" * 80)
    
    iterator = DataIterator(train_ds_batched, epochs=1)
    batch_count = 0
    for batch in iterator:
        batch_count += 1
        if batch_count == 1:
            print(f"First batch from iterator: {batch['image'].shape}")
    print(f"Total batches iterated: {batch_count}")
    
    # Test 7: Mixed precision
    train_ds_mixed = load_mnist_mixed_precision(batch_size=32)
    
    # ========================================================================
    # Dataset Information
    # ========================================================================
    print("\n" + "=" * 80)
    print("Available TFDS Datasets (Popular)")
    print("=" * 80)
    
    datasets_info = {
        'mnist': 'Handwritten digits (28x28, grayscale)',
        'fashion_mnist': 'Fashion items (28x28, grayscale)',
        'cifar10': 'Natural images (32x32, RGB, 10 classes)',
        'cifar100': 'Natural images (32x32, RGB, 100 classes)',
        'imagenet2012': 'ImageNet classification (requires download)',
        'imagenet_resized': 'Resized ImageNet variants',
        'coco': 'Object detection and segmentation',
        'voc': 'Pascal VOC detection',
        'celeb_a': 'Face attributes dataset',
        'oxford_flowers102': 'Flower classification',
        'food101': 'Food classification',
        'stanford_dogs': 'Dog breed classification',
    }
    
    for name, description in datasets_info.items():
        print(f"  • {name:20s} - {description}")
    
    # ========================================================================
    # Best Practices
    # ========================================================================
    print("\n" + "=" * 80)
    print("Best Practices for Data Loading")
    print("=" * 80)
    
    print("""
    1. Use tf.data pipeline features:
       - .cache() to cache preprocessed data
       - .prefetch() for overlapping data loading
       - .shuffle() with appropriate buffer size
       - num_parallel_calls for parallel preprocessing
    
    2. Data augmentation:
       - Apply during training only
       - Use TensorFlow ops for GPU acceleration
       - Keep augmentations reasonable
    
    3. Memory management:
       - Use .cache() after expensive operations
       - Batch before prefetch
       - Consider mixed precision for large images
    
    4. For large datasets:
       - Don't load entire dataset into memory
       - Use streaming with batching
       - Prefetch multiple batches
    
    5. Reproducibility:
       - Set shuffle_files=True with fixed seed
       - Control random augmentation with seed
    """)
    
    print("\n" + "=" * 80)
    print("✓ All data loading examples completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
