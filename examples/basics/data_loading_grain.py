"""
Flax NNX: Data Loading with Grain
==================================
This guide shows how to use Grain for efficient data loading.
Run: pip install grain-nightly && python basics/data_loading_grain.py
"""

import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np
from typing import Iterator, Dict, Any
import functools


import sys
from pathlib import Path

# Add parent directory to path for shared imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Note: Grain is under active development
# Install with: pip install grain-nightly
try:
    import grain.python as grain
    GRAIN_AVAILABLE = True
except ImportError:
    print("Warning: Grain not available. Install with: pip install grain-nightly")
    GRAIN_AVAILABLE = False


# ============================================================================
# 1. SIMPLE IN-MEMORY DATA SOURCE
# ============================================================================

if GRAIN_AVAILABLE:
    class NumpyArraySource(grain.RandomAccessDataSource):
        """Simple data source from numpy arrays."""
        
        def __init__(self, images: np.ndarray, labels: np.ndarray):
            self.images = images
            self.labels = labels
        
        def __len__(self):
            return len(self.images)
        
        def __getitem__(self, index):
            return {
                'image': self.images[index],
                'label': self.labels[index]
            }


# ============================================================================
# 2. BASIC GRAIN DATALOADER
# ============================================================================

def create_simple_grain_loader():
    """Create a simple Grain dataloader."""
    if not GRAIN_AVAILABLE:
        print("Grain not available - skipping")
        return None
    
    print("\n" + "=" * 80)
    print("1. Simple Grain DataLoader")
    print("=" * 80)
    
    # Create dummy data
    num_samples = 1000
    images = np.random.randn(num_samples, 28, 28, 1).astype(np.float32)
    labels = np.random.randint(0, 10, num_samples)
    
    # Create data source
    source = NumpyArraySource(images, labels)
    
    # Create sampler
    sampler = grain.IndexSampler(
        num_records=len(source),
        shuffle=True,
        seed=42,
        num_epochs=1,
        shard_options=grain.NoSharding()
    )
    
    # Create dataloader
    loader = grain.DataLoader(
        data_source=source,
        sampler=sampler,
        worker_count=0,  # 0 for single-process
        worker_buffer_size=1,
    )
    
    # Test iteration
    batch = next(iter(loader))
    print(f"Batch image shape: {batch['image'].shape}")
    print(f"Batch label shape: {batch['label'].shape}")
    
    return loader


# ============================================================================
# 3. BATCHED GRAIN DATALOADER
# ============================================================================

def create_batched_grain_loader(batch_size: int = 32):
    """Create Grain dataloader with batching."""
    if not GRAIN_AVAILABLE:
        print("Grain not available - skipping")
        return None
    
    print("\n" + "=" * 80)
    print("2. Batched Grain DataLoader")
    print("=" * 80)
    
    # Create dummy data
    num_samples = 10000
    images = np.random.randn(num_samples, 28, 28, 1).astype(np.float32)
    labels = np.random.randint(0, 10, num_samples)
    
    source = NumpyArraySource(images, labels)
    
    # Batched sampler
    sampler = grain.IndexSampler(
        num_records=len(source),
        shuffle=True,
        seed=42,
        num_epochs=1,
        shard_options=grain.NoSharding()
    )
    
    # Add batch transformation
    batch_fn = grain.Batch(batch_size=batch_size, drop_remainder=True)
    
    loader = grain.DataLoader(
        data_source=source,
        sampler=sampler,
        worker_count=2,  # Use multiple workers
        worker_buffer_size=2,
        operations=[batch_fn]
    )
    
    batch = next(iter(loader))
    print(f"Batch size: {batch_size}")
    print(f"Batched image shape: {batch['image'].shape}")
    print(f"Batched label shape: {batch['label'].shape}")
    
    # Count total batches
    num_batches = sum(1 for _ in loader)
    print(f"Total batches: {num_batches}")
    
    return loader


# ============================================================================
# 4. DATA TRANSFORMATIONS WITH GRAIN
# ============================================================================

if GRAIN_AVAILABLE:
    class NormalizeTransform(grain.MapTransform):
        """Normalize images to [0, 1] range."""
        
        def map(self, element):
            element['image'] = element['image'] / 255.0
            return element
    
    
    class AugmentTransform(grain.MapTransform):
        """Apply data augmentation."""
        
        def __init__(self, rng_seed: int = 0):
            self.rng = np.random.RandomState(rng_seed)
        
        def map(self, element):
            image = element['image']
            
            # Random horizontal flip
            if self.rng.rand() > 0.5:
                image = np.fliplr(image)
            
            # Random brightness
            brightness_factor = self.rng.uniform(0.8, 1.2)
            image = np.clip(image * brightness_factor, 0, 1)
            
            element['image'] = image.astype(np.float32)
            return element


def create_grain_with_transforms(batch_size: int = 32):
    """Create Grain dataloader with transformations."""
    if not GRAIN_AVAILABLE:
        print("Grain not available - skipping")
        return None
    
    print("\n" + "=" * 80)
    print("3. Grain with Transformations")
    print("=" * 80)
    
    # Create dummy data (0-255 range)
    num_samples = 5000
    images = np.random.randint(0, 256, (num_samples, 28, 28, 1), dtype=np.uint8)
    labels = np.random.randint(0, 10, num_samples)
    
    source = NumpyArraySource(images, labels)
    
    sampler = grain.IndexSampler(
        num_records=len(source),
        shuffle=True,
        seed=42,
        num_epochs=1,
        shard_options=grain.NoSharding()
    )
    
    # Define transformation pipeline
    operations = [
        NormalizeTransform(),
        AugmentTransform(rng_seed=42),
        grain.Batch(batch_size=batch_size, drop_remainder=True)
    ]
    
    loader = grain.DataLoader(
        data_source=source,
        sampler=sampler,
        worker_count=2,
        worker_buffer_size=2,
        operations=operations
    )
    
    batch = next(iter(loader))
    print(f"Transformed batch shape: {batch['image'].shape}")
    print(f"Image value range: [{batch['image'].min():.3f}, {batch['image'].max():.3f}]")
    
    return loader


# ============================================================================
# 5. MULTI-EPOCH TRAINING WITH GRAIN
# ============================================================================

def create_multi_epoch_loader(batch_size: int = 32, num_epochs: int = 3):
    """Create Grain dataloader for multi-epoch training."""
    if not GRAIN_AVAILABLE:
        print("Grain not available - skipping")
        return None
    
    print("\n" + "=" * 80)
    print("4. Multi-Epoch Training")
    print("=" * 80)
    
    num_samples = 1000
    images = np.random.randn(num_samples, 28, 28, 1).astype(np.float32)
    labels = np.random.randint(0, 10, num_samples)
    
    source = NumpyArraySource(images, labels)
    
    # Multi-epoch sampler
    sampler = grain.IndexSampler(
        num_records=len(source),
        shuffle=True,
        seed=42,
        num_epochs=num_epochs,  # Multiple epochs
        shard_options=grain.NoSharding()
    )
    
    loader = grain.DataLoader(
        data_source=source,
        sampler=sampler,
        worker_count=0,
        operations=[grain.Batch(batch_size=batch_size, drop_remainder=True)]
    )
    
    # Count batches across all epochs
    total_batches = sum(1 for _ in loader)
    expected_batches_per_epoch = num_samples // batch_size
    
    print(f"Samples per epoch: {num_samples}")
    print(f"Batch size: {batch_size}")
    print(f"Expected batches per epoch: {expected_batches_per_epoch}")
    print(f"Number of epochs: {num_epochs}")
    print(f"Total batches: {total_batches}")
    
    return loader


# ============================================================================
# 6. SHARDING FOR DISTRIBUTED TRAINING
# ============================================================================

def create_sharded_loader(batch_size: int = 32, num_devices: int = 4):
    """Create Grain dataloader with sharding for multi-device training."""
    if not GRAIN_AVAILABLE:
        print("Grain not available - skipping")
        return None
    
    print("\n" + "=" * 80)
    print("5. Sharded DataLoader (Multi-Device)")
    print("=" * 80)
    
    num_samples = 10000
    images = np.random.randn(num_samples, 28, 28, 1).astype(np.float32)
    labels = np.random.randint(0, 10, num_samples)
    
    source = NumpyArraySource(images, labels)
    
    # Create loaders for each device
    loaders = []
    for device_id in range(num_devices):
        shard_options = grain.ShardByJaxProcess(drop_remainder=True)
        
        sampler = grain.IndexSampler(
            num_records=len(source),
            shuffle=True,
            seed=42,
            num_epochs=1,
            shard_options=shard_options
        )
        
        loader = grain.DataLoader(
            data_source=source,
            sampler=sampler,
            worker_count=0,
            operations=[grain.Batch(batch_size=batch_size, drop_remainder=True)]
        )
        
        loaders.append(loader)
    
    print(f"Created {num_devices} sharded loaders")
    print(f"Total samples: {num_samples}")
    print(f"Samples per device: ~{num_samples // num_devices}")
    
    # Show first batch from first loader
    batch = next(iter(loaders[0]))
    print(f"Batch shape from device 0: {batch['image'].shape}")
    
    return loaders


# ============================================================================
# 7. CUSTOM DATA SOURCE (FILE-BASED)
# ============================================================================

if GRAIN_AVAILABLE:
    class FileBasedSource(grain.RandomAccessDataSource):
        """Data source that loads from files."""
        
        def __init__(self, file_paths: list, labels: list):
            self.file_paths = file_paths
            self.labels = labels
        
        def __len__(self):
            return len(self.file_paths)
        
        def __getitem__(self, index):
            # In real scenario, load from file here
            # For demo, return dummy data
            image = np.random.randn(28, 28, 1).astype(np.float32)
            return {
                'image': image,
                'label': self.labels[index],
                'path': self.file_paths[index]
            }


def create_file_based_loader(batch_size: int = 16):
    """Create Grain dataloader from file paths."""
    if not GRAIN_AVAILABLE:
        print("Grain not available - skipping")
        return None
    
    print("\n" + "=" * 80)
    print("6. File-Based DataLoader")
    print("=" * 80)
    
    # Simulate file paths
    num_files = 500
    file_paths = [f"/data/image_{i:05d}.png" for i in range(num_files)]
    labels = np.random.randint(0, 10, num_files)
    
    source = FileBasedSource(file_paths, labels)
    
    sampler = grain.IndexSampler(
        num_records=len(source),
        shuffle=True,
        seed=42,
        num_epochs=1,
        shard_options=grain.NoSharding()
    )
    
    loader = grain.DataLoader(
        data_source=source,
        sampler=sampler,
        worker_count=2,
        operations=[grain.Batch(batch_size=batch_size, drop_remainder=True)]
    )
    
    batch = next(iter(loader))
    print(f"Batch image shape: {batch['image'].shape}")
    print(f"Batch label shape: {batch['label'].shape}")
    print(f"Sample paths: {batch['path'][:3]}")
    
    return loader


# ============================================================================
# DEMONSTRATION
# ============================================================================

def main():
    print("=" * 80)
    print("Flax NNX Data Loading with Grain Examples")
    print("=" * 80)
    
    if not GRAIN_AVAILABLE:
        print("\n" + "!" * 80)
        print("Grain is not installed!")
        print("Install with: pip install grain-nightly")
        print("!" * 80)
        return
    
    # Test 1: Simple loader
    simple_loader = create_simple_grain_loader()
    
    # Test 2: Batched loader
    batched_loader = create_batched_grain_loader(batch_size=64)
    
    # Test 3: With transformations
    transform_loader = create_grain_with_transforms(batch_size=32)
    
    # Test 4: Multi-epoch
    multi_epoch_loader = create_multi_epoch_loader(batch_size=32, num_epochs=3)
    
    # Test 5: Sharded (multi-device)
    sharded_loaders = create_sharded_loader(batch_size=32, num_devices=4)
    
    # Test 6: File-based
    file_loader = create_file_based_loader(batch_size=16)
    
    # ========================================================================
    # Grain vs TFDS Comparison
    # ========================================================================
    print("\n" + "=" * 80)
    print("Grain vs TensorFlow Datasets (TFDS)")
    print("=" * 80)
    
    print("""
    Grain Advantages:
    ✓ Pure Python - no TensorFlow dependency
    ✓ Better integration with JAX ecosystem
    ✓ Designed for multi-host training
    ✓ More flexible sharding options
    ✓ Lower memory overhead
    ✓ Easier to debug (pure Python)
    
    TFDS Advantages:
    ✓ Larger collection of ready-to-use datasets
    ✓ More mature and stable
    ✓ Better documentation
    ✓ Built-in data augmentation ops
    
    When to use Grain:
    • Multi-host/multi-device training
    • Custom data pipelines
    • Want to avoid TensorFlow dependency
    • Need fine-grained control over data loading
    
    When to use TFDS:
    • Quick prototyping with standard datasets
    • Single-device training
    • Need extensive data augmentation
    • Already using TensorFlow ecosystem
    """)
    
    # ========================================================================
    # Best Practices
    # ========================================================================
    print("\n" + "=" * 80)
    print("Best Practices for Grain")
    print("=" * 80)
    
    print("""
    1. Worker configuration:
       - Use worker_count > 0 for I/O bound tasks
       - Start with 2-4 workers per host
       - Adjust worker_buffer_size based on memory
    
    2. Sharding for multi-device:
       - Use ShardByJaxProcess for data parallelism
       - Set drop_remainder=True for consistent shapes
       - Ensure data is evenly distributed
    
    3. Transformations:
       - Apply cheap transforms first
       - Batch after expensive transforms
       - Use MapTransform for element-wise ops
    
    4. Performance:
       - Profile your data pipeline
       - Use appropriate buffer sizes
       - Consider caching expensive operations
    
    5. Reproducibility:
       - Always set seed for samplers
       - Control randomness in transforms
       - Document data pipeline configuration
    """)
    
    print("\n" + "=" * 80)
    print("✓ All Grain examples completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
