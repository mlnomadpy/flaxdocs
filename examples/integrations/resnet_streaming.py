"""
Flax NNX: ResNet Training with Streaming Data from HuggingFace
===============================================================
Train ResNet on ImageNet-like data streamed from HuggingFace.
Run: pip install datasets pillow && python integrations/resnet_streaming.py
"""

import jax
import jax.numpy as jnp
from flax import nnx
import optax
import numpy as np
from typing import Dict, Tuple
import time


import sys
from pathlib import Path

# Add parent directory to path for shared imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from datasets import load_dataset
    import datasets
    from PIL import Image
    import io
    DATASETS_AVAILABLE = True
except ImportError:
    print("Warning: datasets/PIL not available. Install: pip install datasets pillow")
    DATASETS_AVAILABLE = False


# ============================================================================
# 1. RESNET BUILDING BLOCKS
# ============================================================================

class ResNetBlock(nnx.Module):
    """Basic ResNet block with skip connection."""
    
    def __init__(self, channels: int, stride: int = 1, rngs: nnx.Rngs = None):
        self.conv1 = nnx.Conv(
            channels, channels, kernel_size=(3, 3),
            strides=(stride, stride), padding='SAME', rngs=rngs
        )
        self.bn1 = nnx.BatchNorm(channels, rngs=rngs)
        
        self.conv2 = nnx.Conv(
            channels, channels, kernel_size=(3, 3),
            strides=(1, 1), padding='SAME', rngs=rngs
        )
        self.bn2 = nnx.BatchNorm(channels, rngs=rngs)
        
        # Skip connection (if stride != 1, downsample)
        self.downsample = None
        if stride != 1:
            self.downsample = nnx.Conv(
                channels, channels, kernel_size=(1, 1),
                strides=(stride, stride), rngs=rngs
            )
            self.bn_downsample = nnx.BatchNorm(channels, rngs=rngs)
    
    def __call__(self, x, train: bool = False):
        identity = x
        
        # First conv block
        out = self.conv1(x)
        out = self.bn1(out, use_running_average=not train)
        out = nnx.relu(out)
        
        # Second conv block
        out = self.conv2(out)
        out = self.bn2(out, use_running_average=not train)
        
        # Skip connection
        if self.downsample is not None:
            identity = self.downsample(x)
            identity = self.bn_downsample(identity, use_running_average=not train)
        
        out = out + identity
        out = nnx.relu(out)
        
        return out


class ResNet(nnx.Module):
    """ResNet architecture for image classification."""
    
    def __init__(self, num_classes: int, num_blocks: list = [2, 2, 2, 2],
                 channels: list = [64, 128, 256, 512], rngs: nnx.Rngs = None):
        # Initial conv layer
        self.conv1 = nnx.Conv(
            3, channels[0], kernel_size=(7, 7),
            strides=(2, 2), padding='SAME', rngs=rngs
        )
        self.bn1 = nnx.BatchNorm(channels[0], rngs=rngs)
        
        # ResNet layers
        self.layers = []
        for i, (num_block, channel) in enumerate(zip(num_blocks, channels)):
            stride = 1 if i == 0 else 2
            for j in range(num_block):
                block_stride = stride if j == 0 else 1
                self.layers.append(ResNetBlock(channel, block_stride, rngs))
        
        # Global average pooling and classifier
        self.fc = nnx.Linear(channels[-1], num_classes, rngs=rngs)
    
    def __call__(self, x, train: bool = False):
        # Initial conv
        x = self.conv1(x)
        x = self.bn1(x, use_running_average=not train)
        x = nnx.relu(x)
        x = nnx.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding='SAME')
        
        # ResNet blocks
        for layer in self.layers:
            x = layer(x, train=train)
        
        # Global average pooling
        x = jnp.mean(x, axis=(1, 2))
        
        # Classifier
        x = self.fc(x)
        
        return x


# ============================================================================
# 2. DATA PREPROCESSING
# ============================================================================

def preprocess_image(example, image_size: Tuple[int, int] = (224, 224)):
    """Preprocess image from dataset."""
    try:
        # Handle different image formats
        if 'image' in example:
            img = example['image']
            
            # Convert to PIL Image if needed
            if isinstance(img, dict) and 'bytes' in img:
                img = Image.open(io.BytesIO(img['bytes']))
            elif not isinstance(img, Image.Image):
                # Try to convert
                img = Image.fromarray(np.array(img))
            
            # Convert to RGB
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize
            img = img.resize(image_size, Image.BILINEAR)
            
            # Convert to array and normalize
            img_array = np.array(img).astype(np.float32) / 255.0
            
            # Data augmentation (training only)
            # Random horizontal flip (50% chance)
            if np.random.rand() > 0.5:
                img_array = np.fliplr(img_array)
            
            example['pixel_values'] = img_array
            
            # Get label
            if 'label' in example:
                example['labels'] = example['label']
            
            return example
    
    except Exception as e:
        # Return None for failed preprocessing
        print(f"Warning: Failed to preprocess image: {e}")
        return None


# ============================================================================
# 3. STREAMING DATALOADER
# ============================================================================

class StreamingDataLoader:
    """DataLoader for streaming datasets from HuggingFace."""
    
    def __init__(self, dataset_name: str, split: str = "train",
                 batch_size: int = 32, image_size: Tuple[int, int] = (224, 224),
                 shuffle_buffer: int = 1000):
        self.dataset_name = dataset_name
        self.split = split
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle_buffer = shuffle_buffer
        
        # Load dataset
        print(f"Loading {dataset_name} ({split}) in streaming mode...")
        self.dataset = load_dataset(
            dataset_name,
            split=split,
            streaming=True,
            trust_remote_code=True
        )
        
        # Shuffle if needed
        if shuffle_buffer > 0:
            self.dataset = self.dataset.shuffle(buffer_size=shuffle_buffer)
    
    def __iter__(self):
        batch_images = []
        batch_labels = []
        
        for example in self.dataset:
            # Preprocess
            processed = preprocess_image(example, self.image_size)
            
            if processed is None or 'pixel_values' not in processed:
                continue
            
            batch_images.append(processed['pixel_values'])
            batch_labels.append(processed.get('labels', 0))
            
            if len(batch_images) >= self.batch_size:
                # Yield batch
                yield {
                    'pixel_values': jnp.array(np.stack(batch_images)),
                    'labels': jnp.array(batch_labels)
                }
                batch_images = []
                batch_labels = []


# ============================================================================
# 4. TRAINING FUNCTIONS
# ============================================================================

def compute_loss(logits, labels, num_classes: int):
    """Compute cross-entropy loss."""
    one_hot_labels = jax.nn.one_hot(labels, num_classes)
    return -jnp.mean(jnp.sum(one_hot_labels * jax.nn.log_softmax(logits), axis=-1))


def compute_metrics(logits, labels):
    """Compute accuracy."""
    predictions = jnp.argmax(logits, axis=-1)
    accuracy = jnp.mean(predictions == labels)
    return accuracy


@nnx.jit
def train_step(model: ResNet, optimizer: nnx.Optimizer, batch: Dict,
               num_classes: int):
    """Single training step."""
    
    def loss_fn(model):
        logits = model(batch['pixel_values'], train=True)
        loss = compute_loss(logits, batch['labels'], num_classes)
        return loss, logits
    
    # Compute gradients
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model)
    
    # Update parameters
    optimizer.update(grads)
    
    # Compute metrics
    accuracy = compute_metrics(logits, batch['labels'])
    
    return {'loss': loss, 'accuracy': accuracy}


# ============================================================================
# 5. MAIN TRAINING LOOP
# ============================================================================

def train_resnet_streaming(
    dataset_name: str = "cifar10",
    num_classes: int = 10,
    image_size: Tuple[int, int] = (224, 224),
    batch_size: int = 32,
    num_steps: int = 1000,
    learning_rate: float = 1e-3,
    seed: int = 42
):
    """Train ResNet with streaming data."""
    print("=" * 80)
    print("ResNet Training with Streaming Data")
    print("=" * 80)
    
    print(f"\nConfiguration:")
    print(f"  Dataset: {dataset_name}")
    print(f"  Num classes: {num_classes}")
    print(f"  Image size: {image_size}")
    print(f"  Batch size: {batch_size}")
    print(f"  Num steps: {num_steps}")
    print(f"  Learning rate: {learning_rate}")
    
    # ========================================================================
    # Initialize Model
    # ========================================================================
    print("\nInitializing ResNet...")
    rngs = nnx.Rngs(seed)
    
    # Smaller ResNet for demo (ResNet-18 style)
    model = ResNet(
        num_classes=num_classes,
        num_blocks=[2, 2, 2, 2],
        channels=[64, 128, 256, 512],
        rngs=rngs
    )
    
    # Count parameters
    state = nnx.state(model)
    total_params = sum(x.size for x in jax.tree_util.tree_leaves(state))
    print(f"Total parameters: {total_params:,}")
    
    # ========================================================================
    # Initialize Optimizer
    # ========================================================================
    optimizer = nnx.Optimizer(model, optax.adam(learning_rate))
    
    # ========================================================================
    # Create Streaming DataLoader
    # ========================================================================
    print("\nCreating streaming dataloader...")
    dataloader = StreamingDataLoader(
        dataset_name=dataset_name,
        split="train",
        batch_size=batch_size,
        image_size=image_size,
        shuffle_buffer=1000
    )
    
    # ========================================================================
    # Training Loop
    # ========================================================================
    print("\n" + "=" * 80)
    print("Training")
    print("=" * 80)
    
    step = 0
    running_loss = 0.0
    running_acc = 0.0
    start_time = time.time()
    
    for batch in dataloader:
        # Training step
        metrics = train_step(model, optimizer, batch, num_classes)
        
        running_loss += float(metrics['loss'])
        running_acc += float(metrics['accuracy'])
        step += 1
        
        # Log progress
        if step % 10 == 0:
            avg_loss = running_loss / 10
            avg_acc = running_acc / 10
            elapsed = time.time() - start_time
            steps_per_sec = 10 / elapsed
            
            print(f"Step {step}/{num_steps} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Acc: {avg_acc:.4f} | "
                  f"Speed: {steps_per_sec:.2f} steps/s")
            
            running_loss = 0.0
            running_acc = 0.0
            start_time = time.time()
        
        # Stop after num_steps
        if step >= num_steps:
            break
    
    print("\n" + "=" * 80)
    print("✓ Training Complete!")
    print("=" * 80)
    
    return model


# ============================================================================
# 6. EXAMPLE DATASETS
# ============================================================================

def demo_available_datasets():
    """Show available image datasets on HuggingFace."""
    print("\n" + "=" * 80)
    print("Popular Image Datasets on HuggingFace Hub")
    print("=" * 80)
    
    datasets_info = [
        ("cifar10", "10 classes, 32x32", "60K images"),
        ("cifar100", "100 classes, 32x32", "60K images"),
        ("food101", "101 food categories", "101K images"),
        ("imagenet-1k", "1000 classes", "1.2M images"),
        ("cats_vs_dogs", "Binary classification", "25K images"),
        ("fashion_mnist", "10 fashion categories", "70K images"),
        ("oxford_flowers102", "102 flower species", "8K images"),
        ("stanford_dogs", "120 dog breeds", "20K images"),
        ("scene_parse_150", "Scene parsing", "20K images"),
    ]
    
    print("\n{:<25} {:<25} {:<15}".format("Dataset", "Description", "Size"))
    print("-" * 65)
    for name, desc, size in datasets_info:
        print(f"{name:<25} {desc:<25} {size:<15}")


# ============================================================================
# DEMONSTRATION
# ============================================================================

def main():
    print("=" * 80)
    print("Flax NNX: ResNet Training with Streaming Data")
    print("=" * 80)
    
    if not DATASETS_AVAILABLE:
        print("\n" + "!" * 80)
        print("Required libraries not available!")
        print("Install with: pip install datasets pillow")
        print("!" * 80)
        return
    
    # Show available datasets
    demo_available_datasets()
    
    # ========================================================================
    # Train on CIFAR-10 (lightweight for demo)
    # ========================================================================
    print("\n" + "=" * 80)
    print("Example: Training ResNet on CIFAR-10 (Streaming)")
    print("=" * 80)
    
    model = train_resnet_streaming(
        dataset_name="cifar10",
        num_classes=10,
        image_size=(224, 224),  # Upsample CIFAR-10 to ImageNet size
        batch_size=16,
        num_steps=100,  # Small number for demo
        learning_rate=1e-3,
        seed=42
    )
    
    # ========================================================================
    # Test Inference
    # ========================================================================
    print("\n" + "=" * 80)
    print("Testing Inference")
    print("=" * 80)
    
    # Create random test image
    test_image = jnp.ones((1, 224, 224, 3))
    logits = model(test_image, train=False)
    prediction = jnp.argmax(logits, axis=-1)
    
    print(f"Test image shape: {test_image.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Predicted class: {prediction[0]}")
    
    # ========================================================================
    # Best Practices
    # ========================================================================
    print("\n" + "=" * 80)
    print("Best Practices for Streaming Training")
    print("=" * 80)
    
    print("""
    1. Memory Efficiency:
       ✓ Stream data instead of loading everything
       ✓ Process batches on-the-fly
       ✓ Use appropriate image sizes
       ✓ Implement gradient accumulation if needed
    
    2. Data Augmentation:
       ✓ Apply augmentation during preprocessing
       ✓ Random crops, flips, color jitter
       ✓ Use imgaug or albumentations libraries
       ✓ Cache preprocessed data when possible
    
    3. Performance:
       ✓ Use shuffle buffer for randomness
       ✓ Prefetch batches
       ✓ Use multiple workers if I/O bound
       ✓ JIT compile training step
    
    4. For Large Datasets:
       ✓ ImageNet: Use streaming (1.2M images)
       ✓ Monitor data loading speed
       ✓ Balance preprocessing and training time
       ✓ Consider caching frequent samples
    
    5. Training Tips:
       ✓ Start with smaller resolution
       ✓ Use learning rate warmup
       ✓ Monitor validation metrics
       ✓ Save checkpoints regularly
       ✓ Log to W&B or TensorBoard
    """)
    
    print("\n" + "=" * 80)
    print("✓ All examples completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
