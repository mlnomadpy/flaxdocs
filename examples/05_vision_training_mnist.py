"""
Flax NNX: End-to-End Vision Model Training (MNIST)
===================================================
Complete example of training a CNN on MNIST from scratch.
Run: python 05_vision_training_mnist.py
"""

import jax
import jax.numpy as jnp
from flax import nnx
import optax
import tensorflow_datasets as tfds
import tensorflow as tf
from typing import Dict, Any
import time
from functools import partial

# Disable GPU for TensorFlow
tf.config.set_visible_devices([], 'GPU')


# ============================================================================
# 1. MODEL DEFINITION
# ============================================================================

class CNN(nnx.Module):
    """Convolutional Neural Network for MNIST."""
    
    def __init__(self, num_classes: int = 10, rngs: nnx.Rngs = None):
        # Conv layers
        self.conv1 = nnx.Conv(1, 32, kernel_size=(3, 3), rngs=rngs)
        self.conv2 = nnx.Conv(32, 64, kernel_size=(3, 3), rngs=rngs)
        
        # Batch normalization
        self.bn1 = nnx.BatchNorm(32, rngs=rngs)
        self.bn2 = nnx.BatchNorm(64, rngs=rngs)
        
        # Dense layers
        self.fc1 = nnx.Linear(64 * 5 * 5, 128, rngs=rngs)
        self.fc2 = nnx.Linear(128, num_classes, rngs=rngs)
        
        # Dropout
        self.dropout = nnx.Dropout(rate=0.5, rngs=rngs)
    
    def __call__(self, x, train: bool = False):
        # Conv block 1
        x = self.conv1(x)
        x = self.bn1(x, use_running_average=not train)
        x = nnx.relu(x)
        x = nnx.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x, use_running_average=not train)
        x = nnx.relu(x)
        x = nnx.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        # Flatten
        x = x.reshape((x.shape[0], -1))
        
        # Dense layers
        x = self.fc1(x)
        x = nnx.relu(x)
        x = self.dropout(x, deterministic=not train)
        x = self.fc2(x)
        
        return x


# ============================================================================
# 2. DATA LOADING
# ============================================================================

def load_data(batch_size: int = 128):
    """Load and preprocess MNIST dataset."""
    print("Loading MNIST dataset...")
    
    def preprocess(example):
        image = tf.cast(example['image'], tf.float32) / 255.0
        label = example['label']
        return {'image': image, 'label': label}
    
    # Training data
    train_ds = tfds.load('mnist', split='train', shuffle_files=True)
    train_ds = train_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.cache()
    train_ds = train_ds.shuffle(10000)
    train_ds = train_ds.batch(batch_size)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    
    # Test data
    test_ds = tfds.load('mnist', split='test')
    test_ds = test_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.batch(batch_size)
    test_ds = test_ds.cache()
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)
    
    print(f"✓ Data loaded successfully")
    return train_ds, test_ds


# ============================================================================
# 3. LOSS FUNCTION
# ============================================================================

def cross_entropy_loss(logits, labels):
    """Compute cross-entropy loss."""
    one_hot_labels = jax.nn.one_hot(labels, num_classes=10)
    return -jnp.mean(jnp.sum(one_hot_labels * jax.nn.log_softmax(logits), axis=-1))


# ============================================================================
# 4. METRICS
# ============================================================================

def compute_metrics(logits, labels):
    """Compute accuracy and loss."""
    loss = cross_entropy_loss(logits, labels)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    return {'loss': loss, 'accuracy': accuracy}


# ============================================================================
# 5. TRAINING STEP
# ============================================================================

@nnx.jit
def train_step(model: CNN, optimizer: nnx.Optimizer, batch: Dict[str, jnp.ndarray]):
    """Single training step."""
    
    def loss_fn(model: CNN):
        logits = model(batch['image'], train=True)
        loss = cross_entropy_loss(logits, batch['label'])
        return loss, logits
    
    # Compute gradients
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model)
    
    # Update parameters
    optimizer.update(grads)
    
    # Compute metrics
    metrics = compute_metrics(logits, batch['label'])
    
    return metrics


# ============================================================================
# 6. EVALUATION STEP
# ============================================================================

@nnx.jit
def eval_step(model: CNN, batch: Dict[str, jnp.ndarray]):
    """Single evaluation step."""
    logits = model(batch['image'], train=False)
    return compute_metrics(logits, batch['label'])


# ============================================================================
# 7. TRAINING LOOP
# ============================================================================

def train_epoch(model: CNN, optimizer: nnx.Optimizer, train_ds, epoch: int):
    """Train for one epoch."""
    batch_metrics = []
    
    for batch in tfds.as_numpy(train_ds):
        # Convert to JAX arrays
        batch = {
            'image': jnp.array(batch['image']),
            'label': jnp.array(batch['label'])
        }
        
        metrics = train_step(model, optimizer, batch)
        batch_metrics.append(metrics)
    
    # Average metrics across batches
    epoch_metrics = {
        k: jnp.mean(jnp.array([m[k] for m in batch_metrics]))
        for k in batch_metrics[0].keys()
    }
    
    return epoch_metrics


def evaluate(model: CNN, test_ds):
    """Evaluate on test set."""
    batch_metrics = []
    
    for batch in tfds.as_numpy(test_ds):
        batch = {
            'image': jnp.array(batch['image']),
            'label': jnp.array(batch['label'])
        }
        
        metrics = eval_step(model, batch)
        batch_metrics.append(metrics)
    
    # Average metrics
    eval_metrics = {
        k: jnp.mean(jnp.array([m[k] for m in batch_metrics]))
        for k in batch_metrics[0].keys()
    }
    
    return eval_metrics


# ============================================================================
# 8. MAIN TRAINING FUNCTION
# ============================================================================

def main():
    print("=" * 80)
    print("Flax NNX: End-to-End Vision Model Training")
    print("=" * 80)
    
    # ========================================================================
    # Configuration
    # ========================================================================
    config = {
        'batch_size': 128,
        'num_epochs': 10,
        'learning_rate': 1e-3,
        'seed': 42,
    }
    
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # ========================================================================
    # Load Data
    # ========================================================================
    train_ds, test_ds = load_data(config['batch_size'])
    
    # ========================================================================
    # Initialize Model
    # ========================================================================
    print("\nInitializing model...")
    rngs = nnx.Rngs(config['seed'])
    model = CNN(num_classes=10, rngs=rngs)
    
    # Count parameters
    state = nnx.state(model)
    total_params = sum(x.size for x in jax.tree_util.tree_leaves(state))
    print(f"Total parameters: {total_params:,}")
    
    # ========================================================================
    # Initialize Optimizer
    # ========================================================================
    print("\nInitializing optimizer...")
    optimizer = nnx.Optimizer(model, optax.adam(config['learning_rate']))
    
    # ========================================================================
    # Training Loop
    # ========================================================================
    print("\n" + "=" * 80)
    print("Training")
    print("=" * 80)
    
    best_accuracy = 0.0
    
    for epoch in range(config['num_epochs']):
        start_time = time.time()
        
        # Train
        train_metrics = train_epoch(model, optimizer, train_ds, epoch)
        
        # Evaluate
        eval_metrics = evaluate(model, test_ds)
        
        epoch_time = time.time() - start_time
        
        # Print metrics
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']} ({epoch_time:.2f}s)")
        print(f"  Train Loss: {train_metrics['loss']:.4f}, "
              f"Train Acc: {train_metrics['accuracy']:.4f}")
        print(f"  Test Loss:  {eval_metrics['loss']:.4f}, "
              f"Test Acc:  {eval_metrics['accuracy']:.4f}")
        
        # Track best model
        if eval_metrics['accuracy'] > best_accuracy:
            best_accuracy = eval_metrics['accuracy']
            print(f"  ✓ New best accuracy: {best_accuracy:.4f}")
    
    # ========================================================================
    # Final Results
    # ========================================================================
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"Best Test Accuracy: {best_accuracy:.4f}")
    
    # ========================================================================
    # Test Inference
    # ========================================================================
    print("\n" + "=" * 80)
    print("Testing Inference")
    print("=" * 80)
    
    # Get a single batch
    test_batch = next(iter(tfds.as_numpy(test_ds)))
    test_images = jnp.array(test_batch['image'])
    test_labels = jnp.array(test_batch['label'])
    
    # Make predictions
    logits = model(test_images, train=False)
    predictions = jnp.argmax(logits, axis=-1)
    
    print(f"Batch size: {len(test_labels)}")
    print(f"Sample predictions: {predictions[:10]}")
    print(f"Sample labels:      {test_labels[:10]}")
    print(f"Sample accuracy:    {jnp.mean(predictions == test_labels):.4f}")
    
    # ========================================================================
    # Model Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("Model Summary")
    print("=" * 80)
    
    # Test with single image
    single_image = test_images[0:1]
    print(f"Input shape:  {single_image.shape}")
    
    output = model(single_image, train=False)
    print(f"Output shape: {output.shape}")
    print(f"Output (logits): {output[0]}")
    print(f"Prediction: {jnp.argmax(output[0])}")
    print(f"Confidence: {jax.nn.softmax(output[0])[jnp.argmax(output[0])]:.4f}")
    
    print("\n" + "=" * 80)
    print("✓ Training completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
