"""
Flax NNX: End-to-End Vision Model Training (MNIST)
===================================================
Complete example of training a CNN on MNIST using shared components.

Run: python training/vision_mnist.py
"""

import jax
import jax.numpy as jnp
from flax import nnx
import tensorflow_datasets as tfds
import tensorflow as tf
from typing import Dict
import time
import sys
from pathlib import Path

# Add parent directory to path for shared imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.models import CNN
from shared.training_utils import (
    create_train_step,
    create_eval_step,
    create_optimizer
)

# Disable GPU for TensorFlow (we use JAX for computation)
tf.config.set_visible_devices([], 'GPU')


# ============================================================================
# DATA LOADING
# ============================================================================

def load_mnist_data(batch_size: int = 128):
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
# TRAINING LOOP
# ============================================================================

def train_epoch(model, optimizer, train_ds, train_step_fn):
    """Train for one epoch."""
    total_loss = 0.0
    total_accuracy = 0.0
    num_batches = 0
    
    for batch in train_ds:
        batch_dict = {
            'x': jnp.array(batch['image']),
            'y': jnp.array(batch['label'])
        }
        
        loss, metrics = train_step_fn(model, optimizer, batch_dict)
        
        total_loss += float(loss)
        total_accuracy += float(metrics['accuracy'])
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_accuracy = total_accuracy / num_batches
    
    return avg_loss, avg_accuracy


def evaluate(model, test_ds, eval_step_fn):
    """Evaluate on test set."""
    total_loss = 0.0
    total_accuracy = 0.0
    num_batches = 0
    
    for batch in test_ds:
        batch_dict = {
            'x': jnp.array(batch['image']),
            'y': jnp.array(batch['label'])
        }
        
        loss, metrics = eval_step_fn(model, batch_dict)
        
        total_loss += float(loss)
        total_accuracy += float(metrics['accuracy'])
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_accuracy = total_accuracy / num_batches
    
    return avg_loss, avg_accuracy


# ============================================================================
# MAIN TRAINING SCRIPT
# ============================================================================

def main():
    """Main training function."""
    print("\n" + "=" * 70)
    print("TRAINING CNN ON MNIST - Using Shared Components")
    print("=" * 70 + "\n")
    
    # Hyperparameters
    batch_size = 128
    learning_rate = 0.001
    num_epochs = 3
    
    print("Hyperparameters:")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Epochs: {num_epochs}\n")
    
    # Load data
    train_ds, test_ds = load_mnist_data(batch_size)
    
    # Create model using shared component
    print("Creating CNN model from shared.models...")
    rngs = nnx.Rngs(0)
    model = CNN(num_classes=10, rngs=rngs)
    
    # Count parameters
    params = nnx.state(model, nnx.Param)
    total_params = sum(p.size for p in jax.tree.leaves(params))
    print(f"Total parameters: {total_params:,}")
    print("✓ Model created\n")
    
    # Create optimizer using shared utility
    print("Creating optimizer...")
    optimizer = create_optimizer(model, learning_rate, optimizer_name='adam')
    print("✓ Optimizer created\n")
    
    # Create training and eval steps using shared utilities
    print("Creating train/eval step functions...")
    train_step_fn = create_train_step(loss_fn_name='cross_entropy')
    eval_step_fn = create_eval_step(loss_fn_name='cross_entropy')
    print("✓ Step functions created (JIT compiled)\n")
    
    # Training loop
    print("=" * 70)
    print("TRAINING")
    print("=" * 70)
    
    best_test_acc = 0.0
    
    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(model, optimizer, train_ds, train_step_fn)
        
        # Evaluate
        test_loss, test_acc = evaluate(model, test_ds, eval_step_fn)
        
        epoch_time = time.time() - epoch_start
        
        # Update best accuracy
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_marker = " 🏆"
        else:
            best_marker = ""
        
        # Print results
        print(f"Epoch {epoch:2d}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}{best_marker} | "
              f"Time: {epoch_time:.2f}s")
    
    # Final results
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best test accuracy: {best_test_acc:.4f} ({best_test_acc*100:.2f}%)")
    print("\nKey Components Used:")
    print("  ✓ shared.models.CNN - CNN architecture")
    print("  ✓ shared.training_utils.create_optimizer - Optimizer setup")
    print("  ✓ shared.training_utils.create_train_step - Training step")
    print("  ✓ shared.training_utils.create_eval_step - Evaluation step")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
