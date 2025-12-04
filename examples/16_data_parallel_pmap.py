"""
Flax NNX: Data Parallelism with jax.pmap
==========================================
Complete guide to data parallelism using jax.pmap for training on multiple devices.
Demonstrates synchronous data parallel training with gradient synchronization.

Run: python 16_data_parallel_pmap.py

Key Concepts:
- pmap replicates computation across devices
- Each device processes a shard of the batch
- Gradients are synchronized using pmean
- Model parameters are replicated on all devices
"""

import jax
import jax.numpy as jnp
from flax import nnx
import optax
import numpy as np
from typing import Dict, Tuple
from functools import partial


# ============================================================================
# 1. MODEL DEFINITION
# ============================================================================

class CNNClassifier(nnx.Module):
    """Simple CNN for demonstrating data parallelism."""
    
    def __init__(self, num_classes: int = 10, rngs: nnx.Rngs = None):
        # Convolutional layers
        self.conv1 = nnx.Conv(3, 32, kernel_size=(3, 3), padding='SAME', rngs=rngs)
        self.conv2 = nnx.Conv(32, 64, kernel_size=(3, 3), padding='SAME', rngs=rngs)
        self.conv3 = nnx.Conv(64, 128, kernel_size=(3, 3), padding='SAME', rngs=rngs)
        
        # Batch normalization
        self.bn1 = nnx.BatchNorm(32, rngs=rngs)
        self.bn2 = nnx.BatchNorm(64, rngs=rngs)
        self.bn3 = nnx.BatchNorm(128, rngs=rngs)
        
        # Dense layers
        self.dense1 = nnx.Linear(128 * 4 * 4, 256, rngs=rngs)
        self.dense2 = nnx.Linear(256, num_classes, rngs=rngs)
        
        self.dropout = nnx.Dropout(0.5, rngs=rngs)
    
    def __call__(self, x, train: bool = False):
        # Conv block 1: 32x32 -> 16x16
        x = self.conv1(x)
        x = self.bn1(x, use_running_average=not train)
        x = nnx.relu(x)
        x = nnx.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        # Conv block 2: 16x16 -> 8x8
        x = self.conv2(x)
        x = self.bn2(x, use_running_average=not train)
        x = nnx.relu(x)
        x = nnx.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        # Conv block 3: 8x8 -> 4x4
        x = self.conv3(x)
        x = self.bn3(x, use_running_average=not train)
        x = nnx.relu(x)
        x = nnx.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        # Flatten and dense layers
        x = x.reshape(x.shape[0], -1)
        x = self.dense1(x)
        x = nnx.relu(x)
        x = self.dropout(x, deterministic=not train)
        x = self.dense2(x)
        
        return x


# ============================================================================
# 2. DATA PARALLEL TRAINING WITH PMAP
# ============================================================================

def create_train_state(model, learning_rate: float, momentum: float):
    """Create optimizer state for the model."""
    optimizer = optax.sgd(learning_rate=learning_rate, momentum=momentum)
    opt_state = nnx.Optimizer(model, optimizer)
    return opt_state


def loss_fn(model: CNNClassifier, batch: Dict, train: bool = True):
    """Compute cross-entropy loss."""
    logits = model(batch['image'], train=train)
    
    # One-hot encode labels
    labels_onehot = jax.nn.one_hot(batch['label'], num_classes=10)
    
    # Cross-entropy loss
    loss = optax.softmax_cross_entropy(logits, labels_onehot).mean()
    
    # Compute accuracy
    predictions = jnp.argmax(logits, axis=-1)
    accuracy = jnp.mean(predictions == batch['label'])
    
    return loss, {'accuracy': accuracy}


@partial(jax.pmap, axis_name='devices')
def train_step_pmap(state: nnx.Optimizer, batch: Dict):
    """
    Single training step with data parallelism using pmap.
    
    Key points:
    1. This function is replicated across all devices
    2. Each device gets a shard of the batch (batch dimension is split)
    3. Gradients are computed locally on each device
    4. lax.pmean synchronizes gradients across devices (all-reduce)
    5. All devices update their parameters identically
    
    Args:
        state: Optimizer state (replicated on each device)
        batch: Data batch (sharded across first dimension)
    
    Returns:
        Updated state and metrics (per-device values)
    """
    
    def compute_loss(model):
        return loss_fn(model, batch, train=True)
    
    # Compute gradients on this device's batch shard
    grad_fn = nnx.value_and_grad(compute_loss, has_aux=True)
    (loss, metrics), grads = grad_fn(state.model)
    
    # CRITICAL: Synchronize gradients across all devices
    # pmean computes the mean of gradients across the 'devices' axis
    # This is equivalent to all-reduce with averaging
    grads = jax.lax.pmean(grads, axis_name='devices')
    
    # Also average loss and metrics for consistent logging
    loss = jax.lax.pmean(loss, axis_name='devices')
    metrics = jax.tree.map(lambda x: jax.lax.pmean(x, axis_name='devices'), metrics)
    
    # Update parameters (identical update on all devices)
    state.update(grads)
    
    return state, loss, metrics


@partial(jax.pmap, axis_name='devices')
def eval_step_pmap(model: CNNClassifier, batch: Dict):
    """
    Evaluation step with data parallelism.
    
    Similar to training, but:
    - No gradient computation
    - Model in eval mode (affects dropout, batch norm)
    """
    logits = model(batch['image'], train=False)
    labels_onehot = jax.nn.one_hot(batch['label'], num_classes=10)
    
    loss = optax.softmax_cross_entropy(logits, labels_onehot).mean()
    predictions = jnp.argmax(logits, axis=-1)
    accuracy = jnp.mean(predictions == batch['label'])
    
    # Average metrics across devices
    loss = jax.lax.pmean(loss, axis_name='devices')
    accuracy = jax.lax.pmean(accuracy, axis_name='devices')
    
    return loss, accuracy


# ============================================================================
# 3. DATA PREPARATION FOR PMAP
# ============================================================================

def shard_batch(batch: Dict, num_devices: int) -> Dict:
    """
    Reshape batch to add device dimension for pmap.
    
    Input shape: (total_batch_size, ...)
    Output shape: (num_devices, per_device_batch_size, ...)
    
    pmap expects the first dimension to be the device axis.
    Each device will receive one slice along this dimension.
    
    Example:
        batch_size = 128, num_devices = 8
        Input: (128, 32, 32, 3)
        Output: (8, 16, 32, 32, 3)
        Each device processes 16 images
    """
    def reshape_for_devices(x):
        # Check divisibility
        batch_size = x.shape[0]
        assert batch_size % num_devices == 0, \
            f"Batch size {batch_size} must be divisible by num_devices {num_devices}"
        
        per_device_batch_size = batch_size // num_devices
        # Add device dimension as first axis
        return x.reshape((num_devices, per_device_batch_size) + x.shape[1:])
    
    return jax.tree.map(reshape_for_devices, batch)


def replicate_across_devices(pytree, num_devices: int):
    """
    Replicate a pytree across devices for pmap.
    
    Adds a leading dimension of size num_devices with identical copies.
    Used to replicate model parameters, optimizer state, etc.
    """
    return jax.tree.map(
        lambda x: jnp.broadcast_to(x, (num_devices,) + x.shape),
        pytree
    )


def unreplicate_from_devices(pytree):
    """
    Remove device dimension by taking first device's value.
    
    All devices should have identical values after pmean synchronization,
    so we can just take the first one.
    """
    return jax.tree.map(lambda x: x[0], pytree)


# ============================================================================
# 4. TRAINING LOOP
# ============================================================================

def train_data_parallel():
    """Complete training example with data parallelism."""
    
    # Check available devices
    num_devices = jax.local_device_count()
    print(f"\n{'='*70}")
    print(f"DATA PARALLELISM WITH PMAP")
    print(f"{'='*70}")
    print(f"Available devices: {num_devices}")
    print(f"Device types: {[d.platform for d in jax.local_devices()]}")
    
    if num_devices == 1:
        print("\nWarning: Only 1 device available. pmap will still work but won't")
        print("demonstrate true parallelism. Results will be the same as without pmap.")
    
    # Hyperparameters
    per_device_batch_size = 32
    total_batch_size = per_device_batch_size * num_devices
    num_epochs = 5
    learning_rate = 0.1
    momentum = 0.9
    
    print(f"\nTraining Configuration:")
    print(f"  Per-device batch size: {per_device_batch_size}")
    print(f"  Total batch size: {total_batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Momentum: {momentum}")
    
    # Initialize model on host
    print("\n" + "="*70)
    print("INITIALIZATION")
    print("="*70)
    rng = jax.random.PRNGKey(0)
    rngs = nnx.Rngs(0)
    
    # Create model (on host/first device)
    model = CNNClassifier(num_classes=10, rngs=rngs)
    
    # Create optimizer state
    optimizer_state = create_train_state(model, learning_rate, momentum)
    
    # Replicate state across all devices
    print(f"Replicating model across {num_devices} devices...")
    
    # Extract state for replication
    graphdef, state_arrays = nnx.split(optimizer_state)
    
    # Replicate state
    replicated_state = replicate_across_devices(state_arrays, num_devices)
    
    # Merge back
    optimizer_state = nnx.merge(graphdef, replicated_state)
    
    print("✓ Model replicated")
    
    # Generate synthetic data
    print("\n" + "="*70)
    print("SYNTHETIC DATA GENERATION")
    print("="*70)
    
    num_train_samples = 1000
    num_eval_samples = 200
    
    rng, data_rng = jax.random.split(rng)
    train_images = jax.random.normal(data_rng, (num_train_samples, 32, 32, 3))
    train_labels = jax.random.randint(data_rng, (num_train_samples,), 0, 10)
    
    rng, data_rng = jax.random.split(rng)
    eval_images = jax.random.normal(data_rng, (num_eval_samples, 32, 32, 3))
    eval_labels = jax.random.randint(data_rng, (num_eval_samples,), 0, 10)
    
    print(f"Training samples: {num_train_samples}")
    print(f"Evaluation samples: {num_eval_samples}")
    
    # Training loop
    print("\n" + "="*70)
    print("TRAINING WITH DATA PARALLELISM")
    print("="*70)
    
    num_steps = (num_train_samples // total_batch_size) * num_epochs
    step = 0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 70)
        
        # Shuffle training data
        rng, shuffle_rng = jax.random.split(rng)
        perm = jax.random.permutation(shuffle_rng, num_train_samples)
        train_images_shuffled = train_images[perm]
        train_labels_shuffled = train_labels[perm]
        
        epoch_loss = 0.0
        epoch_acc = 0.0
        num_batches = 0
        
        # Training steps
        for i in range(0, num_train_samples, total_batch_size):
            if i + total_batch_size > num_train_samples:
                break
            
            # Get batch
            batch = {
                'image': train_images_shuffled[i:i + total_batch_size],
                'label': train_labels_shuffled[i:i + total_batch_size]
            }
            
            # Shard batch for pmap (add device dimension)
            batch_sharded = shard_batch(batch, num_devices)
            
            # Training step (executes in parallel on all devices)
            optimizer_state, loss, metrics = train_step_pmap(optimizer_state, batch_sharded)
            
            # Accumulate metrics (loss and metrics are already averaged across devices)
            # Take first device's value since all are identical after pmean
            epoch_loss += float(loss[0])
            epoch_acc += float(metrics['accuracy'][0])
            num_batches += 1
            step += 1
            
            if num_batches % 5 == 0:
                avg_loss = epoch_loss / num_batches
                avg_acc = epoch_acc / num_batches
                print(f"  Step {step:3d} | Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f}")
        
        # Epoch summary
        avg_loss = epoch_loss / num_batches
        avg_acc = epoch_acc / num_batches
        print(f"\n  Epoch {epoch + 1} Summary: Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f}")
        
        # Evaluation
        print(f"\n  Evaluating...")
        eval_loss = 0.0
        eval_acc = 0.0
        num_eval_batches = 0
        
        # Unreplicate model for evaluation (extract from optimizer)
        graphdef, state_arrays = nnx.split(optimizer_state.model)
        unreplicated_state = unreplicate_from_devices(state_arrays)
        eval_model = nnx.merge(graphdef, unreplicated_state)
        
        # Replicate eval model for pmap
        graphdef, state_arrays = nnx.split(eval_model)
        replicated_eval_state = replicate_across_devices(state_arrays, num_devices)
        eval_model_replicated = nnx.merge(graphdef, replicated_eval_state)
        
        for i in range(0, num_eval_samples, total_batch_size):
            if i + total_batch_size > num_eval_samples:
                break
            
            batch = {
                'image': eval_images[i:i + total_batch_size],
                'label': eval_labels[i:i + total_batch_size]
            }
            
            batch_sharded = shard_batch(batch, num_devices)
            loss, accuracy = eval_step_pmap(eval_model_replicated, batch_sharded)
            
            eval_loss += float(loss[0])
            eval_acc += float(accuracy[0])
            num_eval_batches += 1
        
        eval_loss /= num_eval_batches
        eval_acc /= num_eval_batches
        print(f"  Eval: Loss: {eval_loss:.4f} | Acc: {eval_acc:.4f}")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    
    # Explain what happened
    print("\n" + "="*70)
    print("HOW DATA PARALLELISM WORKS")
    print("="*70)
    print("""
1. MODEL REPLICATION:
   - The model parameters are replicated across all devices
   - Each device has an identical copy of the full model
   
2. DATA SHARDING:
   - Each batch is split across devices
   - If batch_size=128 and 8 devices: each device gets 16 examples
   - Shape: (128, 32, 32, 3) -> (8, 16, 32, 32, 3)
   
3. PARALLEL FORWARD PASS:
   - Each device independently computes forward pass on its data shard
   - No communication needed during forward pass
   
4. PARALLEL GRADIENT COMPUTATION:
   - Each device computes gradients on its local data shard
   - Gradients are computed independently in parallel
   
5. GRADIENT SYNCHRONIZATION (All-Reduce):
   - jax.lax.pmean averages gradients across all devices
   - This is a collective communication operation
   - Ensures all devices have the same gradient
   - Communication cost: O(model_size)
   
6. PARAMETER UPDATE:
   - All devices apply the averaged gradient
   - All devices stay in sync (identical parameters)
   
ADVANTAGES:
   ✓ Simple to implement and understand
   ✓ Perfect scaling for compute (N devices = N× throughput)
   ✓ Works with any model size (as long as it fits on one device)
   ✓ No model architecture changes needed
   
LIMITATIONS:
   ✗ All devices must have the same model (memory constraint)
   ✗ Communication overhead (gradient sync)
   ✗ Cannot train models larger than single device memory
   
WHEN TO USE:
   - Your model fits comfortably on a single device
   - You want to process more data per second
   - You want to increase effective batch size
   - Simple parallelization is sufficient
""")


if __name__ == '__main__':
    train_data_parallel()
