"""
Flax NNX: Contrastive Learning with SimCLR
==========================================
Implementation of SimCLR (Simple Framework for Contrastive Learning of Visual Representations).
This example demonstrates self-supervised learning using the NT-Xent loss.

Run: python advanced/simclr_contrastive.py

Reference:
    Chen et al. "A Simple Framework for Contrastive Learning of Visual Representations"
    ICML 2020. https://arxiv.org/abs/2002.05709
"""

import jax
import jax.numpy as jnp
from flax import nnx
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
from typing import Dict, Tuple
import time
from functools import partial


import sys
from pathlib import Path

# Add parent directory to path for shared imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Disable GPU for TensorFlow
tf.config.set_visible_devices([], 'GPU')


# ============================================================================
# 1. DATA AUGMENTATION
# ============================================================================

def random_crop_flip(image, rng, crop_size=28):
    """Apply random crop and horizontal flip."""
    # Random crop
    key1, key2 = jax.random.split(rng)
    h, w = image.shape[1:3]
    top = jax.random.randint(key1, (), 0, h - crop_size + 1)
    left = jax.random.randint(key2, (), 0, w - crop_size + 1)
    image = jax.lax.dynamic_slice(image, (0, top, left, 0), (image.shape[0], crop_size, crop_size, 1))
    
    # Random horizontal flip
    key3 = jax.random.split(key2)[0]
    flip = jax.random.bernoulli(key3, 0.5)
    image = jax.lax.cond(flip, lambda x: jnp.flip(x, axis=2), lambda x: x, image)
    
    return image


def gaussian_blur(image, rng, kernel_size=3, sigma=0.5):
    """Apply Gaussian blur (simplified version)."""
    # Simple averaging blur for demonstration
    kernel = jnp.ones((kernel_size, kernel_size, 1, 1)) / (kernel_size ** 2)
    blurred = jax.lax.conv_general_dilated(
        image.transpose(0, 3, 1, 2),
        kernel.transpose(3, 2, 0, 1),
        window_strides=(1, 1),
        padding='SAME',
        dimension_numbers=('NCHW', 'OIHW', 'NCHW')
    )
    return blurred.transpose(0, 2, 3, 1)


def color_jitter(image, rng, brightness=0.4, contrast=0.4):
    """Apply random brightness and contrast adjustments."""
    key1, key2 = jax.random.split(rng)
    
    # Random brightness
    brightness_factor = 1.0 + jax.random.uniform(key1, (), minval=-brightness, maxval=brightness)
    image = image * brightness_factor
    
    # Random contrast
    contrast_factor = 1.0 + jax.random.uniform(key2, (), minval=-contrast, maxval=contrast)
    mean = jnp.mean(image, axis=(1, 2), keepdims=True)
    image = (image - mean) * contrast_factor + mean
    
    return jnp.clip(image, 0.0, 1.0)


def augment_image(image, rng):
    """Apply full augmentation pipeline."""
    key1, key2, key3 = jax.random.split(rng, 3)
    
    # Apply augmentations
    image = random_crop_flip(image, key1, crop_size=24)
    image = color_jitter(image, key2)
    image = gaussian_blur(image, key3)
    
    return image


# ============================================================================
# 2. ENCODER MODEL
# ============================================================================

class ResNetBlock(nnx.Module):
    """ResNet block for the encoder."""
    
    def __init__(self, features: int, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(features, features, kernel_size=(3, 3), padding='SAME', rngs=rngs)
        self.bn1 = nnx.BatchNorm(features, rngs=rngs)
        self.conv2 = nnx.Conv(features, features, kernel_size=(3, 3), padding='SAME', rngs=rngs)
        self.bn2 = nnx.BatchNorm(features, rngs=rngs)
    
    def __call__(self, x, train: bool = False):
        residual = x
        
        x = self.conv1(x)
        x = self.bn1(x, use_running_average=not train)
        x = nnx.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x, use_running_average=not train)
        
        return nnx.relu(x + residual)


class ContrastiveEncoder(nnx.Module):
    """Encoder network for contrastive learning."""
    
    def __init__(self, hidden_dim: int = 128, rngs: nnx.Rngs = None):
        # Convolutional layers
        self.conv1 = nnx.Conv(1, 32, kernel_size=(3, 3), rngs=rngs)
        self.bn1 = nnx.BatchNorm(32, rngs=rngs)
        
        self.conv2 = nnx.Conv(32, 64, kernel_size=(3, 3), rngs=rngs)
        self.bn2 = nnx.BatchNorm(64, rngs=rngs)
        
        # ResNet blocks
        self.resblock1 = ResNetBlock(64, rngs=rngs)
        self.resblock2 = ResNetBlock(64, rngs=rngs)
        
        # Projection head (maps to lower-dimensional space)
        self.fc1 = nnx.Linear(64 * 5 * 5, 256, rngs=rngs)
        self.fc2 = nnx.Linear(256, hidden_dim, rngs=rngs)
    
    def __call__(self, x, train: bool = False):
        # Encoder
        x = self.conv1(x)
        x = self.bn1(x, use_running_average=not train)
        x = nnx.relu(x)
        x = nnx.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        x = self.conv2(x)
        x = self.bn2(x, use_running_average=not train)
        x = nnx.relu(x)
        x = nnx.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        # ResNet blocks
        x = self.resblock1(x, train=train)
        x = self.resblock2(x, train=train)
        
        # Flatten
        x = x.reshape((x.shape[0], -1))
        
        # Projection head
        x = self.fc1(x)
        x = nnx.relu(x)
        x = self.fc2(x)
        
        return x


# ============================================================================
# 3. NT-XENT LOSS (NORMALIZED TEMPERATURE-SCALED CROSS ENTROPY)
# ============================================================================

def nt_xent_loss(z_i: jnp.ndarray, z_j: jnp.ndarray, temperature: float = 0.5):
    """
    Compute NT-Xent loss for contrastive learning.
    
    Args:
        z_i: Embeddings from first augmentation [batch_size, embedding_dim]
        z_j: Embeddings from second augmentation [batch_size, embedding_dim]
        temperature: Temperature parameter for scaling
    
    Returns:
        loss: Scalar loss value
    
    The NT-Xent loss maximizes agreement between differently augmented views
    of the same image while pushing apart different images.
    
    Formula:
        loss = -log(exp(sim(z_i, z_j)/τ) / Σ exp(sim(z_i, z_k)/τ))
    
    where sim(u, v) = u·v / (||u|| ||v||) is cosine similarity.
    """
    batch_size = z_i.shape[0]
    
    # Normalize embeddings (L2 normalization)
    z_i = z_i / jnp.linalg.norm(z_i, axis=1, keepdims=True)
    z_j = z_j / jnp.linalg.norm(z_j, axis=1, keepdims=True)
    
    # Concatenate embeddings: [2*batch_size, embedding_dim]
    representations = jnp.concatenate([z_i, z_j], axis=0)
    
    # Compute similarity matrix: [2*batch_size, 2*batch_size]
    similarity_matrix = jnp.matmul(representations, representations.T)
    
    # Create positive pair labels
    # For each sample i in the first half, its positive is i+batch_size
    # For each sample i in the second half, its positive is i-batch_size
    labels = jnp.arange(batch_size)
    labels = jnp.concatenate([labels + batch_size, labels])
    
    # Mask out self-similarity (diagonal)
    mask = jnp.eye(2 * batch_size, dtype=bool)
    similarity_matrix = jnp.where(mask, -1e9, similarity_matrix)
    
    # Apply temperature scaling
    similarity_matrix = similarity_matrix / temperature
    
    # Compute cross-entropy loss
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=similarity_matrix,
        labels=labels
    )
    
    return jnp.mean(loss)


# ============================================================================
# 4. DATA LOADING
# ============================================================================

def load_data(batch_size: int = 128):
    """Load MNIST dataset for contrastive learning."""
    print("Loading MNIST dataset...")
    
    def preprocess(example):
        # Only normalize, augmentation happens in training loop
        image = tf.cast(example['image'], tf.float32) / 255.0
        return {'image': image}
    
    # Training data (we don't need labels for contrastive learning)
    train_ds = tfds.load('mnist', split='train', shuffle_files=True)
    train_ds = train_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.cache()
    train_ds = train_ds.shuffle(10000)
    train_ds = train_ds.batch(batch_size)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    
    print(f"✓ Data loaded successfully")
    return train_ds


# ============================================================================
# 5. TRAINING STEP
# ============================================================================

@nnx.jit
def train_step(model: ContrastiveEncoder, optimizer: nnx.Optimizer, 
               batch: Dict[str, jnp.ndarray], rng: jax.random.PRNGKey,
               temperature: float = 0.5):
    """Single contrastive training step."""
    
    # Split RNG for two augmentations
    rng1, rng2 = jax.random.split(rng)
    
    def loss_fn(model: ContrastiveEncoder):
        # Apply two different augmentations
        aug1 = jax.vmap(augment_image, in_axes=(0, None))(batch['image'], rng1)
        aug2 = jax.vmap(augment_image, in_axes=(0, None))(batch['image'], rng2)
        
        # Get embeddings
        z_i = model(aug1, train=True)
        z_j = model(aug2, train=True)
        
        # Compute contrastive loss
        loss = nt_xent_loss(z_i, z_j, temperature=temperature)
        
        return loss
    
    # Compute gradients
    grad_fn = nnx.value_and_grad(loss_fn)
    loss, grads = grad_fn(model)
    
    # Update parameters
    optimizer.update(grads)
    
    return {'loss': loss}


# ============================================================================
# 6. TRAINING LOOP
# ============================================================================

def train_contrastive(
    num_epochs: int = 50,
    batch_size: int = 256,
    learning_rate: float = 3e-4,
    temperature: float = 0.5,
    hidden_dim: int = 128,
):
    """Train contrastive model."""
    print("\n" + "="*70)
    print("CONTRASTIVE LEARNING WITH SIMCLR")
    print("="*70)
    
    # Load data
    train_ds = load_data(batch_size)
    
    # Initialize model and optimizer
    print("\nInitializing model...")
    rng = jax.random.PRNGKey(0)
    model_rng, train_rng = jax.random.split(rng)
    
    model = ContrastiveEncoder(hidden_dim=hidden_dim, rngs=nnx.Rngs(model_rng))
    optimizer = nnx.Optimizer(model, optax.adam(learning_rate))
    
    print(f"✓ Model initialized with {sum(x.size for x in jax.tree_util.tree_leaves(nnx.state(model)))} parameters")
    
    # Training loop
    print(f"\nTraining for {num_epochs} epochs...")
    print(f"Temperature: {temperature}")
    print(f"Hidden dimension: {hidden_dim}")
    print("-" * 70)
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        epoch_loss = []
        
        for batch in train_ds.as_numpy_iterator():
            # Convert to JAX arrays
            batch = {k: jnp.array(v) for k, v in batch.items()}
            
            # Training step
            train_rng, step_rng = jax.random.split(train_rng)
            metrics = train_step(model, optimizer, batch, step_rng, temperature)
            epoch_loss.append(metrics['loss'])
        
        # Compute epoch metrics
        avg_loss = jnp.mean(jnp.array(epoch_loss))
        epoch_time = time.time() - epoch_start
        
        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Time: {epoch_time:.2f}s")
    
    print("\n" + "="*70)
    print("✓ Training completed!")
    print("="*70)
    
    return model


# ============================================================================
# 7. LINEAR EVALUATION (TEST LEARNED REPRESENTATIONS)
# ============================================================================

class LinearClassifier(nnx.Module):
    """Linear classifier on top of frozen encoder."""
    
    def __init__(self, encoder: ContrastiveEncoder, num_classes: int = 10, 
                 rngs: nnx.Rngs = None):
        self.encoder = encoder
        self.classifier = nnx.Linear(128, num_classes, rngs=rngs)
    
    def __call__(self, x, train: bool = False):
        # Get frozen encoder features
        features = self.encoder(x, train=False)
        # Stop gradients to encoder
        features = jax.lax.stop_gradient(features)
        # Apply classifier
        logits = self.classifier(features)
        return logits


def evaluate_representations(pretrained_model: ContrastiveEncoder, 
                            num_epochs: int = 10):
    """Evaluate quality of learned representations with linear probe."""
    print("\n" + "="*70)
    print("LINEAR EVALUATION OF LEARNED REPRESENTATIONS")
    print("="*70)
    
    # Load labeled data
    def preprocess(example):
        image = tf.cast(example['image'], tf.float32) / 255.0
        label = example['label']
        return {'image': image, 'label': label}
    
    train_ds = tfds.load('mnist', split='train[:5000]', shuffle_files=True)
    train_ds = train_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.batch(128).prefetch(tf.data.AUTOTUNE)
    
    test_ds = tfds.load('mnist', split='test')
    test_ds = test_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.batch(128).cache().prefetch(tf.data.AUTOTUNE)
    
    # Create linear classifier
    rng = jax.random.PRNGKey(42)
    classifier = LinearClassifier(pretrained_model, num_classes=10, rngs=nnx.Rngs(rng))
    optimizer = nnx.Optimizer(classifier, optax.adam(1e-3))
    
    # Training step for classifier
    @nnx.jit
    def train_step_classifier(classifier, optimizer, batch):
        def loss_fn(classifier):
            logits = classifier(batch['image'], train=True)
            one_hot_labels = jax.nn.one_hot(batch['label'], num_classes=10)
            loss = -jnp.mean(jnp.sum(one_hot_labels * jax.nn.log_softmax(logits), axis=-1))
            return loss, logits
        
        grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
        (loss, logits), grads = grad_fn(classifier)
        optimizer.update(grads)
        
        accuracy = jnp.mean(jnp.argmax(logits, -1) == batch['label'])
        return {'loss': loss, 'accuracy': accuracy}
    
    # Train linear classifier
    print("\nTraining linear classifier on frozen features...")
    for epoch in range(num_epochs):
        for batch in train_ds.as_numpy_iterator():
            batch = {k: jnp.array(v) for k, v in batch.items()}
            metrics = train_step_classifier(classifier, optimizer, batch)
        
        if (epoch + 1) % 2 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} | "
                  f"Loss: {metrics['loss']:.4f} | "
                  f"Accuracy: {metrics['accuracy']:.4f}")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_accuracies = []
    for batch in test_ds.as_numpy_iterator():
        batch = {k: jnp.array(v) for k, v in batch.items()}
        logits = classifier(batch['image'], train=False)
        accuracy = jnp.mean(jnp.argmax(logits, -1) == batch['label'])
        test_accuracies.append(accuracy)
    
    final_accuracy = jnp.mean(jnp.array(test_accuracies))
    print(f"\n✓ Test Accuracy: {final_accuracy:.4f}")
    print("="*70)


# ============================================================================
# 8. MAIN
# ============================================================================

def main():
    """Main function."""
    # Train contrastive model
    model = train_contrastive(
        num_epochs=30,
        batch_size=256,
        learning_rate=3e-4,
        temperature=0.5,
        hidden_dim=128,
    )
    
    # Evaluate learned representations
    evaluate_representations(model, num_epochs=10)


if __name__ == "__main__":
    main()
