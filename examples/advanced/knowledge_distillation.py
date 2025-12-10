"""
Flax NNX: Knowledge Distillation
=================================
Implementation of knowledge distillation - transferring knowledge from a
large teacher model to a smaller student model.

Run: python advanced/knowledge_distillation.py

Reference:
    Hinton et al. "Distilling the Knowledge in a Neural Network"
    NIPS 2014 Workshop. https://arxiv.org/abs/1503.02531
"""

import jax
import jax.numpy as jnp
from flax import nnx
import optax
import tensorflow_datasets as tfds
import tensorflow as tf
from typing import Dict
import time


import sys
from pathlib import Path

# Add parent directory to path for shared imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Disable GPU for TensorFlow
tf.config.set_visible_devices([], 'GPU')


# ============================================================================
# 1. TEACHER MODEL (LARGE NETWORK)
# ============================================================================

class TeacherCNN(nnx.Module):
    """
    Large teacher model with high capacity.
    This model is pre-trained and will be used to guide the student.
    """
    
    def __init__(self, num_classes: int = 10, rngs: nnx.Rngs = None):
        # Larger architecture
        self.conv1 = nnx.Conv(1, 64, kernel_size=(3, 3), rngs=rngs)
        self.bn1 = nnx.BatchNorm(64, rngs=rngs)
        
        self.conv2 = nnx.Conv(64, 128, kernel_size=(3, 3), rngs=rngs)
        self.bn2 = nnx.BatchNorm(128, rngs=rngs)
        
        self.conv3 = nnx.Conv(128, 128, kernel_size=(3, 3), rngs=rngs)
        self.bn3 = nnx.BatchNorm(128, rngs=rngs)
        
        self.fc1 = nnx.Linear(128 * 3 * 3, 512, rngs=rngs)
        self.fc2 = nnx.Linear(512, 256, rngs=rngs)
        self.fc3 = nnx.Linear(256, num_classes, rngs=rngs)
        
        self.dropout = nnx.Dropout(rate=0.5, rngs=rngs)
    
    def __call__(self, x, train: bool = False):
        x = self.conv1(x)
        x = self.bn1(x, use_running_average=not train)
        x = nnx.relu(x)
        x = nnx.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        x = self.conv2(x)
        x = self.bn2(x, use_running_average=not train)
        x = nnx.relu(x)
        x = nnx.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        x = self.conv3(x)
        x = self.bn3(x, use_running_average=not train)
        x = nnx.relu(x)
        x = nnx.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        x = x.reshape((x.shape[0], -1))
        
        x = self.fc1(x)
        x = nnx.relu(x)
        x = self.dropout(x, deterministic=not train)
        
        x = self.fc2(x)
        x = nnx.relu(x)
        x = self.dropout(x, deterministic=not train)
        
        x = self.fc3(x)
        
        return x


# ============================================================================
# 2. STUDENT MODEL (SMALL NETWORK)
# ============================================================================

class StudentCNN(nnx.Module):
    """
    Small student model with limited capacity.
    This model will learn from the teacher's soft predictions.
    """
    
    def __init__(self, num_classes: int = 10, rngs: nnx.Rngs = None):
        # Much smaller architecture
        self.conv1 = nnx.Conv(1, 16, kernel_size=(3, 3), rngs=rngs)
        self.bn1 = nnx.BatchNorm(16, rngs=rngs)
        
        self.conv2 = nnx.Conv(16, 32, kernel_size=(3, 3), rngs=rngs)
        self.bn2 = nnx.BatchNorm(32, rngs=rngs)
        
        self.fc1 = nnx.Linear(32 * 5 * 5, 64, rngs=rngs)
        self.fc2 = nnx.Linear(64, num_classes, rngs=rngs)
    
    def __call__(self, x, train: bool = False):
        x = self.conv1(x)
        x = self.bn1(x, use_running_average=not train)
        x = nnx.relu(x)
        x = nnx.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        x = self.conv2(x)
        x = self.bn2(x, use_running_average=not train)
        x = nnx.relu(x)
        x = nnx.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        x = x.reshape((x.shape[0], -1))
        
        x = self.fc1(x)
        x = nnx.relu(x)
        x = self.fc2(x)
        
        return x


# ============================================================================
# 3. DISTILLATION LOSS
# ============================================================================

def distillation_loss(
    student_logits: jnp.ndarray,
    teacher_logits: jnp.ndarray,
    labels: jnp.ndarray,
    temperature: float = 3.0,
    alpha: float = 0.5
):
    """
    Combined distillation and classification loss.
    
    The distillation loss consists of two components:
    
    1. Hard Label Loss: Standard cross-entropy with true labels
       L_hard = CE(student_logits, labels)
    
    2. Soft Label Loss: KL divergence between teacher and student distributions
       L_soft = KL(softmax(teacher_logits/T) || softmax(student_logits/T)) * T²
    
    Total Loss:
       L = α * L_hard + (1 - α) * L_soft
    
    Args:
        student_logits: Student model predictions [batch_size, num_classes]
        teacher_logits: Teacher model predictions [batch_size, num_classes]
        labels: True labels [batch_size]
        temperature: Temperature for softening distributions (τ)
                    Higher values produce softer distributions
        alpha: Weight for hard label loss (λ)
               α=1 means standard training, α=0 means pure distillation
    
    Returns:
        Combined loss value
    
    Key Insight:
        The temperature parameter softens the probability distributions,
        revealing the teacher's learned structure. Small probabilities
        become more informative, capturing similarity relationships between
        classes (e.g., "3" is more similar to "8" than to "1").
    """
    num_classes = student_logits.shape[-1]
    
    # 1. Hard label loss (standard cross-entropy)
    one_hot_labels = jax.nn.one_hot(labels, num_classes=num_classes)
    hard_loss = -jnp.mean(
        jnp.sum(one_hot_labels * jax.nn.log_softmax(student_logits), axis=-1)
    )
    
    # 2. Soft label loss (distillation from teacher)
    # Temperature-scaled softmax
    soft_student = jax.nn.log_softmax(student_logits / temperature)
    soft_teacher = jax.nn.softmax(teacher_logits / temperature)
    
    # KL divergence: KL(P||Q) = Σ P(x) * log(P(x) / Q(x))
    soft_loss = -jnp.sum(soft_teacher * soft_student, axis=-1).mean()
    
    # Scale by T² (see paper for derivation)
    soft_loss = soft_loss * (temperature ** 2)
    
    # 3. Combined loss
    total_loss = alpha * hard_loss + (1 - alpha) * soft_loss
    
    return total_loss, hard_loss, soft_loss


# ============================================================================
# 4. DATA LOADING
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
# 5. TRAINING STEPS
# ============================================================================

def cross_entropy_loss(logits, labels):
    """Standard cross-entropy loss."""
    one_hot_labels = jax.nn.one_hot(labels, num_classes=10)
    return -jnp.mean(jnp.sum(one_hot_labels * jax.nn.log_softmax(logits), axis=-1))


@nnx.jit
def train_teacher_step(teacher: TeacherCNN, optimizer: nnx.Optimizer, 
                       batch: Dict[str, jnp.ndarray]):
    """Training step for teacher model (standard supervised learning)."""
    
    def loss_fn(teacher: TeacherCNN):
        logits = teacher(batch['image'], train=True)
        loss = cross_entropy_loss(logits, batch['label'])
        return loss, logits
    
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(teacher)
    optimizer.update(grads)
    
    accuracy = jnp.mean(jnp.argmax(logits, -1) == batch['label'])
    return {'loss': loss, 'accuracy': accuracy}


@nnx.jit
def train_student_step(student: StudentCNN, teacher: TeacherCNN, 
                       optimizer: nnx.Optimizer, batch: Dict[str, jnp.ndarray],
                       temperature: float = 3.0, alpha: float = 0.5):
    """Training step for student model using knowledge distillation."""
    
    def loss_fn(student: StudentCNN):
        # Get student predictions
        student_logits = student(batch['image'], train=True)
        
        # Get teacher predictions (no gradients)
        teacher_logits = teacher(batch['image'], train=False)
        teacher_logits = jax.lax.stop_gradient(teacher_logits)
        
        # Compute distillation loss
        total_loss, hard_loss, soft_loss = distillation_loss(
            student_logits,
            teacher_logits,
            batch['label'],
            temperature=temperature,
            alpha=alpha
        )
        
        return total_loss, (student_logits, hard_loss, soft_loss)
    
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, (logits, hard_loss, soft_loss)), grads = grad_fn(student)
    optimizer.update(grads)
    
    accuracy = jnp.mean(jnp.argmax(logits, -1) == batch['label'])
    return {
        'loss': loss,
        'hard_loss': hard_loss,
        'soft_loss': soft_loss,
        'accuracy': accuracy
    }


@nnx.jit
def eval_step(model, batch: Dict[str, jnp.ndarray]):
    """Evaluation step."""
    logits = model(batch['image'], train=False)
    loss = cross_entropy_loss(logits, batch['label'])
    accuracy = jnp.mean(jnp.argmax(logits, -1) == batch['label'])
    return {'loss': loss, 'accuracy': accuracy}


# ============================================================================
# 6. TRAINING FUNCTIONS
# ============================================================================

def train_teacher(num_epochs: int = 10, batch_size: int = 128, 
                  learning_rate: float = 1e-3):
    """Train teacher model."""
    print("\n" + "="*70)
    print("TRAINING TEACHER MODEL")
    print("="*70)
    
    # Load data
    train_ds, test_ds = load_data(batch_size)
    
    # Initialize teacher and optimizer
    print("\nInitializing teacher model...")
    rng = jax.random.PRNGKey(0)
    teacher = TeacherCNN(num_classes=10, rngs=nnx.Rngs(rng))
    optimizer = nnx.Optimizer(teacher, optax.adam(learning_rate))
    
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(nnx.state(teacher)))
    print(f"✓ Teacher model initialized with {n_params:,} parameters")
    
    # Training loop
    print(f"\nTraining for {num_epochs} epochs...")
    print("-" * 70)
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        train_metrics = []
        
        for batch in train_ds.as_numpy_iterator():
            batch = {k: jnp.array(v) for k, v in batch.items()}
            metrics = train_teacher_step(teacher, optimizer, batch)
            train_metrics.append(metrics)
        
        # Compute average metrics
        avg_train_loss = jnp.mean(jnp.array([m['loss'] for m in train_metrics]))
        avg_train_acc = jnp.mean(jnp.array([m['accuracy'] for m in train_metrics]))
        
        # Evaluate on test set
        test_metrics = []
        for batch in test_ds.as_numpy_iterator():
            batch = {k: jnp.array(v) for k, v in batch.items()}
            metrics = eval_step(teacher, batch)
            test_metrics.append(metrics)
        
        avg_test_loss = jnp.mean(jnp.array([m['loss'] for m in test_metrics]))
        avg_test_acc = jnp.mean(jnp.array([m['accuracy'] for m in test_metrics]))
        
        epoch_time = time.time() - epoch_start
        
        print(f"Epoch {epoch+1:2d}/{num_epochs} | "
              f"Train Loss: {avg_train_loss:.4f} Acc: {avg_train_acc:.4f} | "
              f"Test Loss: {avg_test_loss:.4f} Acc: {avg_test_acc:.4f} | "
              f"Time: {epoch_time:.2f}s")
    
    print("\n✓ Teacher training completed!")
    print(f"  Final test accuracy: {avg_test_acc:.4f}")
    print("="*70)
    
    return teacher


def train_student_with_distillation(
    teacher: TeacherCNN,
    num_epochs: int = 10,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    temperature: float = 3.0,
    alpha: float = 0.5
):
    """Train student model using knowledge distillation."""
    print("\n" + "="*70)
    print("TRAINING STUDENT MODEL WITH KNOWLEDGE DISTILLATION")
    print("="*70)
    print(f"\nDistillation parameters:")
    print(f"  Temperature (τ): {temperature}")
    print(f"  Alpha (λ): {alpha}")
    print(f"  Hard loss weight: {alpha}")
    print(f"  Soft loss weight: {1-alpha}")
    
    # Load data
    train_ds, test_ds = load_data(batch_size)
    
    # Initialize student and optimizer
    print("\nInitializing student model...")
    rng = jax.random.PRNGKey(42)
    student = StudentCNN(num_classes=10, rngs=nnx.Rngs(rng))
    optimizer = nnx.Optimizer(student, optax.adam(learning_rate))
    
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(nnx.state(student)))
    teacher_params = sum(x.size for x in jax.tree_util.tree_leaves(nnx.state(teacher)))
    print(f"✓ Student model initialized with {n_params:,} parameters")
    print(f"  Compression ratio: {teacher_params / n_params:.1f}x smaller")
    
    # Training loop
    print(f"\nTraining for {num_epochs} epochs...")
    print("-" * 70)
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        train_metrics = []
        
        for batch in train_ds.as_numpy_iterator():
            batch = {k: jnp.array(v) for k, v in batch.items()}
            metrics = train_student_step(
                student, teacher, optimizer, batch,
                temperature=temperature, alpha=alpha
            )
            train_metrics.append(metrics)
        
        # Compute average metrics
        avg_train_loss = jnp.mean(jnp.array([m['loss'] for m in train_metrics]))
        avg_hard_loss = jnp.mean(jnp.array([m['hard_loss'] for m in train_metrics]))
        avg_soft_loss = jnp.mean(jnp.array([m['soft_loss'] for m in train_metrics]))
        avg_train_acc = jnp.mean(jnp.array([m['accuracy'] for m in train_metrics]))
        
        # Evaluate on test set
        test_metrics = []
        for batch in test_ds.as_numpy_iterator():
            batch = {k: jnp.array(v) for k, v in batch.items()}
            metrics = eval_step(student, batch)
            test_metrics.append(metrics)
        
        avg_test_acc = jnp.mean(jnp.array([m['accuracy'] for m in test_metrics]))
        
        epoch_time = time.time() - epoch_start
        
        print(f"Epoch {epoch+1:2d}/{num_epochs} | "
              f"Total: {avg_train_loss:.4f} "
              f"Hard: {avg_hard_loss:.4f} "
              f"Soft: {avg_soft_loss:.4f} | "
              f"Train Acc: {avg_train_acc:.4f} "
              f"Test Acc: {avg_test_acc:.4f} | "
              f"Time: {epoch_time:.2f}s")
    
    print("\n✓ Student training completed!")
    print(f"  Final test accuracy: {avg_test_acc:.4f}")
    print("="*70)
    
    return student


def train_student_baseline(num_epochs: int = 10, batch_size: int = 128,
                           learning_rate: float = 1e-3):
    """Train student model without distillation (baseline)."""
    print("\n" + "="*70)
    print("TRAINING STUDENT MODEL WITHOUT DISTILLATION (BASELINE)")
    print("="*70)
    
    # Load data
    train_ds, test_ds = load_data(batch_size)
    
    # Initialize student and optimizer
    print("\nInitializing student model...")
    rng = jax.random.PRNGKey(42)
    student = StudentCNN(num_classes=10, rngs=nnx.Rngs(rng))
    optimizer = nnx.Optimizer(student, optax.adam(learning_rate))
    
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(nnx.state(student)))
    print(f"✓ Student model initialized with {n_params:,} parameters")
    
    # Training loop
    print(f"\nTraining for {num_epochs} epochs...")
    print("-" * 70)
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        train_metrics = []
        
        for batch in train_ds.as_numpy_iterator():
            batch = {k: jnp.array(v) for k, v in batch.items()}
            metrics = train_teacher_step(student, optimizer, batch)
            train_metrics.append(metrics)
        
        avg_train_acc = jnp.mean(jnp.array([m['accuracy'] for m in train_metrics]))
        
        # Evaluate on test set
        test_metrics = []
        for batch in test_ds.as_numpy_iterator():
            batch = {k: jnp.array(v) for k, v in batch.items()}
            metrics = eval_step(student, batch)
            test_metrics.append(metrics)
        
        avg_test_acc = jnp.mean(jnp.array([m['accuracy'] for m in test_metrics]))
        epoch_time = time.time() - epoch_start
        
        print(f"Epoch {epoch+1:2d}/{num_epochs} | "
              f"Train Acc: {avg_train_acc:.4f} "
              f"Test Acc: {avg_test_acc:.4f} | "
              f"Time: {epoch_time:.2f}s")
    
    print("\n✓ Baseline training completed!")
    print(f"  Final test accuracy: {avg_test_acc:.4f}")
    print("="*70)
    
    return student


# ============================================================================
# 7. MAIN
# ============================================================================

def main():
    """Main function demonstrating knowledge distillation."""
    
    print("\n" + "="*70)
    print("KNOWLEDGE DISTILLATION DEMONSTRATION")
    print("="*70)
    print("\nThis example demonstrates:")
    print("1. Training a large teacher model")
    print("2. Training a small student WITH distillation")
    print("3. Training a small student WITHOUT distillation (baseline)")
    print("4. Comparing the results")
    print("="*70)
    
    # Train teacher
    teacher = train_teacher(num_epochs=5, learning_rate=1e-3)
    
    # Train student with distillation
    student_distilled = train_student_with_distillation(
        teacher,
        num_epochs=10,
        learning_rate=1e-3,
        temperature=3.0,
        alpha=0.5
    )
    
    # Train student baseline
    student_baseline = train_student_baseline(num_epochs=10, learning_rate=1e-3)
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("\nKnowledge distillation allows a small student model to achieve")
    print("performance closer to a large teacher model by learning from")
    print("the teacher's soft predictions, not just hard labels.")
    print("\nKey benefits:")
    print("  • Smaller model size (faster inference)")
    print("  • Better generalization")
    print("  • Learns class relationships from teacher")
    print("="*70)


if __name__ == "__main__":
    main()
