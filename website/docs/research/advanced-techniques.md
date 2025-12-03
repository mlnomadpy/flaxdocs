---
sidebar_position: 1
---

# Advanced Research Techniques

Explore cutting-edge techniques for research with Flax.

## Custom Training Loops

Build flexible training loops for research experiments.

### Flexible Training State

```python
from flax import struct
from flax.training import train_state
import optax

@struct.dataclass
class CustomTrainState(train_state.TrainState):
    """Extended training state with additional fields."""
    batch_stats: dict
    dropout_rng: jax.random.PRNGKey
    metrics: dict
    ema_params: dict  # Exponential moving average parameters
    
    def apply_ema(self, decay=0.999):
        """Apply exponential moving average to parameters."""
        new_ema = jax.tree_map(
            lambda ema, p: decay * ema + (1 - decay) * p,
            self.ema_params,
            self.params
        )
        return self.replace(ema_params=new_ema)
```

### Advanced Training Step

```python
@jax.jit
def advanced_train_step(state, batch, dropout_rng):
    """Training step with advanced features."""
    
    def loss_fn(params):
        # Forward pass with dropout and batch norm
        variables = {'params': params, 'batch_stats': state.batch_stats}
        logits, new_variables = state.apply_fn(
            variables,
            batch['image'],
            training=True,
            rngs={'dropout': dropout_rng},
            mutable=['batch_stats']
        )
        
        # Compute loss with label smoothing
        labels_one_hot = jax.nn.one_hot(batch['label'], num_classes=10)
        labels_smooth = labels_one_hot * 0.9 + 0.1 / 10  # Label smoothing
        
        loss = optax.softmax_cross_entropy(
            logits=logits,
            labels=labels_smooth
        ).mean()
        
        # Add auxiliary losses
        l2_loss = 0.5 * sum(
            jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params)
        )
        total_loss = loss + 1e-4 * l2_loss
        
        return total_loss, (logits, new_variables['batch_stats'])
    
    # Compute gradients
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (logits, new_batch_stats)), grads = grad_fn(state.params)
    
    # Gradient clipping
    grads = optax.clip_by_global_norm(1.0).update(grads, state)
    
    # Update state
    state = state.apply_gradients(
        grads=grads,
        batch_stats=new_batch_stats
    )
    
    # Update EMA parameters
    state = state.apply_ema(decay=0.999)
    
    # Compute metrics
    accuracy = jnp.mean(jnp.argmax(logits, -1) == batch['label'])
    metrics = {
        'loss': loss,
        'accuracy': accuracy,
        'grad_norm': optax.global_norm(grads),
    }
    
    return state, metrics
```

## Contrastive Learning

Implement self-supervised learning methods.

### SimCLR-style Contrastive Loss

```python
import jax.numpy as jnp

def nt_xent_loss(z_i, z_j, temperature=0.5):
    """Normalized Temperature-scaled Cross Entropy Loss (NT-Xent)."""
    batch_size = z_i.shape[0]
    
    # Normalize embeddings
    z_i = z_i / jnp.linalg.norm(z_i, axis=1, keepdims=True)
    z_j = z_j / jnp.linalg.norm(z_j, axis=1, keepdims=True)
    
    # Compute similarity matrix
    representations = jnp.concatenate([z_i, z_j], axis=0)
    similarity_matrix = jnp.matmul(representations, representations.T)
    
    # Create labels
    labels = jnp.arange(batch_size)
    labels = jnp.concatenate([labels + batch_size, labels])
    
    # Mask out self-similarity
    mask = jnp.eye(2 * batch_size)
    similarity_matrix = similarity_matrix - mask * 1e9
    
    # Apply temperature scaling
    similarity_matrix = similarity_matrix / temperature
    
    # Compute cross-entropy loss
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=similarity_matrix,
        labels=labels
    ).mean()
    
    return loss

class ContrastiveModel(nn.Module):
    """Model for contrastive learning."""
    
    @nn.compact
    def __call__(self, x, training: bool = False):
        # Encoder
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        x = nn.Conv(features=128, kernel_size=(3, 3))(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        x = x.reshape((x.shape[0], -1))
        
        # Projection head
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dense(features=128)(x)
        
        return x

@jax.jit
def contrastive_train_step(state, batch, rng):
    """Training step for contrastive learning."""
    
    def loss_fn(params):
        # Apply two different augmentations
        rng1, rng2 = jax.random.split(rng)
        
        aug1 = augment(batch['image'], rng1)
        aug2 = augment(batch['image'], rng2)
        
        # Get embeddings
        z_i = state.apply_fn({'params': params}, aug1, training=True)
        z_j = state.apply_fn({'params': params}, aug2, training=True)
        
        # Compute contrastive loss
        loss = nt_xent_loss(z_i, z_j)
        
        return loss
    
    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    
    return state
```

## Meta-Learning (MAML)

Implement Model-Agnostic Meta-Learning.

```python
def maml_train_step(state, support_batch, query_batch, inner_lr=0.01, inner_steps=5):
    """MAML training step."""
    
    def inner_loop(params, support_batch):
        """Inner loop: adapt to task."""
        for _ in range(inner_steps):
            def loss_fn(p):
                logits = state.apply_fn({'params': p}, support_batch['image'])
                loss = optax.softmax_cross_entropy_with_integer_labels(
                    logits=logits,
                    labels=support_batch['label']
                ).mean()
                return loss
            
            grad_fn = jax.grad(loss_fn)
            grads = grad_fn(params)
            
            # Manual SGD update for inner loop
            params = jax.tree_map(
                lambda p, g: p - inner_lr * g,
                params,
                grads
            )
        
        return params
    
    def meta_loss_fn(params):
        """Compute meta-loss on query set after adaptation."""
        # Adapt parameters on support set
        adapted_params = inner_loop(params, support_batch)
        
        # Evaluate on query set
        logits = state.apply_fn({'params': adapted_params}, query_batch['image'])
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits,
            labels=query_batch['label']
        ).mean()
        
        return loss
    
    # Compute meta-gradients
    grad_fn = jax.grad(meta_loss_fn)
    grads = grad_fn(state.params)
    
    # Meta-update
    state = state.apply_gradients(grads=grads)
    
    return state
```

## Neural Architecture Search

Implement differentiable architecture search.

```python
class DARTSCell(nn.Module):
    """DARTS mixed operation cell."""
    num_ops: int = 8
    
    @nn.compact
    def __call__(self, x, arch_params):
        """Apply mixed operation with architecture parameters."""
        
        # Define candidate operations
        ops = [
            lambda x: x,  # Identity
            lambda x: nn.Conv(features=x.shape[-1], kernel_size=(3, 3), padding='SAME')(x),
            lambda x: nn.Conv(features=x.shape[-1], kernel_size=(5, 5), padding='SAME')(x),
            lambda x: nn.avg_pool(x, window_shape=(3, 3), strides=(1, 1), padding='SAME'),
            lambda x: nn.max_pool(x, window_shape=(3, 3), strides=(1, 1), padding='SAME'),
        ]
        
        # Apply weighted sum of operations
        arch_weights = jax.nn.softmax(arch_params)
        result = sum(w * op(x) for w, op in zip(arch_weights, ops))
        
        return result

def nas_train_step(model_state, arch_state, train_batch, val_batch):
    """Training step for neural architecture search."""
    
    # Update architecture parameters on validation set
    def arch_loss_fn(arch_params):
        logits = model_state.apply_fn(
            {'params': model_state.params, 'arch_params': arch_params},
            val_batch['image']
        )
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits,
            labels=val_batch['label']
        ).mean()
        return loss
    
    arch_grads = jax.grad(arch_loss_fn)(arch_state.params)
    arch_state = arch_state.apply_gradients(grads=arch_grads)
    
    # Update model parameters on training set
    def model_loss_fn(params):
        logits = model_state.apply_fn(
            {'params': params, 'arch_params': arch_state.params},
            train_batch['image']
        )
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits,
            labels=train_batch['label']
        ).mean()
        return loss
    
    model_grads = jax.grad(model_loss_fn)(model_state.params)
    model_state = model_state.apply_gradients(grads=model_grads)
    
    return model_state, arch_state
```

## Adversarial Training

Robust training against adversarial examples.

```python
def fgsm_attack(state, batch, epsilon=0.1):
    """Fast Gradient Sign Method attack."""
    
    def loss_fn(x):
        logits = state.apply_fn({'params': state.params}, x)
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits,
            labels=batch['label']
        ).mean()
        return loss
    
    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(batch['image'])
    
    # Create adversarial example
    perturbation = epsilon * jnp.sign(grads)
    adversarial_images = batch['image'] + perturbation
    adversarial_images = jnp.clip(adversarial_images, 0.0, 1.0)
    
    return adversarial_images

@jax.jit
def adversarial_train_step(state, batch, epsilon=0.1):
    """Training step with adversarial examples."""
    
    # Generate adversarial examples
    adv_images = fgsm_attack(state, batch, epsilon)
    
    def loss_fn(params):
        # Loss on clean examples
        clean_logits = state.apply_fn({'params': params}, batch['image'])
        clean_loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=clean_logits,
            labels=batch['label']
        ).mean()
        
        # Loss on adversarial examples
        adv_logits = state.apply_fn({'params': params}, adv_images)
        adv_loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=adv_logits,
            labels=batch['label']
        ).mean()
        
        # Combined loss
        total_loss = 0.5 * clean_loss + 0.5 * adv_loss
        
        return total_loss
    
    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    
    return state
```

## Knowledge Distillation

Transfer knowledge from a teacher model to a student model.

```python
def distillation_loss(student_logits, teacher_logits, labels, temperature=3.0, alpha=0.5):
    """Combined distillation and classification loss."""
    
    # Hard label loss
    hard_loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=student_logits,
        labels=labels
    ).mean()
    
    # Soft label loss (distillation)
    soft_student = jax.nn.log_softmax(student_logits / temperature)
    soft_teacher = jax.nn.softmax(teacher_logits / temperature)
    
    soft_loss = -jnp.sum(soft_teacher * soft_student, axis=-1).mean()
    soft_loss *= (temperature ** 2)
    
    # Combined loss
    total_loss = alpha * hard_loss + (1 - alpha) * soft_loss
    
    return total_loss

@jax.jit
def distillation_train_step(student_state, teacher_params, batch):
    """Training step for knowledge distillation."""
    
    def loss_fn(student_params):
        # Get teacher predictions (no gradients)
        teacher_logits = teacher_state.apply_fn(
            {'params': teacher_params},
            batch['image']
        )
        
        # Get student predictions
        student_logits = student_state.apply_fn(
            {'params': student_params},
            batch['image']
        )
        
        # Compute distillation loss
        loss = distillation_loss(
            student_logits,
            teacher_logits,
            batch['label']
        )
        
        return loss
    
    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(student_state.params)
    student_state = student_state.apply_gradients(grads=grads)
    
    return student_state
```

## Curriculum Learning

Gradually increase task difficulty during training.

```python
class CurriculumScheduler:
    """Manage curriculum difficulty over training."""
    
    def __init__(self, num_stages=5, steps_per_stage=10000):
        self.num_stages = num_stages
        self.steps_per_stage = steps_per_stage
    
    def get_difficulty(self, step):
        """Get current difficulty level (0 to 1)."""
        stage = min(step // self.steps_per_stage, self.num_stages - 1)
        return (stage + 1) / self.num_stages
    
    def sample_batch(self, dataset, difficulty, batch_size):
        """Sample batch according to difficulty."""
        # Filter examples by difficulty
        max_difficulty = int(difficulty * len(dataset))
        available_data = dataset[:max_difficulty]
        
        # Sample batch
        indices = jax.random.choice(
            jax.random.PRNGKey(0),
            len(available_data),
            shape=(batch_size,),
            replace=False
        )
        
        return available_data[indices]

# Usage
curriculum = CurriculumScheduler(num_stages=5, steps_per_stage=5000)

for step in range(num_steps):
    difficulty = curriculum.get_difficulty(step)
    batch = curriculum.sample_batch(dataset, difficulty, batch_size)
    state = train_step(state, batch)
```

## Experiment Reproducibility

Ensure reproducible research experiments.

```python
import random
import numpy as np

def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    
def create_reproducible_experiment(seed=42):
    """Create reproducible experiment setup."""
    set_seed(seed)
    
    # Create deterministic RNG
    rng = jax.random.PRNGKey(seed)
    
    # Log all hyperparameters
    config = {
        'seed': seed,
        'learning_rate': 1e-3,
        'batch_size': 128,
        'num_epochs': 100,
        'model_config': {},
    }
    
    # Save config
    import json
    with open('experiment_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    return rng, config
```

## Next Steps
