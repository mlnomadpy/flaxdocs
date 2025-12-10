"""
Shared training utilities for Flax NNX examples.

Common training functions, loss functions, metrics, and learning rate schedules.
"""

import jax
import jax.numpy as jnp
from flax import nnx
import optax
from typing import Dict, Any, Callable, Tuple
from functools import partial


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

def compute_mse_loss(predictions: jax.Array, targets: jax.Array) -> jax.Array:
    """Compute mean squared error loss.
    
    Args:
        predictions: Model predictions
        targets: Target values
        
    Returns:
        MSE loss value
    """
    return jnp.mean((predictions - targets) ** 2)


def compute_cross_entropy_loss(logits: jax.Array, labels: jax.Array) -> jax.Array:
    """Compute cross-entropy loss.
    
    Args:
        logits: Model logits of shape (batch, num_classes)
        labels: Integer labels of shape (batch,)
        
    Returns:
        Cross-entropy loss value
    """
    # Use optax for numerical stability
    return optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()


# ============================================================================
# METRICS
# ============================================================================

def compute_accuracy(logits: jax.Array, labels: jax.Array) -> float:
    """Compute classification accuracy.
    
    Args:
        logits: Model logits of shape (batch, num_classes)
        labels: Integer labels of shape (batch,)
        
    Returns:
        Accuracy as a float between 0 and 1
    """
    predictions = jnp.argmax(logits, axis=-1)
    return jnp.mean(predictions == labels)


# ============================================================================
# TRAINING STEP
# ============================================================================

def create_train_step(loss_fn_name: str = 'cross_entropy') -> Callable:
    """Create a training step function.
    
    Args:
        loss_fn_name: Name of loss function ('mse', 'cross_entropy')
        
    Returns:
        Training step function
    """
    if loss_fn_name == 'mse':
        loss_fn = compute_mse_loss
        compute_metrics = None
    elif loss_fn_name == 'cross_entropy':
        loss_fn = compute_cross_entropy_loss
        compute_metrics = compute_accuracy
    else:
        raise ValueError(f"Unknown loss function: {loss_fn_name}")
    
    @nnx.jit
    def train_step(
        model: nnx.Module,
        optimizer: nnx.Optimizer,
        batch: Dict[str, jax.Array]
    ) -> Tuple[float, Dict[str, Any]]:
        """Perform one training step.
        
        Args:
            model: The model to train
            optimizer: The optimizer
            batch: Batch of data with 'x' and 'y' keys
            
        Returns:
            Tuple of (loss, metrics_dict)
        """
        def loss_fn_wrapper(model):
            logits = model(batch['x'], train=True)
            loss = loss_fn(logits, batch['y'])
            return loss, logits
        
        # Compute loss and gradients
        (loss, logits), grads = nnx.value_and_grad(loss_fn_wrapper, has_aux=True)(model)
        
        # Update parameters
        optimizer.update(model, grads)
        
        # Compute metrics
        metrics = {'loss': loss}
        if compute_metrics is not None:
            metrics['accuracy'] = compute_metrics(logits, batch['y'])
        
        return loss, metrics
    
    return train_step


# ============================================================================
# EVALUATION STEP
# ============================================================================

def create_eval_step(loss_fn_name: str = 'cross_entropy') -> Callable:
    """Create an evaluation step function.
    
    Args:
        loss_fn_name: Name of loss function ('mse', 'cross_entropy')
        
    Returns:
        Evaluation step function
    """
    if loss_fn_name == 'mse':
        loss_fn = compute_mse_loss
        compute_metrics = None
    elif loss_fn_name == 'cross_entropy':
        loss_fn = compute_cross_entropy_loss
        compute_metrics = compute_accuracy
    else:
        raise ValueError(f"Unknown loss function: {loss_fn_name}")
    
    @nnx.jit
    def eval_step(
        model: nnx.Module,
        batch: Dict[str, jax.Array]
    ) -> Tuple[float, Dict[str, Any]]:
        """Perform one evaluation step.
        
        Args:
            model: The model to evaluate
            batch: Batch of data with 'x' and 'y' keys
            
        Returns:
            Tuple of (loss, metrics_dict)
        """
        # Forward pass in eval mode
        logits = model(batch['x'], train=False)
        loss = loss_fn(logits, batch['y'])
        
        # Compute metrics
        metrics = {'loss': loss}
        if compute_metrics is not None:
            metrics['accuracy'] = compute_metrics(logits, batch['y'])
        
        return loss, metrics
    
    return eval_step


# ============================================================================
# LEARNING RATE SCHEDULES
# ============================================================================

def create_warmup_cosine_schedule(
    init_value: float,
    peak_value: float,
    warmup_steps: int,
    decay_steps: int,
    end_value: float = 0.0
) -> optax.Schedule:
    """Create a warmup + cosine decay learning rate schedule.
    
    Args:
        init_value: Initial learning rate
        peak_value: Peak learning rate after warmup
        warmup_steps: Number of warmup steps
        decay_steps: Number of decay steps after warmup
        end_value: Final learning rate
        
    Returns:
        Learning rate schedule function
    """
    schedules = [
        optax.linear_schedule(
            init_value=init_value,
            end_value=peak_value,
            transition_steps=warmup_steps
        ),
        optax.cosine_decay_schedule(
            init_value=peak_value,
            decay_steps=decay_steps,
            alpha=end_value / peak_value if peak_value > 0 else 0.0
        )
    ]
    
    return optax.join_schedules(schedules, [warmup_steps])


def create_exponential_decay_schedule(
    init_value: float,
    decay_rate: float,
    decay_steps: int,
    staircase: bool = False
) -> optax.Schedule:
    """Create an exponential decay learning rate schedule.
    
    Args:
        init_value: Initial learning rate
        decay_rate: Decay rate (e.g., 0.96 for 4% decay)
        decay_steps: Number of steps per decay
        staircase: If True, decay at discrete intervals
        
    Returns:
        Learning rate schedule function
    """
    return optax.exponential_decay(
        init_value=init_value,
        transition_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=staircase
    )


# ============================================================================
# GRADIENT UTILITIES
# ============================================================================

def clip_gradients(grads: Any, max_norm: float) -> Any:
    """Clip gradients by global norm.
    
    Args:
        grads: Gradient pytree
        max_norm: Maximum gradient norm
        
    Returns:
        Clipped gradients
    """
    return optax.clip_by_global_norm(max_norm)(grads, None)[0]


# ============================================================================
# TRAINING LOOP UTILITIES
# ============================================================================

def create_optimizer(
    model: nnx.Module,
    learning_rate: float,
    optimizer_name: str = 'adam',
    **kwargs
) -> nnx.Optimizer:
    """Create an optimizer for the model.
    
    Args:
        model: The model to optimize
        learning_rate: Learning rate or schedule
        optimizer_name: Name of optimizer ('adam', 'sgd', 'adamw')
        **kwargs: Additional optimizer arguments
        
    Returns:
        Optimizer instance
    """
    if optimizer_name == 'adam':
        tx = optax.adam(learning_rate, **kwargs)
    elif optimizer_name == 'sgd':
        tx = optax.sgd(learning_rate, **kwargs)
    elif optimizer_name == 'adamw':
        tx = optax.adamw(learning_rate, **kwargs)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    # Use wrt parameter for newer Flax versions
    return nnx.Optimizer(model, tx, wrt=nnx.Param)
