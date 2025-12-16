---
sidebar_position: 2
---

# Custom Training Loops

In research, standard training loops often aren't enough. You might need to track complex metrics, manage multiple optimizers, or implement non-standard update rules. Flax's design philosophy is to expose the training state explicitly, giving you full control.

## Why Custom Loops?

Frameworks often abstract away the training loop (`model.fit()`), which is great for standard tasks but restricting for research. Custom loops allow you to:

- **Modify gradients** before application (e.g., gradient clipping, noise injection)
- **Update auxiliary state** like Batch Norm statistics or EMA inputs
- **Implement complex logic** like GAN training (discriminator vs generator steps)
- **Debug easily** by inspecting every intermediate tensor

## Flexible Training State

The `TrainState` pattern is central to Flax. It holds everything that changes during training: parameters, optimizer state, and mutable variables. For research, we often need to extend this.

### Extending TrainState

Here we create a `CustomTrainState` that includes:
- **`batch_stats`**: For Batch Normalization.
- **`dropout_rng`**: Explicit random key for dropout, ensuring reproducibility.
- **`metrics`**: To track arbitrary values during the step.
- **`ema_params`**: Exponential Moving Average of parameters for stable evaluation.

```python
from flax import struct
from flax.training import train_state
import optax
import jax

@struct.dataclass
class CustomTrainState(train_state.TrainState):
    """Extended training state for research experiments."""
    
    # Batch statistics (e.g., for BatchNorm)
    batch_stats: dict
    
    # Random key for dropout (updated every step if needed)
    dropout_rng: jax.random.PRNGKey
    
    # Store arbitrary metrics for the current step
    metrics: dict
    
    # Exponential Moving Average of parameters
    ema_params: dict
    
    def apply_ema(self, decay=0.999):
        """
        Apply exponential moving average to parameters.
        
        New EMA = decay * Old EMA + (1 - decay) * Current Params
        """
        new_ema = jax.tree_map(
            lambda ema, p: decay * ema + (1 - decay) * p,
            self.ema_params,
            self.params
        )
        return self.replace(ema_params=new_ema)
```

## Advanced Training Step

A research-grade training step often involves more than just `loss.backward()`. The example below demonstrates:

1. **State Management**: Handling mutable variables (`batch_stats`).
2. **Auxiliary Losses**: Adding L2 regularization manually.
3. **Advanced Optimization**: Gradient clipping and global norm tracking.
4. **EMA Updates**: Maintaining a separate set of smoothed parameters.

```python
import jax.numpy as jnp

@jax.jit
def advanced_train_step(state, batch, dropout_rng):
    """
    Execute a single training step with advanced features.
    
    Args:
        state: Current CustomTrainState.
        batch: Dictionary containing 'image' and 'label'.
        dropout_rng: PRNGKey for dropout.
        
    Returns:
        Updated state and metrics dictionary.
    """
    
    def loss_fn(params):
        # 1. Forward Pass
        # We must pass mutable=['batch_stats'] to capture updates
        variables = {'params': params, 'batch_stats': state.batch_stats}
        logits, new_variables = state.apply_fn(
            variables,
            batch['image'],
            training=True,
            rngs={'dropout': dropout_rng},
            mutable=['batch_stats']
        )
        
        # 2. Label Smoothing
        # Instead of hard 0/1 labels, we smooth them to prevent overconfidence
        labels_one_hot = jax.nn.one_hot(batch['label'], num_classes=10)
        labels_smooth = labels_one_hot * 0.9 + 0.1 / 10
        
        loss = optax.softmax_cross_entropy(
            logits=logits,
            labels=labels_smooth
        ).mean()
        
        # 3. Auxiliary Losses (L2 Regularization)
        # Manually compute L2 norm of all parameters
        l2_loss = 0.5 * sum(
            jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params)
        )
        total_loss = loss + 1e-4 * l2_loss
        
        return total_loss, (logits, new_variables['batch_stats'])
    
    # 4. Compute Gradients
    # has_aux=True tells JAX that loss_fn returns extra data (logits, new_stats)
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (logits, new_batch_stats)), grads = grad_fn(state.params)
    
    # 5. Gradient Clipping
    # Prevent gradients from exploding by clipping their global norm
    grads = optax.clip_by_global_norm(1.0).update(grads, state)
    
    # 6. Update State
    state = state.apply_gradients(
        grads=grads,
        batch_stats=new_batch_stats
    )
    
    # 7. Update EMA Parameters
    state = state.apply_ema(decay=0.999)
    
    # 8. Compute Metrics
    accuracy = jnp.mean(jnp.argmax(logits, -1) == batch['label'])
    metrics = {
        'loss': loss,
        'accuracy': accuracy,
        'grad_norm': optax.global_norm(grads),  # Track gradient scale
    }
    
    return state, metrics
```

## Best Practices

- **JIT Compilation**: Always decorate your step function with `@jax.jit` for performance.
- **Pure Functions**: Ensure your step function is pure (no side effects). Return the new state explicitly.
- **Metric Collection**: Return a dictionary of scalar metrics. This makes logging to tools like TensorBoard or Weights & Biases trivial.
- **Debugging**: If you hit `NaNs`, use `jax.debug.print` inside the JIT-compiled function or temporarily disable JIT with `jax.disable_jit()` context manager.
