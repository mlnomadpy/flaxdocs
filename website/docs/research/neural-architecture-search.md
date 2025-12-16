---
sidebar_position: 5
---

# Neural Architecture Search

Neural Architecture Search (NAS) automates the design of neural networks. Instead of manually designing layers, we learn the optimal architecture from data.

## Differentiable Architecture Search (DARTS)

Traditional NAS used reinforcement learning or evolutionary algorithms, which are extremely computationally expensive (thousands of GPU hours). **DARTS** (Differentiable Architecture Search) relaxes the discrete search space into a continuous one, allowing us to use standard gradient descent.

### The Search Space

Instead of choosing *one* operation (e.g., 3x3 Conv OR MaxPool), we compute a weighted sum of *all* candidate operations.

Let $\mathcal{O}$ be the set of candidate operations (e.g., identity, conv3x3, maxpool). The output of a mixed operation $\bar{o}(x)$ is:

$$
\bar{o}(x) = \sum_{o \in \mathcal{O}} \frac{\exp(\alpha_o)}{\sum_{o' \in \mathcal{O}} \exp(\alpha_{o'})} o(x)
$$

where $\alpha_o$ are the learnable **architecture parameters**.
- If $\alpha_{conv3x3} \gg \text{others}$, the operation behaves like a 3x3 convolution.
- After training, we pick the operation with the highest $\alpha$.

### Implementing the Mixed Operation

```python
from flax import linen as nn
import jax
import jax.numpy as jnp
import optax

class DARTSCell(nn.Module):
    """
    DARTS mixed operation cell.
    COMPUTES: weighted sum of all candidate operations.
    """
    
    @nn.compact
    def __call__(self, x, arch_params):
        """
        Apply mixed operation.
        
        Args:
            x: Input tensor.
            arch_params: Vector of logits for architecture weights.
        """
        
        # Define candidate operations
        # In a real implementation, you might have separate classes for these
        ops = [
            lambda x: x,  # Identity
            lambda x: nn.Conv(features=x.shape[-1], kernel_size=(3, 3), padding='SAME')(x),
            lambda x: nn.Conv(features=x.shape[-1], kernel_size=(5, 5), padding='SAME')(x),
            # Note: Pooling usually requires same stride/padding handling
            lambda x: nn.avg_pool(x, window_shape=(3, 3), strides=(1, 1), padding='SAME'),
            lambda x: nn.max_pool(x, window_shape=(3, 3), strides=(1, 1), padding='SAME'),
        ]
        
        # Compute softmax weights from architecture parameters
        arch_weights = jax.nn.softmax(arch_params)
        
        # Weighted sum: sum(w_i * op_i(x))
        # This is the "continuous relaxation"
        result = sum(w * op(x) for w, op in zip(arch_weights, ops))
        
        return result
```

## The Bilevel Optimization Problem

DARTS aims to minimize the validation loss with respect to architecture parameters $\alpha$, where the model weights $w$ are optimal for that architecture:

$$
\min_\alpha \mathcal{L}_{val}(w^*(\alpha), \alpha) \\
\text{s.t. } w^*(\alpha) = \text{argmin}_w \mathcal{L}_{train}(w, \alpha)
$$

This is a **bilevel optimization**:
1. **Inner Loop**: Find best weights $w$ for current architecture (on Train set).
2. **Outer Loop**: Find best architecture $\alpha$ assuming optimal weights (on Validation set).

### Alternating Optimization

In practice, we approximate this by alternating updates:
1. **Update $\alpha$**: One step of gradient descent on Validation loss.
2. **Update $w$**: One step of gradient descent on Training loss.

```python
def nas_train_step(model_state, arch_state, train_batch, val_batch):
    """
    Single step of DARTS alternating optimization.
    
    Args:
        model_state: TrainState for model weights (w).
        arch_state: TrainState for architecture parameters (alpha).
        train_batch: Batch for weight update.
        val_batch: Batch for architecture update.
    """
    
    # 1. Update Architecture (Alpha) on Validation Set
    def arch_loss_fn(arch_params):
        # Forward pass using current weights w and candidate alpha
        logits = model_state.apply_fn(
            {'params': model_state.params, 'arch_params': arch_params},
            val_batch['image']
        )
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits,
            labels=val_batch['label']
        ).mean()
        return loss
    
    # Compute gradients wrt alpha
    arch_grads = jax.grad(arch_loss_fn)(arch_state.params)
    # Update alpha
    arch_state = arch_state.apply_gradients(grads=arch_grads)
    
    # 2. Update Weights (w) on Training Set
    def model_loss_fn(params):
        # Forward pass using candidate w and updated alpha
        logits = model_state.apply_fn(
            {'params': params, 'arch_params': arch_state.params},
            train_batch['image']
        )
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits,
            labels=train_batch['label']
        ).mean()
        return loss
    
    # Compute gradients wrt w
    model_grads = jax.grad(model_loss_fn)(model_state.params)
    # Update w
    model_state = model_state.apply_gradients(grads=model_grads)
    
    return model_state, arch_state
```

## Practical Considerations

- **Memory Usage**: DARTS is memory intensive because you must hold *all* candidate operations in memory during the forward/backward pass.
- **Proxy Tasks**: Often search is done on a smaller proxy task (e.g., fewer layers, fewer epochs) and the found architecture is transferred to the full task.
- **Evaluation**: After search, the mixed operations are usually "discretized" by picking the `argmax` operation and retraining from scratch.

## References
- [DARTS: Differentiable Architecture Search](https://arxiv.org/abs/1806.09055) (Liu et al., 2018)
