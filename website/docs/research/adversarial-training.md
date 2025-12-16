---
sidebar_position: 6
---

# Adversarial Training

Deep neural networks are known to be vulnerable to **adversarial examples**: small, imperceptible perturbations to the input that cause the model to make incorrect predictions with high confidence. Adversarial training is the process of training the model on these perturbed examples to improve robustness.

## The Goal: Robust Optimization

Standard training minimizes the expected loss:
$$
\min_\theta \mathbb{E}_{(x,y) \sim \mathcal{D}} [L(f_\theta(x), y)]
$$

Adversarial training solves a **min-max** problem: finding parameters $\theta$ that minimize the loss on the *worst-case* perturbation $\delta$ within a small region $\epsilon$:

$$
\min_\theta \mathbb{E}_{(x,y) \sim \mathcal{D}} \left[ \max_{\|\delta\| \le \epsilon} L(f_\theta(x + \delta), y) \right]
$$

## Fast Gradient Sign Method (FGSM)

Finding the exact "worst-case" perturbation is hard. The **Fast Gradient Sign Method (FGSM)** approximates it by taking a single step in the direction of the gradient of the loss with respect to the input.

The adversarial perturbation $\delta$ is given by:

$$
\delta = \epsilon \cdot \text{sign}(\nabla_x J(\theta, x, y))
$$

where:
- $\epsilon$ is the magnitude of the perturbation (e.g., 0.01).
- $\nabla_x J(\theta, x, y)$ is the gradient of the loss with respect to the input image $x$.
- `sign` takes the element-wise sign (-1 or +1).

The adversarial example is then $x_{adv} = x + \delta$.

### Implementation in JAX

In JAX, we can compute gradients with respect to inputs just as easily as parameters.

```python
import jax
import jax.numpy as jnp
import optax
from flax import linen as nn

def fgsm_attack(state, batch, epsilon=0.1):
    """
    Generate adversarial examples using FGSM.
    
    Args:
        state: TrainState containing model parameters.
        batch: Dictionary with 'image' and 'label'.
        epsilon: Perturbation magnitude.
    """
    
    def loss_fn(x):
        # We only need the loss value to compute gradients wrt x
        logits = state.apply_fn({'params': state.params}, x)
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits,
            labels=batch['label']
        ).mean()
        return loss
    
    # Compute gradient of loss wrt input image
    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(batch['image'])
    
    # Create perturbation: epsilon * sign(gradient)
    perturbation = epsilon * jnp.sign(grads)
    
    # Add perturbation to original image
    adversarial_images = batch['image'] + perturbation
    
    # Clip to valid value range (e.g., 0 to 1 for images)
    adversarial_images = jnp.clip(adversarial_images, 0.0, 1.0)
    
    return adversarial_images
```

## Adversarial Training Step

To train a robust model, we combine clean and adversarial examples. A common strategy is to train on a mixture of both.

```python
@jax.jit
def adversarial_train_step(state, batch, epsilon=0.1):
    """
    Single training step including adversarial examples.
    """
    
    # 1. Generate Adversarial Examples
    # Note: We use the current state to generate them on-the-fly
    adv_images = fgsm_attack(state, batch, epsilon)
    
    def loss_fn(params):
        # 2. Compute Loss on Clean Data (Standard Objective)
        clean_logits = state.apply_fn({'params': params}, batch['image'])
        clean_loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=clean_logits,
            labels=batch['label']
        ).mean()
        
        # 3. Compute Loss on Adversarial Data (Robust Objective)
        adv_logits = state.apply_fn({'params': params}, adv_images)
        adv_loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=adv_logits,
            labels=batch['label']
        ).mean()
        
        # 4. Combined Loss
        # Weighting 50/50 is common, but adjustable
        total_loss = 0.5 * clean_loss + 0.5 * adv_loss
        
        return total_loss
    
    # Compute gradients and update model
    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    
    return state
```

## Considerations

- **Performance Trade-off**: Robust models often have slightly lower accuracy on clean data compared to standard models. This is a known trade-off.
- **Computational Cost**: Generating attacks requires an extra forward and backward pass per step, effectively doubling training time.
- **Iterative Attacks**: FGSM is a "single-step" attack. Stronger attacks like **PGD (Projected Gradient Descent)** take multiple small steps to find a better perturbation, resulting in even more robust models at higher training cost.

## References
- [Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572) (Goodfellow et al., 2014) - The FGSM paper.
