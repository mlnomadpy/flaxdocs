---
sidebar_position: 7
---

# Curriculum Learning

Humans learn better when concepts are introduced in a specific order: simple addition before multiplication, multiplication before calculus. **Curriculum Learning** applies this principle to machine learning: instead of sampling batches randomly ($U(D)$), we sample from a distribution $P_t(D)$ that changes over time to present easy examples first, then gradually harder ones.

## The Theory of Curriculum

A curriculum is defined by a **Difficulty Scorer** $S(x)$ and a **Pacing Function** $\lambda(t)$.

### 1. Difficulty Scoring $S(x)$

How do we measure difficulty? Common heuristics include:

- **Sentence Length** (NLP): Longer sentences are harder.
- **Noise Level**: Examples with lower signal-to-noise ratio.
- **Teacher Uncertainty**: A pre-trained model has high entropy/loss on the example.
- **Transfer Scoring**: Loss from a model trained on a generic dataset.

### 2. Pacing Functions $\lambda(t)$

The pacing function $\lambda(t) \in (0, 1]$ determines the fraction of the dataset available at training step $t$.

**Linear Pacing**:
$$
\lambda(t) = \min(1, \lambda_0 + (1 - \lambda_0) \cdot \frac{t}{T_{growth}})
$$

**Root Pacing** (More time on hard examples):
$$
\lambda(t) = \min(1, \sqrt{\lambda_0^2 + (1 - \lambda_0^2) \cdot \frac{t}{T_{growth}}})
$$

**Geometric Pacing** (More time on easy examples):
$$
\lambda(t) = \min(1, \lambda_0 \cdot \exp(\alpha \cdot t))
$$

Where $\lambda_0$ is the initial data fraction and $T_{growth}$ is the number of steps to reach full dataset.

## Implementation: Dynamic Data Sampling

We can implement curriculum learning by dynamically filtering the dataset during training.

```python
import jax
import jax.numpy as jnp

class CurriculumScheduler:
    """
    Manages the curriculum pacing and data sampling.
    """
    
    def __init__(self, num_stages=10, growth_steps=10000, function='linear'):
        self.num_stages = num_stages
        self.growth_steps = growth_steps
        self.function = function
    
    def get_pacing_rate(self, step):
        """
        Compute available data fraction (lambda) based on step.
        """
        # Linear pacing example
        if self.function == 'linear':
            rate = min(1.0, 0.1 + 0.9 * (step / self.growth_steps))
            return rate
        elif self.function == 'root':
             rate = min(1.0, jnp.sqrt(0.1**2 + (1 - 0.1**2) * (step / self.growth_steps)))
             return rate
        return 1.0
    
    def sample_batch(self, dataset, step, batch_size, rng_key):
        """
        Sample a batch from the 'available' slice of the dataset.
        Assumes dataset is pre-sorted by difficulty!
        """
        pacing_rate = self.get_pacing_rate(step)
        
        # Determine how many examples are 'unlocked'
        num_examples = len(dataset)
        max_index = int(pacing_rate * num_examples)
        max_index = max(max_index, batch_size)  # Ensure minimum data
        
        # Consider only the available slice
        available_data = dataset[:max_index]
        
        # Randomly sample indices from this slice
        indices = jax.random.choice(
            rng_key,
            len(available_data),
            shape=(batch_size,),
            replace=False
        )
        
        return available_data[indices]
```

## Advanced: Self-Paced Learning (SPL)

Pre-defining a curriculum is rigid. **Self-Paced Learning** learns the curriculum jointly with the model parameters $w$.

Objective function:
$$
\min_{w, v} \mathbb{E} \left[ \sum_{i=1}^N v_i L(y_i, f_w(x_i)) - \lambda \sum_{i=1}^N v_i \right]
$$

Where:
- $v_i \in \{0, 1\}$ indicates if example $i$ is selected.
- $\lambda$ is a regularization term (the "age" of the curriculum).

**Optimization**:
1. **Fix $v$, min $w$**: Standard SGD training on selected examples.
2. **Fix $w$, min $v$**: Closed-form solution:
   $$
   v_i^* = \begin{cases} 1 & \text{if } L_i < \lambda \\ 0 & \text{otherwise} \end{cases}
   $$

This means the model trains on all examples with loss smaller than $\lambda$. We gradually increase $\lambda$ to include harder (higher loss) examples.

```python
def self_paced_mask(losses, lambda_threshold):
    """
    Generate mask for Self-Paced Learning.
    Selects examples where loss < lambda.
    """
    return losses < lambda_threshold

@jax.jit
def spl_train_step(state, batch, lambda_threshold):
    """
    Training step with Self-Paced weighting.
    """
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, batch['image'])
        # Compute individual losses
        losses = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=batch['label']
        )
        
        # Compute V matrix (selection)
        # Note: We detach selection from gradient!
        v_mask = jax.lax.stop_gradient(losses < lambda_threshold)
        
        # Weighted loss (only train on selected examples)
        # Avoid division by zero
        mean_loss = jnp.sum(losses * v_mask) / (jnp.sum(v_mask) + 1e-6)
        
        return mean_loss, jnp.mean(v_mask) # Return fraction used
        
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, fraction_used), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    
    return state, fraction_used
```

## Mentorship (Teacher-Student)

Another variation involves a "Teacher" model helping the main model:
1. Train a large Teacher model on the dataset.
2. Teacher scores difficulty of all examples.
3. Student trains using curriculum derived from Teacher's scores.

This is robust because the Teacher's "difficulty" acts as a proxy for the Student's expected error.

## References
- [Curriculum Learning](https://icml.cc/Conferences/2009/papers/592.pdf) (Bengio et al., 2009)
- [Self-Paced Learning for Long-Term Tracking](https://arxiv.org/abs/1301.7397) (Supancic et al., 2013)
- [Automated Curriculum Learning for Neural Networks](https://arxiv.org/abs/1704.03003) (Graves et al., 2017)
