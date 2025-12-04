---
sidebar_position: 2
---

# Meta-Learning with MAML

Learn Model-Agnostic Meta-Learning (MAML) - a powerful technique for few-shot learning that learns good parameter initializations for rapid adaptation to new tasks.

:::info Example Code
See the full implementation: [`examples/14_meta_learning_maml.py`](https://github.com/yourusername/flaxdocs/blob/master/examples/14_meta_learning_maml.py)
:::

## The Few-Shot Learning Challenge

Traditional machine learning needs thousands of examples. Humans learn from just a few:

- See 3 examples of a new bird species → recognize it
- Try a new task a few times → become proficient
- Observe a few demonstrations → imitate behavior

**Few-shot learning** aims to match this capability: learn from $K$ labeled examples per class.

## What is MAML?

**Model-Agnostic Meta-Learning** learns an initialization that allows rapid adaptation to new tasks with just a few gradient steps.

**Key insight**: Some initializations are easier to fine-tune than others. MAML finds the initialization that's universally good across tasks.

```
┌────────────────────────────────────────────┐
│           MAML in One Picture              │
└────────────────────────────────────────────┘

Meta-Parameters θ (good initialization)
            │
     ┌──────┼──────┬──────┐
     │      │      │      │
   Task 1  Task 2 Task 3 ...
     │      │      │
  [Adapt]  [Adapt] [Adapt]  ← Few gradient steps
     │      │      │
    θ₁'    θ₂'    θ₃'  ← Task-specific parameters
     │      │      │
  [Evaluate on query set]
     │      │      │
    L₁     L₂     L₃
     │      │      │
     └──────┴──────┘
            │
    [Update θ to minimize avg loss]
```

## The MAML Algorithm

MAML has two loops:

### Inner Loop (Task Adaptation)

For each task $\mathcal{T}_i$:

1. Sample support set $\mathcal{D}^{\text{sup}}_i = \{(x, y)\}$ (few examples)
2. Compute loss on support set: $\mathcal{L}_{\mathcal{T}_i}(\theta)$
3. Take gradient steps to adapt:

$$
\theta_i' = \theta - \alpha \nabla_\theta \mathcal{L}_{\mathcal{T}_i}(\theta)
$$

**Key**: These are just **temporary** parameter updates for this specific task.

### Outer Loop (Meta-Update)

1. Evaluate adapted parameters on query set: $\mathcal{L}_{\mathcal{T}_i}(\theta_i')$
2. Update meta-parameters to minimize query loss:

$$
\theta \leftarrow \theta - \beta \nabla_\theta \sum_{\mathcal{T}_i} \mathcal{L}_{\mathcal{T}_i}(\theta_i')
$$

**Critical**: We compute gradients through the inner loop adaptation!

## Mathematical Foundation

### The Meta-Objective

MAML optimizes:

$$
\min_\theta \sum_{\mathcal{T}_i \sim p(\mathcal{T})} \mathcal{L}_{\mathcal{T}_i}(f_{\theta_i'})
$$

where $\theta_i' = \theta - \alpha \nabla_\theta \mathcal{L}_{\mathcal{T}_i}(f_\theta)$

This is a **bilevel optimization**:
- **Inner level**: Adapt to task (fixed $\theta$, optimize $\theta'$)
- **Outer level**: Find best initialization (optimize $\theta$)

### Gradient Computation

The meta-gradient requires computing gradients through the adaptation:

$$
\frac{\partial}{\partial \theta} \mathcal{L}_{\mathcal{T}_i}(\theta_i') = \frac{\partial \mathcal{L}_{\mathcal{T}_i}(\theta_i')}{\partial \theta_i'} \cdot \frac{\partial \theta_i'}{\partial \theta}
$$

where:

$$
\frac{\partial \theta_i'}{\partial \theta} = I - \alpha \nabla^2_\theta \mathcal{L}_{\mathcal{T}_i}(\theta)
$$

**First-order MAML** approximates by ignoring second-order term: $\frac{\partial \theta_i'}{\partial \theta} \approx I$

## Implementation in Flax NNX

### Simple Model for Few-Shot Learning

```python
from flax import nnx
import jax.numpy as jnp

class MAMLRegressor(nnx.Module):
    """Simple MLP for regression tasks (e.g., sine wave fitting)."""
    
    def __init__(self, hidden_dim: int = 40, *, rngs: nnx.Rngs):
        self.fc1 = nnx.Linear(1, hidden_dim, rngs=rngs)
        self.fc2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.fc3 = nnx.Linear(hidden_dim, 1, rngs=rngs)
    
    def __call__(self, x):
        x = self.fc1(x)
        x = nnx.relu(x)
        x = self.fc2(x)
        x = nnx.relu(x)
        x = self.fc3(x)
        return x
```

### Inner Loop: Task Adaptation

The inner loop adapts parameters to a specific task:

```python
def inner_loop(model, support_x, support_y, inner_lr=0.01, inner_steps=1):
    """
    Adapt model to a task using support set.
    
    Args:
        model: Model to adapt
        support_x: Support inputs [k_shot, input_dim]
        support_y: Support targets [k_shot, output_dim]
        inner_lr: Learning rate for adaptation
        inner_steps: Number of gradient steps
    
    Returns:
        adapted_params: Parameters adapted to this task
    """
    # Get initial parameters
    params = nnx.state(model, nnx.Param)
    
    # Inner loop adaptation
    for step in range(inner_steps):
        def loss_fn(params):
            # Temporarily update model
            nnx.update(model, params)
            predictions = model(support_x)
            loss = jnp.mean((predictions - support_y) ** 2)
            return loss
        
        # Compute gradients
        grads = jax.grad(loss_fn)(params)
        
        # Manual SGD update (inner loop)
        params = jax.tree_map(
            lambda p, g: p - inner_lr * g,
            params,
            grads
        )
    
    return params
```

**Key points**:
- We manually implement SGD (not using optimizer)
- Updates are temporary (just for this task)
- Gradients will flow through these operations in outer loop

### Meta-Loss: Outer Loop Objective

Compute loss on query set after adaptation:

```python
def maml_meta_loss(model, support_batch, query_batch, inner_lr=0.01, inner_steps=1):
    """
    Compute meta-loss: average loss on query sets after adaptation.
    
    This is the loss we actually optimize (backprop through inner loop).
    """
    batch_size = support_batch['x'].shape[0]
    total_loss = 0.0
    
    for i in range(batch_size):
        # Get support and query for this task
        sup_x = support_batch['x'][i]
        sup_y = support_batch['y'][i]
        que_x = query_batch['x'][i]
        que_y = query_batch['y'][i]
        
        # Inner loop: adapt to task
        adapted_params = inner_loop(model, sup_x, sup_y, inner_lr, inner_steps)
        
        # Evaluate on query set with adapted parameters
        nnx.update(model, adapted_params)
        predictions = model(que_x)
        task_loss = jnp.mean((predictions - que_y) ** 2)
        
        total_loss += task_loss
    
    # Average across tasks
    return total_loss / batch_size
```

### Training Step: Meta-Update

```python
@nnx.jit
def maml_train_step(model: MAMLRegressor, optimizer: nnx.Optimizer,
                    support_batch: Dict, query_batch: Dict,
                    inner_lr: float = 0.01, inner_steps: int = 1):
    """
    MAML training step:
    1. Adapt to each task (inner loop)
    2. Evaluate on query sets
    3. Update meta-parameters (outer loop)
    """
    
    def meta_loss_fn(model):
        return maml_meta_loss(model, support_batch, query_batch, 
                            inner_lr, inner_steps)
    
    # Compute meta-gradients (through inner loop!)
    grad_fn = nnx.value_and_grad(meta_loss_fn)
    loss, grads = grad_fn(model)
    
    # Meta-update (outer loop)
    optimizer.update(grads)
    
    return {'loss': loss}
```

## Understanding MAML Intuitively

### Visualization of Parameter Space

```
┌──────────────────────────────────────────────┐
│         Parameter Space Landscape            │
└──────────────────────────────────────────────┘

     Task 1 optimal ●              ● Task 2 optimal
                      \            /
                       \          /
                        \        /
                         \      /
                          \    /
                           \  /
                      θ* (MAML init)
                           /  \
                          /    \
                         /      \
                        /        \
                       /          \
                      /            \
     Task 3 optimal ●              ● Task 4 optimal

MAML finds θ* in the center, equidistant from all tasks.
A few gradient steps reach any task optimum.
```

### Why It Works

**Without MAML**: Random initialization far from all task optima  
**With MAML**: Initialization central to all tasks, few steps suffice

**Analogy**: 
- Random init = dropped in random city, need detailed directions
- MAML init = dropped in city center, quick walk to any landmark

## Task Distribution

MAML assumes a distribution over tasks. Example for sine wave regression:

```python
def generate_sinusoid_task(amplitude_range=(0.1, 5.0), 
                          phase_range=(0, np.pi)):
    """Generate random sine wave task."""
    amplitude = np.random.uniform(*amplitude_range)
    phase = np.random.uniform(*phase_range)
    
    def task_fn(x):
        return amplitude * np.sin(x + phase)
    
    return task_fn

def sample_task_data(task_fn, n_samples=10):
    """Sample input-output pairs from task."""
    x = np.random.uniform(-5.0, 5.0, size=(n_samples, 1))
    y = task_fn(x)
    return x.astype(np.float32), y.astype(np.float32)
```

**For each task**:
- Sample $K$ support examples (few-shot)
- Sample $Q$ query examples (evaluation)
- Support set for adaptation, query set for meta-gradient

## Training Loop

```python
def train_maml(
    num_iterations: int = 10000,
    meta_batch_size: int = 25,
    meta_lr: float = 1e-3,
    inner_lr: float = 0.01,
    inner_steps: int = 1,
    k_shot: int = 10,
):
    """Train MAML model."""
    
    # Initialize model
    model = MAMLRegressor(hidden_dim=40, rngs=nnx.Rngs(0))
    optimizer = nnx.Optimizer(model, optax.adam(meta_lr))
    
    # Meta-training
    for iteration in range(num_iterations):
        # Sample batch of tasks
        support_batch, query_batch = sample_task_batch(
            batch_size=meta_batch_size,
            k_shot=k_shot,
            k_query=10
        )
        
        # MAML training step
        metrics = maml_train_step(
            model, optimizer, support_batch, query_batch,
            inner_lr=inner_lr, inner_steps=inner_steps
        )
        
        if (iteration + 1) % 1000 == 0:
            print(f"Iteration {iteration+1} | Loss: {metrics['loss']:.4f}")
    
    return model
```

## Evaluation: Few-Shot Adaptation

After meta-training, evaluate on new tasks:

```python
def evaluate_adaptation(model, n_test_tasks=10):
    """Test adaptation to new tasks."""
    
    for task_idx in range(n_test_tasks):
        # Sample new task
        task_fn = generate_sinusoid_task()
        
        # Support set (for adaptation)
        support_x, support_y = sample_task_data(task_fn, n_samples=10)
        
        # Query set (for evaluation)
        query_x, query_y = sample_task_data(task_fn, n_samples=50)
        
        # Evaluate BEFORE adaptation
        pred_before = model(query_x)
        loss_before = jnp.mean((pred_before - query_y) ** 2)
        
        # Adapt to task (1 gradient step)
        adapted_params = inner_loop(model, support_x, support_y, 
                                   inner_lr=0.01, inner_steps=1)
        nnx.update(model, adapted_params)
        
        # Evaluate AFTER adaptation
        pred_after = model(query_x)
        loss_after = jnp.mean((pred_after - query_y) ** 2)
        
        print(f"Task {task_idx+1}: "
              f"Before: {loss_before:.4f} → After: {loss_after:.4f}")
```

**Expected behavior**: Loss drops dramatically after just 1-5 gradient steps!

## Hyperparameters

### Inner Learning Rate α

Controls adaptation speed:

**Too small** (0.001):
- Slow adaptation
- Need many inner steps

**Too large** (0.1):
- Unstable adaptation
- May overshoot optimal

**Good range**: 0.01 - 0.05

### Inner Steps

Number of gradient steps for adaptation:

**1 step**:
- Fastest
- Tests if initialization is truly good

**5-10 steps**:
- More thorough adaptation
- Better final performance

**Trade-off**: More steps = better task performance but slower meta-training

### Meta Learning Rate β

Learning rate for meta-parameters:

**Typical**: 1e-3 (Adam optimizer)

**Key**: Usually smaller than standard supervised learning

### Meta Batch Size

Number of tasks per meta-update:

**Small** (5-10):
- Less memory
- More noise in meta-gradient

**Large** (25-50):
- More stable
- Better gradient estimates

## Practical Considerations

### First-Order MAML

**Full MAML**: Compute second-order derivatives (expensive!)  
**First-Order MAML**: Ignore Hessian (approximation)

```python
# First-order: stop gradients through inner loop
adapted_params = jax.lax.stop_gradient(
    inner_loop(model, support_x, support_y, inner_lr, inner_steps)
)
```

**Trade-off**: 
- First-order: Faster, slightly worse performance
- Full MAML: Slower, theoretically better

Most practitioners use first-order.

### Task Diversity

MAML needs diverse tasks during meta-training:

**Similar tasks**: Model just learns average task  
**Diverse tasks**: Model learns to adapt quickly

**For vision**: Use different classes per task  
**For sine waves**: Vary amplitude and phase widely

### Overfitting to Training Tasks

Monitor performance on held-out tasks:

```python
# Meta-train on tasks 1-1000
train_tasks = sample_tasks(1000)

# Meta-test on tasks 1001-1100  
test_tasks = sample_tasks(100, offset=1000)

# Evaluate adaptation on test tasks
evaluate_on_new_tasks(model, test_tasks)
```

## Extensions and Variants

### Reptile

Simpler alternative to MAML:

1. Sample task
2. Take multiple gradient steps (fully adapt)
3. Move meta-parameters toward adapted parameters

**Advantage**: No gradients through inner loop!  
**Trade-off**: Slightly worse than MAML

### MAML++

Improvements over vanilla MAML:

- Multi-step loss optimization (MSL)
- Annealing inner learning rate
- Batch normalization per step

### Meta-SGD

Learns per-parameter learning rates:

$$
\theta_i' = \theta - \alpha_\theta \odot \nabla_\theta \mathcal{L}_{\mathcal{T}_i}(\theta)
$$

where $\alpha_\theta$ are learned (element-wise learning rates)

## Mathematical Deep Dive

### Why Does MAML Work?

**Theorem** (informal): MAML finds parameters $\theta^*$ such that the expected loss after adaptation is minimized:

$$
\theta^* = \arg\min_\theta \mathbb{E}_{\mathcal{T} \sim p(\mathcal{T})} [\mathcal{L}_\mathcal{T}(\theta - \alpha \nabla \mathcal{L}_\mathcal{T}(\theta))]
$$

**Intuition**: Good initialization makes gradient descent work well across all tasks.

### Connection to Transfer Learning

Standard transfer learning:
```
Pre-train on large dataset → Fine-tune on target task
```

MAML:
```
Meta-train on task distribution → Adapt to new task
```

**Difference**: MAML explicitly optimizes for fine-tuning, not just features.

### Gradient Computation Complexity

**Forward pass**: $O(TK)$ where $T$ = tasks, $K$ = inner steps  
**Backward pass**: $O(TK)$ with first-order, $O(TK^2)$ full MAML

**Memory**: Stores all intermediate activations for backprop through inner loop

## Common Pitfalls

### 1. Too Many Inner Steps
❌ **Problem**: Overfits to support set  
✅ **Solution**: Use 1-5 steps, validate on query set

### 2. Same Support/Query Data
❌ **Problem**: Model memorizes, doesn't generalize  
✅ **Solution**: Always separate support from query

### 3. No Task Diversity
❌ **Problem**: Model doesn't learn to adapt  
✅ **Solution**: Ensure wide range of tasks

### 4. Wrong Learning Rates
❌ **Problem**: No improvement during meta-training  
✅ **Solution**: Tune inner_lr (0.01) and meta_lr (1e-3) separately

## Running the Example

Train MAML on sine wave regression:

```bash
cd examples
python 14_meta_learning_maml.py
```

Expected output:
```
MODEL-AGNOSTIC META-LEARNING (MAML)
Training for 10000 iterations...
Iteration 1000 | Meta Loss: 0.8234
Iteration 2000 | Meta Loss: 0.3421
...

EVALUATING FEW-SHOT ADAPTATION
Task 1 | Before: 2.3456 → After 1 step: 0.1234
Task 2 | Before: 1.9876 → After 1 step: 0.0987
...
```

The dramatic improvement after just one step demonstrates MAML's power!

## Next Steps

- Apply to classification tasks (Omniglot, Mini-ImageNet)
- Try Reptile for simpler implementation
- Experiment with different architectures
- Explore task-conditional models

## References

- **MAML Paper**: [Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](https://arxiv.org/abs/1703.03400) (Finn et al., ICML 2017)
- **Reptile**: [On First-Order Meta-Learning Algorithms](https://arxiv.org/abs/1803.02999)
- **MAML++**: [How to train your MAML](https://arxiv.org/abs/1810.09502)
- **Meta-SGD**: [Meta-SGD: Learning to Learn Quickly for Few-Shot Learning](https://arxiv.org/abs/1707.09835)
