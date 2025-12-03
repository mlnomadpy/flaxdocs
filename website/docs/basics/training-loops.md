---
sidebar_position: 3
---

# Training Loops and Optimization

Learn how to write effective training loops in Flax NNX. This guide covers optimization, gradient computation, JIT compilation, and best practices for stable, fast training.

## Anatomy of a Training Loop

Every training loop has the same basic structure:

```python
# 1. Initialize model and optimizer
model = MyModel(rngs=nnx.Rngs(params=0))
optimizer = nnx.Optimizer(model, optax.adam(learning_rate=1e-3))

# 2. Loop over epochs
for epoch in range(num_epochs):
    
    # 3. Loop over batches
    for batch in train_loader:
        images, labels = batch
        
        # 4. Forward pass: compute loss
        logits = model(images)
        loss = cross_entropy_loss(logits, labels)
        
        # 5. Backward pass: compute gradients
        grads = jax.grad(loss_fn)(model_params)
        
        # 6. Update parameters
        optimizer.update(grads)
    
    # 7. Evaluate
    val_loss = evaluate(model, val_loader)
    print(f"Epoch {epoch}: val_loss = {val_loss:.4f}")
```

But the devil is in the details! Let's understand each part deeply.

## Loss Functions: The Training Objective

### Classification: Cross-Entropy Loss

```python
import jax.numpy as jnp
import optax

def cross_entropy_loss(logits, labels):
    """
    Cross-entropy loss for classification
    
    Args:
        logits: (batch, num_classes) - raw model outputs
        labels: (batch, num_classes) - one-hot encoded targets
    
    Returns:
        scalar loss value
    """
    # Compute log probabilities (numerically stable)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    
    # Cross-entropy: -sum(y * log(y_hat))
    loss = -jnp.sum(labels * log_probs, axis=-1)  # (batch,)
    
    # Average over batch
    return jnp.mean(loss)

# Alternative: use Optax (recommended)
def cross_entropy_loss_optax(logits, labels):
    return optax.softmax_cross_entropy(logits, labels).mean()
```

**Why log_softmax?**
- Numerically stable (avoids exp(large_number) overflow)
- More accurate gradients
- Single operation instead of softmax + log

**Why mean over batch?**
- Makes loss independent of batch size
- Consistent learning rates across different batch sizes
- Standard practice in deep learning

### Regression: Mean Squared Error

```python
def mse_loss(predictions, targets):
    """
    Mean squared error for regression
    
    Args:
        predictions: (batch, output_dim)
        targets: (batch, output_dim)
    """
    squared_diff = (predictions - targets) ** 2
    return jnp.mean(squared_diff)
```

### Adding Regularization

Prevent overfitting with L2 regularization (weight decay):

```python
def loss_with_l2(logits, labels, model, l2_weight=1e-4):
    """Loss with L2 regularization on parameters"""
    
    # Main loss
    ce_loss = cross_entropy_loss(logits, labels)
    
    # L2 regularization: sum of squared weights
    l2_loss = 0.0
    for param in jax.tree_util.tree_leaves(nnx.state(model)):
        if isinstance(param, jnp.ndarray):
            l2_loss += jnp.sum(param ** 2)
    
    return ce_loss + l2_weight * l2_loss
```

**Note**: Modern practice uses weight decay in the optimizer (AdamW) instead of L2 in loss. More on this below.

## Gradient Computation with JAX

### Basic Gradient Computation

```python
def compute_loss(model, batch):
    """Compute loss for a single batch"""
    images, labels = batch
    logits = model(images)
    return cross_entropy_loss(logits, labels)

# Compute gradients
loss, grads = jax.value_and_grad(compute_loss)(model, batch)
```

**Understanding `jax.grad`**:
- Takes a function that returns a scalar
- Returns a function that computes gradients
- By default, differentiates w.r.t. first argument

**Using `value_and_grad`**:
- Returns both loss value and gradients
- More efficient than calling `grad` and computing loss separately
- Use this in training loops!

### The Training Step Pattern

```python
def create_train_step(model, optimizer):
    """Create a JIT-compiled training step function"""
    
    def loss_fn(model):
        """Loss function that returns scalar and auxiliary info"""
        logits = model(batch['images'])
        loss = cross_entropy_loss(logits, batch['labels'])
        
        # Compute accuracy for logging
        preds = jnp.argmax(logits, axis=-1)
        targets = jnp.argmax(batch['labels'], axis=-1)
        accuracy = jnp.mean(preds == targets)
        
        return loss, {'accuracy': accuracy}
    
    @nnx.jit  # JIT compile for speed
    def train_step(model, optimizer, batch):
        # Compute loss and gradients
        grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
        (loss, aux), grads = grad_fn(model)
        
        # Update parameters
        optimizer.update(grads)
        
        return loss, aux
    
    return train_step
```

**Key concepts**:
- **`has_aux=True`**: Loss function returns (loss, aux_dict)
- **Auxiliary outputs**: Return metrics without computing separate forward pass
- **JIT compilation**: Compiles function to optimized XLA code (10-100x speedup)

## Optimizers: Making Updates

### Understanding Optimizers

An optimizer defines **how** to use gradients to update parameters:

```python
# Gradient descent: θ_new = θ_old - lr * ∇loss
θ = θ - learning_rate * grad

# But modern optimizers are much more sophisticated!
```

### Adam: The Default Choice

```python
import optax

# Create Adam optimizer
tx = optax.adam(learning_rate=1e-3)

# Wrap model with NNX optimizer
optimizer = nnx.Optimizer(model, tx)

# Update parameters
optimizer.update(grads)
```

**Why Adam?**
- Adaptive learning rates per parameter
- Handles sparse gradients well
- Relatively insensitive to hyperparameters
- Good default for most problems

**Adam hyperparameters**:
```python
tx = optax.adam(
    learning_rate=1e-3,  # Base learning rate
    b1=0.9,              # Momentum decay
    b2=0.999,            # RMS decay
    eps=1e-8,            # Numerical stability
)
```

### AdamW: Adam with Weight Decay

Better than L2 regularization:

```python
tx = optax.adamw(
    learning_rate=1e-3,
    weight_decay=1e-4,  # Decoupled weight decay
)

optimizer = nnx.Optimizer(model, tx)
```

**AdamW vs Adam + L2**:
- AdamW decouples weight decay from gradient updates
- Better generalization in practice
- Standard for transformers and large models

### SGD with Momentum

Simple but effective:

```python
tx = optax.sgd(
    learning_rate=0.1,
    momentum=0.9,
)

optimizer = nnx.Optimizer(model, tx)
```

**When to use SGD**:
- Training ResNets on ImageNet (better final accuracy)
- When you have good learning rate schedule
- Prefer Adam for quick experimentation

## Learning Rate Schedules

Constant learning rates rarely work well. Use schedules:

### Warmup then Decay

```python
def create_learning_rate_schedule(
    base_lr=1e-3,
    warmup_steps=1000,
    total_steps=100_000,
):
    """Linear warmup followed by cosine decay"""
    
    warmup_schedule = optax.linear_schedule(
        init_value=0.0,
        end_value=base_lr,
        transition_steps=warmup_steps,
    )
    
    decay_schedule = optax.cosine_decay_schedule(
        init_value=base_lr,
        decay_steps=total_steps - warmup_steps,
        alpha=0.01,  # Final LR = 0.01 * base_lr
    )
    
    schedule = optax.join_schedules(
        schedules=[warmup_schedule, decay_schedule],
        boundaries=[warmup_steps],
    )
    
    return schedule

# Use in optimizer
schedule = create_learning_rate_schedule()
tx = optax.adam(learning_rate=schedule)
optimizer = nnx.Optimizer(model, tx)
```

**Why warmup?**
- Prevents divergence at start of training
- Lets optimizer accumulate momentum estimates
- Critical for large batch training

**Why cosine decay?**
- Smooth decay to small learning rate
- Better than step decay in practice
- Common in modern research

### Exponential Decay

```python
schedule = optax.exponential_decay(
    init_value=1e-3,
    transition_steps=1000,
    decay_rate=0.96,
)
```

Every 1000 steps: lr *= 0.96

## Complete Training Loop

Putting it all together:

```python
import jax
import jax.numpy as jnp
from flax import nnx
import optax

def train_model(
    model,
    train_loader,
    val_loader,
    num_epochs=10,
    learning_rate=1e-3,
):
    """Complete training loop with all best practices"""
    
    # 1. Create optimizer with schedule
    total_steps = len(train_loader) * num_epochs
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=1000,
        decay_steps=total_steps,
    )
    
    tx = optax.adamw(learning_rate=schedule, weight_decay=1e-4)
    optimizer = nnx.Optimizer(model, tx)
    
    # 2. Define loss function
    def loss_fn(model, batch):
        logits = model(batch['images'])
        loss = optax.softmax_cross_entropy(
            logits, 
            batch['labels']
        ).mean()
        
        # Compute accuracy
        preds = jnp.argmax(logits, axis=-1)
        targets = jnp.argmax(batch['labels'], axis=-1)
        acc = jnp.mean(preds == targets)
        
        return loss, {'accuracy': acc}
    
    # 3. JIT-compile training step
    @nnx.jit
    def train_step(model, optimizer, batch):
        grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
        (loss, metrics), grads = grad_fn(model, batch)
        optimizer.update(grads)
        return loss, metrics
    
    # 4. Training loop
    for epoch in range(num_epochs):
        # Training
        train_losses = []
        train_accs = []
        
        for batch in train_loader:
            loss, metrics = train_step(model, optimizer, batch)
            train_losses.append(loss)
            train_accs.append(metrics['accuracy'])
        
        # Validation
        val_loss, val_acc = evaluate(model, val_loader)
        
        # Logging
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"  Train Loss: {jnp.mean(jnp.array(train_losses)):.4f}")
        print(f"  Train Acc:  {jnp.mean(jnp.array(train_accs)):.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Val Acc:    {val_acc:.4f}")
    
    return model

def evaluate(model, data_loader):
    """Evaluate model on validation/test set"""
    losses = []
    accs = []
    
    for batch in data_loader:
        logits = model(batch['images'])
        loss = optax.softmax_cross_entropy(
            logits, 
            batch['labels']
        ).mean()
        
        preds = jnp.argmax(logits, axis=-1)
        targets = jnp.argmax(batch['labels'], axis=-1)
        acc = jnp.mean(preds == targets)
        
        losses.append(loss)
        accs.append(acc)
    
    return jnp.mean(jnp.array(losses)), jnp.mean(jnp.array(accs))
```

## Gradient Clipping

Prevent exploding gradients:

```python
# Add gradient clipping to optimizer
tx = optax.chain(
    optax.clip_by_global_norm(1.0),  # Clip gradients to max norm 1.0
    optax.adamw(learning_rate=1e-3, weight_decay=1e-4),
)

optimizer = nnx.Optimizer(model, tx)
```

**When to use**:
- Training RNNs/LSTMs (prone to exploding gradients)
- Training very deep networks
- If you see NaN losses

**How it works**:
```python
# Compute global gradient norm
grad_norm = sqrt(sum(g² for all g in grads))

# If too large, scale down
if grad_norm > max_norm:
    grads = grads * (max_norm / grad_norm)
```

## Mixed Precision Training

Train faster with float16:

```python
# Enable mixed precision
policy = jax.experimental.jax2tf.mixed_precision.default_policy()

# Modify optimizer to handle mixed precision
tx = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.scale_by_adam(),
    optax.scale(-learning_rate),
    optax.apply_if_finite(tx, max_norm=1.0),  # Skip update if NaN/Inf
)
```

**Benefits**:
- 2-3x faster training
- 2x less memory usage
- Enables larger batch sizes

**Challenges**:
- Numerical instability (NaN losses)
- Requires careful scaling
- Not all models benefit equally

## Common Training Issues

### Loss is NaN

**Causes**:
1. Learning rate too high → Exploding gradients
2. Batch norm with small batches
3. Division by zero in loss
4. Numerical overflow in softmax

**Fixes**:
```python
# Lower learning rate
learning_rate = 1e-4  # Instead of 1e-3

# Add gradient clipping
tx = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate),
)

# Use log_softmax instead of softmax
log_probs = jax.nn.log_softmax(logits)

# Check for NaN/Inf
loss = jnp.nan_to_num(loss, nan=1e6, posinf=1e6, neginf=-1e6)
```

### Loss Not Decreasing

**Causes**:
1. Learning rate too small
2. Model not learning anything (dead ReLUs)
3. Data preprocessing issues
4. Optimization problem is too hard

**Debugging**:
```python
# Check gradient norms
grad_norm = jnp.sqrt(sum(
    jnp.sum(g ** 2) for g in jax.tree_util.tree_leaves(grads)
))
print(f"Gradient norm: {grad_norm}")

# If close to 0 → Learning rate too small or vanishing gradients
# If very large → Learning rate too high or exploding gradients

# Check data
print(f"Data mean: {images.mean()}, std: {images.std()}")
# Should be normalized (mean ≈ 0, std ≈ 1)

# Try overfit single batch (sanity check)
for _ in range(100):
    loss = train_step(model, optimizer, single_batch)
print(f"Single batch loss: {loss}")
# Should go close to 0 if model can learn
```

### Training is Slow

**Causes**:
1. Not using JIT compilation
2. Data loading bottleneck
3. Inefficient model operations

**Fixes**:
```python
# Add @nnx.jit to training step (CRITICAL!)
@nnx.jit
def train_step(...):
    ...

# Profile data loading
start = time.time()
for i, batch in enumerate(train_loader):
    if i >= 100: break
print(f"Data throughput: {100 * batch_size / (time.time() - start):.1f} ex/s")
# Should be > 1000 examples/sec for GPU training

# Use prefetch in data pipeline
ds = ds.prefetch(tf.data.AUTOTUNE)
```

## Best Practices Summary

### Optimization
✅ Use AdamW with default hyperparameters (β1=0.9, β2=0.999)  
✅ Add learning rate warmup (1000-5000 steps)  
✅ Use cosine decay for learning rate schedule  
✅ Add gradient clipping for RNNs and large models  
✅ Try weight decay (1e-4 to 1e-5)  

### Training Loop
✅ Always use `@nnx.jit` on training step  
✅ Use `value_and_grad` instead of separate calls  
✅ Return auxiliary metrics (accuracy, etc.) with `has_aux=True`  
✅ Evaluate on validation set every epoch  
✅ Save checkpoints regularly (see checkpointing guide)  

### Debugging
✅ Overfit single batch first (sanity check)  
✅ Monitor gradient norms (should be 0.1-10)  
✅ Check for NaN/Inf in losses  
✅ Profile data loading throughput  
✅ Visualize predictions on validation set  

## Next Steps

Now you can train models effectively! Learn:
- [Save and load checkpoints](./checkpointing) for long training runs
- [Export models](../research/model-export) for deployment
- [Scale to multiple GPUs](../scale/distributed-training)

## Reference Code

Complete working examples:
- [`05_vision_training_mnist.py`](https://github.com/mlnomadpy/flaxdocs/tree/master/examples/05_vision_training_mnist.py) - CNN training
- [`06_language_model_training.py`](https://github.com/mlnomadpy/flaxdocs/tree/master/examples/06_language_model_training.py) - Transformer training
