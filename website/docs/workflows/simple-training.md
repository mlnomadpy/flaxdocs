---
sidebar_position: 1
---

# Your First Training Loop

Learn to write a complete training loop from scratch - no magic, just clear, understandable code.

## The Training Loop Structure

Every training loop has the same five steps:

```python
# 1. Create model and optimizer
# 2. Loop over epochs
#     3. Loop over batches
#         4. Compute loss and gradients
#         5. Update parameters
```

Let's build this step by step.

## Step 1: Setup

```python
import jax
import jax.numpy as jnp
from flax import nnx
import optax

# Create a simple model
class SimpleMLP(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        self.layer1 = nnx.Linear(784, 256, rngs=rngs)
        self.layer2 = nnx.Linear(256, 10, rngs=rngs)
    
    def __call__(self, x):
        x = self.layer1(x)
        x = nnx.relu(x)
        return self.layer2(x)

# Initialize model
model = SimpleMLP(rngs=nnx.Rngs(params=0))

# Create optimizer
optimizer = nnx.Optimizer(model, optax.adam(learning_rate=1e-3))
```

## Step 2: Define Loss Function

```python
def compute_loss(model, batch):
    """Compute cross-entropy loss"""
    images, labels = batch
    
    # Forward pass
    logits = model(images)
    
    # Cross-entropy loss
    loss = optax.softmax_cross_entropy(logits, labels).mean()
    
    return loss
```

## Step 3: Training Step

Here's the magic - compute gradients and update:

```python
def train_step(model, optimizer, batch):
    """Single training step"""
    
    # Compute loss and gradients
    loss, grads = nnx.value_and_grad(compute_loss)(model, batch)
    
    # Update parameters
    optimizer.update(grads)
    
    return loss
```

**What `nnx.value_and_grad` does**:
- Computes the loss (value)
- Computes gradients of loss w.r.t. all parameters (grad)
- Returns both in one efficient pass

## Step 4: Complete Training Loop

```python
def train(model, optimizer, train_loader, num_epochs=10):
    """Full training loop"""
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        # Loop over batches
        for batch in train_loader:
            loss = train_step(model, optimizer, batch)
            epoch_loss += loss
            num_batches += 1
        
        # Print progress
        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch + 1}/{num_epochs}: Loss = {avg_loss:.4f}")

# Run training
train(model, optimizer, train_loader, num_epochs=10)
```

## Adding Validation

Always validate to catch overfitting:

```python
def evaluate(model, val_loader):
    """Evaluate on validation set"""
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch in val_loader:
        images, labels = batch
        
        # Forward pass only (no gradients)
        logits = model(images)
        
        # Compute loss
        loss = optax.softmax_cross_entropy(logits, labels).mean()
        total_loss += loss
        
        # Compute accuracy
        preds = jnp.argmax(logits, axis=-1)
        targets = jnp.argmax(labels, axis=-1)
        correct += jnp.sum(preds == targets)
        total += len(images)
    
    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total
    
    return avg_loss, accuracy

# Use in training loop
def train_with_validation(model, optimizer, train_loader, val_loader, num_epochs=10):
    for epoch in range(num_epochs):
        # Training
        epoch_loss = 0.0
        for batch in train_loader:
            loss = train_step(model, optimizer, batch)
            epoch_loss += loss
        
        # Validation
        val_loss, val_acc = evaluate(model, val_loader)
        
        print(f"Epoch {epoch + 1}: "
              f"Train Loss = {epoch_loss/len(train_loader):.4f}, "
              f"Val Loss = {val_loss:.4f}, "
              f"Val Acc = {val_acc:.2%}")
```

## Making It Fast with JIT

JAX can compile your training step for massive speedup:

```python
# Compile the training step
@nnx.jit
def train_step_fast(model, optimizer, batch):
    """JIT-compiled training step"""
    loss, grads = nnx.value_and_grad(compute_loss)(model, batch)
    optimizer.update(grads)
    return loss

# Use the same way as before - but much faster!
for batch in train_loader:
    loss = train_step_fast(model, optimizer, batch)
```

**First call**: Slow (compiling)  
**Subsequent calls**: Very fast (using compiled code)

## Complete Example with Everything

```python
import jax
import jax.numpy as jnp
from flax import nnx
import optax

# Model
class MNISTModel(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(1, 32, (3, 3), rngs=rngs)
        self.conv2 = nnx.Conv(32, 64, (3, 3), rngs=rngs)
        self.dense1 = nnx.Linear(64 * 5 * 5, 128, rngs=rngs)
        self.dense2 = nnx.Linear(128, 10, rngs=rngs)
    
    def __call__(self, x):
        x = self.conv1(x)
        x = nnx.relu(x)
        x = nnx.max_pool(x, (2, 2), (2, 2))
        
        x = self.conv2(x)
        x = nnx.relu(x)
        x = nnx.max_pool(x, (2, 2), (2, 2))
        
        x = x.reshape(x.shape[0], -1)
        x = self.dense1(x)
        x = nnx.relu(x)
        return self.dense2(x)

# Loss function
def compute_loss(model, batch):
    images, labels = batch
    logits = model(images)
    return optax.softmax_cross_entropy(logits, labels).mean()

# Training step (JIT compiled)
@nnx.jit
def train_step(model, optimizer, batch):
    loss, grads = nnx.value_and_grad(compute_loss)(model, batch)
    optimizer.update(grads)
    return loss

# Evaluation
def evaluate(model, loader):
    correct = 0
    total = 0
    for batch in loader:
        images, labels = batch
        logits = model(images)
        preds = jnp.argmax(logits, axis=-1)
        targets = jnp.argmax(labels, axis=-1)
        correct += jnp.sum(preds == targets)
        total += len(images)
    return correct / total

# Main training loop
def main():
    # Setup
    model = MNISTModel(rngs=nnx.Rngs(params=0))
    optimizer = nnx.Optimizer(model, optax.adam(3e-4))
    
    # Load data (simplified - see data loading guide)
    train_loader = load_mnist_train()
    val_loader = load_mnist_val()
    
    # Train
    for epoch in range(10):
        # Training
        for batch in train_loader:
            loss = train_step(model, optimizer, batch)
        
        # Validation
        accuracy = evaluate(model, val_loader)
        print(f"Epoch {epoch + 1}: Val Accuracy = {accuracy:.2%}")

if __name__ == '__main__':
    main()
```

## Common Mistakes

### Mistake 1: Forgetting to Update

❌ **Wrong**: Computing gradients but not updating
```python
loss, grads = nnx.value_and_grad(compute_loss)(model, batch)
# Forgot to call optimizer.update(grads)!
```

✅ **Right**: Always update after computing gradients
```python
loss, grads = nnx.value_and_grad(compute_loss)(model, batch)
optimizer.update(grads)  # This updates model parameters
```

### Mistake 2: Computing Gradients During Evaluation

❌ **Wrong**: Wasting computation
```python
def evaluate(model, loader):
    for batch in loader:
        loss, grads = nnx.value_and_grad(compute_loss)(model, batch)
        # Don't need gradients during evaluation!
```

✅ **Right**: Just forward pass
```python
def evaluate(model, loader):
    for batch in loader:
        images, labels = batch
        logits = model(images)  # No gradients needed
```

### Mistake 3: Not Shuffling Data

❌ **Wrong**: Training on same order every epoch
```python
for epoch in range(10):
    for batch in train_loader:  # Same order each time
        train_step(model, optimizer, batch)
```

✅ **Right**: Shuffle each epoch
```python
for epoch in range(10):
    shuffled_loader = shuffle_data(train_loader, epoch)
    for batch in shuffled_loader:
        train_step(model, optimizer, batch)
```

## Best Practices

1. **Always validate**: Catch overfitting early
2. **Use JIT compilation**: 10-100x speedup
3. **Print progress**: Know when training stalls
4. **Save checkpoints**: Don't lose trained models (see checkpointing guide)
5. **Monitor learning curves**: Plot loss/accuracy over time

## Hyperparameters to Tune

- **Learning rate**: Most important! Start with 1e-3 or 3e-4
- **Batch size**: 32-256 for most tasks
- **Number of epochs**: Until validation stops improving
- **Optimizer**: Adam is usually a good default

## Next Steps

- [Optimization Strategies](./optimization.md) - Advanced optimizers and schedules
- [Data Loading](./data-loading-simple.md) - Efficient data pipelines
- [Checkpointing](../fundamentals/saving-loading.md) - Save your trained models

## Complete Example

See the full runnable code in [`examples/05_vision_training_mnist.py`](https://github.com/mlnomadpy/flaxdocs/tree/master/examples/05_vision_training_mnist.py).
