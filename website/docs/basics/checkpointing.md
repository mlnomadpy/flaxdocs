---
sidebar_position: 4
---

# State Management and Checkpointing

Understand how Flax NNX manages model state and how to save/load checkpoints for long training runs and deployment.

## Understanding State in Flax NNX

### What is "State"?

In Flax NNX, **state** is everything your model needs to function:

1. **Parameters** (`nnx.Param`): Trainable weights (learned during training)
2. **Variables** (`nnx.Variable`): Non-trainable state (batch norm statistics, etc.)
3. **RNG Keys**: Random number generators for dropout and other stochastic operations

Unlike pure functional approaches, NNX keeps state **inside the module**, making it feel like PyTorch but with JAX's benefits.

### Extracting State from Modules

```python
from flax import nnx

# Create a model
model = MLP(
    in_features=784,
    hidden_features=256,
    out_features=10,
    rngs=nnx.Rngs(params=0)
)

# Extract state as a dictionary
state = nnx.state(model)

# state is a nested dict like:
# {
#   'layers.0.weight': Param(...),
#   'layers.0.bias': Param(...),
#   'layers.1.weight': Param(...),
#   ...
# }

# Get just parameter values (for saving)
param_values = nnx.state(model, nnx.Param)

# Get just variables (batch norm stats, etc.)
variable_values = nnx.state(model, nnx.Variable)
```

**Why this matters**: You need to understand state extraction to save/load checkpoints correctly.

### Updating State

```python
# Load new state into model (in-place mutation)
new_state = {...}  # Loaded from checkpoint
nnx.update(model, new_state)

# Model now has the loaded parameters!
```

This two-function API (`nnx.state` + `nnx.update`) is the foundation of checkpointing.

## Why Checkpointing Matters

Modern training runs can take hours, days, or even weeks. Checkpointing protects against:

- **Hardware failures**: GPU crashes, node failures, power outages
- **Preemption**: Spot instances terminated, job time limits
- **Experimentation**: Compare model versions without retraining
- **Deployment**: Export trained models for production inference

**Rule of thumb**: If training takes > 10 minutes, use checkpointing.

## Orbax: Flax's Checkpointing Library

Orbax handles serialization to disk. It provides:

- **Efficient storage**: Compressed checkpoints with fast I/O
- **Versioning**: Keep multiple checkpoints, auto-prune old ones
- **Async saving**: Save in background without blocking training
- **Distributed checkpointing**: Handle sharded models across devices

### Installing Orbax

```bash
pip install orbax-checkpoint
```

### Basic Checkpointing Pattern

The simplest approach - save model parameters only:

```python
import orbax.checkpoint as ocp
from flax import nnx
import jax.numpy as jnp

# Create model
model = MyModel(rngs=nnx.Rngs(params=0))

# Extract state to save
state = nnx.state(model)

# Create checkpointer
checkpointer = ocp.PyTreeCheckpointer()

# Save to disk
checkpoint_dir = '/tmp/my_model_checkpoint'
checkpointer.save(checkpoint_dir, state)

print(f"Saved checkpoint to {checkpoint_dir}")

# Later: Load checkpoint
loaded_state = checkpointer.restore(checkpoint_dir)

# Update model with loaded state
nnx.update(model, loaded_state)

print("Model restored from checkpoint!")
```

**What gets saved**: A directory with binary files containing your model's arrays in an efficient format.

## Checkpointing Best Practices

### Save Complete Training State

Don't just save model parameters! Save everything needed to resume:

```python
def save_training_checkpoint(
    model, 
    optimizer, 
    epoch, 
    best_val_loss,
    checkpoint_dir
):
    """Save complete training state"""
    
    checkpoint = {
        'model': nnx.state(model),
        'optimizer': nnx.state(optimizer),
        'epoch': epoch,
        'best_val_loss': best_val_loss,
        # Add any other training state
    }
    
    checkpointer = ocp.PyTreeCheckpointer()
    checkpointer.save(checkpoint_dir, checkpoint)

def load_training_checkpoint(
    model,
    optimizer,
    checkpoint_dir
):
    """Restore complete training state"""
    
    checkpointer = ocp.PyTreeCheckpointer()
    checkpoint = checkpointer.restore(checkpoint_dir)
    
    # Update model and optimizer
    nnx.update(model, checkpoint['model'])
    nnx.update(optimizer, checkpoint['optimizer'])
    
    # Return training metadata
    return checkpoint['epoch'], checkpoint['best_val_loss']
```

**Why this matters**: Resuming training without optimizer state means restarting momentum, learning rate schedule, etc.

### Using CheckpointManager for Versioning

Manually managing checkpoint directories is error-prone. Use `CheckpointManager`:

```python
import orbax.checkpoint as ocp

# Create manager
options = ocp.CheckpointManagerOptions(
    max_to_keep=3,  # Keep only last 3 checkpoints
    best_fn=lambda x: x['val_loss'],  # Track best checkpoint
    best_mode='min',  # Lower is better
)

manager = ocp.CheckpointManager(
    directory='/tmp/my_model',
    checkpointers=ocp.PyTreeCheckpointer(),
    options=options,
)

# In training loop
for epoch in range(num_epochs):
    # ... training code ...
    
    # Save checkpoint
    checkpoint = {
        'model': nnx.state(model),
        'optimizer': nnx.state(optimizer),
        'epoch': epoch,
        'val_loss': val_loss,
    }
    
    manager.save(
        step=epoch,
        items=checkpoint,
        metrics={'val_loss': val_loss},  # For best checkpoint tracking
    )

# Later: Restore latest or best checkpoint
latest_step = manager.latest_step()
checkpoint = manager.restore(latest_step)

# Or restore best
best_step = manager.best_step()
best_checkpoint = manager.restore(best_step)
```

**Features**:
- **Auto-pruning**: Deletes old checkpoints automatically
- **Best tracking**: Keeps best checkpoint based on metric
- **Atomic writes**: No corrupted checkpoints from crashes
- **Step management**: Easy to find specific training steps

## Checkpoint Strategies

### Strategy 1: Periodic Saving

Save every N epochs:

```python
# In training loop
for epoch in range(num_epochs):
    # Training...
    
    # Save every 5 epochs
    if (epoch + 1) % 5 == 0:
        save_checkpoint(model, optimizer, epoch, checkpoint_dir)
```

**Pros**: Simple, predictable storage usage  
**Cons**: May lose up to 5 epochs of training

### Strategy 2: Best Model Only

Save only when validation performance improves:

```python
best_val_loss = float('inf')

for epoch in range(num_epochs):
    # Training...
    val_loss = evaluate(model, val_loader)
    
    # Save if improved
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_checkpoint(
            model, 
            optimizer, 
            epoch, 
            f'/tmp/best_model'
        )
        print(f"New best model! Val loss: {val_loss:.4f}")
```

**Pros**: Only keep best model, saves storage  
**Cons**: Can't resume from arbitrary point

### Strategy 3: Combined Approach

Best of both worlds:

```python
manager = ocp.CheckpointManager(
    directory='/tmp/training',
    checkpointers=ocp.PyTreeCheckpointer(),
    options=ocp.CheckpointManagerOptions(
        max_to_keep=3,  # Keep last 3 for resuming
        best_fn=lambda x: x['val_loss'],
        best_mode='min',
    )
)

for epoch in range(num_epochs):
    # Training...
    
    checkpoint = {
        'model': nnx.state(model),
        'optimizer': nnx.state(optimizer),
        'epoch': epoch,
        'val_loss': val_loss,
    }
    
    # Save every epoch
    manager.save(
        step=epoch,
        items=checkpoint,
        metrics={'val_loss': val_loss},
    )

# After training, load best model
best_checkpoint = manager.restore(manager.best_step())
nnx.update(model, best_checkpoint['model'])
```

**Pros**: Can resume training AND use best model  
**Cons**: Uses more storage (but bounded by max_to_keep)

## Async Checkpointing

For large models, saving can take seconds or minutes. Don't block training:

```python
# Enable async saving
manager = ocp.CheckpointManager(
    directory='/tmp/training',
    checkpointers=ocp.PyTreeCheckpointer(),
    options=ocp.CheckpointManagerOptions(
        max_to_keep=3,
        save_interval_steps=1,
        save_on_steps=[],
        keep_time_interval=None,
        enable_async_checkpointing=True,  # KEY: Enable async
    )
)

# Saving happens in background thread
manager.save(step=epoch, items=checkpoint)
# Training continues immediately!

# Before exiting, wait for pending saves
manager.wait_until_finished()
```

**When to use**: Models > 1GB, slow storage (network drives), frequent checkpointing

## Common Pitfalls

### 1. Not Saving Optimizer State

```python
# BAD: Only save model
checkpoint = nnx.state(model)
checkpointer.save(checkpoint_dir, checkpoint)

# GOOD: Save model AND optimizer
checkpoint = {
    'model': nnx.state(model),
    'optimizer': nnx.state(optimizer),
}
checkpointer.save(checkpoint_dir, checkpoint)
```

**Why**: Optimizer has momentum, learning rate schedule state, etc. Without it, resumed training will perform poorly.

### 2. Forgetting to Update Model

```python
# BAD: Load but don't update
loaded_state = checkpointer.restore(checkpoint_dir)
# Model still has random initialization!

# GOOD: Update model
loaded_state = checkpointer.restore(checkpoint_dir)
nnx.update(model, loaded_state)  # ← Critical!
```

### 3. Overwriting Checkpoints

```python
# BAD: Always use same path
checkpointer.save('/tmp/checkpoint', state)  # Overwrites previous!

# GOOD: Use CheckpointManager or version manually
manager.save(step=epoch, items=checkpoint)
# Or: checkpointer.save(f'/tmp/checkpoint_epoch_{epoch}', state)
```

### 4. Not Testing Restore

```python
# Always test your checkpoint loading!
# After training:
test_model = MyModel(rngs=nnx.Rngs(params=42))  # Different init
restored = checkpointer.restore(checkpoint_dir)
nnx.update(test_model, restored)

# Verify it works
test_output = test_model(test_input)
print(f"Loaded model output: {test_output}")
```

## Checkpoint File Organization

Good directory structure:

```
/experiments/
  /my_model_run1/
    /checkpoints/
      /0/           # Epoch 0
      /5/           # Epoch 5
      /10/          # Epoch 10
      /best/        # Best model
    /logs/
      /tensorboard/
      training.log
    config.yaml     # Hyperparameters
    README.md       # Experiment notes
```

## Resuming Training

Complete example:

```python
def train_with_checkpointing(
    model,
    train_loader,
    val_loader,
    checkpoint_dir,
    num_epochs=100,
):
    """Training loop with checkpoint resume support"""
    
    # Create optimizer
    optimizer = nnx.Optimizer(model, optax.adamw(1e-3))
    
    # Setup checkpoint manager
    manager = ocp.CheckpointManager(
        directory=checkpoint_dir,
        checkpointers=ocp.PyTreeCheckpointer(),
        options=ocp.CheckpointManagerOptions(max_to_keep=3),
    )
    
    # Try to resume from checkpoint
    start_epoch = 0
    if manager.latest_step() is not None:
        print(f"Resuming from checkpoint at step {manager.latest_step()}")
        checkpoint = manager.restore(manager.latest_step())
        nnx.update(model, checkpoint['model'])
        nnx.update(optimizer, checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
    
    # Training loop
    for epoch in range(start_epoch, num_epochs):
        # Train for one epoch
        for batch in train_loader:
            loss = train_step(model, optimizer, batch)
        
        # Validate
        val_loss = evaluate(model, val_loader)
        print(f"Epoch {epoch}: val_loss = {val_loss:.4f}")
        
        # Save checkpoint
        checkpoint = {
            'model': nnx.state(model),
            'optimizer': nnx.state(optimizer),
            'epoch': epoch,
            'val_loss': val_loss,
        }
        manager.save(step=epoch, items=checkpoint)
    
    # Load best model at end
    best_checkpoint = manager.restore(manager.best_step())
    nnx.update(model, best_checkpoint['model'])
    
    return model
```

## Next Steps

You now understand state management and checkpointing! Learn:
- [Export models for deployment](./workflows/model-export.md)
- [Scale training to multiple GPUs](../scale/distributed-training.md)
- [Track experiments with W&B](./workflows/observability.md)

## Reference Code

Complete examples:
- [`02_save_load_model.py`](https://github.com/mlnomadpy/flaxdocs/tree/master/examples/02_save_load_model.py) - All checkpointing patterns
- [`05_vision_training_mnist.py`](https://github.com/mlnomadpy/flaxdocs/tree/master/examples/05_vision_training_mnist.py) - Training with checkpoints
