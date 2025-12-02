---
sidebar_position: 3
---

# Model Checkpointing

Learn how to save and restore your Flax models effectively.

## Why Checkpointing?

Checkpointing is crucial for:

- **Resuming Training**: Continue from where you left off if training is interrupted
- **Model Evaluation**: Save the best model during training
- **Deployment**: Export trained models for inference
- **Experimentation**: Compare different model versions

## Basic Checkpointing with Orbax

Flax uses Orbax for checkpointing. Install it first:

```bash
pip install orbax-checkpoint
```

### Simple Save and Restore

```python
from flax.training import orbax_utils
import orbax.checkpoint as ocp

# Create checkpoint manager
checkpoint_dir = '/tmp/flax_checkpoints'
checkpoint_manager = ocp.CheckpointManager(
    checkpoint_dir,
    checkpointers=ocp.PyTreeCheckpointer(),
    options=ocp.CheckpointManagerOptions(max_to_keep=3)
)

# Save checkpoint
ckpt = {'params': state.params, 'step': step}
save_args = orbax_utils.save_args_from_target(ckpt)
checkpoint_manager.save(step, ckpt, save_kwargs={'save_args': save_args})

# Restore checkpoint
restored = checkpoint_manager.restore(step)
state = state.replace(params=restored['params'])
```

## Complete Training State

Save the entire training state including optimizer state:

```python
import orbax.checkpoint as ocp
from flax.training import train_state

# Save complete state
def save_checkpoint(state, step, checkpoint_dir):
    checkpoint_manager = ocp.CheckpointManager(
        checkpoint_dir,
        checkpointers=ocp.PyTreeCheckpointer(),
        options=ocp.CheckpointManagerOptions(
            max_to_keep=3,
            save_interval_steps=1000,
        )
    )
    
    ckpt = {
        'params': state.params,
        'opt_state': state.opt_state,
        'step': state.step,
    }
    
    save_args = orbax_utils.save_args_from_target(ckpt)
    checkpoint_manager.save(step, ckpt, save_kwargs={'save_args': save_args})
    print(f'Saved checkpoint at step {step}')

# Restore complete state
def restore_checkpoint(state, checkpoint_dir):
    checkpoint_manager = ocp.CheckpointManager(
        checkpoint_dir,
        checkpointers=ocp.PyTreeCheckpointer(),
    )
    
    latest_step = checkpoint_manager.latest_step()
    if latest_step is None:
        print('No checkpoint found')
        return state, 0
    
    restored = checkpoint_manager.restore(latest_step)
    state = state.replace(
        params=restored['params'],
        opt_state=restored['opt_state'],
        step=restored['step']
    )
    print(f'Restored checkpoint from step {latest_step}')
    return state, latest_step
```

## Best Model Checkpointing

Save the model with the best validation performance:

```python
class BestCheckpointSaver:
    def __init__(self, checkpoint_dir, metric='loss', mode='min'):
        self.checkpoint_dir = checkpoint_dir
        self.metric = metric
        self.mode = mode  # 'min' or 'max'
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.checkpoint_manager = ocp.CheckpointManager(
            checkpoint_dir,
            checkpointers=ocp.PyTreeCheckpointer(),
            options=ocp.CheckpointManagerOptions(max_to_keep=1)
        )
    
    def should_save(self, current_value):
        if self.mode == 'min':
            return current_value < self.best_value
        else:
            return current_value > self.best_value
    
    def save_if_best(self, state, step, metric_value):
        if self.should_save(metric_value):
            self.best_value = metric_value
            ckpt = {
                'params': state.params,
                'step': step,
                'best_metric': metric_value,
            }
            save_args = orbax_utils.save_args_from_target(ckpt)
            self.checkpoint_manager.save(
                step, 
                ckpt, 
                save_kwargs={'save_args': save_args}
            )
            print(f'Saved best checkpoint: {self.metric}={metric_value:.4f}')
            return True
        return False

# Usage
best_saver = BestCheckpointSaver(
    checkpoint_dir='/tmp/best_model',
    metric='accuracy',
    mode='max'
)

for epoch in range(num_epochs):
    # Training...
    val_accuracy = evaluate(state, val_data)
    best_saver.save_if_best(state, state.step, val_accuracy)
```

## Periodic Checkpointing

Save checkpoints at regular intervals:

```python
def train_with_checkpointing(
    state,
    train_data,
    num_steps,
    checkpoint_dir,
    save_every=1000
):
    checkpoint_manager = ocp.CheckpointManager(
        checkpoint_dir,
        checkpointers=ocp.PyTreeCheckpointer(),
        options=ocp.CheckpointManagerOptions(
            max_to_keep=3,
            save_interval_steps=save_every,
        )
    )
    
    # Try to restore from existing checkpoint
    latest_step = checkpoint_manager.latest_step()
    start_step = 0
    
    if latest_step is not None:
        restored = checkpoint_manager.restore(latest_step)
        state = state.replace(
            params=restored['params'],
            opt_state=restored['opt_state'],
            step=restored['step']
        )
        start_step = int(restored['step'])
        print(f'Resumed from step {start_step}')
    
    for step in range(start_step, num_steps):
        # Training step
        state, loss = train_step(state, next(train_data))
        
        # Periodic checkpoint
        if (step + 1) % save_every == 0:
            ckpt = {
                'params': state.params,
                'opt_state': state.opt_state,
                'step': state.step,
            }
            save_args = orbax_utils.save_args_from_target(ckpt)
            checkpoint_manager.save(
                step, 
                ckpt, 
                save_kwargs={'save_args': save_args}
            )
            print(f'Checkpoint saved at step {step + 1}')
    
    return state
```

## Asynchronous Checkpointing

For large models, use async checkpointing to avoid blocking training:

```python
from orbax.checkpoint import AsyncCheckpointer

async_checkpointer = AsyncCheckpointer(
    ocp.PyTreeCheckpointHandler()
)

checkpoint_manager = ocp.CheckpointManager(
    checkpoint_dir,
    checkpointers=async_checkpointer,
    options=ocp.CheckpointManagerOptions(
        max_to_keep=3,
        save_interval_steps=1000,
    )
)

# Save asynchronously
ckpt = {'params': state.params}
save_args = orbax_utils.save_args_from_target(ckpt)
checkpoint_manager.save(
    step, 
    ckpt, 
    save_kwargs={'save_args': save_args}
)
# Training continues immediately

# Wait for all pending saves before exiting
checkpoint_manager.wait_until_finished()
```

## Checkpoint Structure

Understanding the checkpoint directory structure:

```
checkpoint_dir/
├── 1000/
│   └── checkpoint
├── 2000/
│   └── checkpoint
├── 3000/
│   └── checkpoint
└── checkpoint_manager.json
```

## Legacy Format (pickle-based)

For compatibility with older code:

```python
import pickle
import os

def save_checkpoint_legacy(state, checkpoint_dir, step):
    """Save checkpoint using pickle."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_{step}.pkl')
    
    with open(checkpoint_path, 'wb') as f:
        pickle.dump({
            'params': state.params,
            'opt_state': state.opt_state,
            'step': step,
        }, f)
    
    print(f'Saved checkpoint to {checkpoint_path}')

def restore_checkpoint_legacy(checkpoint_path):
    """Restore checkpoint from pickle."""
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)
    return checkpoint
```

## Best Practices

### 1. Regular Checkpointing

```python
# Save every N steps or epochs
CHECKPOINT_EVERY_STEPS = 1000

if step % CHECKPOINT_EVERY_STEPS == 0:
    save_checkpoint(state, step, checkpoint_dir)
```

### 2. Keep Multiple Checkpoints

```python
# Keep the last 3 checkpoints
options = ocp.CheckpointManagerOptions(max_to_keep=3)
```

### 3. Save Best Model Separately

```python
# Separate directory for best model
best_checkpoint_dir = os.path.join(checkpoint_dir, 'best')
if is_best_model:
    save_checkpoint(state, step, best_checkpoint_dir)
```

### 4. Include Metadata

```python
ckpt = {
    'params': state.params,
    'opt_state': state.opt_state,
    'step': state.step,
    'config': model_config,  # Save model configuration
    'metrics': {'loss': loss, 'accuracy': accuracy},
}
```

## Troubleshooting

### Out of Disk Space

- Reduce `max_to_keep` value
- Use async checkpointing
- Compress checkpoints

### Slow Checkpointing

- Use async checkpointing
- Increase checkpoint interval
- Use faster storage

### Cannot Restore Checkpoint

- Verify checkpoint directory path
- Check if checkpoint files are corrupted
- Ensure model architecture matches

## Next Steps

- [Distributed Training](../scale/distributed-training) - Checkpoint in distributed settings
