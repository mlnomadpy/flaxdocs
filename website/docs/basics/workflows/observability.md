---
sidebar_position: 3
---

# Experiment Tracking and Observability

Learn how to track experiments, monitor training, and debug models using Weights & Biases (W&B). Understand why observability is critical for machine learning research and production.

## Why Observability Matters

Training neural networks is empirical science - you need data to understand what's happening:

### Problems Without Observability

**"My model isn't learning"**
- Is the loss decreasing?
- Are gradients flowing?
- Is the learning rate appropriate?
- → Without metrics, you're flying blind

**"Which hyperparameters work best?"**
- Tried 10 learning rates, which was best?
- Can't remember what you ran yesterday
- → Lost experiments waste time and compute

**"Model works on my laptop, fails on cluster"**
- Different batch sizes? Optimizers? Data ordering?
- Can't reproduce results
- → Unreproducible research is worthless

### What Good Observability Provides

✅ **Real-time monitoring**: See training progress live  
✅ **Experiment comparison**: Compare 100 runs at once  
✅ **Reproducibility**: Track every hyperparameter  
✅ **Debugging**: Diagnose issues with visualizations  
✅ **Collaboration**: Share results with team  
✅ **Publication**: Document experiments for papers  

## Weights & Biases Overview

W&B is the standard for ML experiment tracking. It provides:

- **Automatic logging**: Capture metrics with minimal code
- **Dashboard**: Beautiful visualizations
- **Artifacts**: Version datasets, models, predictions
- **Sweeps**: Automated hyperparameter search
- **Reports**: Collaborative experiment documentation

### Installation and Setup

```bash
pip install wandb
```

Login (one-time):

```python
import wandb

# Get API key from wandb.ai
wandb.login()
```

## Basic Experiment Tracking

### Minimal Working Example

```python
import wandb
from flax import nnx
import optax

# 1. Initialize tracking
wandb.init(
    project="mnist-classification",
    name="baseline-run",
    config={
        "learning_rate": 1e-3,
        "batch_size": 128,
        "epochs": 10,
        "architecture": "CNN",
    }
)

# 2. Create model
model = CNN(rngs=nnx.Rngs(params=0))
optimizer = nnx.Optimizer(model, optax.adam(wandb.config.learning_rate))

# 3. Training loop
for epoch in range(wandb.config.epochs):
    for batch in train_loader:
        loss, metrics = train_step(model, optimizer, batch)
        
        # 4. Log metrics
        wandb.log({
            "train/loss": loss,
            "train/accuracy": metrics['accuracy'],
        })
    
    # Validation
    val_loss, val_acc = evaluate(model, val_loader)
    wandb.log({
        "val/loss": val_loss,
        "val/accuracy": val_acc,
        "epoch": epoch
    })

# 5. Finish tracking
wandb.finish()
```

**Result**: Automatic dashboard with loss/accuracy curves, system metrics, and hyperparameters.

### Understanding the API

**`wandb.init()`**: Starts a new run
- `project`: Group related experiments
- `name`: Human-readable run identifier
- `config`: Hyperparameters to track

**`wandb.log()`**: Log metrics
- Call after each training step
- Metrics grouped by prefix (train/, val/)
- Automatically plots time series

**`wandb.finish()`**: Mark run complete
- Uploads final data
- Releases resources
- Always call at end!

## What to Track

### Essential Metrics

```python
# During training step
wandb.log({
    # Loss values
    "train/loss": loss,
    "train/perplexity": jnp.exp(loss),  # For language models
    
    # Optimization info
    "train/learning_rate": current_lr,
    "train/gradient_norm": grad_norm,
    
    # Performance
    "train/accuracy": accuracy,
    "train/tokens_per_second": throughput,
    
    # Step tracking
    "step": global_step,
})

# After each epoch
wandb.log({
    # Validation metrics
    "val/loss": val_loss,
    "val/accuracy": val_acc,
    
    # Best model tracking
    "val/best_accuracy": best_acc,
    
    # Epoch info
    "epoch": epoch,
    "epoch_time": epoch_duration,
})
```

### Gradient Statistics

Monitor gradient health:

```python
def log_gradient_stats(grads):
    """Log gradient statistics for debugging"""
    
    # Flatten all gradients
    flat_grads = jax.tree_util.tree_leaves(grads)
    
    # Compute statistics
    grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in flat_grads))
    grad_max = max(jnp.max(jnp.abs(g)) for g in flat_grads)
    grad_mean = jnp.mean(jnp.array([jnp.mean(g) for g in flat_grads]))
    
    wandb.log({
        "gradients/norm": grad_norm,
        "gradients/max": grad_max,
        "gradients/mean": grad_mean,
    })

# In training loop
grads = compute_gradients(model, batch)
log_gradient_stats(grads)
optimizer.update(grads)
```

**Why this matters**:
- **Exploding gradients**: norm > 1 → clip or lower LR
- **Vanishing gradients**: norm < 0.01 → network too deep or bad init
- **Dead neurons**: max near zero → change activation or init

### Parameter Statistics

Track parameter evolution:

```python
def log_parameter_stats(model):
    """Log parameter statistics"""
    
    state = nnx.state(model)
    params = jax.tree_util.tree_leaves(state)
    
    # Statistics
    param_norm = jnp.sqrt(sum(jnp.sum(p**2) for p in params))
    param_max = max(jnp.max(jnp.abs(p)) for p in params)
    param_mean = jnp.mean(jnp.array([jnp.mean(p) for p in params]))
    
    wandb.log({
        "parameters/norm": param_norm,
        "parameters/max": param_max,
        "parameters/mean": param_mean,
    })

# Log every N steps
if step % 100 == 0:
    log_parameter_stats(model)
```

## Visualizations

### Custom Plots

```python
# Confusion matrix
import seaborn as sns
import matplotlib.pyplot as plt

def log_confusion_matrix(model, val_loader, class_names):
    """Log confusion matrix"""
    
    # Compute predictions
    all_preds = []
    all_targets = []
    
    for batch in val_loader:
        logits = model(batch['images'])
        preds = jnp.argmax(logits, axis=-1)
        targets = jnp.argmax(batch['labels'], axis=-1)
        
        all_preds.extend(preds)
        all_targets.extend(targets)
    
    # Create confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(all_targets, all_preds)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    
    # Log to W&B
    wandb.log({"confusion_matrix": wandb.Image(fig)})
    plt.close()

# Run at end of training
log_confusion_matrix(model, val_loader, ['cat', 'dog', ...])
```

### Image Logging

```python
# Log example predictions
def log_predictions(model, val_batch):
    """Log model predictions on images"""
    
    images = val_batch['images'][:8]  # First 8 images
    labels = val_batch['labels'][:8]
    
    # Predict
    logits = model(images)
    preds = jnp.argmax(logits, axis=-1)
    probs = jax.nn.softmax(logits, axis=-1)
    
    # Create wandb images with predictions
    wandb_images = []
    for img, true_label, pred_label, prob in zip(images, labels, preds, probs):
        caption = f"True: {true_label}, Pred: {pred_label} ({prob[pred_label]:.2f})"
        wandb_images.append(wandb.Image(img, caption=caption))
    
    wandb.log({"predictions": wandb_images})

# Log every few epochs
if epoch % 5 == 0:
    log_predictions(model, next(iter(val_loader)))
```

### Histograms

```python
# Log weight distributions
def log_weight_histograms(model):
    """Log parameter distributions"""
    
    state = nnx.state(model)
    
    # Log each layer's weights
    for path, param in jax.tree_util.tree_leaves_with_path(state):
        name = '.'.join(str(p.key) for p in path if hasattr(p, 'key'))
        
        if 'weight' in name:
            wandb.log({
                f"weights/{name}": wandb.Histogram(param)
            })

# Log periodically
if step % 1000 == 0:
    log_weight_histograms(model)
```

## Hyperparameter Sweeps

Automatically search hyperparameter space:

### Defining a Sweep

```python
# sweep_config.yaml or in code
sweep_config = {
    'method': 'random',  # or 'grid', 'bayes'
    'metric': {
        'name': 'val/accuracy',
        'goal': 'maximize'
    },
    'parameters': {
        'learning_rate': {
            'distribution': 'log_uniform_values',
            'min': 1e-5,
            'max': 1e-2
        },
        'batch_size': {
            'values': [32, 64, 128, 256]
        },
        'num_layers': {
            'values': [2, 3, 4, 5]
        },
        'hidden_size': {
            'distribution': 'q_uniform',
            'min': 128,
            'max': 512,
            'q': 64  # Step size
        },
        'dropout': {
            'distribution': 'uniform',
            'min': 0.1,
            'max': 0.5
        },
    }
}

# Create sweep
sweep_id = wandb.sweep(sweep_config, project="mnist-classification")
```

### Running Sweep Agents

```python
def train_sweep():
    """Training function for sweep"""
    
    # Initialize with sweep config
    run = wandb.init()
    config = wandb.config
    
    # Create model with sweep hyperparameters
    model = MLP(
        in_features=784,
        hidden_features=config.hidden_size,
        out_features=10,
        num_layers=config.num_layers,
        dropout_rate=config.dropout,
        rngs=nnx.Rngs(params=0)
    )
    
    optimizer = nnx.Optimizer(
        model,
        optax.adam(learning_rate=config.learning_rate)
    )
    
    # Train
    for epoch in range(config.epochs):
        for batch in get_dataloader(batch_size=config.batch_size):
            loss, metrics = train_step(model, optimizer, batch)
            wandb.log({"train/loss": loss, "train/accuracy": metrics['accuracy']})
        
        # Validation
        val_loss, val_acc = evaluate(model, val_loader)
        wandb.log({"val/loss": val_loss, "val/accuracy": val_acc})
    
    wandb.finish()

# Run sweep agent
wandb.agent(sweep_id, function=train_sweep, count=50)  # Run 50 trials
```

### Sweep Strategies

**Random search**:
- Samples hyperparameters randomly
- Good for exploring large spaces
- Easy to parallelize

**Grid search**:
- Tries all combinations
- Exhaustive but expensive
- Best for small spaces

**Bayesian optimization**:
- Uses previous results to guide search
- Most sample-efficient
- Requires sequential runs

## Model Artifacts

Version models and datasets:

### Saving Models as Artifacts

```python
# After training
artifact = wandb.Artifact(
    name='mnist-cnn',
    type='model',
    description='CNN trained on MNIST',
    metadata={
        'accuracy': best_acc,
        'architecture': 'CNN',
        'params': count_parameters(model)
    }
)

# Add model files
artifact.add_file('model.safetensors')
artifact.add_file('config.json')

# Log artifact
wandb.log_artifact(artifact)
```

### Using Artifacts

```python
# Load artifact in new run
run = wandb.init(project="mnist-classification")

artifact = run.use_artifact('mnist-cnn:latest')  # Or specific version
artifact_dir = artifact.download()

# Load model
model = load_model_from_checkpoint(f"{artifact_dir}/model.safetensors")
```

## Best Practices

### Structuring Experiments

```python
# Good: Organized logging
wandb.log({
    # Training metrics
    "train/loss": loss,
    "train/accuracy": acc,
    
    # Validation metrics
    "val/loss": val_loss,
    "val/accuracy": val_acc,
    
    # Optimization
    "opt/learning_rate": lr,
    "opt/gradient_norm": grad_norm,
    
    # System
    "system/gpu_memory": gpu_mem,
    "system/throughput": samples_per_sec,
})

# Bad: Flat namespace
wandb.log({
    "loss": loss,
    "loss2": val_loss,  # Confusing!
    "acc": acc,
    "valacc": val_acc,  # Inconsistent naming
})
```

### Reproducibility Checklist

Track everything needed to reproduce:

```python
config = {
    # Model
    "architecture": "ResNet-18",
    "num_layers": 18,
    "hidden_size": 512,
    
    # Optimization
    "optimizer": "AdamW",
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "lr_schedule": "cosine",
    "warmup_steps": 1000,
    
    # Data
    "dataset": "ImageNet",
    "batch_size": 256,
    "augmentation": "standard",
    
    # Training
    "epochs": 100,
    "seed": 42,
    
    # System
    "jax_version": jax.__version__,
    "flax_version": nnx.__version__,
    "device": jax.devices()[0],
}

wandb.init(project="my-project", config=config)
```

### Offline Mode

Train without internet:

```python
# Set offline mode
import os
os.environ['WANDB_MODE'] = 'offline'

# Train normally
wandb.init(project="my-project")
# ... training ...
wandb.finish()

# Later: Sync offline runs
# wandb sync /path/to/offline/run
```

## Common Patterns

### Early Stopping

```python
best_val_loss = float('inf')
patience = 5
patience_counter = 0

for epoch in range(max_epochs):
    # Training...
    
    # Validation
    val_loss = evaluate(model, val_loader)
    wandb.log({"val/loss": val_loss})
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        
        # Save best model
        save_checkpoint(model, "best_model.safetensors")
        wandb.log({"val/best_loss": best_val_loss})
    else:
        patience_counter += 1
    
    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch}")
        break

# Log final best metric
wandb.run.summary["best_val_loss"] = best_val_loss
```

### Multi-Run Comparison

```python
# Run multiple seeds
for seed in [42, 123, 456, 789, 999]:
    run = wandb.init(
        project="mnist-comparison",
        name=f"seed-{seed}",
        config={"seed": seed}
    )
    
    # Set seed
    rngs = nnx.Rngs(params=seed)
    
    # Train
    model = train_model(rngs=rngs)
    
    # Log results
    val_acc = evaluate(model, val_loader)
    wandb.log({"final_val_accuracy": val_acc})
    
    wandb.finish()

# In W&B UI: Compare all runs to see variance
```

## Debugging with W&B

### Detecting Issues

**Symptoms**:
- Loss explodes → Check gradient norms
- Loss plateaus → Check learning rate schedule
- Accuracy stuck → Visualize predictions

**Debug dashboard**:
```python
# Comprehensive debugging logs
wandb.log({
    "debug/loss": loss,
    "debug/loss_is_nan": jnp.isnan(loss),
    "debug/loss_is_inf": jnp.isinf(loss),
    
    "debug/grad_norm": grad_norm,
    "debug/grad_norm_too_large": grad_norm > 10,
    
    "debug/param_norm": param_norm,
    "debug/param_max": param_max,
    
    "debug/learning_rate": current_lr,
    "debug/batch_mean": batch['images'].mean(),
    "debug/batch_std": batch['images'].std(),
})
```

## Next Steps

You now know how to track experiments professionally! Learn more:
- [Scale training to distributed systems](../../scale/distributed-training.md)
- [Build production pipelines](./model-export.md)

## Reference Code

Complete example:
- [`10_wandb_observability.py`](https://github.com/mlnomadpy/flaxdocs/tree/master/examples/10_wandb_observability.py) - Full W&B integration
