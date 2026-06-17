---
sidebar_position: 2
title: Training Best Practices with Flax
description: Learn proven strategies for training neural networks with Flax including learning rate scheduling, gradient clipping, and regularization techniques.
keywords: [Flax training, learning rate scheduling, gradient clipping, regularization, neural network optimization, training tips]
---

# Training Best Practices

Learn proven strategies and techniques for training neural networks effectively with Flax.

:::note Prerequisites
This guide builds on [Simple Training Loop](/basics/workflows/simple-training).
:::

:::tip What you'll learn
- Build learning rate schedules with warmup, cosine, exponential, and step decay
- Clip gradients by global norm or value via `optax.chain`
- Apply weight decay with `adamw`, plus dropout and BatchNorm in NNX modules
- Speed up training with bfloat16 mixed precision
- Add metrics logging and early stopping to a training loop
:::

## Learning Rate Scheduling

A good learning rate schedule can significantly improve training performance.

### Warmup Schedule

```python
import optax

# Create a warmup schedule
warmup_steps = 1000
peak_lr = 1e-3

schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=peak_lr,
    warmup_steps=warmup_steps,
    decay_steps=10000,
    end_value=1e-5
)

optimizer = optax.adam(learning_rate=schedule)
```

### Common Schedules

```python
# Exponential decay
exponential_schedule = optax.exponential_decay(
    init_value=1e-3,
    transition_steps=1000,
    decay_rate=0.96
)

# Step decay
step_schedule = optax.piecewise_constant_schedule(
    init_value=1e-3,
    boundaries_and_scales={
        5000: 0.5,   # Halve at step 5000
        10000: 0.5,  # Halve again at step 10000
    }
)

# Cosine decay
cosine_schedule = optax.cosine_decay_schedule(
    init_value=1e-3,
    decay_steps=10000,
    alpha=1e-5
)
```

## Gradient Clipping

Prevent exploding gradients by clipping them.

```python
import optax

# Clip by global norm
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=1e-3)
)

# Clip by value
optimizer = optax.chain(
    optax.clip(1.0),
    optax.adam(learning_rate=1e-3)
)
```

## Weight Decay and Regularization

### L2 Regularization

```python
# Using AdamW (Adam with weight decay)
optimizer = optax.adamw(
    learning_rate=1e-3,
    weight_decay=1e-4
)
```

### Dropout

```python
from flax import nnx

class ModelWithDropout(nnx.Module):
    def __init__(self, dropout_rate: float = 0.1, *, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(784, 256, rngs=rngs)
        self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)
        self.linear2 = nnx.Linear(256, 10, rngs=rngs)

    def __call__(self, x, train: bool = False):
        x = self.linear1(x)
        x = nnx.relu(x)
        x = self.dropout(x, deterministic=not train)
        x = self.linear2(x)
        return x

# RNGs supply both the parameter init and the dropout mask
model = ModelWithDropout(rngs=nnx.Rngs(params=0, dropout=1))

# During training — dropout is active
logits = model(x, train=True)

# During evaluation — dropout is disabled
logits = model(x, train=False)
```

## Batch Normalization

```python
import optax

class ModelWithBatchNorm(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(784, 256, rngs=rngs)
        self.bn = nnx.BatchNorm(256, rngs=rngs)
        self.linear2 = nnx.Linear(256, 10, rngs=rngs)

    def __call__(self, x, train: bool = False):
        x = self.linear1(x)
        x = self.bn(x, use_running_average=not train)
        x = nnx.relu(x)
        x = self.linear2(x)
        return x

# NNX tracks the running batch statistics as part of the model state
# automatically — no separate `batch_stats` collection and no TrainState
# juggling. Create the model and optimizer and you're done:
model = ModelWithBatchNorm(rngs=nnx.Rngs(0))
optimizer = nnx.Optimizer(model, optax.adamw(1e-3), wrt=nnx.Param)
# Only nnx.Param is optimized; the BatchNorm stats update in place during
# the forward pass when train=True.
```

## Mixed Precision Training

Use mixed precision to speed up training and reduce memory usage.

```python
# Use bfloat16 or float16
@nnx.jit
def train_step_mixed_precision(model, optimizer, batch):
    def loss_fn(model):
        # Cast inputs to bfloat16
        images = batch['image'].astype(jnp.bfloat16)
        logits = model(images)
        # Cast logits back to float32 for loss computation
        logits = logits.astype(jnp.float32)
        return optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=batch['label']
        ).mean()

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss
```

## Data Augmentation

Improve generalization with data augmentation.

```python
import jax.numpy as jnp
import jax

def random_crop(image, rng, crop_size=24):
    """Randomly crop image."""
    height, width = image.shape[:2]
    y = jax.random.randint(rng, (), 0, height - crop_size + 1)
    x = jax.random.randint(rng, (), 0, width - crop_size + 1)
    return jax.lax.dynamic_slice(image, (y, x, 0), (crop_size, crop_size, image.shape[2]))

def random_flip(image, rng):
    """Randomly flip image horizontally."""
    return jax.lax.cond(
        jax.random.uniform(rng) > 0.5,
        lambda x: jnp.fliplr(x),
        lambda x: x,
        image
    )

def augment_batch(batch, rng):
    """Apply augmentation to batch."""
    batch_size = batch['image'].shape[0]
    rngs = jax.random.split(rng, batch_size)
    
    def augment_single(image, rng):
        rng1, rng2 = jax.random.split(rng)
        image = random_crop(image, rng1)
        image = random_flip(image, rng2)
        return image
    
    augmented_images = jax.vmap(augment_single)(batch['image'], rngs)
    return {'image': augmented_images, 'label': batch['label']}
```

## Monitoring Training

Track metrics effectively during training.

```python
import time
from collections import defaultdict

class MetricsLogger:
    def __init__(self):
        self.metrics = defaultdict(list)
        self.start_time = time.time()
    
    def log(self, step, **kwargs):
        for key, value in kwargs.items():
            self.metrics[key].append((step, float(value)))
    
    def get_average(self, key, last_n=100):
        if key not in self.metrics:
            return None
        values = [v for _, v in self.metrics[key][-last_n:]]
        return sum(values) / len(values) if values else None
    
    def print_summary(self, step):
        elapsed = time.time() - self.start_time
        print(f"\nStep {step} ({elapsed:.1f}s):")
        for key in self.metrics:
            avg = self.get_average(key)
            if avg is not None:
                print(f"  {key}: {avg:.4f}")

# Usage
logger = MetricsLogger()

for step in range(num_steps):
    loss = train_step(model, optimizer, batch)
    logger.log(step, loss=loss)
    
    if step % 100 == 0:
        logger.print_summary(step)
```

## Early Stopping

Implement early stopping to prevent overfitting.

```python
class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
    
    def should_stop(self, current_loss):
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

# Usage
early_stopping = EarlyStopping(patience=5)

for epoch in range(max_epochs):
    # Training...
    val_loss = evaluate(model, val_data)
    
    if early_stopping.should_stop(val_loss):
        print(f"Early stopping at epoch {epoch}")
        break
```

## Best Practices Checklist

- ✅ Use learning rate warmup for stable training
- ✅ Apply gradient clipping to prevent exploding gradients
- ✅ Use weight decay or L2 regularization
- ✅ Implement data augmentation for better generalization
- ✅ Monitor training metrics regularly
- ✅ Use mixed precision for faster training
- ✅ Implement early stopping to prevent overfitting
- ✅ Use appropriate batch sizes for your hardware
- ✅ Save checkpoints regularly
- ✅ Validate hyperparameters with smaller experiments first

## Next steps

- [Model Checkpointing](/basics/checkpointing) - Save and restore models
- [Observability](/basics/workflows/observability) - Track experiments with W&B
- [Training at Scale](/scale) - Scale training across multiple devices
