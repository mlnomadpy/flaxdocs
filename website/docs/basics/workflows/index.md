---
sidebar_position: 0
---

# Training Workflows

Learn the practical skills to train neural networks - from simple training loops to data loading and optimization.

## What You'll Learn

This section covers the essential workflows for training models:

**[Simple Training Loop](./simple-training.md)** - Start here!  
Write your first complete training loop from scratch. Learn gradient computation, parameter updates, and validation.

**[Data Loading](./data-loading-simple.md)**  
Load and preprocess data efficiently. Learn batching, shuffling, and building data pipelines with TensorFlow Datasets.

**[Streaming Large Datasets](./streaming-data.md)**  
Train on datasets larger than memory. Learn to stream data on-demand from HuggingFace and handle massive datasets.

**[Experiment Tracking](./observability.md)**  
Track experiments with Weights & Biases. Monitor training, compare runs, and ensure reproducibility.

**[Model Export](./model-export.md)**  
Export trained models to SafeTensors, ONNX, and HuggingFace Hub for deployment and sharing.

## Why This Section?

You can define perfect models, but if you can't train them properly, they're useless. This section teaches you:

- How to compute gradients and update parameters
- How to load data without bottlenecking your GPU
- How to validate and avoid overfitting
- How to make training fast with JIT compilation

## Prerequisites

Before starting:
- [Your First Model](../fundamentals/your-first-model.md) - You need a model to train!
- Basic Python knowledge

## The Training Workflow

Every training project follows this pattern:

```
1. Load data → 2. Define model → 3. Train → 4. Evaluate → 5. Save
```

This section focuses on steps 1, 3, and 4.

## Quick Example

Here's the complete workflow you'll master:

```python
from flax import nnx
import optax

# 1. Load data
train_loader = create_data_pipeline(split='train', batch_size=128)
val_loader = create_data_pipeline(split='test', batch_size=128)

# 2. Create model and optimizer
model = MyModel(rngs=nnx.Rngs(params=0))
optimizer = nnx.Optimizer(model, optax.adam(3e-4))

# 3. Training loop
@nnx.jit
def train_step(model, optimizer, batch):
    def loss_fn(model):
        images, labels = batch
        logits = model(images)
        return optax.softmax_cross_entropy(logits, labels).mean()
    
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(grads)
    return loss

# 4. Train and evaluate
for epoch in range(10):
    for batch in train_loader:
        loss = train_step(model, optimizer, batch)
    
    accuracy = evaluate(model, val_loader)
    print(f"Epoch {epoch}: Accuracy = {accuracy:.2%}")

# 5. Save (see checkpointing guide)
```

Just 20 lines of actual code!

## Common Challenges

### Challenge 1: Slow Training
**Solution**: Use JIT compilation (`@nnx.jit`), larger batch sizes, and efficient data loading

### Challenge 2: GPU Sitting Idle
**Solution**: Proper data pipeline with prefetching and parallel preprocessing

### Challenge 3: Overfitting
**Solution**: Validation sets, early stopping, and regularization

### Challenge 4: NaN Losses
**Solution**: Lower learning rate, gradient clipping, check data normalization

## Training Best Practices

✅ **Always validate** - Catch overfitting early  
✅ **Use JIT** - 10-100x speedup  
✅ **Shuffle data** - Each epoch in random order  
✅ **Normalize inputs** - Scale to [0,1] or [-1,1]  
✅ **Monitor metrics** - Print loss, accuracy each epoch  
✅ **Save checkpoints** - Don't lose trained models  

## What's Next?

After mastering basic training:
- [Checkpointing](../../basics/checkpointing.md) - Save and load trained models
- [Distributed Training](../../scale/distributed-training.md) - Scale to multiple GPUs
- [Experiment Tracking](./observability.md) - Monitor your training

## Complete Examples

- [`examples/05_vision_training_mnist.py`](https://github.com/mlnomadpy/flaxdocs/tree/master/examples/05_vision_training_mnist.py) - Complete CNN training
- [`examples/06_language_model_training.py`](https://github.com/mlnomadpy/flaxdocs/tree/master/examples/06_language_model_training.py) - Transformer training
- [`examples/03_data_loading_tfds.py`](https://github.com/mlnomadpy/flaxdocs/tree/master/examples/03_data_loading_tfds.py) - Data loading patterns
