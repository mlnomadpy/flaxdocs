---
sidebar_position: 4
title: Streaming Data Training Pipeline
description: Build an end-to-end streaming training pipeline in Flax NNX - stream, shuffle, tokenize, and batch large datasets, with caching, retries, and best practices.
keywords: [streaming training, data pipeline, Flax NNX, HuggingFace datasets, shuffle buffer, tokenization, caching, large datasets, training loop]
---

# Streaming Data Training Pipeline

Build an end-to-end streaming training pipeline in Flax NNX: train on datasets
larger than memory while handling caching, network failures, and performance.

:::note Prerequisites

This page assumes you already understand *what* streaming is and how to read
data larger than memory - the HuggingFace `streaming=True` mechanics, shuffle
buffers, on-the-fly tokenization, and basic batching. If you need a refresher,
read [Streaming Datasets Larger Than Memory](/basics/data-streaming) first.

:::

:::tip What you'll learn
- Assemble streaming, shuffle, tokenize, and batch into one Flax NNX training loop
- Cache tokenized data to disk with `cache_file_name` to skip re-tokenizing
- Handle slow first batches, network timeouts, and poor shuffle quality
- Interleave dataset shards for better randomization
- Decide when *not* to stream (fits in RAM, unreliable network, multiple epochs)
:::

This guide picks up where that primer leaves off and assembles those pieces into
a production training loop. Throughout, we reuse the `tokenize_function` and the
`create_batches` helper introduced in the primer.

## Complete Training Example

```python
from datasets import load_dataset
from transformers import AutoTokenizer
from flax import nnx
import optax

# 1. Setup streaming dataset
dataset = load_dataset(
    'HuggingFaceFW/fineweb-edu',
    name='sample-10BT', 
    split='train',
    streaming=True
)

# 2. Shuffle and tokenize
tokenizer = AutoTokenizer.from_pretrained('gpt2')

dataset = dataset.shuffle(seed=42, buffer_size=10_000)
dataset = dataset.map(
    lambda x: tokenizer(x['text'], max_length=512, truncation=True, padding='max_length'),
    batched=True
)

# 3. Create model
model = GPTModel(vocab_size=50257, rngs=nnx.Rngs(params=0, dropout=1))
optimizer = nnx.Optimizer(model, optax.adamw(3e-4), wrt=nnx.Param)

# 4. Train on stream
num_steps = 10_000
for step, batch in enumerate(create_batches(dataset, batch_size=32)):
    if step >= num_steps:
        break
    
    # Training step
    def loss_fn(model):
        logits = model(batch['input_ids'], train=True)
        # Shift for next-token prediction
        logits = logits[:, :-1]
        targets = batch['input_ids'][:, 1:]
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1)
        ).mean()
        return loss
    
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    
    if step % 100 == 0:
        print(f"Step {step}: Loss = {loss:.4f}")
```

## Caching for Performance

Cache processed data to avoid re-tokenizing:

```python
# Cache tokenized data to disk
dataset = dataset.map(
    tokenize_function,
    batched=True,
    cache_file_name='./cache/tokenized_data'  # Cache location
)

# First run: Tokenizes and saves to cache
# Subsequent runs: Loads from cache (much faster!)
```

## Common Issues

### Issue 1: Slow First Batch

```python
# Problem: First batch takes forever (downloading)

# Solution: Prefetch in background
from datasets import load_dataset

dataset = load_dataset(..., streaming=True)
dataset = dataset.prefetch(10)  # Prefetch 10 batches ahead
```

### Issue 2: Network Timeout

```python
# Problem: Network drops, training crashes

# Solution: Retry logic
from datasets import load_dataset

dataset = load_dataset(
    ..., 
    streaming=True,
    num_proc=1,  # Single process (more stable)
)

# Wrap in try/except
for batch in dataset:
    try:
        loss = train_step(model, optimizer, batch)
    except Exception as e:
        print(f"Skipping batch due to error: {e}")
        continue
```

### Issue 3: Poor Shuffle Quality

```python
# Problem: Buffer too small, data not random

# Solution: Larger buffer + interleave datasets
dataset = dataset.shuffle(buffer_size=50_000)  # Larger buffer

# Or interleave multiple shards
from datasets import interleave_datasets
dataset = interleave_datasets([
    load_dataset(..., split='train[0:25%]', streaming=True),
    load_dataset(..., split='train[25%:50%]', streaming=True),
    load_dataset(..., split='train[50%:75%]', streaming=True),
    load_dataset(..., split='train[75%:100%]', streaming=True),
])
```

## Best Practices

✅ **Use large shuffle buffers** (10k+ for text, 1k+ for images)  
✅ **Cache preprocessed data** when possible  
✅ **Prefetch batches** to hide network latency  
✅ **Monitor network usage** to avoid bottlenecks  
✅ **Have fallback plan** for network failures  

## When NOT to Stream

Don't stream if:
- ❌ Dataset fits in RAM (< 10GB)
- ❌ Network is unreliable
- ❌ Need deterministic ordering
- ❌ Iterating multiple times per epoch

In these cases, download once and train from disk.

## Next steps

- [Track experiments with W&B](/basics/workflows/observability) - Monitor streaming training
- [Export trained models](/basics/workflows/model-export) - Deploy your models
- [ResNet Architecture](/basics/vision/resnet-architecture) - ResNet with streaming images

## Complete Examples

**Organized modular examples:**
- [`examples/integrations/resnet_streaming.py`](https://github.com/mlnomadpy/flaxdocs/tree/master/examples/integrations/resnet_streaming.py) - ResNet with streaming images from HuggingFace
- [`examples/advanced/bert_fineweb.py`](https://github.com/mlnomadpy/flaxdocs/tree/master/examples/advanced/bert_fineweb.py) - BERT training with streaming text from FineWeb
- [`examples/advanced/gpt_training.py`](https://github.com/mlnomadpy/flaxdocs/tree/master/examples/advanced/gpt_training.py) - GPT training with streaming text from FineWeb
