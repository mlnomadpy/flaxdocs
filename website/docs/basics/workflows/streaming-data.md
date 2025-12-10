---
sidebar_position: 4
---

# Streaming Large Datasets

Learn how to train on datasets larger than memory by streaming data on-demand during training.

## Why Streaming?

Modern datasets are huge and won't fit in RAM:
- **FineWeb**: 15 trillion tokens (>10TB)
- **ImageNet-21k**: 14M images (~1TB)  
- **Common Crawl**: Petabytes of text

**Solution**: Stream data during training - fetch batches on-demand instead of loading everything.

## Streaming vs Downloading

**Traditional approach** (download all):
```python
dataset = load_dataset('imagenet-1k')  # Downloads 150GB!
for batch in dataset:
    train_step(batch)
```

**Streaming approach**:
```python
dataset = load_dataset('imagenet-1k', streaming=True)  # No download
for batch in dataset:  # Fetches on-demand
    train_step(batch)
```

**Benefits**:
✅ Start training immediately  
✅ No disk space needed  
✅ Easy to switch datasets  

**Tradeoffs**:
⚠️ Need stable network connection  
⚠️ Slight latency per batch  
⚠️ More complex caching  

## Basic Streaming with HuggingFace

```python
from datasets import load_dataset

# Load in streaming mode
dataset = load_dataset(
    'HuggingFaceFW/fineweb-edu',
    name='sample-10BT',
    split='train',
    streaming=True  # KEY: Don't download
)

# Dataset is iterable, not indexable
# Can't do: dataset[0]  ❌
# Must do: next(iter(dataset))  ✅

# Shuffle with buffer
dataset = dataset.shuffle(
    seed=42,
    buffer_size=10_000  # Shuffle window
)

# Process and iterate
for i, example in enumerate(dataset):
    text = example['text']
    # ... tokenize and train ...
    
    if i >= 10000:  # Train for 10k examples
        break
```

## Understanding Shuffle Buffers

Streaming can't shuffle all data at once (not in memory). Instead, uses a **buffer**:

```python
# In-memory: Perfect shuffle
dataset = dataset.shuffle()  # Shuffles all N examples

# Streaming: Buffer shuffle  
dataset = dataset.shuffle(buffer_size=10_000)
# How it works:
# 1. Load 10k examples into buffer
# 2. Shuffle buffer
# 3. Yield one example
# 4. Load next example, shuffle 10k again
# 5. Repeat
```

**Choosing buffer size**:
- **Larger** = better randomization, more memory
- **Smaller** = less memory, worse randomization  
- **Rule of thumb**: 10-100x batch size

## Tokenization for Text Streaming

Process text on-the-fly:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('gpt2')

def tokenize_function(examples):
    """Tokenize batch of text"""
    return tokenizer(
        examples['text'],
        truncation=True,
        max_length=512,
        padding='max_length',
        return_tensors='np'
    )

# Map tokenization over stream
dataset = dataset.map(
    tokenize_function,
    batched=True,  # Process batches for efficiency
    batch_size=1000,
    remove_columns=['text']  # Don't need raw text anymore
)

# Now iterate over tokenized data
for example in dataset:
    input_ids = example['input_ids']  # Shape: (512,)
    # Train on tokens
```

## Batching Streamed Data

```python
from itertools import islice
import jax.numpy as jnp

def create_batches(dataset, batch_size=32):
    """Create batches from streaming dataset"""
    
    iterator = iter(dataset)
    
    while True:
        # Take batch_size examples
        batch = list(islice(iterator, batch_size))
        
        if not batch:  # No more data
            break
        
        # Stack into arrays
        batch_dict = {}
        for key in batch[0].keys():
            batch_dict[key] = jnp.stack([ex[key] for ex in batch])
        
        yield batch_dict

# Use in training loop
for batch in create_batches(dataset, batch_size=32):
    # batch['input_ids']: (32, 512)
    loss = train_step(model, optimizer, batch)
```

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
optimizer = nnx.Optimizer(model, optax.adamw(3e-4))

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
    optimizer.update(grads)
    
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

## Next Steps

- [Track experiments with W&B](./observability.md) - Monitor streaming training
- [Export trained models](./model-export.md) - Deploy your models
- [Vision streaming example](../vision/resnet-architecture.md) - ResNet with streaming

## Complete Examples

**Organized modular examples:**
- [`examples/integrations/resnet_streaming.py`](https://github.com/mlnomadpy/flaxdocs/tree/master/examples/integrations/resnet_streaming.py) - ResNet with streaming images from HuggingFace
- [`examples/advanced/bert_fineweb.py`](https://github.com/mlnomadpy/flaxdocs/tree/master/examples/advanced/bert_fineweb.py) - BERT training with streaming text from FineWeb
- [`examples/advanced/gpt_training.py`](https://github.com/mlnomadpy/flaxdocs/tree/master/examples/advanced/gpt_training.py) - GPT training with streaming text from FineWeb
