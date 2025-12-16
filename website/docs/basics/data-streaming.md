---
sidebar_position: 5
---

# Streaming Large Datasets

Learn how to interact with datasets larger than memory.

## Why Streaming Matters

Modern datasets are huge:
- **FineWeb**: 15 trillion tokens (>10TB)
- **ImageNet-21k**: 14M images (~1TB)
- **Common Crawl**: Petabytes of text

You can't load these into RAM. Solution: **stream** data during training.

### Streaming vs Downloading

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
- **Start immediately**: No wait for download
- **Disk space**: Don't need TB of storage
- **Flexibility**: Easy to switch datasets

**Tradeoffs**:
- **Network dependency**: Need stable connection
- **Latency**: Slight overhead per batch
- **Caching**: Can cache popular samples

## Streaming with HuggingFace Datasets

### Basic Streaming Pattern

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

### Understanding Shuffle Buffers

Streaming shuffles differently than in-memory:

```python
# In-memory: Perfect shuffle
dataset = dataset.shuffle()  # Shuffles all N examples

# Streaming: Buffer shuffle
dataset = dataset.shuffle(buffer_size=10_000)
# Loads 10k examples, shuffles them, yields one
# Loads next example, shuffles 10k again, yields one
# ...
```

**Choosing buffer size**:
- Larger = better randomization, more memory
- Smaller = less memory, worse randomization
- Rule of thumb: 10-100x batch size

### Tokenization for Streaming

Process text on-the-fly:

```python
from transformers import AutoTokenizer
import jax.numpy as jnp

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

### Batching Streaming Data

```python
from itertools import islice

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
# for batch in create_batches(dataset, batch_size=32):
#     loss = train_step(model, optimizer, batch)
```
