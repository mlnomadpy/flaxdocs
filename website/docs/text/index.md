---
sidebar_position: 0
---

# Natural Language Processing

Learn to build transformer models for text - from basic attention to full GPT-style language models.

## What You'll Build

This section teaches you NLP models from first principles:

**[Simple Transformer](./simple-transformer.md)** - Start here!  
Build a transformer from scratch. Learn self-attention, multi-head attention, and how to generate text with GPT-style models.

## When to Use Transformers

### Use transformers for:
- **Text generation**: Write stories, code, summaries
- **Text classification**: Sentiment analysis, topic classification
- **Question answering**: Answer questions from context
- **Translation**: Convert between languages
- **Text understanding**: Extract information, analyze sentiment

Transformers are the dominant architecture for all NLP tasks since 2017.

## Prerequisites

Before diving into NLP models, make sure you understand:
- [Your First Model](../fundamentals/your-first-model.md) - Basic NNX concepts
- [Simple Training Loop](../workflows/simple-training.md) - How to train models

## Quick Example

Here's the attention mechanism you'll build:

```python
from flax import nnx
import jax.numpy as jnp

class SelfAttention(nnx.Module):
    def __init__(self, embed_dim: int, *, rngs: nnx.Rngs):
        self.query = nnx.Linear(embed_dim, embed_dim, rngs=rngs)
        self.key = nnx.Linear(embed_dim, embed_dim, rngs=rngs)
        self.value = nnx.Linear(embed_dim, embed_dim, rngs=rngs)
        self.scale = embed_dim ** -0.5
    
    def __call__(self, x, mask=None):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        # Attention: softmax(Q @ K^T / sqrt(d)) @ V
        scores = (q @ jnp.swapaxes(k, -2, -1)) * self.scale
        if mask is not None:
            scores = jnp.where(mask, scores, float('-inf'))
        
        attn = jax.nn.softmax(scores, axis=-1)
        return attn @ v

# This is the core of all modern NLP models!
```

## Key Concepts

### Attention Mechanism
The revolutionary idea: each word can "attend" to (look at) every other word to understand context.

### Causal Masking
For text generation, prevent looking at future tokens - only past context allowed.

### Tokenization
Convert text to numbers. Critical for all NLP work:
```python
"Hello world" → [15496, 995]  # Token IDs
```

### Autoregressive Generation
Generate text one token at a time, feeding outputs back as inputs.

## Common NLP Tasks

- **Text Generation**: GPT models generate coherent text
- **Text Understanding**: BERT models understand meaning
- **Translation**: Sequence-to-sequence models
- **Summarization**: Condense long text
- **Question Answering**: Find answers in documents

## Transformer Variants

- **GPT** (Decoder-only): For generation → This guide!
- **BERT** (Encoder-only): For understanding
- **T5** (Encoder-Decoder): For translation/summarization

This section focuses on GPT-style models, which are the most versatile.

## What's Next?

After mastering transformers:
- [BERT Models](./bert-model.md) - Bidirectional understanding
- [Tokenization](./tokenization.md) - Prepare text data
- [Fine-tuning](./fine-tuning.md) - Adapt pretrained models

## Complete Examples

- [`examples/06_language_model_training.py`](https://github.com/mlnomadpy/flaxdocs/tree/master/examples/06_language_model_training.py) - Simple language model
- [`examples/11_bert_fineweb_mteb.py`](https://github.com/mlnomadpy/flaxdocs/tree/master/examples/11_bert_fineweb_mteb.py) - BERT training
- [`examples/12_gpt_fineweb_training.py`](https://github.com/mlnomadpy/flaxdocs/tree/master/examples/12_gpt_fineweb_training.py) - GPT from scratch
