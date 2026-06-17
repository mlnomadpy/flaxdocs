---
sidebar_position: 0
title: NLP with Transformers in Flax NNX
description: Build transformer models for NLP with Flax NNX, from self-attention and causal masking to GPT-style text generation, tokenization, and language modeling.
keywords: [NLP, transformers, Flax NNX, self-attention, GPT, language models, text generation, tokenization, attention mechanism]
---

# Natural Language Processing

Learn to build transformer models for text - from basic attention to full GPT-style language models.

:::note Prerequisites
This guide builds on [Your First Model](/basics/fundamentals/your-first-model) and [Simple Training Loop](/basics/workflows/simple-training).
:::

:::tip What you'll learn
- How self-attention lets each token attend to every other token for context
- Why causal masking blocks attention to future tokens during generation
- How GPT, BERT, and T5 differ (decoder-only, encoder-only, encoder-decoder)
- Generate text autoregressively, one token at a time
:::

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

## Next steps

- [Simple Transformer](/basics/text/simple-transformer) - Build attention and a GPT model from scratch
- [GPT in JAX](/architectures/gpt) - The decoder-only architecture in depth
- [BERT in JAX](/architectures/bert) - The encoder-only architecture for understanding

## Complete Examples

- [`examples/training/language_model.py`](https://github.com/mlnomadpy/flaxdocs/tree/master/examples/training/language_model.py) - Simple language model
- [`examples/advanced/bert_fineweb.py`](https://github.com/mlnomadpy/flaxdocs/tree/master/examples/advanced/bert_fineweb.py) - BERT training
- [`examples/advanced/gpt_training.py`](https://github.com/mlnomadpy/flaxdocs/tree/master/examples/advanced/gpt_training.py) - GPT from scratch
