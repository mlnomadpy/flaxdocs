---
sidebar_position: 2
---

# BERT (Bidirectional Transformers)

BERT (Bidirectional Encoder Representations from Transformers) reads text in both directions to understand context, making it ideal for understanding tasks like classification and question answering.

## Architecture

**Key innovations**:
1. **Bidirectional attention**: Each token sees all tokens (unlike GPT's causal)
2. **Masked language modeling (MLM)**: Predict masked words to learn bi-directional context.

### BERT Layer Implementation

```python
from flax import linen as nnx
import jax.numpy as jnp
import jax

class BERTAttention(nnx.Module):
    """Multi-head self-attention for BERT"""
    
    def __init__(
        self,
        hidden_size: int = 768,
        num_heads: int = 12,
        *,
        rngs: nnx.Rngs
    ):
        assert hidden_size % num_heads == 0
        
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # Q, K, V projections
        self.query = nnx.Linear(hidden_size, hidden_size, rngs=rngs)
        self.key = nnx.Linear(hidden_size, hidden_size, rngs=rngs)
        self.value = nnx.Linear(hidden_size, hidden_size, rngs=rngs)
        
        # Output projection
        self.out = nnx.Linear(hidden_size, hidden_size, rngs=rngs)
        
        self.dropout = nnx.Dropout(0.1, rngs=rngs)
    
    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        *,
        train: bool = True
    ):
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Project to Q, K, V
        q = self.query(hidden_states)
        k = self.key(hidden_states)
        v = self.value(hidden_states)
        
        # Reshape for multi-head: (batch, seq, heads, dim)
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose to (batch, heads, seq, dim)
        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))
        
        # Attention scores: Q @ K.T
        scores = (q @ jnp.swapaxes(k, -2, -1)) / jnp.sqrt(self.head_dim)
        
        # Apply mask (e.g., for padding tokens)
        if attention_mask is not None:
            scores = scores + attention_mask
        
        # Softmax and dropout
        attn_weights = jax.nn.softmax(scores, axis=-1)
        attn_weights = self.dropout(attn_weights, deterministic=not train)
        
        # Weighted sum: weights @ V
        context = attn_weights @ v
        
        # Reshape back
        context = jnp.transpose(context, (0, 2, 1, 3))
        context = context.reshape(batch_size, seq_len, hidden_size)
        
        return self.out(context)
```

## Masked Language Modeling (MLM)

The secret sauce of BERT is its training objective.

```python
import numpy as np

def create_mlm_batch(texts, tokenizer, mask_prob=0.15):
    """
    Create masked language modeling training batch.
    Objective: Predict tokens that were replaced by [MASK].
    """
    
    # Tokenize
    encodings = tokenizer(
        texts,
        padding='max_length',
        truncation=True,
        max_length=512,
        return_tensors='np'
    )
    
    input_ids = encodings['input_ids']
    
    # Create labels (same as input initially)
    labels = input_ids.copy()
    
    # Create mask: Select 15% of tokens
    rand = np.random.rand(*input_ids.shape)
    mask = (rand < mask_prob) & (input_ids != tokenizer.pad_token_id)
    
    # Replace masked positions with [MASK] token
    input_ids[mask] = tokenizer.mask_token_id
    
    # Only compute loss on masked positions (-100 is standard ignore index)
    labels[~mask] = -100
    
    return {
        'input_ids': input_ids,
        'attention_mask': encodings['attention_mask'],
        'labels': labels
    }
```
