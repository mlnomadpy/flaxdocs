---
sidebar_position: 3
---

# GPT (Generative Pre-trained Transformer)

GPT generates text left-to-right (autoregressively). Unlike BERT, it uses **causal attention**, meaning a token can only see previous tokens, not future ones.

## Architecture

| Feature | BERT | GPT |
|---------|------|-----|
| Attention | Bidirectional | Causal (unidirectional) |
| Training | Masked LM | Next token prediction |
| Use case | Understanding | Generation |

### GPT Layer Implementation

```python
from flax import linen as nnx
import jax.numpy as jnp
import jax

class GPTAttention(nnx.Module):
    """Causal self-attention for GPT"""
    
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        *,
        rngs: nnx.Rngs
    ):
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Combined QKV projection (optimization common in GPT)
        self.qkv = nnx.Linear(embed_dim, embed_dim * 3, rngs=rngs)
        self.proj = nnx.Linear(embed_dim, embed_dim, rngs=rngs)
        
        self.dropout = nnx.Dropout(0.1, rngs=rngs)
    
    def __call__(self, x, *, train: bool = True):
        batch, seq_len, embed_dim = x.shape
        
        # Project to Q, K, V
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch, seq_len, 3, self.num_heads, self.head_dim)
        qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention scores
        scores = (q @ jnp.swapaxes(k, -2, -1)) / jnp.sqrt(self.head_dim)
        
        # Causal mask: can't attend to future positions
        mask = jnp.tril(jnp.ones((seq_len, seq_len)))
        scores = jnp.where(mask, scores, float('-inf'))
        
        # Softmax and weighted sum
        attn = jax.nn.softmax(scores, axis=-1)
        attn = self.dropout(attn, deterministic=not train)
        
        out = attn @ v
        out = jnp.transpose(out, (0, 2, 1, 3))
        out = out.reshape(batch, seq_len, embed_dim)
        
        return self.proj(out)
```

## Text Generation Loop

GPT generates text one token at a time.

```python
def generate_text(
    model,
    prompt: str,
    tokenizer,
    max_length: int = 100,
    temperature: float = 1.0,
    top_k: int = 50
):
    """Generate text autoregressively"""
    
    # Tokenize prompt
    input_ids = tokenizer.encode(prompt, return_tensors='np')
    
    for _ in range(max_length):
        # Forward pass
        logits = model(input_ids)  # (1, seq_len, vocab_size)
        next_logits = logits[0, -1, :]  # Prediction for next token
        
        # Temperature scaling
        next_logits = next_logits / temperature
        
        # Top-k sampling
        top_k_logits, top_k_indices = jax.lax.top_k(next_logits, k=top_k)
        
        # Sample from top-k distribution
        probs = jax.nn.softmax(top_k_logits)
        next_token_idx = jax.random.categorical(
            jax.random.PRNGKey(0), 
            jnp.log(probs)
        )
        next_token = top_k_indices[next_token_idx]
        
        # Append to sequence
        input_ids = jnp.concatenate([input_ids, next_token[None, None]], axis=1)
        
        # Stop at end token
        if next_token == tokenizer.eos_token_id:
            break
    
    # Decode to text
    return tokenizer.decode(input_ids[0])
```
