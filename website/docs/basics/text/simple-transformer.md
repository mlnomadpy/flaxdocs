---
sidebar_position: 1
---

# Text Generation with Transformers

Learn to build transformer models for text generation, from simple attention mechanisms to full GPT-style architectures.

## Why Transformers for Text?

Text has different structure than images:
- **Sequential**: Order matters (unlike images)
- **Long-range dependencies**: Words can relate to words far away
- **Variable length**: Sentences can be any length

Transformers handle these through **self-attention** - letting each word attend to all other words.

## Understanding Self-Attention

The core idea: each word looks at all other words and decides which ones are relevant:

```python
import jax
import jax.numpy as jnp
from flax import nnx

class SelfAttention(nnx.Module):
    """Single-head self-attention"""
    
    def __init__(self, embed_dim: int, *, rngs: nnx.Rngs):
        self.embed_dim = embed_dim
        self.scale = embed_dim ** -0.5  # For numerical stability
        
        # Linear projections for queries, keys, values
        self.query = nnx.Linear(embed_dim, embed_dim, rngs=rngs)
        self.key = nnx.Linear(embed_dim, embed_dim, rngs=rngs)
        self.value = nnx.Linear(embed_dim, embed_dim, rngs=rngs)
        
        # Output projection
        self.proj = nnx.Linear(embed_dim, embed_dim, rngs=rngs)
    
    def __call__(self, x: jax.Array, mask: jax.Array | None = None) -> jax.Array:
        """
        Args:
            x: (batch, seq_len, embed_dim)
            mask: (seq_len, seq_len) - optional causal mask
        Returns:
            (batch, seq_len, embed_dim)
        """
        # Project to queries, keys, values
        q = self.query(x)  # (batch, seq_len, embed_dim)
        k = self.key(x)    # (batch, seq_len, embed_dim)
        v = self.value(x)  # (batch, seq_len, embed_dim)
        
        # Compute attention scores: Q @ K^T / sqrt(d)
        scores = (q @ jnp.swapaxes(k, -2, -1)) * self.scale
        # Shape: (batch, seq_len, seq_len)
        
        # Apply causal mask (prevent looking at future tokens)
        if mask is not None:
            scores = jnp.where(mask, scores, float('-inf'))
        
        # Softmax to get attention weights
        attn_weights = jax.nn.softmax(scores, axis=-1)
        # Shape: (batch, seq_len, seq_len)
        
        # Apply attention to values
        out = attn_weights @ v  # (batch, seq_len, embed_dim)
        
        # Output projection
        return self.proj(out)

# Example usage
embed_dim = 128
seq_len = 10
batch = 2

model = SelfAttention(embed_dim, rngs=nnx.Rngs(params=0))
x = jax.random.normal(jax.random.PRNGKey(0), (batch, seq_len, embed_dim))

# Create causal mask (for autoregressive generation)
mask = jnp.tril(jnp.ones((seq_len, seq_len)))  # Lower triangular
mask = mask.reshape(1, seq_len, seq_len)  # Add batch dimension

output = model(x, mask)
print(f"Output shape: {output.shape}")  # (2, 10, 128)
```

## Understanding the Attention Mechanism

**What each part does**:
- **Query (Q)**: "What am I looking for?"
- **Key (K)**: "What do I contain?"
- **Value (V)**: "What do I want to output?"

**The process**:
1. Each token creates a query: "I'm looking for subjects"
2. Each token creates a key: "I'm a verb"
3. Compare queries and keys to get attention weights
4. Use weights to combine values

## Multi-Head Attention

Instead of one attention, use multiple "heads" that learn different patterns:

```python
class MultiHeadAttention(nnx.Module):
    """Multi-head self-attention"""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        rngs: nnx.Rngs
    ):
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Combined projection for efficiency
        self.qkv = nnx.Linear(embed_dim, embed_dim * 3, rngs=rngs)
        self.proj = nnx.Linear(embed_dim, embed_dim, rngs=rngs)
    
    def __call__(self, x: jax.Array, mask: jax.Array | None = None) -> jax.Array:
        batch, seq_len, embed_dim = x.shape
        
        # Project and split into Q, K, V
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch, seq_len, 3, self.num_heads, self.head_dim)
        qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))
        # Shape: (3, batch, num_heads, seq_len, head_dim)
        
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention scores per head
        scores = (q @ jnp.swapaxes(k, -2, -1)) * self.scale
        # Shape: (batch, num_heads, seq_len, seq_len)
        
        # Apply mask
        if mask is not None:
            # Broadcast mask to all heads
            mask = mask.reshape(1, 1, seq_len, seq_len)
            scores = jnp.where(mask, scores, float('-inf'))
        
        # Softmax and apply to values
        attn = jax.nn.softmax(scores, axis=-1)
        out = attn @ v  # (batch, num_heads, seq_len, head_dim)
        
        # Concatenate heads
        out = jnp.transpose(out, (0, 2, 1, 3))
        out = out.reshape(batch, seq_len, embed_dim)
        
        # Output projection
        return self.proj(out)
```

**Why multiple heads?**
- **Different patterns**: One head might learn syntax, another semantics
- **Redundancy**: If one head fails, others can compensate
- **Richer representations**: Combines multiple views of the data

## Transformer Block

A complete transformer block combines attention with a feed-forward network:

```python
class TransformerBlock(nnx.Module):
    """Complete transformer block"""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float = 0.1,
        *,
        rngs: nnx.Rngs
    ):
        # Multi-head attention
        self.attention = MultiHeadAttention(embed_dim, num_heads, rngs=rngs)
        
        # Feed-forward network (position-wise)
        self.mlp = nnx.Sequential(
            nnx.Linear(embed_dim, mlp_dim, rngs=rngs),
            nnx.gelu,
            nnx.Linear(mlp_dim, embed_dim, rngs=rngs),
        )
        
        # Layer normalization (pre-norm style)
        self.norm1 = nnx.LayerNorm(embed_dim, rngs=rngs)
        self.norm2 = nnx.LayerNorm(embed_dim, rngs=rngs)
        
        # Dropout
        self.dropout1 = nnx.Dropout(dropout, rngs=rngs)
        self.dropout2 = nnx.Dropout(dropout, rngs=rngs)
    
    def __call__(
        self, 
        x: jax.Array, 
        mask: jax.Array | None = None,
        *, 
        train: bool = True
    ) -> jax.Array:
        # Attention block with residual connection
        attn_out = self.attention(self.norm1(x), mask)
        if train:
            attn_out = self.dropout1(attn_out)
        x = x + attn_out  # Residual connection
        
        # MLP block with residual connection
        mlp_out = self.mlp(self.norm2(x))
        if train:
            mlp_out = self.dropout2(mlp_out)
        x = x + mlp_out  # Residual connection
        
        return x
```

**Key design choices**:
- **Pre-norm**: Normalize before sublayers (more stable than post-norm)
- **Residual connections**: Like ResNet, helps gradient flow
- **GELU activation**: Smoother than ReLU, works better for transformers

## Complete GPT-Style Model

Now let's build a full language model:

```python
class GPTModel(nnx.Module):
    """GPT-style transformer for text generation"""
    
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        mlp_dim: int,
        dropout: float = 0.1,
        *,
        rngs: nnx.Rngs
    ):
        # Token and position embeddings
        self.token_embedding = nnx.Embed(vocab_size, embed_dim, rngs=rngs)
        self.position_embedding = nnx.Embed(max_seq_len, embed_dim, rngs=rngs)
        
        # Stack of transformer blocks
        self.blocks = [
            TransformerBlock(embed_dim, num_heads, mlp_dim, dropout, rngs=rngs)
            for _ in range(num_layers)
        ]
        
        # Final layer norm
        self.ln_f = nnx.LayerNorm(embed_dim, rngs=rngs)
        
        # Output projection to vocabulary
        self.head = nnx.Linear(embed_dim, vocab_size, rngs=rngs)
        
        # Causal mask (cache for efficiency)
        self.max_seq_len = max_seq_len
    
    def __call__(self, tokens: jax.Array, *, train: bool = True) -> jax.Array:
        """
        Args:
            tokens: (batch, seq_len) - integer token IDs
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        batch, seq_len = tokens.shape
        
        # Create position indices
        positions = jnp.arange(seq_len)
        
        # Embed tokens and positions
        x = self.token_embedding(tokens) + self.position_embedding(positions)
        # Shape: (batch, seq_len, embed_dim)
        
        # Create causal mask
        mask = jnp.tril(jnp.ones((seq_len, seq_len)))
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, mask, train=train)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Project to vocabulary
        logits = self.head(x)
        # Shape: (batch, seq_len, vocab_size)
        
        return logits

# Create a small GPT model
model = GPTModel(
    vocab_size=50257,    # GPT-2 vocabulary size
    max_seq_len=1024,    # Maximum sequence length
    embed_dim=768,       # Embedding dimension
    num_heads=12,        # Number of attention heads
    num_layers=12,       # Number of transformer blocks
    mlp_dim=3072,        # MLP hidden dimension (4x embed_dim)
    dropout=0.1,
    rngs=nnx.Rngs(params=0, dropout=1)
)

# Test with dummy tokens
tokens = jnp.array([[1, 2, 3, 4, 5]])  # Shape: (1, 5)
logits = model(tokens, train=False)
print(f"Output shape: {logits.shape}")  # (1, 5, 50257)
```

## Training a Language Model

Here's a complete training loop:

```python
import optax

def train_language_model():
    # Create model
    model = GPTModel(
        vocab_size=50257,
        max_seq_len=512,
        embed_dim=384,
        num_heads=6,
        num_layers=6,
        mlp_dim=1536,
        rngs=nnx.Rngs(params=0, dropout=1)
    )
    
    # Create optimizer
    optimizer = nnx.Optimizer(
        model,
        optax.adamw(learning_rate=3e-4, weight_decay=0.1)
    )
    
    # Training loop
    for epoch in range(10):
        for batch in train_loader:
            tokens = batch['input_ids']  # (batch, seq_len)
            
            # Forward pass
            def loss_fn(model):
                logits = model(tokens, train=True)
                # Shift for next-token prediction
                logits = logits[:, :-1]  # Remove last prediction
                targets = tokens[:, 1:]   # Remove first token
                
                # Cross-entropy loss
                loss = optax.softmax_cross_entropy_with_integer_labels(
                    logits.reshape(-1, logits.shape[-1]),
                    targets.reshape(-1)
                ).mean()
                return loss
            
            loss, grads = nnx.value_and_grad(loss_fn)(model)
            optimizer.update(grads)
        
        print(f"Epoch {epoch}: Loss = {loss:.4f}")
```

## Text Generation

Once trained, generate text autoregressively:

```python
def generate_text(
    model: GPTModel,
    prompt_tokens: jax.Array,
    max_new_tokens: int = 50,
    temperature: float = 1.0
) -> jax.Array:
    """Generate text one token at a time"""
    
    tokens = prompt_tokens
    
    for _ in range(max_new_tokens):
        # Get predictions
        logits = model(tokens, train=False)
        next_token_logits = logits[:, -1, :]  # Last position
        
        # Apply temperature
        next_token_logits = next_token_logits / temperature
        
        # Sample next token
        next_token = jax.random.categorical(
            jax.random.PRNGKey(0), 
            next_token_logits
        )
        
        # Append to sequence
        tokens = jnp.concatenate([tokens, next_token[:, None]], axis=1)
        
        # Truncate if too long
        if tokens.shape[1] > model.max_seq_len:
            tokens = tokens[:, -model.max_seq_len:]
    
    return tokens

# Example usage
prompt = jnp.array([[1, 2, 3]])  # Your tokenized prompt
generated = generate_text(model, prompt, max_new_tokens=20)
# Convert back to text with your tokenizer
```

## Common Issues

### Issue 1: Causal Mask Shape

❌ **Wrong**: Not broadcasting properly
```python
mask = jnp.tril(jnp.ones((seq_len, seq_len)))
scores = jnp.where(mask, scores, float('-inf'))  # Shape error!
```

✅ **Right**: Add batch and head dimensions
```python
mask = jnp.tril(jnp.ones((seq_len, seq_len)))
mask = mask.reshape(1, 1, seq_len, seq_len)  # (1, 1, seq, seq)
```

### Issue 2: Position Embeddings Range

❌ **Wrong**: Positions out of bounds
```python
positions = jnp.arange(seq_len)  # Can exceed max_seq_len!
```

✅ **Right**: Clip or use relative positions
```python
positions = jnp.minimum(jnp.arange(seq_len), model.max_seq_len - 1)
```

## Key Takeaways

- **Self-attention** lets each token attend to all others
- **Causal mask** prevents looking at future tokens (for generation)
- **Multi-head attention** learns multiple patterns simultaneously
- **Transformer blocks** combine attention + MLP with residual connections
- **Pre-norm** is more stable than post-norm

## Next Steps

Now that you understand transformers, you can explore more advanced NLP topics or apply these concepts to real projects.

## Complete Example

See the full runnable code in [`examples/06_language_model_training.py`](https://github.com/mlnomadpy/flaxdocs/tree/master/examples/06_language_model_training.py) and [`examples/12_gpt_fineweb_training.py`](https://github.com/mlnomadpy/flaxdocs/tree/master/examples/12_gpt_fineweb_training.py).
