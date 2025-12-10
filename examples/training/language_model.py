"""
Flax NNX: End-to-End Language Model Training
=============================================
Complete example of training a simple Transformer language model.
Run: python training/language_model.py
"""

import jax
import jax.numpy as jnp
from flax import nnx
import optax
import numpy as np
from typing import Dict, Tuple
import time



import sys
from pathlib import Path

# Add parent directory to path for shared imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# ============================================================================
# 1. TOKENIZER (SIMPLE CHARACTER-LEVEL)
# ============================================================================

class CharTokenizer:
    """Simple character-level tokenizer."""
    
    def __init__(self, text: str):
        chars = sorted(list(set(text)))
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for ch, i in self.char_to_idx.items()}
        self.vocab_size = len(chars)
    
    def encode(self, text: str) -> np.ndarray:
        return np.array([self.char_to_idx[c] for c in text])
    
    def decode(self, indices: np.ndarray) -> str:
        return ''.join([self.idx_to_char[int(i)] for i in indices])


# ============================================================================
# 2. TRANSFORMER COMPONENTS
# ============================================================================

class MultiHeadAttention(nnx.Module):
    """Multi-head self-attention."""
    
    def __init__(self, d_model: int, num_heads: int, rngs: nnx.Rngs):
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.q_proj = nnx.Linear(d_model, d_model, rngs=rngs)
        self.k_proj = nnx.Linear(d_model, d_model, rngs=rngs)
        self.v_proj = nnx.Linear(d_model, d_model, rngs=rngs)
        self.out_proj = nnx.Linear(d_model, d_model, rngs=rngs)
    
    def __call__(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # Project and split into heads
        q = self.q_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose for attention computation
        q = jnp.transpose(q, (0, 2, 1, 3))  # (batch, heads, seq, head_dim)
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))
        
        # Attention scores
        scores = jnp.matmul(q, jnp.transpose(k, (0, 1, 3, 2))) / jnp.sqrt(self.head_dim)
        
        # Apply causal mask
        if mask is not None:
            scores = jnp.where(mask, scores, -1e9)
        
        # Attention weights
        attn_weights = jax.nn.softmax(scores, axis=-1)
        
        # Apply attention to values
        attn_output = jnp.matmul(attn_weights, v)
        
        # Concatenate heads
        attn_output = jnp.transpose(attn_output, (0, 2, 1, 3))
        attn_output = attn_output.reshape(batch_size, seq_len, self.d_model)
        
        # Final projection
        output = self.out_proj(attn_output)
        
        return output


class FeedForward(nnx.Module):
    """Feed-forward network."""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float, rngs: nnx.Rngs):
        self.fc1 = nnx.Linear(d_model, d_ff, rngs=rngs)
        self.fc2 = nnx.Linear(d_ff, d_model, rngs=rngs)
        self.dropout = nnx.Dropout(dropout, rngs=rngs)
    
    def __call__(self, x, train: bool = False):
        x = self.fc1(x)
        x = nnx.gelu(x)
        x = self.dropout(x, deterministic=not train)
        x = self.fc2(x)
        return x


class TransformerBlock(nnx.Module):
    """Transformer encoder/decoder block."""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, 
                 dropout: float, rngs: nnx.Rngs):
        self.attention = MultiHeadAttention(d_model, num_heads, rngs)
        self.feed_forward = FeedForward(d_model, d_ff, dropout, rngs)
        
        self.norm1 = nnx.LayerNorm(d_model, rngs=rngs)
        self.norm2 = nnx.LayerNorm(d_model, rngs=rngs)
        
        self.dropout1 = nnx.Dropout(dropout, rngs=rngs)
        self.dropout2 = nnx.Dropout(dropout, rngs=rngs)
    
    def __call__(self, x, mask=None, train: bool = False):
        # Self-attention with residual
        attn_out = self.attention(x, mask)
        x = x + self.dropout1(attn_out, deterministic=not train)
        x = self.norm1(x)
        
        # Feed-forward with residual
        ff_out = self.feed_forward(x, train=train)
        x = x + self.dropout2(ff_out, deterministic=not train)
        x = self.norm2(x)
        
        return x


# ============================================================================
# 3. LANGUAGE MODEL
# ============================================================================

class TransformerLM(nnx.Module):
    """Transformer language model."""
    
    def __init__(self, vocab_size: int, d_model: int, num_heads: int,
                 num_layers: int, d_ff: int, max_seq_len: int, 
                 dropout: float = 0.1, rngs: nnx.Rngs = None):
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Token and position embeddings
        self.token_embedding = nnx.Embed(vocab_size, d_model, rngs=rngs)
        self.position_embedding = nnx.Embed(max_seq_len, d_model, rngs=rngs)
        
        # Transformer blocks
        self.blocks = [
            TransformerBlock(d_model, num_heads, d_ff, dropout, rngs)
            for _ in range(num_layers)
        ]
        
        # Output projection
        self.output_proj = nnx.Linear(d_model, vocab_size, rngs=rngs)
        
        self.dropout = nnx.Dropout(dropout, rngs=rngs)
    
    def __call__(self, x, train: bool = False):
        batch_size, seq_len = x.shape
        
        # Create causal mask
        mask = jnp.tril(jnp.ones((seq_len, seq_len)))
        mask = mask[None, None, :, :]  # Add batch and head dimensions
        
        # Token embeddings
        token_emb = self.token_embedding(x)
        
        # Position embeddings
        positions = jnp.arange(seq_len)[None, :]
        pos_emb = self.position_embedding(positions)
        
        # Combine embeddings
        x = token_emb + pos_emb
        x = self.dropout(x, deterministic=not train)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, mask=mask, train=train)
        
        # Project to vocabulary
        logits = self.output_proj(x)
        
        return logits


# ============================================================================
# 4. DATA PREPARATION
# ============================================================================

def create_dataset(text: str, tokenizer: CharTokenizer, 
                   seq_len: int, batch_size: int):
    """Create training dataset from text."""
    # Encode text
    data = tokenizer.encode(text)
    
    # Create sequences
    sequences = []
    for i in range(0, len(data) - seq_len, seq_len // 2):  # 50% overlap
        if i + seq_len + 1 <= len(data):
            sequences.append(data[i:i + seq_len + 1])
    
    sequences = np.array(sequences)
    np.random.shuffle(sequences)
    
    # Split into batches
    num_batches = len(sequences) // batch_size
    sequences = sequences[:num_batches * batch_size]
    
    batches = []
    for i in range(num_batches):
        batch = sequences[i * batch_size:(i + 1) * batch_size]
        # Input and target (shifted by 1)
        x = batch[:, :-1]
        y = batch[:, 1:]
        batches.append((x, y))
    
    return batches


# ============================================================================
# 5. TRAINING
# ============================================================================

def compute_loss(logits, targets):
    """Compute cross-entropy loss."""
    vocab_size = logits.shape[-1]
    logits_flat = logits.reshape(-1, vocab_size)
    targets_flat = targets.reshape(-1)
    
    # Cross entropy
    log_probs = jax.nn.log_softmax(logits_flat, axis=-1)
    one_hot_targets = jax.nn.one_hot(targets_flat, vocab_size)
    loss = -jnp.mean(jnp.sum(one_hot_targets * log_probs, axis=-1))
    
    return loss


def compute_accuracy(logits, targets):
    """Compute token-level accuracy."""
    predictions = jnp.argmax(logits, axis=-1)
    return jnp.mean(predictions == targets)


@nnx.jit
def train_step(model: TransformerLM, optimizer: nnx.Optimizer, 
               x: jnp.ndarray, y: jnp.ndarray):
    """Single training step."""
    
    def loss_fn(model):
        logits = model(x, train=True)
        loss = compute_loss(logits, y)
        return loss, logits
    
    # Compute gradients
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model)
    
    # Update parameters
    optimizer.update(grads)
    
    # Compute metrics
    accuracy = compute_accuracy(logits, y)
    
    return {'loss': loss, 'accuracy': accuracy}


# ============================================================================
# 6. TEXT GENERATION
# ============================================================================

def generate_text(model: TransformerLM, tokenizer: CharTokenizer, 
                  prompt: str, max_length: int = 100, temperature: float = 1.0):
    """Generate text from prompt."""
    # Encode prompt
    tokens = tokenizer.encode(prompt)
    tokens = jnp.array(tokens)[None, :]  # Add batch dimension
    
    for _ in range(max_length):
        # Get predictions
        logits = model(tokens, train=False)
        next_logits = logits[0, -1, :] / temperature
        
        # Sample next token
        probs = jax.nn.softmax(next_logits)
        next_token = jax.random.categorical(
            jax.random.PRNGKey(int(time.time() * 1000)), 
            next_logits
        )
        
        # Append to sequence
        tokens = jnp.concatenate([tokens, next_token[None, None]], axis=1)
        
        # Truncate if too long
        if tokens.shape[1] > model.max_seq_len:
            tokens = tokens[:, -model.max_seq_len:]
    
    # Decode
    generated = tokenizer.decode(tokens[0])
    return generated


# ============================================================================
# 7. MAIN TRAINING
# ============================================================================

def main():
    print("=" * 80)
    print("Flax NNX: Language Model Training")
    print("=" * 80)
    
    # ========================================================================
    # Sample Text (Shakespeare)
    # ========================================================================
    sample_text = """
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die—to sleep,
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to: 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep, perchance to dream—ay, there's the rub:
For in that sleep of death what dreams may come,
When we have shuffled off this mortal coil,
Must give us pause—there's the respect
That makes calamity of so long life.
""" * 20  # Repeat for more training data
    
    # ========================================================================
    # Configuration
    # ========================================================================
    config = {
        'vocab_size': None,  # Will be set by tokenizer
        'd_model': 128,
        'num_heads': 4,
        'num_layers': 4,
        'd_ff': 512,
        'max_seq_len': 64,
        'dropout': 0.1,
        'batch_size': 16,
        'seq_len': 64,
        'num_epochs': 50,
        'learning_rate': 3e-4,
        'seed': 42,
    }
    
    # ========================================================================
    # Prepare Data
    # ========================================================================
    print("\nPreparing data...")
    tokenizer = CharTokenizer(sample_text)
    config['vocab_size'] = tokenizer.vocab_size
    
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Sample characters: {list(tokenizer.char_to_idx.keys())[:20]}")
    
    batches = create_dataset(
        sample_text, tokenizer, config['seq_len'], config['batch_size']
    )
    print(f"Number of batches: {len(batches)}")
    
    # ========================================================================
    # Initialize Model
    # ========================================================================
    print("\nInitializing model...")
    rngs = nnx.Rngs(config['seed'])
    model = TransformerLM(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        d_ff=config['d_ff'],
        max_seq_len=config['max_seq_len'],
        dropout=config['dropout'],
        rngs=rngs
    )
    
    # Count parameters
    state = nnx.state(model)
    total_params = sum(x.size for x in jax.tree_util.tree_leaves(state))
    print(f"Total parameters: {total_params:,}")
    
    # ========================================================================
    # Initialize Optimizer
    # ========================================================================
    optimizer = nnx.Optimizer(model, optax.adam(config['learning_rate']))
    
    # ========================================================================
    # Training Loop
    # ========================================================================
    print("\n" + "=" * 80)
    print("Training")
    print("=" * 80)
    
    for epoch in range(config['num_epochs']):
        start_time = time.time()
        epoch_metrics = []
        
        for x, y in batches:
            x = jnp.array(x)
            y = jnp.array(y)
            metrics = train_step(model, optimizer, x, y)
            epoch_metrics.append(metrics)
        
        # Average metrics
        avg_loss = jnp.mean(jnp.array([m['loss'] for m in epoch_metrics]))
        avg_acc = jnp.mean(jnp.array([m['accuracy'] for m in epoch_metrics]))
        
        epoch_time = time.time() - start_time
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}/{config['num_epochs']} "
                  f"({epoch_time:.2f}s) - "
                  f"Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}")
            
            # Generate sample text
            prompt = "To be"
            generated = generate_text(model, tokenizer, prompt, max_length=50)
            print(f"  Generated: {generated[:100]}...")
    
    # ========================================================================
    # Final Generation
    # ========================================================================
    print("\n" + "=" * 80)
    print("Text Generation Examples")
    print("=" * 80)
    
    prompts = ["To be", "The ", "What "]
    for prompt in prompts:
        generated = generate_text(model, tokenizer, prompt, max_length=100)
        print(f"\nPrompt: '{prompt}'")
        print(f"Generated: {generated}")
    
    print("\n" + "=" * 80)
    print("✓ Training completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
