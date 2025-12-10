"""
Flax NNX: Train GPT from Scratch on FineWeb
============================================
Complete GPT training pipeline with streaming data from HuggingFace FineWeb.
Includes full model architecture, training loop, and text generation.
Run: pip install datasets transformers tiktoken && python advanced/gpt_training.py
"""

import jax
import jax.numpy as jnp
from flax import nnx
import optax
import numpy as np
from typing import Dict, Optional, Tuple
import time


import sys
from pathlib import Path

# Add parent directory to path for shared imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from datasets import load_dataset
    from transformers import AutoTokenizer
    DATASETS_AVAILABLE = True
except ImportError:
    print("Warning: datasets/transformers not available")
    print("Install: pip install datasets transformers")
    DATASETS_AVAILABLE = False


# ============================================================================
# 1. GPT MODEL ARCHITECTURE
# ============================================================================

class GPTAttention(nnx.Module):
    """Multi-head causal self-attention."""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float,
                 max_len: int, rngs: nnx.Rngs):
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        # Q, K, V projections (combined for efficiency)
        self.qkv_proj = nnx.Linear(d_model, 3 * d_model, rngs=rngs)
        self.out_proj = nnx.Linear(d_model, d_model, rngs=rngs)
        
        self.dropout = nnx.Dropout(dropout, rngs=rngs)
        
        # Causal mask
        mask = jnp.tril(jnp.ones((max_len, max_len)))
        self.causal_mask = mask[None, None, :, :]
    
    def __call__(self, x, train: bool = False):
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = jnp.split(qkv, 3, axis=2)
        
        q = q.squeeze(2).transpose(0, 2, 1, 3)  # (batch, heads, seq, head_dim)
        k = k.squeeze(2).transpose(0, 2, 1, 3)
        v = v.squeeze(2).transpose(0, 2, 1, 3)
        
        # Attention scores
        scores = jnp.matmul(q, k.transpose(0, 1, 3, 2)) / jnp.sqrt(self.head_dim)
        
        # Apply causal mask
        mask = self.causal_mask[:, :, :seq_len, :seq_len]
        scores = jnp.where(mask, scores, -1e10)
        
        # Attention weights
        attn_weights = jax.nn.softmax(scores, axis=-1)
        attn_weights = self.dropout(attn_weights, deterministic=not train)
        
        # Apply attention to values
        attn_output = jnp.matmul(attn_weights, v)
        attn_output = attn_output.transpose(0, 2, 1, 3)
        attn_output = attn_output.reshape(batch_size, seq_len, self.d_model)
        
        # Output projection
        output = self.out_proj(attn_output)
        
        return output


class GPTMLP(nnx.Module):
    """GPT MLP (feed-forward) block."""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float, rngs: nnx.Rngs):
        self.fc1 = nnx.Linear(d_model, d_ff, rngs=rngs)
        self.fc2 = nnx.Linear(d_ff, d_model, rngs=rngs)
        self.dropout = nnx.Dropout(dropout, rngs=rngs)
    
    def __call__(self, x, train: bool = False):
        x = self.fc1(x)
        x = nnx.gelu(x)
        x = self.dropout(x, deterministic=not train)
        x = self.fc2(x)
        x = self.dropout(x, deterministic=not train)
        return x


class GPTBlock(nnx.Module):
    """GPT transformer block."""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int,
                 dropout: float, max_len: int, rngs: nnx.Rngs):
        self.attention = GPTAttention(d_model, num_heads, dropout, max_len, rngs)
        self.mlp = GPTMLP(d_model, d_ff, dropout, rngs)
        
        self.ln1 = nnx.LayerNorm(d_model, rngs=rngs)
        self.ln2 = nnx.LayerNorm(d_model, rngs=rngs)
    
    def __call__(self, x, train: bool = False):
        # Pre-norm architecture (GPT-2 style)
        x = x + self.attention(self.ln1(x), train=train)
        x = x + self.mlp(self.ln2(x), train=train)
        return x


class GPTModel(nnx.Module):
    """Complete GPT model."""
    
    def __init__(self, vocab_size: int, max_len: int, d_model: int,
                 num_layers: int, num_heads: int, d_ff: int,
                 dropout: float = 0.1, rngs: nnx.Rngs = None):
        self.max_len = max_len
        self.d_model = d_model
        
        # Token and position embeddings
        self.token_embedding = nnx.Embed(vocab_size, d_model, rngs=rngs)
        self.position_embedding = nnx.Embed(max_len, d_model, rngs=rngs)
        
        # Transformer blocks
        self.blocks = [
            GPTBlock(d_model, num_heads, d_ff, dropout, max_len, rngs)
            for _ in range(num_layers)
        ]
        
        # Final layer norm
        self.ln_f = nnx.LayerNorm(d_model, rngs=rngs)
        
        # LM head
        self.lm_head = nnx.Linear(d_model, vocab_size, use_bias=False, rngs=rngs)
        
        self.dropout = nnx.Dropout(dropout, rngs=rngs)
    
    def __call__(self, input_ids, train: bool = False):
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        token_emb = self.token_embedding(input_ids)
        
        # Position embeddings
        positions = jnp.arange(seq_len)[None, :]
        pos_emb = self.position_embedding(positions)
        
        # Combine embeddings
        x = token_emb + pos_emb
        x = self.dropout(x, deterministic=not train)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, train=train)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # LM head
        logits = self.lm_head(x)
        
        return logits


# ============================================================================
# 2. DATA LOADING FROM FINEWEB
# ============================================================================

def load_fineweb_for_gpt(tokenizer, max_length: int = 1024, 
                         batch_size: int = 8):
    """Load FineWeb dataset for GPT training."""
    if not DATASETS_AVAILABLE:
        print("Datasets not available")
        return None
    
    print("\n" + "=" * 80)
    print("Loading FineWeb Dataset for GPT Training")
    print("=" * 80)
    
    try:
        # Load FineWeb-Edu (educational quality text)
        dataset = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name="sample-10BT",
            split="train",
            streaming=True,
            trust_remote_code=True
        )
        print("✓ FineWeb-Edu loaded successfully")
    except Exception as e:
        print(f"Note: Using dummy data (FineWeb not available: {e})")
        return create_dummy_dataloader_gpt(tokenizer, max_length, batch_size)
    
    def process_batch(texts):
        """Process and tokenize texts for GPT."""
        # Tokenize
        encoded = tokenizer(
            texts,
            truncation=True,
            max_length=max_length + 1,  # +1 for targets
            padding=False,
            return_tensors=None
        )
        
        batch_input_ids = []
        batch_targets = []
        
        for ids in encoded['input_ids']:
            if len(ids) > max_length:
                # Input and target (shifted by 1)
                input_ids = ids[:max_length]
                targets = ids[1:max_length + 1]
                
                batch_input_ids.append(input_ids)
                batch_targets.append(targets)
        
        if len(batch_input_ids) == 0:
            return None
        
        # Pad sequences
        max_seq_len = max(len(seq) for seq in batch_input_ids)
        
        padded_inputs = []
        padded_targets = []
        
        for inp, tgt in zip(batch_input_ids, batch_targets):
            pad_len = max_seq_len - len(inp)
            padded_inputs.append(inp + [tokenizer.pad_token_id] * pad_len)
            padded_targets.append(tgt + [-100] * pad_len)  # -100 = ignore index
        
        return {
            'input_ids': jnp.array(padded_inputs),
            'labels': jnp.array(padded_targets)
        }
    
    def batch_generator():
        texts = []
        for example in dataset:
            text = example.get('text', '')
            if len(text) > 100:  # Filter very short texts
                texts.append(text)
            
            if len(texts) >= batch_size:
                batch = process_batch(texts)
                if batch is not None:
                    yield batch
                texts = []
    
    return batch_generator()


def create_dummy_dataloader_gpt(tokenizer, max_length: int, batch_size: int):
    """Create dummy dataloader for demonstration."""
    print("Creating dummy dataloader...")
    
    sample_texts = [
        "The transformer architecture has revolutionized natural language processing.",
        "Deep learning models can learn complex patterns from large amounts of data.",
        "Attention mechanisms allow models to focus on relevant parts of the input.",
        "Language models are trained to predict the next word in a sequence.",
        "GPT stands for Generative Pre-trained Transformer.",
        "Self-attention enables the model to weigh the importance of different tokens.",
    ] * 20
    
    def batch_generator():
        for _ in range(200):  # 200 batches
            texts = np.random.choice(sample_texts, batch_size, replace=True)
            
            encoded = tokenizer(
                list(texts),
                truncation=True,
                max_length=max_length + 1,
                padding='max_length',
                return_tensors='np'
            )
            
            input_ids = encoded['input_ids'][:, :-1]
            labels = encoded['input_ids'][:, 1:]
            
            # Mask padding tokens in labels
            labels = np.where(
                labels == tokenizer.pad_token_id,
                -100,
                labels
            )
            
            yield {
                'input_ids': jnp.array(input_ids),
                'labels': jnp.array(labels)
            }
    
    return batch_generator()


# ============================================================================
# 3. TRAINING FUNCTIONS
# ============================================================================

def compute_lm_loss(logits, labels):
    """Compute language modeling loss (cross-entropy)."""
    vocab_size = logits.shape[-1]
    
    # Flatten
    logits_flat = logits.reshape(-1, vocab_size)
    labels_flat = labels.reshape(-1)
    
    # Mask for valid targets (not -100)
    mask = labels_flat != -100
    
    if jnp.sum(mask) == 0:
        return jnp.array(0.0)
    
    # Cross entropy
    log_probs = jax.nn.log_softmax(logits_flat, axis=-1)
    target_log_probs = jnp.take_along_axis(
        log_probs, labels_flat[:, None], axis=-1
    ).squeeze(-1)
    
    loss = -jnp.sum(target_log_probs * mask) / jnp.sum(mask)
    
    return loss


def compute_perplexity(loss):
    """Compute perplexity from loss."""
    return jnp.exp(loss)


@nnx.jit
def train_step(model: GPTModel, optimizer: nnx.Optimizer, batch: Dict):
    """Single training step."""
    
    def loss_fn(model):
        logits = model(batch['input_ids'], train=True)
        loss = compute_lm_loss(logits, batch['labels'])
        return loss, logits
    
    # Compute gradients
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model)
    
    # Update parameters
    optimizer.update(grads)
    
    # Compute metrics
    perplexity = compute_perplexity(loss)
    
    # Accuracy
    predictions = jnp.argmax(logits, axis=-1)
    mask = batch['labels'] != -100
    accuracy = jnp.sum((predictions == batch['labels']) * mask) / jnp.maximum(jnp.sum(mask), 1)
    
    return {
        'loss': loss,
        'perplexity': perplexity,
        'accuracy': accuracy
    }


# ============================================================================
# 4. TEXT GENERATION
# ============================================================================

def generate_text(model: GPTModel, tokenizer, prompt: str,
                 max_new_tokens: int = 100, temperature: float = 0.8,
                 top_k: int = 50):
    """Generate text using the trained GPT model."""
    # Encode prompt
    encoded = tokenizer(prompt, return_tensors='np')
    input_ids = jnp.array(encoded['input_ids'])
    
    print(f"\nGenerating text from prompt: '{prompt}'")
    print("-" * 80)
    
    generated_tokens = []
    
    for _ in range(max_new_tokens):
        # Get logits
        logits = model(input_ids, train=False)
        next_token_logits = logits[0, -1, :] / temperature
        
        # Top-k sampling
        if top_k > 0:
            top_k_logits, top_k_indices = jax.lax.top_k(next_token_logits, top_k)
            # Sample from top-k
            probs = jax.nn.softmax(top_k_logits)
            next_token_idx = jax.random.categorical(
                jax.random.PRNGKey(int(time.time() * 1000)), 
                jnp.log(probs)
            )
            next_token = top_k_indices[next_token_idx]
        else:
            # Sample from full distribution
            next_token = jax.random.categorical(
                jax.random.PRNGKey(int(time.time() * 1000)),
                next_token_logits
            )
        
        # Append token
        generated_tokens.append(int(next_token))
        input_ids = jnp.concatenate([
            input_ids,
            jnp.array([[next_token]])
        ], axis=1)
        
        # Truncate if too long
        if input_ids.shape[1] > model.max_len:
            input_ids = input_ids[:, -model.max_len:]
        
        # Stop at EOS token
        if int(next_token) == tokenizer.eos_token_id:
            break
    
    # Decode generated text
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    full_text = prompt + generated_text
    
    print(full_text)
    print("-" * 80)
    
    return full_text


# ============================================================================
# 5. MAIN TRAINING LOOP
# ============================================================================

def main():
    print("=" * 80)
    print("Flax NNX: Train GPT from Scratch on FineWeb")
    print("=" * 80)
    
    if not DATASETS_AVAILABLE:
        print("\n" + "!" * 80)
        print("Required libraries not available!")
        print("Install: pip install datasets transformers")
        print("!" * 80)
        return
    
    # ========================================================================
    # Configuration
    # ========================================================================
    config = {
        # Model (GPT-2 Small-like, scaled down for demo)
        'vocab_size': 50257,  # GPT-2 vocab size
        'max_len': 512,  # Smaller for demo
        'd_model': 256,  # Smaller for demo
        'num_layers': 6,  # Smaller for demo
        'num_heads': 8,
        'd_ff': 1024,
        'dropout': 0.1,
        
        # Training
        'batch_size': 4,
        'learning_rate': 3e-4,
        'num_steps': 200,  # Small for demo
        'warmup_steps': 20,
        'weight_decay': 0.1,
        'grad_clip': 1.0,
        'seed': 42,
    }
    
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # ========================================================================
    # Initialize Tokenizer
    # ========================================================================
    print("\nInitializing GPT-2 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    print(f"✓ Tokenizer loaded (vocab size: {len(tokenizer)})")
    
    # ========================================================================
    # Initialize Model
    # ========================================================================
    print("\nInitializing GPT model...")
    rngs = nnx.Rngs(config['seed'])
    model = GPTModel(
        vocab_size=config['vocab_size'],
        max_len=config['max_len'],
        d_model=config['d_model'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        d_ff=config['d_ff'],
        dropout=config['dropout'],
        rngs=rngs
    )
    
    # Count parameters
    state = nnx.state(model)
    total_params = sum(x.size for x in jax.tree_util.tree_leaves(state))
    print(f"Total parameters: {total_params:,}")
    print(f"Model size: ~{total_params * 4 / 1024 / 1024:.2f} MB (fp32)")
    
    # ========================================================================
    # Initialize Optimizer with Warmup
    # ========================================================================
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config['learning_rate'],
        warmup_steps=config['warmup_steps'],
        decay_steps=config['num_steps'],
        end_value=0.0
    )
    
    optimizer_def = optax.chain(
        optax.clip_by_global_norm(config['grad_clip']),
        optax.adamw(learning_rate=schedule, weight_decay=config['weight_decay'])
    )
    
    optimizer = nnx.Optimizer(model, optimizer_def)
    
    # ========================================================================
    # Load Data
    # ========================================================================
    dataloader = load_fineweb_for_gpt(
        tokenizer,
        max_length=config['max_len'],
        batch_size=config['batch_size']
    )
    
    # ========================================================================
    # Training Loop
    # ========================================================================
    print("\n" + "=" * 80)
    print("Training GPT on FineWeb")
    print("=" * 80)
    
    running_loss = 0.0
    running_ppl = 0.0
    running_acc = 0.0
    start_time = time.time()
    
    for step, batch in enumerate(dataloader):
        if step >= config['num_steps']:
            break
        
        # Training step
        metrics = train_step(model, optimizer, batch)
        
        running_loss += float(metrics['loss'])
        running_ppl += float(metrics['perplexity'])
        running_acc += float(metrics['accuracy'])
        
        # Log every 10 steps
        if (step + 1) % 10 == 0:
            avg_loss = running_loss / 10
            avg_ppl = running_ppl / 10
            avg_acc = running_acc / 10
            elapsed = time.time() - start_time
            steps_per_sec = 10 / elapsed
            
            print(f"Step {step + 1}/{config['num_steps']} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"PPL: {avg_ppl:.2f} | "
                  f"Acc: {avg_acc:.4f} | "
                  f"Speed: {steps_per_sec:.2f} steps/s")
            
            running_loss = 0.0
            running_ppl = 0.0
            running_acc = 0.0
            start_time = time.time()
        
        # Generate sample text every 50 steps
        if (step + 1) % 50 == 0:
            print("\n" + "=" * 80)
            print(f"Sample Generation at Step {step + 1}")
            print("=" * 80)
            
            prompts = [
                "The future of artificial intelligence",
                "Once upon a time",
                "In the field of machine learning",
            ]
            
            for prompt in prompts[:1]:  # Generate from first prompt only
                generate_text(
                    model, tokenizer, prompt,
                    max_new_tokens=50,
                    temperature=0.8,
                    top_k=50
                )
            
            print()
    
    print("\n✓ Training complete!")
    
    # ========================================================================
    # Final Text Generation
    # ========================================================================
    print("\n" + "=" * 80)
    print("Final Text Generation Examples")
    print("=" * 80)
    
    prompts = [
        "The transformer architecture",
        "In recent years, deep learning",
        "The key to understanding",
    ]
    
    for prompt in prompts:
        generate_text(
            model, tokenizer, prompt,
            max_new_tokens=80,
            temperature=0.7,
            top_k=50
        )
        print()
    
    # ========================================================================
    # Best Practices
    # ========================================================================
    print("\n" + "=" * 80)
    print("Best Practices for GPT Training")
    print("=" * 80)
    
    print("""
    1. Data:
       ✓ Use large, high-quality text corpus
       ✓ FineWeb, C4, The Pile, RedPajama
       ✓ Stream data for large datasets
       ✓ Mix different data sources
    
    2. Model Configuration:
       • GPT-2 Small: 124M params, 12 layers, 768 dim
       • GPT-2 Medium: 355M params, 24 layers, 1024 dim
       • GPT-2 Large: 774M params, 36 layers, 1280 dim
       • GPT-2 XL: 1.5B params, 48 layers, 1600 dim
    
    3. Training:
       ✓ Use learning rate warmup
       ✓ Cosine decay schedule
       ✓ Gradient clipping (1.0)
       ✓ Weight decay for regularization
       ✓ Large batch sizes (256-512)
       ✓ Train for 300B+ tokens for good models
    
    4. Optimization:
       ✓ Use bfloat16 mixed precision
       ✓ Gradient accumulation for large batches
       ✓ Gradient checkpointing to save memory
       ✓ Use FlashAttention if available
       ✓ Distributed training across GPUs/TPUs
    
    5. Evaluation:
       ✓ Perplexity on held-out data
       ✓ Few-shot learning benchmarks
       ✓ Generation quality (human eval)
       ✓ Downstream task performance
    
    6. Generation:
       • Temperature: 0.7-1.0 for creativity
       • Top-k: 40-50 for quality
       • Top-p (nucleus): 0.9-0.95
       • Use repetition penalty if needed
    
    7. Scaling Laws:
       • More data > Bigger model (initially)
       • Train longer with more compute
       • Optimal model size depends on compute budget
       • Follow Chinchilla scaling laws
    
    8. Production:
       ✓ Quantize model (int8, int4)
       ✓ Use KV cache for generation
       ✓ Batch generation requests
       ✓ Monitor inference latency
    """)
    
    print("\n" + "=" * 80)
    print("✓ All examples completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
