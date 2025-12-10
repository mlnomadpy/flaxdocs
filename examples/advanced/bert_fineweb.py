"""
Flax NNX: Train BERT on FineWeb and Evaluate on MTEB
=====================================================
Train BERT model from scratch on FineWeb dataset (streaming from HF).
Then evaluate on MTEB (Massive Text Embedding Benchmark).
Run: pip install datasets transformers sentence-transformers && python 11_bert_fineweb_mteb.py
"""

import jax
import jax.numpy as jnp
from flax import nnx
import optax
import numpy as np
from typing import Dict, Optional
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

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ============================================================================
# 1. BERT MODEL ARCHITECTURE
# ============================================================================

class BERTEmbedding(nnx.Module):
    """BERT embeddings: token + position + segment."""
    
    def __init__(self, vocab_size: int, max_len: int, d_model: int, 
                 dropout: float, rngs: nnx.Rngs):
        self.token_emb = nnx.Embed(vocab_size, d_model, rngs=rngs)
        self.pos_emb = nnx.Embed(max_len, d_model, rngs=rngs)
        self.segment_emb = nnx.Embed(2, d_model, rngs=rngs)  # For NSP task
        
        self.norm = nnx.LayerNorm(d_model, rngs=rngs)
        self.dropout = nnx.Dropout(dropout, rngs=rngs)
    
    def __call__(self, input_ids, segment_ids=None, train: bool = False):
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        token_emb = self.token_emb(input_ids)
        
        # Position embeddings
        positions = jnp.arange(seq_len)[None, :]
        pos_emb = self.pos_emb(positions)
        
        # Segment embeddings (default to 0 if not provided)
        if segment_ids is None:
            segment_ids = jnp.zeros_like(input_ids)
        segment_emb = self.segment_emb(segment_ids)
        
        # Combine
        embeddings = token_emb + pos_emb + segment_emb
        embeddings = self.norm(embeddings)
        embeddings = self.dropout(embeddings, deterministic=not train)
        
        return embeddings


class BERTAttention(nnx.Module):
    """Multi-head self-attention for BERT."""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float, 
                 rngs: nnx.Rngs):
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads
        
        self.q_proj = nnx.Linear(d_model, d_model, rngs=rngs)
        self.k_proj = nnx.Linear(d_model, d_model, rngs=rngs)
        self.v_proj = nnx.Linear(d_model, d_model, rngs=rngs)
        self.out_proj = nnx.Linear(d_model, d_model, rngs=rngs)
        
        self.dropout = nnx.Dropout(dropout, rngs=rngs)
    
    def __call__(self, x, mask=None, train: bool = False):
        batch_size, seq_len, _ = x.shape
        
        # Project and split heads
        q = self.q_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose for attention
        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))
        
        # Attention scores
        scores = jnp.matmul(q, jnp.transpose(k, (0, 1, 3, 2))) / jnp.sqrt(self.head_dim)
        
        if mask is not None:
            scores = jnp.where(mask, scores, -1e9)
        
        attn_weights = jax.nn.softmax(scores, axis=-1)
        attn_weights = self.dropout(attn_weights, deterministic=not train)
        
        # Apply attention
        attn_output = jnp.matmul(attn_weights, v)
        attn_output = jnp.transpose(attn_output, (0, 2, 1, 3))
        attn_output = attn_output.reshape(batch_size, seq_len, self.d_model)
        
        output = self.out_proj(attn_output)
        return output


class BERTLayer(nnx.Module):
    """Single BERT transformer layer."""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, 
                 dropout: float, rngs: nnx.Rngs):
        self.attention = BERTAttention(d_model, num_heads, dropout, rngs)
        
        self.ff1 = nnx.Linear(d_model, d_ff, rngs=rngs)
        self.ff2 = nnx.Linear(d_ff, d_model, rngs=rngs)
        
        self.norm1 = nnx.LayerNorm(d_model, rngs=rngs)
        self.norm2 = nnx.LayerNorm(d_model, rngs=rngs)
        
        self.dropout = nnx.Dropout(dropout, rngs=rngs)
    
    def __call__(self, x, mask=None, train: bool = False):
        # Self-attention
        attn_out = self.attention(x, mask, train)
        x = self.norm1(x + self.dropout(attn_out, deterministic=not train))
        
        # Feed-forward
        ff_out = self.ff2(nnx.gelu(self.ff1(x)))
        x = self.norm2(x + self.dropout(ff_out, deterministic=not train))
        
        return x


class BERTModel(nnx.Module):
    """Complete BERT model."""
    
    def __init__(self, vocab_size: int, max_len: int, d_model: int,
                 num_layers: int, num_heads: int, d_ff: int, 
                 dropout: float = 0.1, rngs: nnx.Rngs = None):
        # Embeddings
        self.embeddings = BERTEmbedding(vocab_size, max_len, d_model, dropout, rngs)
        
        # Transformer layers
        self.layers = [
            BERTLayer(d_model, num_heads, d_ff, dropout, rngs)
            for _ in range(num_layers)
        ]
        
        # MLM head
        self.mlm_dense = nnx.Linear(d_model, d_model, rngs=rngs)
        self.mlm_norm = nnx.LayerNorm(d_model, rngs=rngs)
        self.mlm_output = nnx.Linear(d_model, vocab_size, rngs=rngs)
    
    def __call__(self, input_ids, attention_mask=None, train: bool = False):
        # Embeddings
        x = self.embeddings(input_ids, train=train)
        
        # Attention mask
        if attention_mask is not None:
            mask = attention_mask[:, None, None, :]
        else:
            mask = None
        
        # Transformer layers
        for layer in self.layers:
            x = layer(x, mask, train)
        
        return x
    
    def get_mlm_logits(self, hidden_states):
        """Get MLM prediction logits."""
        x = self.mlm_dense(hidden_states)
        x = nnx.gelu(x)
        x = self.mlm_norm(x)
        logits = self.mlm_output(x)
        return logits


# ============================================================================
# 2. DATA LOADING FROM FINEWEB
# ============================================================================

def load_fineweb_streaming(tokenizer, max_length: int = 512, batch_size: int = 8):
    """Load FineWeb dataset in streaming mode."""
    if not DATASETS_AVAILABLE:
        print("Datasets not available")
        return None
    
    print("\n" + "=" * 80)
    print("Loading FineWeb Dataset (Streaming)")
    print("=" * 80)
    
    # Load FineWeb (sample version for demo)
    # Full version: "HuggingFaceFW/fineweb"
    try:
        dataset = load_dataset(
            "HuggingFaceFW/fineweb-edu",  # Educational subset
            name="sample-10BT",  # Small sample
            split="train",
            streaming=True,
            trust_remote_code=True
        )
        print("✓ FineWeb-Edu loaded (streaming)")
    except Exception as e:
        print(f"Note: Using placeholder data (FineWeb not available: {e})")
        # Create dummy data for demonstration
        return create_dummy_dataloader(tokenizer, max_length, batch_size)
    
    def process_batch(examples):
        """Process and tokenize batch."""
        texts = [ex.get('text', '') for ex in examples]
        
        # Tokenize
        encoded = tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='np'
        )
        
        # Create MLM labels
        input_ids = encoded['input_ids']
        labels = input_ids.copy()
        
        # Mask 15% of tokens
        mask_prob = 0.15
        mask_token_id = tokenizer.mask_token_id
        
        rand = np.random.rand(*input_ids.shape)
        mask_indices = rand < mask_prob
        
        labels[~mask_indices] = -100  # Ignore non-masked tokens
        input_ids[mask_indices] = mask_token_id
        
        return {
            'input_ids': jnp.array(input_ids),
            'labels': jnp.array(labels),
            'attention_mask': jnp.array(encoded['attention_mask'])
        }
    
    def batch_generator():
        batch_examples = []
        for example in dataset:
            batch_examples.append(example)
            
            if len(batch_examples) >= batch_size:
                yield process_batch(batch_examples)
                batch_examples = []
    
    return batch_generator()


def create_dummy_dataloader(tokenizer, max_length: int, batch_size: int):
    """Create dummy dataloader for demonstration."""
    print("Creating dummy dataloader for demonstration...")
    
    dummy_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming the world.",
        "Natural language processing enables computers to understand text.",
        "Deep learning models can learn complex patterns from data.",
    ] * (batch_size // 4 + 1)
    
    def batch_generator():
        for _ in range(100):  # Generate 100 batches
            texts = np.random.choice(dummy_texts, batch_size, replace=True)
            
            encoded = tokenizer(
                list(texts),
                padding='max_length',
                truncation=True,
                max_length=max_length,
                return_tensors='np'
            )
            
            input_ids = encoded['input_ids']
            labels = input_ids.copy()
            
            # Create masks
            rand = np.random.rand(*input_ids.shape)
            mask_indices = rand < 0.15
            labels[~mask_indices] = -100
            input_ids[mask_indices] = tokenizer.mask_token_id
            
            yield {
                'input_ids': jnp.array(input_ids),
                'labels': jnp.array(labels),
                'attention_mask': jnp.array(encoded['attention_mask'])
            }
    
    return batch_generator()


# ============================================================================
# 3. TRAINING FUNCTIONS
# ============================================================================

def compute_mlm_loss(logits, labels):
    """Compute masked language modeling loss."""
    # Flatten
    logits_flat = logits.reshape(-1, logits.shape[-1])
    labels_flat = labels.reshape(-1)
    
    # Only compute loss on masked tokens (labels != -100)
    mask = labels_flat != -100
    
    if jnp.sum(mask) == 0:
        return jnp.array(0.0)
    
    # Cross entropy on masked tokens only
    log_probs = jax.nn.log_softmax(logits_flat, axis=-1)
    target_log_probs = jnp.take_along_axis(
        log_probs, labels_flat[:, None], axis=-1
    ).squeeze(-1)
    
    loss = -jnp.sum(target_log_probs * mask) / jnp.sum(mask)
    return loss


@nnx.jit
def train_step(model: BERTModel, optimizer: nnx.Optimizer, batch: Dict):
    """Single training step."""
    
    def loss_fn(model):
        hidden_states = model(
            batch['input_ids'],
            batch['attention_mask'],
            train=True
        )
        logits = model.get_mlm_logits(hidden_states)
        loss = compute_mlm_loss(logits, batch['labels'])
        return loss, logits
    
    # Compute gradients
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model)
    
    # Update
    optimizer.update(grads)
    
    # Accuracy on masked tokens
    predictions = jnp.argmax(logits, axis=-1)
    mask = batch['labels'] != -100
    accuracy = jnp.sum((predictions == batch['labels']) * mask) / jnp.maximum(jnp.sum(mask), 1)
    
    return {'loss': loss, 'accuracy': accuracy}


# ============================================================================
# 4. EVALUATION ON MTEB (SIMPLIFIED)
# ============================================================================

def evaluate_on_mteb_simplified(model: BERTModel, tokenizer):
    """Simplified MTEB evaluation (semantic similarity)."""
    print("\n" + "=" * 80)
    print("Evaluating on MTEB (Simplified)")
    print("=" * 80)
    
    # Sample sentence pairs for similarity
    sentence_pairs = [
        ("The cat sits on the mat", "A feline rests on a rug"),
        ("I love machine learning", "Deep learning is fascinating"),
        ("The weather is nice today", "It's raining heavily"),
    ]
    
    print("\nComputing sentence embeddings...")
    
    def get_sentence_embedding(text):
        """Get [CLS] embedding for sentence."""
        encoded = tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='np'
        )
        
        input_ids = jnp.array(encoded['input_ids'])
        attention_mask = jnp.array(encoded['attention_mask'])
        
        hidden_states = model(input_ids, attention_mask, train=False)
        # Use [CLS] token embedding
        cls_embedding = hidden_states[0, 0, :]
        
        return cls_embedding
    
    for sent1, sent2 in sentence_pairs:
        emb1 = get_sentence_embedding(sent1)
        emb2 = get_sentence_embedding(sent2)
        
        # Cosine similarity
        similarity = jnp.dot(emb1, emb2) / (
            jnp.linalg.norm(emb1) * jnp.linalg.norm(emb2)
        )
        
        print(f"\nSentence 1: {sent1}")
        print(f"Sentence 2: {sent2}")
        print(f"Similarity: {similarity:.4f}")
    
    print("\n✓ Evaluation complete")
    print("\nNote: This is a simplified evaluation.")
    print("For full MTEB: https://github.com/embeddings-benchmark/mteb")


# ============================================================================
# 5. MAIN TRAINING LOOP
# ============================================================================

def main():
    print("=" * 80)
    print("Flax NNX: Train BERT on FineWeb, Evaluate on MTEB")
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
        # Model
        'vocab_size': 30522,  # BERT vocab size
        'max_len': 512,
        'd_model': 256,  # Smaller for demo
        'num_layers': 4,  # Smaller for demo
        'num_heads': 4,
        'd_ff': 1024,
        'dropout': 0.1,
        
        # Training
        'batch_size': 8,
        'learning_rate': 5e-5,
        'num_steps': 100,  # Small for demo
        'seed': 42,
    }
    
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # ========================================================================
    # Initialize Tokenizer
    # ========================================================================
    print("\nInitializing BERT tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    print(f"✓ Tokenizer loaded (vocab size: {len(tokenizer)})")
    
    # ========================================================================
    # Initialize Model
    # ========================================================================
    print("\nInitializing BERT model...")
    rngs = nnx.Rngs(config['seed'])
    model = BERTModel(
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
    
    # ========================================================================
    # Initialize Optimizer
    # ========================================================================
    optimizer = nnx.Optimizer(model, optax.adam(config['learning_rate']))
    
    # ========================================================================
    # Load Data
    # ========================================================================
    dataloader = load_fineweb_streaming(
        tokenizer,
        max_length=config['max_len'],
        batch_size=config['batch_size']
    )
    
    # ========================================================================
    # Training Loop
    # ========================================================================
    print("\n" + "=" * 80)
    print("Training BERT on FineWeb")
    print("=" * 80)
    
    running_loss = 0.0
    running_acc = 0.0
    start_time = time.time()
    
    for step, batch in enumerate(dataloader):
        if step >= config['num_steps']:
            break
        
        # Training step
        metrics = train_step(model, optimizer, batch)
        
        running_loss += float(metrics['loss'])
        running_acc += float(metrics['accuracy'])
        
        # Log every 10 steps
        if (step + 1) % 10 == 0:
            avg_loss = running_loss / 10
            avg_acc = running_acc / 10
            elapsed = time.time() - start_time
            steps_per_sec = 10 / elapsed
            
            print(f"Step {step + 1}/{config['num_steps']} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Acc: {avg_acc:.4f} | "
                  f"Speed: {steps_per_sec:.2f} steps/s")
            
            running_loss = 0.0
            running_acc = 0.0
            start_time = time.time()
    
    print("\n✓ Training complete!")
    
    # ========================================================================
    # Evaluation
    # ========================================================================
    evaluate_on_mteb_simplified(model, tokenizer)
    
    # ========================================================================
    # Best Practices
    # ========================================================================
    print("\n" + "=" * 80)
    print("Best Practices for BERT Training")
    print("=" * 80)
    
    print("""
    1. Data:
       ✓ Use large, diverse text corpus (FineWeb, C4, etc.)
       ✓ Stream data to handle large datasets
       ✓ Implement proper masking strategy (15% tokens)
       ✓ Use whole word masking for better results
    
    2. Training:
       ✓ Use learning rate warmup (10% of steps)
       ✓ Use weight decay for regularization
       ✓ Batch size: 256-2048 (accumulate gradients if needed)
       ✓ Train for 1M steps for good performance
    
    3. Evaluation:
       ✓ MTEB benchmark for embeddings
       ✓ GLUE for downstream tasks
       ✓ Perplexity on held-out data
       ✓ Test on domain-specific tasks
    
    4. Optimizations:
       ✓ Use mixed precision (bfloat16)
       ✓ Gradient checkpointing for memory
       ✓ Multi-host training for scale
       ✓ Monitor GPU utilization
    
    5. MTEB Evaluation:
       • Classification: Sentiment, topic, etc.
       • Clustering: Document grouping
       • Pair Classification: Semantic similarity
       • Reranking: Information retrieval
       • Retrieval: Question answering, search
       • STS: Semantic textual similarity
       • Summarization: Text summarization quality
    
    For full MTEB evaluation, see:
    https://github.com/embeddings-benchmark/mteb
    """)
    
    print("\n" + "=" * 80)
    print("✓ All examples completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
