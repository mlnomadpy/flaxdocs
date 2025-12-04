---
sidebar_position: 2
---

# Streaming Data and Advanced Architectures

Learn how to train on datasets larger than memory using streaming, and build production-grade ResNets, BERT, and GPT models from scratch.

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
for batch in create_batches(dataset, batch_size=32):
    # batch['input_ids']: (32, 512)
    loss = train_step(model, optimizer, batch)
```

## Training ResNet on ImageNet

ResNets use residual connections to train very deep networks. Let's understand the architecture:

### ResNet Building Blocks

**The Core Insight**: Skip connections allow gradients to flow directly through the network.

```python
class ResidualBlock(nnx.Module):
    """Basic residual block: out = F(x) + x"""
    
    def __init__(
        self,
        features: int,
        stride: int = 1,
        *,
        rngs: nnx.Rngs
    ):
        # Main path: two 3x3 convolutions
        self.conv1 = nnx.Conv(
            in_features=features,
            out_features=features,
            kernel_size=(3, 3),
            strides=(stride, stride),
            padding='SAME',
            use_bias=False,
            rngs=rngs
        )
        self.bn1 = nnx.BatchNorm(features, rngs=rngs)
        
        self.conv2 = nnx.Conv(
            in_features=features,
            out_features=features,
            kernel_size=(3, 3),
            padding='SAME',
            use_bias=False,
            rngs=rngs
        )
        self.bn2 = nnx.BatchNorm(features, rngs=rngs)
        
        # Shortcut path: identity or projection
        if stride != 1:
            # Need to downsample skip connection
            self.shortcut = nnx.Sequential(
                nnx.Conv(
                    in_features=features,
                    out_features=features,
                    kernel_size=(1, 1),
                    strides=(stride, stride),
                    use_bias=False,
                    rngs=rngs
                ),
                nnx.BatchNorm(features, rngs=rngs)
            )
        else:
            # Identity shortcut
            self.shortcut = lambda x, train: x
    
    def __call__(self, x, *, train: bool = True):
        # Main path
        out = self.conv1(x)
        out = self.bn1(out, use_running_average=not train)
        out = nnx.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out, use_running_average=not train)
        
        # Skip connection
        identity = self.shortcut(x, train=train) if callable(self.shortcut) else self.shortcut(x)
        
        # Add and activate
        out = out + identity
        out = nnx.relu(out)
        
        return out
```

### Understanding Residual Connections

**Why they work**:
```
Traditional: out = F(x)
- Must learn entire transformation
- Gradients get weaker through layers

Residual: out = F(x) + x
- Only needs to learn the "residue" (difference)
- Gradient flows directly through skip connection
- Can learn identity easily: just make F(x) = 0
```

### Complete ResNet Architecture

```python
class ResNet(nnx.Module):
    """ResNet architecture for ImageNet"""
    
    def __init__(
        self,
        num_classes: int = 1000,
        layers: list[int] = [2, 2, 2, 2],  # ResNet-18
        *,
        rngs: nnx.Rngs
    ):
        # Stem: Initial downsampling
        self.conv1 = nnx.Conv(
            in_features=3,  # RGB
            out_features=64,
            kernel_size=(7, 7),
            strides=(2, 2),
            padding='SAME',
            use_bias=False,
            rngs=rngs
        )
        self.bn1 = nnx.BatchNorm(64, rngs=rngs)
        
        # 4 stages with increasing channels
        self.layer1 = self._make_layer(64, 64, layers[0], stride=1, rngs=rngs)
        self.layer2 = self._make_layer(64, 128, layers[1], stride=2, rngs=rngs)
        self.layer3 = self._make_layer(128, 256, layers[2], stride=2, rngs=rngs)
        self.layer4 = self._make_layer(256, 512, layers[3], stride=2, rngs=rngs)
        
        # Classification head
        self.fc = nnx.Linear(512, num_classes, rngs=rngs)
    
    def _make_layer(self, in_features, out_features, num_blocks, stride, rngs):
        """Create a stack of residual blocks"""
        layers = []
        
        # First block may downsample
        layers.append(ResidualBlock(out_features, stride=stride, rngs=rngs))
        
        # Rest are identity stride
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_features, stride=1, rngs=rngs))
        
        return layers
    
    def __call__(self, x, *, train: bool = True):
        # Stem
        x = self.conv1(x)
        x = self.bn1(x, use_running_average=not train)
        x = nnx.relu(x)
        x = nnx.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding='SAME')
        
        # Stages
        for block in self.layer1:
            x = block(x, train=train)
        for block in self.layer2:
            x = block(x, train=train)
        for block in self.layer3:
            x = block(x, train=train)
        for block in self.layer4:
            x = block(x, train=train)
        
        # Global average pooling
        x = jnp.mean(x, axis=(1, 2))
        
        # Classification
        return self.fc(x)
```

### ResNet Variants

- **ResNet-18**: [2, 2, 2, 2] blocks = 18 layers
- **ResNet-34**: [3, 4, 6, 3] = 34 layers
- **ResNet-50**: [3, 4, 6, 3] with bottleneck blocks = 50 layers
- **ResNet-101**: [3, 4, 23, 3] = 101 layers

Deeper = better accuracy but slower training.

## BERT: Bidirectional Language Understanding

BERT reads text in both directions to understand context.

### BERT Architecture Concepts

**Key innovations**:
1. **Bidirectional attention**: Each token sees all tokens (unlike GPT's causal)
2. **Masked language modeling**: Predict masked words
3. **Pre-training then fine-tuning**: Learn general language, adapt to tasks

### Building BERT Layers

```python
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
        
        # Reshape for multi-head
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        
        q = jnp.transpose(q, (0, 2, 1, 3))  # (batch, heads, seq, head_dim)
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))
        
        # Attention scores
        scores = (q @ jnp.swapaxes(k, -2, -1)) / jnp.sqrt(self.head_dim)
        
        # Apply mask (for padding tokens)
        if attention_mask is not None:
            scores = scores + attention_mask
        
        # Softmax and dropout
        attn_weights = jax.nn.softmax(scores, axis=-1)
        attn_weights = self.dropout(attn_weights, deterministic=not train)
        
        # Weighted sum
        context = attn_weights @ v
        
        # Reshape and project
        context = jnp.transpose(context, (0, 2, 1, 3))
        context = context.reshape(batch_size, seq_len, hidden_size)
        
        return self.out(context)

class BERTLayer(nnx.Module):
    """Complete BERT transformer layer"""
    
    def __init__(
        self,
        hidden_size: int = 768,
        num_heads: int = 12,
        intermediate_size: int = 3072,
        *,
        rngs: nnx.Rngs
    ):
        # Self-attention
        self.attention = BERTAttention(hidden_size, num_heads, rngs=rngs)
        
        # Feed-forward network
        self.intermediate = nnx.Linear(hidden_size, intermediate_size, rngs=rngs)
        self.output = nnx.Linear(intermediate_size, hidden_size, rngs=rngs)
        
        # Layer norm and dropout
        self.ln1 = nnx.LayerNorm(hidden_size, rngs=rngs)
        self.ln2 = nnx.LayerNorm(hidden_size, rngs=rngs)
        self.dropout = nnx.Dropout(0.1, rngs=rngs)
    
    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        *,
        train: bool = True
    ):
        # Attention with residual
        attn_output = self.attention(
            self.ln1(hidden_states),
            attention_mask,
            train=train
        )
        hidden_states = hidden_states + self.dropout(attn_output, deterministic=not train)
        
        # FFN with residual
        intermediate = self.intermediate(self.ln2(hidden_states))
        intermediate = nnx.gelu(intermediate)
        ffn_output = self.output(intermediate)
        hidden_states = hidden_states + self.dropout(ffn_output, deterministic=not train)
        
        return hidden_states
```

### Masked Language Modeling

BERT's training objective:

```python
def create_mlm_batch(texts, tokenizer, mask_prob=0.15):
    """Create masked language modeling training batch"""
    
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
    
    # Create mask
    rand = np.random.rand(*input_ids.shape)
    mask = (rand < mask_prob) & (input_ids != tokenizer.pad_token_id)
    
    # Replace masked positions with [MASK] token
    input_ids[mask] = tokenizer.mask_token_id
    
    # Only compute loss on masked positions
    labels[~mask] = -100  # Ignore in loss
    
    return {
        'input_ids': input_ids,
        'attention_mask': encodings['attention_mask'],
        'labels': labels
    }
```

**How it works**:
1. Randomly mask 15% of tokens
2. Model predicts original token
3. Forces bidirectional understanding

## GPT: Autoregressive Language Model

GPT generates text left-to-right, predicting next tokens.

### GPT vs BERT

| Feature | BERT | GPT |
|---------|------|-----|
| Attention | Bidirectional | Causal (unidirectional) |
| Training | Masked LM | Next token prediction |
| Use case | Understanding | Generation |

### GPT Architecture

```python
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
        
        # Combined QKV projection (more efficient)
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

class GPTBlock(nnx.Module):
    """Complete GPT transformer block"""
    
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: int = 4,
        *,
        rngs: nnx.Rngs
    ):
        self.ln1 = nnx.LayerNorm(embed_dim, rngs=rngs)
        self.attn = GPTAttention(embed_dim, num_heads, rngs=rngs)
        
        self.ln2 = nnx.LayerNorm(embed_dim, rngs=rngs)
        self.mlp = nnx.Sequential(
            nnx.Linear(embed_dim, embed_dim * mlp_ratio, rngs=rngs),
            nnx.gelu,
            nnx.Linear(embed_dim * mlp_ratio, embed_dim, rngs=rngs),
            nnx.Dropout(0.1, rngs=rngs),
        )
    
    def __call__(self, x, *, train: bool = True):
        # Pre-norm architecture
        x = x + self.attn(self.ln1(x), train=train)
        x = x + self.mlp(self.ln2(x))
        return x
```

### Text Generation with GPT

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
        next_logits = logits[0, -1, :]  # Last position
        
        # Temperature scaling
        next_logits = next_logits / temperature
        
        # Top-k sampling
        top_k_logits, top_k_indices = jax.lax.top_k(next_logits, k=top_k)
        
        # Sample from top-k
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
    
    # Decode
    return tokenizer.decode(input_ids[0])
```

## Next Steps

You now understand streaming and advanced architectures! Continue learning:
- [Track experiments with Weights & Biases](../basics/workflows/observability.md)
- [Scale to distributed training](../scale/)

## Reference Code

Complete implementations:
- [`09_resnet_streaming_training.py`](https://github.com/mlnomadpy/flaxdocs/tree/master/examples/09_resnet_streaming_training.py) - ResNet with streaming
- [`11_bert_fineweb_mteb.py`](https://github.com/mlnomadpy/flaxdocs/tree/master/examples/11_bert_fineweb_mteb.py) - BERT pretraining
- [`12_gpt_fineweb_training.py`](https://github.com/mlnomadpy/flaxdocs/tree/master/examples/12_gpt_fineweb_training.py) - GPT training from scratch
