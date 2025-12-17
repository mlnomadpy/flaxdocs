---
sidebar_position: 2
---

# BERT (Bidirectional Transformers)

BERT inputs vectors representing: **Token + Position + Sentence**.
$$E = E_{token} + E_{pos} + E_{seg}$$

## Self-Attention

$$ \text{Attn}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V $$

### 1. Projections

We split the hidden size (768) into 12 heads of size 64.

```python
from flax import linen as nnx
import jax.numpy as jnp
import jax

class BERTAttention(nnx.Module):
    def __init__(self, hidden_size: int = 768, num_heads: int = 12, *, rngs: nnx.Rngs):
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.query = nnx.Linear(hidden_size, hidden_size, rngs=rngs)
        self.key = nnx.Linear(hidden_size, hidden_size, rngs=rngs)
        self.value = nnx.Linear(hidden_size, hidden_size, rngs=rngs)
        self.out = nnx.Linear(hidden_size, hidden_size, rngs=rngs)
```

### 2. Multi-Head Split

In `__call__`, we project input $x$ and reshape to separate heads.
Transposing $(B, L, \text{Heads}, \text{Dim}) \to (B, \text{Heads}, L, \text{Dim})$ allows parallel computation.

```python
    def __call__(self, x):
        B, L, H = x.shape
        
        # Project & Reshape
        q = self.query(x).reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.key(x).reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.value(x).reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
```

### 3. Scores & Softmax

We calculate similarity ($Q \cdot K^T$) and scale by $\sqrt{d}$ to prevent vanishing gradients in softmax.

```python
        # Scaled Dot Product
        scores = jnp.matmul(q, k.swapaxes(-1, -2)) / jnp.sqrt(self.head_dim)
        
        # Probabilities
        weights = jax.nn.softmax(scores, axis=-1)
        
        # Weighted Sum
        context = jnp.matmul(weights, v)
        
        # Recombine Heads
        context = context.transpose(0, 2, 1, 3).reshape(B, L, H)
        return self.out(context)
```

## Masked Language Modeling (MLM)

We aim to predict 15% of invisible words.

### 4. Preparation Logic

We need a function that randomly corrupts inputs.
*   **Mask**: 15% of tokens.
*   **Labels**: Set everything to `-100` except the masked tokens.

```python
import numpy as np

def prepare_batch(examples, tokenizer):
    # 1. Tokenize
    encodings = tokenizer(examples['text'], truncation=True, padding='max_length', 
                        max_length=128, return_tensors='np')
    input_ids = encodings['input_ids']
    
    # 2. Select 15%
    rand = np.random.rand(*input_ids.shape)
    mask = (rand < 0.15) & (input_ids != tokenizer.pad_token_id)
    
    # 3. Create Labels (Ignore unmasked)
    labels = input_ids.copy()
    labels[~mask] = -100
    
    # 4. Corrupt Input with [MASK]
    input_ids[mask] = tokenizer.mask_token_id
    
    return {'input_ids': input_ids, 'labels': labels, 'attention_mask': encodings['attention_mask']}
```

### 5. Loading the Stream

We apply this function to the HuggingFace stream.

```python
from datasets import load_dataset
from transformers import AutoTokenizer

dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
dataset = dataset.shuffle(buffer_size=10_000)
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Apply Mapping
train_loader = dataset.map(
    lambda x: prepare_batch(x, tokenizer), 
    batched=True, 
    batch_size=32, 
    remove_columns=['text']
)
iterator = iter(train_loader)
```

## Training

### 6. The Masked Loss

Standard Cross Entropy averages *all* tokens. We must filter for only the masked ones (where `label != -100`).

```python
import optax
from flax.training import train_state

@jax.jit
def train_step(state, batch):
    def loss_fn(params):
        logits = state.apply_fn(
            {'params': params, 
             'batch_stats': state.batch_stats if hasattr(state, 'batch_stats') else {}}, # Handle optional BN
            batch['input_ids'], 
            batch['attention_mask'] if 'attention_mask' in batch else None,
            train=True
        )
        # Note: Simplified call signature for demo. Real BERT takes attention_mask.
        
        # 1. Raw Loss (includes ignore_index positions)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, batch['labels'])
        
        # 2. Filter: Zero out non-masked positions
        mask = batch['labels'] != -100
        loss = (loss * mask).sum() / (mask.sum() + 1e-6)
        
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss
```

### 7. Execution

```python
# Setup
# (Assuming BERT class is defined similar to GPT but with bidirectional attention)
class BERT(nnx.Module):
    def __init__(self, vocab=30522, *, rngs):
        self.emb = nnx.Embed(vocab, 768, rngs=rngs)
        self.enc = BERTAttention(rngs=rngs)
        self.head = nnx.Linear(768, vocab, rngs=rngs)
    def __call__(self, x, mask=None, train=True):
        x = self.emb(x)
        x = self.enc(x)
        return self.head(x)

model = BERT(rngs=nnx.Rngs(0))
vars = model.init(nnx.Rngs(0), jnp.ones((1, 128), dtype=int))

state = train_state.TrainState.create(
    apply_fn=model.apply,
    params=vars['params'],
    tx=optax.adamw(1e-4),
)

# Run
print("Training...")
for step in range(100):
    try:
        data = next(iterator)
        batch = {k: jnp.array(v) for k, v in data.items()}
        state, loss = train_step(state, batch)
        if step % 10 == 0: print(f"Loss: {loss:.4f}")
    except StopIteration:
        break
```
