---
sidebar_position: 3
---

# GPT (Generative Pre-trained Transformer)

GPT is an **Autoregressive** (or Causal) Language Model. Unlike BERT, which sees the whole sentence, GPT processes text left-to-right. Its goal is simple: **Predict the next word.**

## 1. Causal Attention: Looking back, not forward

Standard attention allows every token to look at every other token ($N \times N$). We must blindfold future tokens using a **Causal Mask** (Lower Triangular Matrix).

$$
\text{Mask} = \begin{bmatrix} 
0 & -\infty & -\infty \\
0 & 0 & -\infty \\
0 & 0 & 0 
\end{bmatrix}
$$

### Implementation (Cell 1)

```python
from flax import linen as nnx
import jax.numpy as jnp
import jax

class GPTAttention(nnx.Module):
    # ... Init similar to BERT ...
    def __init__(self, hidden=768, heads=12, *, rngs):
        self.heads = heads
        self.dim = hidden // heads
        self.qkv = nnx.Linear(hidden, hidden*3, rngs=rngs)
        self.out = nnx.Linear(hidden, hidden, rngs=rngs)

    def __call__(self, x):
        B, L, H = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.heads, self.dim)
        qkv = qkv.transpose(2, 0, 3, 1, 4) # (3, B, Heads, L, Dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Content Scores
        scores = jnp.matmul(q, k.swapaxes(-1, -2)) / jnp.sqrt(self.dim)
        
        # --- Causal Mask ---
        # 1. Create Ones: [[1, 1], [1, 1]]
        # 2. Keep Lower Triangle: [[1, 0], [1, 1]]
        mask = jnp.tril(jnp.ones((L, L)))
        # 3. Apply Mask: Where 0, become -infinity
        scores = jnp.where(mask == 1, scores, -1e9)
        
        weights = jax.nn.softmax(scores, axis=-1)
        return self.out(jnp.matmul(weights, v).transpose(0, 2, 1, 3).reshape(B, L, H))
```

## 2. The Training Trick: Shifting

We feed the **entire sequence** simultaneously.
- **Input**:  `[A, B, C]`
- **Target**: `[B, C, D]`

### Data Setup (Cell 2)

```python
from datasets import load_dataset
from transformers import AutoTokenizer

dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
tokenizer = AutoTokenizer.from_pretrained('gpt2')

def tokenize(ex):
    # We need sequence length + 1 (for the shift)
    return tokenizer(ex['text'], truncation=True, max_length=129, return_overflowing_tokens=True)

dataset = dataset.map(tokenize, batched=True, remove_columns=['text'])
iterator = iter(dataset)
```

## 3. Training Loop

The loss function handles the shifting logic.

### Step Definition (Cell 3)
```python
import optax
from flax.training import train_state

@jax.jit
def train_step(state, batch):
    # Batch is just input_ids: [Batch, 129]
    
    # Input: 0 to T-1
    inputs = batch[:, :-1]
    # Target: 1 to T
    targets = batch[:, 1:]
    
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, inputs, train=True)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss
```

### Execution (Cell 4)
```python
# Models
class GPT(nnx.Module):
    # Simplified GPT Shell
    def __init__(self, vocab=50257, *, rngs):
        self.emb = nnx.Embed(vocab, 768, rngs=rngs)
        self.pos = nnx.Embed(1024, 768, rngs=rngs)
        self.attn = GPTAttention(rngs=rngs)
        self.head = nnx.Linear(768, vocab, rngs=rngs)
    def __call__(self, x, train=True):
        pos = jnp.arange(x.shape[1])[None, :]
        x = self.emb(x) + self.pos(pos)
        x = self.attn(x)
        return self.head(x)

# Init
model = GPT(rngs=nnx.Rngs(0))
x = jnp.ones((1, 128), dtype=jnp.int32)
vars = model.init(nnx.Rngs(0), x)

state = train_state.TrainState.create(
    apply_fn=model.apply,
    params=vars['params'],
    tx=optax.adamw(3e-4),
)

# Run
print("Training GPT...")
for step in range(100):
    try:
        # Get raw batch (simulated batching)
        batch = [next(iterator)['input_ids'] for _ in range(8)]
        # Pad/Stack logic would go here, simplified to direct stack
        batch = jnp.array(batch)[:, :129] 
        
        state, loss = train_step(state, batch)
        if step % 10 == 0:
            print(f"Step {step} | Loss {loss:.4f}")
    except StopIteration:
        break
```

## 4. Sampling (Making it talk)

To generate text, we iterate one token at a time with optional Temperature.

```python
def generate(state, prompt, temp=0.7):
    input_ids = tokenizer.encode(prompt, return_tensors='np')
    
    for _ in range(20):
        # Forward
        logits = state.apply_fn({'params': state.params}, input_ids)
        # Scale
        next_logits = logits[0, -1, :] / temp
        # Sample
        probs = jax.nn.softmax(next_logits)
        next_id = jax.random.categorical(jax.random.PRNGKey(0), jnp.log(probs))
        # Append
        input_ids = jnp.concatenate([input_ids, next_id[None, None]], axis=1)
    
    return tokenizer.decode(input_ids[0])
```

## Limitations & Evolution

GPT models dominate today (ChatGPT, Claude), but they are not perfect:

1.  **Hallucinations**: The model predicts the *most likely* next word, not the *true* one. It can confidently state facts that are statistically probable but factually wrong.
    *   *Evolution*: **RLHF (Reinforcement Learning from Human Feedback)** aligns the model's objective with human truthfulness and safety, not just probability.
2.  **Memory Bandwidth (KV Cache)**: During generation, we must store previous Keys/Values to avoid re-computing them. For long contexts (100k+ tokens), this cache becomes massive.
    *   *Evolution*: **Grouped Query Attention (GQA)** and **Multi-Query Attention (MQA)** share heads to drastically reduce memory footprint.
3.  **Context Limited**: Standard attention limits context windows.
    *   *Evolution*: **Ring Attention** and **RoPE (Rotary Positional Embeddings)** allow training on millions of tokens by distributing the sequence itself across GPUs.

