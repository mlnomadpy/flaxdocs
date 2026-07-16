---
sidebar_position: 2
title: Sequence-to-Sequence with Attention in Flax NNX
description: "Build an encoder-decoder with cross-attention in Flax NNX, train it on a copy-and-reverse task with teacher forcing, and watch attention align source to target."
keywords: [seq2seq, sequence to sequence, attention, encoder decoder, cross-attention, Flax NNX, JAX, nnx.MultiHeadAttention, teacher forcing, GRU, Bahdanau attention, machine translation]
image: img/docusaurus-social-card.jpg
---

# Sequence-to-Sequence with Attention

**Map one sequence to another.** An encoder reads the source, a decoder writes the target, and cross-attention lets every output step look back at the whole input. This guide builds an encoder-decoder in Flax NNX, trains it with teacher forcing on a copy-and-reverse task, and shows how attention learns the source-to-target alignment.

:::note Prerequisites
This guide assumes you have met recurrent cells and attention already. Read [recurrent networks](/applications/sequence/recurrent-networks) for the GRU encoder/decoder, and [simple transformer](/basics/text/simple-transformer) for how multi-head attention works.
:::

:::tip What you'll learn
- The **encoder-decoder** architecture: compress a source sequence into per-step states, then generate a target from them
- How **cross-attention** forms a context vector $c_t = \sum_i \alpha_{t,i} h_i$ over the encoder states with `nnx.MultiHeadAttention`
- **Teacher forcing** — feeding the ground-truth previous token so the decoder trains in parallel
- Why this is a **conditional source → target** model, unlike a decoder-only GPT
- How to score sequence output with **per-token accuracy** and a token-level cross-entropy
:::

:::info Example Code
See the full implementation: [`examples/sequence/seq2seq_attention.py`](https://github.com/mlnomadpy/flaxdocs/tree/master/examples/sequence/seq2seq_attention.py)
:::

## Why Encoder-Decoder?

A recurrent classifier collapses a whole sequence into a single label. But many problems map a sequence to *another sequence* of possibly different content and length: translation, summarization, speech-to-text. The **encoder-decoder** (seq2seq) design splits the job in two:

- an **encoder** reads the source $x_{1:T}$ and produces a stack of hidden states $h_{1:T}$;
- a **decoder** generates the target $y_{1:U}$ one token at a time, conditioned on those states and on what it has already emitted.

The original seq2seq crammed the entire source into the encoder's *final* state — a fixed-size bottleneck that forgets long inputs. **Attention** removes the bottleneck: at each output step the decoder builds a fresh, weighted read over *all* encoder states.

### The task: copy-and-reverse

Our benchmark maps a source sequence to its reverse:

$$
\text{src} = [a, b, c, d] \quad\longrightarrow\quad \text{tgt} = [d, c, b, a]
$$

It is a clean attention probe: to emit output position $t$ correctly, the decoder must attend to source position $T-1-t$ and copy that token. There is no fixed offset the recurrence can memorize — the alignment reverses across the sequence — so a model that solves it has genuinely learned to *point* with attention. Everything is generated in memory; there is nothing to download.

## Attention: a differentiable lookup

At decoder step $t$ we have a query $q_t$ (from the decoder state) and, for every source position $i$, a key $k_i$ and value $v_i$ (from the encoder states). Attention scores each source position, normalizes with a softmax, and returns the value average:

$$
e_{t,i} = \frac{q_t \cdot k_i}{\sqrt{d}}, \qquad
\alpha_{t,i} = \frac{\exp(e_{t,i})}{\sum_{j} \exp(e_{t,j})}, \qquad
c_t = \sum_{i} \alpha_{t,i}\, v_i
$$

The weights $\alpha_{t,\cdot}$ form a soft, differentiable alignment: a distribution over source positions that says *where to look*. For the reverse task, a well-trained model puts almost all of $\alpha_{t,\cdot}$ on position $T-1-t$. **Multi-head** attention runs $H$ of these in parallel over projected subspaces and concatenates the results, so different heads can track different alignment cues.

Because keys/values come from the encoder and queries come from the decoder, this is **cross-attention** — distinct from the **self-attention** a GPT applies within a single stream.

## Building the model in Flax NNX

### Encoder

The encoder is exactly the recurrent stack from the [recurrent networks](/applications/sequence/recurrent-networks) guide: embed the tokens, scan a GRU across time with `nnx.RNN`, and keep *all* per-step states (not just the last), because the decoder attends over the whole stack.

```python
import jax
import jax.numpy as jnp
from flax import nnx
import optax


class Encoder(nnx.Module):
    def __init__(self, vocab, embed, hidden, *, rngs: nnx.Rngs):
        self.embed = nnx.Embed(vocab, embed, rngs=rngs)
        self.rnn = nnx.RNN(nnx.GRUCell(embed, hidden, rngs=rngs))

    def __call__(self, src):
        h = self.embed(src)   # (B, T)        -> (B, T, embed)
        return self.rnn(h)    # (B, T, embed) -> (B, T, hidden)  encoder states
```

### Decoder with cross-attention

The decoder embeds the (teacher-forced) previous tokens, runs its own GRU to get a per-step decoder state, then uses `nnx.MultiHeadAttention` as **cross-attention**: the *query* is the decoder state, the *keys and values* are the encoder states. The context vector is concatenated with the decoder state and projected to the vocabulary.

```python
class Decoder(nnx.Module):
    def __init__(self, vocab, embed, hidden, num_heads, *, rngs: nnx.Rngs):
        # vocab + 1: the extra id is the <bos> start token fed at step 0.
        self.embed = nnx.Embed(vocab + 1, embed, rngs=rngs)
        self.rnn = nnx.RNN(nnx.GRUCell(embed, hidden, rngs=rngs))
        self.cross_attn = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=hidden,      # query dim (decoder state)
            in_kv_features=hidden,   # key/value dim (encoder states)
            qkv_features=hidden,
            decode=False,
            rngs=rngs,
        )
        self.out = nnx.Linear(2 * hidden, vocab, rngs=rngs)

    def __call__(self, dec_in, enc_states):
        d = self.embed(dec_in)          # (B, T)        -> (B, T, embed)
        d = self.rnn(d)                 # (B, T, embed) -> (B, T, hidden)  decoder states
        # Cross-attention: query = decoder states, key/value = encoder states.
        ctx = self.cross_attn(d, enc_states, enc_states)   # (B, T, hidden)  context c_t
        combined = jnp.concatenate([d, ctx], axis=-1)      # (B, T, 2*hidden)
        return self.out(combined)                          # (B, T, vocab)   logits
```

Passing three arguments to `nnx.MultiHeadAttention(inputs_q, inputs_k, inputs_v)` is what makes it cross-attention: `inputs_q` comes from the decoder while `inputs_k` and `inputs_v` come from the encoder. (Calling it with a single argument would be ordinary self-attention.)

### Wiring it together

```python
class Seq2SeqAttention(nnx.Module):
    def __init__(self, vocab, embed, hidden, num_heads, *, rngs: nnx.Rngs):
        self.encoder = Encoder(vocab, embed, hidden, rngs=rngs)
        self.decoder = Decoder(vocab, embed, hidden, num_heads, rngs=rngs)

    def __call__(self, src, dec_in):
        enc_states = self.encoder(src)              # (B, T, hidden)
        return self.decoder(dec_in, enc_states)     # (B, T_out, vocab)
```

## Teacher forcing and the data

During training we feed the decoder the **ground-truth** previous target token rather than its own prediction — this is *teacher forcing*. It lets the decoder process all steps in one parallel pass and gives a stable learning signal early on. The decoder input is the target shifted right by one, with a special `<bos>` (begin-of-sequence) id in the first slot:

$$
\text{dec\_in} = [\,\langle\text{bos}\rangle,\; y_1,\; y_2,\; \dots,\; y_{U-1}\,]
$$

```python
def make_dataset(synthetic=True, *, n=1024, seq_len=8, vocab=12, seed=0):
    key = jax.random.key(seed)
    src = jax.random.randint(key, (n, seq_len), 0, vocab).astype(jnp.int32)
    tgt = src[:, ::-1]                                    # reversed sequence
    bos = jnp.full((n, 1), vocab, dtype=jnp.int32)        # start-of-sequence id
    dec_in = jnp.concatenate([bos, tgt[:, :-1]], axis=1)  # shift-right teacher forcing
    return {"src": src, "dec_in": dec_in, "tgt": tgt}
```

## The training step

The output is a full sequence of logits `(B, T, vocab)`, so we flatten to `(B*T, vocab)` and score every token position with cross-entropy. Per-token accuracy is the fraction of positions predicted correctly.

```python
from shared.training_utils import compute_accuracy, compute_cross_entropy_loss


@nnx.jit
def train_step(model, optimizer, batch):
    def loss_fn(model):
        logits = model(batch["src"], batch["dec_in"])   # (B, T, vocab)
        B, T, V = logits.shape
        flat_logits = logits.reshape(B * T, V)
        flat_tgt = batch["tgt"].reshape(B * T)
        loss = compute_cross_entropy_loss(flat_logits, flat_tgt)
        acc = compute_accuracy(flat_logits, flat_tgt)    # per-token accuracy
        return loss, acc

    (loss, acc), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    optimizer.update(model, grads)
    return loss, acc
```

Build the model and optimizer explicitly, then loop:

```python
model = Seq2SeqAttention(vocab=12, embed=32, hidden=64, num_heads=4, rngs=nnx.Rngs(0))
optimizer = nnx.Optimizer(model, optax.adam(3e-3), wrt=nnx.Param)

for step in range(...):
    loss, acc = train_step(model, optimizer, batch)
```

## Results / What to Expect

The reverse task is fully learnable with attention. On CPU, per-token accuracy climbs past 90% within a dozen epochs and reaches **100%** as the cross-attention locks onto the reversed alignment:

```console
$ python sequence/seq2seq_attention.py
seq2seq+attention  samples=1024 seq_len=8 vocab=12 epochs=30 batch=128
epoch  0  loss 2.3614  token_acc 0.1953
epoch  5  loss 1.6474  token_acc 0.4102
epoch  8  loss 0.7278  token_acc 0.7129
epoch 12  loss 0.1573  token_acc 0.9648
epoch 17  loss 0.0294  token_acc 0.9980
epoch 20  loss 0.0144  token_acc 1.0000
epoch 29  loss 0.0046  token_acc 1.0000
```

Environment knobs `EPOCHS`, `BATCH`, and `SYNTHETIC` let you scale the run; `SYNTHETIC=0` generates a larger, longer (harder) dataset.

## Common Pitfalls

- ❌ Calling `nnx.MultiHeadAttention(x)` with one argument and expecting cross-attention.
  ✅ Cross-attention needs three inputs: `attn(query, key, value)` = `attn(decoder_states, encoder_states, encoder_states)`.

- ❌ Attending over only the encoder's **last** state (the fixed-size bottleneck).
  ✅ Keep *all* per-step states from `nnx.RNN` — attention reads the full `(B, T, hidden)` stack.

- ❌ Feeding the target itself as the decoder input, leaking the answer at each step.
  ✅ Shift right by one and prepend `<bos>`: `dec_in = [<bos>, y_1, ..., y_{U-1}]`.

- ❌ Sizing the output head from `hidden` when you concatenate the context.
  ✅ The decoder state and context are concatenated, so the head is `nnx.Linear(2 * hidden, vocab)`.

- ❌ Setting `qkv_features` so it is not divisible by `num_heads`.
  ✅ Multi-head attention splits `qkv_features` across heads; keep `qkv_features % num_heads == 0` (here `64 % 4 == 0`).

## Next steps

- [Time series](/applications/sequence/time-series) — apply sequence models to forecasting continuous signals.
- [GPT](/architectures/gpt) — drop the encoder and use masked *self*-attention for a decoder-only generative model.

## Complete Example

Full runnable script with the encoder, cross-attention decoder, teacher-forcing data, and the training loop: [`examples/sequence/seq2seq_attention.py`](https://github.com/mlnomadpy/flaxdocs/tree/master/examples/sequence/seq2seq_attention.py).

## References

- Sutskever, Vinyals & Le (2014), *Sequence to Sequence Learning with Neural Networks* — [arXiv:1409.3215](https://arxiv.org/abs/1409.3215).
- Bahdanau, Cho & Bengio (2014), *Neural Machine Translation by Jointly Learning to Align and Translate* — [arXiv:1409.0473](https://arxiv.org/abs/1409.0473).
- Cho et al. (2014), *Learning Phrase Representations using RNN Encoder–Decoder* (GRU) — [arXiv:1406.1078](https://arxiv.org/abs/1406.1078).
- Luong, Pham & Manning (2015), *Effective Approaches to Attention-based Neural Machine Translation* — [arXiv:1508.04025](https://arxiv.org/abs/1508.04025).
- Vaswani et al. (2017), *Attention Is All You Need* — [arXiv:1706.03762](https://arxiv.org/abs/1706.03762).
