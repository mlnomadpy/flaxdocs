---
sidebar_position: 1
title: Recurrent Networks - RNN, LSTM & GRU in Flax NNX
description: "Build RNN, LSTM, GRU and bidirectional recurrent networks in Flax NNX with the nnx.RNN API, master the LSTM gate math, and train on a parity task."
keywords: [recurrent neural network, LSTM, GRU, RNN, Flax NNX, JAX, nnx.RNN, bidirectional RNN, sequence modeling, vanishing gradient, nnx.scan, carry]
image: img/docusaurus-social-card.jpg
---

# Recurrent Networks: RNN, LSTM & GRU

**Teach a network to remember.** Recurrent networks process a sequence one step at a time, carrying a hidden state forward so that earlier tokens can influence later predictions. This guide builds vanilla RNN, LSTM, GRU, and bidirectional classifiers in Flax NNX, derives the LSTM gate equations, and trains them on a parity task that *cannot* be solved without integrating information across every time step.

:::note Prerequisites
Comfortable defining modules and running a training loop? You should have read [your first model](/basics/fundamentals/your-first-model), [understanding state](/basics/fundamentals/understanding-state), and [simple training](/basics/workflows/simple-training). Recurrent layers are stateful, so the state page is especially relevant.
:::

:::tip What you'll learn
- How `nnx.RNN` turns any recurrent **cell** into a layer that scans over time
- The **LSTM gate equations** — forget, input, output — and the two-part carry `(h, c)`
- Why **gating fixes vanishing gradients** where a vanilla RNN fails
- How to swap in `nnx.GRUCell`, `nnx.SimpleCell`, and `nnx.Bidirectional`
- What `nnx.RNN` does under the hood, written out as an explicit `nnx.scan` with **carry threading**
:::

:::info Example Code
See the full implementation: [`examples/sequence/rnn_cells.py`](https://github.com/mlnomadpy/flaxdocs/tree/master/examples/sequence/rnn_cells.py)
:::

## Why Recurrence?

An MLP or CNN sees a fixed-size input all at once. Sequences — text, audio, sensor streams, time series — are different: they have variable length and their *order* carries meaning. A recurrent network processes the sequence step by step, maintaining a **hidden state** $h_t$ that summarizes everything seen so far:

$$
h_t = f(x_t, h_{t-1})
$$

The same cell $f$ is applied at every step, sharing parameters across time. That single recurrence is the whole idea — but making it *learn* long-range dependencies is where LSTMs and GRUs come in.

### The task: parity

Our benchmark is deliberately unforgiving. Given a sequence of integer tokens, predict the **parity** of their sum:

$$
y = \left(\sum_{t=1}^{T} x_t\right) \bmod 2 \in \{0, 1\}
$$

Parity is a perfect recurrent stress test: flipping a *single* token flips the label, so the network must carry an exact running state across the whole sequence. There is no shortcut — a model that ignores any time step cannot beat 50%.

## The vanishing gradient problem

A vanilla RNN computes $h_t = \tanh(W x_t + U h_{t-1} + b)$. Backpropagating the loss at step $T$ to step $t$ multiplies many Jacobians together:

$$
\frac{\partial h_T}{\partial h_t} = \prod_{k=t+1}^{T} \frac{\partial h_k}{\partial h_{k-1}}
= \prod_{k=t+1}^{T} \operatorname{diag}\big(\tanh'(\cdot)\big)\, U^\top
$$

Because $\tanh' \le 1$ and $U$ typically has spectral radius below 1, this product shrinks **exponentially** with $T - t$. Early-step gradients vanish, and the network never learns long-range dependencies. (When the norm is instead above 1, the same product *explodes*.)

## The LSTM: gating to the rescue

The Long Short-Term Memory cell adds a second piece of state — the **cell state** $c_t$ — and three sigmoid **gates** that decide what to keep, write, and read. The full carry is the pair $(h_t, c_t)$.

$$
\begin{aligned}
f_t &= \sigma(W_f x_t + U_f h_{t-1} + b_f) && \text{forget gate} \\
i_t &= \sigma(W_i x_t + U_i h_{t-1} + b_i) && \text{input gate} \\
o_t &= \sigma(W_o x_t + U_o h_{t-1} + b_o) && \text{output gate} \\
\tilde{c}_t &= \tanh(W_c x_t + U_c h_{t-1} + b_c) && \text{candidate} \\
c_t &= f_t \odot c_{t-1} + i_t \odot \tilde{c}_t && \text{cell update} \\
h_t &= o_t \odot \tanh(c_t) && \text{hidden output}
\end{aligned}
$$

Here $\sigma$ is the logistic sigmoid, $\odot$ is elementwise multiplication, and each gate is a vector in $(0, 1)$.

**Why this fixes vanishing gradients.** The cell update is *additive*: $c_t = f_t \odot c_{t-1} + \dots$. The gradient flowing back through the cell state is

$$
\frac{\partial c_t}{\partial c_{t-1}} = \operatorname{diag}(f_t)
$$

When the forget gate stays near 1, this is close to the identity, so gradients travel across many steps **without shrinking** — a "constant error carousel." The network *learns* how long to remember by controlling $f_t$, instead of being forced to forget by the geometry of repeated matrix products.

## Building the model in Flax NNX

Flax NNX separates the **cell** (one step of recurrence) from the **layer** (the scan over time). `nnx.RNN` wraps any cell and applies it across the time axis of a `(B, T, features)` input, returning per-step outputs `(B, T, hidden)`. We embed integer tokens, run the RNN, and classify from the *last* step.

```python
import jax
import jax.numpy as jnp
from flax import nnx
import optax


class LSTMClassifier(nnx.Module):
    def __init__(self, vocab, embed, hidden, n_classes, *, rngs: nnx.Rngs):
        self.embed = nnx.Embed(vocab, embed, rngs=rngs)
        self.rnn = nnx.RNN(nnx.LSTMCell(embed, hidden, rngs=rngs))
        self.head = nnx.Linear(hidden, n_classes, rngs=rngs)

    def __call__(self, tokens):
        h = self.embed(tokens)      # (B, T)        -> (B, T, embed)
        h = self.rnn(h)             # (B, T, embed) -> (B, T, hidden)
        return self.head(h[:, -1])  # last step     -> (B, n_classes)
```

### Swapping the cell: GRU and vanilla RNN

The cell is the only thing that changes. A **GRU** (`nnx.GRUCell`) merges the forget and input gates into a single update gate and carries just `h` — fewer parameters, often comparable accuracy. A **vanilla RNN** (`nnx.SimpleCell`) has no gates at all and is our vanishing-gradient baseline.

```python
class GRUClassifier(nnx.Module):
    def __init__(self, vocab, embed, hidden, n_classes, *, rngs: nnx.Rngs):
        self.embed = nnx.Embed(vocab, embed, rngs=rngs)
        self.rnn = nnx.RNN(nnx.GRUCell(embed, hidden, rngs=rngs))   # gated, single carry h
        self.head = nnx.Linear(hidden, n_classes, rngs=rngs)

    def __call__(self, tokens):
        h = self.embed(tokens)
        h = self.rnn(h)
        return self.head(h[:, -1])


class VanillaRNNClassifier(nnx.Module):
    def __init__(self, vocab, embed, hidden, n_classes, *, rngs: nnx.Rngs):
        self.embed = nnx.Embed(vocab, embed, rngs=rngs)
        self.rnn = nnx.RNN(nnx.SimpleCell(embed, hidden, rngs=rngs))  # no gates
        self.head = nnx.Linear(hidden, n_classes, rngs=rngs)

    def __call__(self, tokens):
        h = self.embed(tokens)
        h = self.rnn(h)
        return self.head(h[:, -1])
```

### Bidirectional: reading both ways

For tasks where the *whole* sequence is available (classification, tagging), a **bidirectional** RNN runs one cell forward and another backward, then concatenates their outputs. Each position now sees both past and future context. `nnx.Bidirectional` handles the reversal, scan, and merge; the classifier head just needs `2 * hidden` inputs.

```python
class BiLSTMClassifier(nnx.Module):
    def __init__(self, vocab, embed, hidden, n_classes, *, rngs: nnx.Rngs):
        self.embed = nnx.Embed(vocab, embed, rngs=rngs)
        forward = nnx.RNN(nnx.LSTMCell(embed, hidden, rngs=rngs))
        backward = nnx.RNN(nnx.LSTMCell(embed, hidden, rngs=rngs))
        self.birnn = nnx.Bidirectional(forward, backward)          # concatenates fwd ++ bwd
        self.head = nnx.Linear(2 * hidden, n_classes, rngs=rngs)

    def __call__(self, tokens):
        h = self.embed(tokens)      # (B, T, embed)
        h = self.birnn(h)           # (B, T, 2*hidden)
        return self.head(h[:, -1])
```

## Under the hood: carry threading with `nnx.scan`

`nnx.RNN` is a convenience wrapper around a `scan`. Writing that scan by hand is the clearest way to understand **carry threading** — the mechanism that passes the recurrent state from one step to the next.

The pattern: initialize the carry with `cell.initialize_carry`, then decorate a per-step function with `@nnx.scan`. The special `nnx.Carry` marker in `in_axes`/`out_axes` says "this argument is threaded, not sliced"; the integer `1` says "slice/stack this argument along the time axis."

```python
def manual_lstm_scan(cell: nnx.LSTMCell, sequence):
    batch = sequence.shape[0]
    in_features = sequence.shape[-1]
    carry = cell.initialize_carry((batch, in_features), nnx.Rngs(0))  # (h, c), each (B, hidden)

    @nnx.scan(in_axes=(nnx.Carry, 1), out_axes=(nnx.Carry, 1))
    def step(carry, x_t):
        carry, y_t = cell(carry, x_t)   # ((h, c), x_t) -> ((h, c), y_t)
        return carry, y_t

    carry, outputs = step(carry, sequence)  # carry: final (h, c); outputs: (B, T, hidden)
    return carry, outputs
```

This produces bit-for-bit the same per-step outputs as `nnx.RNN(cell)(sequence)` — the wrapper is exactly this scan plus some ergonomics (masking, reversal, time-major handling). For an LSTM the carry is the tuple `(h, c)`; for a GRU or vanilla RNN it is just `h`.

## The training step

Standard NNX training: cross-entropy on the last-step logits, `nnx.value_and_grad` with `has_aux=True` to also return logits for the accuracy metric, and `optimizer.update(model, grads)`.

```python
from shared.training_utils import compute_accuracy, compute_cross_entropy_loss


@nnx.jit
def train_step(model, optimizer, batch):
    def loss_fn(model):
        logits = model(batch["x"])
        loss = compute_cross_entropy_loss(logits, batch["y"])
        return loss, logits

    (loss, logits), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    optimizer.update(model, grads)
    acc = compute_accuracy(logits, batch["y"])
    return loss, acc
```

Build the model and optimizer explicitly, then loop:

```python
model = LSTMClassifier(vocab=10, embed=32, hidden=64, n_classes=2, rngs=nnx.Rngs(0))
optimizer = nnx.Optimizer(model, optax.adam(1e-2), wrt=nnx.Param)

for step in range(30):
    loss, acc = train_step(model, optimizer, {"x": x_batch, "y": y_batch})
```

The dataset is generated on the fly — no downloads:

```python
key = jax.random.key(0)
x = jax.random.randint(key, (512, 12), 0, 10)   # tokens in [0, 10)
y = (x.sum(axis=1) % 2).astype(jnp.int32)        # parity label
```

## Results / What to Expect

Parity is fully learnable by a gated recurrent network. On CPU, the LSTM drives cross-entropy toward zero and hits **100% accuracy** within ~30 steps:

```console
$ python sequence/rnn_cells.py
model=lstm samples=512 seq_len=12 epochs=40 batch=128
epoch  0  loss 0.7112  acc 0.4844
epoch  4  loss 0.6722  acc 0.5938
epoch  9  loss 0.5305  acc 0.7422
epoch 19  loss 0.0337  acc 0.9922
epoch 39  loss 0.0006  acc 1.0000
manual scan outputs (4, 12, 64)  carry h (4, 64)  c (4, 64)
```

Try `MODEL=gru` or `MODEL=bilstm` (both solve it), and `MODEL=rnn` for the vanilla baseline — the `nnx.SimpleCell` struggles as you push `seq_len` higher, exactly the vanishing-gradient behavior the gates were designed to cure. Environment knobs `EPOCHS`, `BATCH`, and `SYNTHETIC` let you scale the run.

## Common Pitfalls

- ❌ Feeding `(B, features)` to `nnx.RNN` and wondering why it errors.
  ✅ `nnx.RNN` expects a **time axis**: shape `(B, T, features)`. Embed first, then scan.

- ❌ Classifying from `h[:, 0]` or averaging all steps for a task that needs the full sequence.
  ✅ For parity, use the **last step** `h[:, -1]` (or a bidirectional final state) so the state has seen every token.

- ❌ Building `LSTMCell(embed, hidden)` but wiring the head from `embed`.
  ✅ The RNN outputs **`hidden`**-dim vectors; the head is `nnx.Linear(hidden, n_classes)` (and `2 * hidden` for bidirectional).

- ❌ Expecting a vanilla `nnx.SimpleCell` to match the LSTM on long sequences.
  ✅ Gates exist precisely to carry gradients across time; reach for `LSTMCell`/`GRUCell` when dependencies are long.

- ❌ Putting a Python list of cells in a plain attribute to stack RNN layers.
  ✅ On Flax 0.12 wrap submodule lists in `nnx.List([...])` (and dicts in `nnx.Dict({...})`) so they register as state.

## Next steps

- [Simple Transformer](/basics/text/simple-transformer) — attention replaces recurrence and parallelizes across time.
- [Graph Neural Networks](/applications/scientific/graph-neural-networks) — message passing generalizes recurrence to arbitrary graphs.

## Complete Example

Full runnable script with all four model variants, the manual scan, and the training loop: [`examples/sequence/rnn_cells.py`](https://github.com/mlnomadpy/flaxdocs/tree/master/examples/sequence/rnn_cells.py).

## References

- Hochreiter & Schmidhuber (1997), *Long Short-Term Memory* — [Neural Computation 9(8)](https://doi.org/10.1162/neco.1997.9.8.1735).
- Cho et al. (2014), *Learning Phrase Representations using RNN Encoder–Decoder* (GRU) — [arXiv:1406.1078](https://arxiv.org/abs/1406.1078).
- Pascanu, Mikolov & Bengio (2013), *On the Difficulty of Training Recurrent Neural Networks* — [arXiv:1211.5063](https://arxiv.org/abs/1211.5063).
- Chung et al. (2014), *Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling* — [arXiv:1412.3555](https://arxiv.org/abs/1412.3555).
- Greff et al. (2015), *LSTM: A Search Space Odyssey* — [arXiv:1503.04069](https://arxiv.org/abs/1503.04069).
