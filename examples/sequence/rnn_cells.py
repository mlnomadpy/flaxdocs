"""
Recurrent Networks: RNN / LSTM / GRU / Bidirectional
====================================================
Recurrent classifiers on a synthetic parity task, built with the Flax NNX
`nnx.RNN` / `nnx.LSTMCell` / `nnx.GRUCell` / `nnx.SimpleCell` / `nnx.Bidirectional`
API, plus a hand-written `nnx.scan` loop that shows how carry threading works.

Run: python sequence/rnn_cells.py
"""

import os

import jax
import jax.numpy as jnp
from flax import nnx
import optax

import sys
from pathlib import Path

# Add parent directory to path for shared imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.training_utils import compute_accuracy, compute_cross_entropy_loss


# ============================================================================
# MODELS
# ============================================================================

class LSTMClassifier(nnx.Module):
    """Embed tokens -> LSTM over time -> classify from the last hidden state.

    ``nnx.RNN`` wraps a recurrent cell and scans it across the time axis of a
    ``(B, T, features)`` input, returning the per-step outputs ``(B, T, hidden)``.
    """

    def __init__(self, vocab: int, embed: int, hidden: int, n_classes: int,
                 *, rngs: nnx.Rngs):
        self.embed = nnx.Embed(vocab, embed, rngs=rngs)
        self.rnn = nnx.RNN(nnx.LSTMCell(embed, hidden, rngs=rngs))
        self.head = nnx.Linear(hidden, n_classes, rngs=rngs)

    def __call__(self, tokens):
        h = self.embed(tokens)      # (B, T)      -> (B, T, embed)
        h = self.rnn(h)             # (B, T, embed) -> (B, T, hidden)
        return self.head(h[:, -1])  # last step   -> (B, n_classes)


class GRUClassifier(nnx.Module):
    """Same as LSTMClassifier but with a GRU cell (single gated carry ``h``)."""

    def __init__(self, vocab: int, embed: int, hidden: int, n_classes: int,
                 *, rngs: nnx.Rngs):
        self.embed = nnx.Embed(vocab, embed, rngs=rngs)
        self.rnn = nnx.RNN(nnx.GRUCell(embed, hidden, rngs=rngs))
        self.head = nnx.Linear(hidden, n_classes, rngs=rngs)

    def __call__(self, tokens):
        h = self.embed(tokens)
        h = self.rnn(h)
        return self.head(h[:, -1])


class VanillaRNNClassifier(nnx.Module):
    """Elman/vanilla RNN via ``nnx.SimpleCell`` — no gates, prone to vanishing
    gradients on long sequences. Kept here to contrast against LSTM/GRU."""

    def __init__(self, vocab: int, embed: int, hidden: int, n_classes: int,
                 *, rngs: nnx.Rngs):
        self.embed = nnx.Embed(vocab, embed, rngs=rngs)
        self.rnn = nnx.RNN(nnx.SimpleCell(embed, hidden, rngs=rngs))
        self.head = nnx.Linear(hidden, n_classes, rngs=rngs)

    def __call__(self, tokens):
        h = self.embed(tokens)
        h = self.rnn(h)
        return self.head(h[:, -1])


class BiLSTMClassifier(nnx.Module):
    """Bidirectional LSTM. ``nnx.Bidirectional`` runs a forward and a backward
    RNN and concatenates their outputs, so the head sees ``2 * hidden`` features
    that summarise context from both directions."""

    def __init__(self, vocab: int, embed: int, hidden: int, n_classes: int,
                 *, rngs: nnx.Rngs):
        self.embed = nnx.Embed(vocab, embed, rngs=rngs)
        forward = nnx.RNN(nnx.LSTMCell(embed, hidden, rngs=rngs))
        backward = nnx.RNN(nnx.LSTMCell(embed, hidden, rngs=rngs))
        self.birnn = nnx.Bidirectional(forward, backward)
        self.head = nnx.Linear(2 * hidden, n_classes, rngs=rngs)

    def __call__(self, tokens):
        h = self.embed(tokens)      # (B, T, embed)
        h = self.birnn(h)           # (B, T, 2*hidden)  (fwd ++ bwd)
        return self.head(h[:, -1])


MODELS = {
    "lstm": LSTMClassifier,
    "gru": GRUClassifier,
    "rnn": VanillaRNNClassifier,
    "bilstm": BiLSTMClassifier,
}


# ============================================================================
# MANUAL SCAN — carry threading under the hood
# ============================================================================

def manual_lstm_scan(cell: nnx.LSTMCell, sequence: jax.Array):
    """Reproduce what ``nnx.RNN`` does internally.

    A recurrent layer is a ``scan``: at each step ``t`` the cell maps
    ``(carry, x_t) -> (carry, y_t)``, threading the ``carry`` (for an LSTM the
    pair ``(h, c)``) from one step to the next. ``nnx.scan`` with
    ``in_axes=(nnx.Carry, 1)`` marks the first argument as the threaded carry
    and slices the *other* argument along the time axis (axis 1). ``out_axes``
    likewise stacks the per-step outputs back along axis 1.

    Args:
        cell: an ``nnx.LSTMCell`` with matching ``in_features``.
        sequence: ``(B, T, in_features)`` input.

    Returns:
        ``(carry, outputs)`` where ``carry`` is the final ``(h, c)`` and
        ``outputs`` is ``(B, T, hidden)``.
    """
    batch = sequence.shape[0]
    in_features = sequence.shape[-1]
    carry = cell.initialize_carry((batch, in_features), nnx.Rngs(0))

    @nnx.scan(in_axes=(nnx.Carry, 1), out_axes=(nnx.Carry, 1))
    def step(carry, x_t):
        carry, y_t = cell(carry, x_t)   # (h, c), x_t -> (h, c), y_t
        return carry, y_t

    carry, outputs = step(carry, sequence)
    return carry, outputs


# ============================================================================
# DATA — synthetic parity task
# ============================================================================

def make_dataset(synthetic: bool = True, *, n: int = 512, seq_len: int = 12,
                 vocab: int = 10, seed: int = 0):
    """Parity task: label = (sum of tokens) mod 2.

    Solving it forces the network to *integrate information across every time
    step* — a single missed token flips the label — so it is a clean stress
    test of recurrent memory. There is no external dataset to download here;
    ``synthetic=False`` just generates a larger, longer (harder) version.

    Returns:
        ``(x, y)`` with ``x: (n, seq_len) int32`` tokens and ``y: (n,) int32``
        parity labels in ``{0, 1}``.
    """
    if not synthetic:
        n, seq_len = max(n, 4096), max(seq_len, 20)
    key = jax.random.key(seed)
    x = jax.random.randint(key, (n, seq_len), 0, vocab)
    y = (x.sum(axis=1) % 2).astype(jnp.int32)
    return x, y


# ============================================================================
# TRAIN STEP
# ============================================================================

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


# ============================================================================
# MAIN
# ============================================================================

def main():
    epochs = int(os.environ.get("EPOCHS", 40))
    batch_size = int(os.environ.get("BATCH", 128))
    synthetic = os.environ.get("SYNTHETIC", "1") != "0"
    which = os.environ.get("MODEL", "lstm")

    vocab, embed, hidden, n_classes = 10, 32, 64, 2

    x, y = make_dataset(synthetic=synthetic, vocab=vocab)
    n = x.shape[0]

    rngs = nnx.Rngs(0)
    model = MODELS[which](vocab, embed, hidden, n_classes, rngs=rngs)
    tx = optax.adam(1e-2)
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

    print(f"model={which} samples={n} seq_len={x.shape[1]} "
          f"epochs={epochs} batch={batch_size}")

    step = 0
    for epoch in range(epochs):
        perm = jax.random.permutation(jax.random.key(epoch), n)
        for i in range(0, n - batch_size + 1, batch_size):
            idx = perm[i:i + batch_size]
            batch = {"x": x[idx], "y": y[idx]}
            loss, acc = train_step(model, optimizer, batch)
            step += 1
        print(f"epoch {epoch:2d}  loss {float(loss):.4f}  acc {float(acc):.4f}")

    # Demonstrate the manual scan matches the high-level nnx.RNN wrapper.
    demo_cell = nnx.LSTMCell(embed, hidden, rngs=nnx.Rngs(0))
    seq = jnp.ones((4, x.shape[1], embed))
    (h, c), outputs = manual_lstm_scan(demo_cell, seq)
    print(f"manual scan outputs {outputs.shape}  carry h {h.shape}  c {c.shape}")


if __name__ == "__main__":
    main()
