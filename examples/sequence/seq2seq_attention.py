"""
Sequence-to-Sequence with Attention
====================================
An encoder-decoder that maps a source token sequence to a target sequence,
here the synthetic copy-and-reverse task (target = reversed input). The decoder
uses cross-attention (nnx.MultiHeadAttention) over the encoder states, with
teacher forcing. This is the conditional source->target counterpart to a
decoder-only GPT.

Run: python sequence/seq2seq_attention.py
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
# MODEL
# ============================================================================

class Encoder(nnx.Module):
    """Embed source tokens -> GRU over time -> per-step encoder states.

    ``nnx.RNN`` scans the GRU cell across the time axis of a ``(B, T)`` token
    sequence and returns the full stack of hidden states ``(B, T, hidden)``.
    The decoder attends over *all* of these, not just the last one.
    """

    def __init__(self, vocab: int, embed: int, hidden: int, *, rngs: nnx.Rngs):
        self.embed = nnx.Embed(vocab, embed, rngs=rngs)
        self.rnn = nnx.RNN(nnx.GRUCell(embed, hidden, rngs=rngs))

    def __call__(self, src):
        h = self.embed(src)   # (B, T)      -> (B, T, embed)
        return self.rnn(h)    # (B, T, embed) -> (B, T, hidden)   encoder states


class Decoder(nnx.Module):
    """Teacher-forced GRU decoder with cross-attention over encoder states.

    At every output step the decoder forms a query from its own recurrent
    state and attends over the encoder states (keys/values). The resulting
    context vector is concatenated with the decoder state and projected to the
    vocabulary. This is the encoder-decoder attention of Bahdanau et al.,
    expressed with the built-in ``nnx.MultiHeadAttention``.
    """

    def __init__(self, vocab: int, embed: int, hidden: int, num_heads: int,
                 *, rngs: nnx.Rngs):
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
        d = self.embed(dec_in)          # (B, T)      -> (B, T, embed)
        d = self.rnn(d)                 # (B, T, embed) -> (B, T, hidden)  decoder states
        # Cross-attention: query = decoder states, key/value = encoder states.
        ctx = self.cross_attn(d, enc_states, enc_states)   # (B, T, hidden)  context
        combined = jnp.concatenate([d, ctx], axis=-1)      # (B, T, 2*hidden)
        return self.out(combined)                          # (B, T, vocab)   logits


class Seq2SeqAttention(nnx.Module):
    """Encoder + attention decoder wired together."""

    def __init__(self, vocab: int, embed: int, hidden: int, num_heads: int,
                 *, rngs: nnx.Rngs):
        self.encoder = Encoder(vocab, embed, hidden, rngs=rngs)
        self.decoder = Decoder(vocab, embed, hidden, num_heads, rngs=rngs)

    def __call__(self, src, dec_in):
        enc_states = self.encoder(src)              # (B, T, hidden)
        return self.decoder(dec_in, enc_states)     # (B, T_out, vocab)


# ============================================================================
# DATA — synthetic copy-and-reverse task
# ============================================================================

def make_dataset(synthetic: bool = True, *, n: int = 1024, seq_len: int = 8,
                 vocab: int = 12, seed: int = 0):
    """Copy-and-reverse: the target is the source sequence read backwards.

    ``src = [a, b, c, d]``  ->  ``tgt = [d, c, b, a]``. Solving it forces the
    decoder to align output step ``t`` with source position ``T-1-t`` — an
    alignment that cross-attention learns to express. Everything is generated
    in-memory; ``synthetic=False`` just makes a larger, longer (harder) version.

    Returns a dict with:
        ``src``     (n, T)   int32 source tokens in ``[0, vocab)``
        ``dec_in``  (n, T)   int32 teacher-forcing input: ``<bos>`` + tgt[:-1]
        ``tgt``     (n, T)   int32 target tokens (reversed source)
    where the ``<bos>`` id is ``vocab``.
    """
    if not synthetic:
        n, seq_len = max(n, 8192), max(seq_len, 12)
    key = jax.random.key(seed)
    src = jax.random.randint(key, (n, seq_len), 0, vocab).astype(jnp.int32)
    tgt = src[:, ::-1]                                   # reversed sequence
    bos = jnp.full((n, 1), vocab, dtype=jnp.int32)       # start-of-sequence id
    dec_in = jnp.concatenate([bos, tgt[:, :-1]], axis=1)  # shift-right teacher forcing
    return {"src": src, "dec_in": dec_in, "tgt": tgt}


# ============================================================================
# TRAIN STEP
# ============================================================================

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


# ============================================================================
# MAIN
# ============================================================================

def main():
    epochs = int(os.environ.get("EPOCHS", 30))
    batch_size = int(os.environ.get("BATCH", 128))
    synthetic = os.environ.get("SYNTHETIC", "1") != "0"

    vocab, embed, hidden, num_heads = 12, 32, 64, 4

    data = make_dataset(synthetic=synthetic, vocab=vocab)
    n = data["src"].shape[0]
    seq_len = data["src"].shape[1]

    rngs = nnx.Rngs(0)
    model = Seq2SeqAttention(vocab, embed, hidden, num_heads, rngs=rngs)
    tx = optax.adam(3e-3)
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

    print(f"seq2seq+attention  samples={n} seq_len={seq_len} vocab={vocab} "
          f"epochs={epochs} batch={batch_size}")

    for epoch in range(epochs):
        perm = jax.random.permutation(jax.random.key(epoch), n)
        loss = acc = 0.0
        for i in range(0, n - batch_size + 1, batch_size):
            idx = perm[i:i + batch_size]
            batch = {k: v[idx] for k, v in data.items()}
            loss, acc = train_step(model, optimizer, batch)
        print(f"epoch {epoch:2d}  loss {float(loss):.4f}  token_acc {float(acc):.4f}")


if __name__ == "__main__":
    main()
