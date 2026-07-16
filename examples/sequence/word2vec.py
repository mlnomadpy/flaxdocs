"""
Word2Vec: Skip-Gram with Negative Sampling
===========================================
Learn dense word embeddings from a tiny hardcoded corpus using the skip-gram
with negative sampling (SGNS) objective, built from two ``nnx.Embed`` tables.
After training, cosine-nearest neighbours recover the corpus's semantic themes.

Run: python sequence/word2vec.py
"""

import os

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
import optax

import sys
from pathlib import Path

# Add parent directory to path for shared imports
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================================
# DATA — a tiny themed corpus (fully self-contained, no downloads)
# ============================================================================

# Very frequent "glue" words carry little meaning and co-occur with everything,
# so we drop them before building pairs. This is the tutorial-sized version of
# Word2Vec's frequent-word subsampling and keeps the learned clusters crisp.
STOPWORDS = {"the", "a", "and", "on", "in", "of", "with", "over", "under",
             "across", "toward", "above", "past"}

# Four semantic themes. Words inside a theme co-occur with each other and never
# with words from another theme, so skip-gram should pull each theme together.
CORPUS = [
    # royalty
    "king and queen rule the royal palace",
    "prince and princess wear the golden crown",
    "royal king and queen sit on the throne",
    "crown throne palace king queen prince princess royal",
    "golden crown royal throne palace",
    # ocean
    "boat sail across the deep blue ocean",
    "fish swim under the ocean wave",
    "sailor steer the boat over the ocean water",
    "blue ocean wave boat fish sail water deep",
    "deep ocean water hold the fish and boat",
    # space
    "rocket fly toward a distant star",
    "planet orbit the star and moon glow",
    "rocket reach orbit above the distant planet",
    "star planet moon rocket orbit galaxy comet sky",
    "comet streak past the moon and distant star",
    # music
    "band play a song with guitar and drum",
    "singer sing the melody and guitar play",
    "drum guitar blend the melody of the song",
    "song melody guitar drum band singer tune rhythm",
    "band play a joyful tune and melody",
]

# Ground-truth themes, used only to sanity-check the learned neighbours.
THEMES = {
    "royalty": ["king", "queen", "prince", "princess", "royal", "throne",
                "crown", "palace", "golden", "rule", "wear", "sit"],
    "ocean": ["boat", "ocean", "wave", "fish", "sail", "sailor", "blue",
              "water", "deep", "swim", "steer", "hold"],
    "space": ["rocket", "moon", "star", "planet", "orbit", "galaxy", "comet",
              "sky", "distant", "fly", "reach", "glow", "streak"],
    "music": ["band", "song", "guitar", "drum", "melody", "singer", "tune",
              "rhythm", "play", "sing", "blend", "joyful"],
}


def tokenize(line):
    """Lowercase, split, and drop stopwords."""
    return [w for w in line.lower().split() if w not in STOPWORDS]


def build_vocab(corpus):
    """Map every (non-stopword) token to an integer id (sorted for determinism)."""
    tokens = sorted({w for line in corpus for w in tokenize(line)})
    stoi = {w: i for i, w in enumerate(tokens)}
    itos = tokens
    return stoi, itos


def build_skipgram_pairs(corpus, stoi, window: int = 2):
    """Turn the corpus into (center, context) id pairs.

    For each token we emit one pair per neighbour inside ``[-window, +window]``,
    which is exactly the set of positive examples the skip-gram model must score
    highly. Returns two ``(N,) int32`` arrays.
    """
    centers, contexts = [], []
    for line in corpus:
        ids = [stoi[w] for w in tokenize(line)]
        for i, center in enumerate(ids):
            lo, hi = max(0, i - window), min(len(ids), i + window + 1)
            for j in range(lo, hi):
                if j == i:
                    continue
                centers.append(center)
                contexts.append(ids[j])
    return np.asarray(centers, np.int32), np.asarray(contexts, np.int32)


def sample_negatives(key, n: int, vocab_size: int, k: int):
    """Draw ``k`` uniform-random negative context ids per positive pair.

    Word2Vec's original paper uses a unigram^0.75 distribution; uniform is a fine
    approximation on a small balanced vocabulary and keeps the demo transparent.
    Returns an ``(n, k) int32`` array.
    """
    return jax.random.randint(key, (n, k), 0, vocab_size).astype(jnp.int32)


# ============================================================================
# MODEL — two embedding tables (center "input" + context "output")
# ============================================================================

class SkipGramNS(nnx.Module):
    """Skip-gram with negative sampling.

    Two ``nnx.Embed`` tables of shape ``(vocab, dim)``: ``center`` holds the
    "input" word vectors we ultimately keep, ``context`` holds the "output"
    vectors used to score neighbours. The score of a (center, context) pair is
    their dot product; training pushes true pairs up and sampled negatives down.
    """

    def __init__(self, vocab_size: int, dim: int, *, rngs: nnx.Rngs):
        self.center = nnx.Embed(vocab_size, dim, rngs=rngs)
        self.context = nnx.Embed(vocab_size, dim, rngs=rngs)

    def __call__(self, center_ids, context_ids, negative_ids):
        v_c = self.center(center_ids)              # (B, D)  center vectors
        u_o = self.context(context_ids)            # (B, D)  positive contexts
        u_n = self.context(negative_ids)           # (B, K, D) negatives
        pos_score = jnp.sum(v_c * u_o, axis=-1)                 # (B,)
        neg_score = jnp.einsum("bd,bkd->bk", v_c, u_n)          # (B, K)
        return pos_score, neg_score

    def word_vectors(self):
        """The learned input embeddings — the vectors you export and reuse."""
        return self.center.embedding[...]          # (vocab, D)


def sgns_loss(pos_score, neg_score):
    """Negative-sampling loss (a sigmoid BCE, summed over K negatives).

    $-\\log \\sigma(v_c \\cdot u_o) - \\sum_k \\log \\sigma(-v_c \\cdot u_{n_k})$,
    averaged over the batch. ``jax.nn.log_sigmoid`` is the numerically stable
    ``log(sigmoid(x))``.
    """
    pos = -jax.nn.log_sigmoid(pos_score)                       # want score high
    neg = -jax.nn.log_sigmoid(-neg_score).sum(axis=-1)        # want scores low
    return jnp.mean(pos + neg)


# ============================================================================
# TRAIN STEP
# ============================================================================

@nnx.jit
def train_step(model, optimizer, batch):
    def loss_fn(model):
        pos_score, neg_score = model(
            batch["center"], batch["context"], batch["negatives"]
        )
        loss = sgns_loss(pos_score, neg_score)
        return loss, (pos_score, neg_score)

    (loss, aux), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    optimizer.update(model, grads)
    return loss, aux


# ============================================================================
# NEAREST NEIGHBOURS (cosine)
# ============================================================================

def nearest_neighbors(vectors, itos, word: str, stoi, k: int = 5):
    """Return the ``k`` most cosine-similar words to ``word`` (excluding itself)."""
    normed = vectors / (jnp.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)
    sims = normed @ normed[stoi[word]]
    order = jnp.argsort(sims)[::-1]
    out = []
    for i in order.tolist():
        if itos[i] == word:
            continue
        out.append((itos[i], float(sims[i])))
        if len(out) == k:
            break
    return out


# ============================================================================
# MAIN
# ============================================================================

def main():
    epochs = int(os.environ.get("EPOCHS", 200))
    batch_size = int(os.environ.get("BATCH", 128))
    # SYNTHETIC is honoured for interface consistency; this corpus is always
    # self-contained, so there is nothing to download either way.
    _ = os.environ.get("SYNTHETIC", "1")

    dim = int(os.environ.get("DIM", 32))
    window = int(os.environ.get("WINDOW", 2))
    n_negatives = int(os.environ.get("NEG", 6))

    stoi, itos = build_vocab(CORPUS)
    vocab_size = len(itos)
    centers, contexts = build_skipgram_pairs(CORPUS, stoi, window=window)
    n_pairs = centers.shape[0]

    rngs = nnx.Rngs(0)
    model = SkipGramNS(vocab_size, dim, rngs=rngs)
    optimizer = nnx.Optimizer(model, optax.adam(5e-2), wrt=nnx.Param)

    print(f"vocab={vocab_size} pairs={n_pairs} dim={dim} "
          f"neg={n_negatives} epochs={epochs} batch={batch_size}")

    key = jax.random.key(0)
    centers_j = jnp.asarray(centers)
    contexts_j = jnp.asarray(contexts)
    step = 0
    for epoch in range(epochs):
        key, pkey = jax.random.split(key)
        perm = jax.random.permutation(pkey, n_pairs)
        for i in range(0, n_pairs - batch_size + 1, batch_size) or [0]:
            idx = perm[i:i + batch_size]
            key, nkey = jax.random.split(key)
            batch = {
                "center": centers_j[idx],
                "context": contexts_j[idx],
                "negatives": sample_negatives(
                    nkey, idx.shape[0], vocab_size, n_negatives
                ),
            }
            loss, _ = train_step(model, optimizer, batch)
            step += 1
        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"epoch {epoch:3d}  loss {float(loss):.4f}")

    vectors = model.word_vectors()
    print("\nNearest neighbours (cosine):")
    for probe in ["king", "ocean", "rocket", "guitar"]:
        nbrs = nearest_neighbors(vectors, itos, probe, stoi, k=4)
        pretty = ", ".join(f"{w}:{s:.2f}" for w, s in nbrs)
        print(f"  {probe:8s} -> {pretty}")


if __name__ == "__main__":
    main()
