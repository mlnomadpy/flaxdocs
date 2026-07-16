"""
Flax NNX: Metric Learning with a Siamese Network + Triplet Loss
===============================================================
Learn an embedding where same-class inputs land close together and
different-class inputs are pushed apart, using a shared-weight (Siamese)
embedding network trained with the triplet loss and in-batch hard mining.

Run: python advanced/metric_learning.py

Reference:
    Schroff et al. "FaceNet: A Unified Embedding for Face Recognition and
    Clustering" CVPR 2015. https://arxiv.org/abs/1503.03832
    Hermans et al. "In Defense of the Triplet Loss for Person
    Re-Identification" 2017. https://arxiv.org/abs/1703.07737
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

from shared.models import MLP


# ============================================================================
# 1. SIAMESE EMBEDDING NETWORK
# ============================================================================

class EmbeddingNet(nnx.Module):
    """Shared-weight embedding network: input vector -> L2-normalized embedding.

    A Siamese network is just ONE network applied to every input; the "shared
    weights" come for free because we reuse the same module. The final
    L2 normalization places every embedding on the unit hypersphere, so
    squared Euclidean distance in [0, 4] and cosine distance agree up to a
    constant and the margin has a consistent scale.
    """

    def __init__(self, in_features: int, hidden: int, embed_dim: int,
                 *, rngs: nnx.Rngs, depth: int = 2):
        # Reuse the shared MLP backbone; it maps (B, in_features) -> (B, embed_dim).
        self.backbone = MLP(in_features, hidden, embed_dim, n_layers=depth,
                            rngs=rngs, activation="relu")

    def embed(self, x):
        """Map a batch of inputs to unit-norm embeddings (B, embed_dim)."""
        z = self.backbone(x)
        return z / (jnp.linalg.norm(z, axis=-1, keepdims=True) + 1e-8)

    def __call__(self, x, train: bool = False):
        return self.embed(x)


# ============================================================================
# 2. TRIPLET LOSS + IN-BATCH HARD MINING
# ============================================================================

def pairwise_sq_dist(z):
    """Squared Euclidean distance matrix D[i, j] = ||z_i - z_j||^2, shape (B, B)."""
    sq = jnp.sum(z * z, axis=1)
    d = sq[:, None] + sq[None, :] - 2.0 * (z @ z.T)
    return jnp.maximum(d, 0.0)   # clip tiny negatives from round-off


def batch_hard_triplet_loss(z, labels, margin: float = 0.2):
    """Batch-hard triplet loss (Hermans et al. 2017).

    For every anchor in the batch we mine, *within the same batch*:
      - the HARDEST positive  = the same-class example that is FARTHEST away
      - the HARDEST negative  = the different-class example that is CLOSEST
    and push them to satisfy  d(a, p) + margin <= d(a, n):

        L = mean_a  max(0,  d(a, p*) - d(a, n*) + margin)

    Mining the hardest triplet in-batch is far more sample-efficient than
    random triplets, most of which are already "easy" (loss 0) and contribute
    no gradient. Returns (loss, fraction_of_active_triplets).
    """
    B = z.shape[0]
    D = pairwise_sq_dist(z)

    same = labels[:, None] == labels[None, :]
    eye = jnp.eye(B, dtype=bool)
    pos_mask = same & ~eye          # same class, excluding the anchor itself
    neg_mask = ~same                # different class

    # Hardest positive: largest distance among same-class examples.
    hardest_pos = jnp.max(jnp.where(pos_mask, D, -jnp.inf), axis=1)
    # Hardest negative: smallest distance among different-class examples.
    hardest_neg = jnp.min(jnp.where(neg_mask, D, jnp.inf), axis=1)

    losses = jax.nn.relu(hardest_pos - hardest_neg + margin)
    frac_active = jnp.mean((losses > 0).astype(jnp.float32))
    return jnp.mean(losses), frac_active


# ============================================================================
# 3. VERIFICATION METRIC (positive pairs closer than negative pairs)
# ============================================================================

def verification_accuracy(z, labels):
    """Fraction of anchors whose MEAN same-class distance is smaller than their
    MEAN different-class distance. A single scalar 'triplet loss decreasing' is
    a poor progress signal for retrieval, so we track this ranking metric: it
    starts near 0.5 (random) and rises toward 1.0 as classes separate.

    Also returns (mean positive-pair distance, mean negative-pair distance).
    """
    B = z.shape[0]
    D = pairwise_sq_dist(z)
    same = labels[:, None] == labels[None, :]
    eye = jnp.eye(B, dtype=bool)
    pos_mask = (same & ~eye).astype(jnp.float32)
    neg_mask = (~same).astype(jnp.float32)

    mean_pos = (D * pos_mask).sum(1) / jnp.maximum(pos_mask.sum(1), 1.0)
    mean_neg = (D * neg_mask).sum(1) / jnp.maximum(neg_mask.sum(1), 1.0)

    acc = jnp.mean((mean_pos < mean_neg).astype(jnp.float32))
    return acc, mean_pos.mean(), mean_neg.mean()


# ============================================================================
# 4. DATA (synthetic class-structured vectors by default; MNIST when SYNTHETIC=0)
# ============================================================================

def make_synthetic(n_classes: int = 10, per_class: int = 80, dim: int = 48,
                   center_scale: float = 0.6, noise: float = 1.2, seed: int = 0):
    """Class-structured vectors: each class is a random center + Gaussian noise.

    Offline, self-contained, and deliberately OVERLAPPING (small center scale,
    large noise) so the raw input is only weakly separable and the network has
    to *learn* a good embedding rather than read the class off the input.
    """
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_classes, dim)) * center_scale
    X, Y = [], []
    for c in range(n_classes):
        pts = centers[c] + rng.normal(size=(per_class, dim)) * noise
        X.append(pts)
        Y.extend([c] * per_class)
    X = np.concatenate(X, axis=0).astype(np.float32)
    Y = np.array(Y, dtype=np.int32)
    return X, Y, dim, n_classes


def make_mnist(n: int = 3000):
    """Real MNIST (flattened to 784-dim vectors) when SYNTHETIC=0. Needs tfds."""
    import tensorflow_datasets as tfds
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    ds = tfds.load('mnist', split=f'train[:{n}]', as_supervised=True)
    xs, ys = [], []
    for img, lab in tfds.as_numpy(ds):
        xs.append(img.reshape(-1).astype(np.float32) / 255.0)
        ys.append(int(lab))
    return np.stack(xs), np.array(ys, dtype=np.int32), 784, 10


def build_class_index(Y):
    """Map each class label -> array of row indices (for PK batch sampling)."""
    return {int(c): np.where(Y == c)[0] for c in np.unique(Y)}


def make_pk_batch(X, Y, class_to_idx, P: int, K: int, rng):
    """Sample a balanced 'PK' batch: P classes, K examples each (batch = P*K).

    Balanced batches guarantee that every anchor has both positives (K-1 of
    them) and negatives, which is exactly what in-batch mining needs.
    """
    classes = rng.choice(list(class_to_idx.keys()), size=P, replace=False)
    idxs = []
    for c in classes:
        pool = class_to_idx[int(c)]
        pick = rng.choice(pool, size=K, replace=len(pool) < K)
        idxs.extend(pick.tolist())
    idxs = np.array(idxs)
    return {'x': jnp.asarray(X[idxs]), 'y': jnp.asarray(Y[idxs])}


# ============================================================================
# 5. TRAIN STEP
# ============================================================================

@nnx.jit
def train_step(model, optimizer, batch, margin: float = 0.2):
    """One gradient step of triplet-loss metric learning."""
    def loss_fn(model):
        z = model.embed(batch['x'])                       # (B, embed_dim), unit norm
        loss, frac_active = batch_hard_triplet_loss(z, batch['y'], margin)
        return loss, frac_active

    (loss, frac_active), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    optimizer.update(model, grads)
    return loss, frac_active


# ============================================================================
# 6. VISUALIZATION (2D PCA of embeddings, before vs after training)
# ============================================================================

def save_embedding_scatter(z_before, z_after, labels, n_classes, path):
    """Project embeddings to 2D with PCA and scatter them colored by class.

    Two panels: BEFORE training (random init, classes overlap) vs AFTER
    training (classes form separated clusters). matplotlib is imported lazily
    with the Agg backend so importing this module stays cheap.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def pca2(z):
        """Project rows of z onto their top-2 principal components (numpy SVD)."""
        z = np.asarray(z, dtype=np.float64)
        z = z - z.mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(z, full_matrices=False)
        return z @ vt[:2].T

    labels = np.asarray(labels)
    panels = [
        (pca2(z_before), "Before training (random init)"),
        (pca2(z_after), "After training (triplet loss)"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    cmap = plt.get_cmap("tab10")
    for ax, (proj, title) in zip(axes, panels):
        for c in range(n_classes):
            m = labels == c
            ax.scatter(proj[m, 0], proj[m, 1], s=16, alpha=0.75,
                       color=cmap(c % 10), edgecolors="none", label=f"class {c}")
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        ax.set_xticks([])
        ax.set_yticks([])

    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
                   fontsize=8, framealpha=0.9, title="class")
    fig.suptitle("Metric learning: learned embeddings separate the classes "
                 "(2D PCA)", fontsize=13, y=1.0)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved embedding visualization to {path}")


# ============================================================================
# 7. MAIN
# ============================================================================

def main():
    # Run-scale from env with small CPU-friendly defaults; SYNTHETIC by default.
    epochs = int(os.environ.get('EPOCHS', 8))
    batch = int(os.environ.get('BATCH', 64))
    synthetic = os.environ.get('SYNTHETIC', '1') != '0'
    margin = float(os.environ.get('MARGIN', 0.2))
    embed_dim = int(os.environ.get('EMBED', 32))
    hidden = int(os.environ.get('HIDDEN', 128))

    if synthetic:
        X, Y, in_features, n_classes = make_synthetic()
    else:
        X, Y, in_features, n_classes = make_mnist()

    class_to_idx = build_class_index(Y)

    # Balanced PK sampling: P classes x K examples per batch.
    P = min(n_classes, 8)
    K = max(2, batch // P)
    pk = P * K
    steps_per_epoch = max(1, X.shape[0] // pk)

    print(f"Metric learning (Siamese + triplet loss) on "
          f"{'synthetic class-structured vectors' if synthetic else 'MNIST'}")
    print(f"  data: X={X.shape} classes={n_classes} | in_features={in_features}")
    print(f"  epochs={epochs} batch={pk} (P={P} classes x K={K}) "
          f"embed_dim={embed_dim} margin={margin}")

    model = EmbeddingNet(in_features, hidden, embed_dim, rngs=nnx.Rngs(0))
    optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(nnx.state(model, nnx.Param)))
    print(f"  model: EmbeddingNet with {n_params} parameters\n")

    # Fixed balanced subset (up to 40 per class) used for the before/after
    # embedding visualization, so every class is represented and legible.
    viz_idx = np.concatenate([class_to_idx[int(c)][:40]
                              for c in range(n_classes)])
    viz_x = jnp.asarray(X[viz_idx])
    viz_y = Y[viz_idx]
    # Embeddings at random init (before any training) for the "before" panel.
    z_before = np.asarray(model.embed(viz_x))

    # Fixed balanced batch used to report the verification metric each epoch.
    eval_rng = np.random.default_rng(123)
    eval_batch = make_pk_batch(X, Y, class_to_idx, P, min(K + 4, 12), eval_rng)

    rng = np.random.default_rng(0)
    step = 0
    for epoch in range(1, epochs + 1):
        ep_loss, ep_active, nb = 0.0, 0.0, 0
        for _ in range(steps_per_epoch):
            b = make_pk_batch(X, Y, class_to_idx, P, K, rng)
            loss, frac_active = train_step(model, optimizer, b, margin)
            ep_loss += float(loss)
            ep_active += float(frac_active)
            nb += 1
            step += 1

        z_eval = model.embed(eval_batch['x'])
        acc, dpos, dneg = verification_accuracy(z_eval, eval_batch['y'])
        print(f"  epoch {epoch:2d}/{epochs} | step {step:4d} | "
              f"triplet {ep_loss / nb:.4f} | active {ep_active / nb:.2f} | "
              f"ver_acc {float(acc):.3f} | d+ {float(dpos):.3f} d- {float(dneg):.3f}")

    print("\nDone. The embedding pulls same-class inputs together and pushes "
          "different-class inputs apart (d+ < d-).")

    # Visualize the learned embedding space: random init vs after training.
    z_after = np.asarray(model.embed(viz_x))
    out_path = os.path.join(os.environ.get("OUTDIR", "results"),
                            "metric_embedding.png")
    save_embedding_scatter(z_before, z_after, viz_y, n_classes, out_path)


if __name__ == "__main__":
    main()
