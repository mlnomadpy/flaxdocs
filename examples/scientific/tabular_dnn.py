"""
Deep Learning for Tabular Data
==============================
A tabular DNN that learns a per-column embedding for every categorical feature,
concatenates them with the numeric columns, and feeds an MLP -> classification
head. All data is synthetic and generated offline with jax.random.

Run: python scientific/tabular_dnn.py
"""

import os

import jax
import jax.numpy as jnp
from flax import nnx
import optax

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.training_utils import (
    compute_accuracy,
    compute_cross_entropy_loss,
    compute_mse_loss,
)

# ============================================================================
# SCHEMA: a small mixed-type table
# ============================================================================
# Four numeric columns (already standardized) plus three categorical columns.
# The cardinalities are the number of distinct categories each column can take,
# e.g. a "city" column with 5 values, a "device" column with 8, a "plan" with 3.

NUM_NUMERIC = 4
CAT_CARDINALITIES = (5, 8, 3)   # one entry per categorical column
NUM_CLASSES = 3


def embedding_dims(cardinalities) -> tuple:
    """Pick a small embedding width per categorical column.

    A common rule of thumb is to grow the embedding sub-linearly with the
    cardinality (fastai uses ~min(600, round(1.6 * card**0.56))). For these
    tiny toy columns we use min(8, max(2, card // 2)).
    """
    return tuple(min(8, max(2, card // 2)) for card in cardinalities)


# ============================================================================
# SYNTHETIC DATA (offline): label is a nonlinear function of ALL features
# ============================================================================

def make_dataset(n_samples: int, *, seed: int = 0, task: str = "classification"):
    """Generate a mixed numeric/categorical table with a nonlinear target.

    A hidden generative process (never shown to the model) mixes nonlinear
    numeric interactions with a random per-category effect table, so the model
    genuinely has to (a) learn useful category embeddings and (b) model numeric
    interactions. Returns a dict of jnp arrays; fully offline.
    """
    key = jax.random.key(seed)
    k_num, k_wnum, k_cat_eff = jax.random.split(key, 3)

    # --- numeric columns ~ N(0, 1) (pretend they are pre-standardized) --------
    x_num = jax.random.normal(k_num, (n_samples, NUM_NUMERIC))

    # --- integer-coded categorical columns ------------------------------------
    cat_cols = []
    for i, card in enumerate(CAT_CARDINALITIES):
        k = jax.random.fold_in(key, 100 + i)
        cat_cols.append(jax.random.randint(k, (n_samples,), 0, card))
    x_cat = jnp.stack(cat_cols, axis=1).astype(jnp.int32)   # (N, n_cat)

    # --- hidden ground truth: nonlinear numeric expansion + category effects --
    feat = jnp.concatenate([
        x_num,                                   # linear terms
        jnp.sin(2.0 * x_num[:, :2]),             # nonlinearity
        (x_num[:, 0] * x_num[:, 1])[:, None],    # interaction
        (x_num[:, 2] ** 2)[:, None],             # curvature
    ], axis=-1)
    w = jax.random.normal(k_wnum, (feat.shape[1], NUM_CLASSES))
    score = feat @ w                             # (N, NUM_CLASSES)

    for i, card in enumerate(CAT_CARDINALITIES):
        k = jax.random.fold_in(k_cat_eff, i)
        table = jax.random.normal(k, (card, NUM_CLASSES)) * 1.5
        score = score + table[x_cat[:, i]]       # add each column's effect

    if task == "regression":
        # A single continuous target: the same signal collapsed to a scalar.
        y = score.sum(axis=-1)
        return {"x_num": x_num, "x_cat": x_cat, "y": y}

    y = jnp.argmax(score, axis=-1).astype(jnp.int32)
    return {"x_num": x_num, "x_cat": x_cat, "y": y}


# ============================================================================
# MODEL: per-column embeddings -> concat with numeric -> MLP -> head
# ============================================================================

class TabularDNN(nnx.Module):
    """Embed each categorical column, concat with numeric, then an MLP head.

    Set out_features = NUM_CLASSES for classification, or out_features = 1 for
    regression (the only thing that changes is the head width and the loss).
    """

    def __init__(
        self,
        num_numeric: int,
        cat_cardinalities,
        embed_dims,
        hidden: int,
        out_features: int,
        *,
        rngs: nnx.Rngs,
    ):
        # One embedding table per categorical column. A list of submodules MUST
        # be wrapped in nnx.List so NNX registers them as state (Flax 0.12).
        self.embeddings = nnx.List([
            nnx.Embed(num_embeddings=card, features=dim, rngs=rngs)
            for card, dim in zip(cat_cardinalities, embed_dims)
        ])

        concat_dim = num_numeric + sum(embed_dims)
        self.fc1 = nnx.Linear(concat_dim, hidden, rngs=rngs)
        self.fc2 = nnx.Linear(hidden, hidden, rngs=rngs)
        self.dropout = nnx.Dropout(rate=0.1, rngs=rngs)
        self.head = nnx.Linear(hidden, out_features, rngs=rngs)

    def __call__(self, x_num: jax.Array, x_cat: jax.Array, train: bool = False):
        """x_num: (B, num_numeric) float, x_cat: (B, n_cat) int."""
        # Look up each column's embedding and concatenate everything.
        embeds = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        h = jnp.concatenate([x_num, *embeds], axis=-1)

        h = nnx.relu(self.fc1(h))
        h = self.dropout(h, deterministic=not train)
        h = nnx.relu(self.fc2(h))
        return self.head(h)          # (B, out_features) logits or regression value


def create_model(rngs: nnx.Rngs, *, hidden: int = 64, task: str = "classification"):
    """Build a TabularDNN for the fixed toy schema."""
    out_features = 1 if task == "regression" else NUM_CLASSES
    return TabularDNN(
        num_numeric=NUM_NUMERIC,
        cat_cardinalities=CAT_CARDINALITIES,
        embed_dims=embedding_dims(CAT_CARDINALITIES),
        hidden=hidden,
        out_features=out_features,
        rngs=rngs,
    )


# ============================================================================
# TRAIN STEPS (classification + regression variants)
# ============================================================================

@nnx.jit
def train_step(model: TabularDNN, optimizer: nnx.Optimizer, batch):
    """Classification step: cross-entropy loss, reports accuracy."""
    def loss_fn(model):
        logits = model(batch["x_num"], batch["x_cat"], train=True)
        loss = compute_cross_entropy_loss(logits, batch["y"])
        acc = compute_accuracy(logits, batch["y"])
        return loss, acc

    (loss, acc), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    optimizer.update(model, grads)
    return loss, acc


@nnx.jit
def train_step_regression(model: TabularDNN, optimizer: nnx.Optimizer, batch):
    """Regression step: mean-squared-error on the single scalar head."""
    def loss_fn(model):
        preds = model(batch["x_num"], batch["x_cat"], train=True)[:, 0]
        loss = compute_mse_loss(preds, batch["y"])
        return loss, loss

    (loss, _), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    optimizer.update(model, grads)
    return loss, loss


# ============================================================================
# MAIN
# ============================================================================

def main():
    epochs = int(os.environ.get("EPOCHS", 15))          # passes over the table
    batch_size = int(os.environ.get("BATCH", 256))
    n_samples = int(os.environ.get("N_SAMPLES", 2000))
    task = os.environ.get("TASK", "classification")     # or "regression"
    _ = os.environ.get("SYNTHETIC", "1")                # always synthetic/offline

    data = make_dataset(n_samples, task=task)
    step_fn = train_step_regression if task == "regression" else train_step

    model = create_model(nnx.Rngs(0), task=task)
    optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)

    print(f"Tabular DNN ({task}) | {n_samples} rows | "
          f"{NUM_NUMERIC} numeric + {len(CAT_CARDINALITIES)} categorical columns "
          f"(cardinalities {CAT_CARDINALITIES}, embeds {embedding_dims(CAT_CARDINALITIES)})\n")

    n_batches = max(1, n_samples // batch_size)
    for epoch in range(epochs):
        perm = jax.random.permutation(jax.random.key(epoch), n_samples)
        last_loss, last_acc = 0.0, 0.0
        for b in range(n_batches):
            idx = perm[b * batch_size:(b + 1) * batch_size]
            batch = {
                "x_num": data["x_num"][idx],
                "x_cat": data["x_cat"][idx],
                "y": data["y"][idx],
            }
            last_loss, last_acc = step_fn(model, optimizer, batch)

        if task == "regression":
            print(f"epoch {epoch:2d} | mse {float(last_loss):.4f}")
        else:
            print(f"epoch {epoch:2d} | loss {float(last_loss):.4f} | "
                  f"batch acc {float(last_acc):.3f}")

    if task != "regression":
        logits = model(data["x_num"], data["x_cat"], train=False)
        final_acc = float(compute_accuracy(logits, data["y"]))
        print(f"\nFinal full-dataset accuracy: {final_acc:.3f}")


if __name__ == "__main__":
    main()
