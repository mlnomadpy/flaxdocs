"""
Graph Convolutional Network on Zachary's Karate Club
====================================================
A two-layer GCN that classifies the 34 members of Zachary's Karate Club into
their two real-world factions from only *two* labeled nodes (semi-supervised
node classification) using symmetric-normalized neighborhood aggregation.

Run: python scientific/gcn_karate.py
"""

import os

import jax
import jax.numpy as jnp
from flax import nnx
import optax

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.training_utils import compute_accuracy

# ============================================================================
# THE GRAPH: Zachary's Karate Club (34 nodes, 78 undirected edges)
# ============================================================================
# The classic social-network benchmark. A karate club split into two factions
# after a dispute between the instructor "Mr. Hi" (node 0) and the club
# administrator "Officer" (node 33). We know the full ground-truth split, but
# we only *reveal* those two anchor labels to the model.

NUM_NODES = 34
NUM_CLASSES = 2

# Standard 0-indexed edge list (matches networkx.karate_club_graph()).
EDGES = [
    (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8),
    (0, 10), (0, 11), (0, 12), (0, 13), (0, 17), (0, 19), (0, 21), (0, 31),
    (1, 2), (1, 3), (1, 7), (1, 13), (1, 17), (1, 19), (1, 21), (1, 30),
    (2, 3), (2, 7), (2, 8), (2, 9), (2, 13), (2, 27), (2, 28), (2, 32),
    (3, 7), (3, 12), (3, 13),
    (4, 6), (4, 10),
    (5, 6), (5, 10), (5, 16),
    (6, 16),
    (8, 30), (8, 32), (8, 33),
    (9, 33),
    (13, 33),
    (14, 32), (14, 33),
    (15, 32), (15, 33),
    (18, 32), (18, 33),
    (19, 33),
    (20, 32), (20, 33),
    (22, 32), (22, 33),
    (23, 25), (23, 27), (23, 29), (23, 32), (23, 33),
    (24, 25), (24, 27), (24, 31),
    (25, 31),
    (26, 29), (26, 33),
    (27, 33),
    (28, 31), (28, 33),
    (29, 32), (29, 33),
    (30, 32), (30, 33),
    (31, 32), (31, 33),
    (32, 33),
]

# Ground-truth community for every node (0 = Mr. Hi faction, 1 = Officer
# faction). Used ONLY for evaluation, never for training.
TRUE_LABELS = jnp.array([
    0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
], dtype=jnp.int32)

# The two semi-supervised anchors: the two faction leaders.
ANCHOR_NODES = (0, 33)


# ============================================================================
# ADJACENCY + SYMMETRIC NORMALIZATION
# ============================================================================

def normalize_adjacency(edges, num_nodes: int) -> jax.Array:
    r"""Build the symmetric-normalized adjacency with self-loops.

    Computes  \hat A = D^{-1/2} (A + I) D^{-1/2}, where A is the binary
    adjacency of the undirected graph and D is the degree matrix of A + I.
    """
    a = jnp.zeros((num_nodes, num_nodes), dtype=jnp.float32)
    src = jnp.array([e[0] for e in edges])
    dst = jnp.array([e[1] for e in edges])
    a = a.at[src, dst].set(1.0)
    a = a.at[dst, src].set(1.0)          # undirected -> symmetric

    a = a + jnp.eye(num_nodes)           # add self-loops (A + I)
    deg = a.sum(axis=1)                  # row degrees of (A + I)
    d_inv_sqrt = jnp.power(deg, -0.5)
    d_inv_sqrt = jnp.diag(d_inv_sqrt)
    return d_inv_sqrt @ a @ d_inv_sqrt   # D^{-1/2} (A + I) D^{-1/2}


def make_dataset(synthetic: bool = True):
    """Return the karate-club graph as tiny jnp arrays (fully offline).

    The graph itself is the dataset, so `synthetic=True` (the default) simply
    builds it from the hardcoded edge list. `synthetic=False` rebuilds the same
    graph from networkx to cross-check, if that optional dependency is present.
    """
    edges = EDGES
    labels = TRUE_LABELS
    if not synthetic:
        try:
            import networkx as nx

            g = nx.karate_club_graph()
            edges = list(g.edges())
            labels = jnp.array(
                [0 if g.nodes[i]["club"] == "Mr. Hi" else 1 for i in g.nodes()],
                dtype=jnp.int32,
            )
        except Exception as exc:  # pragma: no cover - offline fallback
            print(f"[make_dataset] networkx unavailable ({exc}); using hardcoded graph.")

    a_hat = normalize_adjacency(edges, NUM_NODES)

    # Identity (one-hot node-id) features: each node is its own basis vector.
    features = jnp.eye(NUM_NODES, dtype=jnp.float32)
    node_ids = jnp.arange(NUM_NODES, dtype=jnp.int32)

    # Semi-supervised mask: 1.0 only on the two anchor nodes.
    train_mask = jnp.zeros((NUM_NODES,), dtype=jnp.float32)
    train_mask = train_mask.at[jnp.array(ANCHOR_NODES)].set(1.0)

    return {
        "a_hat": a_hat,
        "features": features,
        "node_ids": node_ids,
        "labels": labels,
        "train_mask": train_mask,
    }


# ============================================================================
# MODEL: two-layer Graph Convolutional Network
# ============================================================================

class GCNLayer(nnx.Module):
    r"""One graph-convolution layer:  H' = \hat A H W.

    `nnx.Linear` (no bias) supplies the learnable weight W; the fixed
    normalized adjacency \hat A does the neighborhood aggregation via an
    einsum (a plain, parameter-free matmul over the node axis).
    """

    def __init__(self, in_features: int, out_features: int, *, rngs: nnx.Rngs):
        self.linear = nnx.Linear(in_features, out_features, use_bias=False, rngs=rngs)

    def __call__(self, a_hat: jax.Array, h: jax.Array) -> jax.Array:
        h = self.linear(h)                          # H W    -> (N, out)
        h = jnp.einsum("ij,jf->if", a_hat, h)       # \hat A (H W) -> (N, out)
        return h


class GCN(nnx.Module):
    """Two GCN layers mapping node features to class logits."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        num_classes: int,
        *,
        use_embedding: bool = False,
        num_nodes: int = NUM_NODES,
        rngs: nnx.Rngs,
    ):
        self.use_embedding = use_embedding
        if use_embedding:
            # Learn a dense feature per node instead of using one-hot inputs.
            self.embed = nnx.Embed(num_nodes, in_features, rngs=rngs)
        self.gcn1 = GCNLayer(in_features, hidden_features, rngs=rngs)
        self.gcn2 = GCNLayer(hidden_features, num_classes, rngs=rngs)

    def __call__(self, a_hat, features=None, node_ids=None):
        h = self.embed(node_ids) if self.use_embedding else features
        h = nnx.relu(self.gcn1(a_hat, h))           # (N, hidden)
        logits = self.gcn2(a_hat, h)                # (N, num_classes)
        return logits


def create_model(rngs: nnx.Rngs, *, use_embedding: bool = False, hidden_features: int = 16):
    """Build a karate-club GCN (34 one-hot inputs -> hidden -> 2 classes)."""
    in_features = 8 if use_embedding else NUM_NODES
    return GCN(
        in_features=in_features,
        hidden_features=hidden_features,
        num_classes=NUM_CLASSES,
        use_embedding=use_embedding,
        rngs=rngs,
    )


# ============================================================================
# LOSS + TRAIN STEP
# ============================================================================

def masked_cross_entropy(logits, labels, mask):
    """Cross-entropy averaged over the labeled (masked) nodes only."""
    ce = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    return (ce * mask).sum() / mask.sum()


@nnx.jit
def train_step(model: GCN, optimizer: nnx.Optimizer, batch):
    def loss_fn(model):
        logits = model(batch["a_hat"], features=batch["features"], node_ids=batch["node_ids"])
        loss = masked_cross_entropy(logits, batch["labels"], batch["train_mask"])
        acc = compute_accuracy(logits, batch["labels"])   # accuracy over ALL nodes
        return loss, acc

    (loss, acc), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    optimizer.update(model, grads)
    return loss, acc


# ============================================================================
# MAIN
# ============================================================================

def main():
    epochs = int(os.environ.get("EPOCHS", 200))          # full-batch grad steps
    _ = int(os.environ.get("BATCH", NUM_NODES))          # single-graph: unused
    synthetic = os.environ.get("SYNTHETIC", "1") != "0"

    batch = make_dataset(synthetic=synthetic)

    model = create_model(nnx.Rngs(0))
    optimizer = nnx.Optimizer(model, optax.adam(1e-2), wrt=nnx.Param)

    print(f"Training GCN on Zachary's Karate Club for {epochs} steps "
          f"(labeled anchors: nodes {ANCHOR_NODES[0]} and {ANCHOR_NODES[1]})\n")
    for step in range(epochs):
        loss, acc = train_step(model, optimizer, batch)
        if step % 20 == 0 or step == epochs - 1:
            print(f"step {step:4d} | masked loss {float(loss):.4f} | full-graph acc {float(acc):.3f}")

    logits = model(batch["a_hat"], features=batch["features"], node_ids=batch["node_ids"])
    final_acc = float(compute_accuracy(logits, batch["labels"]))
    print(f"\nFinal full-graph accuracy: {final_acc:.3f} "
          f"({int(final_acc * NUM_NODES)}/{NUM_NODES} nodes)")


if __name__ == "__main__":
    main()
