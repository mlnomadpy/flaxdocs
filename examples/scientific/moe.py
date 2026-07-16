"""
Mixture of Experts (MoE) in Flax NNX
====================================
A sparse Mixture-of-Experts layer: N small expert MLPs plus a learnable gate
that top-k routes each input, combines the chosen experts' outputs, and trains
against a task loss plus a load-balancing auxiliary loss on synthetic data.

Run: python scientific/moe.py
"""

import os

import jax
import jax.numpy as jnp
from flax import nnx
import optax

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.models import MLP
from shared.training_utils import compute_accuracy

# Weight on the load-balancing auxiliary loss (task CE dominates the total).
AUX_COEF = 0.01


# ============================================================================
# SYNTHETIC DATA: a Gaussian mixture of `n_classes` well-separated blobs
# ============================================================================
# Fully offline. Each class is a Gaussian blob around its own random mean, so
# different regions of input space carry different labels -- exactly the kind of
# piecewise structure a Mixture of Experts can carve up between specialists.

def make_dataset(n_samples: int, in_features: int, n_classes: int, key):
    k_mean, k_label, k_noise = jax.random.split(key, 3)
    means = jax.random.normal(k_mean, (n_classes, in_features)) * 3.0
    labels = jax.random.randint(k_label, (n_samples,), 0, n_classes)
    noise = jax.random.normal(k_noise, (n_samples, in_features))
    x = means[labels] + noise
    return {"x": x, "y": labels}


# ============================================================================
# THE MoE LAYER: gate + top-k routing + weighted combination + balance loss
# ============================================================================

class MoELayer(nnx.Module):
    r"""Sparse Mixture of Experts over a `d_model`-dim token representation.

    A linear gate scores every expert; each token keeps only its top-`k`
    experts (softmax-weighted), and their outputs are summed. We compute all
    experts densely and mask by the top-k selection -- simple and fast for the
    small expert counts used on CPU.
    """

    def __init__(self, d_model: int, d_hidden: int, n_experts: int, k: int, *, rngs: nnx.Rngs):
        self.n_experts = n_experts
        self.k = k
        # Routing logits: one score per expert. No bias -> pure content routing.
        self.gate = nnx.Linear(d_model, n_experts, use_bias=False, rngs=rngs)
        # Experts are independent MLPs (d_model -> d_hidden -> d_model).
        # MUST be an nnx.List so the submodules register as state on Flax 0.12.
        self.experts = nnx.List([
            MLP(d_model, d_hidden, d_model, n_layers=2, rngs=rngs)
            for _ in range(n_experts)
        ])

    def __call__(self, x: jax.Array, train: bool = False):
        # x: (B, d_model)
        gate_logits = self.gate(x)                              # (B, E)
        router_probs = jax.nn.softmax(gate_logits, axis=-1)     # (B, E) full softmax

        # --- top-k routing ---
        top_vals, top_idx = jax.lax.top_k(gate_logits, self.k)  # (B, k), (B, k)
        top_gates = jax.nn.softmax(top_vals, axis=-1)           # (B, k) combine weights
        one_hot = jax.nn.one_hot(top_idx, self.n_experts)       # (B, k, E)
        # Scatter the k combine weights back onto a dense (B, E) gate matrix.
        dense_gates = jnp.einsum("bk,bke->be", top_gates, one_hot)  # (B, E)

        # --- run every expert, then combine the selected ones ---
        expert_out = jnp.stack(
            [expert(x, train=train) for expert in self.experts], axis=1
        )                                                        # (B, E, d_model)
        y = jnp.einsum("be,bed->bd", dense_gates, expert_out)    # (B, d_model)

        # --- load-balancing auxiliary loss (Switch-Transformer style) ---
        # f_i: fraction of the B*k routing slots that landed on expert i.
        # P_i: mean router probability mass on expert i.
        selection = one_hot.sum(axis=1)                          # (B, E) in {0,1}
        f = selection.sum(axis=0) / (x.shape[0] * self.k)        # (E,) sums to 1
        p = router_probs.mean(axis=0)                            # (E,) sums to 1
        balance_loss = self.n_experts * jnp.sum(f * p)           # minimized (=1) when uniform

        return y, balance_loss, f


# ============================================================================
# THE CLASSIFIER: stem -> MoE (residual + norm) -> linear head
# ============================================================================

class MoEClassifier(nnx.Module):
    """Transformer-FFN-style block: a residual MoE layer feeding a class head."""

    def __init__(
        self,
        in_features: int,
        d_model: int,
        d_hidden: int,
        n_experts: int,
        k: int,
        n_classes: int,
        *,
        rngs: nnx.Rngs,
    ):
        self.stem = nnx.Linear(in_features, d_model, rngs=rngs)
        self.moe = MoELayer(d_model, d_hidden, n_experts, k, rngs=rngs)
        self.norm = nnx.LayerNorm(d_model, rngs=rngs)
        self.head = nnx.Linear(d_model, n_classes, rngs=rngs)

    def __call__(self, x: jax.Array, train: bool = False):
        h = nnx.relu(self.stem(x))                  # (B, d_model)
        moe_out, balance_loss, util = self.moe(h, train=train)
        h = self.norm(h + moe_out)                  # residual around the MoE block
        logits = self.head(h)                       # (B, n_classes)
        return logits, balance_loss, util


def create_model(rngs: nnx.Rngs, *, in_features=32, n_classes=6, n_experts=4, k=2):
    return MoEClassifier(
        in_features=in_features,
        d_model=64,
        d_hidden=128,
        n_experts=n_experts,
        k=k,
        n_classes=n_classes,
        rngs=rngs,
    )


# ============================================================================
# TRAIN STEP: total loss = task CE + AUX_COEF * balance_loss
# ============================================================================

@nnx.jit
def train_step(model: MoEClassifier, optimizer: nnx.Optimizer, batch):
    def loss_fn(model):
        logits, balance_loss, util = model(batch["x"], train=True)
        ce = optax.softmax_cross_entropy_with_integer_labels(logits, batch["y"]).mean()
        total = ce + AUX_COEF * balance_loss
        return total, (ce, balance_loss, util)

    (total, (ce, balance_loss, util)), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    optimizer.update(model, grads)
    return total, ce, balance_loss, util


# ============================================================================
# MAIN
# ============================================================================

def _util_bar(util) -> str:
    """One-line text histogram of expert utilization (fraction of routing slots)."""
    return "  ".join(f"e{i}:{float(u):.2f}" for i, u in enumerate(util))


def save_utilization_plot(util, n_experts: int, k: int, path: str):
    """Bar chart of expert utilization (matplotlib imported lazily; Agg backend).

    Each bar is the fraction of a batch's B*k routing slots that landed on that
    expert. A dashed line marks the uniform 1/E target: with the load-balancing
    aux loss the bars sit close to it (no dead experts) -- exactly what the aux
    loss buys. Values are clipped/normalized for display only. Returns `path`.
    """
    import os
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frac = np.asarray(util, dtype=float)
    ids = np.arange(n_experts)
    uniform = 1.0 / n_experts

    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.bar(ids, frac, color="#4C72B0", edgecolor="white", width=0.7, zorder=3)
    ax.axhline(uniform, ls="--", color="#C44E52", lw=1.8, zorder=4,
               label=f"uniform target 1/E = {uniform:.2f}")

    for b, v in zip(bars, frac):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.006, f"{v:.2f}",
                ha="center", va="bottom", fontsize=10)

    ax.set_xlabel("expert id")
    ax.set_ylabel("routing fraction (share of B·k slots)")
    ax.set_title(f"MoE expert utilization after training (top-{k}-of-{n_experts})")
    ax.set_xticks(ids)
    ax.set_ylim(0, max(float(frac.max()) * 1.25, uniform * 1.6))
    ax.grid(axis="y", ls=":", alpha=0.5, zorder=0)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()

    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    epochs = int(os.environ.get("EPOCHS", 300))
    batch_size = int(os.environ.get("BATCH", 512))
    _ = os.environ.get("SYNTHETIC", "1")  # dataset is synthetic by construction

    in_features, n_classes, n_experts, k = 32, 6, 4, 2
    data = make_dataset(batch_size, in_features, n_classes, jax.random.key(0))

    model = create_model(nnx.Rngs(0), in_features=in_features, n_classes=n_classes,
                         n_experts=n_experts, k=k)
    optimizer = nnx.Optimizer(model, optax.adam(3e-3), wrt=nnx.Param)

    print(f"Training a top-{k}-of-{n_experts} MoE classifier for {epochs} steps "
          f"({n_classes} classes, batch {batch_size})\n")
    for step in range(epochs):
        total, ce, bal, util = train_step(model, optimizer, data)
        if step % 50 == 0 or step == epochs - 1:
            acc = float(compute_accuracy(model(data["x"])[0], data["y"]))
            print(f"step {step:4d} | total {float(total):.4f} | ce {float(ce):.4f} | "
                  f"balance {float(bal):.3f} | acc {acc:.3f}")

    _, _, util = model(data["x"])
    print("\nFinal expert utilization (fraction of routing slots):")
    print("  " + _util_bar(util))

    out_path = os.path.join(os.environ.get("OUTDIR", "results"), "moe_utilization.png")
    save_utilization_plot(util, n_experts, k, out_path)
    print(f"\nSaved expert-utilization plot to {out_path}")


if __name__ == "__main__":
    main()
