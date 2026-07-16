"""
LoRA Parameter-Efficient Fine-Tuning
====================================
Adapt a pretrained MLP to a new task by training only tiny low-rank adapters
while the base weights stay frozen (bit-identical), using nnx.LoRALinear.

Run: python adaptation/lora_finetuning.py
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

from shared.training_utils import compute_mse_loss


# Filter that selects the FROZEN base weights: everything that is an nnx.Param
# but NOT an nnx.LoRAParam. nnx.LoRAParam subclasses nnx.Param, so we must
# subtract it explicitly (intersection via nnx.All, not the comma form which
# would return two separate states).
BASE_PARAMS = nnx.All(nnx.Param, nnx.Not(nnx.LoRAParam))


# ==== MODEL ====

class LoRAMLP(nnx.Module):
    """Two-layer MLP whose Linear layers carry low-rank adapters.

    Each nnx.LoRALinear holds a frozen base kernel/bias (nnx.Param) plus a
    low-rank update A @ B stored as nnx.LoRAParam. At init B = 0, so the
    adapter is a no-op and the module behaves exactly like a plain Linear.
    """

    def __init__(self, in_dim: int, hidden: int, out_dim: int, rank: int,
                 *, rngs: nnx.Rngs):
        self.l1 = nnx.LoRALinear(in_dim, hidden, lora_rank=rank, rngs=rngs)
        self.l2 = nnx.LoRALinear(hidden, out_dim, lora_rank=rank, rngs=rngs)

    def __call__(self, x):
        return self.l2(nnx.relu(self.l1(x)))


# ==== DATA ====

def make_dataset(synthetic: bool = True, n: int = 256, in_dim: int = 8,
                 out_dim: int = 4, seed: int = 0):
    """Synthetic regression: shared inputs X, two targets for two tasks.

    Task A is a random linear map Y_A = X @ W_A. Task B is the SAME inputs but
    a rotated + shifted target Y_B = (Y_A @ R) + b, i.e. a domain shift of A.
    This example is intrinsically synthetic (no downloads); `synthetic=False`
    just draws a larger sample so the script always runs offline on CPU.

    Returns tiny jnp arrays: (X, Y_A, Y_B).
    """
    if not synthetic:
        n = max(n, 1024)
    key = jax.random.PRNGKey(seed)
    kx, ka, kr, kb = jax.random.split(key, 4)

    X = jax.random.normal(kx, (n, in_dim))
    W_a = jax.random.normal(ka, (in_dim, out_dim))
    Y_a = X @ W_a

    # Task B target = task A rotated (orthogonal R) and shifted by b.
    R = jnp.linalg.qr(jax.random.normal(kr, (out_dim, out_dim)))[0]
    b = jax.random.normal(kb, (out_dim,))
    Y_b = Y_a @ R + b
    return X, Y_a, Y_b


# ==== PARAM UTILITIES ====

def count_params(model: nnx.Module, filt) -> int:
    """Total number of scalar parameters selected by `filt`."""
    return int(sum(x.size for x in jax.tree.leaves(nnx.state(model, filt))))


def snapshot(model: nnx.Module, filt):
    """Deep copy of the arrays selected by `filt` (for before/after asserts)."""
    return jax.tree.map(lambda a: jnp.array(a), nnx.state(model, filt))


def trees_equal(a, b) -> bool:
    """True iff every leaf pair is bit-identical."""
    return all(bool(jnp.array_equal(x, y))
               for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)))


# ==== TRAIN STEPS ====

@nnx.jit
def pretrain_step(model, optimizer, batch):
    """Full fine-tuning on task A: trains ALL params (base + adapters)."""
    def loss_fn(model):
        preds = model(batch['x'])
        loss = compute_mse_loss(preds, batch['y'])
        return loss, preds
    (loss, preds), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    optimizer.update(model, grads)
    return loss, preds


@nnx.jit
def adapt_step(model, optimizer, batch):
    """Adapter-only fine-tuning on task B: trains ONLY nnx.LoRAParam.

    Requires BOTH (1) an optimizer built with wrt=nnx.LoRAParam and (2) a
    gradient restricted to the LoRAParam subtree via nnx.DiffState. Because
    LoRAParam subclasses Param, the default wrt=nnx.Param would also move the
    base weights.
    """
    def loss_fn(model):
        preds = model(batch['x'])
        loss = compute_mse_loss(preds, batch['y'])
        return loss, preds
    (loss, preds), grads = nnx.value_and_grad(
        loss_fn, argnums=nnx.DiffState(0, nnx.LoRAParam), has_aux=True)(model)
    optimizer.update(model, grads)
    return loss, preds


# ==== MAIN ====

def main():
    epochs = int(os.environ.get("EPOCHS", "1"))
    batch = int(os.environ.get("BATCH", "128"))
    synthetic = os.environ.get("SYNTHETIC", "1") != "0"
    pretrain_steps = int(os.environ.get("PRETRAIN_STEPS", "200"))
    adapt_steps = int(os.environ.get("ADAPT_STEPS", "200"))

    in_dim, hidden, out_dim, rank = 8, 64, 4, 4
    X, Y_a, Y_b = make_dataset(synthetic=synthetic, n=batch, in_dim=in_dim,
                               out_dim=out_dim)

    rngs = nnx.Rngs(0)
    model = LoRAMLP(in_dim, hidden, out_dim, rank, rngs=rngs)

    n_lora = count_params(model, nnx.LoRAParam)
    n_base = count_params(model, BASE_PARAMS)
    total = n_lora + n_base
    print(f"Base (frozen) params : {n_base}")
    print(f"LoRA (trainable) params: {n_lora}  ({100.0 * n_lora / total:.1f}% of total)")

    # ---- Phase 1: pretrain the whole model on task A (wrt=nnx.Param) ----
    pre_opt = nnx.Optimizer(model, optax.adam(1e-2), wrt=nnx.Param)
    print("\n[pretrain task A]")
    for epoch in range(epochs):
        for step in range(pretrain_steps):
            loss, _ = pretrain_step(model, pre_opt, {'x': X, 'y': Y_a})
        print(f"  epoch {epoch}: task-A MSE = {float(loss):.4f}")

    # ---- Phase 2: freeze base, adapt to task B (wrt=nnx.LoRAParam) ----
    base_before = snapshot(model, BASE_PARAMS)
    lora_before = snapshot(model, nnx.LoRAParam)

    adapt_opt = nnx.Optimizer(model, optax.adam(1e-2), wrt=nnx.LoRAParam)
    print("\n[adapt task B — LoRA only]")
    b_losses = []
    for epoch in range(epochs):
        for step in range(adapt_steps):
            loss, _ = adapt_step(model, adapt_opt, {'x': X, 'y': Y_b})
            b_losses.append(float(loss))
        print(f"  epoch {epoch}: task-B MSE = {float(loss):.4f}")

    base_after = snapshot(model, BASE_PARAMS)
    lora_after = snapshot(model, nnx.LoRAParam)

    print("\n[assertions]")
    print(f"  task-B MSE decreased: {b_losses[-1]:.4f} < {b_losses[0]:.4f} -> "
          f"{b_losses[-1] < b_losses[0]}")
    print(f"  base weights bit-identical: {trees_equal(base_before, base_after)}")
    print(f"  LoRA weights changed: {not trees_equal(lora_before, lora_after)}")


if __name__ == "__main__":
    main()
