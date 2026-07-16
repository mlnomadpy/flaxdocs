"""
RealNVP Normalizing Flow on Two Moons
=====================================
Learn an exact, invertible density with Flax NNX. A stack of affine coupling
layers maps the two-moons data distribution to a standard-normal base; the
tractable log-determinant lets us train by exact maximum likelihood and both
sample (base -> data) and score (data -> base).

Run: python generative/normalizing_flows.py
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

from shared.models import MLP


# ==== AFFINE COUPLING LAYER ====

class AffineCoupling(nnx.Module):
    """RealNVP affine coupling layer.

    A binary `parity` mask keeps half the dimensions unchanged and uses them to
    predict a per-dimension scale ``s`` and shift ``t`` for the other half:

        forward:  z = x_kept + (1-mask) * (x * exp(s) + t),   logdet = sum(s)
        inverse:  x = z_kept + (1-mask) * (z - t) * exp(-s)

    The masked (kept) half passes through unchanged, so the Jacobian is
    triangular and its log-determinant is just ``sum(s)`` over transformed dims.
    Alternating `parity` between stacked layers lets every dimension be
    transformed. ``s`` is squashed with ``tanh`` to keep the scale bounded.
    """

    def __init__(self, dim: int, hidden: int, parity: int, *, rngs: nnx.Rngs):
        self.dim = dim
        self.parity = parity
        # Conditioner: (kept dims) -> concat[s_raw, t], each of size `dim`.
        self.net = MLP(dim, hidden, 2 * dim, n_layers=2, rngs=rngs, activation="gelu")

    def _mask(self):
        # 1.0 where a dim is kept (conditions), 0.0 where it is transformed.
        return (jnp.arange(self.dim) % 2 == self.parity).astype(jnp.float32)

    def _s_t(self, u, mask):
        # Condition only on the kept dims; zero out the rest of the input.
        s_raw, t = jnp.split(self.net(u * mask), 2, axis=-1)
        s = jnp.tanh(s_raw) * (1.0 - mask)   # scale only on transformed dims
        t = t * (1.0 - mask)                 # shift only on transformed dims
        return s, t

    def forward(self, x):
        """Data -> latent, returning (z, logdet) with logdet of shape (B,)."""
        mask = self._mask()
        s, t = self._s_t(x, mask)
        z = x * mask + (1.0 - mask) * (x * jnp.exp(s) + t)
        logdet = jnp.sum(s, axis=-1)
        return z, logdet

    def inverse(self, z):
        """Latent -> data (exact inverse of `forward`)."""
        mask = self._mask()
        s, t = self._s_t(z, mask)  # kept dims are identical to x, so s,t match
        x = z * mask + (1.0 - mask) * (z - t) * jnp.exp(-s)
        return x


# ==== REALNVP: A STACK OF COUPLING LAYERS ====

class RealNVP(nnx.Module):
    """A normalizing flow: composition of alternating affine coupling layers."""

    def __init__(self, dim: int = 2, hidden: int = 64, n_layers: int = 6, *, rngs: nnx.Rngs):
        self.dim = dim
        # Plain lists of submodules MUST be wrapped in nnx.List on Flax 0.12.
        self.layers = nnx.List([
            AffineCoupling(dim, hidden, parity=i % 2, rngs=rngs)
            for i in range(n_layers)
        ])

    def forward(self, x):
        """Data -> base z, accumulating the total log|det dz/dx|."""
        logdet = jnp.zeros(x.shape[0])
        z = x
        for layer in self.layers:
            z, ld = layer.forward(z)
            logdet = logdet + ld
        return z, logdet

    def inverse(self, z):
        """Base z -> data x, applying the layers in reverse order."""
        x = z
        for i in range(len(self.layers) - 1, -1, -1):
            x = self.layers[i].inverse(x)
        return x

    def log_prob(self, x):
        """Exact log-density via change of variables under a N(0, I) base."""
        z, logdet = self.forward(x)
        # log N(z; 0, I) = -0.5 * ||z||^2 - 0.5 * d * log(2*pi)
        logpz = -0.5 * jnp.sum(z ** 2, axis=-1) - 0.5 * self.dim * jnp.log(2.0 * jnp.pi)
        return logpz + logdet

    def sample(self, n: int, seed: int = 0):
        """Draw z ~ N(0, I) and push it through the inverse flow to data space."""
        z = jax.random.normal(jax.random.key(seed), (n, self.dim))
        return self.inverse(z)


# ==== LOSS: NEGATIVE LOG-LIKELIHOOD ====

def nll_loss(model: RealNVP, x):
    """Exact maximum-likelihood objective: mean negative log-density."""
    return -model.log_prob(x).mean()


# ==== TRAIN STEP ====

@nnx.jit
def train_step(model: RealNVP, optimizer: nnx.Optimizer, batch):
    def loss_fn(model):
        nll = nll_loss(model, batch)
        return nll, nll  # aux = nll (scalar) for logging

    (loss, aux), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    optimizer.update(model, grads)
    return loss, aux


# ==== DATA: SYNTHETIC TWO MOONS (NO SKLEARN) ====

def make_two_moons(n: int = 1024, noise: float = 0.1, seed: int = 0):
    """Sample the two-moons distribution as two noisy half circles.

    Fully offline: outer moon is the upper half circle, inner moon is a shifted
    lower half circle. Returns standardized points of shape (n, 2) so the
    N(0, I) base is a good match.
    """
    key = jax.random.key(seed)
    k1, k2, k3 = jax.random.split(key, 3)
    n_out = n // 2
    n_in = n - n_out

    t_out = jnp.pi * jax.random.uniform(k1, (n_out,))
    outer = jnp.stack([jnp.cos(t_out), jnp.sin(t_out)], axis=1)

    t_in = jnp.pi * jax.random.uniform(k2, (n_in,))
    inner = jnp.stack([1.0 - jnp.cos(t_in), 1.0 - jnp.sin(t_in) - 0.5], axis=1)

    x = jnp.concatenate([outer, inner], axis=0)
    x = x + noise * jax.random.normal(k3, x.shape)
    # Standardize to zero mean / unit variance per dim.
    x = (x - x.mean(0)) / (x.std(0) + 1e-6)
    return x


# ==== VISUALIZATION ====

def plot_samples(real, samples, path):
    """Side-by-side scatter of real two-moons vs. samples from the trained flow.

    matplotlib is imported lazily (Agg backend) so importing this module stays
    cheap and the plot renders without a display. If the flow has learned the
    density, both panels trace the same two crescents.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import numpy as np
    real = np.asarray(real)
    samples = np.asarray(samples)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, pts, title, color in (
        (axes[0], real, "Real data (two moons)", "#1f77b4"),
        (axes[1], samples, "Samples from trained flow", "#d62728"),
    ):
        ax.scatter(pts[:, 0], pts[:, 1], s=8, alpha=0.6, color=color, edgecolors="none")
        ax.set_title(title)
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.2)

    # Share the same axis limits so the two distributions are directly comparable.
    lo = min(real.min(), samples.min()) - 0.3
    hi = max(real.max(), samples.max()) + 0.3
    for ax in axes:
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)

    fig.suptitle("RealNVP normalizing flow: learned density matches the data")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved visualization to {path}")


# ==== MAIN ====

def main():
    epochs = int(os.environ.get("EPOCHS", 8))
    batch = int(os.environ.get("BATCH", 128))
    n = int(os.environ.get("SYNTHETIC", 1024))  # dataset size (synthetic by default)

    rngs = nnx.Rngs(0)
    model = RealNVP(dim=2, hidden=64, n_layers=6, rngs=rngs)
    optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)

    data = make_two_moons(n=n, seed=0)

    for epoch in range(epochs):
        perm = jax.random.permutation(jax.random.key(epoch), n)
        data = data[perm]
        total, steps = 0.0, 0
        for i in range(0, n, batch):
            xb = data[i:i + batch]
            loss, _ = train_step(model, optimizer, xb)
            total += float(loss)
            steps += 1
        print(f"epoch {epoch + 1}/{epochs}  nll {total / steps:.4f}")

    samples = model.sample(n, seed=0)
    z, _ = model.forward(data[:256])
    print(f"samples: {samples.shape}  latent: {z.shape}")

    out_path = os.path.join(os.environ.get("OUTDIR", "results"), "flows_samples.png")
    plot_samples(data, samples, out_path)


if __name__ == "__main__":
    main()
