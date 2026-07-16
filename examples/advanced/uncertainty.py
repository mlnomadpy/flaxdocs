"""
Flax NNX: Uncertainty Estimation (MC-Dropout + Deep Ensembles)
==============================================================
Estimate predictive uncertainty on a 1D heteroscedastic regression task and
decompose it into aleatoric (data noise) and epistemic (model) components.
Both a single Monte-Carlo Dropout network and a vmapped deep ensemble show
uncertainty growing away from the training data.

Run: python advanced/uncertainty.py
"""

import os
import jax
import jax.numpy as jnp
from flax import nnx
import optax

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# ============================================================================
# 1. SYNTHETIC HETEROSCEDASTIC 1D DATA
# ============================================================================
# Training inputs live in two clusters, [-4, -2] and [2, 4], leaving a gap in
# the middle and unobserved regions in the tails. A well-calibrated model
# should report LOW uncertainty inside the clusters and HIGH uncertainty in
# the gap and the extrapolation regions.


def true_fn(x):
    """Underlying clean function (no noise)."""
    return jnp.sin(1.5 * x) + 0.1 * x


def noise_std(x):
    """Heteroscedastic aleatoric noise: larger for larger |x|."""
    return 0.05 + 0.05 * jnp.abs(x)


def make_regression_data(key, n_per_cluster=60):
    """Sample training data from two clusters with input-dependent noise."""
    k1, k2, k3 = jax.random.split(key, 3)
    x_left = jax.random.uniform(k1, (n_per_cluster, 1), minval=-4.0, maxval=-2.0)
    x_right = jax.random.uniform(k2, (n_per_cluster, 1), minval=2.0, maxval=4.0)
    x = jnp.concatenate([x_left, x_right], axis=0)
    eps = jax.random.normal(k3, x.shape) * noise_std(x)
    y = true_fn(x) + eps
    return x, y


# ============================================================================
# 2. GAUSSIAN MLP WITH DROPOUT (mean + variance heads)
# ============================================================================
# The network predicts a Gaussian per input: a mean mu(x) and a log-variance
# s(x) = log sigma^2(x). The variance head captures ALEATORIC noise; the
# dropout layers make the network stochastic so repeated forward passes
# reveal EPISTEMIC uncertainty (MC-Dropout).


class GaussianMLP(nnx.Module):
    """MLP that outputs (mu, log_var) with dropout for MC sampling."""

    def __init__(
        self,
        in_features: int = 1,
        hidden: int = 64,
        dropout_rate: float = 0.1,
        *,
        rngs: nnx.Rngs,
    ):
        self.l1 = nnx.Linear(in_features, hidden, rngs=rngs)
        self.l2 = nnx.Linear(hidden, hidden, rngs=rngs)
        self.drop = nnx.Dropout(dropout_rate, rngs=rngs)
        self.mu_head = nnx.Linear(hidden, 1, rngs=rngs)
        self.logvar_head = nnx.Linear(hidden, 1, rngs=rngs)

    def __call__(self, x, train: bool = False):
        h = nnx.relu(self.l1(x))
        h = self.drop(h, deterministic=not train)
        h = nnx.relu(self.l2(h))
        h = self.drop(h, deterministic=not train)
        mu = self.mu_head(h)
        # Clamp log-variance for numerical stability (sigma^2 in ~[e^-6, e^4]).
        log_var = jnp.clip(self.logvar_head(h), -6.0, 4.0)
        return mu, log_var


# ============================================================================
# 3. GAUSSIAN NEGATIVE LOG-LIKELIHOOD LOSS
# ============================================================================
# Training a mean+variance head with the Gaussian NLL lets the network learn
# input-dependent (heteroscedastic) noise instead of a single global sigma.


def gaussian_nll(mu, log_var, y):
    """Mean Gaussian negative log-likelihood (drops the constant term)."""
    inv_var = jnp.exp(-log_var)
    return 0.5 * jnp.mean(log_var + (y - mu) ** 2 * inv_var)


# ============================================================================
# 4. SINGLE-MODEL TRAIN STEP (used for MC-Dropout)
# ============================================================================


@nnx.jit
def train_step(model: GaussianMLP, optimizer: nnx.Optimizer, batch):
    def loss_fn(model):
        mu, log_var = model(batch["x"], train=True)
        loss = gaussian_nll(mu, log_var, batch["y"])
        return loss, mu

    (loss, mu), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    optimizer.update(model, grads)
    return loss, mu


# ============================================================================
# 5. MC-DROPOUT INFERENCE
# ============================================================================
# Keep dropout ON at test time (train=True) and run T stochastic passes. Each
# call advances the dropout RNG, so the T predictions differ. Decompose:
#     predictive mean = mean_t mu_t
#     aleatoric       = mean_t sigma_t^2          (average predicted noise)
#     epistemic       = var_t(mu_t)               (disagreement across passes)
#     total           = aleatoric + epistemic


@nnx.jit
def _mc_forward(model, x):
    return model(x, train=True)  # dropout active


def mc_predict(model, x, n_samples: int = 30):
    mus, variances = [], []
    for _ in range(n_samples):
        mu, log_var = _mc_forward(model, x)
        mus.append(mu)
        variances.append(jnp.exp(log_var))
    mus = jnp.stack(mus)            # (T, N, 1)
    variances = jnp.stack(variances)
    mean = mus.mean(axis=0)
    aleatoric = variances.mean(axis=0)
    epistemic = mus.var(axis=0)
    return {
        "mean": mean,
        "aleatoric": aleatoric,
        "epistemic": epistemic,
        "total": aleatoric + epistemic,
        "samples": mus,
    }


# ============================================================================
# 6. DEEP ENSEMBLE via nnx.vmap
# ============================================================================
# Build M independently-initialized networks (and their optimizers) with a
# single vmapped constructor, then train all M in one vmapped step. Each member
# sees a bootstrap resample of the data, so members disagree where data is
# scarce -> epistemic uncertainty.


@nnx.vmap(in_axes=0, out_axes=0)
def make_ensemble(key):
    """Build one independently-initialized member + its optimizer per key."""
    model = GaussianMLP(dropout_rate=0.1, rngs=nnx.Rngs(key))
    optimizer = nnx.Optimizer(model, optax.adam(1e-2), wrt=nnx.Param)
    return model, optimizer


@nnx.vmap(in_axes=(0, 0, 0), out_axes=0)
def ensemble_train_step(model, optimizer, batch):
    def loss_fn(model):
        mu, log_var = model(batch["x"], train=True)
        return gaussian_nll(mu, log_var, batch["y"])

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss


@nnx.vmap(in_axes=(0, None), out_axes=0)
def _ensemble_forward(model, x):
    return model(x, train=False)  # deterministic; diversity comes from params


def ensemble_predict(models, x):
    mu, log_var = _ensemble_forward(models, x)  # each (M, N, 1)
    variances = jnp.exp(log_var)
    mean = mu.mean(axis=0)
    aleatoric = variances.mean(axis=0)
    epistemic = mu.var(axis=0)
    return {
        "mean": mean,
        "aleatoric": aleatoric,
        "epistemic": epistemic,
        "total": aleatoric + epistemic,
        "members": mu,
    }


def bootstrap_batches(key, x, y, n_models):
    """Resample (x, y) with replacement for each ensemble member -> (M, N, 1)."""
    n = x.shape[0]
    keys = jax.random.split(key, n_models)
    idx = jax.vmap(lambda k: jax.random.randint(k, (n,), 0, n))(keys)  # (M, N)
    return x[idx], y[idx]


# ============================================================================
# 7. MAIN
# ============================================================================


def _region_uncertainty(x_grid, epistemic):
    """Average epistemic uncertainty inside vs. outside the training clusters."""
    x = x_grid.ravel()
    e = epistemic.ravel()
    in_cluster = ((x >= -4) & (x <= -2)) | ((x >= 2) & (x <= 4))
    out_region = ~in_cluster
    return float(e[in_cluster].mean()), float(e[out_region].mean())


# ============================================================================
# 7b. VISUALIZATION: the iconic uncertainty band
# ============================================================================
# One panel per method. Each shows the training scatter, the true function,
# the predictive MEAN as a line, and a shaded +/-2 sigma band. sigma is the
# EPISTEMIC (model-disagreement) standard deviation -- the confidence band on
# the learned function -- which collapses to near-zero over the two data
# clusters and BALLOONS in the middle gap and the extrapolation tails. That
# narrow-on-data / wide-in-the-gap shape is the whole point of estimating
# uncertainty. matplotlib is imported lazily so importing this module stays
# cheap.


def save_uncertainty_plot(x_train, y_train, x_grid, mc, ens, path):
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = np.asarray(x_grid).ravel()
    xt = np.asarray(x_train).ravel()
    yt = np.asarray(y_train).ravel()
    truth = np.asarray(true_fn(x_grid)).ravel()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)
    specs = [
        ("MC-Dropout (T stochastic passes)", mc, "#1f77b4"),
        ("Deep Ensemble (nnx.vmap members)", ens, "#d62728"),
    ]

    for ax, (title, pred, color) in zip(axes, specs):
        mean = np.asarray(pred["mean"]).ravel()
        # Band = +/-2 * epistemic std (confidence on the learned function).
        sigma = np.sqrt(np.asarray(pred["epistemic"]).ravel())

        # Shade the two training clusters so "in-cluster vs gap" is obvious.
        for lo, hi in [(-4, -2), (2, 4)]:
            ax.axvspan(lo, hi, color="0.9", zorder=0)

        # +/-2 sigma epistemic band -- narrow on data, wide in the gap/tails.
        ax.fill_between(
            xs, mean - 2 * sigma, mean + 2 * sigma,
            color=color, alpha=0.25,
            label=r"mean $\pm 2\sigma$ (epistemic)",
        )
        ax.plot(xs, truth, "k--", lw=1.5, alpha=0.7, label="true function")
        ax.plot(xs, mean, color=color, lw=2.2, label="predictive mean")
        ax.scatter(xt, yt, s=14, color="0.25", alpha=0.6,
                   zorder=5, label="training data")

        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_ylim(-3.0, 3.0)
        ax.set_xlim(-7.0, 7.0)
        ax.legend(loc="upper center", fontsize=8, framealpha=0.9, ncol=2)

    fig.suptitle(
        "Predictive uncertainty grows in the gap and the extrapolation tails",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved uncertainty band plot -> {path}")


def main():
    steps = int(os.environ.get("EPOCHS", 500))
    batch_size = int(os.environ.get("BATCH", 64))
    n_models = int(os.environ.get("N_MODELS", 5))
    n_samples = int(os.environ.get("MC_SAMPLES", 30))
    _ = os.environ.get("SYNTHETIC", "1")  # data is always synthetic/offline

    print("=" * 70)
    print("UNCERTAINTY ESTIMATION: MC-Dropout + Deep Ensembles")
    print("=" * 70)

    key = jax.random.key(0)
    dkey, tkey, bkey = jax.random.split(key, 3)
    x_train, y_train = make_regression_data(dkey)
    print(f"Training points: {x_train.shape[0]} (two clusters, heteroscedastic noise)")

    x_grid = jnp.linspace(-7.0, 7.0, 200).reshape(-1, 1)

    # ---- MC-Dropout: train ONE stochastic network --------------------------
    print("\n[1/2] Training a single MC-Dropout network...")
    model = GaussianMLP(dropout_rate=0.1, rngs=nnx.Rngs(0))
    optimizer = nnx.Optimizer(model, optax.adam(1e-2), wrt=nnx.Param)
    for step in range(steps):
        bkey, sub = jax.random.split(bkey)
        idx = jax.random.randint(sub, (batch_size,), 0, x_train.shape[0])
        loss, _ = train_step(model, optimizer, {"x": x_train[idx], "y": y_train[idx]})
        if (step + 1) % max(1, steps // 5) == 0:
            print(f"  step {step + 1:4d}/{steps} | NLL {float(loss):.4f}")

    mc = mc_predict(model, x_grid, n_samples=n_samples)
    mc_in, mc_out = _region_uncertainty(x_grid, mc["epistemic"])
    print(f"  MC-Dropout epistemic: in-cluster {mc_in:.5f} | away {mc_out:.5f}")

    # ---- Deep Ensemble: train M networks with one vmapped step -------------
    print(f"\n[2/2] Training a deep ensemble of {n_models} networks (nnx.vmap)...")
    keys = jax.random.split(tkey, n_models)
    models, optimizers = make_ensemble(keys)
    xb, yb = bootstrap_batches(bkey, x_train, y_train, n_models)
    ens_batch = {"x": xb, "y": yb}
    for step in range(steps * 2):
        losses = ensemble_train_step(models, optimizers, ens_batch)
        if (step + 1) % max(1, (steps * 2) // 5) == 0:
            print(f"  step {step + 1:4d}/{steps * 2} | mean NLL {float(losses.mean()):.4f}")

    ens = ensemble_predict(models, x_grid)
    ens_in, ens_out = _region_uncertainty(x_grid, ens["epistemic"])
    print(f"  Ensemble epistemic:   in-cluster {ens_in:.5f} | away {ens_out:.5f}")

    # ---- Visualization: the iconic uncertainty band ------------------------
    out_path = os.path.join(os.environ.get("OUTDIR", "results"), "uncertainty_band.png")
    save_uncertainty_plot(x_train, y_train, x_grid, mc, ens, out_path)

    print("\n" + "=" * 70)
    print("Uncertainty grows away from the training data for BOTH methods.")
    print("Aleatoric captures data noise; epistemic captures model ignorance.")
    print("=" * 70)


if __name__ == "__main__":
    main()
