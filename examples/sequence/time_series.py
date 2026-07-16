"""
LSTM Time-Series Forecasting
============================
Windowed multi-step forecasting with an ``nnx.RNN(nnx.LSTMCell)`` encoder and a
linear head. The signal is a synthetic sum of sinusoids plus noise, sliced into
sliding windows: read ``L`` past steps, predict the next ``H`` steps (MSE).

Run: python sequence/time_series.py
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


# ============================================================================
# MODEL
# ============================================================================

class LSTMForecaster(nnx.Module):
    """Encode a window of ``L`` past values, forecast the next ``H`` values.

    ``nnx.RNN`` scans ``nnx.LSTMCell`` across the time axis of a
    ``(B, L, in_features)`` window and returns per-step hidden states
    ``(B, L, hidden)``. The last state summarizes the whole window; a
    ``nnx.Linear`` head maps it to the ``H``-step forecast (regression).
    """

    def __init__(self, in_features: int, hidden: int, horizon: int,
                 *, rngs: nnx.Rngs):
        self.rnn = nnx.RNN(nnx.LSTMCell(in_features, hidden, rngs=rngs))
        self.head = nnx.Linear(hidden, horizon, rngs=rngs)

    def __call__(self, x):
        h = self.rnn(x)             # (B, L, in_features) -> (B, L, hidden)
        return self.head(h[:, -1])  # last step           -> (B, horizon)


# ============================================================================
# DATA — synthetic sum-of-sinusoids series + sliding windows
# ============================================================================

def make_series(length: int, *, seed: int = 0) -> jax.Array:
    """Generate a 1-D signal: sum of sinusoids of different periods + noise.

    This is a self-contained stand-in for a real univariate series (energy
    load, temperature, sensor reading): several periodic components make it
    predictable, while the additive noise stops the model from memorizing.

    Returns:
        ``(length,)`` float32 series.
    """
    t = jnp.arange(length, dtype=jnp.float32)
    series = (
        1.0 * jnp.sin(2 * jnp.pi * t / 24.0)     # daily-like cycle
        + 0.5 * jnp.sin(2 * jnp.pi * t / 168.0)  # weekly-like cycle
        + 0.3 * jnp.sin(2 * jnp.pi * t / 7.0)    # short ripple
    )
    noise = 0.1 * jax.random.normal(jax.random.key(seed), (length,))
    return series + noise


def make_windows(series: jax.Array, window: int, horizon: int):
    """Slice a series into ``(input window -> next horizon steps)`` pairs.

    Returns:
        ``x: (N, window, 1)`` input windows and ``y: (N, horizon)`` targets,
        where ``N = len(series) - window - horizon + 1``.
    """
    n = series.shape[0] - window - horizon + 1
    starts = jnp.arange(n)
    x = jax.vmap(lambda s: jax.lax.dynamic_slice(series, (s,), (window,)))(starts)
    y = jax.vmap(
        lambda s: jax.lax.dynamic_slice(series, (s + window,), (horizon,))
    )(starts)
    return x[..., None], y  # (N, window, 1), (N, horizon)


def make_dataset(synthetic: bool = True, *, window: int = 48, horizon: int = 12,
                 seed: int = 0):
    """Build a normalized train/test split from one synthetic series.

    The series is split in time (past = train, future = test) *before*
    windowing, so the test windows are a genuine held-out continuation. We
    standardize using train statistics only, to avoid leaking the future.

    Returns:
        ``(x_tr, y_tr, x_te, y_te, stats)`` with ``stats = (mean, std)``.
    """
    length = 512 if synthetic else 2048
    series = make_series(length, seed=seed)

    split = int(0.8 * length)
    mean = series[:split].mean()
    std = series[:split].std() + 1e-6
    series = (series - mean) / std

    # Overlap the two halves by `window` so the first test target immediately
    # follows the training region without dropping any continuation steps.
    train_series = series[:split]
    test_series = series[split - window:]

    x_tr, y_tr = make_windows(train_series, window, horizon)
    x_te, y_te = make_windows(test_series, window, horizon)
    return x_tr, y_tr, x_te, y_te, (mean, std)


# ============================================================================
# TRAIN / EVAL STEPS
# ============================================================================

@nnx.jit
def train_step(model, optimizer, batch):
    def loss_fn(model):
        preds = model(batch["x"])            # (B, horizon)
        loss = compute_mse_loss(preds, batch["y"])
        return loss, preds

    (loss, preds), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    optimizer.update(model, grads)
    return loss, preds


@nnx.jit
def eval_mse(model, x, y):
    return compute_mse_loss(model(x), y)


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_forecast(history, forecast, truth, path: str):
    """Plot one held-out window: input history, then forecast vs. ground truth.

    The solid line is the ``L`` observed lookback values the model reads. Past
    the forecast origin (dashed vertical line) two lines are overlaid: the
    model's ``H``-step forecast and the true continuation. If the forecast
    tracks the truth, the model has genuinely learned the series' structure.

    Args:
        history: ``(L,)`` observed input window (standardized units).
        forecast: ``(H,)`` model prediction for the next ``H`` steps.
        truth: ``(H,)`` ground-truth continuation.
        path: Destination PNG path.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    history = np.asarray(history)
    forecast = np.asarray(forecast)
    truth = np.asarray(truth)

    L, H = history.shape[0], forecast.shape[0]
    hist_t = np.arange(L)
    fut_t = np.arange(L, L + H)
    # Prepend the last observed point so the forecast/truth lines connect to
    # the history without a visual gap at the origin.
    bridge_t = np.arange(L - 1, L + H)
    fc_line = np.concatenate([history[-1:], forecast])
    tr_line = np.concatenate([history[-1:], truth])

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(hist_t, history, color="#333333", lw=2,
            label=f"Input history (L={L})")
    ax.plot(bridge_t, tr_line, color="#2ca02c", lw=2.2, marker="o", ms=4,
            label="Ground truth")
    ax.plot(bridge_t, fc_line, color="#d62728", lw=2.2, marker="x", ms=6,
            ls="--", label="LSTM forecast")
    ax.axvline(L - 0.5, color="#888888", ls=":", lw=1.5)
    ax.text(L - 0.5, ax.get_ylim()[1], " forecast origin",
            va="top", ha="left", color="#555555", fontsize=9)

    mse = float(np.mean((forecast - truth) ** 2))
    ax.set_title(f"Multi-step forecast on a held-out window "
                 f"(H={H}, MSE={mse:.4f})")
    ax.set_xlabel("time step")
    ax.set_ylabel("value (standardized)")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"saved forecast plot to {path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    epochs = int(os.environ.get("EPOCHS", 40))
    batch_size = int(os.environ.get("BATCH", 64))
    synthetic = os.environ.get("SYNTHETIC", "1") != "0"

    window, horizon, hidden = 48, 12, 64

    x_tr, y_tr, x_te, y_te, _ = make_dataset(
        synthetic=synthetic, window=window, horizon=horizon)
    n = x_tr.shape[0]

    rngs = nnx.Rngs(0)
    model = LSTMForecaster(in_features=1, hidden=hidden, horizon=horizon, rngs=rngs)
    tx = optax.adam(5e-3)
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

    print(f"train_windows={n} test_windows={x_te.shape[0]} "
          f"window={window} horizon={horizon} epochs={epochs} batch={batch_size}")

    for epoch in range(epochs):
        perm = jax.random.permutation(jax.random.key(epoch), n)
        for i in range(0, n - batch_size + 1, batch_size):
            idx = perm[i:i + batch_size]
            batch = {"x": x_tr[idx], "y": y_tr[idx]}
            loss, _ = train_step(model, optimizer, batch)
        test_mse = eval_mse(model, x_te, y_te)
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"epoch {epoch:2d}  train_mse {float(loss):.4f}  "
                  f"test_mse {float(test_mse):.4f}")

    # Forecast the first held-out window and compare to the true continuation.
    forecast = model(x_te[:1])[0]        # (horizon,)
    truth = y_te[0]                      # (horizon,)
    final_mse = float(jnp.mean((forecast - truth) ** 2))
    print(f"held-out forecast MSE (first window): {final_mse:.4f}")
    print(f"forecast[:4] {jnp.round(forecast[:4], 3)}  "
          f"truth[:4] {jnp.round(truth[:4], 3)}")

    # Visualize the forecast against the true continuation.
    out_path = os.path.join(
        os.environ.get("OUTDIR", "results"), "timeseries_forecast.png")
    plot_forecast(x_te[0, :, 0], forecast, truth, out_path)


if __name__ == "__main__":
    main()
