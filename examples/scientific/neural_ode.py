"""
Neural ODE: Fitting a 2D Spiral with a Learned Vector Field
===========================================================
Learn continuous-depth dynamics dy/dt = f_theta(y) that reproduce a spiral
trajectory. A fixed-step RK4 solver (hand-written with jax.lax.scan, no diffrax)
integrates the network, and we backprop straight through the solver.

Run: python scientific/neural_ode.py
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
from shared.training_utils import compute_mse_loss


# ==== GROUND TRUTH: a known linear ODE that traces a decaying spiral ====
# dy/dt = A y with eigenvalues -0.1 +/- i  =>  rotation (period ~2*pi) that
# spirals inward. We integrate THIS with RK4 to manufacture the observations;
# the network never sees A, only the trajectory it produces.
A_TRUE = jnp.array([[-0.1, -1.0],
                    [1.0, -0.1]])
Y0 = jnp.array([2.0, 0.0])   # starting point of the spiral
T = 8.0                       # integrate over t in [0, T] (~1.25 revolutions)


def true_dynamics(t, y):
    """Reference vector field dy/dt = A y (autonomous: t is ignored)."""
    return y @ A_TRUE.T


# ==== SOLVER: fixed-step classic RK4, built by hand with lax.scan ====
def rk4_step(f, y, t, dt):
    """One Runge-Kutta 4 step of dy/dt = f(t, y)."""
    k1 = f(t, y)
    k2 = f(t + dt / 2, y + dt / 2 * k1)
    k3 = f(t + dt / 2, y + dt / 2 * k2)
    k4 = f(t + dt, y + dt * k3)
    return y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def odeint_rk4(f, y0, ts):
    """Integrate dy/dt = f(t, y) over the grid `ts`, returning states (T, ...).

    lax.scan turns the loop into a single differentiable op, so reverse-mode AD
    can flow gradients through every step -- "discretize-then-optimize".
    """
    dt = ts[1] - ts[0]

    def step(y, t):
        y_next = rk4_step(f, y, t, dt)
        return y_next, y_next

    # Scan advances from t0..t_{T-2}; prepend y0 so we return all T states.
    _, ys = jax.lax.scan(step, y0, ts[:-1])
    return jnp.concatenate([y0[None], ys], axis=0)


# ==== MODEL: an MLP vector field integrated by the solver ====
class NeuralODE(nnx.Module):
    """Continuous-depth model: an MLP dynamics f_theta(y) plugged into RK4."""

    def __init__(self, hidden: int = 64, *, rngs: nnx.Rngs):
        # f_theta: R^2 -> R^2. gelu is smooth, giving a well-behaved field.
        self.func = MLP(
            in_features=2,
            hidden_features=hidden,
            out_features=2,
            n_layers=3,
            rngs=rngs,
            activation="gelu",
        )
        # Zero the final layer so the initial field is ~0: the untrained solver
        # then stays near y0 instead of blowing up over many steps. This is the
        # standard "start from the identity flow" trick for Neural ODEs.
        self.func.output.kernel[...] = jnp.zeros_like(self.func.output.kernel[...])
        self.func.output.bias[...] = jnp.zeros_like(self.func.output.bias[...])

    def dynamics(self, t, y):
        """Learned vector field; autonomous, so the time input is unused."""
        return self.func(y)

    def __call__(self, y0, ts):
        """Integrate the learned field from y0 over ts -> trajectory (T, 2)."""
        return odeint_rk4(self.dynamics, y0, ts)


# ==== TRAIN STEP ====
@nnx.jit
def train_step(model: NeuralODE, optimizer: nnx.Optimizer, batch):
    """One optimization step. `batch` = (y0, ts, target_trajectory)."""
    y0, ts, target = batch

    def loss_fn(model):
        pred = model(y0, ts)                 # (T, 2)
        loss = compute_mse_loss(pred, target)
        return loss, pred

    (loss, pred), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    optimizer.update(model, grads)
    return loss, pred


# ==== DATA: synthetic spiral (no download, fully offline) ====
def make_dataset(synthetic: bool = True, n_points: int = 80):
    """Return (ts, true_trajectory) for the ground-truth spiral.

    `synthetic` is accepted for parity with the other guides -- the spiral is
    always generated locally by integrating the known linear ODE.
    """
    ts = jnp.linspace(0.0, T, n_points)
    true_traj = odeint_rk4(true_dynamics, Y0, ts)   # (n_points, 2)
    return ts, true_traj


# ==== VISUALIZATION: phase-plane overlay of true vs learned spiral ====
def save_phase_plot(true_traj, pred_traj, path):
    """Overlay the true spiral (solid) and the learned trajectory (dashed).

    matplotlib is imported lazily (with the headless Agg backend) so that just
    importing this module stays cheap and never needs a display.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import numpy as np
    true_np = np.asarray(true_traj)
    pred_np = np.asarray(pred_traj)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(true_np[:, 0], true_np[:, 1],
            color="#1f77b4", lw=2.5, label="true spiral")
    ax.plot(pred_np[:, 0], pred_np[:, 1],
            color="#d62728", lw=2.0, ls="--", label="Neural ODE (learned)")
    ax.scatter([true_np[0, 0]], [true_np[0, 1]],
               color="black", zorder=5, s=60, label="start $y_0$")

    ax.set_xlabel("$y_1$")
    ax.set_ylabel("$y_2$")
    ax.set_title("Neural ODE recovers the spiral in the phase plane")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"saved phase-plane plot -> {path}")


# ==== MAIN ====
def main():
    steps = int(os.environ.get("EPOCHS", "1000"))       # optimization steps
    n_points = int(os.environ.get("BATCH", "80"))       # observations on curve
    synthetic = os.environ.get("SYNTHETIC", "1") == "1"

    rngs = nnx.Rngs(0)
    model = NeuralODE(hidden=64, rngs=rngs)
    optimizer = nnx.Optimizer(model, optax.adam(5e-3), wrt=nnx.Param)

    ts, true_traj = make_dataset(synthetic=synthetic, n_points=n_points)
    batch = (Y0, ts, true_traj)

    print(f"Fitting dy/dt = f_theta(y) to a spiral on t in [0, {T}]")
    print(f"steps={steps}  points={n_points}\n")

    for step in range(steps):
        loss, _ = train_step(model, optimizer, batch)
        if step % 100 == 0 or step == steps - 1:
            print(f"step {step:5d} | trajectory MSE {float(loss):.4e}")

    pred = model(Y0, ts)
    final_mse = float(compute_mse_loss(pred, true_traj))
    endpoint_err = float(jnp.linalg.norm(pred[-1] - true_traj[-1]))
    print(f"\nfinal trajectory MSE = {final_mse:.4e}")
    print(f"endpoint error |y_pred(T) - y_true(T)| = {endpoint_err:.4f}")

    out_path = os.path.join(os.environ.get("OUTDIR", "results"),
                            "neural_ode_trajectory.png")
    save_phase_plot(true_traj, pred, out_path)


if __name__ == "__main__":
    main()
