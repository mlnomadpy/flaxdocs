"""
Physics-Informed Neural Network: Damped Harmonic Oscillator
===========================================================
Solve the ODE  u'' + 2*zeta*omega*u' + omega^2*u = 0,  u(0)=1, u'(0)=0
by *differentiating the network's output with respect to its input* with
jax.grad and baking the residual straight into the loss. No dataset needed.

Run: python scientific/pinn_oscillator.py
"""

import os

import jax
import jax.numpy as jnp
from flax import nnx
import optax


# ==== PHYSICS: the ODE and its analytic solution ====
# u'' + 2*zeta*omega*u' + omega^2*u = 0 (underdamped, zeta < 1).
OMEGA = 3.0    # natural angular frequency
ZETA = 0.2     # damping ratio (0 < zeta < 1 => oscillatory decay)
T = 4.0        # solve on t in [0, T]  (~2 damped periods)
W_IC = 10.0    # weight on the initial-condition term (soft constraint)


def analytic_solution(t, omega=OMEGA, zeta=ZETA):
    """Closed-form underdamped response with u(0)=1, u'(0)=0 (for reference)."""
    omega_d = omega * jnp.sqrt(1.0 - zeta ** 2)   # damped frequency
    a = zeta * omega
    return jnp.exp(-a * t) * (jnp.cos(omega_d * t) + (a / omega_d) * jnp.sin(omega_d * t))


# ==== MODEL: a tiny t -> u(t) MLP with tanh activations ====
# tanh is smooth, so its 1st and 2nd derivatives are well-behaved -- essential
# because the physics loss differentiates the network twice.
class PINN(nnx.Module):
    """Scalar-in, scalar-out MLP approximating the solution u(t)."""

    def __init__(self, hidden: int = 32, n_layers: int = 4, *, rngs: nnx.Rngs):
        layers = [nnx.Linear(1, hidden, rngs=rngs)]
        for _ in range(n_layers - 1):
            layers.append(nnx.Linear(hidden, hidden, rngs=rngs))
        # Plain python lists crash on Flax 0.12 -- wrap submodules in nnx.List.
        self.layers = nnx.List(layers)
        self.out = nnx.Linear(hidden, 1, rngs=rngs)

    def __call__(self, t):
        """t: (N, 1) collocation points -> (N, 1) predicted displacement."""
        x = t
        for layer in self.layers:
            x = nnx.tanh(layer(x))
        return self.out(x)


# ==== LOSS: the ODE residual built from autodiff of the model ====
def physics_loss(model: PINN, t_coll):
    """Mean-squared ODE residual + initial-condition penalty.

    The key move: define a scalar helper u(t), then take jax.grad twice to get
    u'(t) and u''(t) *with respect to the input t*. vmap evaluates them at every
    collocation point. This is differentiation of the MODEL, not a fit to data.
    """
    def u_fn(t):                       # t: scalar -> scalar u(t)
        return model(t.reshape(1, 1))[0, 0]

    u_t_fn = jax.grad(u_fn)            # du/dt   via autodiff
    u_tt_fn = jax.grad(u_t_fn)        # d2u/dt2 via autodiff-of-autodiff

    t_flat = t_coll[:, 0]
    u = jax.vmap(u_fn)(t_flat)
    u_t = jax.vmap(u_t_fn)(t_flat)
    u_tt = jax.vmap(u_tt_fn)(t_flat)

    residual = u_tt + 2.0 * ZETA * OMEGA * u_t + OMEGA ** 2 * u
    loss_res = jnp.mean(residual ** 2)

    # Initial conditions u(0)=1, u'(0)=0 as a soft penalty.
    t0 = jnp.array(0.0)
    loss_ic = (u_fn(t0) - 1.0) ** 2 + (u_t_fn(t0) - 0.0) ** 2

    total = loss_res + W_IC * loss_ic
    return total, {"residual": loss_res, "ic": loss_ic}


# ==== TRAIN STEP ====
@nnx.jit
def train_step(model: PINN, optimizer: nnx.Optimizer, batch):
    """One optimization step. `batch` is the (N, 1) array of collocation points."""
    def loss_fn(model):
        total, aux = physics_loss(model, batch)
        return total, aux

    (loss, aux), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    optimizer.update(model, grads)
    return loss, aux


# ==== DATA: collocation points (no dataset, PINNs are mesh-free) ====
def make_dataset(synthetic: bool = True, n_collocation: int = 100):
    """Return (N, 1) collocation points on [0, T].

    PINNs need no training data -- the ODE *is* the supervision. `synthetic` is
    accepted for API parity with the other guides; there is nothing to download.
    """
    return jnp.linspace(0.0, T, n_collocation).reshape(-1, 1)


# ==== VISUALIZATION: does the network actually solve the ODE? ====
def plot_solution(model: PINN, t_coll, path: str):
    """Overlay the PINN prediction on the exact analytic solution.

    The two curves nearly coinciding IS the proof: the network was never shown
    the analytic answer, yet the physics-constrained loss recovered it. matplotlib
    is imported lazily (Agg backend) so importing this module stays cheap.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Dense grid for smooth curves; collocation points for the markers.
    t_dense = jnp.linspace(0.0, T, 400).reshape(-1, 1)
    u_pred = model(t_dense)[:, 0]
    u_ref = analytic_solution(t_dense[:, 0])
    t_dense = t_dense[:, 0]
    t_marks = t_coll[:, 0]
    u_marks = model(t_coll)[:, 0]
    max_err = float(jnp.max(jnp.abs(u_pred - u_ref)))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(t_dense, u_ref, color="#111827", linestyle="--", linewidth=2.5,
            label="exact analytic solution")
    ax.plot(t_dense, u_pred, color="#e11d48", linewidth=2.0, alpha=0.9,
            label="PINN prediction $\\hat u_\\theta(t)$")
    ax.scatter(t_marks, u_marks, s=14, color="#2563eb", zorder=3,
               label=f"collocation points (N={len(t_marks)})")
    ax.axhline(0.0, color="#9ca3af", linewidth=0.8, zorder=0)

    ax.set_xlabel("t")
    ax.set_ylabel("u(t)")
    ax.set_title(f"PINN vs. analytic damped oscillator  (max error = {max_err:.4f})")
    ax.legend(loc="upper right", framealpha=0.95)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved solution plot -> {path}")


# ==== MAIN ====
def main():
    steps = int(os.environ.get("EPOCHS", "3000"))          # optimization steps
    n_coll = int(os.environ.get("BATCH", "100"))           # collocation points
    synthetic = os.environ.get("SYNTHETIC", "1") == "1"

    rngs = nnx.Rngs(0)
    model = PINN(hidden=32, n_layers=4, rngs=rngs)
    optimizer = nnx.Optimizer(model, optax.adam(2e-3), wrt=nnx.Param)

    t_coll = make_dataset(synthetic=synthetic, n_collocation=n_coll)

    print(f"Solving u'' + 2*{ZETA}*{OMEGA}*u' + {OMEGA}^2*u = 0 on [0, {T}]")
    print(f"steps={steps}  collocation={n_coll}\n")

    for step in range(steps):
        loss, aux = train_step(model, optimizer, t_coll)
        if step % 500 == 0 or step == steps - 1:
            print(f"step {step:5d} | loss {float(loss):.4e} | "
                  f"residual {float(aux['residual']):.3e} | ic {float(aux['ic']):.3e}")

    # Compare against the analytic solution on a fine grid.
    t_eval = jnp.linspace(0.0, T, 200).reshape(-1, 1)
    pred = model(t_eval)[:, 0]
    ref = analytic_solution(t_eval[:, 0])
    max_err = float(jnp.max(jnp.abs(pred - ref)))
    print(f"\nmax |u_pred - u_analytic| over [0, {T}] = {max_err:.4f}")

    # Save a plot proving the network recovered the physics.
    out_path = os.path.join(os.environ.get("OUTDIR", "results"), "pinn_solution.png")
    plot_solution(model, t_coll, out_path)


if __name__ == "__main__":
    main()
