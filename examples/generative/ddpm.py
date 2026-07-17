"""
Flax NNX: DDPM Diffusion Model on MNIST
=======================================
A minimal denoising diffusion probabilistic model (DDPM): a small
time-conditioned U-Net learns to predict the noise added at each forward
step, then iteratively denoises pure Gaussian noise into images.

Run: python generative/ddpm.py
"""

import os
import jax
import jax.numpy as jnp
from flax import nnx

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.training_utils import create_optimizer, compute_mse_loss


# Default number of diffusion timesteps (kept small so the whole thing is
# CPU-friendly). Override via the T env var in main().
DEFAULT_T = 200


# ==== NOISE SCHEDULE ====

class DiffusionSchedule:
    """Linear-beta noise schedule + closed-form forward process q(x_t | x_0).

    This is a plain Python container of precomputed constants (NOT an
    nnx.Module), so it never shows up in the model's trainable state.
    """

    def __init__(self, T: int = DEFAULT_T, beta_start: float = 1e-4,
                 beta_end: float = 0.02):
        self.T = T
        betas = jnp.linspace(beta_start, beta_end, T)
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alphas_cumprod = jnp.cumprod(self.alphas)          # \bar{alpha}_t
        self.sqrt_acp = jnp.sqrt(self.alphas_cumprod)           # sqrt(\bar{alpha}_t)
        self.sqrt_one_minus_acp = jnp.sqrt(1.0 - self.alphas_cumprod)

    def q_sample(self, x0, t, eps):
        """Sample x_t ~ q(x_t | x_0) = sqrt(acp)*x_0 + sqrt(1-acp)*eps."""
        a = self.sqrt_acp[t][:, None, None, None]
        b = self.sqrt_one_minus_acp[t][:, None, None, None]
        return a * x0 + b * eps


# ==== MODEL ====

class ConvBlock(nnx.Module):
    """Conv -> GroupNorm -> (+ time embedding) -> SiLU.

    The timestep embedding is projected to the channel dimension and added to
    every spatial location, which is how the block becomes time-conditioned.
    The forward pass is wrapped in ``nnx.remat`` to trade compute for memory
    (gradient checkpointing) on the denoiser blocks.
    """

    def __init__(self, in_ch: int, out_ch: int, time_dim: int, *,
                 stride: int = 1, num_groups: int = 8, rngs: nnx.Rngs):
        self.conv = nnx.Conv(in_ch, out_ch, kernel_size=(3, 3),
                             strides=(stride, stride), padding='SAME', rngs=rngs)
        self.norm = nnx.GroupNorm(out_ch, num_groups=num_groups, rngs=rngs)
        self.time_proj = nnx.Linear(time_dim, out_ch, rngs=rngs)

    @nnx.remat
    def __call__(self, x, t_emb):
        h = self.conv(x)
        h = self.norm(h)
        h = h + self.time_proj(t_emb)[:, None, None, :]   # broadcast over H, W
        return nnx.silu(h)


class DDPMUNet(nnx.Module):
    """Compact time-conditioned U-Net that predicts the noise eps.

    One downsample, one upsample, and a single skip connection. Input and
    output are both (B, 28, 28, 1); the network predicts the noise that was
    added to produce ``x_t``.
    """

    def __init__(self, T: int = DEFAULT_T, base: int = 16, time_dim: int = 64,
                 *, rngs: nnx.Rngs):
        # Learned timestep embedding: table lookup + small MLP.
        self.time_embed = nnx.Embed(T, time_dim, rngs=rngs)
        self.time_mlp1 = nnx.Linear(time_dim, time_dim, rngs=rngs)
        self.time_mlp2 = nnx.Linear(time_dim, time_dim, rngs=rngs)

        self.in_conv = nnx.Conv(1, base, kernel_size=(3, 3), padding='SAME',
                                rngs=rngs)                     # (B,28,28,base)
        self.down = ConvBlock(base, base * 2, time_dim, stride=2, rngs=rngs)   # -> 14x14
        self.mid = ConvBlock(base * 2, base * 2, time_dim, stride=1, rngs=rngs)
        self.up = nnx.ConvTranspose(base * 2, base, kernel_size=(3, 3),
                                    strides=(2, 2), padding='SAME', rngs=rngs)  # -> 28x28
        # out_block sees the upsampled features concatenated with the skip.
        self.out_block = ConvBlock(base * 2, base, time_dim, stride=1, rngs=rngs)
        self.out_conv = nnx.Conv(base, 1, kernel_size=(3, 3), padding='SAME',
                                 rngs=rngs)                    # (B,28,28,1)

    def __call__(self, x, t):
        # Timestep embedding shared by every block.
        te = self.time_embed(t)                # (B, time_dim)
        te = nnx.silu(self.time_mlp1(te))
        te = self.time_mlp2(te)

        h0 = nnx.silu(self.in_conv(x))         # (B,28,28,base)  -- skip source
        h1 = self.down(h0, te)                 # (B,14,14,base*2)
        h2 = self.mid(h1, te)                  # (B,14,14,base*2)
        u = nnx.silu(self.up(h2))              # (B,28,28,base)
        cat = jnp.concatenate([u, h0], axis=-1)  # (B,28,28,base*2)  skip connection
        h = self.out_block(cat, te)            # (B,28,28,base)
        return self.out_conv(h)                # (B,28,28,1) predicted noise


# ==== FORWARD DIFFUSION / BATCH SAMPLING ====

def sample_batch(schedule: DiffusionSchedule, x0, rng):
    """Draw a random timestep t and noise eps, then form the noisy input x_t.

    Returns a batch dict consumed directly by ``train_step`` — the schedule is
    only ever touched here, keeping the jitted train step schedule-free.
    """
    key_t, key_e = jax.random.split(rng)
    t = jax.random.randint(key_t, (x0.shape[0],), 0, schedule.T)
    eps = jax.random.normal(key_e, x0.shape)
    x_t = schedule.q_sample(x0, t, eps)
    return {'x_t': x_t, 't': t, 'eps': eps}


# ==== TRAIN STEP ====

@nnx.jit
def train_step(model, optimizer, batch):
    """One optimization step of the simplified DDPM objective.

    L = E_{t, x0, eps} || eps - eps_theta(x_t, t) ||^2
    """
    def loss_fn(model):
        pred = model(batch['x_t'], batch['t'])
        loss = compute_mse_loss(pred, batch['eps'])
        return loss, pred

    (loss, pred), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    optimizer.update(model, grads)
    return loss, pred


# ==== SAMPLING (REVERSE PROCESS) ====

def generate(model, schedule: DiffusionSchedule, num_samples: int, rng):
    """Ancestral DDPM sampling: start from noise, denoise T steps.

    x_{t-1} = 1/sqrt(alpha_t) * (x_t - beta_t/sqrt(1-acp_t) * eps_theta) + sigma_t z

    The reverse loop is expressed with ``jax.lax.fori_loop`` so the whole
    trajectory is a single compiled program rather than a Python loop. We
    ``nnx.split`` the model and thread its (immutable, inference-only) state
    through the loop carry, then ``nnx.merge`` inside the body — the idiomatic
    way to call an NNX module from a raw ``lax`` loop without trace-level errors.
    """
    graphdef, state = nnx.split(model)
    T = schedule.T
    x = jax.random.normal(rng, (num_samples, 28, 28, 1))

    def body(i, carry):
        x, state, key = carry
        m = nnx.merge(graphdef, state)                 # rebuild at loop trace level
        t = T - 1 - i                                  # descend T-1 .. 0
        key, subkey = jax.random.split(key)
        t_batch = jnp.full((num_samples,), t, dtype=jnp.int32)

        eps_theta = m(x, t_batch)
        beta_t = schedule.betas[t]
        alpha_t = schedule.alphas[t]
        acp_t = schedule.alphas_cumprod[t]

        coef = beta_t / jnp.sqrt(1.0 - acp_t)
        mean = (x - coef * eps_theta) / jnp.sqrt(alpha_t)

        noise = jax.random.normal(subkey, x.shape)
        add_noise = (t > 0).astype(x.dtype)            # no noise on the last step
        x = mean + add_noise * jnp.sqrt(beta_t) * noise
        return (x, state, key)

    x, _, _ = jax.lax.fori_loop(0, T, body, (x, state, rng))
    return x


# ==== DATA ====

def make_dataset(n: int = 256, synthetic: bool = True, seed: int = 0):
    """Return (n, 28, 28, 1) images scaled to [-1, 1].

    Synthetic default: random Gaussian blobs (a simple but genuine image
    distribution the model can learn to denoise). Real MNIST is loaded via
    tfds only when synthetic=False.
    """
    if synthetic:
        key = jax.random.PRNGKey(seed)
        k1, k2 = jax.random.split(key)
        grid = jnp.linspace(-1.0, 1.0, 28)
        gx, gy = jnp.meshgrid(grid, grid)                 # (28,28)
        centers = jax.random.uniform(k1, (n, 2), minval=-0.5, maxval=0.5)
        widths = jax.random.uniform(k2, (n, 1), minval=0.15, maxval=0.35)
        cx = centers[:, 0].reshape(n, 1, 1)
        cy = centers[:, 1].reshape(n, 1, 1)
        w = widths.reshape(n, 1, 1)
        dist2 = (gx[None] - cx) ** 2 + (gy[None] - cy) ** 2   # (n,28,28)
        img = jnp.exp(-dist2 / (2.0 * w ** 2))                # (n,28,28) in [0,1]
        img = img * 2.0 - 1.0                                 # -> [-1,1]
        return img[..., None]

    # Real MNIST (opt-in): scale [0,255] -> [-1,1].
    import tensorflow_datasets as tfds
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    ds = tfds.load('mnist', split=f'train[:{n}]', as_supervised=True)
    imgs = jnp.stack([jnp.asarray(img) for img, _ in tfds.as_numpy(ds)])
    imgs = imgs.astype(jnp.float32) / 255.0 * 2.0 - 1.0
    if imgs.ndim == 3:
        imgs = imgs[..., None]
    return imgs


# ==== MAIN ====

def main():
    # Run-scale knobs (small CPU-friendly defaults; default to synthetic data).
    epochs = int(os.environ.get('EPOCHS', 3))
    batch = int(os.environ.get('BATCH', 32))
    T = int(os.environ.get('T', DEFAULT_T))
    synthetic = os.environ.get('SYNTHETIC', '1') != '0'
    n_data = int(os.environ.get('N_DATA', 256))

    print("=" * 60)
    print("DDPM diffusion model on MNIST (Flax NNX)")
    print("=" * 60)
    print(f"  epochs={epochs}  batch={batch}  T={T}  synthetic={synthetic}")

    schedule = DiffusionSchedule(T=T)
    x0 = make_dataset(n=n_data, synthetic=synthetic)
    print(f"  dataset: {x0.shape}  range=[{float(x0.min()):.2f}, {float(x0.max()):.2f}]")

    rngs = nnx.Rngs(0)
    model = DDPMUNet(T=T, rngs=rngs)
    optimizer = create_optimizer(model, 2e-3, optimizer_name='adam')

    n_params = sum(p.size for p in jax.tree.leaves(nnx.state(model, nnx.Param)))
    print(f"  parameters: {n_params:,}\n")

    key = jax.random.PRNGKey(1)
    steps_per_epoch = max(1, x0.shape[0] // batch)
    for epoch in range(1, epochs + 1):
        key, perm_key = jax.random.split(key)
        perm = jax.random.permutation(perm_key, x0.shape[0])
        running = 0.0
        for s in range(steps_per_epoch):
            idx = perm[s * batch:(s + 1) * batch]
            key, bkey = jax.random.split(key)
            b = sample_batch(schedule, x0[idx], bkey)
            loss, _ = train_step(model, optimizer, b)
            running += float(loss)
        print(f"  epoch {epoch:2d}/{epochs} | noise-MSE: {running / steps_per_epoch:.4f}")

    # Generate a handful of samples from pure noise.
    key, gkey = jax.random.split(key)
    n_samples = int(os.environ.get("N_SAMPLES", "16"))
    samples = generate(model, schedule, n_samples, gkey)
    print(f"\n  generated samples: {samples.shape} "
          f"range=[{float(samples.min()):.2f}, {float(samples.max()):.2f}]")

    # Save a sample grid artifact (picked up by the Kaggle runner from results/).
    from shared.training_utils import save_image_grid
    out = os.path.join(os.environ.get("OUTDIR", "results"), "diffusion_samples.png")
    save_image_grid(samples, out, nrow=8, title="Diffusion (DDPM) samples")
    print(f"saved sample grid -> {out}")

    # Forward diffusion: one clean digit progressively corrupted to pure noise.
    # This process is deterministic given the schedule, so it always renders clearly.
    steps_show = [0, schedule.T // 4, schedule.T // 2, 3 * schedule.T // 4, schedule.T - 1]
    clean = x0[:1]
    fwd = []
    for t in steps_show:
        eps = jax.random.normal(jax.random.fold_in(gkey, t + 1), clean.shape)
        fwd.append(schedule.q_sample(clean, jnp.array([t]), eps)[0])
    fwd_out = os.path.join(os.environ.get("OUTDIR", "results"), "diffusion_forward.png")
    save_image_grid(jnp.stack(fwd), fwd_out, nrow=5,
                    title=f"Forward diffusion: t=0 (clean) -> t={schedule.T - 1} (noise)")
    print(f"saved forward-process grid -> {fwd_out}")
    print("=" * 60)


if __name__ == "__main__":
    main()
