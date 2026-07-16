"""
DCGAN on MNIST with Flax NNX
============================
A Deep Convolutional GAN: a transpose-conv Generator and a spectral-normalized
Discriminator play the adversarial minimax game, trained with two optimizers
and the non-saturating generator loss.

Run: python generative/dcgan.py
"""

import os
import jax
import jax.numpy as jnp
from flax import nnx
import optax

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.training_utils import bce_loss


# ==== MODELS ====

class Generator(nnx.Module):
    """Maps a latent vector z -> a 28x28x1 image in [-1, 1].

    Linear(z_dim -> 32*7*7) -> reshape (7,7,32)
      -> ConvTranspose 7->14 (32->16, relu)
      -> ConvTranspose 14->28 (16->1) -> tanh
    """

    def __init__(self, z_dim: int = 64, base: int = 16, *, rngs: nnx.Rngs):
        self.z_dim = z_dim
        self.fc = nnx.Linear(z_dim, base * 2 * 7 * 7, rngs=rngs)
        self.deconv1 = nnx.ConvTranspose(base * 2, base, kernel_size=(3, 3),
                                         strides=(2, 2), padding='SAME', rngs=rngs)  # 7 -> 14
        self.deconv2 = nnx.ConvTranspose(base, 1, kernel_size=(3, 3),
                                         strides=(2, 2), padding='SAME', rngs=rngs)  # 14 -> 28
        self.base = base

    def __call__(self, z):
        h = self.fc(z)
        h = h.reshape(z.shape[0], 7, 7, self.base * 2)
        h = nnx.relu(self.deconv1(h))
        h = self.deconv2(h)
        return nnx.tanh(h)   # outputs in [-1, 1]


class Discriminator(nnx.Module):
    """Classifies 28x28x1 images as real (high logit) or fake (low logit).

    Every conv/linear kernel is wrapped in ``nnx.SpectralNorm`` to bound the
    Lipschitz constant, which stabilizes the adversarial game.
    """

    def __init__(self, base: int = 16, *, rngs: nnx.Rngs):
        # Spectral-normalized conv stack (stride 2, leaky_relu).
        self.conv1 = nnx.SpectralNorm(
            nnx.Conv(1, base, kernel_size=(3, 3), strides=(2, 2),
                     padding='SAME', rngs=rngs), rngs=rngs)          # 28 -> 14
        self.conv2 = nnx.SpectralNorm(
            nnx.Conv(base, base * 2, kernel_size=(3, 3), strides=(2, 2),
                     padding='SAME', rngs=rngs), rngs=rngs)          # 14 -> 7
        # Spectral-normalized linear head -> single logit.
        self.fc = nnx.SpectralNorm(
            nnx.Linear(base * 2 * 7 * 7, 1, rngs=rngs), rngs=rngs)

    def __call__(self, x, train: bool = False):
        # update_stats only when we are actually training the discriminator.
        x = nnx.leaky_relu(self.conv1(x, update_stats=train), negative_slope=0.2)
        x = nnx.leaky_relu(self.conv2(x, update_stats=train), negative_slope=0.2)
        x = x.reshape(x.shape[0], -1)
        return self.fc(x, update_stats=train)   # (B, 1) logit


# ==== DATA ====

def sample_z(key, batch: int, z_dim: int):
    """Draw a batch of standard-normal latent vectors."""
    return jax.random.normal(key, (batch, z_dim))


def make_dataset(synthetic: bool = True, n: int = 512):
    """Return real images as (N, 28, 28, 1) float array scaled to [-1, 1].

    Synthetic mode fabricates smooth blob patterns so the script runs offline;
    set SYNTHETIC=0 to stream real MNIST via tfds.
    """
    if synthetic:
        key = jax.random.PRNGKey(0)
        # A few smooth Gaussian blobs -> non-trivial but cheap structure.
        ys, xs = jnp.meshgrid(jnp.linspace(-1, 1, 28), jnp.linspace(-1, 1, 28),
                              indexing='ij')
        centers = jax.random.uniform(key, (n, 2), minval=-0.5, maxval=0.5)
        imgs = jnp.exp(-((xs[None] - centers[:, 0, None, None]) ** 2
                         + (ys[None] - centers[:, 1, None, None]) ** 2) / 0.1)
        imgs = imgs * 2.0 - 1.0                      # [0,1] -> [-1,1]
        return imgs[..., None]                        # (N, 28, 28, 1)

    import tensorflow_datasets as tfds
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    ds = tfds.load('mnist', split='train', batch_size=-1)
    imgs = jnp.asarray(tfds.as_numpy(ds)['image'], dtype=jnp.float32) / 255.0
    return imgs * 2.0 - 1.0                            # (N, 28, 28, 1) in [-1,1]


# ==== TRAIN STEPS ====

@nnx.jit
def d_step(gen, disc, opt_d, real, z):
    """Update the discriminator: push real logits up, fake logits down."""
    fake = gen(z)   # generator is fixed here (not in the grad set)

    def loss_fn(disc):
        real_logits = disc(real, train=True)
        fake_logits = disc(fake, train=True)
        loss = (bce_loss(real_logits, jnp.ones_like(real_logits))
                + bce_loss(fake_logits, jnp.zeros_like(fake_logits)))
        return loss, (real_logits, fake_logits)

    (loss, aux), grads = nnx.value_and_grad(loss_fn, has_aux=True)(disc)
    opt_d.update(disc, grads)
    return loss, aux


@nnx.jit
def g_step(gen, disc, opt_g, z):
    """Update the generator with the non-saturating loss: make fakes look real.

    ``disc`` is passed as a (non-differentiated) argument so its spectral-norm
    layers stay at the transform's trace level; ``value_and_grad`` differentiates
    argnums=0 only, so no discriminator gradients are produced.
    """

    def loss_fn(gen, disc):
        fake = gen(z)
        # Discriminator is frozen here; don't update its spectral-norm stats.
        fake_logits = disc(fake, train=False)
        loss = bce_loss(fake_logits, jnp.ones_like(fake_logits))
        return loss, fake_logits

    (loss, aux), grads = nnx.value_and_grad(loss_fn, has_aux=True)(gen, disc)
    opt_g.update(gen, grads)
    return loss, aux


# ==== TRAIN LOOP ====

def main():
    epochs = int(os.environ.get("EPOCHS", "2"))
    batch = int(os.environ.get("BATCH", "64"))
    synthetic = os.environ.get("SYNTHETIC", "1") != "0"
    z_dim = 64

    rngs = nnx.Rngs(0)
    gen = Generator(z_dim=z_dim, rngs=rngs)
    disc = Discriminator(rngs=rngs)

    opt_g = nnx.Optimizer(gen, optax.adam(2e-4, b1=0.5), wrt=nnx.Param)
    opt_d = nnx.Optimizer(disc, optax.adam(2e-4, b1=0.5), wrt=nnx.Param)

    data = make_dataset(synthetic=synthetic)
    n = data.shape[0]
    key = jax.random.PRNGKey(42)

    for epoch in range(epochs):
        perm = jax.random.permutation(jax.random.fold_in(key, epoch), n)
        for i in range(0, n - batch + 1, batch):
            real = data[perm[i:i + batch]]
            key, kd, kg = jax.random.split(key, 3)
            d_loss, _ = d_step(gen, disc, opt_d, real, sample_z(kd, batch, z_dim))
            g_loss, _ = g_step(gen, disc, opt_g, sample_z(kg, batch, z_dim))
        print(f"epoch {epoch}: d_loss={float(d_loss):.4f}  g_loss={float(g_loss):.4f}")


if __name__ == "__main__":
    main()
