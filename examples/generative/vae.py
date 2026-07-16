"""
Variational Autoencoder (VAE) on MNIST
======================================
Learn a probabilistic latent space with Flax NNX: an encoder outputs a Gaussian
q(z|x), we sample z via the reparameterization trick, and a decoder reconstructs
the image. Trained by maximizing the ELBO (reconstruction - KL). Explicit NNX RNG
streams (params vs. noise) keep parameter init and latent sampling independent.

Run: python generative/vae.py
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

from shared.models import ConvEncoder, ConvDecoder
from shared.training_utils import bce_loss, kl_divergence


# ==== ENCODER: q(z|x) = N(mu, diag(exp(logvar))) ====

class Encoder(nnx.Module):
    """Conv downsampler -> two heads producing mu and logvar of q(z|x)."""

    def __init__(self, latent_dim: int, base: int = 16, *, rngs: nnx.Rngs):
        # Shared conv body: (B, 28, 28, 1) -> (B, 7, 7, base*2)
        self.body = ConvEncoder(1, base, rngs=rngs)
        feat = base * 2 * 7 * 7  # 32 * 7 * 7 = 1568 for base=16
        self.mu = nnx.Linear(feat, latent_dim, rngs=rngs)
        self.logvar = nnx.Linear(feat, latent_dim, rngs=rngs)

    def __call__(self, x):
        h = self.body(x)
        h = h.reshape(h.shape[0], -1)  # flatten
        return self.mu(h), self.logvar(h)


# ==== VAE: encoder + reparameterization + decoder ====

class VAE(nnx.Module):
    """Full VAE. The `noise` RNG stream drives latent sampling, separate from
    the `params` stream used to initialize weights."""

    def __init__(self, latent_dim: int = 16, base: int = 16, *, rngs: nnx.Rngs):
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim, base, rngs=rngs)
        # Shared decoder: latent -> (B, 28, 28, 1) LOGITS
        self.decoder = ConvDecoder(latent_dim, base, 1, rngs=rngs)
        self.rngs = rngs

    def __call__(self, x):
        mu, logvar = self.encoder(x)
        std = jnp.exp(0.5 * logvar)
        # Reparameterization trick: z = mu + std * eps, eps ~ N(0, I)
        eps = jax.random.normal(self.rngs.noise(), mu.shape)
        z = mu + std * eps
        return self.decoder(z), mu, logvar  # logits, mu, logvar

    def sample(self, n: int, seed: int = 0):
        """Draw z ~ N(0, I) and decode to images in [0, 1]."""
        z = jax.random.normal(jax.random.key(seed), (n, self.latent_dim))
        return nnx.sigmoid(self.decoder(z))


# ==== LOSS: negative ELBO = reconstruction + KL ====

def vae_loss(model: VAE, x):
    logits, mu, logvar = model(x)
    recon = bce_loss(logits, x)          # -E_q[log p(x|z)] (sigmoid BCE, summed)
    kl = kl_divergence(mu, logvar)        # D_KL(q(z|x) || N(0, I))
    return recon + kl, (recon, kl)


# ==== TRAIN STEP ====

@nnx.jit
def train_step(model: VAE, optimizer: nnx.Optimizer, batch):
    def loss_fn(model):
        return vae_loss(model, batch)

    (loss, (recon, kl)), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    optimizer.update(model, grads)
    return loss, (recon, kl)


# ==== DATA ====

def make_dataset(synthetic: bool = True, n: int = 512, seed: int = 0):
    """Return images of shape (N, 28, 28, 1) with values in [0, 1].

    Synthetic default draws smooth Gaussian blobs (offline, learnable structure).
    Set SYNTHETIC=0 to load real MNIST via tfds.
    """
    if synthetic:
        key = jax.random.key(seed)
        cy = jax.random.uniform(key, (n,), minval=8.0, maxval=20.0)
        cx = jax.random.uniform(jax.random.fold_in(key, 1), (n,), minval=8.0, maxval=20.0)
        ys, xs = jnp.mgrid[0:28, 0:28]
        ys = ys.astype(jnp.float32)
        xs = xs.astype(jnp.float32)
        d2 = (ys[None] - cy[:, None, None]) ** 2 + (xs[None] - cx[:, None, None]) ** 2
        imgs = jnp.exp(-d2 / (2.0 * 4.0 ** 2))  # (N, 28, 28) in (0, 1]
        return imgs[..., None]

    import tensorflow_datasets as tfds
    ds = tfds.load("mnist", split="train", as_supervised=True)
    imgs = jnp.stack([jnp.asarray(img) for img, _ in tfds.as_numpy(ds.take(n))])
    imgs = imgs.astype(jnp.float32) / 255.0
    if imgs.ndim == 3:
        imgs = imgs[..., None]
    return imgs


# ==== MAIN ====

def main():
    epochs = int(os.environ.get("EPOCHS", 3))
    batch = int(os.environ.get("BATCH", 64))
    synthetic = os.environ.get("SYNTHETIC", "1") != "0"

    # Two independent RNG streams: params (init) and noise (latent sampling).
    rngs = nnx.Rngs(params=0, noise=1)
    model = VAE(latent_dim=16, rngs=rngs)
    optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)

    data = make_dataset(synthetic=synthetic)
    n = data.shape[0]

    for epoch in range(epochs):
        perm = jax.random.permutation(jax.random.key(epoch), n)
        data = data[perm]
        total, steps = 0.0, 0
        for i in range(0, n, batch):
            xb = data[i:i + batch]
            loss, (recon, kl) = train_step(model, optimizer, xb)
            total += float(loss)
            steps += 1
        print(f"epoch {epoch + 1}/{epochs}  ELBO loss {total / steps:.2f}  "
              f"(recon {float(recon):.2f}  kl {float(kl):.2f})")

    samples = model.sample(64, seed=0)
    print(f"generated samples: {samples.shape}")

    # Save a sample grid artifact (picked up by the Kaggle runner from results/).
    from shared.training_utils import save_image_grid
    out = os.path.join(os.environ.get("OUTDIR", "results"), "vae_samples.png")
    save_image_grid(samples, out, nrow=8, title="VAE samples")
    print(f"saved sample grid -> {out}")

    # Reconstruction grid: 8 originals (top row) over their reconstructions (bottom).
    originals = data[:8]
    recon_logits, _, _ = model(originals)
    recon = nnx.sigmoid(recon_logits)
    pair = jnp.concatenate([originals, recon], axis=0)
    recon_out = os.path.join(os.environ.get("OUTDIR", "results"), "vae_reconstructions.png")
    save_image_grid(pair, recon_out, nrow=8,
                    title="VAE: originals (top) / reconstructions (bottom)")
    print(f"saved reconstruction grid -> {recon_out}")


if __name__ == "__main__":
    main()
