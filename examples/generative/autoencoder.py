"""
Convolutional Autoencoder (+ Denoising) on MNIST with Flax NNX
==============================================================
Compress 28x28 images through a small latent bottleneck and reconstruct them
with learned upsampling (nnx.ConvTranspose). Flip one flag to turn the plain
autoencoder into a denoising autoencoder that cleans up noisy inputs.

Run: python generative/autoencoder.py
"""

import os
import jax
import jax.numpy as jnp
from flax import nnx
import optax

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.models import ConvEncoder, ConvDecoder
from shared.training_utils import bce_loss


# ==== MODEL ====

class ConvAutoencoder(nnx.Module):
    """Encoder -> Linear bottleneck -> Decoder.

    Forward pass:
      x            (B, 28, 28, in_ch)
      -> ConvEncoder                 -> (B, 7, 7, base*2)
      -> flatten + Linear            -> (B, latent_dim)   [the bottleneck]
      -> ConvDecoder (ConvTranspose) -> (B, 28, 28, in_ch) LOGITS
    """

    def __init__(self, latent_dim: int = 32, base: int = 16,
                 in_channels: int = 1, *, rngs: nnx.Rngs):
        self.enc_hw = 7                      # 28 -> 14 -> 7 after two stride-2 convs
        self.enc_channels = base * 2         # ConvEncoder doubles `base`
        self.encoder = ConvEncoder(in_channels, base, rngs=rngs)
        # Bottleneck: squeeze the (7*7*base*2) feature map into `latent_dim`.
        self.bottleneck = nnx.Linear(
            self.enc_channels * self.enc_hw * self.enc_hw, latent_dim, rngs=rngs
        )
        # Decoder mirrors the encoder and returns LOGITS (no final activation).
        self.decoder = ConvDecoder(latent_dim, base, in_channels,
                                   start_hw=self.enc_hw, rngs=rngs)

    def encode(self, x):
        h = self.encoder(x)                  # (B, 7, 7, base*2)
        h = h.reshape(h.shape[0], -1)        # (B, 7*7*base*2)
        return self.bottleneck(h)            # (B, latent_dim)

    def __call__(self, x, train: bool = False):
        z = self.encode(x)                   # bottleneck code
        return self.decoder(z)               # (B, 28, 28, in_ch) logits


# ==== NOISE (for the denoising variant) ====

def add_gaussian_noise(x, key, std: float):
    """Corrupt inputs with Gaussian noise, then clip back to [0, 1]."""
    noisy = x + std * jax.random.normal(key, x.shape)
    return jnp.clip(noisy, 0.0, 1.0)


# ==== TRAIN STEP ====

@nnx.jit
def train_step(model, optimizer, batch):
    """One gradient step. batch['x'] is the (possibly noisy) input,
    batch['target'] is the CLEAN image we reconstruct against."""
    def loss_fn(model):
        logits = model(batch['x'], train=True)          # (B, 28, 28, in_ch)
        loss = bce_loss(logits, batch['target'])        # sigmoid-BCE per pixel
        return loss, logits

    (loss, logits), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    optimizer.update(model, grads)
    return loss, logits


# ==== DATA ====

def make_dataset(synthetic: bool = True, n: int = 512, seed: int = 0):
    """Return CLEAN images of shape (n, 28, 28, 1) in [0, 1].

    synthetic=True  -> binarized random blobs (offline, no downloads).
    synthetic=False -> real MNIST via tfds, scaled to [0, 1].
    """
    if synthetic:
        key = jax.random.key(seed)
        # Smooth-ish random fields, then binarize: values are exactly {0, 1}.
        raw = jax.random.uniform(key, (n, 28, 28, 1))
        images = (raw > 0.5).astype(jnp.float32)
        return images
    else:
        import tensorflow_datasets as tfds
        import tensorflow as tf
        tf.config.set_visible_devices([], 'GPU')
        ds = tfds.load('mnist', split=f'train[:{n}]', as_supervised=True)
        imgs = [tf.cast(img, tf.float32).numpy() / 255.0 for img, _ in tfds.as_numpy(ds)]
        return jnp.asarray(imgs).reshape(-1, 28, 28, 1)


def make_batch(images, idx, *, denoising: bool, noise_std: float, key):
    """Slice a batch and build {'x': input, 'target': clean image}."""
    clean = images[idx]
    if denoising:
        x = add_gaussian_noise(clean, key, noise_std)
    else:
        x = clean
    return {'x': x, 'target': clean}


# ==== MAIN ====

def main():
    # Run-scale from env with small CPU-friendly defaults; SYNTHETIC by default.
    epochs = int(os.environ.get('EPOCHS', 4))
    batch = int(os.environ.get('BATCH', 64))
    synthetic = os.environ.get('SYNTHETIC', '1') != '0'
    denoising = os.environ.get('DENOISE', '0') != '0'
    noise_std = float(os.environ.get('NOISE_STD', 0.5))
    latent_dim = int(os.environ.get('LATENT', 32))

    mode = 'DENOISING autoencoder' if denoising else 'autoencoder'
    print(f"Convolutional {mode} on {'synthetic' if synthetic else 'MNIST'} data")
    print(f"  epochs={epochs} batch={batch} latent_dim={latent_dim} "
          f"noise_std={noise_std if denoising else 0}")

    images = make_dataset(synthetic=synthetic)
    n = images.shape[0]
    print(f"  dataset: {images.shape}  (clean targets in [0, 1])")

    model = ConvAutoencoder(latent_dim=latent_dim, rngs=nnx.Rngs(0))
    optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)

    key = jax.random.key(1)
    step = 0
    for epoch in range(1, epochs + 1):
        key, perm_key = jax.random.split(key)
        order = jax.random.permutation(perm_key, n)
        epoch_loss, nb = 0.0, 0
        for start in range(0, n - batch + 1, batch):
            idx = order[start:start + batch]
            key, noise_key = jax.random.split(key)
            b = make_batch(images, idx, denoising=denoising,
                           noise_std=noise_std, key=noise_key)
            loss, _ = train_step(model, optimizer, b)
            epoch_loss += float(loss)
            nb += 1
            step += 1
        print(f"  epoch {epoch:2d}/{epochs} | steps {step:4d} | "
              f"BCE {epoch_loss / max(nb, 1):8.2f}")

    # Reconstruction grid: 8 originals (top row) over their reconstructions (bottom).
    from shared.training_utils import save_image_grid
    originals = images[:8]
    recon = nnx.sigmoid(model(originals))
    pair = jnp.concatenate([originals, recon], axis=0)
    recon_out = os.path.join(os.environ.get("OUTDIR", "results"), "ae_reconstructions.png")
    save_image_grid(pair, recon_out, nrow=8,
                    title="Autoencoder: originals (top) / reconstructions (bottom)")
    print(f"saved reconstruction grid -> {recon_out}")

    print("Done. The decoder's nnx.ConvTranspose layers upsample 7->14->28.")


if __name__ == "__main__":
    main()
