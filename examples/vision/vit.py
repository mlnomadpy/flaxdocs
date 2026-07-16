"""
Vision Transformer (ViT) Classifier on MNIST with Flax NNX
==========================================================
Patchify 28x28 images into non-overlapping 7x7 patches, embed each patch,
prepend a learned CLS token + positional embeddings, run a small transformer
encoder, and classify from the CLS token. Synthetic (offline) by default.

Run: python vision/vit.py
"""

import os
import jax
import jax.numpy as jnp
from flax import nnx
import optax

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.models import TransformerBlock
from shared.training_utils import compute_cross_entropy_loss, compute_accuracy


# ==== PATCH EMBEDDING ====

class PatchEmbed(nnx.Module):
    """Split an image into non-overlapping patches and linearly embed each one.

    Implemented as a single strided convolution: a kernel of size `patch_size`
    with stride `patch_size` applies one linear map per patch (this is exactly a
    per-patch flatten + Linear). For 28x28 with patch=7 the output grid is 4x4,
    which we flatten into a sequence of 16 patch tokens.
    """

    def __init__(self, patch_size: int, in_channels: int, dim: int, *, rngs: nnx.Rngs):
        self.proj = nnx.Conv(
            in_channels, dim,
            kernel_size=(patch_size, patch_size),
            strides=(patch_size, patch_size),
            padding="VALID",
            rngs=rngs,
        )

    def __call__(self, x):
        x = self.proj(x)                       # (B, H/p, W/p, dim)
        b, h, w, d = x.shape
        return x.reshape(b, h * w, d)          # (B, num_patches, dim)


# ==== MODEL ====

class ViT(nnx.Module):
    """A compact Vision Transformer for image classification.

    Pipeline:
      x            (B, 28, 28, 1)
      -> PatchEmbed                 -> (B, 16, dim)   # 16 non-overlapping patches
      -> prepend learned CLS token  -> (B, 17, dim)
      -> add learned pos embeddings -> (B, 17, dim)
      -> depth x TransformerBlock   -> (B, 17, dim)   # global attention from layer 1
      -> LayerNorm + take CLS token -> (B, dim)
      -> Linear head                -> (B, num_classes) logits
    """

    def __init__(
        self,
        *,
        image_size: int = 28,
        patch_size: int = 7,
        in_channels: int = 1,
        num_classes: int = 10,
        dim: int = 64,
        depth: int = 4,
        num_heads: int = 4,
        mlp_dim: int = 128,
        dropout: float = 0.1,
        rngs: nnx.Rngs,
    ):
        assert image_size % patch_size == 0, "image_size must be divisible by patch_size"
        num_patches = (image_size // patch_size) ** 2  # 4 * 4 = 16

        self.patch_embed = PatchEmbed(patch_size, in_channels, dim, rngs=rngs)

        # Learned CLS token and positional embeddings, stored as trainable Params.
        # Small-scale init (0.02) matches the ViT / BERT convention.
        self.cls_token = nnx.Param(0.02 * jax.random.normal(rngs.params(), (1, 1, dim)))
        self.pos_embed = nnx.Param(
            0.02 * jax.random.normal(rngs.params(), (1, num_patches + 1, dim))
        )

        # A stack of pre-norm transformer encoder blocks. MUST be nnx.List so the
        # submodules are tracked as a pytree on Flax 0.12.
        self.blocks = nnx.List([
            TransformerBlock(
                d_model=dim,
                num_heads=num_heads,
                d_ff=mlp_dim,
                dropout=dropout,
                rngs=rngs,
                causal=False,               # bidirectional: every token sees every token
            )
            for _ in range(depth)
        ])

        self.norm = nnx.LayerNorm(dim, rngs=rngs)
        self.head = nnx.Linear(dim, num_classes, rngs=rngs)

    def __call__(self, x, train: bool = False):
        b = x.shape[0]

        tokens = self.patch_embed(x)                          # (B, num_patches, dim)

        # Prepend the CLS token to every example in the batch.
        cls = jnp.broadcast_to(self.cls_token[...], (b, 1, tokens.shape[-1]))
        tokens = jnp.concatenate([cls, tokens], axis=1)       # (B, num_patches+1, dim)

        # Add learned positional embeddings (broadcast over the batch).
        tokens = tokens + self.pos_embed[...]

        for block in self.blocks:
            tokens = block(tokens, train=train)               # (B, num_patches+1, dim)

        tokens = self.norm(tokens)
        cls_out = tokens[:, 0]                                 # (B, dim) — the CLS token
        return self.head(cls_out)                             # (B, num_classes) logits


# ==== TRAIN STEP ====

@nnx.jit
def train_step(model, optimizer, batch):
    """One gradient step of cross-entropy classification."""
    def loss_fn(model):
        logits = model(batch["x"], train=True)                # (B, num_classes)
        loss = compute_cross_entropy_loss(logits, batch["y"])
        return loss, logits

    (loss, logits), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    optimizer.update(model, grads)
    accuracy = compute_accuracy(logits, batch["y"])
    return loss, accuracy


# ==== DATA ====

def make_dataset(synthetic: bool = True, n: int = 512, num_classes: int = 10, seed: int = 0):
    """Return (images, labels) with images of shape (n, 28, 28, 1) in [0, 1].

    synthetic=True  -> offline images WITH per-class signal: each class lights up
                       a distinct 7x7 patch cell of the 4x4 grid, plus Gaussian
                       noise. This gives a learnable pattern a ViT can pick up in
                       a few dozen steps (no downloads).
    synthetic=False -> real MNIST via tfds, scaled to [0, 1].
    """
    if synthetic:
        key = jax.random.key(seed)
        k_lab, k_noise = jax.random.split(key)
        labels = jax.random.randint(k_lab, (n,), 0, num_classes)

        # One bright-patch template per class: class c -> grid cell (c // 4, c % 4).
        grid = 28 // 7  # 4 cells per side, 16 total (covers 10 classes)
        templates = jnp.zeros((num_classes, 28, 28, 1))
        for c in range(num_classes):
            r, col = c // grid, c % grid
            templates = templates.at[c, r * 7:(r + 1) * 7, col * 7:(col + 1) * 7, 0].set(1.0)

        images = templates[labels] + 0.3 * jax.random.normal(k_noise, (n, 28, 28, 1))
        images = jnp.clip(images, 0.0, 1.0)
        return images, labels

    import tensorflow_datasets as tfds
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    ds = tfds.load("mnist", split=f"train[:{n}]", as_supervised=True)
    imgs, labs = [], []
    for img, lab in tfds.as_numpy(ds):
        imgs.append(img.astype("float32") / 255.0)
        labs.append(int(lab))
    return jnp.asarray(imgs).reshape(-1, 28, 28, 1), jnp.asarray(labs)


# ==== MAIN ====

def main():
    # Run-scale from env with small CPU-friendly defaults; SYNTHETIC by default.
    epochs = int(os.environ.get("EPOCHS", 20))
    batch = int(os.environ.get("BATCH", 64))
    synthetic = os.environ.get("SYNTHETIC", "1") != "0"
    num_classes = 10

    print(f"Vision Transformer on {'synthetic' if synthetic else 'MNIST'} data")
    print(f"  epochs={epochs} batch={batch}")

    images, labels = make_dataset(synthetic=synthetic, num_classes=num_classes)
    n = images.shape[0]
    print(f"  dataset: images {images.shape}, labels {labels.shape}")

    model = ViT(num_classes=num_classes, dim=64, depth=4, num_heads=4, rngs=nnx.Rngs(0))
    optimizer = nnx.Optimizer(model, optax.adamw(1e-3), wrt=nnx.Param)

    key = jax.random.key(1)
    step = 0
    for epoch in range(1, epochs + 1):
        key, perm_key = jax.random.split(key)
        order = jax.random.permutation(perm_key, n)
        ep_loss, ep_acc, nb = 0.0, 0.0, 0
        for start in range(0, n - batch + 1, batch):
            idx = order[start:start + batch]
            b = {"x": images[idx], "y": labels[idx]}
            loss, acc = train_step(model, optimizer, b)
            ep_loss += float(loss)
            ep_acc += float(acc)
            nb += 1
            step += 1
        print(f"  epoch {epoch:2d}/{epochs} | steps {step:4d} | "
              f"loss {ep_loss / max(nb, 1):.4f} | acc {ep_acc / max(nb, 1):.3f}")

    print("Done. Every patch token attends to every other from the first layer — "
          "a global receptive field, unlike a CNN's local kernels.")


if __name__ == "__main__":
    main()
