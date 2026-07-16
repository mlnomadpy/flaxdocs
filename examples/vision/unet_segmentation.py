"""
U-Net Semantic Segmentation on Synthetic Shapes with Flax NNX
=============================================================
A full encoder-decoder U-Net that labels every pixel: two downsampling stages,
a bottleneck, and a mirrored decoder that upsamples with nnx.ConvTranspose while
concatenating encoder features through skip connections. Data is generated on the
fly (random circles/squares + binary masks) so it runs offline on CPU.

Run: python vision/unet_segmentation.py
"""

import os
import jax
import jax.numpy as jnp
from flax import nnx

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.training_utils import bce_loss


# ==== MODEL ====

class DoubleConv(nnx.Module):
    """(Conv -> GroupNorm -> ReLU) x 2 at a fixed spatial resolution.

    The two-conv block is the basic building unit of U-Net. GroupNorm (not
    BatchNorm) keeps normalization batch-independent, which plays nicely with
    `nnx.jit` and small CPU batches. Channel counts must stay divisible by
    `num_groups`.
    """

    def __init__(self, in_ch: int, out_ch: int, *, num_groups: int = 8,
                 rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(in_ch, out_ch, kernel_size=(3, 3),
                              padding='SAME', rngs=rngs)
        self.norm1 = nnx.GroupNorm(out_ch, num_groups=num_groups, rngs=rngs)
        self.conv2 = nnx.Conv(out_ch, out_ch, kernel_size=(3, 3),
                              padding='SAME', rngs=rngs)
        self.norm2 = nnx.GroupNorm(out_ch, num_groups=num_groups, rngs=rngs)

    def __call__(self, x):
        x = nnx.relu(self.norm1(self.conv1(x)))
        x = nnx.relu(self.norm2(self.conv2(x)))
        return x


class UNet(nnx.Module):
    """A 2-level U-Net for binary semantic segmentation.

    Encoder halves the resolution twice (max-pool), the bottleneck runs at the
    coarsest scale, and the decoder upsamples back with `nnx.ConvTranspose`. At
    each decoder stage the upsampled features are CONCATENATED with the matching
    encoder features (the skip connections) before another DoubleConv. Output is
    per-pixel logits of shape (B, H, W, out_channels) at the input resolution.

    Spatial trace for a 32x32 input:
      x   (B,32,32,in)
      s1 = enc1(x)                 -> (B,32,32,base)      [skip 1]
      s2 = enc2(pool(s1))          -> (B,16,16,base*2)    [skip 2]
      b  = bottleneck(pool(s2))    -> (B, 8, 8,base*4)
      up2(b) ++ s2 -> dec2         -> (B,16,16,base*2)
      up1     ++ s1 -> dec1        -> (B,32,32,base)
      out_conv                     -> (B,32,32,out)       logits
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1,
                 base: int = 16, *, rngs: nnx.Rngs):
        # --- Encoder (contracting path) ---
        self.enc1 = DoubleConv(in_channels, base, rngs=rngs)       # full res
        self.enc2 = DoubleConv(base, base * 2, rngs=rngs)          # 1/2 res

        # --- Bottleneck ---
        self.bottleneck = DoubleConv(base * 2, base * 4, rngs=rngs)  # 1/4 res

        # --- Decoder (expanding path) ---
        # ConvTranspose doubles the resolution; the following DoubleConv sees the
        # upsampled features concatenated with the encoder skip (hence 2x in_ch).
        self.up2 = nnx.ConvTranspose(base * 4, base * 2, kernel_size=(3, 3),
                                     strides=(2, 2), padding='SAME', rngs=rngs)
        self.dec2 = DoubleConv(base * 4, base * 2, rngs=rngs)      # concat -> base*4
        self.up1 = nnx.ConvTranspose(base * 2, base, kernel_size=(3, 3),
                                     strides=(2, 2), padding='SAME', rngs=rngs)
        self.dec1 = DoubleConv(base * 2, base, rngs=rngs)         # concat -> base*2

        # 1x1 conv projects to per-pixel class logits (no activation).
        self.out_conv = nnx.Conv(base, out_channels, kernel_size=(1, 1),
                                 rngs=rngs)

    def __call__(self, x, train: bool = False):
        # Encoder: keep s1, s2 as skip sources BEFORE each downsample.
        s1 = self.enc1(x)                                        # (B,H,  W,  base)
        s2 = self.enc2(nnx.max_pool(s1, (2, 2), strides=(2, 2)))  # (B,H/2,W/2,base*2)
        b = self.bottleneck(nnx.max_pool(s2, (2, 2), strides=(2, 2)))  # (B,H/4,...,base*4)

        # Decoder: upsample, concatenate the skip, then DoubleConv.
        d2 = self.up2(b)                                         # (B,H/2,W/2,base*2)
        d2 = self.dec2(jnp.concatenate([d2, s2], axis=-1))       # skip 2
        d1 = self.up1(d2)                                        # (B,H,  W,  base)
        d1 = self.dec1(jnp.concatenate([d1, s1], axis=-1))       # skip 1

        return self.out_conv(d1)                                 # (B,H,W,out) logits


# ==== METRICS ====

def iou_dice(logits, masks, threshold: float = 0.5, eps: float = 1e-6):
    """Mean IoU and Dice between thresholded predictions and binary masks.

    IoU  = |P ∩ G| / |P ∪ G|      (a.k.a. Jaccard index)
    Dice = 2|P ∩ G| / (|P| + |G|) (a.k.a. F1 over pixels)
    """
    preds = (jax.nn.sigmoid(logits) > threshold).astype(jnp.float32)
    axes = (1, 2, 3)
    inter = jnp.sum(preds * masks, axis=axes)
    p_area = jnp.sum(preds, axis=axes)
    g_area = jnp.sum(masks, axis=axes)
    union = p_area + g_area - inter
    iou = jnp.mean((inter + eps) / (union + eps))
    dice = jnp.mean((2.0 * inter + eps) / (p_area + g_area + eps))
    return iou, dice


# ==== TRAIN STEP ====

@nnx.jit
def train_step(model, optimizer, batch):
    """One optimization step of per-pixel sigmoid binary cross-entropy.

    L = mean_b  sum_pixels  BCE(sigmoid(logit_p), mask_p)
    """
    def loss_fn(model):
        logits = model(batch['image'], train=True)          # (B, H, W, 1)
        loss = bce_loss(logits, batch['mask'])              # per-pixel sigmoid BCE
        return loss, logits

    (loss, logits), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    optimizer.update(model, grads)
    return loss, logits


# ==== DATA ====

def make_dataset(n: int = 128, size: int = 32, max_shapes: int = 3,
                 seed: int = 0):
    """Generate (images, masks), each (n, size, size, 1), fully offline.

    Each image gets `max_shapes` bright circles/squares dropped on a dim,
    slightly noisy background. The mask is 1 wherever any shape covers a pixel.
    Everything is built with jax.random — no downloads.
    """
    key = jax.random.key(seed)
    k_cy, k_cx, k_r, k_shape, k_noise = jax.random.split(key, 5)

    # Pixel coordinate grid, shared by every image.
    coords = jnp.arange(size, dtype=jnp.float32)
    gy, gx = jnp.meshgrid(coords, coords, indexing='ij')     # (size, size)

    margin = 0.18 * size
    cy = jax.random.uniform(k_cy, (n, max_shapes), minval=margin, maxval=size - margin)
    cx = jax.random.uniform(k_cx, (n, max_shapes), minval=margin, maxval=size - margin)
    radius = jax.random.uniform(k_r, (n, max_shapes),
                                minval=0.09 * size, maxval=0.20 * size)
    is_circle = jax.random.bernoulli(k_shape, 0.5, (n, max_shapes))

    # Broadcast grid (1,1,H,W) against per-shape params (n,K,1,1) -> (n,K,H,W).
    dy = gy[None, None] - cy[..., None, None]
    dx = gx[None, None] - cx[..., None, None]
    r = radius[..., None, None]
    circ = (dy ** 2 + dx ** 2) <= r ** 2                      # disk
    square = (jnp.abs(dy) <= r) & (jnp.abs(dx) <= r)          # axis-aligned box
    per_shape = jnp.where(is_circle[..., None, None], circ, square)

    mask = jnp.any(per_shape, axis=1).astype(jnp.float32)     # (n, H, W) union

    # Image: dim background + bright foreground + mild Gaussian noise, clipped.
    noise = 0.05 * jax.random.normal(k_noise, (n, size, size))
    image = 0.15 + 0.70 * mask + noise
    image = jnp.clip(image, 0.0, 1.0)

    return image[..., None], mask[..., None]                  # (n,H,W,1) each


# ==== VISUALIZATION ====

def save_prediction_grid(model, images, masks, path: str, n_show: int = 4,
                         threshold: float = 0.5):
    """Save an INPUT | GROUND-TRUTH | PREDICTED comparison grid as a PNG.

    One row per example, three columns. The predicted column shows the raw
    sigmoid probability as a heatmap with the thresholded boundary overlaid, so
    a reader can see the U-Net reconstructs the mask per-pixel. matplotlib is
    imported lazily so importing this module never requires it.
    """
    import os
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_show = int(min(n_show, images.shape[0]))
    logits = model(images[:n_show], train=False)          # (n, H, W, 1)
    probs = np.asarray(jax.nn.sigmoid(logits))[..., 0]    # (n, H, W)
    imgs = np.asarray(images[:n_show])[..., 0]
    gts = np.asarray(masks[:n_show])[..., 0]

    # Per-example IoU for the caption / titles.
    preds_bin = (probs > threshold).astype(np.float32)
    inter = (preds_bin * gts).sum(axis=(1, 2))
    union = preds_bin.sum(axis=(1, 2)) + gts.sum(axis=(1, 2)) - inter
    ious = (inter + 1e-6) / (union + 1e-6)

    col_titles = ["Input image", "Ground-truth mask", "Predicted P(foreground)"]
    fig, axes = plt.subplots(n_show, 3, figsize=(6.6, 2.2 * n_show))
    axes = np.array(axes).reshape(n_show, 3)
    for r in range(n_show):
        axes[r, 0].imshow(imgs[r], cmap="gray", vmin=0.0, vmax=1.0)
        axes[r, 1].imshow(gts[r], cmap="gray", vmin=0.0, vmax=1.0)
        im = axes[r, 2].imshow(probs[r], cmap="magma", vmin=0.0, vmax=1.0)
        # Overlay the thresholded prediction boundary in cyan.
        axes[r, 2].contour(preds_bin[r], levels=[0.5], colors="cyan",
                           linewidths=1.2)
        axes[r, 2].set_ylabel(f"IoU {ious[r]:.2f}", fontsize=9)
        for c in range(3):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
        if r == 0:
            for c in range(3):
                axes[r, c].set_title(col_titles[c], fontsize=10)

    fig.suptitle(
        f"U-Net segmentation on held-out shapes — mean IoU {ious.mean():.2f}",
        fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


# ==== MAIN ====

def main():
    # Run-scale from env with small CPU-friendly defaults; always offline.
    epochs = int(os.environ.get('EPOCHS', 6))
    batch = int(os.environ.get('BATCH', 16))
    size = int(os.environ.get('IMG_SIZE', 32))
    n_data = int(os.environ.get('N_DATA', 128))
    base = int(os.environ.get('BASE', 16))

    print("=" * 60)
    print("U-Net semantic segmentation on synthetic shapes (Flax NNX)")
    print("=" * 60)
    print(f"  epochs={epochs}  batch={batch}  size={size}x{size}  n={n_data}")

    images, masks = make_dataset(n=n_data, size=size)
    fg_frac = float(masks.mean())
    print(f"  data: images {images.shape}, masks {masks.shape} "
          f"(foreground {fg_frac*100:.1f}% of pixels)")

    model = UNet(in_channels=1, out_channels=1, base=base, rngs=nnx.Rngs(0))
    import optax
    optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)

    n_params = sum(p.size for p in jax.tree.leaves(nnx.state(model, nnx.Param)))
    print(f"  parameters: {n_params:,}\n")

    key = jax.random.key(1)
    steps_per_epoch = max(1, n_data // batch)
    for epoch in range(1, epochs + 1):
        key, perm_key = jax.random.split(key)
        perm = jax.random.permutation(perm_key, n_data)
        running = 0.0
        for s in range(steps_per_epoch):
            idx = perm[s * batch:(s + 1) * batch]
            b = {'image': images[idx], 'mask': masks[idx]}
            loss, _ = train_step(model, optimizer, b)
            running += float(loss)
        # Full-dataset IoU/Dice at epoch end (eval mode is identical here).
        logits = model(images, train=False)
        iou, dice = iou_dice(logits, masks)
        print(f"  epoch {epoch:2d}/{epochs} | BCE {running/steps_per_epoch:8.2f} "
              f"| IoU {float(iou):.3f} | Dice {float(dice):.3f}")

    print("\nDone. The decoder's nnx.ConvTranspose layers upsample the mask "
          "back to full resolution; skips restore fine boundaries.")

    # Visualize predictions on a FRESH (held-out) synthetic batch so the grid
    # demonstrates generalization, not memorization of the training set.
    test_images, test_masks = make_dataset(n=8, size=size, seed=12345)
    out_dir = os.environ.get("OUTDIR", "results")
    out_path = os.path.join(out_dir, "unet_masks.png")
    save_prediction_grid(model, test_images, test_masks, out_path, n_show=4)
    print(f"Saved prediction grid to {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
