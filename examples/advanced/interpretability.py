"""
Interpretability: Saliency, Integrated Gradients & Grad-CAM (Flax NNX)
=====================================================================
Attribute a CNN's prediction back to its input with three classic methods:
vanilla gradient saliency, Integrated Gradients (with a completeness check),
and Grad-CAM on a convolutional feature map. Trains a tiny CNN on
self-contained synthetic shapes so the whole thing runs offline on CPU.

Run: python advanced/interpretability.py
"""

import os
import jax
import jax.numpy as jnp
from flax import nnx
import optax

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.training_utils import compute_cross_entropy_loss, compute_accuracy


# ==== MODEL ====

class SaliencyCNN(nnx.Module):
    """A tiny CNN that exposes its last conv feature map.

    The forward pass is split into two halves so gradients can be taken w.r.t.
    either the INPUT (saliency / Integrated Gradients) or the intermediate
    feature map A (Grad-CAM). A Global-Average-Pool head keeps the feature
    extractor fully convolutional, which is exactly what Grad-CAM assumes:

      x               (B, 28, 28, 1)
      -> features()   -> A: (B, 14, 14, 32)     [target of Grad-CAM]
      -> classify()   -> logits: (B, num_classes)
    """

    def __init__(self, num_classes: int = 3, *, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(1, 16, kernel_size=(3, 3), rngs=rngs)
        self.conv2 = nnx.Conv(16, 32, kernel_size=(3, 3), rngs=rngs)
        self.fc1 = nnx.Linear(32, 64, rngs=rngs)
        self.fc2 = nnx.Linear(64, num_classes, rngs=rngs)

    def features(self, x):
        """Input -> last conv feature map A (before the classifier head)."""
        x = nnx.relu(self.conv1(x))
        x = nnx.max_pool(x, window_shape=(2, 2), strides=(2, 2))   # 28 -> 14
        x = nnx.relu(self.conv2(x))                                # (B, 14, 14, 32)
        return x

    def classify_from_features(self, feat):
        """Feature map A -> class logits via Global Average Pooling + MLP."""
        pooled = jnp.mean(feat, axis=(1, 2))          # GAP over space -> (B, 32)
        h = nnx.relu(self.fc1(pooled))
        return self.fc2(h)

    def __call__(self, x, train: bool = False):
        return self.classify_from_features(self.features(x))


# ==== ATTRIBUTION METHODS ====

def vanilla_saliency(model, x, target: int):
    """Vanilla gradient saliency: |d logit_target / d input|.

    x is a single image with a batch dim, shape (1, 28, 28, 1). The returned
    saliency map has the SAME shape as the input.
    """
    def logit_fn(inp):
        return model(inp)[0, target]              # scalar target logit
    grad = jax.grad(logit_fn)(x)                  # d logit / d x, shape == x
    return jnp.abs(grad)


def integrated_gradients(model, x, target: int, baseline=None, steps: int = 256):
    r"""Integrated Gradients along the straight-line path baseline -> x.

        IG_i = (x_i - x'_i) * \int_0^1 (d f / d x_i)(x' + a (x - x')) da

    The integral is approximated with the midpoint Riemann rule over `steps`
    points. Satisfies the completeness axiom: IG.sum() ~= f(x) - f(baseline).
    """
    if baseline is None:
        baseline = jnp.zeros_like(x)              # black-image baseline

    def logit_fn(inp):
        return model(inp)[0, target]
    grad_fn = jax.grad(logit_fn)

    # Midpoint rule: alphas at the centre of each of `steps` sub-intervals.
    alphas = (jnp.arange(steps) + 0.5) / steps
    interp = baseline + alphas[:, None, None, None, None] * (x - baseline)  # (S,1,28,28,1)
    grads = jax.vmap(grad_fn)(interp)             # gradient at each point on the path
    avg_grads = grads.mean(axis=0)                # approximates the path integral
    return (x - baseline) * avg_grads


def completeness_gap(model, x, target: int, ig, baseline=None):
    """Return (IG.sum(), f(x) - f(baseline)) for the completeness axiom."""
    if baseline is None:
        baseline = jnp.zeros_like(x)
    lhs = float(jnp.sum(ig))
    rhs = float(model(x)[0, target] - model(baseline)[0, target])
    return lhs, rhs


def grad_cam(model, x, target: int):
    """Grad-CAM: weight the conv feature map by its gradient importance.

    1. A       = features(x)                       (1, 14, 14, K)
    2. alpha_k = mean_{i,j} d logit_target / d A    (global-average-pooled grads)
    3. CAM     = ReLU(sum_k alpha_k * A_k)         coarse (14x14) heatmap
    4. upsample CAM to the input resolution and normalise to [0, 1].
    """
    feat = model.features(x)                      # (1, 14, 14, K)

    def logit_fn(f):
        return model.classify_from_features(f)[0, target]
    grads = jax.grad(logit_fn)(feat)              # d logit / d A

    weights = grads.mean(axis=(1, 2), keepdims=True)          # (1, 1, 1, K)
    cam = nnx.relu(jnp.sum(weights * feat, axis=-1))          # (1, 14, 14)
    cam = cam / (cam.max() + 1e-8)                            # normalise to [0, 1]
    cam = jax.image.resize(cam, (cam.shape[0], x.shape[1], x.shape[2]),
                           method='bilinear')                 # -> (1, 28, 28)
    return cam


# ==== TRAIN STEP ====

@nnx.jit
def train_step(model, optimizer, batch):
    """One supervised gradient step (standard classification)."""
    def loss_fn(model):
        logits = model(batch['x'], train=True)
        loss = compute_cross_entropy_loss(logits, batch['y'])
        return loss, logits

    (loss, logits), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    optimizer.update(model, grads)
    return loss, logits


# ==== DATA ====

# Three translation-invariant shape classes: a filled disk, a square outline,
# and a plus sign. Class depends on SHAPE (not position), so a GAP-head CNN can
# classify it while the shape lives at a random location the attributions
# should recover.
SHAPE_NAMES = ('disk', 'square-ring', 'plus')


def make_dataset(n: int, key, noise: float = 0.1):
    """Synthetic labelled images of shape (n, 28, 28, 1), values in [0, 1].

    Each image draws one of three shapes at a random location, plus noise.
    The shape's pixels are exactly what a good attribution map should recover.
    """
    ks = jax.random.split(key, 4)
    labels = jax.random.randint(ks[0], (n,), 0, len(SHAPE_NAMES))
    cy = jax.random.randint(ks[1], (n,), 8, 20).astype(jnp.float32)   # random centre
    cx = jax.random.randint(ks[2], (n,), 8, 20).astype(jnp.float32)

    yy, xx = jnp.meshgrid(jnp.arange(28), jnp.arange(28), indexing='ij')
    dy = yy[None] - cy[:, None, None]
    dx = xx[None] - cx[:, None, None]
    dist = jnp.sqrt(dy ** 2 + dx ** 2)                 # Euclidean distance field
    cheb = jnp.maximum(jnp.abs(dy), jnp.abs(dx))       # Chebyshev (square) field

    disk = (dist <= 4.0).astype(jnp.float32)                                 # filled disk
    ring = ((cheb <= 5) & (cheb >= 4)).astype(jnp.float32)                   # square outline
    plus = (((jnp.abs(dy) <= 1) | (jnp.abs(dx) <= 1)) & (cheb <= 5)).astype(jnp.float32)
    shapes = jnp.stack([disk, ring, plus], axis=0)     # (3, n, 28, 28)

    img = shapes[labels, jnp.arange(n)]                # gather each image's shape
    img = jnp.clip(img + noise * jax.random.normal(ks[3], (n, 28, 28)), 0.0, 1.0)
    return img[..., None], labels                      # (n, 28, 28, 1), (n,)


def make_batch(images, labels, idx):
    return {'x': images[idx], 'y': labels[idx]}


# ==== MAIN ====

def main():
    # Run-scale from env with small CPU-friendly defaults; SYNTHETIC by default.
    epochs = int(os.environ.get('EPOCHS', 10))
    batch = int(os.environ.get('BATCH', 64))
    n = int(os.environ.get('N', 1024))
    ig_steps = int(os.environ.get('IG_STEPS', 256))
    _ = os.environ.get('SYNTHETIC', '1')          # data is always synthetic (offline)
    num_classes = len(SHAPE_NAMES)

    print("Interpretability: saliency / integrated gradients / Grad-CAM")
    print(f"  epochs={epochs} batch={batch} n={n} classes={num_classes} "
          f"ig_steps={ig_steps} shapes={SHAPE_NAMES}")

    key = jax.random.key(0)
    images, labels = make_dataset(n, key)
    print(f"  dataset: {images.shape}  labels in [0, {num_classes - 1}]")

    model = SaliencyCNN(num_classes=num_classes, rngs=nnx.Rngs(0))
    optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)

    # --- Train the classifier (attribution is only meaningful once it learns) ---
    perm_key = jax.random.key(1)
    step = 0
    for epoch in range(1, epochs + 1):
        perm_key, sub = jax.random.split(perm_key)
        order = jax.random.permutation(sub, n)
        ep_loss, ep_acc, nb = 0.0, 0.0, 0
        for start in range(0, n - batch + 1, batch):
            idx = order[start:start + batch]
            b = make_batch(images, labels, idx)
            loss, logits = train_step(model, optimizer, b)
            ep_loss += float(loss)
            ep_acc += float(compute_accuracy(logits, b['y']))
            nb += 1
            step += 1
        print(f"  epoch {epoch:2d}/{epochs} | steps {step:4d} | "
              f"loss {ep_loss / nb:.4f} | acc {ep_acc / nb:.3f}")

    # --- Attribute a single test image toward its predicted class ---
    x = images[0:1]                                # (1, 28, 28, 1)
    target = int(jnp.argmax(model(x)[0]))
    print(f"\nExplaining image 0 (true shape '{SHAPE_NAMES[int(labels[0])]}', "
          f"predicted '{SHAPE_NAMES[target]}')")

    sal = vanilla_saliency(model, x, target)
    print(f"  saliency map shape:   {sal.shape}  (== input {x.shape})")

    ig = integrated_gradients(model, x, target, steps=ig_steps)
    lhs, rhs = completeness_gap(model, x, target, ig)
    rel = abs(lhs - rhs) / (abs(rhs) + 1e-8)
    print(f"  integrated gradients: shape {ig.shape}")
    print(f"    completeness: IG.sum()={lhs:+.6f}  f(x)-f(baseline)={rhs:+.6f}  "
          f"|gap|={abs(lhs - rhs):.2e}  rel={rel:.2e}")

    cam = grad_cam(model, x, target)
    print(f"  grad-cam heatmap:     shape {cam.shape}  "
          f"range [{float(cam.min()):.2f}, {float(cam.max()):.2f}]")

    print("\nDone. Attributions localise the class-defining shape in the input.")


if __name__ == "__main__":
    main()
