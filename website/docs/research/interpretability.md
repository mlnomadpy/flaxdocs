---
sidebar_position: 12
title: Saliency, Integrated Gradients & Grad-CAM in JAX
description: "Attribute a CNN's predictions to input pixels in JAX and Flax NNX — vanilla gradient saliency, Integrated Gradients with a completeness check, and Grad-CAM."
keywords: [interpretability, saliency maps, integrated gradients, Grad-CAM, JAX, Flax, NNX, attribution, explainability, feature attribution, XAI]
---

# Interpretability: Saliency, Integrated Gradients & Grad-CAM

**Open the black box.** Take a trained CNN and ask *which pixels drove this prediction?* — with three classic, gradient-based attribution methods you can implement in a few lines of JAX.

:::note Prerequisites
A research-grade guide. You should be comfortable with a convolutional classifier first — see the [simple CNN](/basics/vision/simple-cnn) page. The core trick, differentiating a logit **with respect to the input**, is the same one used in [adversarial training](/research/adversarial-training), so that page is great background.
:::

:::tip What you'll learn
- Compute **vanilla gradient saliency** as $|\partial \text{logit} / \partial x|$ with a single `jax.grad` call w.r.t. the input
- Implement **Integrated Gradients** along a baseline path and verify the **completeness axiom** numerically
- Build a simple **Grad-CAM** by differentiating a logit w.r.t. a convolutional feature map
- Understand why a fully-convolutional + Global-Average-Pool head is the natural setting for Grad-CAM
- Recognise the failure modes: gradient saturation, noisy maps, and degenerate (all-zero) heatmaps
:::

:::info Example Code
See the full implementation: [`examples/advanced/interpretability.py`](https://github.com/mlnomadpy/flaxdocs/tree/master/examples/advanced/interpretability.py)
:::

## Why Interpretability?

A model that predicts *"plus sign"* is only trustworthy if it does so **for the right reasons**. Attribution methods answer a concrete question: how much did each input feature contribute to a particular output? For an image classifier $f$ with class score $f_c$, an attribution assigns a value to every input pixel $x_i$, producing a heatmap you can overlay on the image.

The three methods below trade off along a familiar axis:

- **Vanilla saliency** — one gradient, instant, but noisy and prone to *saturation*.
- **Integrated Gradients** — averages gradients along a path; satisfies clean axioms at the cost of many forward passes.
- **Grad-CAM** — coarse but robust; uses gradients at a *feature map* rather than the raw input.

All three are *post-hoc*: they explain an already-trained network without changing it.

## The Model

We need a CNN that exposes its **last convolutional feature map** so Grad-CAM can hook into it. We split the forward pass into `features()` (input → feature map $A$) and `classify_from_features()` (feature map → logits). The classifier head uses **Global Average Pooling**, which keeps the feature extractor fully convolutional — exactly what Grad-CAM assumes.

```python
import jax
import jax.numpy as jnp
from flax import nnx


class SaliencyCNN(nnx.Module):
    """A tiny CNN that exposes its last conv feature map."""

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
```

We train this on a **self-contained synthetic dataset**: each 28×28 image contains one of three shapes — a filled disk, a square outline, or a plus — drawn at a *random* location. Because the class depends on the shape, not its position, a GAP-head CNN can classify it, and the shape's pixels are exactly what a good attribution map should recover.

## 1. Vanilla Gradient Saliency

The simplest attribution: how much does the class score change if we nudge each pixel? That is just the magnitude of the gradient of the target logit with respect to the input (Simonyan et al., 2013):

$$
S_i(x) = \left| \frac{\partial f_c(x)}{\partial x_i} \right|
$$

In JAX, gradients w.r.t. the input are no harder than gradients w.r.t. parameters — you differentiate a function of `x` while the model is captured in the closure:

```python
def vanilla_saliency(model, x, target: int):
    """Vanilla gradient saliency: |d logit_target / d input|.

    x is a single image with a batch dim, shape (1, 28, 28, 1). The returned
    saliency map has the SAME shape as the input.
    """
    def logit_fn(inp):
        return model(inp)[0, target]              # scalar target logit
    grad = jax.grad(logit_fn)(x)                  # d logit / d x, shape == x
    return jnp.abs(grad)
```

**The catch: saturation.** Once a feature strongly activates the correct class, its gradient often *flattens to zero* — the network is already confident, so nudging that pixel barely moves the logit. Vanilla saliency then assigns near-zero importance to the very pixels that matter. Integrated Gradients fixes exactly this.

## 2. Integrated Gradients

Integrated Gradients (Sundararajan et al., 2017) accumulates gradients along a straight-line path from an information-less **baseline** $x'$ (here, a black image) to the actual input $x$. For pixel $i$:

$$
IG_i = (x_i - x'_i)\int_0^1 \frac{\partial f}{\partial x_i}\big(x' + \alpha (x - x')\big)\, d\alpha
$$

By walking from the baseline to the input, IG sees the gradient *before* the network saturates, so it no longer misses confident features. We approximate the integral with a midpoint Riemann sum and vectorise the path with `jax.vmap`:

```python
def integrated_gradients(model, x, target: int, baseline=None, steps: int = 256):
    if baseline is None:
        baseline = jnp.zeros_like(x)              # black-image baseline

    def logit_fn(inp):
        return model(inp)[0, target]
    grad_fn = jax.grad(logit_fn)

    # Midpoint rule: alphas at the centre of each of `steps` sub-intervals.
    alphas = (jnp.arange(steps) + 0.5) / steps
    interp = baseline + alphas[:, None, None, None, None] * (x - baseline)
    grads = jax.vmap(grad_fn)(interp)             # gradient at each point on the path
    avg_grads = grads.mean(axis=0)                # approximates the path integral
    return (x - baseline) * avg_grads
```

### The Completeness Axiom

IG's headline property is **completeness**: the attributions sum exactly to the difference in output between the input and the baseline.

$$
\sum_i IG_i = f_c(x) - f_c(x')
$$

This is a great built-in test — if your implementation is correct, the two sides match. Our example checks it directly:

```python
def completeness_gap(model, x, target: int, ig, baseline=None):
    """Return (IG.sum(), f(x) - f(baseline)) for the completeness axiom."""
    if baseline is None:
        baseline = jnp.zeros_like(x)
    lhs = float(jnp.sum(ig))
    rhs = float(model(x)[0, target] - model(baseline)[0, target])
    return lhs, rhs
```

Because a ReLU network is piecewise-linear, the path integral is essentially exact — the two sides agree to a relative error of $\sim 10^{-4}$ with a few hundred steps. If your gap is large, you almost always just need **more integration steps**.

## 3. Grad-CAM

Grad-CAM (Selvaraju et al., 2017) produces a *class-discriminative localisation map* by looking at the gradient flowing into the last convolutional feature map $A \in \mathbb{R}^{h\times w\times K}$, rather than the raw pixels. First, global-average-pool the gradients to get one importance weight per channel:

$$
\alpha_k^c = \frac{1}{h\,w}\sum_{i}\sum_{j} \frac{\partial f_c}{\partial A^k_{ij}}
$$

Then take a ReLU-ed weighted combination of the feature maps (ReLU keeps only evidence *for* the class):

$$
L^c_{\text{Grad-CAM}} = \mathrm{ReLU}\!\left(\sum_k \alpha_k^c\, A^k\right)
$$

The `features()` / `classify_from_features()` split lets us differentiate the logit w.r.t. the feature map, then upsample the coarse map back to input resolution:

```python
def grad_cam(model, x, target: int):
    feat = model.features(x)                      # (1, 14, 14, K)

    def logit_fn(f):
        return model.classify_from_features(f)[0, target]
    grads = jax.grad(logit_fn)(feat)              # d logit / d A

    weights = grads.mean(axis=(1, 2), keepdims=True)          # (1, 1, 1, K) = alpha_k
    cam = nnx.relu(jnp.sum(weights * feat, axis=-1))          # (1, 14, 14)
    cam = cam / (cam.max() + 1e-8)                            # normalise to [0, 1]
    cam = jax.image.resize(cam, (cam.shape[0], x.shape[1], x.shape[2]),
                           method='bilinear')                 # -> (1, 28, 28)
    return cam
```

## Training Step

Attribution is only meaningful once the network has actually learned the task, so we first train the classifier with an ordinary supervised step:

```python
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
```

Set up the model and optimizer with the modern NNX API:

```python
model = SaliencyCNN(num_classes=3, rngs=nnx.Rngs(0))
optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
```

## Results / What to Expect

Running the example trains the CNN to 100% accuracy on the synthetic shapes in ~160 steps, then attributes one test image. The key checks are the **attribution properties**, not a decreasing loss:

```
Interpretability: saliency / integrated gradients / Grad-CAM
  epochs=10 batch=64 n=1024 classes=3 ig_steps=256 shapes=('disk', 'square-ring', 'plus')
  dataset: (1024, 28, 28, 1)  labels in [0, 2]
  epoch  8/10 | steps  128 | loss 0.4168 | acc 1.000
  epoch 10/10 | steps  160 | loss 0.2108 | acc 1.000

Explaining image 0 (true shape 'square-ring', predicted 'square-ring')
  saliency map shape:   (1, 28, 28, 1)  (== input (1, 28, 28, 1))
  integrated gradients: shape (1, 28, 28, 1)
    completeness: IG.sum()=+6.877846  f(x)-f(baseline)=+6.875711  |gap|=2.13e-03  rel=3.10e-04
  grad-cam heatmap:     shape (1, 28, 28)  range [0.00, 0.95]
```

Three properties to verify:

- **Saliency map shape equals the input shape** — one importance value per input element.
- **Completeness holds**: `IG.sum()` ≈ `f(x) − f(baseline)` (here to a relative error of `3.1e-04`).
- **Grad-CAM is non-degenerate**: the heatmap spans `[0, 0.95]` rather than collapsing to all zeros.

## Common Pitfalls

**1. Gradient saturation makes vanilla saliency blank**
❌ Rely on raw gradients for a confident model → the most important pixels get ≈0 importance.
✅ Use Integrated Gradients (or SmoothGrad) to integrate over less-saturated regions of the path.

**2. Integrated Gradients with too few steps**
❌ Use a handful of steps and see `IG.sum()` drift away from `f(x) − f(baseline)`.
✅ Increase `steps` (128–256 is plenty here) and check the completeness gap as a correctness test.

**3. A meaningless baseline**
❌ Pick a baseline that isn't information-less (e.g. a real image) and misread the attributions.
✅ Use a black image (or an average/blurred input) so IG measures contribution *relative to "nothing"*.

**4. Grad-CAM heatmap collapses to all zeros**
❌ Global-average-pool the gradients over a head that uses *spatial position* (e.g. flatten → dense) — the positive and negative weights cancel and ReLU zeros the map.
✅ Attribute a fully-convolutional extractor with a **GAP head**, the setting Grad-CAM was designed for.

**5. Explaining an untrained model**
❌ Compute attributions before the network learns the task → you visualise noise.
✅ Train to good accuracy first; the example asserts accuracy > 0.9 before attributing.

## Next steps

- [Uncertainty Estimation](/research/uncertainty) — go beyond "which pixels?" to "how confident?" with calibrated predictions.
- [Adversarial Training](/research/adversarial-training) — the same input-gradient trick, used to *attack* and harden models.
- Back to the [Research hub](/research/advanced-techniques).

## Complete Example

**Saliency, Integrated Gradients & Grad-CAM implementation:**
- [`examples/advanced/interpretability.py`](https://github.com/mlnomadpy/flaxdocs/tree/master/examples/advanced/interpretability.py) — trains a tiny CNN on synthetic shapes, then computes all three attribution maps with a completeness check.

## References

- **Saliency Maps**: [Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps](https://arxiv.org/abs/1312.6034) (Simonyan et al., 2013)
- **Integrated Gradients**: [Axiomatic Attribution for Deep Networks](https://arxiv.org/abs/1703.01365) (Sundararajan et al., ICML 2017)
- **Grad-CAM**: [Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/abs/1610.02391) (Selvaraju et al., ICCV 2017)
- **CAM**: [Learning Deep Features for Discriminative Localization](https://arxiv.org/abs/1512.04150) (Zhou et al., CVPR 2016)
- **SmoothGrad**: [SmoothGrad: removing noise by adding noise](https://arxiv.org/abs/1706.03825) (Smilkov et al., 2017)
