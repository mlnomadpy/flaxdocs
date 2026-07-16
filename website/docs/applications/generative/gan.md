---
sidebar_position: 3
title: DCGAN in Flax NNX - Adversarial Image Generation
description: "Build a DCGAN on MNIST with Flax NNX: the minimax game, the non-saturating generator loss, two optimizers, and SpectralNorm to tame mode collapse."
keywords: [GAN, DCGAN, generative adversarial network, flax nnx, jax, spectral normalization, non-saturating loss, mode collapse, minimax, image generation]
---

# Deep Convolutional GAN (DCGAN)

Pit a forger against a detective: two networks train against each other until the fakes become indistinguishable from real digits.

:::note Prerequisites
- [Variational Autoencoder (VAE)](/applications/generative/vae) — the `nnx.ConvTranspose` decoder and MNIST scaling this guide builds on.
- [Custom Training Loops](/research/custom-training-loops) — GANs need two optimizers and hand-written alternating updates.
:::

:::tip What you'll learn
- The **minimax game** behind GANs and how a generator and discriminator co-evolve.
- The **non-saturating** generator loss, and why the naive minimax objective stalls early.
- Driving **two independent `nnx.Optimizer`s** — one per network — with separate `@nnx.jit` steps.
- Applying **`nnx.SpectralNorm`** to the discriminator to bound its Lipschitz constant and stabilize training.
- What **mode collapse** looks like and how to recognize healthy adversarial losses.
:::

:::info Example Code
The complete, runnable script for this guide lives at
[`examples/generative/dcgan.py`](https://github.com/mlnomadpy/flaxdocs/tree/master/examples/generative/dcgan.py).
:::

## The adversarial idea

A [VAE](/applications/generative/vae) learns an *explicit* density by maximizing a
likelihood bound. A **generative adversarial network** takes the opposite route:
it learns the data distribution *implicitly* by playing a game.

Two networks compete:

- The **generator** $G$ maps a random latent vector $z \sim \mathcal{N}(0, I)$ to
  a fake image $G(z)$. It never sees real data directly — it only learns from the
  discriminator's feedback.
- The **discriminator** $D$ takes an image and outputs a single logit: high for
  real, low for fake.

$D$ is trained to tell real from fake; $G$ is trained to fool $D$. As $D$ gets
sharper, $G$ is pushed to produce ever more realistic samples, until — at the
theoretical optimum — the fakes match the real distribution and $D$ can only
guess.

## The math: a minimax game

Goodfellow et al. framed this as a two-player minimax game with value function

$$
\min_G \max_D \; \mathbb{E}_{x \sim p_{\text{data}}}\big[\log D(x)\big]
\;+\; \mathbb{E}_{z \sim p_z}\big[\log\big(1 - D(G(z))\big)\big]
$$

The discriminator maximizes this: it wants $D(x) \to 1$ on real data and
$D(G(z)) \to 0$ on fakes. The generator minimizes the second term: it wants
$D(G(z)) \to 1$.

### The non-saturating trick

In practice, minimizing $\log(1 - D(G(z)))$ directly is a bad idea. Early in
training the discriminator easily rejects the generator's garbage, so
$D(G(z)) \approx 0$ — exactly where $\log(1 - D(G(z)))$ is **flat**, giving the
generator almost no gradient. The generator starves precisely when it needs to
learn the most.

The fix is the **non-saturating** loss: instead of minimizing
$\log(1 - D(G(z)))$, the generator *maximizes* $\log D(G(z))$ — equivalently, it
minimizes the binary cross-entropy of the fake logits against the label **1**:

$$
\mathcal{L}_G = -\,\mathbb{E}_{z}\big[\log D(G(z))\big], \qquad
\mathcal{L}_D = -\,\mathbb{E}_{x}\big[\log D(x)\big] - \mathbb{E}_{z}\big[\log\big(1 - D(G(z))\big)\big]
$$

Both are just sigmoid **binary cross-entropy** with the right target labels —
$1$ for "should look real", $0$ for "should look fake" — which is numerically
stable when computed from logits.

## Building the generator

The generator mirrors a VAE decoder: project the latent vector into a small
spatial grid, then upsample with strided `nnx.ConvTranspose` layers. The final
`nnx.tanh` puts pixels in $[-1, 1]$, so **real images are scaled to $[-1, 1]$ too**.

```python
from flax import nnx

class Generator(nnx.Module):
    def __init__(self, z_dim: int = 64, base: int = 16, *, rngs: nnx.Rngs):
        self.z_dim = z_dim
        self.base = base
        self.fc = nnx.Linear(z_dim, base * 2 * 7 * 7, rngs=rngs)
        self.deconv1 = nnx.ConvTranspose(base * 2, base, kernel_size=(3, 3),
                                         strides=(2, 2), padding='SAME', rngs=rngs)  # 7 -> 14
        self.deconv2 = nnx.ConvTranspose(base, 1, kernel_size=(3, 3),
                                         strides=(2, 2), padding='SAME', rngs=rngs)  # 14 -> 28

    def __call__(self, z):
        h = self.fc(z)
        h = h.reshape(z.shape[0], 7, 7, self.base * 2)
        h = nnx.relu(self.deconv1(h))
        h = self.deconv2(h)
        return nnx.tanh(h)   # outputs in [-1, 1]
```

## Building the discriminator (with SpectralNorm)

The discriminator is a plain strided-conv classifier, but every conv/linear
kernel is wrapped in **`nnx.SpectralNorm`**. Spectral normalization divides each
weight matrix by its largest singular value (estimated with power iteration),
which bounds the network's Lipschitz constant. That keeps the discriminator from
becoming arbitrarily confident — a runaway $D$ produces exploding gradients and
collapses the game — so the two players stay in balance.

```python
class Discriminator(nnx.Module):
    def __init__(self, base: int = 16, *, rngs: nnx.Rngs):
        self.conv1 = nnx.SpectralNorm(
            nnx.Conv(1, base, kernel_size=(3, 3), strides=(2, 2),
                     padding='SAME', rngs=rngs), rngs=rngs)          # 28 -> 14
        self.conv2 = nnx.SpectralNorm(
            nnx.Conv(base, base * 2, kernel_size=(3, 3), strides=(2, 2),
                     padding='SAME', rngs=rngs), rngs=rngs)          # 14 -> 7
        self.fc = nnx.SpectralNorm(
            nnx.Linear(base * 2 * 7 * 7, 1, rngs=rngs), rngs=rngs)

    def __call__(self, x, train: bool = False):
        # update_stats only when we are actually training the discriminator.
        x = nnx.leaky_relu(self.conv1(x, update_stats=train), negative_slope=0.2)
        x = nnx.leaky_relu(self.conv2(x, update_stats=train), negative_slope=0.2)
        x = x.reshape(x.shape[0], -1)
        return self.fc(x, update_stats=train)   # (B, 1) logit
```

`nnx.SpectralNorm` keeps its power-iteration estimate (`u`, `sigma`) as
`BatchStat` variables, not `Param`s — so they are *not* differentiated by
`wrt=nnx.Param`, and we only refresh them (`update_stats=True`) while the
discriminator is the one being optimized.

## Two optimizers

A GAN has two sets of parameters updated by two different objectives, so it needs
**two optimizers**. We use Adam with a low learning rate and $\beta_1 = 0.5$ — the
DCGAN recipe that damps the momentum term Adam would otherwise carry across the
adversarial oscillation.

```python
import optax

rngs = nnx.Rngs(0)
gen = Generator(z_dim=64, rngs=rngs)
disc = Discriminator(rngs=rngs)

opt_g = nnx.Optimizer(gen, optax.adam(2e-4, b1=0.5), wrt=nnx.Param)
opt_d = nnx.Optimizer(disc, optax.adam(2e-4, b1=0.5), wrt=nnx.Param)
```

## The two training steps

Each network gets its own `@nnx.jit` step. The discriminator step pushes real
logits up and fake logits down; the generator step uses the non-saturating loss,
labeling its own fakes as **real**.

The shared `bce_loss(logits, targets)` is exactly sigmoid binary cross-entropy
computed from logits, so we feed it target tensors of ones or zeros.

```python
from shared.training_utils import bce_loss

@nnx.jit
def d_step(gen, disc, opt_d, real, z):
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
    def loss_fn(gen, disc):
        fake = gen(z)
        fake_logits = disc(fake, train=False)   # discriminator frozen
        loss = bce_loss(fake_logits, jnp.ones_like(fake_logits))
        return loss, fake_logits

    # Pass disc as a (non-differentiated) argument so its spectral-norm layers
    # stay at the transform's trace level; value_and_grad differentiates
    # argnums=0 (gen) only.
    (loss, aux), grads = nnx.value_and_grad(loss_fn, has_aux=True)(gen, disc)
    opt_g.update(gen, grads)
    return loss, aux
```

Two subtleties worth calling out:

- In `d_step`, the fake batch is computed *outside* `loss_fn`, so the generator
  is treated as a constant — no gradients leak into $G$ while training $D$.
- In `g_step`, we differentiate w.r.t. `gen` only, but we still pass `disc` **into**
  `value_and_grad`. `nnx.SpectralNorm` mutates its layer's weights in place on
  every call; passing `disc` as a transform argument keeps that mutation at the
  correct JAX trace level and avoids a `TraceContextError`.

The training loop just alternates them, one discriminator update then one
generator update per batch:

```python
d_loss, _ = d_step(gen, disc, opt_d, real, sample_z(kd, batch, z_dim))
g_loss, _ = g_step(gen, disc, opt_g, sample_z(kg, batch, z_dim))
```

## Results / What to Expect

Running the script on CPU with the synthetic defaults produces output like:

```console
$ python generative/dcgan.py
epoch 0: d_loss=1.2278  g_loss=0.6531
epoch 1: d_loss=1.0854  g_loss=0.6295
```

Do **not** expect the losses to fall monotonically — that is the wrong mental
model for a GAN. Because $G$ and $D$ optimize *opposing* objectives, the losses
oscillate: when the generator improves, the discriminator's job gets harder (its
loss rises), and vice versa. A healthy run keeps both losses **finite and bounded**
and hovering in a stable range, with $d\_loss$ roughly around $2\log 2 \approx
1.39$ and $g\_loss$ near $\log 2 \approx 0.69$ at the theoretical equilibrium
where $D(x) = D(G(z)) = 0.5$.

What you *should* watch for instead: `NaN`/`inf` losses (the discriminator
overpowered the generator — spectral norm is your first defense), or a
$g\_loss$ that climbs to the sky while $d\_loss$ crashes to zero (the
discriminator won outright and the generator's gradient vanished).

Set `SYNTHETIC=0` to train on real MNIST via `tfds`, and tune `EPOCHS` / `BATCH`
through environment variables.

## Common Pitfalls

- ❌ Using the naive minimax generator loss, minimizing `log(1 - D(G(z)))`.
  ✅ Use the **non-saturating** loss — BCE of the fake logits against label `1` —
  so the generator gets strong gradients even while its samples are poor.

- ❌ Feeding raw $[0, 1]$ images to a discriminator whose generator ends in `tanh`.
  ✅ Scale real images to $[-1, 1]$ (e.g. `img * 2 - 1`) so real and fake share the
  same range.

- ❌ Sharing one optimizer, or letting the generator step leak gradients into the
  discriminator.
  ✅ Keep **two** `nnx.Optimizer`s; compute the fake batch outside `loss_fn` in
  `d_step`, and differentiate `g_step` w.r.t. the generator only.

- ❌ Applying `sigmoid` inside the discriminator and then using BCE (double sigmoid,
  unstable).
  ✅ Return raw **logits** from the discriminator and use sigmoid-BCE from logits.

- ❌ An unconstrained discriminator that grows arbitrarily confident, exploding the
  generator's gradients (classic **mode collapse**, where $G$ emits one repeated
  sample).
  ✅ Wrap the discriminator's conv/linear kernels in `nnx.SpectralNorm` to bound its
  Lipschitz constant and keep the game balanced.

## Next steps

- [Diffusion Models (DDPM)](/applications/generative/diffusion) — swap the
  adversarial game for iterative denoising, trading training instability for many
  sampling steps.
- [Variational Autoencoder (VAE)](/applications/generative/vae) — the explicit,
  likelihood-based alternative to the implicit GAN objective.

## Complete Example

The full, verified script is at
[`examples/generative/dcgan.py`](https://github.com/mlnomadpy/flaxdocs/tree/master/examples/generative/dcgan.py)
— a CPU-friendly DCGAN with synthetic-data defaults, spectral-normalized
discriminator, two optimizers, and alternating `@nnx.jit` train steps.

## References

- Goodfellow et al., *Generative Adversarial Networks* (2014). [arXiv:1406.2661](https://arxiv.org/abs/1406.2661)
- Radford, Metz & Chintala, *Unsupervised Representation Learning with Deep Convolutional GANs (DCGAN)* (2015). [arXiv:1511.06434](https://arxiv.org/abs/1511.06434)
- Miyato et al., *Spectral Normalization for Generative Adversarial Networks* (2018). [arXiv:1802.05957](https://arxiv.org/abs/1802.05957)
- Arjovsky, Chintala & Bottou, *Wasserstein GAN* (2017). [arXiv:1701.07875](https://arxiv.org/abs/1701.07875)
