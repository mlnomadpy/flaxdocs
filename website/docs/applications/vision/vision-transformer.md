---
sidebar_position: 1
title: Vision Transformer (ViT) Classifier in Flax NNX
description: "Build a Vision Transformer (ViT) in Flax NNX: patchify images, prepend a CLS token with learned positional embeddings, and classify with a transformer encoder."
keywords: [vision transformer, ViT, flax nnx, jax, patch embedding, self-attention, CLS token, image classification, MNIST]
---

# Vision Transformer (ViT)

Treat an image as a short sequence of patches and let self-attention do the rest — a transformer, not a convolution, learns to classify.

:::note Prerequisites
This guide builds on [Simple CNN](/basics/vision/simple-cnn) for the image-classification setup, [Text Generation with Transformers](/basics/text/simple-transformer) for self-attention, and [ResNet Architecture](/architectures/resnet) for the convolutional baseline we contrast against.
:::

:::tip What you'll learn
- Patchify a 28×28 image into non-overlapping 7×7 patches with a single strided `nnx.Conv`
- Prepend a learned **CLS token** and add learned **positional embeddings** stored as `nnx.Param`
- Stack pre-norm transformer encoder blocks (reusing the shared `TransformerBlock`) with `nnx.List`
- Classify from the CLS token and train with cross-entropy
- Understand why ViTs have a **global receptive field from layer 1** — and why that makes them data-hungry vs CNNs
:::

:::info Example Code
Full runnable script: [`examples/vision/vit.py`](https://github.com/mlnomadpy/flaxdocs/tree/master/examples/vision/vit.py)
:::

## Why a transformer for images?

A CNN processes an image with small kernels that only see a local neighborhood; a pixel on the left of the image cannot influence one on the right until many layers of pooling have shrunk the spatial grid. A **Vision Transformer** takes the opposite stance: it chops the image into a handful of patches, treats them as a sequence of tokens, and applies self-attention so that *every patch can attend to every other patch in the very first layer*. The receptive field is global immediately.

The recipe (from Dosovitskiy et al., 2020) is short:

1. **Patchify** the image into $N$ non-overlapping patches.
2. **Linearly embed** each patch into a $D$-dimensional token.
3. **Prepend** a special learnable classification (CLS) token and **add** learned positional embeddings.
4. Run a standard **transformer encoder**.
5. Read the final CLS token and send it through a linear **classification head**.

### Patch embedding

Split an image $x \in \mathbb{R}^{H \times W \times C}$ into $N = HW / P^2$ patches of size $P \times P$. Each patch is flattened to $\mathbb{R}^{P^2 C}$ and projected by a shared matrix $E \in \mathbb{R}^{(P^2 C) \times D}$. We prepend a learnable token $x_\text{cls}$ and add positional embeddings $E_\text{pos}$:

$$
z_0 = \big[\, x_\text{cls};\ x_p^1 E;\ x_p^2 E;\ \dots;\ x_p^N E \,\big] + E_\text{pos},
\qquad E_\text{pos} \in \mathbb{R}^{(N+1) \times D}.
$$

For our $28\times28$ MNIST-sized image with patch size $P=7$, there are $N = (28/7)^2 = 16$ patches, so the sequence has $N+1 = 17$ tokens (16 patches + CLS). Applying a convolution with kernel size $P$ and **stride** $P$ computes exactly one linear map per non-overlapping patch, which is why patch embedding is just a strided `nnx.Conv`.

### Self-attention

Inside each encoder block, tokens mix through scaled dot-product attention:

$$
\text{Attention}(Q, K, V) = \operatorname{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V.
$$

Because $Q K^\top$ is an $(N+1) \times (N+1)$ matrix, every token — including the CLS token — forms a weighted combination of *all* tokens. There is no locality prior baked in; attention learns which patches matter. Positional embeddings are what tell the otherwise permutation-invariant attention *where* each patch came from.

## Patchifying the image

We implement patch embedding as one strided convolution and flatten the resulting grid into a sequence.

```python
import jax
import jax.numpy as jnp
from flax import nnx
import optax

from shared.models import TransformerBlock
from shared.training_utils import compute_cross_entropy_loss, compute_accuracy


class PatchEmbed(nnx.Module):
    """Split an image into non-overlapping patches and linearly embed each one."""

    def __init__(self, patch_size: int, in_channels: int, dim: int, *, rngs: nnx.Rngs):
        self.proj = nnx.Conv(
            in_channels, dim,
            kernel_size=(patch_size, patch_size),
            strides=(patch_size, patch_size),
            padding="VALID",                    # non-overlapping patches
            rngs=rngs,
        )

    def __call__(self, x):
        x = self.proj(x)                        # (B, H/p, W/p, dim)
        b, h, w, d = x.shape
        return x.reshape(b, h * w, d)           # (B, num_patches, dim)
```

The `padding="VALID"` plus `strides=(patch_size, patch_size)` is the whole trick: the kernel lands on each patch exactly once with no overlap, so a $28\times28$ image becomes a $4\times4$ grid, flattened to 16 patch tokens.

## Building the ViT

The CLS token and positional embeddings are plain learnable arrays — we store them as `nnx.Param`. The encoder is a stack of the shared `TransformerBlock`, wrapped in `nnx.List` so Flax tracks the submodules correctly.

```python
class ViT(nnx.Module):
    def __init__(self, *, image_size=28, patch_size=7, in_channels=1, num_classes=10,
                 dim=64, depth=4, num_heads=4, mlp_dim=128, dropout=0.1, rngs: nnx.Rngs):
        num_patches = (image_size // patch_size) ** 2      # 16

        self.patch_embed = PatchEmbed(patch_size, in_channels, dim, rngs=rngs)

        # Learned CLS token and positional embeddings, small-scale init (0.02).
        self.cls_token = nnx.Param(0.02 * jax.random.normal(rngs.params(), (1, 1, dim)))
        self.pos_embed = nnx.Param(
            0.02 * jax.random.normal(rngs.params(), (1, num_patches + 1, dim))
        )

        # A stack of pre-norm encoder blocks. MUST be nnx.List on Flax 0.12.
        self.blocks = nnx.List([
            TransformerBlock(d_model=dim, num_heads=num_heads, d_ff=mlp_dim,
                             dropout=dropout, rngs=rngs, causal=False)
            for _ in range(depth)
        ])

        self.norm = nnx.LayerNorm(dim, rngs=rngs)
        self.head = nnx.Linear(dim, num_classes, rngs=rngs)

    def __call__(self, x, train: bool = False):
        b = x.shape[0]
        tokens = self.patch_embed(x)                       # (B, num_patches, dim)

        # Prepend the CLS token to every example.
        cls = jnp.broadcast_to(self.cls_token[...], (b, 1, tokens.shape[-1]))
        tokens = jnp.concatenate([cls, tokens], axis=1)    # (B, num_patches+1, dim)

        # Add learned positional embeddings (broadcast over the batch).
        tokens = tokens + self.pos_embed[...]

        for block in self.blocks:
            tokens = block(tokens, train=train)            # global attention every layer

        tokens = self.norm(tokens)
        cls_out = tokens[:, 0]                             # (B, dim) — the CLS token
        return self.head(cls_out)                         # (B, num_classes) logits
```

The shapes flow like this:

```
x            (B, 28, 28, 1)
PatchEmbed   (B, 16, 64)     # 16 non-overlapping 7x7 patches -> dim 64
+ CLS token  (B, 17, 64)     # prepend one learned token
+ pos embed  (B, 17, 64)     # learned position per token
encoder x4   (B, 17, 64)     # every token attends to all 17
CLS -> head  (B, 10)         # classify from token 0
```

Instantiate it with an explicit `nnx.Rngs` and pair it with AdamW:

```python
model = ViT(num_classes=10, dim=64, depth=4, num_heads=4, rngs=nnx.Rngs(0))
optimizer = nnx.Optimizer(model, optax.adamw(1e-3), wrt=nnx.Param)
```

## The training step

This is the standard NNX pattern: a `loss_fn` closed over the model, `nnx.value_and_grad` with `has_aux=True`, then `optimizer.update`. We return the batch accuracy as a lightweight metric.

```python
@nnx.jit
def train_step(model, optimizer, batch):
    def loss_fn(model):
        logits = model(batch["x"], train=True)              # (B, num_classes)
        loss = compute_cross_entropy_loss(logits, batch["y"])
        return loss, logits

    (loss, logits), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    optimizer.update(model, grads)
    accuracy = compute_accuracy(logits, batch["y"])
    return loss, accuracy
```

## Results / What to Expect

The script defaults to tiny, offline **synthetic** data (no downloads): each class lights up a distinct 7×7 patch cell of the 4×4 grid, plus Gaussian noise. That gives a clean per-patch signal a ViT can learn on CPU in seconds. Loss falls and accuracy rises steadily:

```console
$ python vision/vit.py
Vision Transformer on synthetic data
  epochs=20 batch=64
  dataset: images (512, 28, 28, 1), labels (512,)
  epoch  1/20 | steps    8 | loss 2.5321 | acc 0.117
  epoch  5/20 | steps   40 | loss 2.2699 | acc 0.150
  epoch 10/20 | steps   80 | loss 2.0057 | acc 0.281
  epoch 15/20 | steps  120 | loss 1.1706 | acc 0.605
  epoch 20/20 | steps  160 | loss 0.5468 | acc 0.852
Done. Every patch token attends to every other from the first layer — ...
```

On a single fixed batch the model is much faster — 40 steps take the loss from ~2.79 to ~0.39 and accuracy from ~0.12 to ~0.97 (this is what the verification harness checks). Point the script at real MNIST with `SYNTHETIC=0` (requires `tensorflow-datasets`).

**Honest caveat on data-hunger.** A ViT has none of a CNN's built-in inductive biases — no locality, no translation equivariance. It must *learn* that neighboring patches are related, which takes more data and regularization. On small datasets a plain ViT typically **underperforms a comparable CNN/ResNet**; the transformer only pulls ahead at scale (large datasets or pretraining), or with heavy augmentation and distillation (see DeiT). Reach for a ViT when you have data or a pretrained checkpoint, not as a drop-in win on a few thousand images.

## Common Pitfalls

❌ **Storing the encoder blocks in a plain Python `list`.**
✅ Wrap them in `nnx.List([...])` — on Flax 0.12 a bare list is not tracked as a pytree and training crashes.

❌ **Forgetting positional embeddings.** Self-attention is permutation-invariant, so without them the model cannot tell a top-left patch from a bottom-right one.
✅ Add a learned `pos_embed` of shape `(1, num_patches + 1, dim)` — sized for the patches *plus* the CLS token.

❌ **Overlapping patches from the wrong conv config** (e.g. `padding="SAME"` or a stride smaller than the patch size).
✅ Use `padding="VALID"` with `strides=(patch_size, patch_size)` so each patch is embedded exactly once.

❌ **Sizing `pos_embed` to `num_patches` and then indexing `tokens[:, 0]` as the CLS token.** The shapes silently mismatch or you classify from a patch.
✅ Prepend the CLS token *first*, size positional embeddings to `num_patches + 1`, and read token 0.

❌ **Expecting the ViT to beat a CNN on a few thousand images.**
✅ Use strong augmentation, distillation, or a pretrained backbone — or just use a CNN/ResNet when data is scarce.

## Next steps

- [U-Net Image Segmentation](/applications/vision/unet-segmentation) — go from whole-image labels to dense per-pixel predictions.
- [GPT Architecture](/architectures/gpt) — the same transformer encoder machinery, made autoregressive with a causal mask.

## Complete Example

[`examples/vision/vit.py`](https://github.com/mlnomadpy/flaxdocs/tree/master/examples/vision/vit.py) — a runnable Vision Transformer with patch embedding, a learned CLS token and positional embeddings, synthetic-by-default data, and an MNIST switch.

## References

- Dosovitskiy et al., [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) (2020) — the original ViT.
- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (2017) — the transformer and scaled dot-product attention.
- Touvron et al., [Training data-efficient image transformers & distillation through attention](https://arxiv.org/abs/2012.12877) (2020) — DeiT, making ViTs work on smaller data.
