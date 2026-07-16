---
sidebar_position: 2
title: Toy CLIP Cross-Modal Contrastive Learning
description: "Build a tiny CLIP in Flax NNX — align synthetic images and captions in one embedding space with a symmetric InfoNCE loss and batch retrieval accuracy."
keywords: [CLIP, contrastive learning, cross-modal, InfoNCE, NT-Xent, image-text, flax nnx, jax, retrieval, temperature, dual encoder]
---

# Toy CLIP: Cross-Modal Contrastive Learning

**Teach an image encoder and a text encoder to agree.** This is CLIP at toy scale — align synthetic "digit-like" images with their captions in one shared embedding space, with no labels beyond "these two go together."

:::note Prerequisites
This sits on top of two towers. Comfortable with a [simple CNN](/basics/vision/simple-cnn) for the image side and a [simple transformer / embedding](/basics/text/simple-transformer) for the text side? Good. CLIP is a *cross-modal* twist on [contrastive learning](/research/contrastive-learning) — read that first for the single-modality (SimCLR) version.
:::

:::tip What you'll learn
- Why CLIP aligns **two modalities** instead of two augmentations of one (the key difference from SimCLR)
- Build a **dual encoder**: a CNN image tower and an `nnx.Embed` + mean-pool text tower projecting into a shared space
- The **symmetric InfoNCE** loss — cross-entropy in both directions with the batch diagonal as the correct pairing
- Why you **L2-normalize** embeddings and divide by a **temperature**
- Measure **batch retrieval accuracy** (argmax over each row equals the diagonal) as the task metric
:::

:::info Example Code
See the full, verified implementation: [`examples/adaptation/clip_toy.py`](https://github.com/mlnomadpy/flaxdocs/tree/master/examples/adaptation/clip_toy.py)
:::

## The Motivation

[SimCLR](/research/contrastive-learning) learns image representations by pulling
together two *augmented views of the same image* and pushing apart different
images. Everything happens inside **one modality**.

**CLIP** (Contrastive Language–Image Pre-training) changes the positive pair:
instead of two crops of one photo, the positive pair is *an image and its
caption* — two **different modalities** describing the same thing. There are no
hand-crafted augmentations; the "view" of a cat photo is simply the text "a
photo of a cat." Train on enough image–text pairs and the two encoders converge
on a **shared embedding space** where matching images and captions land near
each other, enabling zero-shot classification and cross-modal retrieval.

This page builds that mechanism end to end at a scale that runs on a CPU in
seconds. Images are synthetic "digit-like" blobs (one fixed pattern per class);
captions are the template **"a photo of the digit N"** rendered as token ids.
The learning signal is identical to real CLIP — only the data is tiny.

## The Math

Each tower maps its input to a vector, which we **L2-normalize** so similarity
is pure cosine similarity. For a batch of $B$ image–caption pairs, let
$u_i$ be the normalized image embedding and $v_j$ the normalized text embedding.
The scaled similarity matrix (logits) is

$$
\ell_{ij} = \frac{u_i^\top v_j}{\tau}, \qquad \|u_i\| = \|v_j\| = 1,
$$

where $\tau$ is a **temperature** (smaller $\tau$ sharpens the distribution).
Within a batch, pair $i$'s image should match pair $i$'s caption and *no other*,
so the correct target for every row is its own index — the **diagonal**. That
turns the problem into two classification tasks, and CLIP averages both:

$$
\mathcal{L} = \tfrac{1}{2}\Big(
\underbrace{-\tfrac{1}{B}\sum_{i} \log \frac{e^{\ell_{ii}}}{\sum_{j} e^{\ell_{ij}}}}_{\text{image}\to\text{text}}
\;+\;
\underbrace{-\tfrac{1}{B}\sum_{i} \log \frac{e^{\ell_{ii}}}{\sum_{j} e^{\ell_{ji}}}}_{\text{text}\to\text{image}}
\Big).
$$

The first term treats each **image as a query** over the $B$ captions; the
second treats each **caption as a query** over the $B$ images. This is exactly
the InfoNCE / NT-Xent objective, made **symmetric** across the two modalities.
The other $B-1$ entries in each row are in-batch negatives, so a larger batch
means a harder, more informative task.

## The Model

Two encoders, one shared projection dimension. The image tower reuses the
shared `ConvEncoder` (a small stride-2 CNN); the text tower embeds tokens and
mean-pools over the sequence. Both project to `proj_dim` and L2-normalize.

```python
import jax, jax.numpy as jnp
from flax import nnx
import optax
from shared.models import ConvEncoder
from shared.training_utils import compute_cross_entropy_loss

class CLIPToy(nnx.Module):
    def __init__(self, num_classes, *, proj_dim=64, embed_dim=64,
                 img_size=28, base=16, temperature=0.07, rngs: nnx.Rngs):
        self.temperature = temperature  # static float, baked into jit

        # Image tower: ConvEncoder halves H,W twice -> img_size/4 spatial.
        self.image_encoder = ConvEncoder(in_channels=1, base=base, rngs=rngs)
        feat_hw = img_size // 4
        self.image_proj = nnx.Linear(base * 2 * feat_hw * feat_hw, proj_dim, rngs=rngs)

        # Text tower: token embedding + mean-pool + projection.
        self.token_embed = nnx.Embed(vocab_size(num_classes), embed_dim, rngs=rngs)
        self.text_proj = nnx.Linear(embed_dim, proj_dim, rngs=rngs)

    @staticmethod
    def _l2_normalize(x):
        return x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)

    def encode_image(self, images):
        h = self.image_encoder(images)                 # (B, H/4, W/4, base*2)
        h = h.reshape((h.shape[0], -1))                # flatten
        return self._l2_normalize(self.image_proj(h))  # (B, proj_dim)

    def encode_text(self, tokens):
        emb = self.token_embed(tokens).mean(axis=1)     # embed + mean-pool
        return self._l2_normalize(self.text_proj(emb))  # (B, proj_dim)

    def __call__(self, images, tokens):
        return self.encode_image(images), self.encode_text(tokens)
```

Only the final token of each caption varies with the class, so the text tower
must learn to route that discriminative digit token through the mean-pool. Note
CLIP uses a *learnable* temperature (a `log`-scale `nnx.Param`); we keep it a
fixed float here so the toy's loss curve stays clean and reproducible.

### The captions and images

Captions follow a single template, so the vocabulary is five shared words plus
one distinct token per digit:

```python
_TEMPLATE_WORDS = ["a", "photo", "of", "the", "digit"]
SEQ_LEN = len(_TEMPLATE_WORDS) + 1

def vocab_size(num_classes):
    return len(_TEMPLATE_WORDS) + num_classes

def build_captions(num_classes):
    template = jnp.broadcast_to(jnp.arange(len(_TEMPLATE_WORDS)),
                                (num_classes, len(_TEMPLATE_WORDS)))
    digit_tokens = len(_TEMPLATE_WORDS) + jnp.arange(num_classes)
    return jnp.concatenate([template, digit_tokens[:, None]], axis=1)
```

Each batch samples **distinct** classes so the diagonal is the *unique* correct
pairing — that keeps batch retrieval accuracy an honest signal (two identical
captions in one batch would make the target ambiguous).

## The Symmetric InfoNCE Loss

The loss is the two-directional cross-entropy from the math above. Because the
correct label for row $i$ is $i$, the targets are just `arange(B)`:

```python
def clip_loss(img_emb, txt_emb, temperature):
    logits = (img_emb @ txt_emb.T) / temperature          # (B, B)
    labels = jnp.arange(img_emb.shape[0])
    loss_i2t = compute_cross_entropy_loss(logits, labels)      # image -> text
    loss_t2i = compute_cross_entropy_loss(logits.T, labels)    # text  -> image
    return 0.5 * (loss_i2t + loss_t2i)

def retrieval_accuracy(img_emb, txt_emb):
    sims = img_emb @ txt_emb.T
    labels = jnp.arange(img_emb.shape[0])
    acc_i2t = jnp.mean(jnp.argmax(sims, axis=1) == labels)     # image -> text
    acc_t2i = jnp.mean(jnp.argmax(sims, axis=0) == labels)     # text  -> image
    return 0.5 * (acc_i2t + acc_t2i)
```

## The Train Step

Standard NNX training: differentiate the loss, return the retrieval accuracy as
aux, and let the optimizer update all parameters.

```python
@nnx.jit
def train_step(model, optimizer, batch):
    def loss_fn(model):
        img_emb = model.encode_image(batch["image"])
        txt_emb = model.encode_text(batch["tokens"])
        loss = clip_loss(img_emb, txt_emb, model.temperature)
        acc = retrieval_accuracy(img_emb, txt_emb)
        return loss, acc
    (loss, acc), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    optimizer.update(model, grads)
    return loss, acc

model = CLIPToy(num_classes=10, rngs=nnx.Rngs(0))
optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
```

## Results / What to Expect

The defaults are offline and synthetic. On CPU the loss collapses toward zero
within ~50 steps and **batch retrieval accuracy reaches 1.0** — every image
retrieves its own caption and vice versa:

```console
$ python adaptation/clip_toy.py
============================================================
Toy CLIP — cross-modal contrastive learning
============================================================
classes=10  vocab=15  seq_len=6  batch=8  synthetic=True
trainable params: 110336

[train] 1 epoch(s) x 300 steps
  epoch 0 step   0 | loss 2.3067 | retrieval acc 0.062
  epoch 0 step  50 | loss 0.0004 | retrieval acc 1.000
  epoch 0 step 100 | loss 0.0001 | retrieval acc 1.000
  epoch 0 step 150 | loss 0.0001 | retrieval acc 1.000
  epoch 0 step 200 | loss 0.0000 | retrieval acc 1.000
  epoch 0 step 250 | loss 0.0000 | retrieval acc 1.000

[assertions]
  loss decreased:      0.0000 < 2.3067 -> True
  retrieval improved:  1.000 > 0.062 -> True
```

For a contrastive objective the scalar loss is only half the story — the signal
that matters is the **task metric**, retrieval accuracy, climbing from chance
($1/B$) to 1.0. Set `SYNTHETIC=0` to swap the synthetic blobs for per-class mean
MNIST images (falls back to synthetic if MNIST cannot be loaded offline), or
tune `BATCH`, `STEPS`, and `NUM_CLASSES` via environment variables.

## Common Pitfalls

- ❌ Skipping L2-normalization and feeding raw embeddings into the dot product.
  Logit magnitudes then depend on vector norms, and the temperature stops meaning
  anything.
  ✅ Normalize both towers to unit length so logits are true cosine similarities.

- ❌ Using a one-directional cross-entropy (image→text only). The text tower gets
  a weaker gradient and alignment is lopsided.
  ✅ Average **both** directions: `0.5 * (loss_i2t + loss_t2i)` on `logits` and `logits.T`.

- ❌ Letting duplicate classes into a batch, then trusting retrieval accuracy.
  Two identical captions make the diagonal target ambiguous and the metric lies.
  ✅ Sample **distinct** classes per batch so the diagonal is the unique positive.

- ❌ Treating this like SimCLR and augmenting one modality into two views.
  CLIP's positive pair is *cross-modal* (image ↔ caption), not two crops.
  ✅ Keep one image encoder and one text encoder; the caption *is* the second view.

- ❌ Setting the temperature far too high (e.g. $\tau=1$) with normalized vectors.
  Logits are capped at $\pm 1$, the softmax barely separates positives, and
  learning crawls.
  ✅ Use a small $\tau$ (CLIP uses ~0.07) — or make it a learnable `log`-scale param.

## Next steps

- [Contrastive Learning with SimCLR](/research/contrastive-learning) — the
  single-modality cousin: same InfoNCE math, but positives are augmentations.
- [LoRA Parameter-Efficient Fine-Tuning](/applications/adaptation/lora-finetuning)
  — once you have a pretrained encoder, adapt it to new tasks with tiny adapters.

## Complete Example

The full, verified script is at
[`examples/adaptation/clip_toy.py`](https://github.com/mlnomadpy/flaxdocs/tree/master/examples/adaptation/clip_toy.py)
— a CPU-friendly, offline toy CLIP with synthetic image–caption defaults, a
symmetric InfoNCE loss, batch retrieval-accuracy reporting, and an optional
MNIST-prototype mode (`SYNTHETIC=0`) that degrades gracefully offline.

## References

- Radford et al., *Learning Transferable Visual Models From Natural Language Supervision* (CLIP, 2021). [arXiv:2103.00020](https://arxiv.org/abs/2103.00020)
- Oord, Li & Vinyals, *Representation Learning with Contrastive Predictive Coding* (InfoNCE, 2018). [arXiv:1807.03748](https://arxiv.org/abs/1807.03748)
- Chen et al., *A Simple Framework for Contrastive Learning of Visual Representations* (SimCLR, 2020). [arXiv:2002.05709](https://arxiv.org/abs/2002.05709)
- Jia et al., *Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision* (ALIGN, 2021). [arXiv:2102.05918](https://arxiv.org/abs/2102.05918)
