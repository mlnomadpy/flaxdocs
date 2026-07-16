"""
Toy CLIP: Cross-Modal Contrastive Learning
==========================================
A minimal CLIP at toy scale: align synthetic "digit-like" images with their
templated captions ("a photo of the digit N") in a shared embedding space using
a symmetric InfoNCE (NT-Xent) loss over a batch.

Run: python adaptation/clip_toy.py
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

from shared.models import ConvEncoder
from shared.training_utils import compute_cross_entropy_loss


# ==== VOCAB / CAPTION TEMPLATE ====
# Every caption is "a photo of the digit <N>". Only the final token varies with
# the class, so the text tower must learn to route the discriminative digit
# token through mean-pooling. Shared word ids 0..4, digit-token id = 5 + class.
_TEMPLATE_WORDS = ["a", "photo", "of", "the", "digit"]
SEQ_LEN = len(_TEMPLATE_WORDS) + 1  # 5 template words + 1 digit token


def build_captions(num_classes: int) -> jnp.ndarray:
    """Token-id matrix of shape (num_classes, SEQ_LEN) for the caption template."""
    template = jnp.arange(len(_TEMPLATE_WORDS))            # ids 0..4, shared
    digit_tokens = len(_TEMPLATE_WORDS) + jnp.arange(num_classes)  # id 5 + c
    template = jnp.broadcast_to(template, (num_classes, len(_TEMPLATE_WORDS)))
    return jnp.concatenate([template, digit_tokens[:, None]], axis=1)


def vocab_size(num_classes: int) -> int:
    return len(_TEMPLATE_WORDS) + num_classes


# ==== SYNTHETIC IMAGE PROTOTYPES ====

def build_prototypes(key, num_classes: int, img_size: int = 28) -> jnp.ndarray:
    """One fixed random "digit-like" pattern per class, in [0, 1].

    Each class gets a distinct low-frequency blob so a small CNN can tell them
    apart. Shape: (num_classes, img_size, img_size, 1).
    """
    raw = jax.random.normal(key, (num_classes, img_size, img_size, 1))
    # Smooth with a 3x3 box blur so patterns look blob-like, not pure noise.
    kernel = jnp.ones((3, 3, 1, 1)) / 9.0
    raw = jax.lax.conv_general_dilated(
        raw, kernel, window_strides=(1, 1), padding="SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )
    lo = raw.min(axis=(1, 2, 3), keepdims=True)
    hi = raw.max(axis=(1, 2, 3), keepdims=True)
    return (raw - lo) / (hi - lo + 1e-6)


def try_mnist_prototypes(num_classes: int, img_size: int = 28):
    """Best-effort per-class MEAN MNIST image (SYNTHETIC=0). Offline-safe.

    Returns prototypes of shape (num_classes, img_size, img_size, 1) or None if
    MNIST cannot be loaded (no network / tfds). Never raises.
    """
    try:  # pragma: no cover - non-default, environment dependent
        import numpy as np
        import tensorflow_datasets as tfds
        ds = tfds.load("mnist", split="train", batch_size=-1)
        data = tfds.as_numpy(ds)
        images, labels = data["image"].astype("float32") / 255.0, data["label"]
        protos = []
        for c in range(num_classes):
            protos.append(images[labels == (c % 10)].mean(axis=0))
        return jnp.asarray(np.stack(protos))  # (num_classes, 28, 28, 1)
    except Exception as exc:  # noqa: BLE001 - fall back to synthetic
        print(f"  [SYNTHETIC=0] MNIST unavailable ({type(exc).__name__}); "
              f"using synthetic prototypes instead.")
        return None


def make_batch(key, prototypes, captions, batch_size: int, noise: float = 0.15):
    """Sample a batch of image-text pairs with DISTINCT classes.

    Distinct classes keep the diagonal the unique correct pairing, so batch
    retrieval accuracy is a clean signal. Returns a dict of jnp arrays.
    """
    num_classes = prototypes.shape[0]
    k_cls, k_noise = jax.random.split(key)
    replace = batch_size > num_classes
    ids = jax.random.choice(k_cls, num_classes, (batch_size,), replace=replace)
    images = prototypes[ids] + noise * jax.random.normal(k_noise, prototypes[ids].shape)
    return {"image": images, "tokens": captions[ids], "labels": ids}


# ==== MODEL ====

class CLIPToy(nnx.Module):
    """Two encoders projecting images and captions into one L2-normalized space.

    Image tower: shared ConvEncoder (a small CNN) -> linear projection.
    Text tower : nnx.Embed -> mean-pool over the sequence -> linear projection.
    A fixed temperature scales the cosine-similarity logits (CLIP uses a
    learnable log-scale; fixed keeps this toy's loss curve clean).
    """

    def __init__(self, num_classes: int, *, proj_dim: int = 64, embed_dim: int = 64,
                 img_size: int = 28, base: int = 16, temperature: float = 0.07,
                 rngs: nnx.Rngs):
        self.temperature = temperature  # static (plain float): baked into jit

        # Image tower: ConvEncoder halves H,W twice -> (img_size/4) spatial, base*2 ch.
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
        emb = self.token_embed(tokens)                 # (B, SEQ_LEN, embed_dim)
        emb = emb.mean(axis=1)                          # mean-pool over sequence
        return self._l2_normalize(self.text_proj(emb))  # (B, proj_dim)

    def __call__(self, images, tokens):
        return self.encode_image(images), self.encode_text(tokens)


# ==== CONTRASTIVE LOSS + METRIC ====

def clip_loss(img_emb, txt_emb, temperature: float):
    """Symmetric InfoNCE: cross-entropy in BOTH directions with a diagonal target.

    logits[i, j] = <img_i, txt_j> / temperature. The correct pairing for row i
    is column i, so the labels are arange(B) for image->text and, by symmetry,
    for text->image on the transposed logits.
    """
    logits = (img_emb @ txt_emb.T) / temperature       # (B, B)
    labels = jnp.arange(img_emb.shape[0])
    loss_i2t = compute_cross_entropy_loss(logits, labels)      # image queries
    loss_t2i = compute_cross_entropy_loss(logits.T, labels)    # text queries
    return 0.5 * (loss_i2t + loss_t2i)


def retrieval_accuracy(img_emb, txt_emb):
    """Symmetric batch retrieval accuracy (argmax over each row == diagonal)."""
    sims = img_emb @ txt_emb.T
    labels = jnp.arange(img_emb.shape[0])
    acc_i2t = jnp.mean(jnp.argmax(sims, axis=1) == labels)     # image->text
    acc_t2i = jnp.mean(jnp.argmax(sims, axis=0) == labels)     # text->image
    return 0.5 * (acc_i2t + acc_t2i)


# ==== TRAIN STEP ====

@nnx.jit
def train_step(model: CLIPToy, optimizer: nnx.Optimizer, batch):
    def loss_fn(model):
        img_emb = model.encode_image(batch["image"])
        txt_emb = model.encode_text(batch["tokens"])
        loss = clip_loss(img_emb, txt_emb, model.temperature)
        acc = retrieval_accuracy(img_emb, txt_emb)
        return loss, acc
    (loss, acc), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    optimizer.update(model, grads)
    return loss, acc


# ==== MAIN ====

def main():
    epochs = int(os.environ.get("EPOCHS", "1"))
    batch_size = int(os.environ.get("BATCH", "8"))
    steps = int(os.environ.get("STEPS", "300"))
    num_classes = int(os.environ.get("NUM_CLASSES", "10"))
    synthetic = os.environ.get("SYNTHETIC", "1") != "0"
    img_size = 28

    print("=" * 60)
    print("Toy CLIP — cross-modal contrastive learning")
    print("=" * 60)

    # ---- data ----
    key = jax.random.PRNGKey(0)
    proto_key, data_key = jax.random.split(key)
    prototypes = None
    if not synthetic:
        prototypes = try_mnist_prototypes(num_classes, img_size)
    if prototypes is None:
        prototypes = build_prototypes(proto_key, num_classes, img_size)
    captions = build_captions(num_classes)
    print(f"classes={num_classes}  vocab={vocab_size(num_classes)}  "
          f"seq_len={SEQ_LEN}  batch={batch_size}  synthetic={synthetic}")

    # ---- model + optimizer ----
    model = CLIPToy(num_classes, img_size=img_size, rngs=nnx.Rngs(0))
    optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
    n_params = sum(x.size for x in jax.tree.leaves(nnx.state(model, nnx.Param)))
    print(f"trainable params: {n_params}")

    # ---- train ----
    print(f"\n[train] {epochs} epoch(s) x {steps} steps")
    losses, accs = [], []
    for epoch in range(epochs):
        for step in range(steps):
            data_key, k = jax.random.split(data_key)
            batch = make_batch(k, prototypes, captions, batch_size)
            loss, acc = train_step(model, optimizer, batch)
            losses.append(float(loss))
            accs.append(float(acc))
            if step % 50 == 0:
                print(f"  epoch {epoch} step {step:3d} | "
                      f"loss {float(loss):.4f} | retrieval acc {float(acc):.3f}")

    print("\n[assertions]")
    print(f"  loss decreased:      {losses[-1]:.4f} < {losses[0]:.4f} -> "
          f"{losses[-1] < losses[0]}")
    print(f"  retrieval improved:  {accs[-1]:.3f} > {accs[0]:.3f} -> "
          f"{accs[-1] > accs[0]}")


if __name__ == "__main__":
    main()
