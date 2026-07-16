---
sidebar_position: 1
title: LoRA Parameter-Efficient Fine-Tuning in Flax NNX
description: "Fine-tune a frozen Flax NNX model by training only low-rank adapters with nnx.LoRALinear and nnx.LoRAParam — base weights stay frozen, bit-identical."
keywords: [LoRA, parameter-efficient fine-tuning, PEFT, low-rank adaptation, nnx.LoRALinear, nnx.LoRAParam, flax nnx, jax, frozen weights, DiffState]
---

# LoRA Parameter-Efficient Fine-Tuning

Adapt a pretrained model to a new task by training only tiny low-rank adapters while every base weight stays frozen — bit-for-bit unchanged.

:::note Prerequisites
This builds on the models you adapt and on Flax NNX state filtering. Comfortable
with a [transformer / MLP](/basics/text/simple-transformer) and with
[state and parameter filtering](/basics/fundamentals/understanding-state)? Good.
LoRA is a *different* efficiency lever than
[knowledge distillation](/research/knowledge-distillation) — this page contrasts them.
:::

:::tip What you'll learn
- Why low-rank updates $W' = W + \tfrac{\alpha}{r}BA$ can fine-tune a model with a tiny fraction of the parameters
- Attach adapters with `nnx.LoRALinear` and scope training to `nnx.LoRAParam`
- Why adapter-only training needs **both** `nnx.Optimizer(..., wrt=nnx.LoRAParam)` **and** `nnx.value_and_grad(..., argnums=nnx.DiffState(0, nnx.LoRAParam))`
- Filter frozen base weights with the intersection `nnx.All(nnx.Param, nnx.Not(nnx.LoRAParam))`
- Prove the base is frozen (bit-identical before/after) while adapters learn a new task
:::

:::info Example Code
See the full, verified implementation: [`examples/adaptation/lora_finetuning.py`](https://github.com/mlnomadpy/flaxdocs/tree/master/examples/adaptation/lora_finetuning.py)
:::

## The Motivation

Fully fine-tuning a large model means updating — and storing a fresh copy of —
*every* weight for *every* downstream task. For a model with billions of
parameters that is expensive to train and wasteful to serve: one 10 GB checkpoint
per task.

**LoRA (Low-Rank Adaptation)** takes a different route. Freeze the pretrained
weights entirely and inject a small, trainable, low-rank update alongside each
weight matrix. You train only the adapters — often **well under 1%** of the
parameters — and ship a few megabytes per task instead of a full checkpoint.

Contrast this with [knowledge distillation](/research/knowledge-distillation),
the other efficiency lever: distillation *compresses* knowledge into a smaller
student network (a new, cheaper model), whereas LoRA *keeps the original model
intact* and bolts on cheap, swappable adapters. Distillation changes the model;
LoRA changes almost nothing about it.

## The Math

A pretrained linear layer computes $h = Wx$ with a frozen weight matrix
$W \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$. LoRA hypothesizes that the
*update* needed to adapt to a new task has low intrinsic rank, so it constrains
the change to a rank-$r$ factorization:

$$
W' = W + \Delta W = W + \frac{\alpha}{r}\, B A,
\qquad B \in \mathbb{R}^{d_\text{out} \times r},\;
A \in \mathbb{R}^{r \times d_\text{in}},\; r \ll \min(d_\text{in}, d_\text{out}).
$$

Only $A$ and $B$ are trained; $W$ never moves. The scalar $\alpha/r$ scales the
update. At initialization $B = 0$ (and $A$ is small random), so $\Delta W = 0$
and the adapted model starts out **identical** to the pretrained one — training
can only improve from the pretrained baseline.

**Why "under 1%"?** A full weight matrix has $d_\text{in} \cdot d_\text{out}$
parameters; the adapter has $r\,(d_\text{in} + d_\text{out})$. For a
$4096 \times 4096$ attention projection with $r = 8$ that is
$8 \cdot 8192 = 65{,}536$ adapter params versus $16.7$M base params — about
**0.4%**. The savings grow as the matrices get bigger. (In the toy
$\mathbb{R}^8 \to \mathbb{R}^4$ model below the matrices are *tiny*, so the
adapters are a large fraction — the effect is illustrative, not the payoff.)

Flax's `nnx.LoRALinear` implements exactly this: it holds the frozen base kernel
as an `nnx.Param` and the factors $A, B$ as `nnx.LoRAParam`, and its forward pass
is $y = Wx + (xA)B$ with $B$ initialized to zero.

## The Model

`nnx.LoRALinear` is a drop-in for `nnx.Linear` that carries adapters. Stack two
of them into an MLP:

```python
import jax, jax.numpy as jnp
from flax import nnx
import optax

class LoRAMLP(nnx.Module):
    def __init__(self, in_dim, hidden, out_dim, rank, *, rngs: nnx.Rngs):
        self.l1 = nnx.LoRALinear(in_dim, hidden, lora_rank=rank, rngs=rngs)
        self.l2 = nnx.LoRALinear(hidden, out_dim, lora_rank=rank, rngs=rngs)

    def __call__(self, x):
        return self.l2(nnx.relu(self.l1(x)))
```

Each `LoRALinear` splits into two kinds of variables. `nnx.LoRAParam` is a
**subclass** of `nnx.Param`, which is the crux of everything below: a naive
"train all `nnx.Param`" filter would sweep in the adapters *and* the base. To
select only the frozen base, take the intersection of "is a Param" and "is not a
LoRAParam":

```python
# Frozen base weights = Param AND (NOT LoRAParam).
# Use nnx.All (intersection). The comma form nnx.state(m, A, B) returns TWO
# states, which is not what we want here.
BASE_PARAMS = nnx.All(nnx.Param, nnx.Not(nnx.LoRAParam))

def count_params(model, filt):
    return int(sum(x.size for x in jax.tree.leaves(nnx.state(model, filt))))
```

## Pretrain, Then Freeze and Adapt

The narrative: **pretrain** the whole model on task A (a random linear map),
then **freeze** it and **adapt** to task B (the same inputs, but a rotated and
shifted target — a domain shift) by training *only* the adapters.

### Phase 1 — full fine-tuning on task A

Standard training with `wrt=nnx.Param` updates everything (base + adapters):

```python
@nnx.jit
def pretrain_step(model, optimizer, batch):
    def loss_fn(model):
        preds = model(batch['x'])
        loss = jnp.mean((preds - batch['y']) ** 2)
        return loss, preds
    (loss, preds), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    optimizer.update(model, grads)
    return loss, preds

model = LoRAMLP(8, 64, 4, rank=4, rngs=nnx.Rngs(0))
pre_opt = nnx.Optimizer(model, optax.adam(1e-2), wrt=nnx.Param)  # trains all
```

### Phase 2 — adapter-only fine-tuning on task B

Now freeze the base. Adapter-only training requires **two** changes that must
agree, because `LoRAParam` is a subclass of `Param`:

1. Build the optimizer with `wrt=nnx.LoRAParam` so it only holds/updates adapter state.
2. Restrict the gradient to the adapter subtree with `argnums=nnx.DiffState(0, nnx.LoRAParam)`.

```python
@nnx.jit
def adapt_step(model, optimizer, batch):
    def loss_fn(model):
        preds = model(batch['x'])
        loss = jnp.mean((preds - batch['y']) ** 2)
        return loss, preds
    (loss, preds), grads = nnx.value_and_grad(
        loss_fn,
        argnums=nnx.DiffState(0, nnx.LoRAParam),  # gradient only w.r.t. adapters
        has_aux=True,
    )(model)
    optimizer.update(model, grads)
    return loss, preds

adapt_opt = nnx.Optimizer(model, optax.adam(1e-2), wrt=nnx.LoRAParam)  # adapters only
```

If you set only one of the two (e.g. keep the default `wrt=nnx.Param`), the base
weights *will* move — LoRA's whole promise silently breaks. To make the freeze a
checked invariant, snapshot the base before and after:

```python
def snapshot(model, filt):
    return jax.tree.map(lambda a: jnp.array(a), nnx.state(model, filt))

base_before = snapshot(model, BASE_PARAMS)
# ... run adapt_step in a loop on task B ...
base_after = snapshot(model, BASE_PARAMS)
assert all(bool(jnp.array_equal(x, y))
           for x, y in zip(jax.tree.leaves(base_before),
                           jax.tree.leaves(base_after)))  # bit-identical
```

## Results / What to Expect

Running the example on CPU (defaults are offline and synthetic) prints the
parameter split, drives task-A MSE to near zero, then adapts to the shifted
task B using only the adapters — while the base stays bit-identical:

```console
$ python adaptation/lora_finetuning.py
Base (frozen) params : 836
LoRA (trainable) params: 560  (40.1% of total)

[pretrain task A]
  epoch 0: task-A MSE = 0.0051

[adapt task B — LoRA only]
  epoch 0: task-B MSE = 0.0641

[assertions]
  task-B MSE decreased: 0.0641 < 21.1328 -> True
  base weights bit-identical: True
  LoRA weights changed: True
```

Task-B loss falls from ~21 to ~0.06 by moving **only** the 560 adapter values;
the 836 base parameters are frozen to the bit. (Remember: the 40% adapter share
is an artifact of the tiny $8 \to 64 \to 4$ layers; on a real large model the same
recipe puts the adapters well under 1%.)

## Common Pitfalls

- ❌ `nnx.Optimizer(model, tx, wrt=nnx.Param)` for adapter-only training.
  Since `nnx.LoRAParam` **is** an `nnx.Param`, this trains the base too.
  ✅ Use `wrt=nnx.LoRAParam` **and** `argnums=nnx.DiffState(0, nnx.LoRAParam)` — both.

- ❌ Setting `wrt=nnx.LoRAParam` on the optimizer but leaving the gradient as the
  default (differentiates all params). The optimizer only *applies* to adapters,
  but you still pay to compute base gradients — and a shape mismatch can bite.
  ✅ Match them: scope the gradient with `nnx.DiffState(0, nnx.LoRAParam)`.

- ❌ Filtering the base with the comma form `nnx.state(m, nnx.Param, nnx.Not(nnx.LoRAParam))`,
  expecting the frozen weights. That returns **two** separate states.
  ✅ Take the intersection: `nnx.All(nnx.Param, nnx.Not(nnx.LoRAParam))`.

- ❌ Expecting the adapted model to differ from the pretrained one at step 0.
  With $B = 0$, $\Delta W = 0$ and outputs are identical until you train.
  ✅ That's by design — LoRA starts from the pretrained baseline and can only add.

- ❌ Assuming LoRA always saves ">99%" regardless of shape. On small matrices
  $r(d_\text{in}+d_\text{out})$ can rival $d_\text{in}\,d_\text{out}$.
  ✅ The savings scale with matrix size; pick $r \ll \min(d_\text{in}, d_\text{out})$.

## Next steps

- [Knowledge Distillation](/research/knowledge-distillation) — the complementary
  efficiency lever: compress into a smaller student instead of freezing and adapting.
- [Variational Autoencoders (VAE)](/applications/generative/vae) — more practice
  with custom `nnx.value_and_grad` training loops and losses.

## Complete Example

The full, verified script is at
[`examples/adaptation/lora_finetuning.py`](https://github.com/mlnomadpy/flaxdocs/tree/master/examples/adaptation/lora_finetuning.py)
— a CPU-friendly, offline LoRA demo with synthetic-regression defaults, a
pretrain→freeze→adapt loop, parameter-count reporting, and frozen-base assertions.

## References

- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models* (2021). [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
- Aghajanyan, Zettlemoyer & Gupta, *Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning* (2020). [arXiv:2012.13255](https://arxiv.org/abs/2012.13255)
- Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs* (2023). [arXiv:2305.14314](https://arxiv.org/abs/2305.14314)
