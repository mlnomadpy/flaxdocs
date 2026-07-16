---
sidebar_position: 4
title: Deep Learning for Tabular Data in Flax NNX
description: "Build a tabular DNN in Flax NNX with per-column categorical embeddings concatenated with numeric features, then an MLP head for classification or regression."
keywords: [tabular deep learning, categorical embeddings, entity embeddings, mixed features, nnx.Embed, tabular neural network, classification, regression, Flax NNX, JAX]
---

# Deep Learning for Tabular Data

Turn a spreadsheet of mixed numeric and categorical columns into a differentiable
model by giving every category its own learnable vector.

:::note Prerequisites
You should be comfortable building modules in [Your First Model](/basics/fundamentals/your-first-model)
and running a [Simple Training Loop](/basics/workflows/simple-training). No dataset
download is needed — the table is generated synthetically and offline.
:::

:::tip What you'll learn
- Why raw integer category codes hurt and **entity embeddings** ($\texttt{nnx.Embed}$) fix it
- How to give **each categorical column its own embedding table** and concatenate with numeric features
- A clean `TabularDNN` module: embeddings → concat → MLP → head
- Swapping a **classification head** for a **regression head** (only the head width and loss change)
- Training with cross-entropy and watching accuracy climb from chance to ~98%
:::

:::info Example Code
See the full implementation: [`examples/scientific/tabular_dnn.py`](https://github.com/mlnomadpy/flaxdocs/tree/master/examples/scientific/tabular_dnn.py)
:::

## Why tabular data is different

Most real-world business data is **tabular**: rows of records with a mix of
**numeric** columns (age, price, temperature) and **categorical** columns (city,
device type, subscription plan). Gradient-boosted trees dominate here, but neural
networks are attractive when you want to share a backbone across tasks, ingest
huge cardinalities, or fuse tables with text/images.

The catch is the categorical columns. A category like `city = 3` is *not*
three-times `city = 1` — the integer code is an arbitrary label, not a
magnitude. Feeding it as a raw float tells the network a lie about ordering and
distance. The two classic fixes:

- **One-hot encoding** — a $C$-wide sparse vector per column. Correct, but blows
  up with high cardinality and learns nothing shared between categories.
- **Entity embeddings** — map each category id to a small dense, *learnable*
  vector. This is exactly what an embedding layer does for words, applied to
  table columns. Similar categories can end up with similar vectors, and the
  width stays small even for thousands of categories.

We use entity embeddings, one table per categorical column.

## The recipe: embed, concatenate, MLP

For a row with numeric features $x_{\text{num}} \in \mathbb{R}^{d_{\text{num}}}$
and categorical codes $c_1, \dots, c_K$, each column $k$ owns an embedding table
$E_k \in \mathbb{R}^{V_k \times m_k}$ (with $V_k$ the cardinality and $m_k$ the
embedding width). We look up each code and concatenate everything into one vector:

$$
z = \big[\, x_{\text{num}} \;\Vert\; E_1[c_1] \;\Vert\; \cdots \;\Vert\; E_K[c_K] \,\big]
\in \mathbb{R}^{d_{\text{num}} + \sum_k m_k}
$$

That combined vector then flows through a plain MLP and a task head:

$$
h = \sigma(W_2\,\sigma(W_1 z)), \qquad \hat y = W_{\text{head}}\, h
$$

For **classification** the head outputs $C$ logits and we use softmax
cross-entropy. For **regression** the head outputs a single scalar and we use
mean-squared error — the only things that change are the head width and the loss.

### How wide should an embedding be?

A category with 3 values needs far fewer dimensions than one with 3,000. A common
heuristic grows the width sub-linearly with cardinality (fastai uses
$\min(600, \lceil 1.6\,V^{0.56}\rceil)$). For our tiny toy columns we use a
small rule of thumb:

```python
def embedding_dims(cardinalities):
    # min(8, max(2, card // 2)) per categorical column
    return tuple(min(8, max(2, card // 2)) for card in cardinalities)
```

## The synthetic table

There is no download. We generate four numeric columns (standard normal) and
three integer-coded categorical columns, then build the label from a **hidden
generative process** the model never sees: a nonlinear expansion of the numeric
features (sines, a product interaction, a square) plus a random per-category
effect table. The label is the arg-max of the resulting class scores, so the
model must learn *both* numeric interactions and useful category embeddings.

```python
import jax
import jax.numpy as jnp

NUM_NUMERIC = 4
CAT_CARDINALITIES = (5, 8, 3)   # one entry per categorical column
NUM_CLASSES = 3

def make_dataset(n_samples, *, seed=0, task="classification"):
    key = jax.random.key(seed)
    k_num, k_wnum, k_cat_eff = jax.random.split(key, 3)

    x_num = jax.random.normal(k_num, (n_samples, NUM_NUMERIC))
    cat_cols = [
        jax.random.randint(jax.random.fold_in(key, 100 + i), (n_samples,), 0, card)
        for i, card in enumerate(CAT_CARDINALITIES)
    ]
    x_cat = jnp.stack(cat_cols, axis=1).astype(jnp.int32)   # (N, n_cat)

    # hidden ground truth: nonlinear numeric expansion + category effects
    feat = jnp.concatenate([
        x_num,
        jnp.sin(2.0 * x_num[:, :2]),
        (x_num[:, 0] * x_num[:, 1])[:, None],
        (x_num[:, 2] ** 2)[:, None],
    ], axis=-1)
    score = feat @ jax.random.normal(k_wnum, (feat.shape[1], NUM_CLASSES))
    for i, card in enumerate(CAT_CARDINALITIES):
        table = jax.random.normal(jax.random.fold_in(k_cat_eff, i), (card, NUM_CLASSES)) * 1.5
        score = score + table[x_cat[:, i]]

    y = jnp.argmax(score, axis=-1).astype(jnp.int32)
    return {"x_num": x_num, "x_cat": x_cat, "y": y}
```

## The model in NNX

The module holds **one `nnx.Embed` per categorical column**. Because it is a
*list of submodules*, it must be wrapped in `nnx.List` so NNX registers the
tables as state — a bare Python list crashes on Flax 0.12.

```python
from flax import nnx

class TabularDNN(nnx.Module):
    def __init__(self, num_numeric, cat_cardinalities, embed_dims,
                 hidden, out_features, *, rngs: nnx.Rngs):
        # one embedding table per categorical column (MUST be nnx.List)
        self.embeddings = nnx.List([
            nnx.Embed(num_embeddings=card, features=dim, rngs=rngs)
            for card, dim in zip(cat_cardinalities, embed_dims)
        ])
        concat_dim = num_numeric + sum(embed_dims)
        self.fc1 = nnx.Linear(concat_dim, hidden, rngs=rngs)
        self.fc2 = nnx.Linear(hidden, hidden, rngs=rngs)
        self.dropout = nnx.Dropout(rate=0.1, rngs=rngs)
        self.head = nnx.Linear(hidden, out_features, rngs=rngs)

    def __call__(self, x_num, x_cat, train: bool = False):
        # x_num: (B, num_numeric) float, x_cat: (B, n_cat) int
        embeds = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        h = jnp.concatenate([x_num, *embeds], axis=-1)
        h = nnx.relu(self.fc1(h))
        h = self.dropout(h, deterministic=not train)
        h = nnx.relu(self.fc2(h))
        return self.head(h)          # (B, out_features)
```

The **classification vs. regression** switch lives entirely in `out_features`:

```python
def create_model(rngs, *, hidden=64, task="classification"):
    out_features = 1 if task == "regression" else NUM_CLASSES
    return TabularDNN(
        num_numeric=NUM_NUMERIC,
        cat_cardinalities=CAT_CARDINALITIES,
        embed_dims=embedding_dims(CAT_CARDINALITIES),
        hidden=hidden,
        out_features=out_features,
        rngs=rngs,
    )
```

## The train step

Standard NNX pattern: `nnx.value_and_grad` with `has_aux=True` to carry the
metric, then `optimizer.update`. For classification we use softmax cross-entropy
and report accuracy.

```python
import optax

@nnx.jit
def train_step(model, optimizer, batch):
    def loss_fn(model):
        logits = model(batch["x_num"], batch["x_cat"], train=True)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, batch["y"]).mean()
        acc = jnp.mean(logits.argmax(-1) == batch["y"])
        return loss, acc

    (loss, acc), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    optimizer.update(model, grads)
    return loss, acc

model = create_model(nnx.Rngs(0))
optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
```

The **regression variant** swaps only the head width, the loss, and squeezes the
scalar output:

```python
@nnx.jit
def train_step_regression(model, optimizer, batch):
    def loss_fn(model):
        preds = model(batch["x_num"], batch["x_cat"], train=True)[:, 0]  # (B,)
        loss = jnp.mean((preds - batch["y"]) ** 2)                       # MSE
        return loss, loss
    (loss, _), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    optimizer.update(model, grads)
    return loss, loss
```

## Results / What to expect

Mini-batch training over 2,000 synthetic rows runs in a few seconds on CPU.
Cross-entropy falls steadily and accuracy climbs from the ~33% three-class
baseline into the low 0.80s on the full dataset (individual batches often hit
higher):

```console
$ python scientific/tabular_dnn.py
Tabular DNN (classification) | 2000 rows | 4 numeric + 3 categorical columns (cardinalities (5, 8, 3), embeds (2, 4, 2))

epoch  0 | loss 0.9649 | batch acc 0.535
epoch  4 | loss 0.7292 | batch acc 0.699
epoch  8 | loss 0.6128 | batch acc 0.742
epoch 12 | loss 0.5966 | batch acc 0.770
epoch 14 | loss 0.5344 | batch acc 0.805

Final full-dataset accuracy: 0.818
```

On a small fixed batch trained harder (higher learning rate, 40 steps) the model
essentially memorizes the mapping — loss $1.26 \to 0.08$ and accuracy
$0.19 \to 0.98$ — confirming the embeddings carry real signal. The regression
head on the same features drives MSE down by ~20x over the same 40 steps:

```console
reg mse:  104.42 -> 5.22
```

## Common pitfalls

**Feeding raw category integers as numeric features.**

❌ Treats `city = 3` as three-times `city = 1`, inventing a false ordering.
```python
h = jnp.concatenate([x_num, x_cat.astype(jnp.float32)], axis=-1)
```
✅ Look each code up in a learnable embedding table.
```python
embeds = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
h = jnp.concatenate([x_num, *embeds], axis=-1)
```

**Storing the per-column embeddings in a plain Python list.**

❌ A bare list of submodules is not registered as state and crashes on Flax 0.12.
```python
self.embeddings = [nnx.Embed(c, d, rngs=rngs) for c, d in ...]
```
✅ Wrap submodule collections in `nnx.List`.
```python
self.embeddings = nnx.List([nnx.Embed(c, d, rngs=rngs) for c, d in ...])
```

**One shared embedding table for all columns.**

❌ Column 0's category `2` and column 2's category `2` are unrelated; sharing a
table forces them to collide.
```python
self.embed = nnx.Embed(max_cardinality, dim, rngs=rngs)   # one table, wrong ids
```
✅ Give each column its own table sized to its own cardinality.
```python
self.embeddings = nnx.List([nnx.Embed(card, dim, rngs=rngs)
                            for card, dim in zip(cards, dims)])
```

**Off-by-one on `num_embeddings`.**

❌ Sizing the table to the max id, not the count, indexes out of bounds for the
top category.
```python
nnx.Embed(num_embeddings=x_cat[:, i].max(), features=dim, rngs=rngs)
```
✅ Size it to the cardinality (max id + 1).
```python
nnx.Embed(num_embeddings=int(x_cat[:, i].max()) + 1, features=dim, rngs=rngs)
```

**Comparing a `(B, 1)` regression output against a `(B,)` target.**

❌ Broadcasting turns the MSE into a `(B, B)` matrix and silently trains on garbage.
```python
loss = jnp.mean((model(x_num, x_cat) - y) ** 2)   # (B,1) vs (B,) -> (B,B)
```
✅ Squeeze the head to `(B,)` first.
```python
loss = jnp.mean((model(x_num, x_cat)[:, 0] - y) ** 2)
```

## Next steps

- [Mixture of Experts](/applications/scientific/mixture-of-experts) — route each
  row to specialized sub-networks, a natural next step for heterogeneous tables.
- [Graph Neural Networks (GCN)](/applications/scientific/graph-neural-networks) —
  when rows are not independent but connected in a relational structure.

## Complete Example

[`examples/scientific/tabular_dnn.py`](https://github.com/mlnomadpy/flaxdocs/tree/master/examples/scientific/tabular_dnn.py)
— the full, runnable tabular DNN with synthetic mixed-type data, per-column
embeddings, and both the classification and regression training loops.

## References

- Guo & Berkhahn (2016), *Entity Embeddings of Categorical Variables* — [arXiv:1604.06737](https://arxiv.org/abs/1604.06737)
- Gorishniy et al. (2021), *Revisiting Deep Learning Models for Tabular Data* — [arXiv:2106.11959](https://arxiv.org/abs/2106.11959)
- Arik & Pfister (2019), *TabNet: Attentive Interpretable Tabular Learning* — [arXiv:1908.07442](https://arxiv.org/abs/1908.07442)
- Huang et al. (2020), *TabTransformer: Tabular Data Modeling Using Contextual Embeddings* — [arXiv:2012.06678](https://arxiv.org/abs/2012.06678)
