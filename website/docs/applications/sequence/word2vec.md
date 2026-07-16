---
sidebar_position: 4
title: Word2Vec Skip-Gram with Negative Sampling in NNX
description: "Train Word2Vec skip-gram embeddings with negative sampling in Flax NNX using two nnx.Embed tables, then recover semantic clusters with cosine nearest neighbours."
keywords: [word2vec, skip-gram, negative sampling, SGNS, word embeddings, nnx.Embed, flax nnx, jax, cosine similarity, nearest neighbours, distributional semantics]
image: img/docusaurus-social-card.jpg
---

# Word2Vec: Skip-Gram with Negative Sampling

**Turn words into geometry.** Word2Vec learns a dense vector for every word so that words used in similar contexts land near each other. This guide builds the skip-gram with negative sampling (SGNS) model in Flax NNX from two `nnx.Embed` tables, derives the negative-sampling loss, trains on a tiny self-contained corpus, and shows the learned clusters via cosine nearest neighbours.

:::note Prerequisites
You should be comfortable defining a module and running a training loop — see [your first model](/basics/fundamentals/your-first-model). Word2Vec is the embedding layer of the sequence family, so the [recurrent networks](/applications/sequence/recurrent-networks) guide is good background on how tokens become vectors.
:::

:::tip What you'll learn
- The **skip-gram objective** — predict context words from a center word
- Why full softmax is intractable and how **negative sampling** replaces it
- How to build SGNS from **two `nnx.Embed` tables** (input + output vectors)
- The **negative-sampling loss** written as a numerically stable sigmoid BCE
- How to read out embeddings and probe them with **cosine nearest neighbours**
:::

:::info Example Code
See the full implementation: [`examples/sequence/word2vec.py`](https://github.com/mlnomadpy/flaxdocs/tree/master/examples/sequence/word2vec.py)
:::

## The distributional hypothesis

> "You shall know a word by the company it keeps." — J.R. Firth

Word2Vec operationalizes this idea: a word's meaning is captured by the distribution of words that surround it. If *king* and *queen* appear in the same slots — next to *royal*, *throne*, *palace* — their vectors should be close. We never label anything; the corpus's own co-occurrence statistics are the supervision.

### The skip-gram model

Skip-gram frames it as a prediction task: given a **center** word $w_c$, predict each **context** word $w_o$ within a window of size $m$. Over a corpus of length $T$ the objective is to maximize the average log-probability of the true context words:

$$
\frac{1}{T} \sum_{t=1}^{T} \sum_{-m \le j \le m,\, j \ne 0} \log P(w_{t+j} \mid w_t)
$$

The natural choice for $P$ is a softmax over the whole vocabulary,

$$
P(w_o \mid w_c) = \frac{\exp(u_{w_o}^\top v_{w_c})}{\sum_{w=1}^{V} \exp(u_{w}^\top v_{w_c})},
$$

where each word has **two** vectors: an *input* vector $v_w$ (used when it is the center) and an *output* vector $u_w$ (used when it is a context). The problem: that denominator sums over the entire vocabulary $V$, so every gradient step costs $O(V)$. For real vocabularies (millions of words) this is hopeless.

## Negative sampling

Negative sampling replaces the expensive softmax with a much cheaper **binary** problem. Instead of asking "which of all $V$ words is the context?", we ask "is this specific (center, context) pair real, or fake?" For each true pair we draw $K$ **negative** words from a noise distribution and train the model to tell the real pair apart from the fakes.

The loss for a single positive pair $(w_c, w_o)$ with negatives $w_{n_1}, \dots, w_{n_K}$ is

$$
\mathcal{L} = -\log \sigma\!\left(u_{w_o}^\top v_{w_c}\right) \;-\; \sum_{k=1}^{K} \log \sigma\!\left(-u_{w_{n_k}}^\top v_{w_c}\right)
$$

where $\sigma(x) = 1/(1 + e^{-x})$ is the logistic sigmoid. The first term pushes the score of the true pair **up** (toward $\sigma \to 1$); each second term pushes a random pair's score **down** (toward $\sigma \to 0$). This is exactly a sigmoid **binary cross-entropy** with label $1$ for the true context and $0$ for the negatives — but summed over only $K + 1$ words instead of $V$.

The original paper samples negatives from the unigram distribution raised to the $3/4$ power, $P_n(w) \propto \text{count}(w)^{0.75}$, which slightly boosts rare words. On a small balanced vocabulary, uniform sampling works fine and keeps the demo transparent.

## Building the model in Flax NNX

The model is just the two vector tables — `center` holds the input vectors $v_w$, `context` holds the output vectors $u_w$. The forward pass looks up the center vector, the positive context vector, and the $K$ negative vectors, then returns their dot-product scores.

```python
import jax
import jax.numpy as jnp
from flax import nnx
import optax


class SkipGramNS(nnx.Module):
    def __init__(self, vocab_size, dim, *, rngs: nnx.Rngs):
        self.center = nnx.Embed(vocab_size, dim, rngs=rngs)   # input vectors v_w
        self.context = nnx.Embed(vocab_size, dim, rngs=rngs)  # output vectors u_w

    def __call__(self, center_ids, context_ids, negative_ids):
        v_c = self.center(center_ids)              # (B, D)    center vectors
        u_o = self.context(context_ids)            # (B, D)    positive contexts
        u_n = self.context(negative_ids)           # (B, K, D) negatives
        pos_score = jnp.sum(v_c * u_o, axis=-1)                 # (B,)
        neg_score = jnp.einsum("bd,bkd->bk", v_c, u_n)          # (B, K)
        return pos_score, neg_score

    def word_vectors(self):
        return self.center.embedding[...]          # (vocab, D) — the vectors you keep
```

After training we export the **center** table as the word vectors (a common alternative is to average `center + context`).

### The negative-sampling loss

Written with `jax.nn.log_sigmoid` — the numerically stable `log(sigmoid(x))` — the loss is a direct transcription of the math above:

```python
def sgns_loss(pos_score, neg_score):
    pos = -jax.nn.log_sigmoid(pos_score)                  # true pair -> score high
    neg = -jax.nn.log_sigmoid(-neg_score).sum(axis=-1)   # negatives -> scores low
    return jnp.mean(pos + neg)
```

## Preparing the data

There is nothing to download: a handful of themed sentences are hardcoded in the script. We tokenize, drop a small stopword set (the tutorial-sized version of Word2Vec's frequent-word subsampling), and slide a window over each sentence to emit `(center, context)` pairs.

```python
def build_skipgram_pairs(corpus, stoi, window=2):
    centers, contexts = [], []
    for line in corpus:
        ids = [stoi[w] for w in tokenize(line)]        # tokenize drops stopwords
        for i, center in enumerate(ids):
            lo, hi = max(0, i - window), min(len(ids), i + window + 1)
            for j in range(lo, hi):
                if j != i:
                    centers.append(center)
                    contexts.append(ids[j])
    return np.asarray(centers, np.int32), np.asarray(contexts, np.int32)


def sample_negatives(key, n, vocab_size, k):
    return jax.random.randint(key, (n, k), 0, vocab_size).astype(jnp.int32)
```

The corpus is built from four disjoint themes — royalty, ocean, space, music — whose words co-occur with each other but never across themes. That gives skip-gram a clean signal and makes the learned clusters easy to see.

## The training step

Standard NNX: `nnx.value_and_grad` with `has_aux=True` so we can return the scores alongside the loss, then `optimizer.update(model, grads)`. Fresh negatives are sampled every step, so the model sees new noise each time.

```python
@nnx.jit
def train_step(model, optimizer, batch):
    def loss_fn(model):
        pos_score, neg_score = model(
            batch["center"], batch["context"], batch["negatives"]
        )
        loss = sgns_loss(pos_score, neg_score)
        return loss, (pos_score, neg_score)

    (loss, aux), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    optimizer.update(model, grads)
    return loss, aux
```

Build the model and optimizer explicitly, then loop over shuffled minibatches:

```python
model = SkipGramNS(vocab_size, dim=32, rngs=nnx.Rngs(0))
optimizer = nnx.Optimizer(model, optax.adam(5e-2), wrt=nnx.Param)

for step in range(40):
    key, nkey = jax.random.split(key)
    batch = {
        "center": centers,
        "context": contexts,
        "negatives": sample_negatives(nkey, n_pairs, vocab_size, K),
    }
    loss, _ = train_step(model, optimizer, batch)
```

## Probing the embeddings

The payoff is geometric: after training, the **cosine similarity** between word vectors reflects semantic relatedness.

$$
\text{sim}(a, b) = \frac{v_a^\top v_b}{\lVert v_a \rVert \, \lVert v_b \rVert}
$$

```python
def nearest_neighbors(vectors, itos, word, stoi, k=5):
    normed = vectors / (jnp.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)
    sims = normed @ normed[stoi[word]]
    order = jnp.argsort(sims)[::-1]
    return [(itos[i], float(sims[i])) for i in order.tolist()
            if itos[i] != word][:k]
```

## Results / What to Expect

The negative-sampling loss drops from ~4.9 to well under 1.5 within 40 steps, and every probe word's nearest neighbours come from its own theme — `king` pulls in `queen` and `royal`, `ocean` pulls in `sail` and `fish`:

```console
$ python sequence/word2vec.py
vocab=49 pairs=324 dim=32 neg=6 epochs=200 batch=128
epoch   0  loss 4.7754
epoch  20  loss 1.3206
epoch 100  loss 1.2696
epoch 199  loss 1.2935

Nearest neighbours (cosine):
  king     -> queen:0.72, rule:0.69, palace:0.68, princess:0.65
  ocean    -> sail:0.80, sailor:0.63, wave:0.61, fish:0.60
  rocket   -> star:0.76, planet:0.64, comet:0.64, reach:0.58
  guitar   -> song:0.74, blend:0.67, drum:0.65, singer:0.62
```

The loss plateaus rather than going to zero — that is expected. With random negatives there is always residual noise the model cannot (and should not) fit; the useful signal is in the **relative geometry** of the vectors, not the absolute loss. Environment knobs `EPOCHS`, `BATCH`, `DIM`, `WINDOW`, and `NEG` let you scale the run.

## Common Pitfalls

- ❌ Using one embedding table and dotting a word with itself.
  ✅ Skip-gram keeps **two** tables (`center` for input, `context` for output); the score is `center(w_c) · context(w_o)`.

- ❌ Computing the loss with a full softmax over the vocabulary.
  ✅ That is $O(V)$ per step. Use **negative sampling**: one positive plus $K$ sampled negatives, scored with a sigmoid.

- ❌ Applying `sigmoid` then `log` by hand and hitting `NaN`s from `log(0)`.
  ✅ Use `jax.nn.log_sigmoid(x)` and `jax.nn.log_sigmoid(-x)` — the stable form of `log σ`.

- ❌ Reusing the **same** negatives every step.
  ✅ Resample negatives each step (split a fresh `jax.random.key`) so the model sees varied noise.

- ❌ Putting a Python list of embedding tables in a plain attribute to hold many tables.
  ✅ On Flax 0.12 wrap submodule lists in `nnx.List([...])` (and dicts in `nnx.Dict({...})`) so they register as state.

## Next steps

- [Simple Transformer](/basics/text/simple-transformer) — contextual embeddings replace one-vector-per-word with representations that depend on the whole sentence.
- [CLIP](/applications/adaptation/clip) — the same "pull matching pairs together, push mismatches apart" idea, scaled up to align images and text.

## Complete Example

Full runnable script — corpus, vocab, skip-gram pairs, negative sampling, the SGNS model, training loop, and cosine nearest neighbours: [`examples/sequence/word2vec.py`](https://github.com/mlnomadpy/flaxdocs/tree/master/examples/sequence/word2vec.py).

## References

- Mikolov et al. (2013), *Efficient Estimation of Word Representations in Vector Space* — [arXiv:1301.3781](https://arxiv.org/abs/1301.3781).
- Mikolov et al. (2013), *Distributed Representations of Words and Phrases and their Compositionality* (negative sampling) — [arXiv:1310.4546](https://arxiv.org/abs/1310.4546).
- Goldberg & Levy (2014), *word2vec Explained: Deriving Mikolov et al.'s Negative-Sampling Word-Embedding Method* — [arXiv:1402.3722](https://arxiv.org/abs/1402.3722).
- Levy & Goldberg (2014), *Neural Word Embedding as Implicit Matrix Factorization* — [NeurIPS 2014](https://papers.nips.cc/paper/2014/hash/feab05aa91085b7a8012516bc3533958-Abstract.html).
