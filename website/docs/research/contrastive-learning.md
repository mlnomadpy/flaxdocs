---
sidebar_position: 1
---

# Contrastive Learning with SimCLR

Learn self-supervised representation learning through contrastive methods. SimCLR enables training powerful visual representations without labeled data by contrasting augmented views of images.

:::info Example Code
See the full implementation: [`examples/13_contrastive_learning_simclr.py`](https://github.com/yourusername/flaxdocs/blob/master/examples/13_contrastive_learning_simclr.py)
:::

## Why Contrastive Learning?

Traditional supervised learning requires massive labeled datasets. Contrastive learning learns from the data itself:

- **Self-supervised**: No labels needed during pre-training
- **Powerful representations**: Often matches supervised performance with linear probing
- **Data efficient**: Can leverage unlabeled data at scale

**Key insight**: Different augmented views of the same image should be similar, while views from different images should be dissimilar.

## The SimCLR Framework

SimCLR (Simple Framework for Contrastive Learning of Visual Representations) consists of four components:

```
Input Image → Augmentation Pipeline → Encoder → Projection Head → Contrastive Loss
```

### 1. Data Augmentation Pipeline

Apply two random augmentations to create positive pairs:

```python
def augment_image(image, rng):
    """Apply augmentation pipeline."""
    key1, key2, key3 = jax.random.split(rng, 3)
    
    # Random crop and flip
    image = random_crop_flip(image, key1, crop_size=24)
    
    # Color jittering (brightness, contrast)
    image = color_jitter(image, key2, brightness=0.4, contrast=0.4)
    
    # Gaussian blur
    image = gaussian_blur(image, key3)
    
    return image
```

**Why augmentation matters**: Forces the model to learn features invariant to these transformations, capturing semantic content rather than superficial patterns.

### 2. Encoder Network

Maps images to representations:

```python
class ContrastiveEncoder(nnx.Module):
    """Encoder with ResNet blocks for better features."""
    
    def __init__(self, hidden_dim: int = 128, *, rngs: nnx.Rngs):
        # Convolutional layers
        self.conv1 = nnx.Conv(1, 32, kernel_size=(3, 3), rngs=rngs)
        self.bn1 = nnx.BatchNorm(32, rngs=rngs)
        
        self.conv2 = nnx.Conv(32, 64, kernel_size=(3, 3), rngs=rngs)
        self.bn2 = nnx.BatchNorm(64, rngs=rngs)
        
        # ResNet blocks for richer representations
        self.resblock1 = ResNetBlock(64, rngs=rngs)
        self.resblock2 = ResNetBlock(64, rngs=rngs)
        
        # Projection head (2-layer MLP)
        self.fc1 = nnx.Linear(64 * 5 * 5, 256, rngs=rngs)
        self.fc2 = nnx.Linear(256, hidden_dim, rngs=rngs)
    
    def __call__(self, x, train: bool = False):
        # Encoder: extract features
        x = self.conv1(x)
        x = self.bn1(x, use_running_average=not train)
        x = nnx.relu(x)
        x = nnx.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        x = self.conv2(x)
        x = self.bn2(x, use_running_average=not train)
        x = nnx.relu(x)
        x = nnx.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        x = self.resblock1(x, train=train)
        x = self.resblock2(x, train=train)
        
        # Projection head
        x = x.reshape((x.shape[0], -1))
        x = self.fc1(x)
        x = nnx.relu(x)
        x = self.fc2(x)
        
        return x
```

### 3. NT-Xent Loss (The Math)

The **Normalized Temperature-scaled Cross Entropy** loss is the heart of SimCLR.

**Setup**: For batch size $N$, we create $2N$ augmented views. Each sample $i$ has one positive pair (its other augmentation) and $2N-2$ negative pairs (all other views).

**Loss formula** for a positive pair $(i, j)$:

$$
\ell_{i,j} = -\log \frac{\exp(\text{sim}(z_i, z_j) / \tau)}{\sum_{k=1}^{2N} \mathbb{1}_{[k \neq i]} \exp(\text{sim}(z_i, z_k) / \tau)}
$$

Where:
- $\text{sim}(u, v) = \frac{u^T v}{\|u\| \|v\|}$ is **cosine similarity**
- $\tau$ is the **temperature** parameter
- $z_i, z_j$ are the projected embeddings

**Final loss** averages over all positive pairs:

$$
\mathcal{L} = \frac{1}{2N} \sum_{k=1}^{N} [\ell_{2k-1, 2k} + \ell_{2k, 2k-1}]
$$

**Implementation**:

```python
def nt_xent_loss(z_i: jnp.ndarray, z_j: jnp.ndarray, temperature: float = 0.5):
    """
    NT-Xent loss for contrastive learning.
    
    Maximizes agreement between differently augmented views
    of the same image while pushing apart different images.
    """
    batch_size = z_i.shape[0]
    
    # L2 normalize (makes similarity = cosine similarity)
    z_i = z_i / jnp.linalg.norm(z_i, axis=1, keepdims=True)
    z_j = z_j / jnp.linalg.norm(z_j, axis=1, keepdims=True)
    
    # Concatenate all representations
    representations = jnp.concatenate([z_i, z_j], axis=0)  # [2N, d]
    
    # Compute similarity matrix
    similarity_matrix = jnp.matmul(representations, representations.T)  # [2N, 2N]
    
    # Labels: each sample's positive is at index +batch_size
    labels = jnp.arange(batch_size)
    labels = jnp.concatenate([labels + batch_size, labels])
    
    # Mask out self-similarity
    mask = jnp.eye(2 * batch_size, dtype=bool)
    similarity_matrix = jnp.where(mask, -1e9, similarity_matrix)
    
    # Temperature scaling
    similarity_matrix = similarity_matrix / temperature
    
    # Cross-entropy loss
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=similarity_matrix,
        labels=labels
    )
    
    return jnp.mean(loss)
```

## Training Loop

The training step applies two augmentations and computes the contrastive loss:

```python
@nnx.jit
def train_step(model: ContrastiveEncoder, optimizer: nnx.Optimizer, 
               batch: Dict[str, jnp.ndarray], rng: jax.random.PRNGKey,
               temperature: float = 0.5):
    """Contrastive training step."""
    
    # Split RNG for independent augmentations
    rng1, rng2 = jax.random.split(rng)
    
    def loss_fn(model: ContrastiveEncoder):
        # Apply two different augmentations to same batch
        aug1 = jax.vmap(augment_image, in_axes=(0, None))(batch['image'], rng1)
        aug2 = jax.vmap(augment_image, in_axes=(0, None))(batch['image'], rng2)
        
        # Get embeddings
        z_i = model(aug1, train=True)
        z_j = model(aug2, train=True)
        
        # Compute contrastive loss
        loss = nt_xent_loss(z_i, z_j, temperature=temperature)
        
        return loss
    
    # Compute gradients and update
    grad_fn = nnx.value_and_grad(loss_fn)
    loss, grads = grad_fn(model)
    optimizer.update(grads)
    
    return {'loss': loss}
```

## Evaluating Representations: Linear Probing

After contrastive pre-training, evaluate by training a **linear classifier** on frozen features:

```python
class LinearClassifier(nnx.Module):
    """Linear probe on frozen encoder."""
    
    def __init__(self, encoder: ContrastiveEncoder, num_classes: int = 10, 
                 *, rngs: nnx.Rngs):
        self.encoder = encoder
        self.classifier = nnx.Linear(128, num_classes, rngs=rngs)
    
    def __call__(self, x, train: bool = False):
        # Get frozen features
        features = self.encoder(x, train=False)
        features = jax.lax.stop_gradient(features)
        
        # Linear classification
        logits = self.classifier(features)
        return logits

# Train only the classifier
model = LinearClassifier(pretrained_encoder, num_classes=10, rngs=nnx.Rngs(0))
optimizer = nnx.Optimizer(model, optax.adam(1e-3))

# Training step only updates classifier weights
@nnx.jit
def train_classifier_step(model, optimizer, batch):
    def loss_fn(model):
        logits = model(batch['image'], train=True)
        one_hot = jax.nn.one_hot(batch['label'], num_classes=10)
        loss = -jnp.mean(jnp.sum(one_hot * jax.nn.log_softmax(logits), axis=-1))
        return loss, logits
    
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model)
    optimizer.update(grads)
    
    accuracy = jnp.mean(jnp.argmax(logits, -1) == batch['label'])
    return {'loss': loss, 'accuracy': accuracy}
```

**Good performance** (85-95% of supervised accuracy) indicates learned representations capture semantic information.

## Understanding the Components

### Why Cosine Similarity?

Cosine similarity normalizes for magnitude, focusing on direction:

$$
\text{sim}(u, v) = \frac{u^T v}{\|u\| \|v\|} = \cos(\theta)
$$

**Benefits**:
- **Scale invariant**: Two vectors with same direction have similarity 1, regardless of magnitude
- **Bounded**: Always in $[-1, 1]$, helping optimization stability
- **Geometric**: Directly measures angle between vectors

### Temperature Parameter $\tau$

Temperature controls distribution sharpness:

**Low temperature** ($\tau = 0.1$):
- Very peaked distributions
- Focuses on hardest negatives
- Can be unstable

**High temperature** ($\tau = 1.0$):
- Smooth distributions  
- Treats all negatives equally
- Weaker learning signal

**Optimal** ($\tau = 0.5$):
- Balances hard negatives with stability
- Typically best performance

**Mathematical effect**:

$$
P(k|i) = \frac{\exp(\text{sim}(z_i, z_k) / \tau)}{\sum_j \exp(\text{sim}(z_i, z_j) / \tau)}
$$

As $\tau \to 0$: $P$ becomes one-hot (most similar only)  
As $\tau \to \infty$: $P$ becomes uniform (all equally important)

### Why Projection Head?

SimCLR adds a 2-layer MLP after the encoder:

```python
# Projection head
x = self.fc1(x)  # Linear layer
x = nnx.relu(x)  # Nonlinearity
x = self.fc2(x)  # Final projection
```

**Key finding**: Train with projection head, but **discard it** for downstream tasks. Use encoder features $h$, not projections $z$.

**Why it helps**: Creates a better space for contrastive learning without constraining the encoder representations.

## Practical Considerations

### Batch Size Matters

Larger batches = more negative examples = better representations:

- **Small** (64): Limited negatives, weaker signal
- **Medium** (256): Good balance for most tasks
- **Large** (4096): Best results, requires significant GPU memory

**If GPU limited**, use gradient accumulation:

```python
# Accumulate gradients over multiple steps
accumulation_steps = 4
batch_size_per_step = 64
effective_batch_size = 256  # 4 * 64
```

### Augmentation is Critical

Quality of augmentations directly impacts representation quality:

**Essential augmentations**:
- Random cropping (most important!)
- Random horizontal flip
- Color jittering

**Beneficial augmentations**:
- Gaussian blur (+1-2% improvement)
- Grayscale
- Solarization

### Training Duration

- **100 epochs**: Learns low-level features (edges, textures)
- **400 epochs**: Learns mid-level features (parts, patterns)
- **1000 epochs**: Learns high-level semantic features

Longer training generally improves downstream task performance.

### Learning Rate Schedule

Use cosine decay with warmup:

```python
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=3e-4,
    warmup_steps=1000,
    decay_steps=num_epochs * steps_per_epoch,
    end_value=1e-5
)
optimizer = nnx.Optimizer(model, optax.adam(schedule))
```

## Mathematical Insights

### Connection to Mutual Information

SimCLR maximizes a lower bound on mutual information between views:

$$
I(z_i; z_j) \geq \log(2N) - \mathcal{L}_{\text{NT-Xent}}
$$

Minimizing contrastive loss = maximizing information shared between augmented views.

### Gradient Analysis

The gradient with respect to embedding $z_i$ is:

$$
\frac{\partial \ell_{i,j}}{\partial z_i} = \frac{1}{\tau} \left[ \sum_{k \neq i} P_{ik} z_k - z_j \right]
$$

where $P_{ik} = \frac{\exp(\text{sim}(z_i, z_k)/\tau)}{\sum_l \exp(\text{sim}(z_i, z_l)/\tau)}$

**Interpretation**:
- **Positive** $z_j$ is pulled closer (negative sign)
- **Negatives** $z_k$ are pushed away (positive sign)
- Weighted by similarity (hard negatives pushed more)

### Alignment and Uniformity

SimCLR optimizes two properties:

**Alignment**: Positive pairs close together

$$
\ell_{\text{align}} = \mathbb{E}_{(x, x^+)} [\|f(x) - f(x^+)\|^2]
$$

**Uniformity**: Features spread on hypersphere

$$
\ell_{\text{uniform}} = \log \mathbb{E}_{x, y} [e^{-2\|f(x) - f(y)\|^2}]
$$

NT-Xent loss balances both: align positives, separate all pairs uniformly.

## Extensions and Variants

### SimCLRv2
- Bigger models, deeper projection heads
- Selective kernel networks
- Self-distillation during fine-tuning

### MoCo (Momentum Contrast)
- Uses momentum encoder
- Queue of negative examples
- More memory efficient

### BYOL (Bootstrap Your Own Latent)
- No negative pairs needed
- Uses predictor head
- Only positive pairs

## Common Pitfalls

### 1. Weak Augmentation
❌ **Problem**: Model learns trivial shortcuts  
✅ **Solution**: Use strong, diverse augmentations

### 2. Small Batch Size
❌ **Problem**: Too few negatives, poor representations  
✅ **Solution**: Use 256+ batch size or gradient accumulation

### 3. Wrong Temperature
❌ **Problem**: Training unstable or doesn't converge  
✅ **Solution**: Start with $\tau = 0.5$, tune if needed

### 4. Using Projection Head for Downstream
❌ **Problem**: Worse performance on downstream tasks  
✅ **Solution**: Use encoder features $h$, discard projection $z$

## Running the Example

Train a contrastive model on MNIST:

```bash
cd examples
python 13_contrastive_learning_simclr.py
```

Expected output:
```
Training for 30 epochs...
Epoch   5/30 | Loss: 2.8432 | Time: 15.23s
Epoch  10/30 | Loss: 2.1245 | Time: 15.18s
...
✓ Training completed!

LINEAR EVALUATION
Training linear classifier on frozen features...
Test Accuracy: 0.9123
```

## Next Steps

- Try different augmentation strategies
- Experiment with batch sizes and temperature
- Apply to your own datasets
- Explore variants like MoCo or BYOL

## References

- **SimCLR Paper**: [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709) (Chen et al., ICML 2020)
- **SimCLRv2**: [Big Self-Supervised Models are Strong Semi-Supervised Learners](https://arxiv.org/abs/2006.10029)
- **Understanding Contrastive Learning**: [What Makes for Good Views for Contrastive Learning?](https://arxiv.org/abs/2005.10243)
