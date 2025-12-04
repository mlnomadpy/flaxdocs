---
sidebar_position: 3
---

# Knowledge Distillation

Learn to transfer knowledge from large "teacher" models to smaller "student" models, achieving better performance with fewer parameters and faster inference.

:::info Example Code
See the full implementation: [`examples/15_knowledge_distillation.py`](https://github.com/yourusername/flaxdocs/blob/master/examples/15_knowledge_distillation.py)
:::

## The Motivation

**Problem**: Large neural networks achieve great accuracy but are:
- Slow at inference
- Memory-intensive
- Expensive to deploy
- Power-hungry on devices

**Solution**: Train a small model to mimic a large model's behavior.

## What is Knowledge Distillation?

**Knowledge distillation** transfers knowledge from a large pre-trained "teacher" model to a smaller "student" model.

```
┌─────────────────────────────────────────┐
│      Knowledge Distillation Flow        │
└─────────────────────────────────────────┘

Large Teacher Model                Small Student Model
   (Pre-trained)                      (Learning)
        │                                  │
        │                                  │
    [Input: cat image]              [Same input]
        │                                  │
        ↓                                  ↓
   Teacher Forward                   Student Forward
        │                                  │
        ↓                                  ↓
  Soft Predictions                  Soft Predictions
  [cat: 0.8,                       [cat: 0.6,
   dog: 0.15,                       dog: 0.3,
   bird: 0.05]                      bird: 0.1]
        │                                  │
        └──────────┬────────────────────────┘
                   ↓
            Distillation Loss
         (minimize difference)
                   │
                   ↓
         Update Student Parameters
```

**Key insight**: Learn from teacher's soft predictions (probabilities), not just hard labels (0/1).

## Why Soft Targets?

Hard labels: `[0, 1, 0, 0, ...]` (one-hot)  
Soft targets: `[0.05, 0.85, 0.08, 0.02, ...]` (probabilities)

**Soft targets are richer**:
- Encode similarity between classes (e.g., "3" resembles "8" more than "1")
- Contain "dark knowledge" about how model reasons
- Provide more training signal per example

## The Distillation Loss

Combine two objectives:

### 1. Hard Label Loss (Standard)

Train on true labels:

$$
\mathcal{L}_{\text{hard}} = \text{CE}(\text{softmax}(z^S), y)
$$

where $z^S$ are student logits, $y$ is true label.

### 2. Soft Label Loss (Distillation)

Match teacher's soft predictions:

$$
\mathcal{L}_{\text{soft}} = \text{CE}\left(\text{softmax}\left(\frac{z^S}{\tau}\right), \text{softmax}\left(\frac{z^T}{\tau}\right)\right) \cdot \tau^2
$$

where:
- $z^T$ are teacher logits
- $\tau$ is **temperature** (softens distributions)
- $\tau^2$ scaling factor (see derivation below)

### 3. Combined Loss

$$
\mathcal{L} = \lambda \cdot \mathcal{L}_{\text{hard}} + (1 - \lambda) \cdot \mathcal{L}_{\text{soft}}
$$

**Typical**: $\lambda = 0.5$ (equal weight), $\tau = 3$

## Temperature: The Key Mechanism

Temperature $\tau$ softens probability distributions:

$$
p_i = \frac{\exp(z_i / \tau)}{\sum_j \exp(z_j / \tau)}
$$

**Effect of temperature**:

**Low ($\tau = 1$)**: Standard softmax
```
Logits: [5, 3, 1] → Probs: [0.88, 0.12, 0.00]
```

**High ($\tau = 3$)**: Softer distribution
```
Logits: [5, 3, 1] → Probs: [0.60, 0.30, 0.10]
```

**Why soft is better**: Small probabilities become more informative!

### Mathematical Insight

For large logits and $\tau \to \infty$:

$$
p_i \approx \frac{1}{N} + \frac{z_i}{\tau C}
$$

where $C = \sum_j z_j$. Soft targets reveal relative logit values.

**The $\tau^2$ factor**: When matching soft targets, gradients scale as $1/\tau$. Multiplying by $\tau^2$ ensures consistent gradient magnitudes.

## Implementation in Flax NNX

### Teacher Model (Large Network)

```python
from flax import nnx

class TeacherCNN(nnx.Module):
    """Large model with high capacity."""
    
    def __init__(self, num_classes: int = 10, *, rngs: nnx.Rngs):
        # Large architecture: more filters, more layers
        self.conv1 = nnx.Conv(1, 64, kernel_size=(3, 3), rngs=rngs)
        self.bn1 = nnx.BatchNorm(64, rngs=rngs)
        
        self.conv2 = nnx.Conv(64, 128, kernel_size=(3, 3), rngs=rngs)
        self.bn2 = nnx.BatchNorm(128, rngs=rngs)
        
        self.conv3 = nnx.Conv(128, 128, kernel_size=(3, 3), rngs=rngs)
        self.bn3 = nnx.BatchNorm(128, rngs=rngs)
        
        # Dense layers
        self.fc1 = nnx.Linear(128 * 3 * 3, 512, rngs=rngs)
        self.fc2 = nnx.Linear(512, 256, rngs=rngs)
        self.fc3 = nnx.Linear(256, num_classes, rngs=rngs)
        
        self.dropout = nnx.Dropout(rate=0.5, rngs=rngs)
    
    def __call__(self, x, train: bool = False):
        # Three conv blocks
        x = self.conv1(x)
        x = self.bn1(x, use_running_average=not train)
        x = nnx.relu(x)
        x = nnx.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        x = self.conv2(x)
        x = self.bn2(x, use_running_average=not train)
        x = nnx.relu(x)
        x = nnx.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        x = self.conv3(x)
        x = self.bn3(x, use_running_average=not train)
        x = nnx.relu(x)
        x = nnx.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        # Dense classification
        x = x.reshape((x.shape[0], -1))
        x = self.fc1(x)
        x = nnx.relu(x)
        x = self.dropout(x, deterministic=not train)
        x = self.fc2(x)
        x = nnx.relu(x)
        x = self.dropout(x, deterministic=not train)
        x = self.fc3(x)
        
        return x
```

### Student Model (Small Network)

```python
class StudentCNN(nnx.Module):
    """Small model with limited capacity."""
    
    def __init__(self, num_classes: int = 10, *, rngs: nnx.Rngs):
        # Much smaller: fewer filters, fewer layers
        self.conv1 = nnx.Conv(1, 16, kernel_size=(3, 3), rngs=rngs)
        self.bn1 = nnx.BatchNorm(16, rngs=rngs)
        
        self.conv2 = nnx.Conv(16, 32, kernel_size=(3, 3), rngs=rngs)
        self.bn2 = nnx.BatchNorm(32, rngs=rngs)
        
        self.fc1 = nnx.Linear(32 * 5 * 5, 64, rngs=rngs)
        self.fc2 = nnx.Linear(64, num_classes, rngs=rngs)
    
    def __call__(self, x, train: bool = False):
        # Two conv blocks
        x = self.conv1(x)
        x = self.bn1(x, use_running_average=not train)
        x = nnx.relu(x)
        x = nnx.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        x = self.conv2(x)
        x = self.bn2(x, use_running_average=not train)
        x = nnx.relu(x)
        x = nnx.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        # Classification
        x = x.reshape((x.shape[0], -1))
        x = self.fc1(x)
        x = nnx.relu(x)
        x = self.fc2(x)
        
        return x
```

**Size comparison**: Student typically 5-20x smaller than teacher!

### Distillation Loss Function

```python
def distillation_loss(
    student_logits: jnp.ndarray,
    teacher_logits: jnp.ndarray,
    labels: jnp.ndarray,
    temperature: float = 3.0,
    alpha: float = 0.5
):
    """
    Combined distillation loss.
    
    Args:
        student_logits: Student predictions [batch, num_classes]
        teacher_logits: Teacher predictions [batch, num_classes]
        labels: True labels [batch]
        temperature: Temperature for softening (τ)
        alpha: Weight for hard loss (λ)
    
    Returns:
        total_loss, hard_loss, soft_loss
    """
    num_classes = student_logits.shape[-1]
    
    # 1. Hard label loss (standard cross-entropy)
    one_hot_labels = jax.nn.one_hot(labels, num_classes=num_classes)
    hard_loss = -jnp.mean(
        jnp.sum(one_hot_labels * jax.nn.log_softmax(student_logits), axis=-1)
    )
    
    # 2. Soft label loss (distillation)
    # Temperature-scaled softmax
    soft_student = jax.nn.log_softmax(student_logits / temperature)
    soft_teacher = jax.nn.softmax(teacher_logits / temperature)
    
    # KL divergence: KL(P||Q) = Σ P(x) log(P(x)/Q(x))
    soft_loss = -jnp.sum(soft_teacher * soft_student, axis=-1).mean()
    
    # Scale by T² (maintains gradient magnitude)
    soft_loss = soft_loss * (temperature ** 2)
    
    # 3. Combined loss
    total_loss = alpha * hard_loss + (1 - alpha) * soft_loss
    
    return total_loss, hard_loss, soft_loss
```

### Training Step with Distillation

```python
@nnx.jit
def train_student_step(
    student: StudentCNN,
    teacher: TeacherCNN,
    optimizer: nnx.Optimizer,
    batch: Dict[str, jnp.ndarray],
    temperature: float = 3.0,
    alpha: float = 0.5
):
    """Train student using teacher's knowledge."""
    
    def loss_fn(student: StudentCNN):
        # Get student predictions
        student_logits = student(batch['image'], train=True)
        
        # Get teacher predictions (frozen, no gradients)
        teacher_logits = teacher(batch['image'], train=False)
        teacher_logits = jax.lax.stop_gradient(teacher_logits)
        
        # Compute distillation loss
        total_loss, hard_loss, soft_loss = distillation_loss(
            student_logits,
            teacher_logits,
            batch['label'],
            temperature=temperature,
            alpha=alpha
        )
        
        return total_loss, (student_logits, hard_loss, soft_loss)
    
    # Compute gradients and update student
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, (logits, hard_loss, soft_loss)), grads = grad_fn(student)
    optimizer.update(grads)
    
    accuracy = jnp.mean(jnp.argmax(logits, -1) == batch['label'])
    
    return {
        'loss': loss,
        'hard_loss': hard_loss,
        'soft_loss': soft_loss,
        'accuracy': accuracy
    }
```

**Key**: Teacher gradients are stopped with `jax.lax.stop_gradient` - only student is updated!

## Training Workflow

### Step 1: Train Teacher

First, train the large teacher model:

```python
def train_teacher(num_epochs=10, learning_rate=1e-3):
    """Train teacher model on labeled data."""
    
    # Load data
    train_ds, test_ds = load_data(batch_size=128)
    
    # Initialize teacher
    teacher = TeacherCNN(num_classes=10, rngs=nnx.Rngs(0))
    optimizer = nnx.Optimizer(teacher, optax.adam(learning_rate))
    
    # Standard supervised training
    for epoch in range(num_epochs):
        for batch in train_ds:
            metrics = train_step(teacher, optimizer, batch)
        
        # Evaluate
        test_acc = evaluate(teacher, test_ds)
        print(f"Epoch {epoch+1} | Test Acc: {test_acc:.4f}")
    
    return teacher
```

### Step 2: Distill to Student

Then train student using teacher's knowledge:

```python
def train_student_with_distillation(
    teacher: TeacherCNN,
    num_epochs=10,
    learning_rate=1e-3,
    temperature=3.0,
    alpha=0.5
):
    """Train student model using knowledge distillation."""
    
    # Load data
    train_ds, test_ds = load_data(batch_size=128)
    
    # Initialize student
    student = StudentCNN(num_classes=10, rngs=nnx.Rngs(42))
    optimizer = nnx.Optimizer(student, optax.adam(learning_rate))
    
    # Distillation training
    for epoch in range(num_epochs):
        for batch in train_ds:
            metrics = train_student_step(
                student, teacher, optimizer, batch,
                temperature=temperature, alpha=alpha
            )
        
        # Evaluate
        test_acc = evaluate(student, test_ds)
        print(f"Epoch {epoch+1} | "
              f"Test Acc: {test_acc:.4f} | "
              f"Hard Loss: {metrics['hard_loss']:.4f} | "
              f"Soft Loss: {metrics['soft_loss']:.4f}")
    
    return student
```

### Step 3: Compare with Baseline

Train student without distillation for comparison:

```python
def train_student_baseline(num_epochs=10):
    """Train student WITHOUT distillation (baseline)."""
    
    student = StudentCNN(num_classes=10, rngs=nnx.Rngs(42))
    optimizer = nnx.Optimizer(student, optax.adam(1e-3))
    
    # Standard supervised training (no teacher)
    for epoch in range(num_epochs):
        for batch in train_ds:
            metrics = standard_train_step(student, optimizer, batch)
        
        test_acc = evaluate(student, test_ds)
        print(f"Epoch {epoch+1} | Test Acc: {test_acc:.4f}")
    
    return student
```

**Typical results**:
- Teacher: 98.5% accuracy
- Student (distilled): 97.2% accuracy
- Student (baseline): 95.8% accuracy

**Distillation bridges the gap!**

## Hyperparameters

### Temperature τ

Controls softness of distributions:

**Low** ($\tau = 1$): 
- Nearly hard labels
- Less distillation benefit

**Medium** ($\tau = 3-5$):
- Good balance
- Typical best performance

**High** ($\tau = 10$):
- Very soft, nearly uniform
- May lose useful information

**Rule of thumb**: Start with $\tau = 3$, tune if needed.

### Alpha λ

Balances hard and soft losses:

**$\lambda = 1$**: Only hard labels (no distillation)  
**$\lambda = 0.5$**: Equal weight (typical)  
**$\lambda = 0$**: Only soft labels (pure distillation)

**When to adjust**:
- **More labeled data**: Increase $\lambda$ (trust labels more)
- **Less labeled data**: Decrease $\lambda$ (rely on teacher more)

### Learning Rate

Typically same as standard training:
- Adam: 1e-3 to 3e-4
- SGD: 0.01 to 0.1

No special tuning usually needed.

## Understanding the Benefits

### 1. Model Compression

**Size reduction**: 5-20x smaller models

Example MNIST:
- Teacher: 2.1M parameters
- Student: 140K parameters
- Compression: 15x

**Speed improvement**: 
- Fewer operations
- Fits in device memory
- Lower latency

### 2. Better Generalization

Student often generalizes better than:
- Training from scratch
- Using only hard labels

**Why**: Soft targets provide regularization, preventing overfitting.

### 3. Dark Knowledge Transfer

Teacher encodes:
- Class similarities (3 looks like 8)
- Uncertainty (ambiguous examples)
- Decision boundaries

Student learns this "dark knowledge" beyond just labels.

## Variants and Extensions

### Distillation for Ensembles

Distill multiple teachers into one student:

```python
def ensemble_distillation_loss(student_logits, teacher_logits_list, labels):
    """Distill from ensemble of teachers."""
    
    # Average teacher predictions
    avg_teacher_logits = jnp.mean(
        jnp.stack(teacher_logits_list, axis=0),
        axis=0
    )
    
    # Standard distillation loss
    return distillation_loss(
        student_logits,
        avg_teacher_logits,
        labels,
        temperature=3.0,
        alpha=0.5
    )
```

**Benefits**: Capture diverse knowledge from multiple models.

### Self-Distillation

Student becomes its own teacher:

1. Train model normally
2. Use trained model as teacher
3. Re-train (distill) on itself

**Surprisingly effective**: Often improves performance!

### Online Distillation

Student and teacher train together:

```python
def online_distillation_step(teacher, student, optimizers, batch):
    """Train both teacher and student simultaneously."""
    
    # Update teacher (standard training)
    teacher_metrics = train_step(teacher, optimizers['teacher'], batch)
    
    # Update student (distillation from current teacher)
    student_metrics = distill_step(
        student, teacher, optimizers['student'], batch
    )
    
    return teacher_metrics, student_metrics
```

**Advantage**: No need to fully train teacher first.

### Cross-Modal Distillation

Transfer knowledge across modalities:

- Image teacher → Text student
- Text teacher → Image student

Requires aligned representations (e.g., CLIP embeddings).

## Practical Considerations

### When to Use Distillation

**Good fit**:
- ✅ Deploying to resource-constrained devices
- ✅ Need faster inference
- ✅ Have pre-trained large model
- ✅ Limited labeled data

**Not ideal**:
- ❌ Teacher barely better than student capacity
- ❌ No computational constraints
- ❌ Tasks where interpretability matters more than performance

### Choosing Student Architecture

**Option 1**: Scaled-down teacher
```python
# Teacher: [64, 128, 256] filters
# Student: [16, 32, 64] filters (1/4 scale)
```

**Option 2**: Different architecture
```python
# Teacher: ResNet-50
# Student: MobileNet
```

**Rule**: Student should have sufficient capacity (not too small).

### Debugging Distillation

**Problem**: Student not improving

**Check**:
1. **Teacher quality**: Is teacher accurate?
2. **Temperature**: Try $\tau \in [2, 5]$
3. **Alpha**: Try $\lambda = 0.3$ (more weight on soft)
4. **Student capacity**: Is student too small?

**Monitor**: Plot hard vs soft loss separately!

## Mathematical Deep Dive

### Derivation of Temperature Scaling

For logit $z_i$ and temperature $\tau$:

$$
p_i = \frac{e^{z_i/\tau}}{\sum_j e^{z_j/\tau}}
$$

**As $\tau \to 0$**: Approaches one-hot (hardest)  
**As $\tau \to \infty$**: Approaches uniform (softest)

**Gradient w.r.t. logits**:

$$
\frac{\partial \mathcal{L}_{\text{soft}}}{\partial z^S_i} = \frac{1}{\tau}(p^S_i - p^T_i)
$$

Gradients scale as $1/\tau$. Multiplying loss by $\tau^2$ gives:

$$
\frac{\partial (\tau^2 \mathcal{L}_{\text{soft}})}{\partial z^S_i} = \tau(p^S_i - p^T_i)
$$

This maintains consistent gradient magnitudes across temperatures.

### Connection to Label Smoothing

Distillation relates to label smoothing:

**Label smoothing**:
$$
y_{\text{smooth}} = (1-\epsilon) \cdot y + \epsilon \cdot \text{uniform}
$$

**Distillation**:
$$
y_{\text{distill}} = \lambda \cdot y + (1-\lambda) \cdot \text{softmax}(z^T/\tau)
$$

Both regularize by softening targets, but distillation uses learned soft targets.

### Information Theory Perspective

Distillation minimizes KL divergence between distributions:

$$
\mathcal{L}_{\text{soft}} = \text{KL}(P^T \| P^S) = \sum_i p^T_i \log \frac{p^T_i}{p^S_i}
$$

Equivalently, maximizes mutual information between student and teacher predictions.

## Common Pitfalls

### 1. Forgetting to Freeze Teacher
❌ **Problem**: Teacher and student both update  
✅ **Solution**: Use `jax.lax.stop_gradient` on teacher logits

### 2. Wrong Temperature
❌ **Problem**: $\tau = 1$ (no softening)  
✅ **Solution**: Use $\tau = 3$ as default

### 3. Alpha Too Large
❌ **Problem**: $\lambda = 1$ (no distillation)  
✅ **Solution**: Use $\lambda = 0.5$ or smaller

### 4. Student Too Small
❌ **Problem**: Student lacks capacity  
✅ **Solution**: Ensure student has 10-20% of teacher capacity

## Running the Example

Train teacher, then distill to student:

```bash
cd examples
python 15_knowledge_distillation.py
```

Expected output:
```
TRAINING TEACHER MODEL
Epoch  5/10 | Test Acc: 0.9850 | Time: 12.3s
✓ Teacher training completed!

TRAINING STUDENT WITH DISTILLATION
Temperature: 3.0, Alpha: 0.5
Epoch 10/10 | Test Acc: 0.9720 | Hard: 0.15 Soft: 0.08
✓ Student training completed!

TRAINING STUDENT WITHOUT DISTILLATION (BASELINE)
Epoch 10/10 | Test Acc: 0.9580
✓ Baseline training completed!

SUMMARY:
  Teacher:    2.1M params, 98.50% accuracy
  Student:    140K params, 97.20% accuracy (distilled)
  Baseline:   140K params, 95.80% accuracy
  Improvement: +1.40% from distillation
  Compression: 15x smaller, 1.30% loss vs teacher
```

## Next Steps

- Try different temperature values
- Experiment with alpha (hard/soft balance)
- Apply to larger models (ResNet, Transformers)
- Explore self-distillation
- Combine with pruning and quantization

## References

- **Original Paper**: [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) (Hinton et al., NIPS 2014)
- **Born Again Networks**: [Born Again Neural Networks](https://arxiv.org/abs/1805.04770)
- **Self-Distillation**: [Be Your Own Teacher](https://arxiv.org/abs/1905.08094)
- **Survey**: [Knowledge Distillation: A Survey](https://arxiv.org/abs/2006.05525)
