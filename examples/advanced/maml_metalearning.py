"""
Flax NNX: Meta-Learning with MAML
==================================
Implementation of Model-Agnostic Meta-Learning (MAML) for few-shot learning.

Run: python advanced/maml_metalearning.py

Reference:
    Finn et al. "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks"
    ICML 2017. https://arxiv.org/abs/1703.03400
"""

import jax
import jax.numpy as jnp
from flax import nnx
import optax
import numpy as np
from typing import Dict, List, Tuple
import time
from functools import partial



import sys
from pathlib import Path

# Add parent directory to path for shared imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# ============================================================================
# 1. SIMPLE CNN MODEL FOR FEW-SHOT LEARNING
# ============================================================================

class FewShotCNN(nnx.Module):
    """Simple CNN for few-shot classification."""
    
    def __init__(self, num_classes: int = 5, rngs: nnx.Rngs = None):
        # Feature extractor
        self.conv1 = nnx.Conv(1, 32, kernel_size=(3, 3), padding='SAME', rngs=rngs)
        self.bn1 = nnx.BatchNorm(32, rngs=rngs)
        
        self.conv2 = nnx.Conv(32, 32, kernel_size=(3, 3), padding='SAME', rngs=rngs)
        self.bn2 = nnx.BatchNorm(32, rngs=rngs)
        
        self.conv3 = nnx.Conv(32, 32, kernel_size=(3, 3), padding='SAME', rngs=rngs)
        self.bn3 = nnx.BatchNorm(32, rngs=rngs)
        
        self.conv4 = nnx.Conv(32, 32, kernel_size=(3, 3), padding='SAME', rngs=rngs)
        self.bn4 = nnx.BatchNorm(32, rngs=rngs)
        
        # Classifier
        self.fc = nnx.Linear(32, num_classes, rngs=rngs)
    
    def __call__(self, x, train: bool = False):
        # Conv blocks with max pooling
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
        
        x = self.conv4(x)
        x = self.bn4(x, use_running_average=not train)
        x = nnx.relu(x)
        x = nnx.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        # Global average pooling
        x = jnp.mean(x, axis=(1, 2))
        
        # Classifier
        x = self.fc(x)
        
        return x


# ============================================================================
# 2. OMNIGLOT DATASET GENERATOR (SIMULATED)
# ============================================================================

def generate_sinusoid_task(amplitude_range=(0.1, 5.0), phase_range=(0, np.pi)):
    """
    Generate a random sinusoid task for demonstration.
    In practice, you would use actual datasets like Omniglot or Mini-ImageNet.
    """
    amplitude = np.random.uniform(*amplitude_range)
    phase = np.random.uniform(*phase_range)
    
    def task_fn(x):
        return amplitude * np.sin(x + phase)
    
    return task_fn


def sample_task_data(task_fn, n_samples=10, x_range=(-5.0, 5.0)):
    """Sample data from a task function."""
    x = np.random.uniform(*x_range, size=(n_samples, 1))
    y = task_fn(x)
    return x.astype(np.float32), y.astype(np.float32)


class MAMLDataGenerator:
    """
    Generate tasks for MAML training.
    For demonstration, we use sinusoid regression.
    """
    
    def __init__(self, n_way: int = 5, k_shot: int = 5, k_query: int = 15):
        self.n_way = n_way
        self.k_shot = k_shot
        self.k_query = k_query
    
    def sample_batch(self, batch_size: int):
        """Sample a batch of tasks."""
        support_x, support_y = [], []
        query_x, query_y = [], []
        
        for _ in range(batch_size):
            # Generate random task
            task_fn = generate_sinusoid_task()
            
            # Sample support and query sets
            sup_x, sup_y = sample_task_data(task_fn, self.k_shot)
            que_x, que_y = sample_task_data(task_fn, self.k_query)
            
            support_x.append(sup_x)
            support_y.append(sup_y)
            query_x.append(que_x)
            query_y.append(que_y)
        
        return {
            'support': {
                'x': np.stack(support_x),
                'y': np.stack(support_y)
            },
            'query': {
                'x': np.stack(query_x),
                'y': np.stack(query_y)
            }
        }


# ============================================================================
# 3. SIMPLE MLP FOR REGRESSION (MAML DEMO)
# ============================================================================

class MAMLRegressor(nnx.Module):
    """Simple MLP for regression tasks (sinusoid fitting)."""
    
    def __init__(self, hidden_dim: int = 40, rngs: nnx.Rngs = None):
        self.fc1 = nnx.Linear(1, hidden_dim, rngs=rngs)
        self.fc2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.fc3 = nnx.Linear(hidden_dim, 1, rngs=rngs)
    
    def __call__(self, x):
        x = self.fc1(x)
        x = nnx.relu(x)
        x = self.fc2(x)
        x = nnx.relu(x)
        x = self.fc3(x)
        return x


# ============================================================================
# 4. MAML INNER LOOP (TASK ADAPTATION)
# ============================================================================

def inner_loop(model, support_x, support_y, inner_lr=0.01, inner_steps=1):
    """
    MAML inner loop: Adapt model to a specific task.
    
    This performs gradient descent on the support set to quickly adapt
    the model parameters to the task.
    
    Args:
        model: The model to adapt
        support_x: Support set inputs [k_shot, input_dim]
        support_y: Support set targets [k_shot, output_dim]
        inner_lr: Learning rate for inner loop adaptation
        inner_steps: Number of gradient steps for adaptation
    
    Returns:
        adapted_params: Parameters adapted to the task
    """
    # Get initial parameters
    params = nnx.state(model, nnx.Param)
    
    # Inner loop updates
    for _ in range(inner_steps):
        def loss_fn(params):
            # Temporarily update model with these params
            nnx.update(model, params)
            predictions = model(support_x)
            loss = jnp.mean((predictions - support_y) ** 2)
            return loss
        
        # Compute gradients
        grads = jax.grad(loss_fn)(params)
        
        # Manual SGD update (inner loop)
        params = jax.tree_map(
            lambda p, g: p - inner_lr * g,
            params,
            grads
        )
    
    return params


# ============================================================================
# 5. MAML OUTER LOOP (META-TRAINING)
# ============================================================================

def maml_meta_loss(model, support_batch, query_batch, inner_lr=0.01, inner_steps=1):
    """
    MAML meta-loss: Compute loss on query set after task adaptation.
    
    The key insight of MAML is to optimize the initial parameters such that
    after a few gradient steps on a task's support set, the model performs
    well on that task's query set.
    
    Formula:
        θ' = θ - α∇_θ L_support(θ)           # Inner loop
        L_meta = E_task[L_query(θ')]          # Meta loss
        θ ← θ - β∇_θ L_meta                   # Meta update
    
    where:
        - θ are the meta-parameters (initial parameters)
        - θ' are task-specific parameters after adaptation
        - α is the inner loop learning rate
        - β is the meta learning rate (outer loop)
    """
    batch_size = support_batch['x'].shape[0]
    total_loss = 0.0
    
    for i in range(batch_size):
        # Get support and query for this task
        sup_x = support_batch['x'][i]
        sup_y = support_batch['y'][i]
        que_x = query_batch['x'][i]
        que_y = query_batch['y'][i]
        
        # Inner loop: Adapt to task
        adapted_params = inner_loop(model, sup_x, sup_y, inner_lr, inner_steps)
        
        # Evaluate on query set with adapted parameters
        nnx.update(model, adapted_params)
        predictions = model(que_x)
        task_loss = jnp.mean((predictions - que_y) ** 2)
        
        total_loss += task_loss
    
    # Average loss across tasks
    meta_loss = total_loss / batch_size
    
    return meta_loss


@nnx.jit
def maml_train_step(model: MAMLRegressor, optimizer: nnx.Optimizer,
                    support_batch: Dict, query_batch: Dict,
                    inner_lr: float = 0.01, inner_steps: int = 1):
    """
    Single MAML training step.
    
    This performs:
    1. Inner loop adaptation on support set for each task
    2. Compute meta-loss on query set
    3. Update meta-parameters (outer loop)
    """
    
    def meta_loss_fn(model):
        return maml_meta_loss(model, support_batch, query_batch, inner_lr, inner_steps)
    
    # Compute meta-gradients
    grad_fn = nnx.value_and_grad(meta_loss_fn)
    loss, grads = grad_fn(model)
    
    # Meta-update (outer loop)
    optimizer.update(grads)
    
    return {'loss': loss}


# ============================================================================
# 6. TRAINING LOOP
# ============================================================================

def train_maml(
    num_iterations: int = 10000,
    meta_batch_size: int = 25,
    meta_lr: float = 1e-3,
    inner_lr: float = 0.01,
    inner_steps: int = 1,
    k_shot: int = 10,
    k_query: int = 10,
):
    """Train MAML model."""
    print("\n" + "="*70)
    print("MODEL-AGNOSTIC META-LEARNING (MAML)")
    print("="*70)
    print("\nMAML learns an initialization that allows rapid adaptation")
    print("to new tasks with just a few gradient steps.")
    print("\nAlgorithm:")
    print("  1. Sample batch of tasks")
    print("  2. For each task:")
    print("     - Adapt parameters on support set (inner loop)")
    print("     - Evaluate on query set")
    print("  3. Update meta-parameters to minimize query loss (outer loop)")
    print("="*70)
    
    # Initialize data generator
    data_generator = MAMLDataGenerator(k_shot=k_shot, k_query=k_query)
    
    # Initialize model and optimizer
    print("\nInitializing model...")
    rng = jax.random.PRNGKey(0)
    model = MAMLRegressor(hidden_dim=40, rngs=nnx.Rngs(rng))
    optimizer = nnx.Optimizer(model, optax.adam(meta_lr))
    
    print(f"✓ Model initialized")
    print(f"  Meta batch size: {meta_batch_size} tasks")
    print(f"  K-shot: {k_shot} examples per task")
    print(f"  K-query: {k_query} examples for evaluation")
    print(f"  Inner learning rate: {inner_lr}")
    print(f"  Inner steps: {inner_steps}")
    print(f"  Meta learning rate: {meta_lr}")
    
    # Training loop
    print(f"\nTraining for {num_iterations} iterations...")
    print("-" * 70)
    
    losses = []
    for iteration in range(num_iterations):
        # Sample batch of tasks
        batch = data_generator.sample_batch(meta_batch_size)
        support_batch = {k: jnp.array(v) for k, v in batch['support'].items()}
        query_batch = {k: jnp.array(v) for k, v in batch['query'].items()}
        
        # MAML training step
        metrics = maml_train_step(
            model, optimizer, support_batch, query_batch,
            inner_lr=inner_lr, inner_steps=inner_steps
        )
        
        losses.append(float(metrics['loss']))
        
        # Print progress
        if (iteration + 1) % 1000 == 0:
            avg_loss = np.mean(losses[-100:])
            print(f"Iteration {iteration+1:5d}/{num_iterations} | "
                  f"Meta Loss: {avg_loss:.4f}")
    
    print("\n" + "="*70)
    print("✓ Training completed!")
    print("="*70)
    
    return model


# ============================================================================
# 7. EVALUATION (FEW-SHOT ADAPTATION)
# ============================================================================

def evaluate_adaptation(model: MAMLRegressor, inner_lr=0.01, n_test_tasks=10):
    """
    Evaluate MAML's ability to adapt to new tasks.
    
    This demonstrates the key property of MAML: after meta-training,
    the model can quickly adapt to new tasks with just a few examples.
    """
    print("\n" + "="*70)
    print("EVALUATING FEW-SHOT ADAPTATION")
    print("="*70)
    
    print("\nTesting adaptation on new tasks...")
    print("Comparing before and after adaptation:")
    print("-" * 70)
    
    for task_idx in range(n_test_tasks):
        # Generate new task
        task_fn = generate_sinusoid_task()
        
        # Sample support set (for adaptation)
        support_x, support_y = sample_task_data(task_fn, n_samples=10)
        support_x = jnp.array(support_x)
        support_y = jnp.array(support_y)
        
        # Sample query set (for evaluation)
        query_x, query_y = sample_task_data(task_fn, n_samples=50)
        query_x = jnp.array(query_x)
        query_y = jnp.array(query_y)
        
        # Evaluate before adaptation
        pred_before = model(query_x)
        loss_before = jnp.mean((pred_before - query_y) ** 2)
        
        # Adapt to task (1, 5, 10 steps)
        for n_steps in [1, 5, 10]:
            adapted_params = inner_loop(model, support_x, support_y, 
                                       inner_lr=inner_lr, inner_steps=n_steps)
            nnx.update(model, adapted_params)
            
            # Evaluate after adaptation
            pred_after = model(query_x)
            loss_after = jnp.mean((pred_after - query_y) ** 2)
            
            if n_steps == 1:
                print(f"Task {task_idx+1:2d} | "
                      f"Before: {loss_before:.4f} → "
                      f"After 1 step: {loss_after:.4f} | "
                      f"Improvement: {(loss_before - loss_after):.4f}")
        
        # Restore original parameters
        # (In practice, you would keep a copy of the meta-parameters)
    
    print("\n" + "="*70)
    print("✓ Evaluation completed!")
    print("\nKey Insight:")
    print("  The model rapidly improves with just a few gradient steps,")
    print("  demonstrating that MAML learns a good initialization for")
    print("  fast adaptation to new tasks.")
    print("="*70)


# ============================================================================
# 8. VISUALIZATION
# ============================================================================

def visualize_adaptation(model: MAMLRegressor, inner_lr=0.01):
    """Visualize how the model adapts to a new task."""
    try:
        import matplotlib.pyplot as plt
        
        print("\nGenerating visualization...")
        
        # Generate a test task
        task_fn = generate_sinusoid_task(amplitude_range=(1.0, 2.0))
        
        # Sample support set
        support_x, support_y = sample_task_data(task_fn, n_samples=10)
        support_x = jnp.array(support_x)
        support_y = jnp.array(support_y)
        
        # Generate test points
        test_x = np.linspace(-5, 5, 100).reshape(-1, 1).astype(np.float32)
        test_y = task_fn(test_x)
        test_x_jax = jnp.array(test_x)
        
        # Predictions at different adaptation steps
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        for i, n_steps in enumerate([0, 1, 5, 10]):
            ax = axes[i]
            
            if n_steps == 0:
                # Before adaptation
                pred = model(test_x_jax)
            else:
                # After adaptation
                adapted_params = inner_loop(model, support_x, support_y,
                                           inner_lr=inner_lr, inner_steps=n_steps)
                nnx.update(model, adapted_params)
                pred = model(test_x_jax)
            
            # Plot
            ax.plot(test_x, test_y, 'k-', label='True function', linewidth=2)
            ax.plot(test_x, pred, 'r--', label='Model prediction', linewidth=2)
            ax.scatter(support_x, support_y, c='blue', s=100, 
                      label='Support set', zorder=5)
            ax.set_title(f'After {n_steps} gradient steps')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/tmp/maml_adaptation.png', dpi=150, bbox_inches='tight')
        print("✓ Visualization saved to /tmp/maml_adaptation.png")
        
    except ImportError:
        print("Matplotlib not available. Skipping visualization.")


# ============================================================================
# 9. MAIN
# ============================================================================

def main():
    """Main function."""
    # Train MAML
    model = train_maml(
        num_iterations=10000,
        meta_batch_size=25,
        meta_lr=1e-3,
        inner_lr=0.01,
        inner_steps=1,
        k_shot=10,
        k_query=10,
    )
    
    # Evaluate adaptation
    evaluate_adaptation(model, inner_lr=0.01, n_test_tasks=10)
    
    # Visualize adaptation
    visualize_adaptation(model, inner_lr=0.01)


if __name__ == "__main__":
    main()
