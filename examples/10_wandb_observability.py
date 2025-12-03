"""
Flax NNX: Observability with Weights & Biases (W&B)
====================================================
Track experiments, log metrics, and visualize training with W&B.
Run: pip install wandb && python 10_wandb_observability.py
"""

import jax
import jax.numpy as jnp
from flax import nnx
import optax
import numpy as np
from typing import Dict
import time

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    print("Warning: wandb not available. Install with: pip install wandb")
    WANDB_AVAILABLE = False


# ============================================================================
# EXAMPLE MODEL
# ============================================================================

class SimpleCNN(nnx.Module):
    """Simple CNN for demonstration."""
    
    def __init__(self, num_classes: int = 10, rngs: nnx.Rngs = None):
        self.conv1 = nnx.Conv(1, 32, kernel_size=(3, 3), rngs=rngs)
        self.conv2 = nnx.Conv(32, 64, kernel_size=(3, 3), rngs=rngs)
        self.bn1 = nnx.BatchNorm(32, rngs=rngs)
        self.bn2 = nnx.BatchNorm(64, rngs=rngs)
        self.fc1 = nnx.Linear(64 * 5 * 5, 128, rngs=rngs)
        self.fc2 = nnx.Linear(128, num_classes, rngs=rngs)
        self.dropout = nnx.Dropout(rate=0.5, rngs=rngs)
    
    def __call__(self, x, train: bool = False):
        x = self.conv1(x)
        x = self.bn1(x, use_running_average=not train)
        x = nnx.relu(x)
        x = nnx.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        x = self.conv2(x)
        x = self.bn2(x, use_running_average=not train)
        x = nnx.relu(x)
        x = nnx.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        x = x.reshape((x.shape[0], -1))
        x = self.fc1(x)
        x = nnx.relu(x)
        x = self.dropout(x, deterministic=not train)
        x = self.fc2(x)
        
        return x


# ============================================================================
# 1. BASIC W&B INTEGRATION
# ============================================================================

def train_with_wandb_basic(config: Dict):
    """Basic W&B integration example."""
    if not WANDB_AVAILABLE:
        print("W&B not available")
        return
    
    print("\n" + "=" * 80)
    print("1. Basic W&B Integration")
    print("=" * 80)
    
    # Initialize W&B
    run = wandb.init(
        project="flax-nnx-demo",
        name="basic-example",
        config=config,
        mode="offline"  # Use "online" to actually sync to cloud
    )
    
    print(f"W&B Run: {run.name}")
    print(f"W&B URL: {run.url}")
    
    # Create dummy training data
    for step in range(100):
        # Simulate metrics
        loss = 1.0 - (step / 100) + np.random.randn() * 0.1
        accuracy = (step / 100) + np.random.randn() * 0.05
        
        # Log metrics
        wandb.log({
            "train/loss": loss,
            "train/accuracy": accuracy,
            "step": step
        })
        
        if step % 20 == 0:
            print(f"Step {step}: Loss={loss:.4f}, Acc={accuracy:.4f}")
    
    # Finish run
    wandb.finish()
    print("✓ Basic logging complete")


# ============================================================================
# 2. COMPREHENSIVE LOGGING
# ============================================================================

class WandbLogger:
    """Comprehensive W&B logger for Flax NNX."""
    
    def __init__(self, project: str, name: str, config: Dict,
                 log_model: bool = True, mode: str = "offline"):
        self.project = project
        self.name = name
        self.log_model = log_model
        
        # Initialize W&B
        self.run = wandb.init(
            project=project,
            name=name,
            config=config,
            mode=mode
        )
        
        print(f"✓ W&B initialized: {self.run.url}")
    
    def log_metrics(self, metrics: Dict, step: int, prefix: str = ""):
        """Log training/validation metrics."""
        log_dict = {f"{prefix}/{k}": v for k, v in metrics.items()}
        log_dict["step"] = step
        wandb.log(log_dict)
    
    def log_hyperparameters(self, hparams: Dict):
        """Log hyperparameters."""
        wandb.config.update(hparams)
    
    def log_model_info(self, model: nnx.Module):
        """Log model architecture information."""
        state = nnx.state(model)
        total_params = sum(x.size for x in jax.tree_util.tree_leaves(state))
        
        wandb.config.update({
            "model/total_parameters": total_params,
            "model/type": type(model).__name__,
        })
        
        print(f"✓ Model info logged: {total_params:,} parameters")
    
    def log_images(self, images: np.ndarray, predictions: np.ndarray,
                   labels: np.ndarray, step: int, num_images: int = 8):
        """Log images with predictions."""
        images_to_log = []
        
        for i in range(min(num_images, len(images))):
            img = wandb.Image(
                images[i],
                caption=f"True: {labels[i]}, Pred: {predictions[i]}"
            )
            images_to_log.append(img)
        
        wandb.log({"predictions": images_to_log, "step": step})
    
    def log_histogram(self, data: Dict[str, np.ndarray], step: int):
        """Log histograms (e.g., weights, gradients)."""
        for name, values in data.items():
            wandb.log({
                f"histograms/{name}": wandb.Histogram(values),
                "step": step
            })
    
    def log_confusion_matrix(self, predictions: np.ndarray, 
                            labels: np.ndarray, class_names: list, step: int):
        """Log confusion matrix."""
        wandb.log({
            "confusion_matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=labels,
                preds=predictions,
                class_names=class_names
            ),
            "step": step
        })
    
    def log_gradients(self, grads, step: int):
        """Log gradient statistics."""
        grad_norms = {}
        
        for name, grad in grads.items():
            if hasattr(grad, 'value'):
                grad = grad.value
            norm = float(jnp.linalg.norm(jnp.array(grad).flatten()))
            grad_norms[f"gradients/{name}"] = norm
        
        wandb.log({**grad_norms, "step": step})
    
    def watch_model(self, model: nnx.Module, log_freq: int = 100):
        """Watch model (log gradients and parameters)."""
        # Note: W&B watch is primarily for PyTorch
        # For Flax, we manually log what we need
        print("✓ Model watching enabled (manual logging)")
    
    def finish(self):
        """Finish W&B run."""
        wandb.finish()
        print("✓ W&B run finished")


# ============================================================================
# 3. TRAINING WITH COMPREHENSIVE LOGGING
# ============================================================================

def train_with_comprehensive_logging():
    """Train model with comprehensive W&B logging."""
    if not WANDB_AVAILABLE:
        print("W&B not available")
        return
    
    print("\n" + "=" * 80)
    print("2. Comprehensive W&B Logging")
    print("=" * 80)
    
    # Configuration
    config = {
        "model": "SimpleCNN",
        "dataset": "MNIST",
        "batch_size": 128,
        "learning_rate": 1e-3,
        "num_epochs": 5,
        "optimizer": "adam",
        "seed": 42,
    }
    
    # Initialize logger
    logger = WandbLogger(
        project="flax-nnx-comprehensive",
        name="cnn-mnist",
        config=config,
        mode="offline"
    )
    
    # Initialize model
    rngs = nnx.Rngs(config["seed"])
    model = SimpleCNN(num_classes=10, rngs=rngs)
    
    # Log model info
    logger.log_model_info(model)
    
    # Initialize optimizer
    optimizer = nnx.Optimizer(model, optax.adam(config["learning_rate"]))
    
    # Training loop (simulated)
    print("\nTraining...")
    num_steps = 100
    
    for step in range(num_steps):
        # Simulate training metrics
        train_loss = 1.0 - (step / num_steps) + np.random.randn() * 0.1
        train_acc = (step / num_steps) + np.random.randn() * 0.05
        
        # Log training metrics
        logger.log_metrics({
            "loss": train_loss,
            "accuracy": train_acc,
        }, step=step, prefix="train")
        
        # Periodic validation
        if step % 20 == 0:
            val_loss = train_loss * 0.9
            val_acc = train_acc * 1.05
            
            logger.log_metrics({
                "loss": val_loss,
                "accuracy": val_acc,
            }, step=step, prefix="val")
            
            print(f"Step {step}/{num_steps} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f}")
        
        # Log sample images (every 50 steps)
        if step % 50 == 0 and step > 0:
            # Simulate batch
            sample_images = np.random.randn(8, 28, 28, 1)
            sample_labels = np.random.randint(0, 10, 8)
            sample_preds = np.random.randint(0, 10, 8)
            
            logger.log_images(
                sample_images, sample_preds, sample_labels, step
            )
    
    # Log confusion matrix at end
    all_preds = np.random.randint(0, 10, 100)
    all_labels = np.random.randint(0, 10, 100)
    class_names = [str(i) for i in range(10)]
    
    logger.log_confusion_matrix(all_preds, all_labels, class_names, step=num_steps)
    
    # Finish
    logger.finish()
    print("✓ Training complete")


# ============================================================================
# 4. HYPERPARAMETER SWEEP
# ============================================================================

def hyperparameter_sweep():
    """Run hyperparameter sweep with W&B."""
    if not WANDB_AVAILABLE:
        print("W&B not available")
        return
    
    print("\n" + "=" * 80)
    print("3. Hyperparameter Sweep")
    print("=" * 80)
    
    # Define sweep configuration
    sweep_config = {
        'method': 'random',  # or 'grid', 'bayes'
        'metric': {
            'name': 'val/accuracy',
            'goal': 'maximize'
        },
        'parameters': {
            'learning_rate': {
                'distribution': 'log_uniform_values',
                'min': 1e-4,
                'max': 1e-2
            },
            'batch_size': {
                'values': [32, 64, 128]
            },
            'dropout_rate': {
                'distribution': 'uniform',
                'min': 0.1,
                'max': 0.5
            }
        }
    }
    
    print("Sweep configuration:")
    print(f"  Method: {sweep_config['method']}")
    print(f"  Metric: {sweep_config['metric']['name']}")
    print(f"  Parameters: {list(sweep_config['parameters'].keys())}")
    
    def train_with_config():
        """Train with W&B config."""
        run = wandb.init(mode="offline")
        config = wandb.config
        
        print(f"\nTrying configuration:")
        print(f"  Learning rate: {config.learning_rate}")
        print(f"  Batch size: {config.batch_size}")
        print(f"  Dropout rate: {config.dropout_rate}")
        
        # Simulate training
        for step in range(50):
            # Simulate metrics based on hyperparameters
            val_acc = 0.5 + (step / 50) * 0.3
            val_acc += config.learning_rate * 10  # Better with higher LR (demo)
            val_acc += np.random.randn() * 0.05
            
            wandb.log({
                "val/accuracy": val_acc,
                "step": step
            })
        
        wandb.finish()
    
    # Note: In practice, uncomment this to run sweep
    # sweep_id = wandb.sweep(sweep_config, project="flax-nnx-sweep")
    # wandb.agent(sweep_id, train_with_config, count=5)
    
    print("\n✓ Sweep example configured (commented out to avoid actual run)")


# ============================================================================
# 5. WHY W&B IS IMPORTANT
# ============================================================================

def explain_wandb_importance():
    """Explain why observability with W&B matters."""
    print("\n" + "=" * 80)
    print("Why Observability with W&B Matters")
    print("=" * 80)
    
    print("""
    🎯 Key Benefits of Using W&B:
    
    1. EXPERIMENT TRACKING
       • Track all experiments in one place
       • Compare different runs side-by-side
       • Never lose track of what worked
       • Reproducible experiments
    
    2. REAL-TIME MONITORING
       • Watch training progress live
       • Catch issues early (NaN losses, etc.)
       • Monitor GPU/memory usage
       • Get alerts on failures
    
    3. HYPERPARAMETER OPTIMIZATION
       • Automated hyperparameter sweeps
       • Visualize parameter importance
       • Find optimal configurations faster
       • Save time and compute resources
    
    4. COLLABORATION
       • Share results with team
       • Comment on runs
       • Create reports and presentations
       • Version control for ML experiments
    
    5. DEBUGGING
       • Visualize model predictions
       • Track gradient flows
       • Monitor weight distributions
       • Identify training issues quickly
    
    6. PRODUCTION READINESS
       • Track model versions
       • Monitor model performance over time
       • A/B testing different models
       • Audit trail for compliance
    
    📊 What to Log:
    
    Essential:
    • Training/validation loss and metrics
    • Learning rate schedule
    • Model architecture and parameters
    • Hyperparameters and config
    
    Recommended:
    • Sample predictions with images
    • Confusion matrices
    • Gradient norms
    • Weight histograms
    • System metrics (GPU, memory)
    
    Advanced:
    • Attention maps (for transformers)
    • Feature visualizations
    • Model explanations
    • Dataset statistics
    
    💡 Best Practices:
    
    1. Name runs descriptively
    2. Use tags for organization
    3. Log early and often
    4. Create custom charts
    5. Write good run notes
    6. Save model checkpoints
    7. Share results with team
    8. Create reports for stakeholders
    
    🚀 Getting Started:
    
    1. Sign up: https://wandb.ai
    2. Install: pip install wandb
    3. Login: wandb login
    4. Initialize: wandb.init(project="my-project")
    5. Log: wandb.log({"metric": value})
    6. Finish: wandb.finish()
    
    ⚡ Pro Tips:
    
    • Use offline mode for development
    • Enable autoresume for long training
    • Use artifacts for dataset versioning
    • Set up alerts for critical metrics
    • Integrate with Slack/email notifications
    """)


# ============================================================================
# DEMONSTRATION
# ============================================================================

def main():
    print("=" * 80)
    print("Flax NNX: Observability with Weights & Biases")
    print("=" * 80)
    
    if not WANDB_AVAILABLE:
        print("\n" + "!" * 80)
        print("W&B is not installed!")
        print("Install with: pip install wandb")
        print("Sign up at: https://wandb.ai")
        print("!" * 80)
        explain_wandb_importance()
        return
    
    # Example 1: Basic logging
    config = {
        "learning_rate": 1e-3,
        "batch_size": 128,
        "epochs": 10,
    }
    train_with_wandb_basic(config)
    
    # Example 2: Comprehensive logging
    train_with_comprehensive_logging()
    
    # Example 3: Hyperparameter sweep
    hyperparameter_sweep()
    
    # Explain importance
    explain_wandb_importance()
    
    print("\n" + "=" * 80)
    print("✓ All W&B examples completed!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Sign up at https://wandb.ai")
    print("2. Run: wandb login")
    print("3. Change mode='offline' to mode='online'")
    print("4. View your experiments at wandb.ai")


if __name__ == "__main__":
    main()
