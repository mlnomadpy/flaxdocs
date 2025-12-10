"""
Flax NNX: Saving and Loading Models
====================================
This guide shows how to save and load Flax NNX models.
Run: python basics/save_load_model.py
"""

import jax
import jax.numpy as jnp
from flax import nnx
import orbax.checkpoint as ocp
from pathlib import Path
import tempfile



import sys
from pathlib import Path

# Add parent directory to path for shared imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# ============================================================================
# DEFINE A SIMPLE MODEL
# ============================================================================

class SimpleModel(nnx.Module):
    """Simple model for demonstration."""
    
    def __init__(self, in_features: int, out_features: int, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(in_features, 128, rngs=rngs)
        self.linear2 = nnx.Linear(128, out_features, rngs=rngs)
        self.dropout = nnx.Dropout(rate=0.5, rngs=rngs)
        self.bn = nnx.BatchNorm(128, rngs=rngs)
    
    def __call__(self, x, train: bool = False):
        x = self.linear1(x)
        x = self.bn(x, use_running_average=not train)
        x = nnx.relu(x)
        x = self.dropout(x, deterministic=not train)
        x = self.linear2(x)
        return x


# ============================================================================
# METHOD 1: SAVE/LOAD USING NNX STATE
# ============================================================================

def save_model_nnx(model: nnx.Module, path: str):
    """Save model using NNX state abstraction."""
    print(f"\nSaving model to: {path}")
    
    # Extract model state (parameters, batch stats, etc.)
    state = nnx.state(model)
    
    # Create checkpointer
    checkpointer = ocp.PyTreeCheckpointer()
    
    # Save
    checkpointer.save(path, state)
    print(f"✓ Model saved successfully")


def load_model_nnx(model: nnx.Module, path: str):
    """Load model using NNX state abstraction."""
    print(f"\nLoading model from: {path}")
    
    # Create checkpointer
    checkpointer = ocp.PyTreeCheckpointer()
    
    # Load state
    restored_state = checkpointer.restore(path)
    
    # Update model with restored state
    nnx.update(model, restored_state)
    print(f"✓ Model loaded successfully")
    
    return model


# ============================================================================
# METHOD 2: SAVE/LOAD WITH CHECKPOINT MANAGER
# ============================================================================

def save_with_checkpoint_manager(model: nnx.Module, step: int, checkpoint_dir: str):
    """Save model using Orbax CheckpointManager for versioning."""
    print(f"\nSaving checkpoint at step {step} to: {checkpoint_dir}")
    
    # Get model state
    state = nnx.state(model)
    
    # Create checkpoint manager
    options = ocp.CheckpointManagerOptions(
        max_to_keep=3,  # Keep last 3 checkpoints
        create=True
    )
    
    checkpoint_manager = ocp.CheckpointManager(
        checkpoint_dir,
        checkpointers=ocp.PyTreeCheckpointer(),
        options=options
    )
    
    # Save checkpoint
    checkpoint_manager.save(
        step,
        args=ocp.args.PyTreeSave(state)
    )
    
    print(f"✓ Checkpoint saved at step {step}")
    return checkpoint_manager


def load_with_checkpoint_manager(model: nnx.Module, checkpoint_dir: str, 
                                  step: int = None):
    """Load model from checkpoint manager."""
    print(f"\nLoading checkpoint from: {checkpoint_dir}")
    
    # Create checkpoint manager
    checkpoint_manager = ocp.CheckpointManager(
        checkpoint_dir,
        checkpointers=ocp.PyTreeCheckpointer()
    )
    
    # Get latest step if not specified
    if step is None:
        step = checkpoint_manager.latest_step()
        if step is None:
            raise ValueError("No checkpoints found")
        print(f"Loading latest checkpoint at step {step}")
    
    # Create abstract state for restoration
    abstract_state = nnx.state(model)
    
    # Restore
    restored_state = checkpoint_manager.restore(
        step,
        args=ocp.args.PyTreeRestore(abstract_state)
    )
    
    # Update model
    nnx.update(model, restored_state)
    print(f"✓ Checkpoint loaded from step {step}")
    
    return model, step


# ============================================================================
# METHOD 3: SAVE/LOAD ONLY PARAMETERS (COMPACT)
# ============================================================================

def save_parameters_only(model: nnx.Module, path: str):
    """Save only trainable parameters (most compact)."""
    print(f"\nSaving parameters only to: {path}")
    
    # Get only parameters (excluding batch stats, etc.)
    state = nnx.state(model)
    
    # Filter for parameters only
    params = {}
    for key, value in state.items():
        if 'kernel' in str(key) or 'bias' in str(key) or 'scale' in str(key):
            params[key] = value
    
    checkpointer = ocp.PyTreeCheckpointer()
    checkpointer.save(path, params)
    print(f"✓ Parameters saved successfully")


def load_parameters_only(model: nnx.Module, path: str):
    """Load only parameters."""
    print(f"\nLoading parameters from: {path}")
    
    checkpointer = ocp.PyTreeCheckpointer()
    params = checkpointer.restore(path)
    
    # Update model with parameters
    nnx.update(model, params)
    print(f"✓ Parameters loaded successfully")
    
    return model


# ============================================================================
# METHOD 4: SAVE/LOAD WITH METADATA
# ============================================================================

def save_with_metadata(model: nnx.Module, path: str, metadata: dict):
    """Save model with additional metadata."""
    print(f"\nSaving model with metadata to: {path}")
    
    state = nnx.state(model)
    
    # Combine state and metadata
    checkpoint = {
        'model_state': state,
        'metadata': metadata
    }
    
    checkpointer = ocp.PyTreeCheckpointer()
    checkpointer.save(path, checkpoint)
    print(f"✓ Model and metadata saved")
    print(f"  Metadata: {metadata}")


def load_with_metadata(model: nnx.Module, path: str):
    """Load model and metadata."""
    print(f"\nLoading model with metadata from: {path}")
    
    checkpointer = ocp.PyTreeCheckpointer()
    checkpoint = checkpointer.restore(path)
    
    # Extract and update model
    nnx.update(model, checkpoint['model_state'])
    metadata = checkpoint['metadata']
    
    print(f"✓ Model and metadata loaded")
    print(f"  Metadata: {metadata}")
    
    return model, metadata


# ============================================================================
# DEMONSTRATION
# ============================================================================

def main():
    print("=" * 80)
    print("Flax NNX Model Saving and Loading Examples")
    print("=" * 80)
    
    # Create temporary directory for checkpoints
    temp_dir = tempfile.mkdtemp()
    print(f"\nUsing temporary directory: {temp_dir}")
    
    # ========================================================================
    # Initialize model
    # ========================================================================
    print("\n" + "=" * 80)
    print("Initializing Model")
    print("=" * 80)
    
    rngs = nnx.Rngs(42)
    model = SimpleModel(in_features=10, out_features=5, rngs=rngs)
    
    # Create some dummy data
    x = jnp.ones((4, 10))
    original_output = model(x, train=False)
    
    print(f"Original output:\n{original_output}")
    print(f"Original first layer weights sum: {jnp.sum(model.linear1.kernel.value):.4f}")
    
    # ========================================================================
    # Test 1: Basic Save/Load with NNX State
    # ========================================================================
    print("\n" + "=" * 80)
    print("Test 1: Basic Save/Load")
    print("=" * 80)
    
    save_path_1 = Path(temp_dir) / "model_basic"
    save_model_nnx(model, str(save_path_1))
    
    # Create new model and load
    model_loaded = SimpleModel(in_features=10, out_features=5, rngs=rngs)
    load_model_nnx(model_loaded, str(save_path_1))
    
    # Verify
    loaded_output = model_loaded(x, train=False)
    print(f"\nLoaded output:\n{loaded_output}")
    print(f"Loaded first layer weights sum: {jnp.sum(model_loaded.linear1.kernel.value):.4f}")
    print(f"Outputs match: {jnp.allclose(original_output, loaded_output)}")
    
    # ========================================================================
    # Test 2: Checkpoint Manager with Versioning
    # ========================================================================
    print("\n" + "=" * 80)
    print("Test 2: Checkpoint Manager with Versioning")
    print("=" * 80)
    
    checkpoint_dir = Path(temp_dir) / "checkpoints"
    
    # Simulate training with multiple checkpoints
    for step in [100, 200, 300, 400]:
        # Modify model slightly (simulate training)
        model.linear1.kernel.value = model.linear1.kernel.value * 1.01
        save_with_checkpoint_manager(model, step, str(checkpoint_dir))
    
    # Load latest checkpoint
    model_from_manager = SimpleModel(in_features=10, out_features=5, rngs=rngs)
    model_from_manager, loaded_step = load_with_checkpoint_manager(
        model_from_manager, str(checkpoint_dir)
    )
    
    print(f"\nLoaded from step: {loaded_step}")
    
    # Load specific checkpoint
    model_specific = SimpleModel(in_features=10, out_features=5, rngs=rngs)
    model_specific, _ = load_with_checkpoint_manager(
        model_specific, str(checkpoint_dir), step=200
    )
    print(f"Loaded specific checkpoint from step 200")
    
    # ========================================================================
    # Test 3: Parameters Only (Compact)
    # ========================================================================
    print("\n" + "=" * 80)
    print("Test 3: Save/Load Parameters Only")
    print("=" * 80)
    
    save_path_3 = Path(temp_dir) / "model_params"
    save_parameters_only(model, str(save_path_3))
    
    model_params = SimpleModel(in_features=10, out_features=5, rngs=rngs)
    load_parameters_only(model_params, str(save_path_3))
    
    print("✓ Parameters loaded successfully")
    
    # ========================================================================
    # Test 4: Save/Load with Metadata
    # ========================================================================
    print("\n" + "=" * 80)
    print("Test 4: Save/Load with Metadata")
    print("=" * 80)
    
    metadata = {
        'model_name': 'SimpleModel',
        'version': '1.0',
        'training_steps': 1000,
        'accuracy': 0.95,
        'hyperparameters': {
            'learning_rate': 0.001,
            'batch_size': 32
        }
    }
    
    save_path_4 = Path(temp_dir) / "model_with_metadata"
    save_with_metadata(model, str(save_path_4), metadata)
    
    model_with_meta = SimpleModel(in_features=10, out_features=5, rngs=rngs)
    model_with_meta, loaded_metadata = load_with_metadata(
        model_with_meta, str(save_path_4)
    )
    
    # ========================================================================
    # Best Practices
    # ========================================================================
    print("\n" + "=" * 80)
    print("Best Practices for Checkpointing")
    print("=" * 80)
    
    print("""
    1. Use CheckpointManager for training:
       - Automatically manages checkpoint versions
       - Handles cleanup of old checkpoints
       - Supports async saving
    
    2. Save complete state during training:
       - Include optimizer state
       - Include training step/epoch
       - Include metrics and metadata
    
    3. For deployment/inference:
       - Save only parameters to reduce size
       - Include model metadata for versioning
    
    4. Checkpoint frequency:
       - Save regularly based on steps/epochs
       - Save best model based on validation metric
       - Keep multiple checkpoints for safety
    
    5. Loading strategies:
       - Always verify model architecture matches
       - Handle missing or extra parameters gracefully
       - Validate outputs after loading
    """)
    
    print("\n" + "=" * 80)
    print("✓ All save/load tests completed successfully!")
    print("=" * 80)
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)
    print(f"\nCleaned up temporary directory: {temp_dir}")


if __name__ == "__main__":
    main()
