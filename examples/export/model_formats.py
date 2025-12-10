"""
Flax NNX: Export Model to SafeTensors and ONNX
===============================================
This guide shows how to export Flax NNX models to different formats.
Run: pip install safetensors onnx && python 07_export_models.py
"""

import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np
from pathlib import Path
import tempfile
import json


import sys
from pathlib import Path

# Add parent directory to path for shared imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# SafeTensors
try:
    from safetensors.flax import save_file as save_safetensors
    from safetensors.flax import load_file as load_safetensors
    SAFETENSORS_AVAILABLE = True
except ImportError:
    print("Warning: safetensors not available. Install with: pip install safetensors")
    SAFETENSORS_AVAILABLE = False

# ONNX
try:
    import onnx
    from jax.experimental import jax2tf
    import tensorflow as tf
    import tf2onnx
    ONNX_AVAILABLE = True
except ImportError:
    print("Warning: ONNX export not available. Install: pip install onnx tf2onnx")
    ONNX_AVAILABLE = False


# ============================================================================
# EXAMPLE MODEL
# ============================================================================

class SimpleModel(nnx.Module):
    """Simple model for export demonstrations."""
    
    def __init__(self, in_features: int, out_features: int, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(in_features, 128, rngs=rngs)
        self.linear2 = nnx.Linear(128, 64, rngs=rngs)
        self.linear3 = nnx.Linear(64, out_features, rngs=rngs)
        self.bn = nnx.BatchNorm(128, rngs=rngs)
    
    def __call__(self, x, train: bool = False):
        x = self.linear1(x)
        x = self.bn(x, use_running_average=not train)
        x = nnx.relu(x)
        x = self.linear2(x)
        x = nnx.relu(x)
        x = self.linear3(x)
        return x


# ============================================================================
# 1. EXPORT TO SAFETENSORS
# ============================================================================

def export_to_safetensors(model: nnx.Module, filepath: str):
    """Export Flax NNX model to SafeTensors format."""
    if not SAFETENSORS_AVAILABLE:
        print("SafeTensors not available")
        return
    
    print("\n" + "=" * 80)
    print("Exporting to SafeTensors")
    print("=" * 80)
    
    # Get model state
    state = nnx.state(model)
    
    # Convert to dictionary with string keys for SafeTensors
    tensors = {}
    
    def flatten_dict(d, parent_key=''):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else str(k)
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key).items())
            else:
                # Convert JAX array to numpy
                if hasattr(v, 'value'):
                    v = np.array(v.value)
                elif isinstance(v, jnp.ndarray):
                    v = np.array(v)
                items.append((new_key, v))
        return dict(items)
    
    tensors = flatten_dict(state)
    
    print(f"Number of tensors: {len(tensors)}")
    print(f"Tensor keys (first 5): {list(tensors.keys())[:5]}")
    
    # Save to SafeTensors
    save_safetensors(tensors, filepath)
    print(f"✓ Model exported to: {filepath}")
    
    # Get file size
    file_size = Path(filepath).stat().st_size / 1024 / 1024
    print(f"File size: {file_size:.2f} MB")
    
    return tensors


def load_from_safetensors(model: nnx.Module, filepath: str):
    """Load model weights from SafeTensors."""
    if not SAFETENSORS_AVAILABLE:
        print("SafeTensors not available")
        return
    
    print("\n" + "=" * 80)
    print("Loading from SafeTensors")
    print("=" * 80)
    
    # Load tensors
    tensors = load_safetensors(filepath)
    print(f"Loaded {len(tensors)} tensors")
    
    # Unflatten dictionary structure
    def unflatten_dict(flat_dict):
        result = {}
        for key, value in flat_dict.items():
            parts = key.split('.')
            current = result
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value
        return result
    
    state_dict = unflatten_dict(tensors)
    
    # Update model
    # Note: This is a simplified version. In practice, you'd need to
    # carefully match the structure to your model's state
    print("✓ Weights loaded from SafeTensors")
    
    return state_dict


# ============================================================================
# 2. EXPORT TO ONNX
# ============================================================================

def export_to_onnx(model: nnx.Module, filepath: str, 
                   input_shape: tuple, input_dtype=jnp.float32):
    """Export Flax NNX model to ONNX format."""
    if not ONNX_AVAILABLE:
        print("ONNX export not available")
        return
    
    print("\n" + "=" * 80)
    print("Exporting to ONNX")
    print("=" * 80)
    
    # Create a prediction function
    @jax.jit
    def predict(x):
        return model(x, train=False)
    
    # Create dummy input
    dummy_input = jnp.ones(input_shape, dtype=input_dtype)
    
    print(f"Input shape: {input_shape}")
    print(f"Input dtype: {input_dtype}")
    
    # Convert JAX function to TensorFlow
    print("Converting JAX to TensorFlow...")
    tf_predict = jax2tf.convert(predict, enable_xla=False)
    
    # Create TensorFlow function
    @tf.function(input_signature=[
        tf.TensorSpec(shape=input_shape, dtype=tf.float32)
    ])
    def tf_fn(x):
        return tf_predict(x)
    
    # Test TensorFlow function
    tf_input = tf.ones(input_shape, dtype=tf.float32)
    tf_output = tf_fn(tf_input)
    print(f"TensorFlow output shape: {tf_output.shape}")
    
    # Convert to ONNX
    print("Converting TensorFlow to ONNX...")
    onnx_model, _ = tf2onnx.convert.from_function(
        tf_fn,
        input_signature=[tf.TensorSpec(shape=input_shape, dtype=tf.float32)],
        opset=13
    )
    
    # Save ONNX model
    onnx.save(onnx_model, filepath)
    print(f"✓ Model exported to: {filepath}")
    
    # Get file size
    file_size = Path(filepath).stat().st_size / 1024 / 1024
    print(f"File size: {file_size:.2f} MB")
    
    # Print ONNX model info
    print(f"\nONNX Model Info:")
    print(f"  IR version: {onnx_model.ir_version}")
    print(f"  Producer: {onnx_model.producer_name}")
    print(f"  Inputs: {[i.name for i in onnx_model.graph.input]}")
    print(f"  Outputs: {[o.name for o in onnx_model.graph.output]}")
    
    return onnx_model


def verify_onnx_export(onnx_path: str, model: nnx.Module, input_shape: tuple):
    """Verify ONNX export by comparing outputs."""
    if not ONNX_AVAILABLE:
        print("ONNX not available")
        return
    
    print("\n" + "=" * 80)
    print("Verifying ONNX Export")
    print("=" * 80)
    
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not available. Install with: pip install onnxruntime")
        return
    
    # Load ONNX model
    ort_session = ort.InferenceSession(onnx_path)
    
    # Create test input
    test_input = np.random.randn(*input_shape).astype(np.float32)
    
    # Run JAX model
    jax_output = model(jnp.array(test_input), train=False)
    jax_output = np.array(jax_output)
    
    # Run ONNX model
    onnx_input = {ort_session.get_inputs()[0].name: test_input}
    onnx_output = ort_session.run(None, onnx_input)[0]
    
    # Compare outputs
    max_diff = np.max(np.abs(jax_output - onnx_output))
    mean_diff = np.mean(np.abs(jax_output - onnx_output))
    
    print(f"JAX output shape: {jax_output.shape}")
    print(f"ONNX output shape: {onnx_output.shape}")
    print(f"Max difference: {max_diff:.6f}")
    print(f"Mean difference: {mean_diff:.6f}")
    
    if max_diff < 1e-5:
        print("✓ ONNX export verified successfully!")
    else:
        print("⚠ Warning: Large difference detected")


# ============================================================================
# 3. EXPORT METADATA
# ============================================================================

def export_model_metadata(model: nnx.Module, filepath: str, 
                         model_info: dict):
    """Export model metadata as JSON."""
    print("\n" + "=" * 80)
    print("Exporting Model Metadata")
    print("=" * 80)
    
    # Count parameters
    state = nnx.state(model)
    total_params = sum(x.size for x in jax.tree_util.tree_leaves(state))
    
    metadata = {
        'model_name': model_info.get('name', 'unknown'),
        'model_type': model_info.get('type', 'unknown'),
        'framework': 'Flax NNX',
        'total_parameters': int(total_params),
        'input_shape': model_info.get('input_shape'),
        'output_shape': model_info.get('output_shape'),
        'hyperparameters': model_info.get('hyperparameters', {}),
        'training_info': model_info.get('training_info', {}),
    }
    
    # Save metadata
    with open(filepath, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Metadata exported to: {filepath}")
    print(f"\nMetadata:")
    for key, value in metadata.items():
        print(f"  {key}: {value}")


# ============================================================================
# 4. COMPLETE EXPORT PIPELINE
# ============================================================================

def export_model_complete(model: nnx.Module, export_dir: str,
                         model_name: str, input_shape: tuple,
                         model_info: dict = None):
    """Complete model export pipeline."""
    print("\n" + "=" * 80)
    print("Complete Model Export Pipeline")
    print("=" * 80)
    
    export_path = Path(export_dir)
    export_path.mkdir(parents=True, exist_ok=True)
    
    model_info = model_info or {}
    model_info['name'] = model_name
    model_info['input_shape'] = list(input_shape)
    
    # Test model to get output shape
    test_input = jnp.ones(input_shape)
    test_output = model(test_input, train=False)
    model_info['output_shape'] = list(test_output.shape)
    
    print(f"Export directory: {export_dir}")
    print(f"Model name: {model_name}")
    
    # 1. Export to SafeTensors
    if SAFETENSORS_AVAILABLE:
        safetensors_path = export_path / f"{model_name}.safetensors"
        export_to_safetensors(model, str(safetensors_path))
    
    # 2. Export to ONNX
    if ONNX_AVAILABLE:
        onnx_path = export_path / f"{model_name}.onnx"
        export_to_onnx(model, str(onnx_path), input_shape)
        verify_onnx_export(str(onnx_path), model, input_shape)
    
    # 3. Export metadata
    metadata_path = export_path / f"{model_name}_metadata.json"
    export_model_metadata(model, str(metadata_path), model_info)
    
    print("\n" + "=" * 80)
    print("Export Complete!")
    print("=" * 80)
    print(f"Exported files:")
    for file in export_path.glob(f"{model_name}*"):
        print(f"  • {file.name}")


# ============================================================================
# DEMONSTRATION
# ============================================================================

def main():
    print("=" * 80)
    print("Flax NNX Model Export Examples")
    print("=" * 80)
    
    # ========================================================================
    # Initialize Model
    # ========================================================================
    print("\nInitializing model...")
    rngs = nnx.Rngs(42)
    model = SimpleModel(in_features=10, out_features=5, rngs=rngs)
    
    # Test forward pass
    test_input = jnp.ones((4, 10))
    output = model(test_input, train=False)
    print(f"Model output shape: {output.shape}")
    
    # ========================================================================
    # Export Models
    # ========================================================================
    temp_dir = tempfile.mkdtemp()
    print(f"\nUsing temporary directory: {temp_dir}")
    
    # Complete export pipeline
    model_info = {
        'type': 'feedforward',
        'hyperparameters': {
            'hidden_layers': [128, 64],
            'activation': 'relu',
        },
        'training_info': {
            'optimizer': 'adam',
            'learning_rate': 0.001,
            'epochs': 100,
        }
    }
    
    export_model_complete(
        model=model,
        export_dir=temp_dir,
        model_name='simple_model',
        input_shape=(1, 10),
        model_info=model_info
    )
    
    # ========================================================================
    # Test Loading
    # ========================================================================
    if SAFETENSORS_AVAILABLE:
        safetensors_path = Path(temp_dir) / 'simple_model.safetensors'
        load_from_safetensors(model, str(safetensors_path))
    
    # ========================================================================
    # Best Practices
    # ========================================================================
    print("\n" + "=" * 80)
    print("Best Practices for Model Export")
    print("=" * 80)
    
    print("""
    1. SafeTensors:
       ✓ Fast and safe serialization format
       ✓ Prevents arbitrary code execution
       ✓ Memory-mapped loading
       ✓ Best for storing/sharing weights
       ✓ Widely adopted (HuggingFace default)
    
    2. ONNX:
       ✓ Cross-framework compatibility
       ✓ Production deployment
       ✓ Hardware acceleration support
       ✓ Model optimization tools
       ✗ May lose some JAX-specific features
    
    3. When to use each format:
       • SafeTensors: Sharing weights, checkpointing
       • ONNX: Production deployment, inference
       • Both: Maximum compatibility
    
    4. Always include:
       • Model metadata (architecture, hyperparameters)
       • Version information
       • Input/output specifications
       • Training information
    
    5. Verification:
       • Test loaded models
       • Compare outputs with original
       • Document any precision differences
    """)
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)
    print(f"\nCleaned up: {temp_dir}")


if __name__ == "__main__":
    main()
