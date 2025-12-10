"""
Flax NNX: Basic Model Definition
=================================
This guide shows how to define models using Flax NNX, demonstrating how to use
shared components from the examples library.

Run: python basics/model_definition.py
"""

import jax
import jax.numpy as jnp
from flax import nnx
import sys
from pathlib import Path

# Add parent directory to path for shared imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.models import MLP, CNN


# ============================================================================
# 1. SIMPLE LINEAR MODEL
# ============================================================================

class SimpleLinear(nnx.Module):
    """A simple linear layer model."""
    
    def __init__(self, in_features: int, out_features: int, rngs: nnx.Rngs):
        # NNX uses explicit RNG handling
        self.linear = nnx.Linear(in_features, out_features, rngs=rngs)
    
    def __call__(self, x):
        return self.linear(x)


def demo_simple_linear():
    """Demonstrate simple linear model."""
    print("=" * 70)
    print("1. SIMPLE LINEAR MODEL")
    print("=" * 70)
    
    # Create model
    rngs = nnx.Rngs(0)
    model = SimpleLinear(in_features=10, out_features=5, rngs=rngs)
    
    # Test forward pass
    x = jnp.ones((4, 10))
    output = model(x)
    
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"✓ Simple linear model works!")
    print()


# ============================================================================
# 2. MULTI-LAYER PERCEPTRON (MLP) - Using Shared Component
# ============================================================================

def demo_mlp():
    """Demonstrate MLP from shared components."""
    print("=" * 70)
    print("2. MULTI-LAYER PERCEPTRON (MLP) - Shared Component")
    print("=" * 70)
    
    # Create MLP using shared component
    rngs = nnx.Rngs(0)
    model = MLP(
        in_features=784,        # MNIST flattened
        hidden_features=128,
        out_features=10,        # 10 classes
        n_layers=3,
        rngs=rngs
    )
    
    # Test forward pass
    x = jnp.ones((32, 784))
    output = model(x)
    
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Count parameters
    params = nnx.state(model, nnx.Param)
    total_params = sum(p.size for p in jax.tree.leaves(params))
    print(f"Total parameters: {total_params:,}")
    print(f"✓ MLP works! (from shared.models)")
    print()


# ============================================================================
# 3. CONVOLUTIONAL NEURAL NETWORK (CNN) - Using Shared Component
# ============================================================================

def demo_cnn():
    """Demonstrate CNN from shared components."""
    print("=" * 70)
    print("3. CONVOLUTIONAL NEURAL NETWORK (CNN) - Shared Component")
    print("=" * 70)
    
    # Create CNN using shared component
    rngs = nnx.Rngs(0)
    model = CNN(num_classes=10, rngs=rngs)
    
    # Test forward pass (MNIST-like input)
    x = jnp.ones((16, 28, 28, 1))
    output = model(x, train=False)
    
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Count parameters
    params = nnx.state(model, nnx.Param)
    total_params = sum(p.size for p in jax.tree.leaves(params))
    print(f"Total parameters: {total_params:,}")
    print(f"✓ CNN works! (from shared.models)")
    print()


# ============================================================================
# 4. MODEL INSPECTION
# ============================================================================

def demo_model_inspection():
    """Demonstrate model inspection capabilities."""
    print("=" * 70)
    print("4. MODEL INSPECTION")
    print("=" * 70)
    
    rngs = nnx.Rngs(0)
    model = MLP(in_features=10, hidden_features=20, out_features=5, n_layers=2, rngs=rngs)
    
    # Get model state
    state = nnx.state(model)
    print("Model state structure:")
    print(jax.tree.map(lambda x: f"  {x.shape} {x.dtype}", state))
    
    # Get only parameters
    params = nnx.state(model, nnx.Param)
    num_params = sum(p.size for p in jax.tree.leaves(params))
    print(f"\nTotal trainable parameters: {num_params:,}")
    
    # Get graph node (structure)
    graphdef, state = nnx.split(model)
    print(f"\nGraphDef type: {type(graphdef)}")
    print(f"✓ Model inspection complete!")
    print()


# ============================================================================
# MAIN DEMO
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("FLAX NNX: BASIC MODEL DEFINITION EXAMPLES")
    print("Using Shared Components from examples/shared/")
    print("=" * 70 + "\n")
    
    demo_simple_linear()
    demo_mlp()
    demo_cnn()
    demo_model_inspection()
    
    print("=" * 70)
    print("✓ All demos completed successfully!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("- SimpleLinear: Basic single-layer model")
    print("- MLP: Multi-layer perceptron with ReLU activations")
    print("- CNN: Convolutional network for image data")
    print("- All models use explicit RNG handling with nnx.Rngs")
    print("- Shared components in shared.models can be reused across examples")
    print()
