"""
Integration tests for basic model definition examples.

Tests that the refactored examples run correctly.
"""

import pytest
import jax
import jax.numpy as jnp
from flax import nnx
import sys
from pathlib import Path

# Add examples directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.models import MLP, CNN


@pytest.mark.integration
class TestModelDefinitionExample:
    """Integration tests for model definition example."""
    
    def test_simple_linear(self):
        """Test SimpleLinear model works."""
        class SimpleLinear(nnx.Module):
            def __init__(self, in_features: int, out_features: int, rngs: nnx.Rngs):
                self.linear = nnx.Linear(in_features, out_features, rngs=rngs)
            def __call__(self, x):
                return self.linear(x)
        
        rngs = nnx.Rngs(0)
        model = SimpleLinear(in_features=10, out_features=5, rngs=rngs)
        x = jnp.ones((4, 10))
        output = model(x)
        
        assert output.shape == (4, 5)
        assert jnp.all(jnp.isfinite(output))
    
    def test_mlp_from_shared(self):
        """Test MLP from shared components works."""
        rngs = nnx.Rngs(0)
        model = MLP(
            in_features=784,
            hidden_features=128,
            out_features=10,
            n_layers=3,
            rngs=rngs
        )
        
        x = jnp.ones((32, 784))
        output = model(x)
        
        assert output.shape == (32, 10)
        assert jnp.all(jnp.isfinite(output))
        
        # Check parameters
        params = nnx.state(model, nnx.Param)
        total_params = sum(p.size for p in jax.tree.leaves(params))
        assert total_params > 0
    
    def test_cnn_from_shared(self):
        """Test CNN from shared components works."""
        rngs = nnx.Rngs(0)
        model = CNN(num_classes=10, rngs=rngs)
        
        # MNIST-like input
        x = jnp.ones((16, 28, 28, 1))
        output = model(x, train=False)
        
        assert output.shape == (16, 10)
        assert jnp.all(jnp.isfinite(output))
        
        # Check parameters
        params = nnx.state(model, nnx.Param)
        total_params = sum(p.size for p in jax.tree.leaves(params))
        assert total_params > 0
    
    def test_model_inspection(self):
        """Test model inspection works."""
        rngs = nnx.Rngs(0)
        model = MLP(in_features=10, hidden_features=20, out_features=5, n_layers=2, rngs=rngs)
        
        # Get model state
        state = nnx.state(model)
        assert state is not None
        
        # Get parameters
        params = nnx.state(model, nnx.Param)
        num_params = sum(p.size for p in jax.tree.leaves(params))
        assert num_params == 745  # Expected from demo
        
        # Split model
        graphdef, state = nnx.split(model)
        assert graphdef is not None
        assert state is not None
