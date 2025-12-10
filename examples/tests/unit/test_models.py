"""
Unit tests for shared model architectures.

Tests common model components like MLP, CNN, and Transformer blocks.
"""

import pytest
import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np


@pytest.mark.unit
class TestMLP:
    """Tests for Multi-Layer Perceptron."""
    
    def test_mlp_creation(self):
        """Test MLP can be created with valid parameters."""
        from shared.models import MLP
        
        rngs = nnx.Rngs(0)
        model = MLP(
            in_features=10,
            hidden_features=20,
            out_features=5,
            n_layers=2,
            rngs=rngs
        )
        assert model is not None
    
    def test_mlp_forward_shape(self):
        """Test MLP forward pass output shape."""
        from shared.models import MLP
        
        batch_size = 4
        in_features = 10
        out_features = 5
        
        rngs = nnx.Rngs(0)
        model = MLP(
            in_features=in_features,
            hidden_features=20,
            out_features=out_features,
            n_layers=2,
            rngs=rngs
        )
        
        x = jnp.ones((batch_size, in_features))
        output = model(x)
        
        assert output.shape == (batch_size, out_features)
    
    def test_mlp_output_is_finite(self):
        """Test MLP produces finite outputs."""
        from shared.models import MLP
        
        rngs = nnx.Rngs(0)
        model = MLP(in_features=10, hidden_features=20, out_features=5, n_layers=2, rngs=rngs)
        
        x = jnp.ones((4, 10))
        output = model(x)
        
        assert jnp.all(jnp.isfinite(output))


@pytest.mark.unit
class TestCNN:
    """Tests for Convolutional Neural Network."""
    
    def test_cnn_creation(self):
        """Test CNN can be created."""
        from shared.models import CNN
        
        rngs = nnx.Rngs(0)
        model = CNN(num_classes=10, rngs=rngs)
        assert model is not None
    
    def test_cnn_forward_shape(self):
        """Test CNN forward pass output shape for MNIST-like input."""
        from shared.models import CNN
        
        batch_size = 4
        num_classes = 10
        
        rngs = nnx.Rngs(0)
        model = CNN(num_classes=num_classes, rngs=rngs)
        
        # MNIST-like input: 28x28x1
        x = jnp.ones((batch_size, 28, 28, 1))
        output = model(x, train=False)
        
        assert output.shape == (batch_size, num_classes)
    
    def test_cnn_train_vs_eval_mode(self):
        """Test CNN behaves differently in train vs eval mode."""
        from shared.models import CNN
        
        rngs = nnx.Rngs(0)
        model = CNN(num_classes=10, rngs=rngs)
        
        x = jnp.ones((2, 28, 28, 1))
        
        # Get outputs in both modes
        output_train = model(x, train=True)
        output_eval = model(x, train=False)
        
        # Both should be valid
        assert jnp.all(jnp.isfinite(output_train))
        assert jnp.all(jnp.isfinite(output_eval))
        assert output_train.shape == output_eval.shape


@pytest.mark.unit
class TestTransformerBlock:
    """Tests for Transformer block components."""
    
    def test_attention_creation(self):
        """Test multi-head attention can be created."""
        from shared.models import MultiHeadAttention
        
        rngs = nnx.Rngs(0)
        attn = MultiHeadAttention(
            d_model=64,
            num_heads=4,
            dropout=0.1,
            rngs=rngs
        )
        assert attn is not None
    
    def test_attention_forward_shape(self):
        """Test attention output shape matches input shape."""
        from shared.models import MultiHeadAttention
        
        batch_size = 2
        seq_len = 10
        d_model = 64
        
        rngs = nnx.Rngs(0)
        attn = MultiHeadAttention(d_model=d_model, num_heads=4, dropout=0.1, rngs=rngs)
        
        x = jnp.ones((batch_size, seq_len, d_model))
        output = attn(x, train=False)
        
        assert output.shape == (batch_size, seq_len, d_model)
    
    def test_transformer_block_creation(self):
        """Test transformer block can be created."""
        from shared.models import TransformerBlock
        
        rngs = nnx.Rngs(0)
        block = TransformerBlock(
            d_model=64,
            num_heads=4,
            d_ff=256,
            dropout=0.1,
            rngs=rngs
        )
        assert block is not None
    
    def test_transformer_block_forward(self):
        """Test transformer block forward pass."""
        from shared.models import TransformerBlock
        
        batch_size = 2
        seq_len = 10
        d_model = 64
        
        rngs = nnx.Rngs(0)
        block = TransformerBlock(d_model=d_model, num_heads=4, d_ff=256, dropout=0.1, rngs=rngs)
        
        x = jnp.ones((batch_size, seq_len, d_model))
        output = block(x, train=False)
        
        assert output.shape == (batch_size, seq_len, d_model)
        assert jnp.all(jnp.isfinite(output))


@pytest.mark.unit
class TestResNetBlock:
    """Tests for ResNet block."""
    
    def test_resnet_block_creation(self):
        """Test ResNet block can be created."""
        from shared.models import ResNetBlock
        
        rngs = nnx.Rngs(0)
        block = ResNetBlock(in_channels=32, out_channels=64, stride=1, rngs=rngs)
        assert block is not None
    
    def test_resnet_block_forward_same_channels(self):
        """Test ResNet block forward with same input/output channels."""
        from shared.models import ResNetBlock
        
        batch_size = 2
        channels = 32
        height, width = 16, 16
        
        rngs = nnx.Rngs(0)
        block = ResNetBlock(in_channels=channels, out_channels=channels, stride=1, rngs=rngs)
        
        x = jnp.ones((batch_size, height, width, channels))
        output = block(x, train=False)
        
        assert output.shape == (batch_size, height, width, channels)
    
    def test_resnet_block_forward_different_channels(self):
        """Test ResNet block forward with different input/output channels."""
        from shared.models import ResNetBlock
        
        batch_size = 2
        in_channels = 32
        out_channels = 64
        height, width = 16, 16
        
        rngs = nnx.Rngs(0)
        block = ResNetBlock(in_channels=in_channels, out_channels=out_channels, stride=1, rngs=rngs)
        
        x = jnp.ones((batch_size, height, width, in_channels))
        output = block(x, train=False)
        
        assert output.shape == (batch_size, height, width, out_channels)
    
    def test_resnet_block_with_stride(self):
        """Test ResNet block with stride > 1."""
        from shared.models import ResNetBlock
        
        batch_size = 2
        channels = 32
        height, width = 16, 16
        stride = 2
        
        rngs = nnx.Rngs(0)
        block = ResNetBlock(in_channels=channels, out_channels=channels, stride=stride, rngs=rngs)
        
        x = jnp.ones((batch_size, height, width, channels))
        output = block(x, train=False)
        
        expected_height = height // stride
        expected_width = width // stride
        assert output.shape == (batch_size, expected_height, expected_width, channels)
