"""
Shared model architectures for Flax NNX examples.

Common neural network components that can be reused across examples.
"""

import jax
import jax.numpy as jnp
from flax import nnx
from typing import Optional


class MLP(nnx.Module):
    """Multi-layer perceptron with configurable hidden layers."""
    
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        n_layers: int,
        rngs: nnx.Rngs,
        activation: str = "relu"
    ):
        """
        Args:
            in_features: Input dimension
            hidden_features: Hidden layer dimension
            out_features: Output dimension
            n_layers: Number of hidden layers
            rngs: Random number generators
            activation: Activation function name ('relu', 'gelu')
        """
        self.activation = activation
        
        # Build layers list
        layers_list = []
        
        # Input layer
        layers_list.append(nnx.Linear(in_features, hidden_features, rngs=rngs))
        
        # Hidden layers
        for _ in range(n_layers - 1):
            layers_list.append(nnx.Linear(hidden_features, hidden_features, rngs=rngs))
        
        # Use nnx.List for proper pytree handling
        self.layers = nnx.List(layers_list)
        
        # Output layer
        self.output = nnx.Linear(hidden_features, out_features, rngs=rngs)
    
    def __call__(self, x, train: bool = False):
        """Forward pass through the MLP.
        
        Args:
            x: Input tensor of shape (batch, in_features)
            train: Whether in training mode (for consistency with other models;
                  unused in vanilla MLP without dropout/batchnorm)
            
        Returns:
            Output tensor of shape (batch, out_features)
            
        Note:
            The `train` parameter is included for API consistency with CNN and
            Transformer models that use dropout/batchnorm. In this vanilla MLP
            implementation without such layers, it has no effect.
        """
        # Forward through hidden layers with activation
        for layer in self.layers:
            x = layer(x)
            if self.activation == "relu":
                x = nnx.relu(x)
            elif self.activation == "gelu":
                x = nnx.gelu(x)
        
        # Output layer (no activation)
        x = self.output(x)
        return x


class CNN(nnx.Module):
    """Simple CNN for image classification (MNIST/CIFAR style).
    
    Note: This CNN is designed for 28x28 images (MNIST-like). For other
    input sizes, consider creating a custom CNN or use the more flexible
    ResNet architecture.
    """
    
    def __init__(self, num_classes: int, rngs: nnx.Rngs, input_channels: int = 1):
        """
        Args:
            num_classes: Number of output classes
            rngs: Random number generators
            input_channels: Number of input channels (default: 1 for MNIST)
            
        Note:
            This architecture expects 28x28 input images. After two conv+pool
            layers (with SAME padding), the spatial dimensions become 7x7,
            resulting in a flattened size of 64 * 7 * 7 = 3136.
        """
        # Convolutional layers
        self.conv1 = nnx.Conv(input_channels, 32, kernel_size=(3, 3), rngs=rngs)
        self.conv2 = nnx.Conv(32, 64, kernel_size=(3, 3), rngs=rngs)
        
        # Batch normalization
        self.bn1 = nnx.BatchNorm(32, rngs=rngs)
        self.bn2 = nnx.BatchNorm(64, rngs=rngs)
        
        # Fully connected layers
        # For MNIST 28x28: after conv(SAME)+pool(2x2) twice: 28->14->7
        # So 64 * 7 * 7 = 3136
        self.fc1 = nnx.Linear(64 * 7 * 7, 128, rngs=rngs)
        self.fc2 = nnx.Linear(128, num_classes, rngs=rngs)
        
        # Dropout
        self.dropout = nnx.Dropout(rate=0.5, rngs=rngs)
    
    def __call__(self, x, train: bool = False):
        """Forward pass through CNN.
        
        Args:
            x: Input images of shape (batch, height, width, channels)
            train: Whether in training mode
            
        Returns:
            Logits of shape (batch, num_classes)
        """
        # Conv block 1
        x = self.conv1(x)
        x = self.bn1(x, use_running_average=not train)
        x = nnx.relu(x)
        x = nnx.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x, use_running_average=not train)
        x = nnx.relu(x)
        x = nnx.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        # Flatten
        x = x.reshape((x.shape[0], -1))
        
        # FC layers
        x = self.fc1(x)
        x = nnx.relu(x)
        x = self.dropout(x, deterministic=not train)
        x = self.fc2(x)
        
        return x


class MultiHeadAttention(nnx.Module):
    """Multi-head attention mechanism."""
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float,
        rngs: nnx.Rngs,
        causal: bool = False,
        max_len: int = 512
    ):
        """
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
            rngs: Random number generators
            causal: Whether to use causal (autoregressive) masking
            max_len: Maximum sequence length for causal mask
        """
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads
        self.causal = causal
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        # Q, K, V projections
        self.qkv_proj = nnx.Linear(d_model, 3 * d_model, rngs=rngs)
        self.out_proj = nnx.Linear(d_model, d_model, rngs=rngs)
        
        self.dropout = nnx.Dropout(dropout, rngs=rngs)
        
        # Causal mask if needed
        if causal:
            mask = jnp.tril(jnp.ones((max_len, max_len)))
            self.causal_mask = mask[None, None, :, :]
    
    def __call__(self, x, train: bool = False):
        """Apply multi-head attention.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            train: Whether in training mode
            
        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = jnp.split(qkv, 3, axis=2)
        
        q = q.squeeze(2).transpose(0, 2, 1, 3)  # (batch, heads, seq, head_dim)
        k = k.squeeze(2).transpose(0, 2, 1, 3)
        v = v.squeeze(2).transpose(0, 2, 1, 3)
        
        # Attention scores
        scores = jnp.matmul(q, k.transpose(0, 1, 3, 2)) / jnp.sqrt(self.head_dim)
        
        # Apply causal mask if needed
        if self.causal:
            mask = self.causal_mask[:, :, :seq_len, :seq_len]
            scores = jnp.where(mask, scores, -1e10)
        
        # Attention weights
        attn_weights = jax.nn.softmax(scores, axis=-1)
        attn_weights = self.dropout(attn_weights, deterministic=not train)
        
        # Apply attention to values
        attn_output = jnp.matmul(attn_weights, v)
        attn_output = attn_output.transpose(0, 2, 1, 3)
        attn_output = attn_output.reshape(batch_size, seq_len, self.d_model)
        
        # Output projection
        output = self.out_proj(attn_output)
        
        return output


class TransformerBlock(nnx.Module):
    """Transformer block with attention and feed-forward."""
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float,
        rngs: nnx.Rngs,
        causal: bool = False
    ):
        """
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout rate
            rngs: Random number generators
            causal: Whether to use causal attention
        """
        self.attention = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            rngs=rngs,
            causal=causal
        )
        
        # Feed-forward network
        self.ff1 = nnx.Linear(d_model, d_ff, rngs=rngs)
        self.ff2 = nnx.Linear(d_ff, d_model, rngs=rngs)
        self.dropout = nnx.Dropout(dropout, rngs=rngs)
        
        # Layer normalization
        self.ln1 = nnx.LayerNorm(d_model, rngs=rngs)
        self.ln2 = nnx.LayerNorm(d_model, rngs=rngs)
    
    def __call__(self, x, train: bool = False):
        """Apply transformer block.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            train: Whether in training mode
            
        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        # Attention with residual
        attn_output = self.attention(self.ln1(x), train=train)
        x = x + self.dropout(attn_output, deterministic=not train)
        
        # Feed-forward with residual
        ff_output = self.ff1(self.ln2(x))
        ff_output = nnx.gelu(ff_output)
        ff_output = self.dropout(ff_output, deterministic=not train)
        ff_output = self.ff2(ff_output)
        x = x + self.dropout(ff_output, deterministic=not train)
        
        return x


class ResNetBlock(nnx.Module):
    """ResNet residual block."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        rngs: nnx.Rngs
    ):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            stride: Stride for convolution
            rngs: Random number generators
        """
        self.conv1 = nnx.Conv(
            in_channels,
            out_channels,
            kernel_size=(3, 3),
            strides=(stride, stride),
            padding='SAME',
            rngs=rngs
        )
        self.bn1 = nnx.BatchNorm(out_channels, rngs=rngs)
        
        self.conv2 = nnx.Conv(
            out_channels,
            out_channels,
            kernel_size=(3, 3),
            padding='SAME',
            rngs=rngs
        )
        self.bn2 = nnx.BatchNorm(out_channels, rngs=rngs)
        
        # Projection shortcut if needed
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nnx.Conv(
                in_channels,
                out_channels,
                kernel_size=(1, 1),
                strides=(stride, stride),
                rngs=rngs
            )
            self.bn_shortcut = nnx.BatchNorm(out_channels, rngs=rngs)
        else:
            self.shortcut = None
    
    def __call__(self, x, train: bool = False):
        """Apply ResNet block.
        
        Args:
            x: Input tensor of shape (batch, height, width, channels)
            train: Whether in training mode
            
        Returns:
            Output tensor of shape (batch, height', width', out_channels)
        """
        residual = x
        
        # First conv
        x = self.conv1(x)
        x = self.bn1(x, use_running_average=not train)
        x = nnx.relu(x)
        
        # Second conv
        x = self.conv2(x)
        x = self.bn2(x, use_running_average=not train)
        
        # Shortcut connection
        if self.shortcut is not None:
            residual = self.shortcut(residual)
            residual = self.bn_shortcut(residual, use_running_average=not train)
        
        # Add residual and activate
        x = x + residual
        x = nnx.relu(x)
        
        return x
