"""
Flax NNX: Basic Model Definition
=================================
This guide shows how to define models using Flax NNX.
Run: python basics/01_basic_model_definition.py
"""

import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np



import sys
from pathlib import Path

# Add parent directory to path for shared imports
sys.path.insert(0, str(Path(__file__).parent.parent))

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


# ============================================================================
# 2. MULTI-LAYER PERCEPTRON (MLP)
# ============================================================================

class MLP(nnx.Module):
    """Multi-layer perceptron with configurable hidden layers."""
    
    def __init__(self, in_features: int, hidden_features: int, 
                 out_features: int, n_layers: int, rngs: nnx.Rngs):
        self.layers = []
        
        # Input layer
        self.layers.append(nnx.Linear(in_features, hidden_features, rngs=rngs))
        
        # Hidden layers
        for _ in range(n_layers - 1):
            self.layers.append(nnx.Linear(hidden_features, hidden_features, rngs=rngs))
        
        # Output layer
        self.output = nnx.Linear(hidden_features, out_features, rngs=rngs)
    
    def __call__(self, x):
        # Forward through hidden layers with ReLU
        for layer in self.layers:
            x = nnx.relu(layer(x))
        
        # Output layer (no activation)
        x = self.output(x)
        return x


# ============================================================================
# 3. CONVOLUTIONAL NEURAL NETWORK (CNN)
# ============================================================================

class SimpleCNN(nnx.Module):
    """Simple CNN for image classification."""
    
    def __init__(self, num_classes: int, rngs: nnx.Rngs):
        # Convolutional layers
        self.conv1 = nnx.Conv(1, 32, kernel_size=(3, 3), rngs=rngs)
        self.conv2 = nnx.Conv(32, 64, kernel_size=(3, 3), rngs=rngs)
        
        # Fully connected layers
        self.fc1 = nnx.Linear(64 * 5 * 5, 128, rngs=rngs)  # Adjusted for MNIST size
        self.fc2 = nnx.Linear(128, num_classes, rngs=rngs)
        
        # Dropout
        self.dropout = nnx.Dropout(rate=0.5, rngs=rngs)
    
    def __call__(self, x, train: bool = False):
        # Conv block 1
        x = self.conv1(x)
        x = nnx.relu(x)
        x = nnx.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        # Conv block 2
        x = self.conv2(x)
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


# ============================================================================
# 4. RESIDUAL BLOCK & RESNET-STYLE MODEL
# ============================================================================

class ResidualBlock(nnx.Module):
    """Residual block with skip connection."""
    
    def __init__(self, features: int, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(features, features, kernel_size=(3, 3), 
                              padding='SAME', rngs=rngs)
        self.conv2 = nnx.Conv(features, features, kernel_size=(3, 3), 
                              padding='SAME', rngs=rngs)
        self.norm1 = nnx.BatchNorm(features, rngs=rngs)
        self.norm2 = nnx.BatchNorm(features, rngs=rngs)
    
    def __call__(self, x, train: bool = False):
        residual = x
        
        x = self.conv1(x)
        x = self.norm1(x, use_running_average=not train)
        x = nnx.relu(x)
        
        x = self.conv2(x)
        x = self.norm2(x, use_running_average=not train)
        
        # Skip connection
        x = x + residual
        x = nnx.relu(x)
        
        return x


class MiniResNet(nnx.Module):
    """Mini ResNet-style architecture."""
    
    def __init__(self, num_classes: int, rngs: nnx.Rngs):
        # Initial conv
        self.conv_init = nnx.Conv(1, 32, kernel_size=(3, 3), 
                                   padding='SAME', rngs=rngs)
        self.norm_init = nnx.BatchNorm(32, rngs=rngs)
        
        # Residual blocks
        self.res_block1 = ResidualBlock(32, rngs=rngs)
        self.res_block2 = ResidualBlock(32, rngs=rngs)
        
        # Classification head
        self.fc = nnx.Linear(32, num_classes, rngs=rngs)
    
    def __call__(self, x, train: bool = False):
        # Initial conv
        x = self.conv_init(x)
        x = self.norm_init(x, use_running_average=not train)
        x = nnx.relu(x)
        
        # Residual blocks
        x = self.res_block1(x, train=train)
        x = self.res_block2(x, train=train)
        
        # Global average pooling
        x = jnp.mean(x, axis=(1, 2))
        
        # Classification
        x = self.fc(x)
        
        return x


# ============================================================================
# 5. TRANSFORMER BLOCK
# ============================================================================

class TransformerBlock(nnx.Module):
    """Simple Transformer encoder block."""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, rngs: nnx.Rngs):
        # Multi-head attention
        self.attention = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=d_model,
            decode=False,
            rngs=rngs
        )
        
        # Feed-forward network
        self.ff1 = nnx.Linear(d_model, d_ff, rngs=rngs)
        self.ff2 = nnx.Linear(d_ff, d_model, rngs=rngs)
        
        # Layer normalization
        self.norm1 = nnx.LayerNorm(d_model, rngs=rngs)
        self.norm2 = nnx.LayerNorm(d_model, rngs=rngs)
        
        # Dropout
        self.dropout = nnx.Dropout(rate=0.1, rngs=rngs)
    
    def __call__(self, x, train: bool = False):
        # Self-attention with residual
        attn_out = self.attention(x)
        x = x + self.dropout(attn_out, deterministic=not train)
        x = self.norm1(x)
        
        # Feed-forward with residual
        ff_out = self.ff2(nnx.relu(self.ff1(x)))
        x = x + self.dropout(ff_out, deterministic=not train)
        x = self.norm2(x)
        
        return x


# ============================================================================
# DEMONSTRATION & TESTING
# ============================================================================

def main():
    print("=" * 80)
    print("Flax NNX Model Definition Examples")
    print("=" * 80)
    
    # Initialize RNG
    rngs = nnx.Rngs(0)
    
    # ========================================================================
    # Test 1: Simple Linear Model
    # ========================================================================
    print("\n1. Simple Linear Model")
    print("-" * 40)
    
    model = SimpleLinear(in_features=10, out_features=5, rngs=rngs)
    x = jnp.ones((4, 10))  # Batch of 4
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {nnx.state(model)}")
    
    # ========================================================================
    # Test 2: MLP
    # ========================================================================
    print("\n2. Multi-Layer Perceptron")
    print("-" * 40)
    
    mlp = MLP(in_features=784, hidden_features=256, 
              out_features=10, n_layers=3, rngs=rngs)
    x = jnp.ones((8, 784))  # Batch of 8
    output = mlp(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Count parameters
    state = nnx.state(mlp)
    total_params = sum(x.size for x in jax.tree_util.tree_leaves(state))
    print(f"Total parameters: {total_params:,}")
    
    # ========================================================================
    # Test 3: CNN
    # ========================================================================
    print("\n3. Convolutional Neural Network")
    print("-" * 40)
    
    cnn = SimpleCNN(num_classes=10, rngs=rngs)
    x = jnp.ones((4, 28, 28, 1))  # MNIST-like images
    
    # Training mode
    output_train = cnn(x, train=True)
    print(f"Input shape: {x.shape}")
    print(f"Output shape (train): {output_train.shape}")
    
    # Inference mode
    output_test = cnn(x, train=False)
    print(f"Output shape (test): {output_test.shape}")
    
    # ========================================================================
    # Test 4: ResNet-style Model
    # ========================================================================
    print("\n4. ResNet-style Model")
    print("-" * 40)
    
    resnet = MiniResNet(num_classes=10, rngs=rngs)
    x = jnp.ones((4, 28, 28, 1))
    
    output = resnet(x, train=True)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    state = nnx.state(resnet)
    total_params = sum(x.size for x in jax.tree_util.tree_leaves(state))
    print(f"Total parameters: {total_params:,}")
    
    # ========================================================================
    # Test 5: Transformer Block
    # ========================================================================
    print("\n5. Transformer Block")
    print("-" * 40)
    
    transformer = TransformerBlock(d_model=512, num_heads=8, 
                                   d_ff=2048, rngs=rngs)
    x = jnp.ones((4, 32, 512))  # (batch, seq_len, d_model)
    
    output = transformer(x, train=True)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # ========================================================================
    # Bonus: Model Inspection
    # ========================================================================
    print("\n" + "=" * 80)
    print("Model Inspection Examples")
    print("=" * 80)
    
    # Get model state (parameters)
    mlp_state = nnx.state(mlp)
    print(f"\nMLP state keys: {list(mlp_state.keys())[:5]}...")  # Show first 5
    
    # Get specific layer parameters
    print(f"\nFirst layer weight shape: {mlp.layers[0].kernel.value.shape}")
    print(f"First layer bias shape: {mlp.layers[0].bias.value.shape}")
    
    print("\n" + "=" * 80)
    print("✓ All models created successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
