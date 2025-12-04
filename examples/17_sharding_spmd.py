"""
Flax NNX: Automatic Sharding with jax.jit (SPMD)
=================================================
Demonstrates automatic array sharding and SPMD (Single Program Multiple Data)
parallelism using JAX's modern sharding API with jax.jit.

Run: python 17_sharding_spmd.py

Key Concepts:
- Automatic sharding with NamedSharding and PartitionSpec
- jax.jit compiles with sharding constraints
- No pmap needed - compiler handles parallelism
- Can shard both data and model dimensions
- More flexible than pmap for complex sharding patterns
"""

import jax
import jax.numpy as jnp
from flax import nnx
import optax
import numpy as np
from typing import Dict, Tuple
from functools import partial
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import mesh_utils


# ============================================================================
# 1. MODEL DEFINITION
# ============================================================================

class TransformerBlock(nnx.Module):
    """Transformer block for demonstrating model sharding."""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, rngs: nnx.Rngs = None):
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Multi-head attention
        self.q_proj = nnx.Linear(d_model, d_model, rngs=rngs)
        self.k_proj = nnx.Linear(d_model, d_model, rngs=rngs)
        self.v_proj = nnx.Linear(d_model, d_model, rngs=rngs)
        self.out_proj = nnx.Linear(d_model, d_model, rngs=rngs)
        
        # Feed-forward network
        self.ff1 = nnx.Linear(d_model, d_ff, rngs=rngs)
        self.ff2 = nnx.Linear(d_ff, d_model, rngs=rngs)
        
        # Layer normalization
        self.ln1 = nnx.LayerNorm(d_model, rngs=rngs)
        self.ln2 = nnx.LayerNorm(d_model, rngs=rngs)
    
    def attention(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        
        q = q.transpose(0, 2, 1, 3)  # (batch, heads, seq, head_dim)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        
        # Attention scores
        scores = jnp.matmul(q, k.transpose(0, 1, 3, 2)) / jnp.sqrt(self.head_dim)
        attn_weights = jax.nn.softmax(scores, axis=-1)
        
        # Apply attention
        attn_output = jnp.matmul(attn_weights, v)
        attn_output = attn_output.transpose(0, 2, 1, 3)
        attn_output = attn_output.reshape(batch_size, seq_len, self.d_model)
        
        # Output projection
        return self.out_proj(attn_output)
    
    def __call__(self, x):
        # Attention with residual
        attn_out = self.attention(self.ln1(x))
        x = x + attn_out
        
        # Feed-forward with residual
        ff_out = self.ff2(nnx.relu(self.ff1(self.ln2(x))))
        x = x + ff_out
        
        return x


class ShardedTransformer(nnx.Module):
    """Transformer model for sequence classification."""
    
    def __init__(self, vocab_size: int, d_model: int, num_heads: int,
                 d_ff: int, num_layers: int, num_classes: int,
                 max_len: int = 512, rngs: nnx.Rngs = None):
        # Embedding layer
        self.embedding = nnx.Embed(vocab_size, d_model, rngs=rngs)
        
        # Positional encoding
        self.pos_encoding = self.create_pos_encoding(max_len, d_model)
        
        # Transformer blocks
        self.blocks = [
            TransformerBlock(d_model, num_heads, d_ff, rngs=rngs)
            for _ in range(num_layers)
        ]
        
        # Classification head
        self.ln_final = nnx.LayerNorm(d_model, rngs=rngs)
        self.classifier = nnx.Linear(d_model, num_classes, rngs=rngs)
    
    def create_pos_encoding(self, max_len: int, d_model: int):
        """Create sinusoidal positional encoding."""
        position = jnp.arange(max_len)[:, None]
        div_term = jnp.exp(jnp.arange(0, d_model, 2) * (-jnp.log(10000.0) / d_model))
        
        pos_encoding = jnp.zeros((max_len, d_model))
        pos_encoding = pos_encoding.at[:, 0::2].set(jnp.sin(position * div_term))
        pos_encoding = pos_encoding.at[:, 1::2].set(jnp.cos(position * div_term))
        
        return pos_encoding
    
    def __call__(self, x):
        # x shape: (batch_size, seq_len)
        seq_len = x.shape[1]
        
        # Embedding + positional encoding
        x = self.embedding(x)
        x = x + self.pos_encoding[:seq_len, :]
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Classification (use first token like BERT's [CLS])
        x = self.ln_final(x[:, 0, :])
        logits = self.classifier(x)
        
        return logits


# ============================================================================
# 2. SHARDING CONFIGURATION
# ============================================================================

def create_mesh(num_devices: int):
    """
    Create a device mesh for sharding.
    
    We create a 2D mesh with axes:
    - 'data': for data parallelism (sharding batch dimension)
    - 'model': for model parallelism (sharding model dimensions)
    
    Example with 8 devices:
    - (8, 1): Pure data parallelism
    - (4, 2): 4-way data, 2-way model parallelism
    - (2, 4): 2-way data, 4-way model parallelism
    - (1, 8): Pure model parallelism
    """
    # For this example, we'll use pure data parallelism
    # In practice, you'd configure this based on your model size
    devices = mesh_utils.create_device_mesh((num_devices, 1))
    mesh = Mesh(devices, axis_names=('data', 'model'))
    
    print(f"\nDevice Mesh:")
    print(f"  Shape: {devices.shape}")
    print(f"  Axis names: {mesh.axis_names}")
    print(f"  Devices:\n{devices}")
    
    return mesh


def create_sharding_specs(mesh: Mesh):
    """
    Create sharding specifications for different tensor types.
    
    PartitionSpec defines how each dimension of a tensor is sharded:
    - P('data'): shard along 'data' mesh axis
    - P('model'): shard along 'model' mesh axis
    - P('data', 'model'): shard first dim along 'data', second along 'model'
    - P(None): replicate (no sharding)
    """
    specs = {
        # Data sharding: shard batch dimension only
        'data': NamedSharding(mesh, P('data', None)),
        
        # Model sharding: shard model dimension only
        # Used for large weight matrices
        'model': NamedSharding(mesh, P(None, 'model')),
        
        # 2D sharding: shard both batch and model dimensions
        'data_model': NamedSharding(mesh, P('data', 'model')),
        
        # Replicated: no sharding, replicate on all devices
        'replicated': NamedSharding(mesh, P(None)),
    }
    
    return specs


def shard_params(params, mesh: Mesh, strategy: str = 'replicated'):
    """
    Apply sharding to model parameters.
    
    Different strategies:
    - 'replicated': All parameters replicated (like pmap)
    - 'fsdp': Shard parameters along data axis (like FSDP)
    - 'tensor_parallel': Shard large matrices along model axis
    """
    specs = create_sharding_specs(mesh)
    
    if strategy == 'replicated':
        # Simplest: replicate all parameters
        spec = specs['replicated']
        return jax.tree.map(
            lambda x: jax.device_put(x, spec),
            params
        )
    
    elif strategy == 'fsdp':
        # Shard parameters along first dimension (FSDP-style)
        # Useful for very large models
        spec = specs['data']
        
        def shard_param(path, x):
            # Shard weight matrices, replicate biases
            if len(x.shape) >= 2 and 'kernel' in str(path):
                return jax.device_put(x, specs['data'])
            else:
                return jax.device_put(x, specs['replicated'])
        
        return jax.tree_util.tree_map_with_path(shard_param, params)
    
    elif strategy == 'tensor_parallel':
        # Shard large weight matrices along model axis
        spec_model = specs['model']
        spec_replicated = specs['replicated']
        
        def shard_param(path, x):
            # Shard large weight matrices along second dimension
            if len(x.shape) >= 2 and 'kernel' in str(path) and x.shape[-1] >= 256:
                return jax.device_put(x, spec_model)
            else:
                return jax.device_put(x, spec_replicated)
        
        return jax.tree_util.tree_map_with_path(shard_param, params)
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


# ============================================================================
# 3. TRAINING WITH AUTOMATIC SHARDING
# ============================================================================

def loss_fn(model: ShardedTransformer, batch: Dict):
    """Compute cross-entropy loss."""
    logits = model(batch['input_ids'])
    labels_onehot = jax.nn.one_hot(batch['label'], num_classes=10)
    
    loss = optax.softmax_cross_entropy(logits, labels_onehot).mean()
    
    predictions = jnp.argmax(logits, axis=-1)
    accuracy = jnp.mean(predictions == batch['label'])
    
    return loss, {'accuracy': accuracy}


def create_sharded_train_step(mesh: Mesh, sharding_specs):
    """
    Create a training step with automatic sharding.
    
    Key difference from pmap:
    - Use jax.jit instead of jax.pmap
    - Specify in_shardings and out_shardings
    - Compiler automatically handles communication
    - More flexible sharding patterns
    """
    
    # Define input/output shardings
    # Data is sharded along batch dimension
    data_sharding = sharding_specs['data']
    replicated_sharding = sharding_specs['replicated']
    
    @partial(
        jax.jit,
        # Specify how inputs are sharded
        in_shardings=(replicated_sharding, data_sharding),
        # Specify how outputs should be sharded
        out_shardings=(replicated_sharding, replicated_sharding, replicated_sharding)
    )
    def train_step(state: nnx.Optimizer, batch: Dict):
        """
        Training step with automatic sharding.
        
        The compiler will:
        1. Recognize input sharding patterns
        2. Propagate shardings through computation
        3. Insert communication ops as needed
        4. Optimize collective operations
        """
        
        def compute_loss(model):
            return loss_fn(model, batch)
        
        # Compute gradients
        grad_fn = nnx.value_and_grad(compute_loss, has_aux=True)
        (loss, metrics), grads = grad_fn(state.model)
        
        # Update parameters
        state.update(grads)
        
        return state, loss, metrics
    
    return train_step


# ============================================================================
# 4. TRAINING DEMONSTRATION
# ============================================================================

def train_with_sharding():
    """Complete training example with automatic sharding."""
    
    print(f"\n{'='*70}")
    print(f"AUTOMATIC SHARDING WITH JAX.JIT (SPMD)")
    print(f"{'='*70}")
    
    # Device setup
    num_devices = jax.local_device_count()
    print(f"\nAvailable devices: {num_devices}")
    print(f"Device types: {[d.platform for d in jax.local_devices()]}")
    
    if num_devices == 1:
        print("\nNote: Only 1 device available. Sharding will still work")
        print("but will demonstrate the API without actual parallelism.")
    
    # Create device mesh
    mesh = create_mesh(num_devices)
    sharding_specs = create_sharding_specs(mesh)
    
    # Hyperparameters
    vocab_size = 1000
    d_model = 256
    num_heads = 8
    d_ff = 512
    num_layers = 4
    num_classes = 10
    seq_len = 64
    batch_size = 32
    num_epochs = 3
    learning_rate = 1e-3
    
    print(f"\nModel Configuration:")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Model dimension: {d_model}")
    print(f"  Num heads: {num_heads}")
    print(f"  Num layers: {num_layers}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Batch size: {batch_size}")
    
    # Initialize model
    print("\n" + "="*70)
    print("MODEL INITIALIZATION")
    print("="*70)
    
    rngs = nnx.Rngs(0)
    model = ShardedTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers,
        num_classes=num_classes,
        rngs=rngs
    )
    
    # Create optimizer
    optimizer = optax.adam(learning_rate)
    state = nnx.Optimizer(model, optimizer)
    
    # Shard parameters
    print("\nApplying parameter sharding...")
    
    # We'll use the 'replicated' strategy for simplicity
    # In practice, you might use 'fsdp' or 'tensor_parallel' for large models
    with mesh:
        graphdef, params = nnx.split(state)
        params_sharded = shard_params(params, mesh, strategy='replicated')
        state = nnx.merge(graphdef, params_sharded)
    
    print("✓ Parameters sharded")
    
    # Inspect sharding
    print("\nParameter Sharding:")
    _, state_arrays = nnx.split(state.model)
    
    def print_sharding_info(path, x):
        path_str = '.'.join(str(p) for p in path)
        if hasattr(x, 'sharding'):
            print(f"  {path_str:50s} {str(x.shape):20s} {x.sharding.spec}")
    
    jax.tree_util.tree_map_with_path(print_sharding_info, state_arrays)
    
    # Create training step
    train_step = create_sharded_train_step(mesh, sharding_specs)
    
    # Generate synthetic data
    print("\n" + "="*70)
    print("SYNTHETIC DATA GENERATION")
    print("="*70)
    
    num_samples = 500
    rng = jax.random.PRNGKey(0)
    
    input_ids = jax.random.randint(rng, (num_samples, seq_len), 0, vocab_size)
    labels = jax.random.randint(rng, (num_samples,), 0, num_classes)
    
    print(f"Training samples: {num_samples}")
    print(f"Input shape: {input_ids.shape}")
    
    # Training loop
    print("\n" + "="*70)
    print("TRAINING WITH AUTOMATIC SHARDING")
    print("="*70)
    
    with mesh:
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 70)
            
            epoch_loss = 0.0
            epoch_acc = 0.0
            num_batches = 0
            
            for i in range(0, num_samples, batch_size):
                if i + batch_size > num_samples:
                    break
                
                # Get batch
                batch = {
                    'input_ids': input_ids[i:i + batch_size],
                    'label': labels[i:i + batch_size]
                }
                
                # Shard batch data
                batch_sharded = jax.tree.map(
                    lambda x: jax.device_put(x, sharding_specs['data']),
                    batch
                )
                
                # Training step
                state, loss, metrics = train_step(state, batch_sharded)
                
                # Accumulate metrics
                epoch_loss += float(loss)
                epoch_acc += float(metrics['accuracy'])
                num_batches += 1
            
            # Epoch summary
            avg_loss = epoch_loss / num_batches
            avg_acc = epoch_acc / num_batches
            print(f"\n  Epoch {epoch + 1} Summary: Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f}")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    
    # Explanation
    print("\n" + "="*70)
    print("HOW AUTOMATIC SHARDING (SPMD) WORKS")
    print("="*70)
    print("""
1. DEVICE MESH:
   - Create a logical grid of devices
   - Axes have names (e.g., 'data', 'model')
   - Example: (8, 1) mesh = 8 data parallel, 1 model parallel
   
2. SHARDING SPECIFICATION (PartitionSpec):
   - Defines how each tensor dimension is sharded
   - P('data', None): shard first dim along 'data' axis
   - P(None, 'model'): shard second dim along 'model' axis
   - P(None): replicate across all devices
   
3. AUTOMATIC SHARDING PROPAGATION:
   - Compiler analyzes computation graph
   - Propagates sharding through operations
   - Inserts communication (all-reduce, all-gather) automatically
   - Optimizes collective operations
   
4. JIT COMPILATION WITH SHARDINGS:
   - in_shardings: how inputs are already sharded
   - out_shardings: how outputs should be sharded
   - Compiler generates SPMD code
   - Single program runs on all devices
   
5. ADVANTAGES OVER PMAP:
   ✓ More flexible sharding patterns
   ✓ Can mix data and model parallelism freely
   ✓ Better performance (optimized collectives)
   ✓ Easier to express complex sharding
   ✓ Works with any number of devices
   ✓ Automatic resharding between operations
   
6. SHARDING STRATEGIES:
   
   a) Data Parallelism:
      - Shard batch dimension: P('data', None)
      - Replicate model parameters
      - Same as pmap but more flexible
   
   b) Tensor Parallelism:
      - Shard weight matrices: P(None, 'model')
      - Split large matrices across devices
      - Useful for huge transformer layers
   
   c) FSDP (Fully Sharded Data Parallel):
      - Shard parameters: P('data')
      - All-gather before use, shard after
      - Saves memory for large models
   
   d) 2D Parallelism:
      - Combine data and model parallelism
      - P('data', 'model')
      - Best for very large scale training

WHEN TO USE:
   - Any multi-device training (replaces pmap)
   - Models that don't fit on single device
   - Need flexible sharding patterns
   - Want automatic optimization
   - Modern JAX code (recommended over pmap)

KEY CONCEPTS:
   - SPMD: Single Program, Multiple Data
   - XLA compiler handles low-level details
   - Sharding is declarative (what, not how)
   - Compiler optimizes communication
   - Can mix strategies in same model
""")
    
    # Show actual sharding of final state
    print("\n" + "="*70)
    print("FINAL PARAMETER SHARDING")
    print("="*70)
    
    _, final_state = nnx.split(state.model)
    
    total_params = 0
    sharded_params = 0
    
    def count_params(path, x):
        nonlocal total_params, sharded_params
        if hasattr(x, 'size'):
            total_params += x.size
            if hasattr(x, 'sharding') and x.sharding.spec != P(None):
                sharded_params += x.size
    
    jax.tree_util.tree_map_with_path(count_params, final_state)
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Sharded parameters: {sharded_params:,}")
    print(f"Replicated parameters: {total_params - sharded_params:,}")


if __name__ == '__main__':
    train_with_sharding()
