"""
Flax NNX: FSDP-Style Fully Sharded Data Parallel
==================================================
Demonstrates Fully Sharded Data Parallel (FSDP) training pattern,
where model parameters are sharded across devices to save memory.

Run: python distributed/fsdp_sharding.py

Key Concepts:
- Shard parameters across devices (reduce memory per device)
- All-gather parameters when needed for computation
- Shard gradients during backward pass
- Reduce-scatter gradients after backward
- Much lower memory usage than data parallelism
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



import sys
from pathlib import Path

# Add parent directory to path for shared imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# ============================================================================
# 1. LARGE MODEL FOR FSDP DEMONSTRATION
# ============================================================================

class LargeTransformer(nnx.Module):
    """
    Large transformer model that benefits from FSDP.
    
    In FSDP, we shard the parameters across devices, which means:
    - Each device only stores a fraction of the parameters
    - Memory usage per device is reduced
    - Can train larger models than with standard data parallelism
    """
    
    def __init__(self, vocab_size: int, d_model: int, num_heads: int,
                 d_ff: int, num_layers: int, num_classes: int,
                 rngs: nnx.Rngs = None):
        self.d_model = d_model
        
        # Large embedding layer (often a memory bottleneck)
        self.embedding = nnx.Embed(vocab_size, d_model, rngs=rngs)
        
        # Multiple transformer layers
        self.layers = []
        for _ in range(num_layers):
            layer = {
                'attn_q': nnx.Linear(d_model, d_model, rngs=rngs),
                'attn_k': nnx.Linear(d_model, d_model, rngs=rngs),
                'attn_v': nnx.Linear(d_model, d_model, rngs=rngs),
                'attn_out': nnx.Linear(d_model, d_model, rngs=rngs),
                'ff1': nnx.Linear(d_model, d_ff, rngs=rngs),
                'ff2': nnx.Linear(d_ff, d_model, rngs=rngs),
                'ln1': nnx.LayerNorm(d_model, rngs=rngs),
                'ln2': nnx.LayerNorm(d_model, rngs=rngs),
            }
            self.layers.append(layer)
        
        # Output layers
        self.ln_final = nnx.LayerNorm(d_model, rngs=rngs)
        self.output = nnx.Linear(d_model, num_classes, rngs=rngs)
        
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
    
    def attention(self, x, layer_dict):
        """Multi-head attention."""
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = layer_dict['attn_q'](x)
        k = layer_dict['attn_k'](x)
        v = layer_dict['attn_v'](x)
        
        # Reshape for multi-head
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        
        # Attention
        scores = jnp.matmul(q, k.transpose(0, 1, 3, 2)) / jnp.sqrt(self.head_dim)
        attn_weights = jax.nn.softmax(scores, axis=-1)
        attn_out = jnp.matmul(attn_weights, v)
        
        # Reshape back
        attn_out = attn_out.transpose(0, 2, 1, 3)
        attn_out = attn_out.reshape(batch_size, seq_len, self.d_model)
        
        return layer_dict['attn_out'](attn_out)
    
    def __call__(self, x):
        # Embedding
        x = self.embedding(x)
        
        # Transformer layers
        for layer_dict in self.layers:
            # Attention with residual
            attn_out = self.attention(layer_dict['ln1'](x), layer_dict)
            x = x + attn_out
            
            # Feed-forward with residual
            ff_out = layer_dict['ff2'](nnx.relu(layer_dict['ff1'](layer_dict['ln2'](x))))
            x = x + ff_out
        
        # Output
        x = self.ln_final(x)
        x = jnp.mean(x, axis=1)  # Global average pooling
        logits = self.output(x)
        
        return logits


# ============================================================================
# 2. FSDP SHARDING UTILITIES
# ============================================================================

def create_fsdp_mesh(num_devices: int):
    """
    Create a 1D mesh for FSDP.
    
    FSDP typically uses a 1D mesh where all devices are in the 'fsdp' axis.
    Parameters are sharded along this axis.
    """
    devices = mesh_utils.create_device_mesh((num_devices,))
    mesh = Mesh(devices, axis_names=('fsdp',))
    
    print(f"\nFSDP Mesh:")
    print(f"  Devices: {num_devices}")
    print(f"  Axis: 'fsdp'")
    print(f"  Topology: 1D (all devices in one axis)")
    
    return mesh


def shard_params_fsdp(params, mesh: Mesh, shard_size_threshold: int = 1024):
    """
    Shard parameters across devices for FSDP.
    
    Strategy:
    - Large parameters (>threshold elements): shard along first dimension
    - Small parameters (biases, layer norms): replicate
    
    This mimics PyTorch FSDP behavior where large tensors are sharded
    and small tensors are replicated for efficiency.
    
    Args:
        params: Model parameters
        mesh: Device mesh
        shard_size_threshold: Minimum size to shard (elements)
    """
    # Create shardings
    sharded_spec = NamedSharding(mesh, P('fsdp'))
    replicated_spec = NamedSharding(mesh, P())
    
    def shard_param(path, param):
        path_str = '/'.join(str(p) for p in path)
        
        # Decision logic for sharding
        should_shard = (
            param.size >= shard_size_threshold and
            len(param.shape) >= 2 and
            'kernel' in path_str.lower()
        )
        
        if should_shard:
            # Shard along first dimension
            spec = sharded_spec
            action = "SHARD"
        else:
            # Replicate small parameters
            spec = replicated_spec
            action = "REPLICATE"
        
        sharded = jax.device_put(param, spec)
        
        print(f"  {action:10s} {path_str:60s} {str(param.shape):20s} -> {spec.spec}")
        
        return sharded
    
    print("\nSharding Parameters:")
    sharded_params = jax.tree_util.tree_map_with_path(shard_param, params)
    
    return sharded_params


def analyze_memory_usage(params, mesh: Mesh):
    """
    Analyze memory usage with and without FSDP.
    
    Shows the memory savings from sharding parameters.
    """
    num_devices = len(mesh.devices.flat)
    
    # Calculate total parameters and bytes
    total_params = 0
    total_bytes = 0
    sharded_bytes_per_device = 0
    replicated_bytes = 0
    
    def count_param(path, param):
        nonlocal total_params, total_bytes, sharded_bytes_per_device, replicated_bytes
        
        param_count = param.size
        param_bytes = param.size * param.dtype.itemsize
        
        total_params += param_count
        total_bytes += param_bytes
        
        # Check if this parameter is sharded
        if hasattr(param, 'sharding'):
            if param.sharding.spec == P('fsdp'):
                # Sharded: each device stores 1/N of the parameter
                sharded_bytes_per_device += param_bytes / num_devices
            else:
                # Replicated: each device stores the full parameter
                replicated_bytes += param_bytes
    
    jax.tree_util.tree_map_with_path(count_param, params)
    
    # Total memory per device
    memory_per_device_fsdp = sharded_bytes_per_device + replicated_bytes
    memory_per_device_replicated = total_bytes
    
    print(f"\nMemory Analysis:")
    print(f"  Total Parameters: {total_params:,}")
    print(f"  Total Memory: {total_bytes / 1e6:.2f} MB")
    print(f"\n  WITHOUT FSDP (Replicated):")
    print(f"    Memory per device: {memory_per_device_replicated / 1e6:.2f} MB")
    print(f"    Total across {num_devices} devices: {memory_per_device_replicated * num_devices / 1e6:.2f} MB")
    print(f"\n  WITH FSDP (Sharded):")
    print(f"    Sharded memory per device: {sharded_bytes_per_device / 1e6:.2f} MB")
    print(f"    Replicated memory per device: {replicated_bytes / 1e6:.2f} MB")
    print(f"    Total per device: {memory_per_device_fsdp / 1e6:.2f} MB")
    print(f"    Memory savings: {(1 - memory_per_device_fsdp / memory_per_device_replicated) * 100:.1f}%")
    print(f"    Can train {memory_per_device_replicated / memory_per_device_fsdp:.1f}× larger model!")


# ============================================================================
# 3. FSDP TRAINING STEP
# ============================================================================

def loss_fn(model, batch):
    """Compute cross-entropy loss."""
    logits = model(batch['input_ids'])
    labels_onehot = jax.nn.one_hot(batch['label'], num_classes=10)
    
    loss = optax.softmax_cross_entropy(logits, labels_onehot).mean()
    
    predictions = jnp.argmax(logits, axis=-1)
    accuracy = jnp.mean(predictions == batch['label'])
    
    return loss, {'accuracy': accuracy}


def create_fsdp_train_step(mesh: Mesh):
    """
    Create training step with FSDP.
    
    Key operations:
    1. Parameters start sharded across devices
    2. During forward pass, all-gather parameters as needed
    3. Compute loss and gradients
    4. Gradients are sharded (reduce-scatter)
    5. Update sharded parameters
    
    The compiler automatically inserts all-gather and reduce-scatter ops.
    """
    
    # Sharding for data (shard batch dimension)
    data_sharding = NamedSharding(mesh, P('fsdp'))
    
    # Sharding for parameters (will be sharded in the state)
    param_sharding = NamedSharding(mesh, P('fsdp'))
    
    @partial(jax.jit, donate_argnums=(0,))
    def train_step(state: nnx.Optimizer, batch: Dict):
        """
        FSDP training step.
        
        What happens under the hood:
        1. Parameters are sharded across devices
        2. All-gather: temporarily gather full parameters for computation
        3. Forward pass with full parameters
        4. Backward pass computes gradients
        5. Reduce-scatter: average and shard gradients
        6. Update sharded parameters
        
        This is all automatic thanks to JAX's sharding propagation!
        """
        
        def compute_loss(model):
            return loss_fn(model, batch)
        
        # Compute gradients
        # JAX will automatically:
        # - All-gather sharded parameters before forward pass
        # - Compute gradients
        # - Reduce-scatter gradients (average + shard)
        grad_fn = nnx.value_and_grad(compute_loss, has_aux=True)
        (loss, metrics), grads = grad_fn(state.model)
        
        # Update parameters (they remain sharded)
        state.update(grads)
        
        return state, loss, metrics
    
    return train_step


# ============================================================================
# 4. TRAINING DEMONSTRATION
# ============================================================================

def main():
    print(f"\n{'='*70}")
    print(f"FULLY SHARDED DATA PARALLEL (FSDP)")
    print(f"{'='*70}")
    
    # Check devices
    num_devices = jax.local_device_count()
    print(f"\nAvailable devices: {num_devices}")
    
    if num_devices == 1:
        print("\nNote: Only 1 device. FSDP will work but won't show memory savings.")
        print("Memory savings are most apparent with 8+ devices.")
    
    # Model configuration
    vocab_size = 10000
    d_model = 512
    num_heads = 8
    d_ff = 2048
    num_layers = 6
    num_classes = 10
    seq_len = 128
    batch_size = 16
    learning_rate = 1e-4
    
    print(f"\nModel Configuration:")
    print(f"  Vocabulary: {vocab_size:,}")
    print(f"  Model dimension: {d_model}")
    print(f"  Feed-forward dimension: {d_ff}")
    print(f"  Attention heads: {num_heads}")
    print(f"  Layers: {num_layers}")
    print(f"  Sequence length: {seq_len}")
    
    # Create mesh
    mesh = create_fsdp_mesh(num_devices)
    
    # Initialize model
    print("\n" + "="*70)
    print("MODEL INITIALIZATION")
    print("="*70)
    
    rngs = nnx.Rngs(0)
    model = LargeTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers,
        num_classes=num_classes,
        rngs=rngs
    )
    
    print("\n✓ Model initialized")
    
    # Count parameters before sharding
    graphdef, params = nnx.split(model)
    
    total_params = sum(p.size for p in jax.tree.leaves(params))
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Approximate model size: {total_params * 4 / 1e6:.2f} MB (float32)")
    
    # Apply FSDP sharding
    print("\n" + "="*70)
    print("APPLYING FSDP SHARDING")
    print("="*70)
    
    with mesh:
        sharded_params = shard_params_fsdp(params, mesh, shard_size_threshold=1024)
    
    # Analyze memory usage
    with mesh:
        analyze_memory_usage(sharded_params, mesh)
    
    # Reconstruct model with sharded params
    model = nnx.merge(graphdef, sharded_params)
    
    # Create optimizer
    optimizer = optax.adam(learning_rate)
    state = nnx.Optimizer(model, optimizer)
    
    # Create training step
    train_step = create_fsdp_train_step(mesh)
    
    # Generate synthetic data
    print("\n" + "="*70)
    print("SYNTHETIC DATA")
    print("="*70)
    
    num_samples = 200
    rng = jax.random.PRNGKey(0)
    
    input_ids = jax.random.randint(rng, (num_samples, seq_len), 0, vocab_size)
    labels = jax.random.randint(rng, (num_samples,), 0, num_classes)
    
    print(f"Training samples: {num_samples}")
    
    # Training loop
    print("\n" + "="*70)
    print("TRAINING WITH FSDP")
    print("="*70)
    
    num_epochs = 3
    
    # Sharding for input data
    data_sharding = NamedSharding(mesh, P('fsdp'))
    
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
                    lambda x: jax.device_put(x, data_sharding),
                    batch
                )
                
                # Training step
                state, loss, metrics = train_step(state, batch_sharded)
                
                epoch_loss += float(loss)
                epoch_acc += float(metrics['accuracy'])
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            avg_acc = epoch_acc / num_batches
            print(f"\n  Epoch {epoch + 1} Summary: Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f}")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    
    # Detailed explanation
    print("\n" + "="*70)
    print("HOW FSDP (FULLY SHARDED DATA PARALLEL) WORKS")
    print("="*70)
    print("""
1. PARAMETER SHARDING:
   - Each device stores only a fraction of model parameters
   - Example: With 8 devices, each device stores 1/8 of parameters
   - Massive memory savings for large models
   
2. ALL-GATHER OPERATION:
   - Before computing a layer, all-gather its parameters
   - Temporarily reconstruct full parameters from shards
   - All devices get the same full parameters
   - This happens automatically during forward pass
   
3. FORWARD PASS:
   - Parameters are gathered just-in-time for each layer
   - Compute forward pass with full parameters
   - Can optionally discard gathered parameters after use (activation checkpointing)
   
4. BACKWARD PASS:
   - Compute gradients with respect to full parameters
   - Gradients are initially "full" size
   
5. REDUCE-SCATTER OPERATION:
   - Average gradients across devices (reduce)
   - Shard averaged gradients (scatter)
   - Each device keeps only its shard of gradients
   - Matches the parameter sharding pattern
   
6. PARAMETER UPDATE:
   - Each device updates its shard of parameters
   - No additional communication needed
   - Parameters remain sharded
   
MEMORY BREAKDOWN (per device):
   
   Without FSDP (Standard Data Parallel):
   - Parameters: 100% (replicated)
   - Gradients: 100% (replicated)
   - Optimizer states: 100% (replicated for Adam: 2× params)
   - Activations: 100% / num_devices (sharded batch)
   Total: ~400% of model size per device
   
   With FSDP:
   - Parameters: 100% / num_devices (sharded)
   - Gradients: 100% / num_devices (sharded)
   - Optimizer states: 200% / num_devices (sharded)
   - Activations: 100% / num_devices (sharded batch)
   Total: ~400% / num_devices of model size per device

COMPARISON WITH OTHER STRATEGIES:

   Data Parallelism (pmap):
   ✓ Simple implementation
   ✗ Full model on each device
   ✗ Memory = O(model_size)
   ✓ No extra communication
   
   FSDP:
   ✓ Shard everything
   ✓ Memory = O(model_size / num_devices)
   ✗ All-gather and reduce-scatter overhead
   ✓ Can train much larger models
   
   Pipeline Parallelism:
   ✓ Split model across devices
   ✓ Memory = O(model_size / num_stages)
   ✗ Pipeline bubbles (idle time)
   ✗ Requires sequential architecture
   
   Tensor Parallelism:
   ✓ Shard individual operations
   ✓ Good for very wide layers
   ✗ More communication
   ✗ Requires careful layer design

ADVANTAGES:
   ✓ Train models larger than single device memory
   ✓ Linear scaling of memory with devices
   ✓ Works with any model architecture
   ✓ Combines well with other strategies
   ✓ Minimal code changes needed
   
LIMITATIONS:
   ✗ Communication overhead (all-gather, reduce-scatter)
   ✗ Slower than data parallel for small models
   ✗ Best with fast interconnect (NVLink, InfiniBand)
   ✗ Need many devices for best efficiency

WHEN TO USE:
   - Model doesn't fit on single device with data parallelism
   - Training very large models (GPT, LLaMA scale)
   - Have 8+ devices with fast interconnect
   - Want to maximize model size given hardware
   - Need to combine with other parallelism strategies

REAL-WORLD EXAMPLES:
   - Meta LLaMA: 65B parameters with FSDP
   - GPT-3: 175B parameters (pipeline + tensor + FSDP)
   - Large Vision Transformers: ViT-G with FSDP

IMPLEMENTATION NOTES:
   - JAX automatically handles all-gather and reduce-scatter
   - Sharding spec P('fsdp') tells compiler to shard
   - Compiler optimizes communication patterns
   - Can mix with tensor parallelism: P('fsdp', 'tensor')
   
PERFORMANCE TIPS:
   1. Use larger batch sizes to amortize communication
   2. Enable gradient accumulation for effective batch size
   3. Use mixed precision (bfloat16) to reduce communication
   4. Combine with activation checkpointing for memory
   5. Profile to find communication bottlenecks
""")


if __name__ == '__main__':
    main()
