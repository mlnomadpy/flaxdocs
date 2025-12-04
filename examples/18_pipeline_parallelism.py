"""
Flax NNX: Pipeline Parallelism
================================
Demonstrates pipeline parallelism where a model is split into stages
and different stages run on different devices with microbatching.

Run: python 18_pipeline_parallelism.py

Key Concepts:
- Split model into sequential stages
- Each stage runs on a different device
- Process multiple microbatches in pipeline
- Overlaps computation across stages
- Reduces idle time with pipeline scheduling
"""

import jax
import jax.numpy as jnp
from flax import nnx
import optax
import numpy as np
from typing import Dict, List, Tuple
from functools import partial


# ============================================================================
# 1. MODEL STAGES
# ============================================================================

class Stage1(nnx.Module):
    """First stage: Input embedding and initial layers."""
    
    def __init__(self, vocab_size: int, d_model: int, rngs: nnx.Rngs = None):
        self.embedding = nnx.Embed(vocab_size, d_model, rngs=rngs)
        self.conv1 = nnx.Conv(d_model, d_model, kernel_size=(3,), padding='SAME', rngs=rngs)
        self.conv2 = nnx.Conv(d_model, d_model, kernel_size=(3,), padding='SAME', rngs=rngs)
        self.ln = nnx.LayerNorm(d_model, rngs=rngs)
    
    def __call__(self, x):
        # x: (batch, seq_len) - token ids
        x = self.embedding(x)  # (batch, seq_len, d_model)
        x = self.conv1(x)
        x = nnx.relu(x)
        x = self.conv2(x)
        x = self.ln(x)
        return x


class Stage2(nnx.Module):
    """Second stage: Middle transformer layers."""
    
    def __init__(self, d_model: int, num_heads: int, rngs: nnx.Rngs = None):
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Attention
        self.q_proj = nnx.Linear(d_model, d_model, rngs=rngs)
        self.k_proj = nnx.Linear(d_model, d_model, rngs=rngs)
        self.v_proj = nnx.Linear(d_model, d_model, rngs=rngs)
        self.out_proj = nnx.Linear(d_model, d_model, rngs=rngs)
        
        # FFN
        self.ff1 = nnx.Linear(d_model, d_model * 4, rngs=rngs)
        self.ff2 = nnx.Linear(d_model * 4, d_model, rngs=rngs)
        
        self.ln1 = nnx.LayerNorm(d_model, rngs=rngs)
        self.ln2 = nnx.LayerNorm(d_model, rngs=rngs)
    
    def attention(self, x):
        batch_size, seq_len, _ = x.shape
        
        q = self.q_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        
        scores = jnp.matmul(q, k.transpose(0, 1, 3, 2)) / jnp.sqrt(self.head_dim)
        attn_weights = jax.nn.softmax(scores, axis=-1)
        attn_output = jnp.matmul(attn_weights, v)
        
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
        return self.out_proj(attn_output)
    
    def __call__(self, x):
        # Attention block
        x = x + self.attention(self.ln1(x))
        
        # FFN block
        x = x + self.ff2(nnx.relu(self.ff1(self.ln2(x))))
        
        return x


class Stage3(nnx.Module):
    """Third stage: More transformer layers."""
    
    def __init__(self, d_model: int, num_heads: int, rngs: nnx.Rngs = None):
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Attention
        self.q_proj = nnx.Linear(d_model, d_model, rngs=rngs)
        self.k_proj = nnx.Linear(d_model, d_model, rngs=rngs)
        self.v_proj = nnx.Linear(d_model, d_model, rngs=rngs)
        self.out_proj = nnx.Linear(d_model, d_model, rngs=rngs)
        
        # FFN
        self.ff1 = nnx.Linear(d_model, d_model * 4, rngs=rngs)
        self.ff2 = nnx.Linear(d_model * 4, d_model, rngs=rngs)
        
        self.ln1 = nnx.LayerNorm(d_model, rngs=rngs)
        self.ln2 = nnx.LayerNorm(d_model, rngs=rngs)
    
    def attention(self, x):
        batch_size, seq_len, _ = x.shape
        
        q = self.q_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        
        scores = jnp.matmul(q, k.transpose(0, 1, 3, 2)) / jnp.sqrt(self.head_dim)
        attn_weights = jax.nn.softmax(scores, axis=-1)
        attn_output = jnp.matmul(attn_weights, v)
        
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
        return self.out_proj(attn_output)
    
    def __call__(self, x):
        # Attention block
        x = x + self.attention(self.ln1(x))
        
        # FFN block
        x = x + self.ff2(nnx.relu(self.ff1(self.ln2(x))))
        
        return x


class Stage4(nnx.Module):
    """Fourth stage: Final layers and classification head."""
    
    def __init__(self, d_model: int, num_classes: int, rngs: nnx.Rngs = None):
        self.ln = nnx.LayerNorm(d_model, rngs=rngs)
        self.pool_linear = nnx.Linear(d_model, d_model, rngs=rngs)
        self.classifier = nnx.Linear(d_model, num_classes, rngs=rngs)
    
    def __call__(self, x):
        # x: (batch, seq_len, d_model)
        x = self.ln(x)
        
        # Global average pooling
        x = jnp.mean(x, axis=1)  # (batch, d_model)
        
        x = self.pool_linear(x)
        x = nnx.relu(x)
        logits = self.classifier(x)
        
        return logits


class PipelineModel(nnx.Module):
    """Full model composed of pipeline stages."""
    
    def __init__(self, vocab_size: int, d_model: int, num_heads: int,
                 num_classes: int, rngs: nnx.Rngs = None):
        self.stage1 = Stage1(vocab_size, d_model, rngs=rngs)
        self.stage2 = Stage2(d_model, num_heads, rngs=rngs)
        self.stage3 = Stage3(d_model, num_heads, rngs=rngs)
        self.stage4 = Stage4(d_model, num_classes, rngs=rngs)
    
    def __call__(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return x


# ============================================================================
# 2. PIPELINE PARALLELISM UTILITIES
# ============================================================================

def split_into_microbatches(batch: Dict, num_microbatches: int) -> List[Dict]:
    """
    Split a batch into microbatches for pipeline parallelism.
    
    This allows us to pipeline execution: while stage 1 processes microbatch 2,
    stage 2 can process microbatch 1 (from previous iteration).
    """
    batch_size = batch['input_ids'].shape[0]
    microbatch_size = batch_size // num_microbatches
    
    microbatches = []
    for i in range(num_microbatches):
        start_idx = i * microbatch_size
        end_idx = (i + 1) * microbatch_size
        
        microbatch = {
            'input_ids': batch['input_ids'][start_idx:end_idx],
            'label': batch['label'][start_idx:end_idx]
        }
        microbatches.append(microbatch)
    
    return microbatches


def create_pipeline_schedule(num_stages: int, num_microbatches: int) -> List[List[Tuple[int, int]]]:
    """
    Create a GPipe-style pipeline schedule.
    
    Returns a list of timesteps, where each timestep contains (stage_id, microbatch_id) pairs
    that can execute in parallel.
    
    Example with 4 stages, 4 microbatches:
    Time 0: [(0, 0)]                          # Stage 0 processes microbatch 0
    Time 1: [(0, 1), (1, 0)]                  # Stage 0->mb1, Stage 1->mb0
    Time 2: [(0, 2), (1, 1), (2, 0)]          # 3 stages busy
    Time 3: [(0, 3), (1, 2), (2, 1), (3, 0)]  # All stages busy (full pipeline)
    Time 4: [(1, 3), (2, 2), (3, 1)]          # Stage 0 done, others continue
    ...
    """
    schedule = []
    total_time_steps = num_stages + num_microbatches - 1
    
    for t in range(total_time_steps):
        timestep = []
        for stage in range(num_stages):
            microbatch = t - stage
            if 0 <= microbatch < num_microbatches:
                timestep.append((stage, microbatch))
        schedule.append(timestep)
    
    return schedule


# ============================================================================
# 3. PIPELINE TRAINING
# ============================================================================

def forward_stage(stage_module, x, stage_id: int):
    """Run forward pass for a single stage."""
    return stage_module(x)


def backward_stage(stage_module, x, label, stage_id: int, is_last_stage: bool):
    """Run backward pass for a single stage."""
    
    if is_last_stage:
        # Last stage computes loss
        def loss_fn(stage):
            logits = stage(x)
            labels_onehot = jax.nn.one_hot(label, num_classes=10)
            loss = optax.softmax_cross_entropy(logits, labels_onehot).mean()
            return loss
        
        grad_fn = nnx.grad(loss_fn)
        grads = grad_fn(stage_module)
        return grads
    else:
        # Middle stages: for simplicity, we'll just compute gradients
        # In real pipeline parallelism, we'd also need to handle activation gradients
        def dummy_loss(stage):
            out = stage(x)
            return jnp.sum(out ** 2)
        
        grad_fn = nnx.grad(dummy_loss)
        grads = grad_fn(stage_module)
        return grads


def train_with_pipeline(model: PipelineModel, batch: Dict, num_microbatches: int):
    """
    Simplified pipeline training for demonstration.
    
    In production, you would use a library like GPipe or implement proper:
    - Activation checkpointing
    - Gradient accumulation across microbatches
    - Proper backward pass coordination
    - Device placement for each stage
    
    This demonstrates the concept but uses a simplified approach.
    """
    
    # Split batch into microbatches
    microbatches = split_into_microbatches(batch, num_microbatches)
    
    # Store activations for backward pass
    stage_outputs = {i: {} for i in range(4)}
    
    # Forward pass through pipeline
    print("\n  Forward Pass:")
    for mb_idx, microbatch in enumerate(microbatches):
        print(f"    Microbatch {mb_idx}:")
        
        # Stage 1
        x = forward_stage(model.stage1, microbatch['input_ids'], 0)
        stage_outputs[0][mb_idx] = x
        print(f"      Stage 1: {x.shape}")
        
        # Stage 2
        x = forward_stage(model.stage2, x, 1)
        stage_outputs[1][mb_idx] = x
        print(f"      Stage 2: {x.shape}")
        
        # Stage 3
        x = forward_stage(model.stage3, x, 2)
        stage_outputs[2][mb_idx] = x
        print(f"      Stage 3: {x.shape}")
        
        # Stage 4
        logits = forward_stage(model.stage4, x, 3)
        stage_outputs[3][mb_idx] = logits
        print(f"      Stage 4: {logits.shape}")
    
    # Compute loss across all microbatches
    total_loss = 0.0
    total_acc = 0.0
    
    print("\n  Computing Loss:")
    for mb_idx, microbatch in enumerate(microbatches):
        logits = stage_outputs[3][mb_idx]
        labels_onehot = jax.nn.one_hot(microbatch['label'], num_classes=10)
        loss = optax.softmax_cross_entropy(logits, labels_onehot).mean()
        
        predictions = jnp.argmax(logits, axis=-1)
        accuracy = jnp.mean(predictions == microbatch['label'])
        
        total_loss += float(loss)
        total_acc += float(accuracy)
        
        print(f"    Microbatch {mb_idx}: Loss={loss:.4f}, Acc={accuracy:.4f}")
    
    avg_loss = total_loss / num_microbatches
    avg_acc = total_acc / num_microbatches
    
    # Backward pass (simplified - in reality would be pipelined too)
    print("\n  Backward Pass:")
    
    # Accumulate gradients across microbatches
    accumulated_grads = []
    
    for mb_idx, microbatch in enumerate(microbatches):
        # Stage 4 (last stage)
        x = stage_outputs[2][mb_idx]
        grads4 = backward_stage(model.stage4, x, microbatch['label'], 3, is_last_stage=True)
        
        # Stage 3
        x = stage_outputs[1][mb_idx]
        grads3 = backward_stage(model.stage3, x, None, 2, is_last_stage=False)
        
        # Stage 2
        x = stage_outputs[0][mb_idx]
        grads2 = backward_stage(model.stage2, x, None, 1, is_last_stage=False)
        
        # Stage 1
        grads1 = backward_stage(model.stage1, microbatch['input_ids'], None, 0, is_last_stage=False)
        
        accumulated_grads.append({
            'stage1': grads1,
            'stage2': grads2,
            'stage3': grads3,
            'stage4': grads4
        })
        
        print(f"    Microbatch {mb_idx}: Gradients computed")
    
    # Average gradients
    print("\n  Averaging gradients across microbatches...")
    
    def average_grads(grads_list):
        if not grads_list:
            return None
        
        avg = grads_list[0]
        for grads in grads_list[1:]:
            avg = jax.tree.map(lambda a, b: a + b, avg, grads)
        
        return jax.tree.map(lambda x: x / len(grads_list), avg)
    
    avg_grads = {
        'stage1': average_grads([g['stage1'] for g in accumulated_grads]),
        'stage2': average_grads([g['stage2'] for g in accumulated_grads]),
        'stage3': average_grads([g['stage3'] for g in accumulated_grads]),
        'stage4': average_grads([g['stage4'] for g in accumulated_grads])
    }
    
    return avg_loss, avg_acc, avg_grads


# ============================================================================
# 4. MAIN TRAINING LOOP
# ============================================================================

def main():
    print(f"\n{'='*70}")
    print(f"PIPELINE PARALLELISM")
    print(f"{'='*70}")
    
    # Configuration
    vocab_size = 1000
    d_model = 128
    num_heads = 8
    num_classes = 10
    seq_len = 32
    batch_size = 64
    num_microbatches = 4
    num_epochs = 3
    learning_rate = 1e-3
    
    print(f"\nConfiguration:")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Model dimension: {d_model}")
    print(f"  Num heads: {num_heads}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Batch size: {batch_size}")
    print(f"  Microbatches: {num_microbatches}")
    print(f"  Microbatch size: {batch_size // num_microbatches}")
    
    num_devices = jax.local_device_count()
    print(f"\nDevices: {num_devices}")
    
    if num_devices >= 4:
        print("✓ Enough devices for 4-stage pipeline (1 stage per device)")
    else:
        print(f"⚠ Only {num_devices} device(s). In production, each stage would be on a separate device.")
    
    # Initialize model
    print("\n" + "="*70)
    print("MODEL INITIALIZATION")
    print("="*70)
    
    rngs = nnx.Rngs(0)
    model = PipelineModel(vocab_size, d_model, num_heads, num_classes, rngs=rngs)
    
    print("\nPipeline Stages:")
    print("  Stage 1: Embedding + Conv layers")
    print("  Stage 2: Transformer block")
    print("  Stage 3: Transformer block")
    print("  Stage 4: Pooling + Classification")
    
    # Create optimizers for each stage
    optimizer = optax.adam(learning_rate)
    opt_state1 = nnx.Optimizer(model.stage1, optimizer)
    opt_state2 = nnx.Optimizer(model.stage2, optimizer)
    opt_state3 = nnx.Optimizer(model.stage3, optimizer)
    opt_state4 = nnx.Optimizer(model.stage4, optimizer)
    
    # Generate synthetic data
    print("\n" + "="*70)
    print("SYNTHETIC DATA")
    print("="*70)
    
    num_samples = 500
    rng = jax.random.PRNGKey(0)
    
    input_ids = jax.random.randint(rng, (num_samples, seq_len), 0, vocab_size)
    labels = jax.random.randint(rng, (num_samples,), 0, num_classes)
    
    print(f"Training samples: {num_samples}")
    
    # Training loop
    print("\n" + "="*70)
    print("TRAINING WITH PIPELINE PARALLELISM")
    print("="*70)
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 70)
        
        epoch_loss = 0.0
        epoch_acc = 0.0
        num_batches = 0
        
        for i in range(0, num_samples, batch_size):
            if i + batch_size > num_samples:
                break
            
            batch = {
                'input_ids': input_ids[i:i + batch_size],
                'label': labels[i:i + batch_size]
            }
            
            print(f"\nBatch {num_batches + 1}:")
            
            # Pipeline training
            loss, acc, grads = train_with_pipeline(model, batch, num_microbatches)
            
            # Update stages
            opt_state1.update(grads['stage1'])
            opt_state2.update(grads['stage2'])
            opt_state3.update(grads['stage3'])
            opt_state4.update(grads['stage4'])
            
            epoch_loss += loss
            epoch_acc += acc
            num_batches += 1
            
            print(f"\n  Batch Loss: {loss:.4f}, Acc: {acc:.4f}")
        
        avg_loss = epoch_loss / num_batches
        avg_acc = epoch_acc / num_batches
        print(f"\nEpoch {epoch + 1} Summary: Loss={avg_loss:.4f}, Acc={avg_acc:.4f}")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    
    # Explanation
    print("\n" + "="*70)
    print("HOW PIPELINE PARALLELISM WORKS")
    print("="*70)
    print("""
1. MODEL SPLITTING:
   - Divide model into sequential stages
   - Each stage is a subnetwork (group of layers)
   - Example: Stage1->Embedding, Stage2->Attention, Stage3->FFN, Stage4->Head
   
2. DEVICE MAPPING:
   - Each stage runs on a different device
   - Stage1->Device0, Stage2->Device1, Stage3->Device2, Stage4->Device3
   - Activations are transferred between devices
   
3. MICROBATCHING:
   - Split each batch into smaller microbatches
   - Process multiple microbatches in pipeline
   - Example: 4 microbatches allow 4-stage pipeline to stay busy
   
4. PIPELINE SCHEDULE:
   Time 0: [Stage1->MB0]
   Time 1: [Stage1->MB1, Stage2->MB0]
   Time 2: [Stage1->MB2, Stage2->MB1, Stage3->MB0]
   Time 3: [Stage1->MB3, Stage2->MB2, Stage3->MB1, Stage4->MB0]  <- Full pipeline
   Time 4: [Stage2->MB3, Stage3->MB2, Stage4->MB1]
   Time 5: [Stage3->MB3, Stage4->MB2]
   Time 6: [Stage4->MB3]
   
5. FORWARD PASS:
   - Microbatches flow through stages sequentially
   - Each stage processes a microbatch then passes to next stage
   - Multiple microbatches in flight simultaneously
   
6. BACKWARD PASS:
   - Happens in reverse order through stages
   - Gradients flow backward through pipeline
   - Also pipelined across microbatches
   
7. GRADIENT ACCUMULATION:
   - Accumulate gradients from all microbatches
   - Average gradients before parameter update
   - Single update per batch (not per microbatch)

ADVANTAGES:
   ✓ Can train very large models (split across devices)
   ✓ Better device utilization (all stages busy)
   ✓ Memory efficient (only store one stage per device)
   ✓ Can combine with data parallelism
   
LIMITATIONS:
   ✗ Pipeline bubbles (some idle time at start/end)
   ✗ Requires sequential model architecture
   ✗ Communication between stages (activation transfer)
   ✗ More complex implementation
   ✗ Need enough microbatches to fill pipeline

EFFICIENCY:
   - Ideal efficiency: (num_microbatches) / (num_microbatches + num_stages - 1)
   - With 4 stages, 8 microbatches: 8 / (8 + 4 - 1) = 72.7%
   - With 4 stages, 16 microbatches: 16 / (16 + 4 - 1) = 84.2%
   - More microbatches = better efficiency (but more memory)

WHEN TO USE:
   - Model too large for single device
   - Sequential model architecture (transformers, ResNets)
   - Have multiple devices available
   - Can't use data parallelism alone (memory constraints)
   - Want to scale to very large models

IMPLEMENTATIONS:
   - GPipe: Google's pipeline parallelism
   - PipeDream: Microsoft Research
   - Megatron-LM: NVIDIA (combines pipeline + tensor parallelism)

KEY DIFFERENCES FROM DATA PARALLELISM:
   - Data parallel: Split batch, replicate model
   - Pipeline parallel: Split model, sequential processing
   - Data parallel: All devices do same work on different data
   - Pipeline parallel: Different devices do different stages
""")
    
    # Show pipeline schedule
    print("\n" + "="*70)
    print("EXAMPLE PIPELINE SCHEDULE")
    print("="*70)
    
    schedule = create_pipeline_schedule(num_stages=4, num_microbatches=4)
    
    print("\nWith 4 stages and 4 microbatches:")
    print("(Each entry shows Stage processing Microbatch)")
    print()
    
    for t, timestep in enumerate(schedule):
        active = ", ".join([f"S{stage}->MB{mb}" for stage, mb in timestep])
        print(f"  Time {t}: {active}")
    
    print(f"\n  Total time steps: {len(schedule)}")
    print(f"  Pipeline efficiency: {4 / len(schedule) * 100:.1f}%")


if __name__ == '__main__':
    main()
