---
sidebar_position: 8
---

# Experiment Reproducibility

Reproducibility is the bedrock of scientific research. In deep learning, subtle factors like random seed initialization, nondeterministic GPU operations, and floating-point associativity can lead to vastly different results across runs.

JAX and Flax are designed with reproducibility in mind, primarily through explicit RNG key handling.

## The Sources of Randomness

To ensure your experiment is reproducible, you must control three sources of randomness:

1.  **Python/Numpy Randomness**: Standard libraries used for data loading/shuffling.
2.  **JAX/Flax Randomness**: Model initialization and stochastic layers (Dropout).
3.  **Hardware Determinism**: GPU operations that may be nondeterministic for speed.

## 1. Controlling Seeds

### Global Seeds

While JAX is functional, your data loading pipeline (e.g., PyTorch DataLoader or TensorFlow Datasets) likely relies on global state.

```python
import random
import numpy as np
import tensorflow as tf  # if using tf.data
import torch             # if using pytorch dataloaders

def set_global_seeds(seed=42):
    """Set random seeds for all standard libraries."""
    random.seed(seed)
    np.random.seed(seed)
    # tf.random.set_seed(seed)
    # torch.manual_seed(seed)
    print(f"Global seeds set to {seed}")
```

### JAX PRNG Keys (Explicit Randomness)

JAX does not use global state for randomness. Instead, you must explicitly pass a `PRNGKey`. This is a feature, not a bug, making randomness reproducible and forkable.

```python
import jax

def create_reproducible_rng(seed=42):
    """
    Create the root PRNG key. 
    All subsequent keys should be derived from this one using jax.random.split.
    """
    return jax.random.PRNGKey(seed)

# GOOD Usage
root_key = jax.random.PRNGKey(0)
init_key, train_key = jax.random.split(root_key)
params = model.init(init_key, dummy_input)

# BAD Usage
# params = model.init(jax.random.PRNGKey(0), dummy_input) 
# ^ Hardcoding seeds inside functions makes composition impossible/opaque.
```

## 2. Configuration Management

Reproducing a "model" means reproducing the code **and** the configuration. Hardcoding parameters (`lr = 0.001`) effectively loses that information for future reference.

Use a structured configuration object (like `ml_collections`, `hydra`, or `pydantic`) and save it to disk.

```python
import json
import os
from ml_collections import ConfigDict

def save_experiment_config(config: ConfigDict, log_dir: str):
    """Save the exact configuration used for this run."""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    config_path = os.path.join(log_dir, 'config.json')
    
    with open(config_path, 'w') as f:
        # Serializes the config to a readable JSON string
        f.write(config.to_json(indent=2))
    
    print(f"Config saved to {config_path}")

# Example Usage
config = ConfigDict()
config.seed = 42
config.optimizer = ConfigDict()
config.optimizer.learning_rate = 1e-3
config.optimizer.type = 'adamw'
config.model_arch = 'resnet18'

save_experiment_config(config, './logs/exp_001')
```

## 3. Hardware Determinism

On GPUs, some highly optimized operations (like `scatter_add` or certain CuDNN convolutions) are non-deterministic because the order of atomic additions is not guaranteed. Since floating point addition is not associative (`(a+b)+c != a+(b+c)`), this order variance changes the result bitwise.

### Handling Non-Determinism in JAX/XLA

In JAX, you can force XLA to use only deterministic kernels. This might come at a performance cost (sometimes significant).

```python
import os

# Set environment variable BEFORE importing jax
# This forces the XLA compiler to avoid non-deterministic algorithms
os.environ['XLA_FLAGS'] = '--xla_gpu_deterministic_ops=true'

import jax

# Verify setting
print("JAX running with deterministic ops:", os.environ.get('XLA_FLAGS'))
```

**Note**: If you run distributed training across multiple nodes, ensuring identical hardware topology is also required for bitwise reproducibility using `pmap`.

## 4. Debugging Divergence

If two "identical" runs diverge, how do you find the cause?

### Step 1: Check Initialization
Assert that initial parameters are identical.
```python
# Run A
jax.numpy.save('params_A.npy', params)

# Run B
params_A = jax.numpy.load('params_A.npy')
diff = jax.tree_map(lambda x, y: jnp.max(jnp.abs(x - y)), params, params_A)
print("Init Diff:", diff) # Should be 0.0
```

### Step 2: Check First Batch
Log the checksum of the first batch of data. This catches data loading randomness.
```python
batch_checksum = jnp.sum(batch['image'])
print(f"Batch 0 checksum: {batch_checksum}")
```

### Step 3: Check Gradients
If init and data match, check gradients after step 1.
```python
grads_checksum = jax.tree_util.tree_reduce(lambda x, y: x + jnp.sum(y), grads, 0)
print(f"Grads 0 checksum: {grads_checksum}")
```

## Checklist for Reproducibility

- [ ] **Code Versioning**: specific git commit hash is logged.
- [ ] **Environment**: `requirements.txt` or Docker container is saved.
- [ ] **Config**: All hyperparameters are saved to a file (JSON/YAML).
- [ ] **Data Split**: Train/Val/Test splits are static files, not random splits.
- [ ] **Seeds**: Global seeds set for Numpy/Python.
- [ ] **Keys**: JAX PRNG keys explicitly split and passed.
- [ ] **Determinism**: XLA deterministic flags set if precision is critical.
