---
sidebar_position: 1
---

# Model Export and Deployment

Learn how to export your trained Flax NNX models to standard formats for deployment in production environments, edge devices, and integration with other frameworks.

## Why Export Models?

After training, you need to deploy models. But Flax/JAX isn't always available in production:

- **Mobile/Edge**: iOS/Android devices don't run JAX
- **Web browsers**: Need JavaScript-compatible formats
- **Other frameworks**: TensorFlow Serving, ONNX Runtime, etc.
- **Language interop**: Call models from C++, Rust, Go
- **Performance**: Optimized runtimes can be faster than JAX

Common export formats:
- **SafeTensors**: Universal weight format (recommended)
- **ONNX**: Cross-framework standard
- **HuggingFace Hub**: Share models with the community

## SafeTensors: The Modern Standard

SafeTensors is a safe, fast format for storing tensors.

### Why SafeTensors?

**Advantages over pickle**:
- **Secure**: No code execution (pickle can run arbitrary code!)
- **Fast**: Memory-mapped loading, no deserialization
- **Cross-platform**: Works with PyTorch, Transformers, Diffusers
- **Metadata**: Store model config alongside weights

**When to use**: Default choice for saving model weights.

### Basic SafeTensors Export

```python
from safetensors.flax import save_file, load_file
from flax import nnx
import jax.numpy as jnp

# Train your model
model = MyModel(rngs=nnx.Rngs(params=0))
# ... training ...

# Extract parameters as regular dict
state = nnx.state(model)

# Convert to dictionary of numpy arrays
tensors = {}
for path, param in jax.tree_util.tree_leaves_with_path(state):
    # Convert path to string key
    key = '.'.join(str(p.key) for p in path if hasattr(p, 'key'))
    # Convert JAX array to numpy
    tensors[key] = jnp.array(param.value) if hasattr(param, 'value') else jnp.array(param)

# Save to file
save_file(tensors, 'model.safetensors')

print(f"Saved {len(tensors)} tensors to model.safetensors")
```

### Loading SafeTensors

```python
# Load from file
loaded_tensors = load_file('model.safetensors')

# Create new model (same architecture)
new_model = MyModel(rngs=nnx.Rngs(params=1))

# Reconstruct state structure
new_state = nnx.state(new_model)

# Update with loaded weights
for key, value in loaded_tensors.items():
    # Navigate to correct location in state tree
    parts = key.split('.')
    current = new_state
    for part in parts[:-1]:
        current = current[part]
    current[parts[-1]] = value

# Update model
nnx.update(new_model, new_state)

# Model ready to use!
output = new_model(input_data)
```

### Adding Metadata

Store configuration with weights:

```python
from safetensors.flax import save_file
import json

# Model configuration
metadata = {
    'model_type': 'MLP',
    'in_features': '784',
    'hidden_features': '256',
    'out_features': '10',
    'num_layers': '3',
    'activation': 'relu',
    'trained_epochs': '50',
    'val_accuracy': '98.5',
}

# Save with metadata
save_file(
    tensors=tensors,
    filename='model.safetensors',
    metadata=metadata
)

# Load and read metadata
from safetensors import safe_open

with safe_open('model.safetensors', framework='flax') as f:
    metadata = f.metadata()
    print(f"Model type: {metadata['model_type']}")
    print(f"Val accuracy: {metadata['val_accuracy']}")
    
    # Load tensors
    for key in f.keys():
        tensor = f.get_tensor(key)
```

## ONNX: Universal Model Format

ONNX (Open Neural Network Exchange) enables model portability across frameworks.

### Why ONNX?

**Use cases**:
- **Production serving**: ONNX Runtime is highly optimized
- **Cross-framework**: Train in JAX, deploy in PyTorch
- **Hardware acceleration**: Optimized for GPUs, CPUs, edge devices
- **Tooling**: Great support for optimization, quantization

**Limitations**:
- Not all JAX operations supported
- Dynamic shapes can be tricky
- Debugging exported models is harder

### JAX to ONNX Export

JAX doesn't directly export to ONNX. Strategy: JAX → TensorFlow → ONNX

```python
import jax
import jax.numpy as jnp
from flax import nnx
import tensorflow as tf
import tf2onnx

# Step 1: Create a tracing function
def create_trace_fn(model):
    """Create a function that can be traced"""
    
    @jax.jit
    def predict(x):
        return model(x)
    
    return predict

# Step 2: Convert JAX function to TensorFlow
model = MyModel(rngs=nnx.Rngs(params=0))
# ... train model ...

predict_fn = create_trace_fn(model)

# Get example input
example_input = jnp.ones((1, 784))  # Batch size 1
example_output = predict_fn(example_input)

# Convert to TensorFlow function
@tf.function(input_signature=[
    tf.TensorSpec(shape=[None, 784], dtype=tf.float32, name='input')
])
def tf_predict(x):
    # Convert TF tensor to JAX array
    x_jax = jnp.array(x.numpy())
    # Run JAX model
    output_jax = predict_fn(x_jax)
    # Convert back to TF
    return tf.constant(output_jax)

# Step 3: Export to ONNX
onnx_model, _ = tf2onnx.convert.from_function(
    tf_predict,
    input_signature=[tf.TensorSpec([None, 784], tf.float32, name='input')],
    opset=13,
    output_path='model.onnx'
)

print("Exported to model.onnx")
```

### Understanding ONNX Conversion

**Key concepts**:

**Opset version**: Set of supported operations
- Higher opset = more operations supported
- Opset 13+ recommended for modern models

**Static vs dynamic shapes**:
```python
# Static batch size (faster)
input_signature=[tf.TensorSpec([32, 784], tf.float32)]

# Dynamic batch size (more flexible)
input_signature=[tf.TensorSpec([None, 784], tf.float32)]
```

**Input/output names**:
```python
# Name inputs for clarity
tf.TensorSpec([None, 784], tf.float32, name='pixel_values')
# In ONNX Runtime: results = session.run(None, {'pixel_values': input})
```

### Running ONNX Models

Use ONNX Runtime for inference:

```python
import onnxruntime as ort
import numpy as np

# Load model
session = ort.InferenceSession('model.onnx')

# Check input/output info
print("Inputs:")
for input in session.get_inputs():
    print(f"  {input.name}: shape={input.shape}, dtype={input.type}")

print("Outputs:")
for output in session.get_outputs():
    print(f"  {output.name}: shape={output.shape}, dtype={output.type}")

# Run inference
input_data = np.random.randn(1, 784).astype(np.float32)
outputs = session.run(None, {'input': input_data})

logits = outputs[0]
predicted_class = np.argmax(logits, axis=-1)
print(f"Predicted class: {predicted_class}")
```

### ONNX Optimization

Make models faster with optimization:

```python
import onnx
from onnxruntime.transformers import optimizer

# Load ONNX model
model = onnx.load('model.onnx')

# Optimize
optimized_model = optimizer.optimize_model(
    'model.onnx',
    model_type='bert',  # or 'gpt2', 'vit', etc.
    num_heads=8,
    hidden_size=512,
)

optimized_model.save_model_to_file('model_optimized.onnx')

# Benchmark improvement
# Original: 100ms/batch
# Optimized: 50ms/batch (2x faster!)
```

## HuggingFace Hub Integration

Share models with the community and load pre-trained weights.

### Why HuggingFace Hub?

**Benefits**:
- **Discovery**: 500k+ models searchable
- **Versioning**: Git-based model versioning
- **Collaboration**: Share within teams or publicly
- **Infrastructure**: Free hosting with CDN
- **Ecosystem**: Integrates with Transformers, Datasets, Spaces

### Uploading Models

```python
from huggingface_hub import HfApi, create_repo
from flax import nnx
import jax.numpy as jnp

# Train your model
model = MyModel(rngs=nnx.Rngs(params=0))
# ... training ...

# Save model locally first
state = nnx.state(model)
# ... save as safetensors ...

# Create repository
api = HfApi()
repo_id = "username/my-awesome-model"

create_repo(
    repo_id=repo_id,
    repo_type="model",
    private=False,  # Set True for private repos
)

# Upload files
api.upload_file(
    path_or_fileobj="model.safetensors",
    path_in_repo="model.safetensors",
    repo_id=repo_id,
)

# Upload model card (README.md)
model_card = """
---
license: mit
tags:
- flax
- jax
- image-classification
datasets:
- mnist
metrics:
- accuracy
---

# My Awesome Model

Trained on MNIST with Flax NNX.

## Usage

```python
from flax import nnx
from huggingface_hub import hf_hub_download

# Download weights
weights_path = hf_hub_download(
    repo_id="username/my-awesome-model",
    filename="model.safetensors"
)

# Load into model
# ... (see SafeTensors section)
```

## Performance

- Validation Accuracy: 98.5%
- Test Accuracy: 98.2%
"""

api.upload_file(
    path_or_fileobj=model_card.encode(),
    path_in_repo="README.md",
    repo_id=repo_id,
)

print(f"Model uploaded to https://huggingface.co/{repo_id}")
```

### Downloading Models

```python
from huggingface_hub import hf_hub_download, list_repo_files

# List available files
repo_id = "username/my-awesome-model"
files = list_repo_files(repo_id)
print(f"Available files: {files}")

# Download specific file
weights_path = hf_hub_download(
    repo_id=repo_id,
    filename="model.safetensors",
    cache_dir="./models"  # Local cache
)

# Load into your model
# ... (use SafeTensors loading code) ...
```

### Model Cards Best Practices

A good model card includes:

```markdown
---
license: apache-2.0
tags:
- flax
- jax
- text-classification
- sentiment-analysis
language:
- en
datasets:
- imdb
metrics:
- accuracy
- f1
model-index:
- name: my-sentiment-model
  results:
  - task:
      type: text-classification
    dataset:
      name: IMDB
      type: imdb
    metrics:
    - type: accuracy
      value: 92.5
---

# Model Name

## Model Description
Brief description of what the model does.

## Training Data
What data was used for training.

## Training Procedure
Hyperparameters, optimization details.

## Evaluation Results
Performance on test set.

## Limitations and Biases
Known issues, when not to use.

## How to Use
Code examples for loading and inference.
```

## Deployment Patterns

### Pattern 1: REST API with FastAPI

```python
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from safetensors.flax import load_file

app = FastAPI()

# Load model at startup
model = MyModel(rngs=nnx.Rngs(params=0))
weights = load_file('model.safetensors')
# ... update model with weights ...

class PredictionRequest(BaseModel):
    data: list[list[float]]  # (batch, features)

class PredictionResponse(BaseModel):
    predictions: list[int]
    probabilities: list[list[float]]

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    # Convert to JAX array
    x = jnp.array(request.data)
    
    # Run model
    logits = model(x)
    probs = jax.nn.softmax(logits, axis=-1)
    preds = jnp.argmax(logits, axis=-1)
    
    return PredictionResponse(
        predictions=preds.tolist(),
        probabilities=probs.tolist()
    )

# Run with: uvicorn app:app --host 0.0.0.0 --port 8000
```

### Pattern 2: Batch Processing

```python
def batch_inference(model, data_loader, batch_size=256):
    """Efficient batch processing for large datasets"""
    
    all_predictions = []
    
    for batch in data_loader:
        # Process batch
        logits = model(batch)
        preds = jnp.argmax(logits, axis=-1)
        
        all_predictions.append(preds)
    
    return jnp.concatenate(all_predictions)

# Use with dataloader
predictions = batch_inference(model, test_loader)
```

### Pattern 3: ONNX Runtime Serving

```python
import onnxruntime as ort
from concurrent.futures import ThreadPoolExecutor

class ONNXPredictor:
    """Thread-safe ONNX model serving"""
    
    def __init__(self, model_path, num_threads=4):
        # Create session with optimization
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = num_threads
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        
        self.session = ort.InferenceSession(
            model_path,
            sess_options=sess_options
        )
        self.input_name = self.session.get_inputs()[0].name
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Run inference"""
        outputs = self.session.run(None, {self.input_name: x})
        return outputs[0]
    
    def predict_batch(self, batches: list[np.ndarray]) -> list[np.ndarray]:
        """Parallel batch processing"""
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(self.predict, batches))
        return results

# Use predictor
predictor = ONNXPredictor('model.onnx')
predictions = predictor.predict(input_data)
```

## Export Checklist

Before deploying, verify:

✅ **Model loads correctly**: Test load/inference cycle  
✅ **Output shapes match**: Same as training  
✅ **Numerical accuracy**: Compare exported vs original (< 1e-5 difference)  
✅ **Performance benchmark**: Measure latency/throughput  
✅ **Error handling**: Graceful failures for invalid inputs  
✅ **Version tracking**: Tag releases, document changes  
✅ **Documentation**: Usage examples, input/output specs  

## Common Export Issues

### Issue 1: Shape Mismatches

```python
# Problem: Model expects (batch, height, width, channels)
# But ONNX has (batch, channels, height, width)

# Solution: Add transpose
def export_with_transpose(model, example_input):
    def wrapped_predict(x):
        # NCHW → NHWC
        x = jnp.transpose(x, (0, 2, 3, 1))
        output = model(x)
        return output
    
    return wrapped_predict
```

### Issue 2: Unsupported Operations

```python
# Problem: Custom operations not in ONNX spec

# Solution: Replace with standard ops
# Instead of custom activation:
def custom_activation(x):
    return jnp.where(x > 0, x, 0.01 * x)  # LeakyReLU

# Use standard nnx.leaky_relu:
output = nnx.leaky_relu(x, negative_slope=0.01)
```

### Issue 3: Dynamic Shapes

```python
# Problem: Shape depends on input

# Solution: Use None for dynamic dimensions
@tf.function(input_signature=[
    tf.TensorSpec([None, None, 3], tf.float32)  # Variable height/width
])
def flexible_predict(x):
    return model(x)
```

## Next Steps

You can now export models for deployment! Learn:
- [Stream large datasets during training](./streaming-data.md)
- [Track experiments with W&B](./observability.md)
- [Build advanced architectures](../../research/advanced-techniques.md)

## Reference Code

**Complete modular examples:**
- [`examples/export/model_formats.py`](https://github.com/mlnomadpy/flaxdocs/tree/master/examples/export/model_formats.py) - SafeTensors and ONNX export patterns
- [`examples/integrations/huggingface.py`](https://github.com/mlnomadpy/flaxdocs/tree/master/examples/integrations/huggingface.py) - HuggingFace Hub model upload and sharing
