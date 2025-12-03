"""
Flax NNX: HuggingFace Hub Integration
======================================
Upload models to HuggingFace Hub and stream data for training.
Run: pip install huggingface_hub datasets && python 08_huggingface_integration.py
"""

import jax
import jax.numpy as jnp
from flax import nnx
import optax
import numpy as np
from pathlib import Path
import tempfile
import json

# HuggingFace
try:
    from huggingface_hub import HfApi, create_repo, upload_folder
    from huggingface_hub import login as hf_login
    HF_HUB_AVAILABLE = True
except ImportError:
    print("Warning: huggingface_hub not available. Install: pip install huggingface_hub")
    HF_HUB_AVAILABLE = False

try:
    from datasets import load_dataset, Dataset
    import datasets
    DATASETS_AVAILABLE = True
except ImportError:
    print("Warning: datasets not available. Install: pip install datasets")
    DATASETS_AVAILABLE = False

try:
    from safetensors.flax import save_file as save_safetensors
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False


# ============================================================================
# EXAMPLE MODEL
# ============================================================================

class TextClassifier(nnx.Module):
    """Simple text classifier."""
    
    def __init__(self, vocab_size: int, embed_dim: int, num_classes: int, 
                 rngs: nnx.Rngs):
        self.embedding = nnx.Embed(vocab_size, embed_dim, rngs=rngs)
        self.fc1 = nnx.Linear(embed_dim, 128, rngs=rngs)
        self.fc2 = nnx.Linear(128, num_classes, rngs=rngs)
        self.dropout = nnx.Dropout(rate=0.5, rngs=rngs)
    
    def __call__(self, x, train: bool = False):
        # Embedding
        x = self.embedding(x)
        # Average pooling over sequence length
        x = jnp.mean(x, axis=1)
        # Classification layers
        x = self.fc1(x)
        x = nnx.relu(x)
        x = self.dropout(x, deterministic=not train)
        x = self.fc2(x)
        return x


# ============================================================================
# 1. UPLOAD MODEL TO HUGGINGFACE HUB
# ============================================================================

def upload_model_to_hub(
    model: nnx.Module,
    repo_name: str,
    model_name: str = "model",
    token: str = None,
    private: bool = True,
    model_card: str = None
):
    """Upload Flax NNX model to HuggingFace Hub."""
    if not HF_HUB_AVAILABLE or not SAFETENSORS_AVAILABLE:
        print("HuggingFace Hub or SafeTensors not available")
        return
    
    print("\n" + "=" * 80)
    print("Uploading Model to HuggingFace Hub")
    print("=" * 80)
    
    # Create temporary directory for model files
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir)
    
    try:
        # 1. Save model weights
        print("Saving model weights...")
        state = nnx.state(model)
        
        # Flatten state for SafeTensors
        def flatten_dict(d, parent_key=''):
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}.{k}" if parent_key else str(k)
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key).items())
                else:
                    if hasattr(v, 'value'):
                        v = np.array(v.value)
                    items.append((new_key, v))
            return dict(items)
        
        tensors = flatten_dict(state)
        save_safetensors(tensors, str(temp_path / f"{model_name}.safetensors"))
        print(f"✓ Saved {len(tensors)} tensors")
        
        # 2. Create config
        config = {
            "model_type": "flax_nnx",
            "framework": "flax_nnx",
            "total_parameters": sum(x.size for x in jax.tree_util.tree_leaves(state)),
        }
        
        with open(temp_path / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        print("✓ Created config.json")
        
        # 3. Create model card
        if model_card is None:
            model_card = f"""---
tags:
- flax
- nnx
- jax
library_name: flax
---

# {repo_name}

This is a Flax NNX model trained with JAX.

## Model Details

- **Framework**: Flax NNX
- **Total Parameters**: {config['total_parameters']:,}

## Usage

```python
from flax import nnx
from safetensors.flax import load_file
import jax.numpy as jnp

# Load weights
weights = load_file("model.safetensors")

# Initialize your model and load weights
# model = YourModel(...)
# nnx.update(model, weights)
```

## Training

This model was trained using Flax NNX.
"""
        
        with open(temp_path / "README.md", "w") as f:
            f.write(model_card)
        print("✓ Created README.md")
        
        # 4. Create/get repository
        print(f"\nCreating repository: {repo_name}")
        api = HfApi(token=token)
        
        try:
            repo_url = create_repo(
                repo_name,
                token=token,
                private=private,
                exist_ok=True
            )
            print(f"✓ Repository created/updated: {repo_url}")
        except Exception as e:
            print(f"Repository creation: {e}")
        
        # 5. Upload files
        print("\nUploading files...")
        api.upload_folder(
            folder_path=str(temp_path),
            repo_id=repo_name,
            repo_type="model",
            token=token,
        )
        
        print("\n" + "=" * 80)
        print("✓ Upload Complete!")
        print("=" * 80)
        print(f"Model available at: https://huggingface.co/{repo_name}")
        
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)


# ============================================================================
# 2. STREAM DATA FROM HUGGINGFACE DATASETS
# ============================================================================

def stream_dataset_from_hub(dataset_name: str, split: str = "train",
                           batch_size: int = 32, streaming: bool = True):
    """Stream dataset from HuggingFace Hub."""
    if not DATASETS_AVAILABLE:
        print("Datasets library not available")
        return None
    
    print("\n" + "=" * 80)
    print(f"Streaming Dataset: {dataset_name}")
    print("=" * 80)
    
    # Load dataset in streaming mode
    dataset = load_dataset(
        dataset_name,
        split=split,
        streaming=streaming,
        trust_remote_code=True
    )
    
    print(f"Dataset: {dataset_name}")
    print(f"Split: {split}")
    print(f"Streaming: {streaming}")
    
    if streaming:
        # For streaming datasets, we can't get the length
        print("Streaming mode: Length not available")
        
        # Show first few examples
        print("\nFirst example:")
        first_example = next(iter(dataset))
        for key, value in first_example.items():
            if isinstance(value, str):
                print(f"  {key}: {value[:100]}...")
            else:
                print(f"  {key}: {value}")
    else:
        print(f"Dataset size: {len(dataset)}")
    
    return dataset


def create_dataloader_from_hf(dataset, batch_size: int = 32, 
                              tokenizer_fn=None):
    """Create dataloader from HuggingFace dataset."""
    print("\n" + "=" * 80)
    print("Creating DataLoader")
    print("=" * 80)
    
    def batch_generator():
        batch = []
        for example in dataset:
            # Apply tokenization if provided
            if tokenizer_fn:
                example = tokenizer_fn(example)
            
            batch.append(example)
            
            if len(batch) >= batch_size:
                # Convert to JAX arrays
                batched = {}
                for key in batch[0].keys():
                    values = [b[key] for b in batch]
                    if isinstance(values[0], (int, float)):
                        batched[key] = jnp.array(values)
                    elif isinstance(values[0], np.ndarray):
                        batched[key] = jnp.array(np.stack(values))
                
                yield batched
                batch = []
    
    print(f"✓ DataLoader created (batch_size={batch_size})")
    return batch_generator()


# ============================================================================
# 3. TRAIN WITH STREAMING DATA
# ============================================================================

def train_with_streaming_data(
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    dataset_name: str,
    num_steps: int = 100,
    batch_size: int = 32
):
    """Train model with streaming data from HuggingFace."""
    if not DATASETS_AVAILABLE:
        print("Datasets library not available")
        return
    
    print("\n" + "=" * 80)
    print("Training with Streaming Data")
    print("=" * 80)
    
    # Load streaming dataset
    dataset = load_dataset(
        dataset_name,
        split="train",
        streaming=True,
        trust_remote_code=True
    )
    
    # Simple tokenizer (for demo - use proper tokenizer in production)
    def simple_tokenize(example):
        # This is dataset-specific - adjust as needed
        text = example.get('text', '')
        # Simple character encoding (demo only)
        tokens = [ord(c) % 256 for c in text[:100]]
        # Pad to fixed length
        tokens = tokens + [0] * (100 - len(tokens))
        example['input_ids'] = np.array(tokens[:100])
        return example
    
    # Create dataloader
    dataloader = create_dataloader_from_hf(
        dataset,
        batch_size=batch_size,
        tokenizer_fn=simple_tokenize
    )
    
    # Training loop
    print(f"\nTraining for {num_steps} steps...")
    
    for step, batch in enumerate(dataloader):
        if step >= num_steps:
            break
        
        # Dummy training step (replace with actual training)
        if 'input_ids' in batch:
            # Forward pass
            # logits = model(batch['input_ids'], train=True)
            # ... compute loss and update
            pass
        
        if (step + 1) % 10 == 0:
            print(f"Step {step + 1}/{num_steps}")
    
    print("✓ Training complete")


# ============================================================================
# 4. EXAMPLE: IMDB SENTIMENT CLASSIFICATION
# ============================================================================

def example_imdb_streaming():
    """Example: Train sentiment classifier on streaming IMDB data."""
    if not DATASETS_AVAILABLE:
        print("Datasets library not available")
        return
    
    print("\n" + "=" * 80)
    print("Example: IMDB Sentiment Classification (Streaming)")
    print("=" * 80)
    
    # Load IMDB dataset
    dataset = load_dataset("imdb", split="train", streaming=True)
    
    print("Streaming IMDB dataset...")
    
    # Process first few examples
    for i, example in enumerate(dataset):
        if i >= 3:
            break
        print(f"\nExample {i + 1}:")
        print(f"  Text: {example['text'][:100]}...")
        print(f"  Label: {'Positive' if example['label'] == 1 else 'Negative'}")


# ============================================================================
# 5. EXAMPLE: WIKIPEDIA STREAMING
# ============================================================================

def example_wikipedia_streaming():
    """Example: Stream Wikipedia data for pretraining."""
    if not DATASETS_AVAILABLE:
        print("Datasets library not available")
        return
    
    print("\n" + "=" * 80)
    print("Example: Wikipedia Streaming")
    print("=" * 80)
    
    # Load Wikipedia dataset (English, small subset)
    try:
        dataset = load_dataset(
            "wikipedia",
            "20220301.en",
            split="train",
            streaming=True,
            trust_remote_code=True
        )
        
        print("Streaming Wikipedia dataset...")
        
        # Process first article
        for i, example in enumerate(dataset):
            if i >= 1:
                break
            print(f"\nArticle {i + 1}:")
            print(f"  Title: {example.get('title', 'N/A')}")
            print(f"  Text: {example.get('text', '')[:200]}...")
    
    except Exception as e:
        print(f"Note: Wikipedia dataset requires configuration: {e}")


# ============================================================================
# DEMONSTRATION
# ============================================================================

def main():
    print("=" * 80)
    print("Flax NNX HuggingFace Integration Examples")
    print("=" * 80)
    
    # ========================================================================
    # 1. Streaming Datasets
    # ========================================================================
    
    # Example 1: IMDB
    if DATASETS_AVAILABLE:
        example_imdb_streaming()
    
    # Example 2: Wikipedia
    if DATASETS_AVAILABLE:
        example_wikipedia_streaming()
    
    # ========================================================================
    # 2. Model Upload (Demo - requires authentication)
    # ========================================================================
    print("\n" + "=" * 80)
    print("Model Upload to HuggingFace Hub")
    print("=" * 80)
    
    print("""
To upload a model to HuggingFace Hub:

1. Create an account at https://huggingface.co
2. Generate an access token at https://huggingface.co/settings/tokens
3. Login using:
   ```python
   from huggingface_hub import login
   login(token="your_token_here")
   ```

4. Upload your model:
   ```python
   upload_model_to_hub(
       model=model,
       repo_name="username/model-name",
       token="your_token_here",
       private=False
   )
   ```

Example (commented out to avoid actual upload):
""")
    
    # Demo model (don't actually upload)
    rngs = nnx.Rngs(42)
    model = TextClassifier(
        vocab_size=1000,
        embed_dim=128,
        num_classes=2,
        rngs=rngs
    )
    
    print("\nModel initialized (demo only, not uploading)")
    print(f"Model type: {type(model).__name__}")
    
    # Uncomment to actually upload (requires authentication):
    # upload_model_to_hub(
    #     model=model,
    #     repo_name="your-username/my-flax-model",
    #     token="your_hf_token",
    #     private=True
    # )
    
    # ========================================================================
    # Best Practices
    # ========================================================================
    print("\n" + "=" * 80)
    print("Best Practices for HuggingFace Integration")
    print("=" * 80)
    
    print("""
    1. Streaming Datasets:
       ✓ Use streaming=True for large datasets
       ✓ Process data on-the-fly
       ✓ No need to download entire dataset
       ✓ Perfect for pretraining language models
    
    2. Model Sharing:
       ✓ Always include model card (README.md)
       ✓ Document training details
       ✓ Include usage examples
       ✓ Specify license
       ✓ Use SafeTensors format
    
    3. Popular Datasets:
       • Text: wikitext, c4, bookcorpus, pile
       • Vision: imagenet, coco, places365
       • Audio: librispeech, common_voice
       • Multimodal: conceptual_captions, wit
    
    4. Authentication:
       • Use environment variable: HF_TOKEN
       • Or use huggingface-cli login
       • Store tokens securely
    
    5. Data Processing:
       • Use proper tokenizers
       • Implement efficient batching
       • Cache processed data when possible
       • Use multiprocessing for CPU-bound tasks
    """)
    
    print("\n" + "=" * 80)
    print("✓ Examples completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
