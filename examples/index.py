#!/usr/bin/env python3
"""
Flax NNX Examples Index
=======================
Quick reference for all available examples organized by category.
"""

EXAMPLES = {
    "Basics": {
        "description": "Fundamental concepts and building blocks",
        "examples": [
            {
                "name": "Model Definition",
                "file": "basics/model_definition.py",
                "description": "Define models using shared components (MLP, CNN)",
                "concepts": ["nnx.Module", "shared.models", "model inspection"]
            },
            {
                "name": "Model Definition (Original)",
                "file": "basics/01_basic_model_definition.py",
                "description": "Original self-contained model definition examples",
                "concepts": ["MLP", "CNN", "ResNet blocks", "Transformers"]
            },
            {
                "name": "Save and Load Models",
                "file": "basics/save_load_model.py",
                "description": "Checkpoint management with Orbax",
                "concepts": ["nnx.state", "orbax", "checkpointing"]
            },
            {
                "name": "Data Loading (TFDS)",
                "file": "basics/data_loading_tfds.py",
                "description": "Load data using TensorFlow Datasets",
                "concepts": ["tfds", "batching", "preprocessing"]
            },
            {
                "name": "Data Loading (Grain)",
                "file": "basics/data_loading_grain.py",
                "description": "Pure Python data loading with Grain",
                "concepts": ["grain", "custom datasets", "sharding"]
            }
        ]
    },
    "Training": {
        "description": "End-to-end training examples",
        "examples": [
            {
                "name": "Vision Training (MNIST)",
                "file": "training/vision_mnist.py",
                "description": "Train CNN on MNIST using shared components",
                "concepts": ["shared utilities", "train loop", "evaluation"]
            },
            {
                "name": "Language Model Training",
                "file": "training/language_model.py",
                "description": "Train Transformer language model",
                "concepts": ["transformers", "attention", "text generation"]
            }
        ]
    },
    "Export": {
        "description": "Model export and deployment",
        "examples": [
            {
                "name": "Model Formats",
                "file": "export/model_formats.py",
                "description": "Export to SafeTensors and ONNX",
                "concepts": ["safetensors", "onnx", "deployment"]
            }
        ]
    },
    "Integrations": {
        "description": "Integration with ML ecosystem",
        "examples": [
            {
                "name": "HuggingFace Integration",
                "file": "integrations/huggingface.py",
                "description": "Upload models and stream datasets from HF Hub",
                "concepts": ["huggingface_hub", "datasets", "model sharing"]
            },
            {
                "name": "ResNet Streaming",
                "file": "integrations/resnet_streaming.py",
                "description": "Train ResNet with streaming data",
                "concepts": ["streaming", "resnet", "large datasets"]
            },
            {
                "name": "Weights & Biases",
                "file": "integrations/wandb.py",
                "description": "Experiment tracking and visualization",
                "concepts": ["wandb", "logging", "hyperparameter sweeps"]
            }
        ]
    },
    "Advanced": {
        "description": "Advanced techniques and architectures",
        "examples": [
            {
                "name": "BERT Training",
                "file": "advanced/bert_fineweb.py",
                "description": "Train BERT on FineWeb, evaluate on MTEB",
                "concepts": ["bert", "mlm", "sentence embeddings"]
            },
            {
                "name": "GPT Training",
                "file": "advanced/gpt_training.py",
                "description": "Train GPT from scratch on FineWeb",
                "concepts": ["gpt", "causal lm", "text generation"]
            },
            {
                "name": "Contrastive Learning (SimCLR)",
                "file": "advanced/simclr_contrastive.py",
                "description": "Self-supervised learning with SimCLR",
                "concepts": ["simclr", "contrastive learning", "augmentation"]
            },
            {
                "name": "Meta-Learning (MAML)",
                "file": "advanced/maml_metalearning.py",
                "description": "Model-Agnostic Meta-Learning",
                "concepts": ["maml", "meta-learning", "few-shot"]
            },
            {
                "name": "Knowledge Distillation",
                "file": "advanced/knowledge_distillation.py",
                "description": "Transfer knowledge from teacher to student",
                "concepts": ["distillation", "teacher-student", "compression"]
            }
        ]
    },
    "Distributed": {
        "description": "Multi-device and distributed training",
        "examples": [
            {
                "name": "Data Parallel (pmap)",
                "file": "distributed/data_parallel_pmap.py",
                "description": "Data parallelism with pmap",
                "concepts": ["pmap", "data parallelism", "multi-gpu"]
            },
            {
                "name": "SPMD Sharding",
                "file": "distributed/sharding_spmd.py",
                "description": "Single Program Multiple Data with sharding",
                "concepts": ["spmd", "sharding", "mesh"]
            },
            {
                "name": "Pipeline Parallelism",
                "file": "distributed/pipeline_parallel.py",
                "description": "Pipeline model parallelism",
                "concepts": ["pipeline", "model parallelism", "stages"]
            },
            {
                "name": "FSDP Sharding",
                "file": "distributed/fsdp_sharding.py",
                "description": "Fully Sharded Data Parallelism",
                "concepts": ["fsdp", "sharding", "large models"]
            }
        ]
    }
}


def print_examples():
    """Print all examples organized by category."""
    print("\n" + "=" * 80)
    print("FLAX NNX EXAMPLES - COMPLETE INDEX")
    print("=" * 80 + "\n")
    
    for category, data in EXAMPLES.items():
        print(f"📁 {category.upper()}")
        print(f"   {data['description']}")
        print()
        
        for i, example in enumerate(data['examples'], 1):
            print(f"   {i}. {example['name']}")
            print(f"      File: {example['file']}")
            print(f"      {example['description']}")
            print(f"      Concepts: {', '.join(example['concepts'])}")
            print()
        
        print()
    
    print("=" * 80)
    print(f"Total: {sum(len(data['examples']) for data in EXAMPLES.values())} examples")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    print_examples()
