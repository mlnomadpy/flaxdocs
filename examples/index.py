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
            },
            {
                "name": "DQN Reinforcement Learning",
                "file": "advanced/dqn_reinforcement_learning.py",
                "description": "Deep Q-Network reinforcement learning",
                "concepts": ["dqn", "reinforcement learning", "replay buffer"]
            },
            {
                "name": "Metric Learning (Siamese/Triplet)",
                "file": "advanced/metric_learning.py",
                "description": "Learn an embedding where same-class inputs cluster (triplet loss)",
                "concepts": ["metric learning", "triplet loss", "siamese", "mining"]
            },
            {
                "name": "Interpretability & Saliency",
                "file": "advanced/interpretability.py",
                "description": "Saliency, Integrated Gradients, and Grad-CAM via jax.grad on inputs",
                "concepts": ["saliency", "integrated gradients", "grad-cam", "attribution"]
            },
            {
                "name": "Uncertainty Estimation",
                "file": "advanced/uncertainty.py",
                "description": "MC-dropout and deep ensembles for predictive uncertainty",
                "concepts": ["mc-dropout", "deep ensembles", "nnx.vmap", "calibration"]
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
    },
    "ImageNet": {
        "description": "Standalone ImageNet training application",
        "examples": [
            {
                "name": "ImageNet Training",
                "file": "imagenet/main.py",
                "description": "Standalone ResNet ImageNet training app",
                "concepts": ["imagenet", "resnet", "large-scale training"]
            }
        ]
    },
    "Generative": {
        "description": "Generative models — learn to sample data, not just classify it",
        "examples": [
            {
                "name": "Autoencoder (+ Denoising)",
                "file": "generative/autoencoder.py",
                "description": "Compress and reconstruct images through a bottleneck",
                "concepts": ["nnx.ConvTranspose", "ConvEncoder/ConvDecoder", "reconstruction loss"]
            },
            {
                "name": "Variational Autoencoder (VAE)",
                "file": "generative/vae.py",
                "description": "Probabilistic latent + reparameterization; sample new digits",
                "concepts": ["reparameterization", "ELBO", "nnx.Rngs noise stream"]
            },
            {
                "name": "DCGAN",
                "file": "generative/dcgan.py",
                "description": "Generator vs discriminator with two optimizers + spectral norm",
                "concepts": ["adversarial training", "nnx.SpectralNorm", "two optimizers"]
            },
            {
                "name": "Diffusion Model (DDPM)",
                "file": "generative/ddpm.py",
                "description": "Iterative denoising with a small time-conditioned U-Net",
                "concepts": ["diffusion", "nnx.Embed timestep", "nnx.GroupNorm", "sampling loop"]
            },
            {
                "name": "Normalizing Flows (RealNVP)",
                "file": "generative/normalizing_flows.py",
                "description": "Invertible network with exact likelihood via change of variables",
                "concepts": ["RealNVP", "affine coupling", "log-det Jacobian"]
            }
        ]
    },
    "Sequence": {
        "description": "Sequence models — recurrence and order",
        "examples": [
            {
                "name": "Recurrent Networks (RNN/LSTM/GRU)",
                "file": "sequence/rnn_cells.py",
                "description": "The nnx.RNN API family + manual nnx.scan on a parity task",
                "concepts": ["nnx.RNN", "nnx.LSTMCell/GRUCell", "nnx.Bidirectional", "nnx.scan"]
            },
            {
                "name": "Seq2Seq with Attention",
                "file": "sequence/seq2seq_attention.py",
                "description": "Encoder-decoder with cross-attention on a copy/reverse task",
                "concepts": ["seq2seq", "cross-attention", "nnx.MultiHeadAttention", "teacher forcing"]
            },
            {
                "name": "Time-Series Forecasting",
                "file": "sequence/time_series.py",
                "description": "LSTM multi-step forecasting on synthetic sinusoids",
                "concepts": ["forecasting", "sliding windows", "nnx.LSTMCell"]
            },
            {
                "name": "Word2Vec (skip-gram)",
                "file": "sequence/word2vec.py",
                "description": "Learn word embeddings with skip-gram + negative sampling",
                "concepts": ["nnx.Embed", "negative sampling", "embeddings"]
            }
        ]
    },
    "Scientific": {
        "description": "Graphs, scientific ML, and structured data",
        "examples": [
            {
                "name": "Graph Neural Network (GCN)",
                "file": "scientific/gcn_karate.py",
                "description": "Semi-supervised node classification on Zachary's Karate Club",
                "concepts": ["message passing", "normalized adjacency", "nnx.Einsum"]
            },
            {
                "name": "Physics-Informed NN (PINN)",
                "file": "scientific/pinn_oscillator.py",
                "description": "Solve an ODE by differentiating the model output w.r.t. its input",
                "concepts": ["jax.grad on input", "residual loss", "scientific ML"]
            },
            {
                "name": "Neural ODE",
                "file": "scientific/neural_ode.py",
                "description": "Continuous-depth model integrated through a differentiable RK4 solver",
                "concepts": ["neural ODE", "jax.lax.scan solver", "continuous depth"]
            },
            {
                "name": "Tabular Deep Learning",
                "file": "scientific/tabular_dnn.py",
                "description": "MLP with categorical embeddings for structured data",
                "concepts": ["nnx.Embed categoricals", "tabular", "regression/classification"]
            },
            {
                "name": "Mixture of Experts (MoE)",
                "file": "scientific/moe.py",
                "description": "Sparse top-k expert routing with a load-balancing loss",
                "concepts": ["MoE", "top-k gating", "load balancing", "nnx.Einsum"]
            }
        ]
    },
    "Vision": {
        "description": "Vision beyond CNNs — attention and dense prediction",
        "examples": [
            {
                "name": "Vision Transformer (ViT)",
                "file": "vision/vit.py",
                "description": "Patch embedding + transformer encoder for image classification",
                "concepts": ["ViT", "patch embedding", "nnx.MultiHeadAttention", "CLS token"]
            },
            {
                "name": "U-Net Segmentation",
                "file": "vision/unet_segmentation.py",
                "description": "Encoder-decoder with skip connections for per-pixel masks",
                "concepts": ["U-Net", "nnx.ConvTranspose", "skip connections", "segmentation"]
            }
        ]
    },
    "Adaptation": {
        "description": "Fine-tuning and adaptation of existing models",
        "examples": [
            {
                "name": "LoRA Fine-Tuning",
                "file": "adaptation/lora_finetuning.py",
                "description": "Freeze a model, train only low-rank adapters",
                "concepts": ["nnx.LoRALinear", "nnx.Optimizer(wrt=nnx.LoRAParam)", "nnx.DiffState"]
            },
            {
                "name": "CLIP (toy)",
                "file": "adaptation/clip_toy.py",
                "description": "Align image and text encoders with a symmetric contrastive loss",
                "concepts": ["CLIP", "cross-modal", "contrastive", "dual encoders"]
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
