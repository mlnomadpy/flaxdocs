---
sidebar_position: 1
slug: /
title: Train Neural Networks with Flax & JAX
description: Comprehensive guide to training neural networks with Flax NNX and JAX. Learn distributed training, model optimization, and production-ready ML workflows.
keywords: [Flax, JAX, NNX, neural networks, machine learning, deep learning, training, distributed training, TPU, GPU, tutorial, guide]
image: /img/docusaurus-social-card.jpg
---

# Welcome to Learn Flax NNX Training

Your comprehensive guide to mastering neural network training with **Flax NNX** and **JAX** from the ground up! 🚀

## What is Flax NNX?

Flax NNX is the new neural network API built on JAX, combining the best of functional and object-oriented programming. It provides a flexible, high-performance framework for machine learning research and production:

- **Pythonic & Intuitive**: Easy to learn, familiar OOP style with explicit state management
- **Explicit RNGs**: No hidden randomness, full control over reproducibility
- **JIT Compilation**: Lightning-fast execution with JAX's XLA compiler
- **Automatic Differentiation**: Effortless gradient computation for any function
- **Scalability**: Easy scaling from single GPU to TPU pods with minimal code changes
- **Production Ready**: Used in real-world systems at Google and beyond

## Why Learn Flax NNX?

Unlike older neural network libraries that hide state management and randomness, Flax NNX gives you explicit control over every aspect of your models. This makes debugging easier, scaling simpler, and helps you truly understand what's happening in your training loop.

This guide will teach you the **concepts and patterns** behind Flax NNX training, not just show you code to copy. You'll learn:

- **How** Flax NNX manages model state and parameters
- **Why** explicit RNG handling makes your training reproducible
- **When** to use different optimization patterns
- **What** makes a good training loop architecture

## Documentation Structure

This documentation is organized into small, focused guides that won't overwhelm you:

### 🎯 Fundamentals

Start with the core concepts that apply everywhere:
- **Your First Model**: Build a simple neural network from scratch
- **Understanding State**: How NNX manages parameters and variables

These fundamentals take ~15 minutes and are essential for everything else.

[Start with Fundamentals →](/basics/fundamentals)

### 🏃 Training Workflows

Learn the practical skills to train models:
- **Simple Training Loop**: Write your first complete training loop
- **Data Loading**: Build efficient data pipelines without bottlenecks

Short, focused guides that get you training quickly.

[Learn Training Workflows →](/basics/workflows)

### 🖼️ Computer Vision

Build image models step-by-step:
- **Simple CNN**: Your first convolutional network for image classification
- **ResNet**: Deep networks with skip connections

Each guide is self-contained and builds one complete model.

[Explore Computer Vision →](/basics/vision)

### 📝 Natural Language Processing

Build text models from scratch:
- **Simple Transformer**: Understand attention and build GPT-style models

Clear explanations of how transformers actually work.

[Explore NLP →](/basics/text)

### 📈 Scale

Take your training to production scale:
- **Distributed Training**: Multiple GPUs and TPUs
- **Performance Optimization**: Make training faster

[Learn about Scaling →](/scale/)

### 🔬 Research

Advanced patterns for cutting-edge research:
- **Model Export**: ONNX, SafeTensors, HuggingFace formats
- **Observability**: Track experiments with W&B
- **Advanced Architectures**: Building ResNets, Transformers, BERT, and GPT from scratch

[Explore Research Topics →](/research/advanced-techniques)

### 🧩 Applications

Small, self-contained implementations across model families and domains:
- **Generative**: autoencoders, VAEs, GANs, and diffusion models
- **Sequence**: recurrent networks (RNN/LSTM/GRU) and time series
- **Scientific**: graph neural networks and physics-informed networks (PINNs)
- **Adaptation**: parameter-efficient fine-tuning with LoRA

[Explore Applications →](/applications)

## How to Use This Documentation

### If you're brand new:
1. Start with [Fundamentals →](/basics/fundamentals) (~15 min)
2. Learn [Training Workflows →](/basics/workflows) (~20 min)
3. Choose your domain: [Vision](/basics/vision) or [Text](/basics/text)

### If you know the basics:
- Jump directly to [Computer Vision](/basics/vision) or [NLP](/basics/text)
- Each guide is self-contained and buildable in isolation

### If you need specific examples:
- See the [`/examples`](https://github.com/mlnomadpy/flaxdocs/tree/master/examples) directory
- **21 complete, organized examples** covering all topics:
  - **Basics** (5 examples): Model definition, checkpointing, data loading
  - **Training** (2 examples): Vision and language model training
  - **Export** (1 example): Model deployment formats
  - **Integrations** (3 examples): HuggingFace, W&B, streaming data
  - **Advanced** (5 examples): BERT, GPT, SimCLR, MAML, distillation
  - **Distributed** (4 examples): Multi-device training strategies
- All examples use **shared, tested components** for consistency
- View the complete index: `python examples/index.py`

## What Makes This Different?

**Small, focused guides**: Each page teaches ONE concept completely. No 5000-word mega-guides.

**Domain-organized**: Vision models in vision/, text models in text/. Find what you need quickly.

**Example-driven**: Every concept has working code you can run immediately.

**No overwhelm**: Start simple, build up gradually. You won't drown in complexity.

## Reference Code

All documentation includes conceptual explanations with code snippets. For complete runnable examples, see the [`/examples`](https://github.com/mlnomadpy/flaxdocs/tree/master/examples) directory in the repository:

- **21 organized examples** in modular structure:
  - `basics/` - Model definition, checkpointing, data loading
  - `training/` - End-to-end vision and language model training
  - `export/` - Model deployment (SafeTensors, ONNX)
  - `integrations/` - HuggingFace Hub, W&B, streaming datasets
  - `advanced/` - BERT, GPT, SimCLR, MAML, knowledge distillation
  - `distributed/` - pmap, SPMD, pipeline parallelism, FSDP
- **Shared component library** (`shared/models.py`, `shared/training_utils.py`) with tested, reusable code
- Each file is extensively commented for learning
- Run `python examples/index.py` to see all available examples with descriptions

## Getting Help

- **GitHub Issues**: Report bugs or request features in our [GitHub repository](https://github.com/mlnomadpy/flaxdocs)
- **Flax Official Docs**: Check out the [official Flax documentation](https://flax.readthedocs.io/)
- **JAX Documentation**: Learn more about [JAX](https://jax.readthedocs.io/)

## Contributing

We welcome contributions! If you'd like to improve this documentation:

1. Fork the [repository](https://github.com/mlnomadpy/flaxdocs)
2. Make your changes
3. Submit a pull request

Happy training with Flax! 🎉
