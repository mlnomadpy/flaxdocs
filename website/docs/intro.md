---
sidebar_position: 1
slug: /
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

This documentation is organized to progressively build your understanding:

### 🚀 Basics

Master the core concepts of Flax NNX:
- **Model Definition**: Understanding modules, parameters, and state
- **Data Loading**: Efficient pipelines with TFDS and Grain
- **Training Loops**: Writing effective, JIT-compiled training code
- **Checkpointing**: Saving and loading model state correctly
- **Best Practices**: Common patterns and anti-patterns

[Start Learning the Basics →](/docs/basics/model-definition)

### 📈 Scale

Take your training to production scale:
- **Distributed Training**: Understanding data and model parallelism
- **Performance Optimization**: Memory management and compute efficiency
- **Large Batch Training**: Scaling strategies and stability techniques
- **Multi-Host Setup**: Coordinating training across multiple machines

[Learn about Scaling →](/docs/scale/distributed-training)

### 🔬 Research

Advanced patterns for cutting-edge research:
- **Model Export**: Converting models to ONNX, SafeTensors, and HuggingFace format
- **Streaming Training**: Handling datasets larger than memory
- **Observability**: Tracking experiments with Weights & Biases
- **Advanced Architectures**: Building ResNets, Transformers, BERT, and GPT from scratch

[Explore Research Topics →](/docs/research/model-export)

## Reference Code

All documentation includes conceptual explanations with code snippets. For complete runnable examples, see the [`/examples`](https://github.com/mlnomadpy/flaxdocs/tree/master/examples) directory in the repository:

- 12 self-contained Python scripts covering all topics
- Each file is extensively commented for learning
- Use as reference when implementing your own training pipelines

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
