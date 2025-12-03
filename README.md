# Learn Flax & JAX: Neural Network Training Guide

A comprehensive learning resource for anyone interested in training neural networks with Flax and JAX. Whether you're just getting started or looking to master advanced techniques, this guide has you covered! 🚀

[![Deploy to GitHub Pages](https://github.com/mlnomadpy/flaxdocs/actions/workflows/deploy.yml/badge.svg)](https://github.com/mlnomadpy/flaxdocs/actions/workflows/deploy.yml)

## 🎓 Who Is This For?

- **Beginners** wanting to learn Flax NNX from scratch
- **ML Engineers** transitioning from PyTorch or TensorFlow
- **Researchers** exploring functional programming approaches to deep learning
- **Students** building practical skills in modern neural network frameworks

## 🚀 Quick Start with Runnable Examples

**NEW!** Check out our [**complete runnable examples**](./examples/) for Flax NNX:

- 🎯 **Basics**: Model definition, saving/loading, data loading
- 🏃 **Training**: End-to-end MNIST CNN, Transformer LM
- 📦 **Export**: SafeTensors, ONNX formats
- 🤗 **HuggingFace**: Model upload, streaming datasets
- 🔬 **Advanced**: ResNet streaming, BERT, GPT training
- 📊 **Observability**: Weights & Biases integration

Each guide is a complete, self-contained Python file you can run immediately!

👉 **[View All Examples →](./examples/)**

## 🌟 What You'll Learn

This guide takes you from zero to hero with Flax NNX for training neural networks:

### 🚀 Basics
- **Getting Started**: Learn to set up your environment and create your first Flax NNX model
- **Training Best Practices**: Master learning rate scheduling, gradient clipping, and regularization
- **Model Checkpointing**: Understand how to save and restore your models with Orbax

### 📈 Scale
- **Distributed Training**: Learn to scale training across multiple GPUs/TPUs with data and model parallelism
- **Performance Optimization**: Discover mixed precision training, gradient accumulation, and optimization tricks
- **Multi-Host Training**: Master distributed training across multiple machines

### 🔬 Research
- **Advanced Techniques**: Explore contrastive learning, meta-learning, NAS, and adversarial training
- **Custom Training Loops**: Build flexible, custom training loops for research
- **Experiment Tracking**: Learn reproducible research practices with W&B integration

## 🚀 Getting Started with This Guide

### Prerequisites
- Node.js 20.0 or higher
- npm or yarn

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/mlnomadpy/flaxdocs.git
   cd flaxdocs/website
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Start the development server**
   ```bash
   npm start
   ```
   
   This opens the guide in your browser. Most changes are reflected live without restarting the server.

4. **Build the website**
   ```bash
   npm run build
   ```
   
   This generates static content into the `build` directory.

## 📚 Learning Path

New to Flax/JAX? We recommend following this learning path:

1. **Start with Basics**: Read through the Getting Started guide
2. **Practice Training**: Work through Training Best Practices examples
3. **Save Your Progress**: Learn Checkpointing techniques
4. **Scale Up**: Move to Distributed Training when you're ready
5. **Go Advanced**: Explore Research techniques for cutting-edge methods

## 📝 Contributing

We welcome contributions! Help us make this learning resource even better:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-guide`
3. **Make your changes** in the `website/docs` directory
4. **Test locally**: Run `npm start` in the `website` directory
5. **Build to verify**: Run `npm run build` to ensure everything works
6. **Commit your changes**: `git commit -m 'Add amazing guide'`
7. **Push to your fork**: `git push origin feature/amazing-guide`
8. **Open a Pull Request**

### Documentation Structure

```
website/
├── docs/
│   ├── intro.md                          # Landing page
│   ├── basics/                           # Basic training guides
│   │   ├── getting-started.md
│   │   ├── training-best-practices.md
│   │   └── checkpointing.md
│   ├── scale/                            # Scaling guides
│   │   └── distributed-training.md
│   └── research/                         # Advanced research techniques
│       └── advanced-techniques.md
├── blog/                                 # Blog posts
├── src/                                  # React components
└── static/                               # Static assets
```

### Writing Guidelines

- Use clear, beginner-friendly language
- Include practical code examples with detailed explanations
- Add links to relevant learning resources
- Test all code snippets before submitting
- Focus on teaching concepts, not just showing code

## 🔧 Built With

- [Docusaurus](https://docusaurus.io/) - Documentation framework
- [TypeScript](https://www.typescriptlang.org/) - Type safety
- [GitHub Pages](https://pages.github.com/) - Hosting
- [GitHub Actions](https://github.com/features/actions) - CI/CD

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Acknowledgments

- [Flax Team](https://github.com/google/flax) - For creating an amazing functional neural network library
- [JAX Team](https://github.com/google/jax) - For the powerful numerical computing framework
- All contributors and learners who help improve this guide

## 📞 Questions & Support

- **Issues**: Report bugs or request new topics in [GitHub Issues](https://github.com/mlnomadpy/flaxdocs/issues)
- **Discussions**: Ask questions and share your learning in [GitHub Discussions](https://github.com/mlnomadpy/flaxdocs/discussions)
- **Official Docs**: Supplement your learning with [Flax official documentation](https://flax.readthedocs.io/)

## 🌐 Accessing the Guide

The guide is automatically deployed and available online:

- **Live Guide**: https://mlnomadpy.github.io/flaxdocs/
- **CI/CD**: Automated via GitHub Actions (see `.github/workflows/deploy.yml`)

### Manual Deployment

For contributors who need to deploy manually:

```bash
cd website
npm run build
# The build output is in website/build/
# Can be deployed to any static hosting service
```

---

**Made with ❤️ for learners exploring Flax and JAX**

