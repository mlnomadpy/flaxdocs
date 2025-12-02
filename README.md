# Flax Training Documentation

Comprehensive guides for training neural networks with Flax - because the official docs aren't cutting it! 🚀

[![Deploy to GitHub Pages](https://github.com/mlnomadpy/flaxdocs/actions/workflows/deploy.yml/badge.svg)](https://github.com/mlnomadpy/flaxdocs/actions/workflows/deploy.yml)

## 🌟 What's Inside

This documentation covers everything you need to know about training models with Flax:

### 🚀 Basics
- **Getting Started**: Set up your environment and create your first Flax model
- **Training Best Practices**: Learning rate scheduling, gradient clipping, regularization
- **Model Checkpointing**: Save and restore your models effectively

### 📈 Scale
- **Distributed Training**: Scale across multiple GPUs/TPUs with data and model parallelism
- **Performance Optimization**: Mixed precision, gradient accumulation, and more
- **Multi-Host Training**: Configure and run training across multiple machines

### 🔬 Research
- **Advanced Techniques**: Contrastive learning, meta-learning, NAS, adversarial training
- **Custom Training Loops**: Build flexible training loops for research
- **Experiment Tracking**: Reproducible research practices

## 🚀 Quick Start

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
   
   This command starts a local development server and opens up a browser window. Most changes are reflected live without having to restart the server.

4. **Build the website**
   ```bash
   npm run build
   ```
   
   This command generates static content into the `build` directory and can be served using any static contents hosting service.

## 📝 Contributing

We welcome contributions! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-guide`
3. **Make your changes** in the `website/docs` directory
4. **Test locally**: Run `npm start` in the `website` directory
5. **Build to verify**: Run `npm run build` to ensure no broken links
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

- Use clear, concise language
- Include code examples with explanations
- Add links to relevant resources
- Test all code snippets before submitting
- Follow the existing documentation style

## 🔧 Built With

- [Docusaurus](https://docusaurus.io/) - Documentation framework
- [TypeScript](https://www.typescriptlang.org/) - Type safety
- [GitHub Pages](https://pages.github.com/) - Hosting
- [GitHub Actions](https://github.com/features/actions) - CI/CD

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Acknowledgments

- [Flax Team](https://github.com/google/flax) - For creating an amazing neural network library
- [JAX Team](https://github.com/google/jax) - For the foundation Flax is built on
- All contributors who help improve this documentation

## 📞 Contact & Support

- **Issues**: Report bugs or request features in [GitHub Issues](https://github.com/mlnomadpy/flaxdocs/issues)
- **Discussions**: Join conversations in [GitHub Discussions](https://github.com/mlnomadpy/flaxdocs/discussions)
- **Official Flax**: Check out [Flax official documentation](https://flax.readthedocs.io/)

## 🌐 Deployment

The documentation is automatically deployed to GitHub Pages when changes are pushed to the `main` branch:

- **Live Site**: https://mlnomadpy.github.io/flaxdocs/
- **CI/CD**: Automated via GitHub Actions (see `.github/workflows/deploy.yml`)

### Manual Deployment

If you need to deploy manually:

```bash
cd website
npm run build
# The build output is in website/build/
# Can be deployed to any static hosting service
```

---

**Made with ❤️ for the Flax community**

