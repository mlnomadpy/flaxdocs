#!/bin/bash
# Quick start script for Flax NNX examples
# Run: bash quickstart.sh

set -e

echo "=========================================="
echo "Flax NNX Quick Start"
echo "=========================================="
echo ""

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not found"
    exit 1
fi

echo "✓ Python 3 found: $(python3 --version)"
echo ""

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "✓ Virtual environment created"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip -q

echo "✓ Pip upgraded"
echo ""

# Install core dependencies
echo "Installing core dependencies (this may take a few minutes)..."
pip install -q jax jaxlib flax optax orbax-checkpoint numpy

echo "✓ Core dependencies installed"
echo ""

# Install optional dependencies
read -p "Install optional dependencies? (datasets, transformers, wandb) [y/N]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Installing optional dependencies..."
    pip install -q tensorflow-datasets datasets transformers \
        safetensors onnx wandb pillow tiktoken
    echo "✓ Optional dependencies installed"
fi

echo ""
echo "=========================================="
echo "Installation Complete! 🎉"
echo "=========================================="
echo ""
echo "Quick Start Commands:"
echo ""
echo "  # Activate environment (if not already active)"
echo "  source venv/bin/activate"
echo ""
echo "  # Run basic model definition example"
echo "  python examples/basics/model_definition.py"
echo ""
echo "  # Run MNIST CNN training"
echo "  python examples/training/vision_mnist.py"
echo ""
echo "  # Run GPT training"
echo "  python examples/advanced/gpt_training.py"
echo ""
echo "See README.md for full list of examples!"
echo ""
