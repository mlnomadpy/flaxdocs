---
sidebar_position: 0
title: Computer Vision with Flax NNX
description: Build computer vision models in Flax NNX, from simple CNNs for image classification to ResNets with skip connections for deeper, stronger networks.
keywords: [computer vision, CNN, ResNet, image classification, Flax NNX, convolutional neural network, MNIST, skip connections]
---

# Computer Vision

Learn to build neural networks for visual tasks - from simple CNNs to advanced architectures like ResNet.

:::note Prerequisites
This guide builds on [Your First Model](/basics/fundamentals/your-first-model) and [Simple Training Loop](/basics/workflows/simple-training).
:::

:::tip What you'll learn
- When to reach for a CNN versus a ResNet for an image task
- How convolutions and pooling capture spatial patterns and translation invariance
- How skip connections let you train networks 10+ layers deep
- Build a SimpleCNN that hits 90%+ accuracy on MNIST
:::

## What You'll Build

This section teaches you computer vision models step-by-step:

**[Simple CNN](./simple-cnn.md)** - Start here!  
Build your first convolutional neural network for image classification. Learn convolutions, pooling, and how to train on MNIST.

**[ResNet Architecture](./resnet-architecture.md)**  
Go deeper with residual networks. Learn skip connections that enable training 50+ layer networks.

## When to Use These Models

### Use CNNs when:
- Working with images (classification, detection, segmentation)
- Need to capture spatial patterns
- Want translation invariance
- Have limited compute (CNNs are efficient)

### Use ResNets when:
- Need deeper networks (10+ layers)
- Want state-of-the-art image classification
- Building on pretrained models (transfer learning)
- Need strong feature extractors

## Prerequisites

Before diving into vision models, make sure you understand:
- [Your First Model](../fundamentals/your-first-model.md) - Basic NNX concepts
- [Simple Training Loop](../workflows/simple-training.md) - How to train models

## Quick Example

Here's a simple CNN you'll build:

```python
from flax import nnx

class SimpleCNN(nnx.Module):
    def __init__(self, num_classes: int, *, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(1, 32, (3, 3), rngs=rngs)
        self.conv2 = nnx.Conv(32, 64, (3, 3), rngs=rngs)
        self.dense = nnx.Linear(64 * 5 * 5, num_classes, rngs=rngs)
    
    def __call__(self, x):
        # Conv blocks
        x = self.conv1(x)
        x = nnx.relu(x)
        x = nnx.max_pool(x, (2, 2), (2, 2))
        
        x = self.conv2(x)
        x = nnx.relu(x)
        x = nnx.max_pool(x, (2, 2), (2, 2))
        
        # Classifier
        x = x.reshape(x.shape[0], -1)
        return self.dense(x)

# 90%+ accuracy on MNIST with this simple model!
```

## Common Vision Tasks

- **Image Classification**: "Is this a cat or dog?"
- **Object Detection**: "Where are the objects in this image?"
- **Semantic Segmentation**: "Label every pixel"
- **Transfer Learning**: Use pretrained models on new tasks

This section focuses on classification, which is the foundation for all other tasks.

## Next steps

- [Simple CNN](/basics/vision/simple-cnn) - Build your first convolutional classifier
- [ResNet Architecture](/basics/vision/resnet-architecture) - Go deeper with skip connections
- [Streaming Data](/basics/workflows/streaming-data) - Train on datasets larger than memory

## Complete Examples

- [`examples/training/vision_mnist.py`](https://github.com/mlnomadpy/flaxdocs/tree/master/examples/training/vision_mnist.py) - Complete MNIST training
- [`examples/integrations/resnet_streaming.py`](https://github.com/mlnomadpy/flaxdocs/tree/master/examples/integrations/resnet_streaming.py) - ResNet with streaming data
