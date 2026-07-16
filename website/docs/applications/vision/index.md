---
sidebar_position: 0
title: Advanced Vision in Flax NNX
description: Build a Vision Transformer (ViT) and a U-Net for segmentation in Flax NNX — vision beyond CNNs, with attention and dense per-pixel prediction.
keywords: [vision transformer, ViT, U-Net, semantic segmentation, flax nnx, jax, attention, dense prediction]
---

# Advanced Vision

CNNs and ResNets are the workhorses of image classification, but they aren't the
whole story. This track covers two architectures that go beyond the convolutional
backbone: the **Vision Transformer**, which brings attention to images, and the
**U-Net**, which produces a full-resolution output for dense prediction.

## What you'll build

- **[Vision Transformer (ViT)](/applications/vision/vision-transformer)** —
  split an image into patches, embed them as tokens, and classify with a
  transformer encoder. A global receptive field from the very first layer, using
  the built-in `nnx.MultiHeadAttention`.
- **[U-Net Segmentation](/applications/vision/unet-segmentation)** — an
  encoder-decoder with skip connections that labels *every pixel*, using
  `nnx.ConvTranspose` to upsample back to full resolution.

## Prerequisites

You should have built a [CNN](/basics/vision/simple-cnn) and understand
[transformers](/basics/text/simple-transformer) (ViT reuses the encoder) and
[ResNet skip connections](/basics/vision/resnet-architecture) (U-Net generalizes them).

## Next steps

- Start with the [Vision Transformer](/applications/vision/vision-transformer).
- The [U-Net](/applications/vision/unet-segmentation) is also the denoiser
  backbone behind [diffusion models](/applications/generative/diffusion).
