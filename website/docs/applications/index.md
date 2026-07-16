---
sidebar_position: 0
title: Flax NNX Applications — Build Real Models Across Domains
description: A tour of small, self-contained Flax NNX implementations across generative models, sequence models, graphs, scientific ML, vision, and fine-tuning.
keywords: [flax nnx applications, jax examples, generative models, rnn, graph neural network, pinn, lora, vision transformer, flax tutorials]
---

# Applications

You've learned the fundamentals, seen the canonical architectures, and scaled a
training run. This section is where it all pays off: **small, self-contained
implementations of real model families you can build with Flax NNX** — each one
runnable, each one teaching a distinct idea.

Every guide here follows the same recipe: the math in a few lines, a from-scratch
model in idiomatic NNX, a training step, and results you can reproduce. Datasets
are tiny (MNIST, synthetic, or built-in) so everything runs on a laptop.

## The domains

### 🎨 [Generative Models](/applications/generative)
Learn to *generate* data, not just classify it: autoencoders, VAEs, GANs, and
diffusion models. Showcases `nnx.ConvTranspose` for learned upsampling.

### 🔁 [Sequence Models & Time Series](/applications/sequence)
Recurrence and order: RNNs, LSTMs, and GRUs via the `nnx.RNN` API family, plus
sequence-to-sequence and forecasting.

### 🔬 [Graphs, Scientific & Structured](/applications/scientific)
Beyond grids and sequences: graph neural networks, physics-informed networks
(PINNs) that differentiate *through the model*, and structured/tabular data.

### 🧬 [Multimodal & Adaptation](/applications/adaptation)
Adapt and combine models: parameter-efficient fine-tuning with LoRA
(`nnx.LoRALinear`) and cross-modal alignment.

## Prerequisites

These guides assume you're comfortable with the [training loop](/basics/workflows/simple-training)
and [Flax NNX state](/basics/fundamentals/understanding-state). Individual guides
link the specific extra background they need.

## How to use this section

Each sub-category has a suggested reading order that builds one idea at a time
(e.g. Autoencoder → VAE → GAN → Diffusion). Jump straight to a domain that
interests you, or work through a whole track.

## Next steps

- Start with [Generative Models](/applications/generative) — the most popular track.
- Or explore [Sequence Models](/applications/sequence) if you work with text or time series.
- Need to scale one of these up? See [Distributed Training](/scale/).
