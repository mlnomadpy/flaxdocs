---
sidebar_position: 0
title: Generative Models in Flax NNX
description: Build autoencoders, VAEs, GANs, and diffusion models from scratch in Flax NNX — learn to generate images with nnx.ConvTranspose and custom losses.
keywords: [generative models, autoencoder, VAE, GAN, diffusion, DDPM, flax nnx, jax, ConvTranspose]
---

# Generative Models

Generative models learn the *distribution* of data so they can produce new
samples — new digits, new images — rather than just labeling existing ones. This
track builds four families from scratch, each adding one new idea about how to
model that distribution.

## What you'll build

- **[Autoencoders](/applications/generative/autoencoder)** — compress and
  reconstruct images through a bottleneck. Introduces `nnx.ConvTranspose` for
  learned upsampling (the decoder primitive every later guide reuses).
- **[Variational Autoencoders (VAE)](/applications/generative/vae)** — put a
  *probability distribution* over the latent space and sample new digits, using
  the reparameterization trick and the ELBO objective.
- **[Generative Adversarial Networks (GAN)](/applications/generative/gan)** — a
  generator and discriminator locked in a game; learn density *implicitly* with
  two optimizers and spectral normalization.
- **[Diffusion Models (DDPM)](/applications/generative/diffusion)** — generate by
  iteratively denoising pure noise; a small U-Net predicts the noise at each step.
- **[Normalizing Flows](/applications/generative/normalizing-flows)** — an
  *invertible* network that gives you the exact likelihood via the change-of-variables
  formula (RealNVP coupling layers).

## Suggested order

Read top to bottom: each guide builds on the previous one's intuition
(reconstruction → latent distribution → adversarial → iterative denoising).

## Prerequisites

You should be comfortable with [CNNs](/basics/vision/simple-cnn) and the
[training loop](/basics/workflows/simple-training). VAE and beyond lean on
[Flax NNX state and RNGs](/basics/fundamentals/understanding-state).

## Next steps

- Start with the [Autoencoder](/applications/generative/autoencoder).
- Compare with self-supervised [Contrastive Learning](/research/contrastive-learning).
