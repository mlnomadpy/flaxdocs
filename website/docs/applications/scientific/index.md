---
sidebar_position: 0
title: Graphs, Scientific & Structured Data in Flax NNX
description: Build graph neural networks and physics-informed neural networks (PINNs) in Flax NNX — new data modalities and differentiating through the model with JAX autodiff.
keywords: [graph neural network, GCN, physics-informed neural network, PINN, scientific machine learning, flax nnx, jax autodiff]
---

# Graphs, Scientific & Structured

Grids (images) and sequences (text) aren't the only structures worth modeling.
This track ventures into **relational data (graphs)** and **scientific ML**,
where JAX's real superpower — differentiating arbitrary functions — takes center
stage.

## What you'll build

- **[Graph Neural Networks (GCN)](/applications/scientific/graph-neural-networks)** —
  message passing over a graph for semi-supervised node classification, using
  `nnx.Einsum` for adjacency-weighted aggregation.
- **[Physics-Informed Neural Networks (PINN)](/applications/scientific/pinn)** —
  solve a differential equation by baking it into the loss, differentiating the
  network's *output with respect to its input* via `jax.grad`. No dataset needed.
- **[Neural ODEs](/applications/scientific/neural-ode)** — continuous-depth models;
  learn the *dynamics* and integrate them through a differentiable ODE solver.
- **[Tabular Deep Learning](/applications/scientific/tabular)** — MLPs with
  categorical embeddings for structured/tabular data (classification and regression).
- **[Mixture of Experts (MoE)](/applications/scientific/mixture-of-experts)** —
  sparse conditional computation with top-k routing and a load-balancing loss.

## Prerequisites

You should be comfortable with the [training loop](/basics/workflows/simple-training)
and [custom training loops](/research/custom-training-loops). PINNs assume basic
calculus (derivatives, differential equations).

## Next steps

- Start with [Graph Neural Networks](/applications/scientific/graph-neural-networks).
- Then try [Physics-Informed Neural Networks](/applications/scientific/pinn).
