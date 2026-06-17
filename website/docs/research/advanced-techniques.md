---
sidebar_position: 1
title: Advanced Research Techniques in JAX
description: "Index of advanced JAX and Flax NNX research techniques — custom training loops, contrastive learning, MAML, distillation, NAS, adversarial training, and RL."
keywords: [JAX research, Flax NNX, custom training loops, meta-learning, knowledge distillation, neural architecture search, adversarial training, reinforcement learning]
---

# Advanced Research Techniques

The Research section collects advanced techniques that go beyond standard supervised training. Each page is a research-grade, from-scratch implementation in JAX and Flax NNX — with the math, runnable code, and the design decisions that matter. These guides assume you already know how to train a model; they focus on the *non-standard* ideas layered on top.

:::note Prerequisites
These pages move fast and assume solid fundamentals. Make sure you are comfortable with a [simple training loop](/basics/workflows/simple-training) and the [training best practices](/basics/training-best-practices) before diving in.
:::

## Start here

New to this section? Begin with [Custom Training Loops](/research/custom-training-loops). Almost every technique below builds on the explicit `TrainState` and custom training-step patterns it introduces.

## Representation and model efficiency

Learn powerful representations and shrink models without losing accuracy.

- **[Contrastive Learning](/research/contrastive-learning)** — SimCLR self-supervised representation learning: the NT-Xent loss, augmentation pipeline, and linear-probe evaluation, all without labels.
- **[Knowledge Distillation](/research/knowledge-distillation)** — Train small, fast student models from large teachers using temperature-scaled soft targets.

## Learning paradigms

Move beyond a single fixed dataset and objective.

- **[Meta-Learning](/research/meta-learning)** — MAML / learning-to-learn: find an initialization that adapts to new tasks in a few gradient steps.
- **[Reinforcement Learning](/research/reinforcement-learning)** — Deep Q-Networks (DQN) in JAX with experience replay, target networks, and JAX-native gymnax environments.
- **[Curriculum Learning](/research/curriculum-learning)** — Easy-to-hard training schedules with difficulty scoring, pacing functions, and self-paced learning.

## Robustness and rigor

Make models trustworthy and results reproducible.

- **[Adversarial Training](/research/adversarial-training)** — Build robustness against adversarial examples using FGSM and the min-max robust optimization objective.
- **[Experiment Reproducibility](/research/experiment-reproducibility)** — Deterministic, reproducible runs via explicit PRNG keys, config management, and XLA determinism.

## Search and control

Take fine-grained control of the architecture and the training step itself.

- **[Custom Training Loops](/research/custom-training-loops)** — Fine-grained control over the training step: extend `TrainState`, add gradient clipping, EMA, and auxiliary losses.
- **[Neural Architecture Search](/research/neural-architecture-search)** — DARTS / automating architecture design via continuous relaxation and bilevel optimization.

