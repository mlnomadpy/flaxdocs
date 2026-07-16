---
sidebar_position: 0
title: Multimodal & Adaptation in Flax NNX
description: Parameter-efficient fine-tuning with LoRA and cross-modal models in Flax NNX — adapt frozen models with nnx.LoRALinear and train under 1% of parameters.
keywords: [LoRA, fine-tuning, PEFT, parameter-efficient, multimodal, CLIP, flax nnx, jax, nnx.LoRALinear]
---

# Multimodal & Adaptation

Training from scratch is expensive. This track is about **reusing and adapting**
models — attaching small trainable adapters to a frozen network, and aligning
models across different modalities (like images and text).

## What you'll build

- **[LoRA Fine-Tuning](/applications/adaptation/lora-finetuning)** — freeze a full
  model and train only tiny low-rank adapters (`nnx.LoRALinear`), updating well
  under 1% of the parameters. Uses `nnx.Optimizer(wrt=nnx.LoRAParam)` to scope
  training to just the adapters.
- **[CLIP (toy)](/applications/adaptation/clip)** — align an image encoder and a
  text encoder in a shared embedding space with a symmetric contrastive loss.

## Prerequisites

You should have built a [transformer](/basics/text/simple-transformer) (the model
we adapt) and understand [Flax NNX state and parameter filtering](/basics/fundamentals/understanding-state).

## Next steps

- Build [LoRA Fine-Tuning](/applications/adaptation/lora-finetuning).
- Contrast with [Knowledge Distillation](/research/knowledge-distillation), a
  different efficiency approach (compress into a smaller student).
