---
sidebar_position: 0
title: Sequence Models & Time Series in Flax NNX
description: Build recurrent networks (RNN, LSTM, GRU) and sequence models in Flax NNX using the nnx.RNN API family and nnx.scan.
keywords: [rnn, lstm, gru, sequence model, time series, flax nnx, jax, nnx.RNN, nnx.scan, recurrent]
---

# Sequence Models & Time Series

Not all data lives on a grid. Text, audio, and time series are **ordered
sequences** where what came before shapes what comes next. This track covers the
recurrent model family — the pre-transformer workhorses that are still the right
tool for many small sequence tasks.

## What you'll build

- **[Recurrent Networks (RNN / LSTM / GRU)](/applications/sequence/recurrent-networks)** —
  the carry-state primitive and the full `nnx.RNN`, `nnx.LSTMCell`, `nnx.GRUCell`,
  and `nnx.Bidirectional` API family, plus the manual `nnx.scan` view under the hood.

More guides in this track (seq2seq with attention, time-series forecasting, and
learned word embeddings) are on the way.

## Prerequisites

You should have built [your first model](/basics/fundamentals/your-first-model)
and understand [Flax NNX state](/basics/fundamentals/understanding-state) —
recurrence is all about threading state (the *carry*) through time.

## Next steps

- Build [Recurrent Networks](/applications/sequence/recurrent-networks).
- Compare with the attention-based [Transformer](/basics/text/simple-transformer).
