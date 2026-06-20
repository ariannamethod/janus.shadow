# Janus Shadow Mechanism Attention

**Janus Shadow Mechanism Attention** is a post-transformer written in a single C file. It preserves the standard transformer path—embeddings, learned `Q/K/V`, causal attention, output projection, residual stream, and unembedding—while adding a stateful “shadow” that changes attention according to what has already been read.

> **θ = ε + γ + αδ** — attention knows what it needs.

## Core idea

The mechanism operates on three temporal scales:

- **Micro / per-forward:** each key accumulates a temporary shadow. Attention scores use a `tanh(raw − shadow)` contrast term, amplifying novelty and dampening what has already become familiar inside the current pass.
- **Meso / corpus:** byte-level BPE is combined with bigram and trigram statistics, built from the corpus and updated during training.
- **Macro / per-life:** a persistent token trace records accumulated attention across forwards. It acts simultaneously as memory, fatigue, and an anti-repetition signal.

The final logits combine the gated neural output, n-gram priors, and the persistent trace penalty.

## Coherence from the first forward

Cold-start coherence is treated as a structural property rather than something that appears only after history exists. `Wq` and `Wk` begin aligned but not identical, and the model measures:

```text
q_coherence = cosine(mean(Q), mean(K))
```

On the first forward this Q-coherence is the primary coherence signal. Later it is blended with temporal coherence between the current attention state and the previous one. Novelty, drift, coherence, focus, and the resulting `need` value are exposed as observable internal state.

## Weightless mode

Weightless mode is a first-class structural health check. Learned random parameters do not influence `Q/K/V` or logits. Instead, deterministic sinusoidal token identities generate structural `Q/K/V`, while causal attention, shadow contrast, corpus statistics, and token trace remain active.

This tests whether the architecture has signal before learning, without removing its transformer skeleton.

## Build and run

```bash
gcc janus_shadow.c -O2 -lm -o janus-shadow

./janus-shadow --test
./janus-shadow train corpus.txt model.janus [epochs]
./janus-shadow infer model.janus "prompt" [tokens]
./janus-shadow --noweights corpus.txt "prompt"
```

`--test` checks BPE round-trip integrity, weightless generation, training, weighted generation, and the internal shadow metrics.

## Current scope

This is a small research implementation rather than a production language model. Its purpose is to make the Janus Shadow mechanism inspectable: one file, deterministic initialization, causal trace snapshots, bounded trigram storage, explicit coherence, and a readable forward/backward path.

It belongs to the broader **Janus Method** line alongside **Janus Echo Attention Mechanism** and **RRPRAM**: architectures in which attention is not stateless, but observes the consequences of its own reading across multiple scales of time.
