---
pinned_commit: b1039d0634e1dec63d5d2f67ea96b8845dfe6a61
status: current blockers before mapper architecture upgrade
date: 2026-05-09
owner: s3d-i
module: train/stage_2/model_mapper_v1
---

# Mapper V1 Current Status

This note records the current Stage 2 mapper status before starting a larger
architecture upgrade.

## Summary

The current mapper is still a teacher-forced Phase B model. It trains against
tokenized 8s windows with cached control-teacher memory, but it does not yet
have a rollout evaluation loop that validates generated windows under the same
grammar and long-note carry constraints used by inference.

The main blockers are:

1. teacher-forcing-only evaluation;
2. stale mapper window index generation that misses terminal song windows;
3. a design mismatch between the mapper decoder and available global song
   information.

## 1. Teacher Forcing Only

Current training and evaluation are teacher forced. Metrics report performance
on gold-prefix fragments, not autoregressive rollouts.

Rollout evaluation is intentionally not the default yet because long-note close
behavior is high risk. A model that looks acceptable under teacher forcing can
still drift during generation, miss close events, close on the wrong lane, or
enter a state where the hard grammar has no good next token. This matters more
for mapper v1 than a simple event model because LN carry state crosses token
and window boundaries.

Before treating mapper metrics as reliable, add rollout evaluation that checks:

- exact window completion;
- legal grammar-constrained generation;
- LN carry-in and carry-out consistency;
- close timing and lane correctness;
- dead-end and max-token failure rates;
- generated density and section-shape drift.

Until then, teacher-forced loss is only a training signal, not evidence that the
model can generate stable charts.

## 2. Current Window Index Is Stale

The current mapper window index was generated before
`9d5f1a324e5211d7a73d75f3d7bd3ed1c7627c68`.

That means the index does not correctly include terminal song windows when the
song end is not aligned to an exact 8s mapper window. In practice, this prevents
the mapper from learning end-of-song behavior for many charts because those
terminal fragments are absent or underrepresented.

There is a later helper,
`train/stage_2/data/build_mapper_v1_end_window_index.py`, that appends terminal
windows to an existing index. That is useful evidence of the issue, but the next
training run should not depend on patching a stale artifact as the long-term
contract.

We need a new mapper index design that explicitly defines:

- fixed 8s stride windows for normal coverage;
- terminal windows for non-8s-aligned song endings;
- short-song behavior;
- BOS/EOS labeling policy;
- LN carry-in and carry-out reconstruction at every selected window boundary;
- cache keys and reports that make index provenance auditable.

The new index should be regenerated from the source map index and should make
terminal coverage part of the first-class generation logic.

## 3. Mapper Global Context Mismatch

The current control teacher already has a full-song path: it consumes full-song
packed mel, dense timing, and song-position features, pools them by
`global_stride`, and fuses the resulting global memory into local control
memory. Current mapper training can therefore receive full-song information
indirectly through cached `control_memory_8s`.

The mapper decoder itself, however, only cross-attends to the 8s control memory
slice. It does not directly cross-attend to pooled full-song mel, full-song
timing, or global control memory. This is the biggest design mismatch in the
current model.

The likely next architecture should keep local 8s control memory for precise
event placement, but add a mapper-side global memory stream. That stream should
be pooled or otherwise compressed; raw full-resolution full-song cross-attention
would be too expensive and noisy for normal training.

A practical target design:

- preserve `control_memory_8s` as the local decoder memory;
- cache or compute pooled full-song global memory alongside the local memory;
- let the mapper decoder cross-attend to global mel/timing/control memory
  through a gated block;
- initialize the new global gate near zero so existing behavior is the starting
  point;
- train with fresh optimizer state, optionally warm-starting compatible current
  mapper weights.

This is an architecture upgrade, not a small continuation run.
