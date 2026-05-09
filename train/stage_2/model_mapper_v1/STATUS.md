---
pinned_commit: 607ffdf17d203ff1860817c6d68d769adcb14adb
status: current blockers before mapper architecture upgrade
date: 2026-05-09
owner: s3d-i
module: train/stage_2/model_mapper_v1
---

# Mapper V1 Current Status

This note records the current Stage 2 mapper status before starting a larger
architecture upgrade.

## Summary

The current mapper is a teacher-forced Phase B model. It trains against
tokenized 8s windows with cached control-teacher memory and already has the
essential V1 supervised targets needed for V2 teacher-forced training.

The main V2 blockers are:

1. terminal mapper window coverage needs a first-class index/data contract;
2. a design mismatch between the mapper decoder and available global song
   information.

Rollout evaluation remains useful, especially for long-note generation quality,
but it is not a Mapper V2 design dependency for now.

## 1. Teacher Forcing Status

Current training and evaluation are teacher forced. Metrics report performance
on gold-prefix fragments, not autoregressive rollouts.

For Mapper V2 design, keep the V1 teacher-forced target contract: fragment
tokens, target states, carry-in/out, close labels, density targets, density
confidence, cached local control memory, and hard grammar masks.

## 2. Terminal Window Coverage

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

The mapper architecture and sample path are compatible with padded 8s windows.
The critical data requirement is that the mapper window-selection gate admits
explicit terminal windows instead of filtering them out before the padding path
runs.

We still need a new mapper index design that explicitly defines:

- fixed 8s stride windows for normal coverage;
- terminal windows for non-8s-aligned song endings;
- short-song behavior;
- BOS/EOS labeling policy;
- LN carry-in and carry-out reconstruction at every selected window boundary;
- cache keys and reports that make index provenance auditable.

The new index should be regenerated from the source map index, and terminal
coverage should be part of the first-class generation logic.

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
