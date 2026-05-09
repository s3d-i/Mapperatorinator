---
pinned_commit: 607ffdf17d203ff1860817c6d68d769adcb14adb
status: initial model and training entrypoint
date: 2026-05-09
owner: s3d-i
module: train/stage_2/model_mapper_v2
---

# Mapper V2

Current code status:

- `MapperV2Model` and `MapperV2Config` exist in `model.py`;
- V2 reuses the V1 tokenizer, replay state, grammar mask, adapters, and
  teacher-forced loss target contract;
- V2 adds a trainable mapper-side full-song mel/timing encoder plus gated
  decoder cross-attention;
- V2 also injects direct window-level global position conditioning into the
  decoder, using the target window start/center, real source song length, and
  seconds-to-end features;
- `MapperV1WindowDataset` can now opt into full-song context on cached local
  control-teacher batches with `include_full_song_context=True`;
- `train.stage_2.training.mapper_v2` is the V2 Phase B training entrypoint. It
  reuses the V1 teacher-forced loss, local cache format, and mapper window
  targets while constructing `MapperV2Model` and requiring full-song context for
  global runs.

Teacher cache precompute remains shared with V1 because V2 consumes the same
local `control_memory_8s` and `density_teacher_8s` tensors. Formal V2 training
should use:

```bash
uv run python -m train.stage_2.training.mapper_v2 --config train/stage_2/training/configs/stage2_mapper_v2_phase_b_global_mps.yaml
```

Mapper V2 is an architecture upgrade over `model_mapper_v1`, not a replacement
for the V1 teacher-forced target contract.

V1 already has the essential supervised mapper targets:

- tokenized 8s mapper fragments;
- explicit `LNCarryState` carry-in and carry-out;
- carry-aware hard grammar masks;
- long-note close labels;
- density target and confidence tensors;
- cached local `control_memory_8s` and `density_teacher_8s`.

V2 should keep those targets and add a mapper-side full-song context path.

## Architecture Target

Keep the V1 local path:

- decoder self-attention over mapper tokens;
- cross-attention to local 8s control memory;
- state-prior, LN-close, density, and grammar loss paths.

Add a global path:

- consume full-song packed mel `[B, F, 160]`;
- consume full-song dense timing v2 `[B, F, 4]`;
- respect full-song padding masks and frame counts;
- pool or encode the full-song stream to a bounded memory length;
- let the mapper decoder cross-attend to this global memory through a gated
  block initialized near zero.
- add a direct global window-position projection to the decoder input hidden so
  local token queries know where the current 8s write window sits in the whole
  song, without replacing the V1 local time features.

The first V2 implementation trains with the same teacher-forced losses as V1.
Rollout evaluation is not a V2 design dependency for now.

## Data Contract

The mapper window path is compatible with padded 8s windows. Terminal coverage
depends on the window-selection gate admitting explicit terminal windows, then
the existing sample path pads `full_mel`, `full_dense_timing_v2`, masks, and
control-teacher slices to the selected 8s span.

V2 should treat terminal windows as first-class data, not as a post-hoc patched
artifact.
