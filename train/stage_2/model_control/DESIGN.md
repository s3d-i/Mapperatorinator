---
pinned_commit: 26cc8382e58199f78be581dbd1aee806315bb31c
status: frozen
date: 2026-05-04
owner: s3d-i
---

# Stage 2 Control Model Design

This document freezes the Stage 2 control model contract for implementation.
Workers should implement this design before making mapper-decoder changes.

## Goal

Train a control encoder that predicts `control_v3` features from audio,
oracle dense timing v2, and requested difficulty. The trained encoder must also
produce latent `control_memory` that the later mapper can cross-attend to.

The mapper design is intentionally out of scope for this slice.

## Inputs

Use the existing Stage 2 control-window dataset as the source of full-song
features:

- `full_mel`: `[B, T, 160]`
- `full_dense_timing_v2`: `[B, T, 4]`
- `padding_mask`: `[B, T]`, `True` means padded
- `normalized_difficulty`: `[B]`, continuous scalar in `[-1, 1]`
- `target_start_frame`: `[B]`
- `control_v3_target`: `[B, 100, 20]`

The model must train on fixed 12 second contexts:

- frame hop: `20ms`
- context length: `600` frames
- target length: `100` frames
- target is the center 2 seconds of the 12 second context

For each sample, slice:

```text
context_start_frame = target_start_frame - 250
context_end_frame = target_start_frame + 350
target_offset_in_context = 250
```

Out-of-song context frames are zero padded and marked in the context padding
mask. The target remains exactly `target_start_frame : target_start_frame + 100`.

## Targets

`control_v3_target` has 20 channels:

- 12 value channels from `VALUE_FEATURE_NAMES`
- 8 confidence channels from `CONFIDENCE_FEATURE_NAMES`

The model predicts both during training:

- `value_pred`: `[B, 100, 12]`
- `confidence_pred`: `[B, 100, 8]`
- `compound_confidence_pred`: `[B, 100, 1]`

The 8 confidence channels are training and diagnostics signals. They are not
part of the downstream mapper contract. The single compound confidence scalar is
reserved for diagnostics and possible mapper ablations.

## Downstream Contract

Expose this output object:

```python
@dataclass(frozen=True)
class ControlEncoderOutput:
    value_pred: torch.Tensor                  # [B, 100, 12]
    confidence_pred: torch.Tensor             # [B, 100, 8]
    compound_confidence_pred: torch.Tensor    # [B, 100, 1]
    control_memory: torch.Tensor              # [B, 600, D]
    memory_padding_mask: torch.Tensor         # [B, 600]
```

The later mapper should consume:

- `value_pred`
- `control_memory`
- `memory_padding_mask`

The later mapper should not consume all 8 confidence predictions by default.

## Architecture

Use a Transformer-owned hybrid encoder:

```text
mel/timing context
  -> input projection / conv stem
  -> difficulty FiLM
  -> Transformer encoder blocks
  -> difficulty FiLM after each block
  -> full context hidden states = control_memory
  -> center 100-frame slice
  -> value, confidence, and compound-confidence heads
```

### Input Features

Concatenate mel and timing channels:

```text
input = concat(context_mel, context_dense_timing_v2)  # [B, 600, 164]
```

Difficulty is not concatenated as a repeated input channel. Use it through FiLM.

### Conv Stem

The conv stem extracts local rhythm detail before self-attention. A small
default is enough:

- `Linear(164, d_model)` or `Conv1d(164, d_model, kernel_size=1)`
- 2 residual temporal conv blocks
- kernel size `5` or `7`
- same-length padding
- GELU activation
- dropout from config

The stem must preserve `[B, 600, d_model]`.

### Transformer Encoder

Use a bidirectional encoder over the 600 context frames:

- default `d_model`: `256`
- default heads: `4` or `8`
- default layers: `4`
- default feed-forward dim: `1024`
- dropout: configurable, default `0.1`
- `batch_first=True`
- support `src_key_padding_mask`

Normal full attention is acceptable at 600 frames. Do not implement full-song
hierarchical attention in this slice.

### Difficulty FiLM

Use continuous difficulty conditioning:

```python
h = h * (1.0 + gamma) + beta
```

`gamma` and `beta` come from an MLP over `normalized_difficulty`.

Apply FiLM:

- once after the conv stem
- once after every Transformer block

Initialize FiLM near identity:

- final gamma projection weights and bias start at zero
- final beta projection weights and bias start at zero

This makes the initial model behave like an unconditional encoder and learn
difficulty modulation only where useful.

## Prediction Heads

Slice the center target hidden states:

```text
center_hidden = control_memory[:, 250:350]  # [B, 100, D]
```

Heads:

- `value_head`: `D -> 12`
- `confidence_head`: `D -> 8`
- `compound_confidence_head`: `D -> 1`

Apply output ranges:

- value channels: clamp or sigmoid only where the target contract requires
  `[0, 1]`; otherwise allow signed outputs such as `hand_balance_signed`
- confidence channels: sigmoid
- compound confidence: sigmoid

Do not silently change target semantics. Use feature-name based range handling.

## Loss

Do not train with flat MSE over all 20 channels.

Use three loss groups:

```text
total_loss =
  value_loss
  + confidence_loss_weight * confidence_loss
  + compound_confidence_loss_weight * compound_confidence_loss
```

Recommended starting weights:

- `confidence_loss_weight = 0.25`
- `compound_confidence_loss_weight = 0.10`

### Value Loss

Use confidence-weighted regression for the 12 value channels. Confidence weights
come from the target confidence channels, not predicted confidence.

Feature families:

```text
density_level, density_burst        -> density_confidence
ln_change_rate_gated                -> ln_change_confidence
chord_ratio                         -> chord_confidence
jack_excess                         -> jack_confidence
jack_streak_exposure                -> jack_streak_confidence
hand_balance_signed                 -> hand_confidence
hand_imbalance_abs                  -> hand_confidence
repeat_exact, repeat_shift,
repeat_motion                       -> repeat_confidence
hold_occupancy                      -> control_confidence
```

Apply feature weights:

```text
density_level           1.00
density_burst           0.35
hold_occupancy          1.00
ln_change_rate_gated    0.60
chord_ratio             1.00
jack_excess             0.90
jack_streak_exposure    1.00
hand_balance_signed     0.60
hand_imbalance_abs      0.90
repeat_exact            1.00
repeat_shift            1.00
repeat_motion           1.00
```

Use MSE for the first implementation. Huber loss is allowed later only after a
measured reason.

### Confidence Loss

Train confidence predictions against the 8 target confidence channels using MSE
or BCE-with-logits. If the head applies sigmoid, use MSE.

### Compound Confidence Target

Define the compound confidence target as the mean of the 8 target confidence
channels:

```text
compound_confidence_target = mean(target_confidence_channels, dim=-1)
```

This scalar is diagnostic and reserved for future mapper ablations.

## Metrics

Report at least:

- total loss
- value loss
- confidence loss
- compound confidence loss
- per-value-feature MAE
- per-value-feature weighted MSE
- per-confidence-feature MAE
- compound confidence MAE

Keep metrics keyed by feature name.

## Training Entry Point

Create a Stage 2 control training entrypoint under `train/stage_2/training/`.

Required behavior:

- YAML config loading with unknown-key rejection
- deterministic seed
- device selection: `auto`, `cpu`, `mps`, `cuda`
- DataLoader using existing `ControlWindowDataset`
- fixed 12s context slicing in collate or a wrapper dataset
- train/eval split support before trusting larger runs
- checkpoint save and resume
- JSON report under `train/artifacts/runs/stage2_control/`

Use `uv run` for all Python commands.

## First Implementation Tasks

Implement in this order:

1. `ControlContextDataset` or context-collate helper that converts full-song
   batch tensors into `[B, 600, 160]`, `[B, 600, 4]`, and masks.
2. Unit tests for context slicing at song start, middle, and song end.
3. Control encoder model and output dataclass.
4. Unit tests for model shapes, mask handling, FiLM identity initialization,
   and parameter budget.
5. Loss module with feature-name based confidence weighting.
6. Unit tests for loss channel mapping and confidence weighting.
7. Training entrypoint with synthetic smoke mode.
8. Checkpoint/resume tests.
9. Tiny overfit run on a small subset.

## Non-Goals

- Do not implement mapper decoder changes in this slice.
- Do not train on BeatThis dense timing in this slice.
- Do not implement full-song hierarchical attention in this slice.
- Do not pass all 8 predicted confidence channels to the downstream mapper.
- Do not preserve Stage 1 compatibility wrappers unless explicitly requested.

## Assumptions

- Oracle dense timing v2 is good enough for initial control training.
- The current timing design and filtered dataset quality are good enough that a
  later BeatThis dense timing substitution should not require a new control
  model contract.
- This assumption must be verified later with BeatThis substitution diagnostics,
  but it does not block the first control-model implementation.
