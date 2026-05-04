---
pinned_commit: 85b7e196aca1596e4170cb218c43e33941ecbedd
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
- `target_valid_mask`: `[B, 100]`, `True` means the target frame is inside
  the song and may contribute to loss and metrics
- `ln_change_n_eff_target`: `[B, 100]`, training-only diagnostic support for
  `ln_change_rate_gated`

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
mask. The target remains aligned to
`target_start_frame : target_start_frame + 100`, but target frames beyond
`frame_count` must have `target_valid_mask=False`.

The target tensor may keep zero-filled out-of-song values for fixed shape, but
all target losses and target metrics must ignore those frames. Compute:

```text
target_valid_mask[b, i] =
  target_start_frame[b] + i < frame_count[b]
```

## Targets

`control_v3_target` has 20 channels:

- 12 value channels from `VALUE_FEATURE_NAMES`
- 8 confidence channels from `CONFIDENCE_FEATURE_NAMES`

The model predicts value and confidence channels during training:

- `value_pred`: `[B, 100, 12]`
- `confidence_pred`: `[B, 100, 8]`

The 8 confidence channels are training and diagnostics signals. They are not
part of the downstream mapper contract. The single compound confidence scalar is
reserved for diagnostics and possible mapper ablations, but it is derived from
the predicted `control_confidence` channel instead of learned by a separate head.

`ln_change_n_eff_target` is not part of `control_v3_target`, is not predicted by
the model, and is not part of the downstream mapper contract. Load it from the
`ln_change_n_eff` diagnostic column in the saved `control_v3` time series and
resample it onto the same 100 target frame centers as `control_v3_target`.

## Downstream Contract

Expose this output object:

```python
@dataclass(frozen=True)
class ControlEncoderOutput:
    value_pred: torch.Tensor                  # [B, 100, 12]
    confidence_pred: torch.Tensor             # [B, 100, 8]
    compound_confidence_pred: torch.Tensor    # [B, 100, 1], derived alias
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
  -> value and confidence heads
  -> derived compound-confidence output
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

Apply output ranges:

- value channels: clamp or sigmoid only where the target contract requires
  `[0, 1]`; otherwise allow signed outputs such as `hand_balance_signed`
- confidence channels: sigmoid

Do not silently change target semantics. Use feature-name based range handling.

Derive:

```text
compound_confidence_pred =
  confidence_pred[..., control_confidence_index : control_confidence_index + 1]
```

Do not add an independent compound-confidence head in the first implementation.

## Loss

Do not train with flat MSE over all 20 channels.

Use two loss groups:

```text
total_loss =
  value_loss
  + confidence_loss_weight * confidence_loss
```

Recommended starting weights:

- `confidence_loss_weight = 0.25`

### Value Loss

Use robust weighted regression for the 12 value channels. These channels are
continuous control strengths, not binary labels; do not replace the primary
value loss with BCE or focal BCE.

For each value feature `f`:

```text
loss_f = weighted_mean(
  smooth_l1(pred_f - target_f, delta[f]),
  target_valid_mask
    * confidence_weight[f]
    * feature_weight[f]
    * sparse_multiplier[f]
)
```

Confidence weights come from target confidence channels, not predicted
confidence. Normalize weighted losses by the sum of effective weights, not by
`B * 100`, so masked target tails, low-support LN-change frames, and sparse
boosting do not change loss scale accidentally.

Use configurable Huber/SmoothL1 deltas. Starting values:

```text
density_level, ln_change_rate_gated    0.20
bounded [0, 1] value channels          0.10
hand_balance_signed                    0.10
```

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
hold_occupancy                      -> 1.0
```

`hold_occupancy` is deterministic LN interval occupancy. It should only be
weighted by `target_valid_mask`, not by `control_confidence`,
`density_confidence`, or any unrelated support signal.

For `ln_change_rate_gated`, also multiply by a support weight derived from
`ln_change_n_eff_target`:

```text
ln_change_support_weight =
  clip((ln_change_n_eff_target - 2.0) / 1.0, 0, 1)
```

This keeps full weight at the audit threshold `n_eff >= 3.0`, fades partial
support between `2.0` and `3.0`, and masks very weak support below `2.0`.

If `ln_change_n_eff_target` is not loaded in a first smoke implementation, state
that the run is confidence-only for LN-change and set
`ln_change_support_weight = 1.0`. Do not pretend support-aware weighting is
active without the diagnostic sidecar.

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

Sparse continuous features need target-dependent balancing in addition to the
static feature weights. Apply sparse balancing to:

```text
jack_excess
hand_imbalance_abs
repeat_exact
repeat_shift
repeat_motion
```

Use:

```text
sparse_multiplier[f] =
  1 + sparse_boost[f] * smoothstep(target_f, sparse_low[f], sparse_high[f])
```

Starting defaults:

```text
sparse_low[f] = 0.10
sparse_high[f] = 0.50
sparse_boost[f] = 4.0
```

Allow `sparse_boost` up to `8.0` if validation shows systematic positive-target
underprediction. Do not apply sparse boost to `jack_streak_exposure` initially;
it is much less sparse and should first train with `jack_streak_confidence`.

For `hand_balance_signed`, train the sign only when the target has meaningful
hand imbalance:

```text
hand_balance_weight =
  target_valid_mask
  * hand_confidence
  * feature_weight[hand_balance_signed]
  * clip(target_hand_imbalance_abs / 0.25, 0, 1)
```

Mirror augmentation must flip `hand_balance_signed`.

### Confidence Loss

Train confidence predictions against the 8 target confidence channels. Use MSE
or Huber on sigmoid outputs. If the implementation keeps pre-sigmoid logits for
loss computation, soft BCE-with-logits is also allowed, but the exposed
`confidence_pred` must remain sigmoid-ranged. Every confidence loss term is
multiplied by `target_valid_mask`.

The `control_confidence` target is already one of the 8 target confidence
channels. Supervise it directly as part of `confidence_loss`.

### Compound Confidence Output

Do not define a separate `compound_confidence_target` as
`mean(target_confidence_channels)`. The 8 channels already include
`control_confidence`, which is itself the Stage 1 aggregate confidence. Expose
`compound_confidence_pred` as the predicted `control_confidence` channel.

## Metrics

Report at least:

- total loss
- value loss
- confidence loss
- per-value-feature MAE
- per-value-feature weighted Huber/SmoothL1
- per-confidence-feature MAE
- control/compound confidence MAE
- target valid frame rate
- masked target frame count
- `ln_change_rate_gated` support-weight mean and masked frame count
- sparse positive-frame MAE for `jack_excess`, `hand_imbalance_abs`, and
  `repeat_*` where target is at least `0.50`
- sparse positive-window max-target versus max-pred MAE

Keep metrics keyed by feature name.

## Training Entry Point

Create a Stage 2 control training entrypoint under `train/stage_2/training/`.

Required behavior:

- YAML config loading with unknown-key rejection
- deterministic seed
- device selection: `auto`, `cpu`, `mps`, `cuda`
- DataLoader using existing `ControlWindowDataset`
- fixed 12s context slicing in collate or a wrapper dataset
- `target_valid_mask` generation before loss computation
- `ln_change_n_eff_target` loading or derivation for support-aware
  `ln_change_rate_gated` loss
- train/eval split support before trusting larger runs
- checkpoint save and resume
- JSON report under `train/artifacts/runs/stage2_control/`

Use `uv run` for all Python commands.

## First Implementation Tasks

Implement in this order:

1. `ControlContextDataset` or context-collate helper that converts full-song
   batch tensors into `[B, 600, 160]`, `[B, 600, 4]`, context masks, and
   `target_valid_mask`.
2. Unit tests for context slicing and target masks at song start, middle, and
   song end.
3. Control encoder model and output dataclass.
4. Unit tests for model shapes, mask handling, FiLM identity initialization,
   and parameter budget.
5. Loss module with robust value regression, feature-name based confidence
   weighting, `target_valid_mask`, sparse balancing, and
   `ln_change_n_eff_target` support weighting.
6. Unit tests for loss channel mapping, confidence weighting, target masking,
   sparse balancing, hand-balance magnitude gating, LN-change support weighting,
   and derived compound-confidence output.
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
