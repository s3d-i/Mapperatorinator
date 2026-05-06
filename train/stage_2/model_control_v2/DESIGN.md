---
pinned_commit: b47d35c7fd5cacc9b44b31b62abc225b567ce7ca
status: frozen
date: 2026-05-07
owner: s3d-i
feature_contract_version: control_v4
design_revision: v1.2
module: train/stage_2/model_control_v2
supersedes_for_new_work: train/stage_2/model_control/DESIGN.md
---

# Stage 2 Control V4 Design Spec

This document freezes the `control_v4` feature and model contract for the new
`model_control_v2` implementation. The goal is a local coarse control prior,
not a complete chart-feature oracle.

## Decision Summary

Train exactly four value controls:

```text
density_level_v4
density_burst_v4
chord_ratio_v4
technical_pressure_v4
```

Model output is:

```text
control_v4_pred: [B, 20, 4]
```

Do not train or expose confidence channels as model outputs. Confidence remains
artifact metadata, loss weighting, and diagnostics only.

The design intentionally fixes the issues observed in the first Stage 2 control
run:

- average regression loss must not hide peak underprediction;
- confidence/easy channels must not dilute the main task;
- random-window eval must not be the only sparse-feature judge.

Revision `v1.1` made four implementation constraints explicit:

- split construction happens before any fitted normalizer, positive threshold, or
  final artifact write;
- confidence and validity masks have fixed formulas and a floor for negative
  frame supervision;
- peak losses use masked 500ms max-pooling;
- feature scalar weights are applied only after per-feature weighted means, so
  they cannot be canceled by denominator normalization.

Revision `v1.2` integrates the second review pass with three more hard
constraints:

- the 100ms target grid is center-aligned to pooled 20ms hidden states;
- sparse soft normalization is continuous at zero;
- warm-start is allowed only when the backbone architecture is exactly
  compatible with the checkpoint.

## Scope

In scope:

- `control_v4` target extraction and artifacts;
- group-aware train/val/test split;
- feature-balanced window index;
- standalone audio-to-control model;
- control-specific loss and eval.

Out of scope:

- mapper decoder changes;
- joint mapper training;
- compatibility wrappers for the previous 20-channel `control_v3` model;
- preserving the old confidence-head contract.

## Inputs

The standalone control model consumes the same context family as the existing
Stage 2 control encoder:

- `context_mel`: `[B, 600, 160]`
- `context_dense_timing_v2`: `[B, 600, 4]`
- `context_padding_mask`: `[B, 600]`, `True` means padded
- `normalized_difficulty`: `[B]`, continuous scalar in `[-1, 1]`

Frame and window geometry:

```text
context hop:          20ms
context length:       600 frames = 12s
target window:        center 100 context frames = 2s
target source grid:   100ms
target frame count:   20
target shape:         [B, 20, 4]
```

Slice the context as before:

```text
context_start_frame = target_start_frame - 250
context_end_frame   = target_start_frame + 350
target_offset       = 250
target_20ms_span    = context[:, 250:350]
target_100ms_hidden = avg_pool_5_frames(target_20ms_span)
```

V4 target windows must start on the 100ms source grid:

```text
20ms_frame_center_s[f] = 0.01 + 0.02 * f
source_grid[n] = 0.05 + 0.10 * n
target_start_20ms_frame = target_start_100ms_frame * 5
target_time_s[k] = 0.05 + 0.10 * (target_start_100ms_frame + k)
window_start_s = 0.10 * target_start_100ms_frame
window_end_s = window_start_s + 2.0
```

Do not create 20ms-stride v4 target windows. The 100ms target frame is the
source-grid feature value for that bin center, and the matching model hidden
state is the average of the corresponding five 20ms encoder states. `time_s` in
the timeseries artifact is the 100ms bin center, not the left edge.

The v4 head must directly output 20 target frames. A 100-frame output with
loss-time pooling is allowed only as a temporary smoke-test bridge, not as the
frozen contract.

## Feature Extraction Contract

Use the 100ms source grid directly. Do not perform per-song normalization.
All normalizers are fitted on valid train-split frames only and written to
metadata.

Use the triangular kernel already used by v2/v3 feature code:

```text
K_L(u) = max(0, 1 - abs(u) / L) / L
```

Use:

```text
alpha = 1.15
key_count = 4
```

### Soft Robust Normalizer

Normalizer slopes are metadata-configurable. First-run defaults:

```text
density_level_slope = 3.0
density_burst_slope = 2.5
technical_slope     = 2.5
```

For dense nonnegative signals:

```text
denom = max(q90 - q10, 1e-6)
z = (x - q50) / denom
soft_robust01(x, slope) = sigmoid(slope * z)
```

For sparse nonnegative signals, use a continuous zero-preserving variant:

```text
denom = max(q90_pos - q10_pos, 1e-6)
s = sigmoid(slope * (x - q50_pos) / denom)
s0 = sigmoid(slope * (0.0 - q50_pos) / denom)
y = (s - s0) / max(1.0 - s0, 1e-6)
zero_preserving_soft_robust01(x, slope) =
  0.0 if x <= 0
  clip01(y) otherwise
```

Do not use a discontinuous sparse normalizer that jumps from `0.0` at `x = 0`
to a nontrivial positive value at `x = epsilon`.

The sparse normalizer quantiles are fitted over positive valid train values:

```text
positive = raw_value > 1e-8 and valid_control_mask
```

Minimum positive-frame blockers:

```text
density_burst_v4:
  min_positive_frames = max(50000, 0.002 * train_valid_frame_count)
technical_pressure_v4:
  min_positive_frames = max(5000, 0.0005 * train_valid_frame_count)
chord_ratio_v4:
  no sparse normalizer, but positive boost quantiles require
  positive_chord_frames >= max(50000, 0.002 * train_valid_frame_count)
```

If a sparse feature has too few positive train frames to fit stable quantiles,
artifact construction must fail before training.

### density_level_v4

Definition:

```text
D_med(t) = sum_i chord_size_i ** alpha * K_3.0s(t - onset_i)
density_level_raw(t) = log1p(D_med(t))
density_level_v4(t) =
  soft_robust01(density_level_raw(t), density_level_slope)
```

Interpretation:

```text
sustained mapping intensity / section energy budget
```

This is a high-confidence feature. It should remain the strongest control axis.

### density_burst_v4

Definition:

```text
D_short(t) = sum_i chord_size_i ** alpha * K_0.50s(t - onset_i)
D_med(t)   = sum_i chord_size_i ** alpha * K_3.0s(t - onset_i)
burst_raw(t) = max(0, log1p(D_short(t)) - log1p(D_med(t)))
density_burst_v4(t) =
  zero_preserving_soft_robust01(burst_raw(t), density_burst_slope)
```

Interpretation:

```text
local accent / spike prior
```

This feature is not a hard object-count constraint. It should encourage local
emphasis around short musical spikes and chord accents. Training must explicitly
protect peak recall because pointwise average loss underfits this feature.

### chord_ratio_v4

Definition:

```text
chord_strength_i = (chord_size_i - 1) / (key_count - 1)
chord_ratio_raw(t) =
  weighted_average(chord_strength_i, kernel=K_2.0s)
chord_ratio_v4(t) = clip01(chord_ratio_raw(t))
```

Do not confidence-gate the value. A zero value should mean low simultaneity
pressure, not low support. Save confidence separately and use it for loss
weighting and diagnostics.

Interpretation:

```text
simultaneity / chord pressure
```

### technical_pressure_v4

This feature replaces the mapper-facing meaning of `jack_streak_exposure`.
The model and mapper should treat it as weak technical texture pressure, not as
a jack command.

Per onset event:

```text
technical_event_weight = 0

for each active column:
  gap = t_cur - last_time_same_col
  if 0 < gap < 0.22s:
    risk = (1 - gap / 0.22s) ** 2
    streak_factor = min(streak_len - 1, 4) / 4
    technical_event_weight += risk * streak_factor
```

Then:

```text
technical_raw(t) =
  sum_i technical_event_weight_i * K_1.25s(t - onset_i)
technical_pressure_v4(t) =
  zero_preserving_soft_robust01(log1p(technical_raw(t)), technical_slope)
```

Interpretation:

```text
weak compact / repeated-lane / hand-pressure-heavy texture prior
```

This is experimental. It must use low global loss weight, strong positive
weighting, and feature-positive eval. It may be removed from mapper conditioning
later if ablations show that it behaves like a noisy density proxy.

## Validity And Confidence Contract

Confidence is not predicted by the model. It is artifact-side metadata used for
loss weighting and diagnostics. Do not reuse a generic v3 compound confidence for
these columns.

Required loss-facing confidence columns:

```text
density_confidence
chord_confidence
technical_confidence
```

Additional support diagnostics may be stored, but training code must not depend
on extra confidence columns unless this contract is revised.

### Effective Support

For an event set and kernel length `L`:

```text
sum_w(t)  = sum_i K_L(t - event_time_i)
sum_w2(t) = sum_i K_L(t - event_time_i) ** 2
n_eff(t)  = sum_w(t) ** 2 / max(sum_w2(t), 1e-12)

support_confidence(n_eff, n0, k):
  x = (n_eff - n0) / k
  return clip01(1.0 - exp(-max(0.0, x)))
```

`n_eff` is computed from support event counts and kernel weights, not from target
value magnitude.

### valid_control_mask

```text
valid_control_mask(t) =
  finite_features(t)
  and t >= first_hit_start_s
  and t <= last_object_end_s
  and t < audio_duration_s
  and map_passed_feature_extraction_validation

first_hit_start_s = min(hit.start_s)
last_object_end_s = max(hit.end_s if present else hit.start_s)
```

Window target masks are derived from source rows:

```text
target_valid_mask[b, k] =
  source row exists
  and valid_control_mask(source_time_s)
```

Invalid target frames must have all four final `*_loss_weight` values set to
`0.0`.

Do not mark rests inside `[first_hit_start_s, last_object_end_s]` invalid.
Middle rests are real low-density training targets. Intro/outro outside playable
object range may be invalid.

### density_confidence

`density_confidence` uses medium-density onset support:

```text
density_support_times = onset_times
density_n_eff_med = n_eff from K_3.0s over onset_times
density_confidence =
  support_confidence(density_n_eff_med, n0=1.0, k=2.0)
```

The density-level point loss does not multiply this confidence:
`w_density = valid_control_mask`. Density burst uses this family confidence.
Do not use short-window support confidence to downweight `density_burst_v4`,
because that suppresses the short spikes the feature is meant to learn.

### chord_confidence

Chord confidence uses onset support for the ratio denominator:

```text
chord_support_times = onset_times
chord_n_eff = n_eff from K_2.0s over onset_times
chord_confidence =
  support_confidence(chord_n_eff, n0=3.0, k=4.0)
```

### technical_confidence

Technical confidence is based on positive technical event support. Ordinary
zero-tech frames still receive supervision through the technical loss-confidence
floor.

```text
technical_event_times =
  times where technical_event_weight > 0
technical_n_eff =
  n_eff from K_1.25s over technical_event_times

technical_confidence =
  support_confidence(technical_n_eff, n0=1.0, k=2.0)
```

## Artifact Contract

Artifacts live under:

```text
train/artifacts/features/control_v4/
```

Required files:

```text
control_v4_timeseries.parquet
control_v4_map_summary.parquet
control_v4_window_index.parquet
control_v4_metadata.json
```

Raw-only build cache:

```text
train/artifacts/features/control_v4/raw/control_v4_raw_timeseries.parquet
control_v4_fit_metadata.tmp.json
```

The raw cache may be extracted for all split-assigned maps because it is
deterministic and unfitted. It must not be consumed by training as a final
artifact. It may contain only raw unbounded signals, debug support values, split
labels, and `valid_control_mask`.

The raw cache must not contain:

```text
density_level_v4
density_burst_v4
chord_ratio_v4
technical_pressure_v4
*_loss_weight
normalizer_quantiles
positive_boost_quantiles
sampler buckets
baselines
```

Normalizer fitting, positive-threshold fitting, difficulty weights, and
baselines must use only:

```text
split == "train"
and valid_control_mask == true
```

Final bounded targets for train, val, and test must all be written from the
train-fitted quantiles in `control_v4_fit_metadata.tmp.json`.

### Timeseries Columns

Each 100ms row must include map identity columns carried from the Stage 2 index
plus:

```text
time_s
density_level_v4
density_burst_v4
chord_ratio_v4
technical_pressure_v4
density_level_loss_weight
density_burst_loss_weight
chord_loss_weight
technical_loss_weight
density_level_raw
density_burst_raw
chord_ratio_raw
technical_raw
density_confidence
chord_confidence
technical_confidence
valid_control_mask
split
difficulty_bin
```

The four `*_loss_weight` columns are final frame-level base weights after
validity, confidence, feature-positive boosting, and technical difficulty
balancing. Feature scalar weights from the loss config are not baked into these
columns.

### Map Summary Columns

At minimum:

```text
map identity columns
duration_s
normalized_difficulty
difficulty_bin
split
valid_frame_count
density_level_mean
density_burst_max
chord_ratio_p95
technical_pressure_max
nonzero rates for each feature
saturation rates for each feature
```

### Window Index Columns

Each candidate 2s target window stores:

```text
map identity columns
window_start_s
window_end_s
target_start_20ms_frame
target_start_100ms_frame
split
difficulty_bin
target_valid_frame_count
density_level_mean
density_burst_max
chord_ratio_p95
technical_pressure_max
is_uniform_candidate
is_high_burst
is_high_chord
is_high_tech
is_rare_difficulty
sampler_bucket_primary_debug
```

The index must support separate uniform, high-burst, high-chord,
high-technical, and rare-difficulty sampling. Bucket membership is boolean and
non-exclusive; `sampler_bucket_primary_debug` is only a reporting label.

Default window generation:

```text
TARGET_SOURCE_GRID_STEP = 0.10s
TARGET_SOURCE_FRAME_COUNT = 20
TARGET_WINDOW_SECONDS = 2.0
TARGET_WINDOW_STRIDE_SOURCE_FRAMES = 20
TARGET_WINDOW_STRIDE_SECONDS = 2.0
require_full_target_window = true
require_all_20_target_rows_exist = true
min_valid_control_frames = 20
```

Include a window only if:

```text
target_start_100ms_frame + 20 <= source_frame_count_100ms
and all 20 source rows exist
and sum(valid_control_mask in target window) >= min_valid_control_frames
```

Optional overlap for data-limited experiments may use
`TARGET_WINDOW_STRIDE_SOURCE_FRAMES = 10`, but the frozen default is
non-overlapping 2s windows. A debug artifact may relax
`min_valid_control_frames` to `12`; formal standalone v4 training uses
full-valid windows. The loss remains mask-aware for smoke tests, artifact bugs,
and later mapper reuse.

### Metadata

`control_v4_metadata.json` must include:

```text
schema_version
feature_contract_version = "control_v4"
pinned_commit
artifact_created_at
source_index_path
group_split_config
split_seed
group_key_method
group_key_columns
train_group_count
val_group_count
test_group_count
train_valid_frame_count
val_valid_frame_count
test_valid_frame_count
train_split_sha
normalizer_fit_scope = "train_valid_frames_only"
positive_threshold_fit_scope = "train_positive_valid_frames_only"
feature_config
normalizer_config
normalizer_quantiles
positive_boost_quantiles
confidence_config
window_index_config
sampler_bucket_thresholds
difficulty_bin_counts_by_split
window_bucket_counts_by_split
artifact_audit_summary
```

The normalizer metadata must be sufficient to reproduce every bounded target
value exactly. `confidence_config` must include every `support_confidence`
parameter, support event definition, kernel length, and loss-confidence floor.
Optional support diagnostics must be listed separately from the required
loss-facing confidence columns.

Minimum confidence metadata:

```text
confidence_config:
  density:
    output_column: density_confidence
    support_source: onset_times
    kernel_L: 3.0
    n0: 1.0
    k: 2.0
    loss_floor_for_burst: 0.25
  chord:
    output_column: chord_confidence
    support_source: onset_times
    kernel_L: 2.0
    n0: 3.0
    k: 4.0
    loss_floor: 0.25
  technical:
    output_column: technical_confidence
    support_source: technical_event_times
    kernel_L: 1.25
    n0: 1.0
    k: 2.0
    loss_floor: 0.35
```

## Splits

Use group-aware splitting before fitting normalizers, positive thresholds, or
final bounded targets. No normalizer or positive threshold may be fit before the
split exists.

Default:

```text
train: 80%
val:   10%
test:  10%
```

Split by group so that alternate difficulties of the same song do not cross
train/eval boundaries. Default group key:

```text
audio_sha256 if available
else f"{shard}::{normalized_audio_path}"
```

If neither audio hash nor audio path is available, artifact construction must
fail. Do not use beatmap path as the primary group key.

Difficulty bins:

```python
def difficulty_bin(diff: float) -> str:
    if 2.0 <= diff < 3.0:
        return "2-3"
    if 3.0 <= diff < 4.0:
        return "3-4"
    if 4.0 <= diff < 5.0:
        return "4-5"
    if 5.0 <= diff <= 6.0:
        return "5-6"
    raise ValueError("control_v4 supports difficulty in [2.0, 6.0]")
```

Required difficulty bins:

```text
2-3
3-4
4-5
5-6
```

Every split must contain enough windows from each bin for eval. If this cannot
be satisfied, artifact construction must report the deficit and fail instead of
silently falling back to random windows.

Formal-artifact fail-fast minimums:

```text
min_groups_per_bin_per_eval_split = 8
min_windows_per_bin_per_eval_split = 1024
min_total_windows_per_eval_split = 8192
```

On failure, report the missing bin, available group count, available window
count, and a suggested fallback. Do not silently use random-window fallback. A
small-data smoke artifact may relax these gates only with an explicit
`--allow-small-split-smoke` flag recorded in metadata.

Use group-level greedy stratified assignment:

```text
seed = 1337
split ratios: train=0.80, val=0.10, test=0.10

input per group:
  difficulty-bin map/window count vector

sort groups by total count descending
for each group:
  assign to the split that minimizes objective

objective =
  sum_bins (
    (current_count[split, bin] + group_count[bin] - target_count[split, bin])
    / max(target_count[split, bin], 1)
  ) ** 2
  + 0.1 * total_count_balance_error
```

`train_split_sha` is:

```text
sha256("\n".join(sorted(train_map_identity_strings)).encode()).hexdigest()
```

## Two-Pass Artifact Build

The artifact builder has four phases.

Phase A: manifest and split.

```text
source index
-> map manifest
-> group-aware split
-> split manifest
```

The split manifest must contain:

```text
group_key
split
difficulty_bin
beatmap_path
audio_path
source index / filtered index / beatmap_id
frame_count_20ms
source_frame_count_100ms
```

Phase B: raw feature extraction.

```text
split manifest
-> raw unbounded feature cache for all split-assigned maps
```

Phase C: train-only fitting.

Fit only from train split valid frames:

```text
normalizer_quantiles
positive_boost_quantiles
difficulty_balance_weight
global and difficulty-bin baselines
```

Phase D: materialize final artifacts.

```text
final bounded timeseries
map summary
window index
metadata
```

The artifact builder must run in this order:

1. Load source Stage 2 map index.
2. Filter supported 4K mania maps with difficulty in `[2, 6]`.
3. Compute `group_key`.
4. Build group-aware train/val/test split.
5. Validate split difficulty-bin coverage.
6. Extract raw v4 timeseries for all maps, carrying split labels.
7. Fit normalizers and positive-boost thresholds from train valid frames only.
8. Apply train-fitted normalizers and weights to all splits.
9. Write final v4 timeseries, map summary, window index, and metadata.
10. Run final artifact audit.
11. Train model.

Raw extraction may touch all splits. Fitting must only touch train rows with
`valid_control_mask = true`; sparse positive-threshold fitting must additionally
require `raw_value > 1e-8`.

Any artifact containing bounded target values, fitted quantiles, positive
thresholds, frame-level loss weights, sampler buckets, or baselines must be
generated after split assignment. Raw unbounded caches may be computed before
fitting only if they are clearly marked raw-only and are never consumed as final
training artifacts without train-only fitting.

## Loss Weights

Frame-level loss weights are computed from train-split quantiles. Do not use
fixed value thresholds like `0.50` to decide sparse positives.

Positive boost:

```text
smoothstep(y, lo, hi) =
  r * r * (3 - 2 * r)
  where r = clip((y - lo) / max(hi - lo, 1e-6), 0, 1)

positive_boost(y, lo, hi, boost) =
  1 + boost * smoothstep(y, lo, hi)
```

Use positive-value train quantiles for sparse/rare features:

```text
burst: q70_pos, q95_pos
chord: q75_pos, q95_pos
tech:  q70_pos, q95_pos
```

Keep a confidence floor so low-support negative frames remain weakly supervised:

```text
density_confidence_floor_for_burst = 0.25
chord_confidence_floor = 0.25
technical_confidence_floor = 0.35

loss_confidence(c, floor) = floor + (1.0 - floor) * c
```

Frame weights:

```text
w_density =
  valid_control_mask

w_burst =
  valid_control_mask
  * loss_confidence(density_confidence, floor=0.25)
  * positive_boost(y_burst, q70_burst_pos, q95_burst_pos, boost=4.0)

w_chord =
  valid_control_mask
  * loss_confidence(chord_confidence, floor=0.25)
  * positive_boost(y_chord, q75_chord_pos, q95_chord_pos, boost=2.0)

w_tech =
  valid_control_mask
  * loss_confidence(technical_confidence, floor=0.35)
  * positive_boost(y_tech, q70_tech_pos, q95_tech_pos, boost=6.0)
  * difficulty_balance_weight
```

`difficulty_balance_weight` is computed from train-window difficulty bins:

```text
raw_bin_weight[b] =
  total_train_windows / (num_bins * train_windows_in_bin[b])

clipped[b] = clip(raw_bin_weight[b], 0.5, 2.0)

mean_after_clip =
  sum_b clipped[b] * train_windows_in_bin[b] / total_train_windows

difficulty_balance_weight[b] =
  clipped[b] / mean_after_clip
```

Frame-level weights do not include feature scalar weights. Per-feature losses are
first normalized by frame-level weight sum. Feature scalar weights are applied
only when combining those per-feature losses, so they cannot be canceled by the
same denominator.

## Model

Reuse the existing Stage 2 control backbone shape:

```text
mel + dense_timing_v2
  -> input projection
  -> temporal conv stem
  -> positional embedding
  -> difficulty FiLM
  -> Transformer encoder blocks
  -> control_memory
```

V4 output head:

```python
center_hidden = control_memory[:, 250:350]       # [B, 100, D]
center_mask = memory_padding_mask[:, 250:350]    # [B, 100], True means padded

hidden_group = center_hidden.reshape(B, 20, 5, D)
mask_group = (~center_mask).reshape(B, 20, 5).to(center_hidden.dtype)
denom = mask_group.sum(dim=2).clamp_min(1.0)
pooled_hidden = (hidden_group * mask_group[..., None]).sum(dim=2) / denom[..., None]
pooled_hidden = torch.where(
    mask_group.sum(dim=2, keepdim=True) > 0,
    pooled_hidden,
    torch.zeros_like(pooled_hidden),
)

value_logits = value_head(pooled_hidden)
value_pred = torch.sigmoid(value_logits)
```

The loss weights mask out target frames whose corresponding pooled context is
fully padded.

Expose:

```python
@dataclass(frozen=True)
class ControlEncoderV4Output:
    value_pred: torch.Tensor           # [B, 20, 4]
    control_memory: torch.Tensor       # [B, 600, D]
    memory_padding_mask: torch.Tensor  # [B, 600]
```

There is no `confidence_head`, no `confidence_pred`, and no derived compound
confidence output in the v4 contract.

Model presets:

```text
Preset A: control_v4_d256_l4_scratch
d_model = 256
heads = 4
layers = 4
ffn_dim = 1024
dropout = 0.1
conv_blocks = 2
conv_kernel_size = 5
batch_size = 16 or 24

Preset B: control_v4_d384_l3_warmstart
d_model = 384
heads = 8
layers = 3
ffn_dim = 1536
dropout = 0.1
conv_blocks = 2
conv_kernel_size = 5
batch_size = 12
```

Warm-start must use a preset whose backbone architecture exactly matches the
checkpoint. A `d256/l4` v4 run must not silently warm-start a `d384/l3`
checkpoint.

## Training Loss

Predictions and targets are bounded:

```text
p: [B, 20, 4] in [0, 1]
y: [B, 20, 4] in [0, 1]
w: [B, 20, 4]
```

Total loss:

```text
L_total =
    L_point
  + 0.5 * L_pool
  + 0.5 * L_peak
```

Rank loss is disabled in the first implementation.

### Point Loss

Use weighted SmoothL1/Huber with:

```text
beta = 0.05
```

Feature scalar weights:

```text
density_level_v4        1.00
density_burst_v4        0.75
chord_ratio_v4          1.00
technical_pressure_v4   0.35
```

If technical pressure becomes a density proxy or harms ordinary-window metrics,
lower `technical_pressure_v4` to `0.15` before removing the feature entirely.

Per-feature point losses:

```text
L_point_j =
  sum(frame_weight_j * smooth_l1(p_j, y_j, beta=0.05))
  / sum(frame_weight_j).clamp_min(eps)
```

If a batch has no effective weight for a feature, that feature contributes
`0.0` for the batch.

Combine features after per-feature normalization:

```text
L_point =
    1.00 * L_point_density
  + 0.75 * L_point_burst
  + 1.00 * L_point_chord
  + 0.35 * L_point_tech
```

Do not multiply `feature_scalar_j` into `frame_weight_j` before dividing by the
effective weight sum; that cancels the scalar. Do not divide `L_point` by `B*T`.

### Pool Loss

Apply 500ms average-pool loss to the stable budget features:

```text
pool features:
  density_level_v4, scalar = 1.0
  chord_ratio_v4,   scalar = 1.0
```

At 100ms target resolution, `avg_pool_500ms` uses `kernel=5`, `stride=5`.
Use masked average pooling. Groups with no valid frames do not contribute.

Masked average pool:

```python
def masked_avg_pool_500ms(x, valid_mask):
    # x: [B, 20]
    # valid_mask: [B, 20] bool
    x_blocks = x.reshape(B, 4, 5)
    m_blocks = valid_mask.reshape(B, 4, 5).to(x.dtype)
    denom = m_blocks.sum(dim=-1)
    block_valid = denom > 0
    avg = (x_blocks * m_blocks).sum(dim=-1) / denom.clamp_min(1.0)
    avg = avg.masked_fill(~block_valid, 0.0)
    return avg, block_valid
```

Pool weights:

```text
pool_weight_density, pool_block_valid_density =
  masked_avg_pool_500ms(target_valid_mask.float(), target_valid_mask)

pool_weight_chord, pool_block_valid_chord =
  masked_avg_pool_500ms(
    loss_confidence(chord_confidence, floor=0.25),
    target_valid_mask
  )
```

The pool loss does not use positive boosts. It applies pool feature scalars only
after each feature's pooled weighted mean is computed.

Pool component:

```text
pred_pool_j, block_valid_j = masked_avg_pool_500ms(p_j, target_valid_mask)
tgt_pool_j, _             = masked_avg_pool_500ms(y_j, target_valid_mask)
pool_weight_j             = pool_weight_j * block_valid_j.float()

L_pool_j =
  sum(pool_weight_j * smooth_l1(pred_pool_j, tgt_pool_j, beta=0.05))
  / sum(pool_weight_j).clamp_min(eps)

L_pool =
  1.0 * L_pool_density + 1.0 * L_pool_chord
```

### Peak Loss

Apply 500ms max-pool loss to peak-sensitive features:

```text
peak features:
  density_burst_v4,      scalar = 1.0
  technical_pressure_v4, scalar = 0.5
```

Masked max-pool:

```python
def masked_max_pool_500ms(x, valid_mask):
    # x: [B, 20]
    # valid_mask: [B, 20] bool
    x_blocks = x.reshape(B, 4, 5)
    m_blocks = valid_mask.reshape(B, 4, 5)
    block_valid = m_blocks.any(dim=-1)
    x_masked = x_blocks.masked_fill(~m_blocks, -torch.inf)
    block_max = x_masked.max(dim=-1).values
    block_max = block_max.masked_fill(~block_valid, 0.0)
    return block_max, block_valid
```

For each peak feature:

```text
pred_peak, block_valid = masked_max_pool_500ms(p_feature, target_valid_mask)
tgt_peak, _            = masked_max_pool_500ms(y_feature, target_valid_mask)
block_weight, _        = masked_max_pool_500ms(frame_weight_feature, target_valid_mask)
peak_weight_j          = block_weight * block_valid.float()
```

Peak component:

```text
L_peak_j =
  sum(peak_weight_j * smooth_l1(pred_peak_j, tgt_peak_j, beta=0.05))
  / sum(peak_weight_j).clamp_min(eps)

L_peak =
  L_peak_burst + 0.5 * L_peak_tech
```

If a batch has no effective peak weight for a feature, that component returns
`0.0`, not `NaN`.

This is mandatory. If positive-slice eval still shows burst underprediction,
add a later `density_burst_v4` top-quantile focal BCE auxiliary, but do not add
it in the first implementation.

## Sampler

Training batches must be feature-balanced:

```text
40% uniform windows
20% high density_burst windows
15% high chord_ratio windows
15% high technical_pressure windows
10% hard difficulty / rare-bin windows
```

Bucket scores:

```text
score_burst = max(y_burst in window)
score_chord = p95(y_chord in window)
score_tech  = max(y_tech in window)
score_level = mean(y_density in window)
```

High-feature buckets are defined by train-split window-score quantiles:

```text
q80_burst = q80(score_burst over train windows where score_burst > 0)
q80_chord = q80(score_chord over train windows where score_chord > 0)
q80_tech  = q80(score_tech  over train windows where score_tech  > 0)

high_burst = score_burst >= q80_burst
high_chord = score_chord >= q80_chord
high_tech  = score_tech  >= q80_tech
```

A window can belong to multiple buckets. The sampler chooses a bucket first,
then samples uniformly from that bucket.

Bucket minimums:

```text
len(high_burst_train) >= max(512, 0.001 * train_window_count)
len(high_chord_train) >= max(512, 0.001 * train_window_count)
len(high_tech_train)  >= max(256, 0.0005 * train_window_count)
```

Formal artifact minimums are stricter:

```text
train high_burst windows >= 2048
train high_chord windows >= 2048
train high_tech windows  >= 1024
val/test total windows per split >= 8192
val/test windows per difficulty bin >= 1024
val/test groups per difficulty bin >= 8
```

If `high_tech < 1024`, mark `technical_pressure_v4` as
`experimental_low_support = true` and fail the formal training artifact. A debug
artifact may still be produced to inspect why support is low.

Rare-bin windows are sampled inversely to train difficulty-bin counts:

```text
bin_weight = 1 / train_window_count_by_bin
normalize mean to 1.0
clip to [0.5, 2.0]
```

Oversampling applies only to training. Validation and test metrics must always
include uniform-distribution eval plus feature-positive slices.

## Evaluation

Eval has three mandatory layers.

### Uniform Eval

On real-distribution windows report, per feature:

```text
MAE
weighted SmoothL1
Pearson
Spearman
lift vs global mean baseline
lift vs difficulty-bin mean baseline
prediction mean
target mean
bias
```

### Positive-Slice Eval

Report feature-positive subsets:

```text
density_burst_top10
chord_ratio_top20
technical_pressure_top10
density_level_by_decile
```

For each subset:

```text
target_mean_subset
pred_mean_subset
subset_bias
top-k recall
positive MAE
lift vs global mean baseline
lift vs difficulty-bin mean baseline
```

Top-k recall is computed by ranking the same eval population twice, once by
target and once by prediction, then reporting:

```text
count(target_top_k intersect pred_top_k) / count(target_top_k)
```

For `top10`, `k` is 10% of the evaluated population; for `top20`, `k` is 20%.

### Difficulty-Stratified Eval

Report all uniform and positive-slice metrics by:

```text
2-3
3-4
4-5
5-6
```

An eval split with zero windows in any required bin is invalid.

### Technical Proxy Checks

Technical pressure must not silently become a density proxy. Report:

```text
technical_vs_density_partial_corr
technical_ordinary_overprediction
technical_false_positive_top10_rate_bottom50
```

Ordinary technical windows:

```text
ordinary_tech = target_technical_pressure_v4 <= q50_tech_valid
ordinary_tech_pred_mean
ordinary_tech_target_mean
ordinary_tech_bias
```

False-positive top10 rate:

```text
technical_false_positive_top10_rate_bottom50 =
  fraction(pred_tech_top10 that lies in target_tech_bottom50)
```

Checkpoint thresholds:

```text
ordinary_overprediction <= train_target_tech_mean + 0.05
technical_false_positive_top10_rate_bottom50 <= 0.20
partial_corr_tech_given_density > 0.05
```

Partial correlation report:

```text
corr(pred_tech, target_tech)
corr(pred_tech, target_density)
partial_corr(pred_tech, target_tech | density_level, difficulty_bin)
```

If `partial_corr <= 0.05` and `corr(pred_tech, target_density)` is high, treat
`technical_pressure_v4` as a density proxy and do not feed it to the mapper
without ablation evidence.

## Model Selection

Do not select checkpoints by total average loss alone.

Primary selection metrics:

```text
density_level_v4 Spearman
density_burst_top10 pred_mean / target_mean
density_burst_top10 recall
chord_ratio_top20 recall
technical_pressure_top10 recall
ordinary-window overprediction for technical_pressure_v4
```

Initial success thresholds:

```text
density_level_v4:
  Spearman >= 0.85

density_burst_v4:
  Spearman >= 0.35
  top10 pred_mean / target_mean >= 0.55
  top10 recall above global and difficulty-bin baselines

chord_ratio_v4:
  Spearman >= 0.45
  top20 recall above global and difficulty-bin baselines

technical_pressure_v4:
  Spearman >= 0.25
  top10 recall above difficulty-bin baseline
  no obvious ordinary-window overprediction
```

These thresholds are training-readiness gates for the standalone control model,
not final mapper-quality guarantees.

## Training Schedule

Warm-start is allowed only when backbone architecture matches exactly:

```text
input_projection
conv_stem
position shape
FiLM modules
encoder layer count
d_model
heads
ffn_dim
```

Otherwise the run must start from scratch or use an explicit partial-load script
that reports loaded and skipped keys.

Compatible warm-start procedure:

```text
load compatible backbone weights
discard old value/confidence heads
initialize new 4-channel value head
```

Optimizer:

```text
optimizer = AdamW
weight_decay = 0.01
grad_clip = 1.0
warmup_steps = 1000
scheduler = cosine
min_lr_ratio = 0.10
max_steps = 16000 to 24000
eval_every = 500
save_every = 500
log_every = 25
```

Scratch learning rates:

```text
head_lr = 3e-4
encoder_lr = 1e-4
```

Warm-start learning rates:

```text
head_lr    = 3e-4
encoder_top_lr = 8e-5
encoder_lower_lr = 3e-5
```

MPS local starting points:

```text
warm-start d384/l3:
  batch_size = 12
  num_workers = 0
  max_cached_maps = 16

scratch d256/l4:
  batch_size = 16
  if stable, try 24
```

Warm-start stage 1:

```text
2000 steps
freeze input_projection + conv_stem + lower encoder layers
train new v4 head + last encoder block + output_norm
head_lr = 3e-4
top_encoder_lr = 5e-5
```

Warm-start stage 2:

```text
remaining steps
unfreeze all
head_lr = 2e-4
encoder_lr = 8e-5
```

Scratch training:

```text
single stage is acceptable
head_lr = 3e-4
encoder_lr = 1e-4
```

Early stopping and checkpoint choice must include positive-slice metrics.

## Artifact Audit Before Training

Before any training run, audit and save:

```text
finite rate
range min/max in [0, 1]
nonzero rate
top10/top20 target means
per-difficulty distributions
feature correlation matrix
saturation rate near 0 and near 1
positive window counts
window bucket counts
split difficulty coverage
```

Saturation checks:

```text
near_one_rate = mean(y >= 0.98 over valid train frames)

density_burst_v4 near_one_rate > 0.05 => fail
technical_pressure_v4 near_one_rate > 0.03 => fail

top10 = y[y >= q90_valid]
if mean(top10) > 0.97 and std(top10) < 0.02:
  fail "top10 collapsed near 1.0"
```

Artifact blockers:

- finite rate below `1.0`;
- target values outside `[0, 1]`;
- train/val/test missing required difficulty bins;
- normalizer fitted on any non-train split;
- `density_burst_v4` positive frames below threshold;
- `technical_pressure_v4` positive frames below threshold;
- positive chord frames below threshold for positive-boost quantiles;
- high-burst, high-chord, or high-technical train bucket below formal minimum;
- `density_burst_v4` top10 collapsed near `1.0`;
- `technical_pressure_v4` top10 collapsed near `1.0`;
- val/test total or per-bin window count below minimum;
- val/test groups per difficulty bin below minimum;
- bounded target, loss weight, sampler bucket, fitted quantile, or baseline
  generated before split assignment.

Training blockers:

- `density_level_v4` Spearman below `0.80` after warmup;
- `density_burst_top10` pred/target ratio stays below `0.35`;
- chord top20 recall not above difficulty-bin baseline;
- technical ordinary overprediction above gate;
- technical false-positive top10 rate bottom50 above gate;
- technical partial correlation given density below gate.

## Mapper Integration Later

When this encoder is used by the mapper, mapper training must not see only
ground-truth controls.

Condition mix:

```text
50% ground-truth v4 controls
25% predicted v4 controls
25% noisy/dropout v4 controls
```

Also apply:

```text
drop entire control condition sometimes
drop individual features sometimes
add small Gaussian noise
```

The three stronger mapper conditions are:

```text
density_level_v4
density_burst_v4
chord_ratio_v4
```

`technical_pressure_v4` remains weak-condition plus ablation until mapper output
quality proves it improves texture without causing unwanted jack overuse.

## Required Experiments

Do not freeze these settings without a small ablation.

Experiment A: density burst raw definition.

Run raw-feature audits before training:

```text
A1: L_short = 0.40s, L_med = 3.0s
A2: L_short = 0.50s, L_med = 3.0s
A3: L_short = 0.75s, L_med = 3.0s
```

Selection checks:

```text
bounded_pos_q95 - bounded_pos_q70 >= 0.10
near_one_rate <= 0.15
top10 windows visually correspond to local accents, not every ordinary onset
density_burst_top10 not identical to density_level_top10
```

Experiment B: density burst normalizer slope.

```text
run slopes: 2.5, 3.0, 4.0
compare:
  density_burst_top10 pred_mean / target_mean
  density_burst_top10 recall
  near_one_rate
  top10 std
  ordinary burst overprediction
```

Experiment C: technical pressure support.

```text
C1: L_tech = 1.00s
C2: L_tech = 1.25s
C3: L_tech = 1.50s
```

Selection checks:

```text
positive window count enough
not purely density-correlated
top windows are recognizably compact/repeated-lane patterns
```

Experiment D: loss ablation.

```text
run:
  point only
  point + pool
  point + pool + peak

success:
  peak loss improves density_burst_top10 pred/target ratio
  without excessive ordinary burst bias
```

Experiment E: sampler ablation.

```text
E1: uniform sampler
E2: feature-balanced sampler
```

Expected result:

```text
E2 improves burst/chord/tech positive-slice recall
E2 must not introduce unacceptable ordinary overprediction
```

Experiment F: technical pressure usefulness.

```text
F1: without technical_pressure_v4
F2: with technical_pressure_v4 weight 0.15
F3: with technical_pressure_v4 weight 0.35
```

Standalone metrics are not enough for technical pressure. Mapper ablation must
check whether it improves texture, causes unwanted jack overuse, or degrades
clean stream.

Mapper integration ablation:

```text
G1: density + burst + chord
G2: density + burst + chord + technical
```

`technical_pressure_v4` becomes a stronger mapper condition only if G2 improves
texture without unwanted jack overuse or clean-stream degradation.

## Implementation Order

1. Build control_v4 map manifest from the Stage 2 index.
2. Assign group-aware train/val/test split.
3. Extract raw 100ms control_v4 feature cache for all split-assigned maps.
4. Fit normalizers, positive thresholds, difficulty weights, and baselines using train split only.
5. Materialize bounded control_v4 timeseries and final frame-level loss weights for all splits.
6. Build feature-balanced window index from final bounded timeseries.
7. Run artifact audit and fail-fast gates.
8. Implement `ControlEncoderV4`.
9. Implement v4 loss with masked point, pool, and peak terms.
10. Implement uniform, positive-slice, difficulty-stratified, and technical proxy eval.
11. Run synthetic smoke tests.
12. Run tiny overfit.
13. Run standalone control training and select by positive-slice metrics.

All Python commands in this repository must be run through `uv run`.
