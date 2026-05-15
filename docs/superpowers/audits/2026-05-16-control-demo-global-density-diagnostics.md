---
pinned_commit: 3ba4d409e729815f4029c10ae553713f72ccf1bf
status: diagnostic
date: 2026-05-16
owner: s3d-i
run: stage2_control_demo_global_d384_l3_stride16_b6
checkpoint: train/artifacts/runs/stage2_control_demo/stage2_control_demo_global_d384_l3_stride16_b6/checkpoints/checkpoint_step_002000.pt
---

# Control Demo Global Density Diagnostics

## Summary

The `control_demo_global` step-2000 checkpoint does not provide strong evidence
that global context is helping yet.

The stronger diagnosis is:

1. The global fusion path is still near its weak initialization.
2. The density demo loss underweights low-density and rest frames because it
   multiplies point loss by `density_confidence`.
3. The eval split is too narrow and audio-leaky for generalization claims.

This is therefore not primarily a model-size problem. Before spending a full
12k-step run on this exact setup, fix the density loss/eval contract and use a
proper audio-group stratified eval split.

## Current Confidence Calculation

The density confidence used by `control_demo_global` comes from the saved
`control_v3` target, not from the model. The demo target keeps exactly:

```text
control_demo_target = [density_level, density_confidence]
```

The source formula is inherited from `train/stage1_oracle/features/control_v3.py`
via `control_v2.density_features`.

For each 100ms grid frame, density support is computed from grouped onset times
using the medium triangular kernel:

```text
K_L(u) = max(0, 1 - abs(u) / L) / L
L = density_L_med = 3.0s
```

The density value uses chord-weighted support:

```text
event_weight_i = chord_size_i ** density_alpha
density_alpha = 1.15
d_med(t) = sum_i event_weight_i * K_3.0s(t - onset_i)
density_level(t) = log1p(d_med(t))
```

The confidence, however, is based on effective support count from kernel weights,
not the chord-weighted density magnitude:

```text
sum_w(t)  = sum_i K_3.0s(t - onset_i)
sum_w2(t) = sum_i K_3.0s(t - onset_i) ** 2
n_eff(t)  = sum_w(t) ** 2 / max(sum_w2(t), 1e-12)
```

Then:

```text
density_n_eff_min = 1.0
density_gate_scale = 2.0

x = (n_eff - density_n_eff_min) / density_gate_scale
density_confidence = clip01(1.0 - exp(-max(0.0, x)))
```

Equivalent:

```text
if n_eff <= 1.0:
  density_confidence = 0.0
else:
  density_confidence = 1.0 - exp(-(n_eff - 1.0) / 2.0)
```

After feature extraction, all feature and confidence channels are edge-neutralized
to `0.0` outside `valid_control_mask`, where:

```text
valid_control_mask = grid >= first_hit_start and grid <= last_object_end
```

Stage 2 then interpolates the 100ms control timeseries onto 20ms target frame
centers for the 100-frame demo target window.

## Current Demo Loss

`ControlDemoModelLoss.value_weights` computes:

```text
weight = target_valid_mask * density_confidence * density_loss_weight
```

So low-support low-density frames can have little or zero loss weight, even when
they are valid playable-map frames. This is why the model can improve weighted
SmoothL1 while still miscalibrating rests and low-density sections.

For v4, density should not use this confidence as the primary point-loss weight.
The v4 density-level base weight should be:

```text
w_density = valid_control_mask
```

Density confidence can remain artifact metadata and diagnostics, and can still
be useful for burst or support-sensitive auxiliary targets. It should not erase
ordinary low-density supervision for `density_level_v4`.

## Checkpoint Loss History

Saved eval points for
`stage2_control_demo_global_d384_l3_stride16_b6`:

| step | eval loss | eval density MAE |
| ---: | ---: | ---: |
| 1 | 0.08808 | 0.21561 |
| 500 | 0.08967 | 0.21405 |
| 1000 | 0.08551 | 0.21498 |
| 1500 | 0.08400 | 0.21535 |
| 2000 | 0.08281 | 0.22587 |

The weighted loss improves slowly after step 1000, but plain MAE does not
improve. This already suggests calibration drift under the confidence-weighted
loss.

## Prediction Diagnostics At Step 2000

Eval inference was run over the saved eval split:

```text
eval windows: 1,036
eval frames: 102,854
```

Frame-level metrics:

```text
target_mean: 2.25534
pred_mean:   2.35022
bias:        0.09488
MAE:         0.22587
RMSE:        0.41627
Pearson:     0.84217
Spearman:    0.89095
```

The model ranks density well, but calibration is compressed:

```text
target q01/q05/q50/q95/q99: 0.000 / 0.000 / 2.446 / 2.929 / 3.071
pred   q01/q05/q50/q95/q99: 0.991 / 1.539 / 2.414 / 2.867 / 2.939
```

Target-decile calibration:

| target decile | target mean | pred mean | bias |
| ---: | ---: | ---: | ---: |
| 1 | 0.538 | 1.568 | 1.030 |
| 2 | 1.799 | 1.978 | 0.179 |
| 3 | 2.087 | 2.137 | 0.050 |
| 4 | 2.275 | 2.268 | -0.007 |
| 5 | 2.392 | 2.367 | -0.025 |
| 6 | 2.494 | 2.482 | -0.012 |
| 7 | 2.567 | 2.533 | -0.034 |
| 8 | 2.646 | 2.602 | -0.043 |
| 9 | 2.802 | 2.741 | -0.061 |
| 10 | 2.954 | 2.825 | -0.129 |

The key failure is not rank ordering. It is low-density overprediction plus
high-density underprediction.

## Confidence Weighting Effect

On the same eval split, density confidence by target decile:

| target decile | target mean | confidence mean | zero-confidence rate | loss-weight share |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.538 | 0.415 | 0.536 | 0.044 |
| 2 | 1.799 | 0.999 | 0.000 | 0.106 |
| 3 | 2.087 | 1.000 | 0.000 | 0.106 |
| 4 | 2.275 | 1.000 | 0.000 | 0.106 |
| 5 | 2.392 | 1.000 | 0.000 | 0.106 |
| 6 | 2.494 | 1.000 | 0.000 | 0.106 |
| 7 | 2.567 | 1.000 | 0.000 | 0.106 |
| 8 | 2.646 | 1.000 | 0.000 | 0.106 |
| 9 | 2.802 | 1.000 | 0.000 | 0.106 |
| 10 | 2.954 | 1.000 | 0.000 | 0.106 |

The bottom decile gets only about `4.4%` of total weighted density loss despite
being `10%` of valid frames. This directly explains why low-density calibration
is weak.

## Global Branch Diagnosis

The global fusion gates stayed almost unchanged:

| checkpoint | fusion 0 gate | fusion 1 gate |
| --- | ---: | ---: |
| step 1 | 0.05021 | 0.05021 |
| step 500 | 0.04996 | 0.05006 |
| step 1000 | 0.05026 | 0.05039 |
| step 1500 | 0.05054 | 0.05071 |
| step 2000 | 0.05080 | 0.05099 |

An ablation using the same step-2000 weights showed that bypassing the global
modules was slightly better on this eval set:

| mode | weighted loss | MAE | Pearson |
| --- | ---: | ---: | ---: |
| full global | 0.08281 | 0.22587 | 0.84217 |
| local only, same weights | 0.08112 | 0.22329 | 0.84172 |

This means the current run has not really validated global context. The local
pretrained path is carrying the checkpoint.

## Eval Split Diagnosis

The current eval split is too small and not audio-disjoint:

```text
source windows: 707,767
train windows:  706,731
eval windows:   1,036

train beatmaps: 9,227
train audio:    3,640

eval beatmaps:  15
eval audio:     15
eval 5-6 windows: 0
eval audio overlap with train: 14 / 15
```

The split is beatmap-disjoint but mostly not audio-disjoint. Alternate
difficulties of the same audio can appear across train and eval. This makes the
reported eval useful as a smoke metric, but not as a reliable generalization
metric.

## Dataset Target Distribution

The full `control_v3` density timeseries is not obviously too narrow:

```text
rows: 16,858,417
density_level mean: 2.36958
density_level std:  0.59029

q01/q05/q50/q95/q99/max:
0.000 / 1.334 / 2.458 / 3.106 / 3.272 / 3.515

density_level <= 0.05: 1.91%
density_level >= 2.0:  81.63%
```

The dataset is skewed toward active mapping frames, but there are still enough
low-density frames that a valid-frame-weighted density loss should learn them
better than the current confidence-weighted demo loss.

## Recommendation

For the next v4 control run:

1. Use audio-group train/val/test splitting before normalizer fitting.
2. Make eval stratified by difficulty bin, including `5-6`.
3. Increase eval size well beyond 15 beatmaps.
4. Train `density_level_v4` with `w_density = valid_control_mask`.
5. Keep confidence as metadata/diagnostics for density, not as the point-loss
   gate.
6. Add calibration metrics: prediction quantiles, target-decile bias,
   low-density MAE, Pearson, Spearman, and baseline lift.
7. Test global context with an explicit local-vs-global ablation after the loss
   and split are fixed.

The current evidence points to loss/eval-contract problems first, global-branch
training recipe second, and raw model capacity third.
