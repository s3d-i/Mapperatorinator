---
date: 2026-04-26
drafted_on: 2026-04-26
effective_on: 2026-04-26
pinned_commit: af829dc2b72706b83d79dd5aa269d0989e5c2305
updates:
  - "[2026-04-25-control-v2-feature-diagnostics.md](2026-04-25-control-v2-feature-diagnostics.md)"
  - "[2026-04-25-control-v2-requested-missing-diagnostics.md](2026-04-25-control-v2-requested-missing-diagnostics.md)"
source_links:
  - "[control_v3.py](../../../train/stage1_oracle/features/control_v3.py)"
  - "[control_v3_artifact.py](../../../train/stage1_oracle/features/control_v3_artifact.py)"
  - "[control_v3_audit.py](../../../train/stage1_oracle/features/control_v3_audit.py)"
  - "[control_feature_visualization_v3.ipynb](../../../train/notebooks/control_feature_visualization_v3.ipynb)"
artifact_links:
  - "[control_feature_visualization_v3_full_eligible_executed.ipynb](../../../train/artifacts/features/control_v3_audit/control_feature_visualization_v3_full_eligible_executed.ipynb)"
  - "[control_v3_artifact_metadata_4k_no_timing_anomalies_2to6.json](../../../train/artifacts/features/control_v3_artifact_metadata_4k_no_timing_anomalies_2to6.json)"
  - "[control_v3_timeseries_4k_no_timing_anomalies_2to6.parquet](../../../train/artifacts/features/control_v3_timeseries_4k_no_timing_anomalies_2to6.parquet)"
  - "[control_v3_map_summary_4k_no_timing_anomalies_2to6.parquet](../../../train/artifacts/features/control_v3_map_summary_4k_no_timing_anomalies_2to6.parquet)"
  - "[control_v3_section_audit_8s_stride4.parquet](../../../train/artifacts/features/control_v3_audit/control_v3_section_audit_8s_stride4.parquet)"
test_links:
  - "[test_train_stage1_control_v3_features.py](../../../tests/test_train_stage1_control_v3_features.py)"
  - "[test_train_stage1_control_v3_artifact.py](../../../tests/test_train_stage1_control_v3_artifact.py)"
  - "[test_train_stage1_control_v3_audit.py](../../../tests/test_train_stage1_control_v3_audit.py)"
  - "[test_control_feature_visualization_v3_notebook.py](../../../tests/test_control_feature_visualization_v3_notebook.py)"
analysis_scope: "基于已执行 v3 notebook、保存的 v3 artifacts、当前 checkout 中的 v3 feature/audit/test 代码，以及本结论中列出的 audit summaries；本文不重新跑全量 parquet 计算。"
artifact_metadata_code_commit: 2349c32ce68d7c736c777bee1d28989763c59fb9
artifact_metadata_code_dirty: true
source_note: "当前 pinned checkout 的 control_v3.py 已将 jack_streak_confidence 纳入 control_confidence；本文仍建议训练时优先使用 per-family confidence，而不是只依赖 control_confidence。"
---

# Control V3 Feature Readiness For Control Encoder Training

## 总判断

可以开始训练 control encoder，但不要把所有 feature 当成同等干净的 control axis。

更准确地说：

- `v3` 已经解决 `v2` 的两个大问题：
  - `repeat_rhythm` 已从 model channels 删除，原来的 saturation blocker 消失。
  - `ln_change_rate` 从 raw model-facing value 改成 `ln_change_rate_gated`，contract 明显更合理。
- `101/103` gates passed 是强 positive signal，说明现在 feature system 已经不是 “audit 不过不能用” 的状态。
- 但剩下两个问题不是完全无关紧要：
  - `ln_change_rate_gated` 仍有 `3.5024%` sections 的 `low_support_high_value`。
  - `density_burst` 在 section-level p95 上几乎 saturated，且是 visible pointwise failure queue 的主角。

所以结论是：

`v3` features 已经可以用于训练 control encoder；但 `ln_change_rate_gated` 和
`density_burst` 应该作为 yellow features 处理，训练时需要
confidence/support weighting 或较低 loss weight。其余大部分 features 可以视为
ready。

## 数据完整性和 artifact 状态

这部分已经基本过关：

| item | result |
| --- | ---: |
| maps | 10,977 |
| timeseries rows | 16,858,417 |
| section rows | 407,491 |
| section unique maps | 10,977 |
| artifact error_count | 0 |
| schema_version | 3 |
| feature_contract_version | 3 |
| model channels | 20 |
| finite_model_stats | 1.000000 |
| valid_fraction mean | 0.993986 |
| required feature-aligned diagnostics missing columns | 0 |

这说明现在不是 extractor/runtime 层面的失败。feature-aligned diagnostics 也已经
存在，不再有 `v2` 那种 “audit 可能假通过，因为 peak-aligned diagnostics 缺失”
的问题。

## Model Channels

当前 model-facing channels 是：

```text
density_level
density_burst
hold_occupancy
ln_change_rate_gated
chord_ratio
jack_excess
jack_streak_exposure
hand_balance_signed
hand_imbalance_abs
repeat_exact
repeat_shift
repeat_motion
density_confidence
ln_change_confidence
chord_confidence
jack_confidence
jack_streak_confidence
hand_confidence
repeat_confidence
control_confidence
```

这比 `v2` 更合理：

- `repeat_rhythm` 已删除。
- `ln_change_rate_raw` 不再是 model-facing channel。
- `ln_change_rate_gated` 进入 model-facing。
- `ln_change_rate_raw` 被降级为 diagnostic-only。
- confidence channels 保留，适合训练时做 loss weighting / masking。

这个 contract 现在基本成立。

## 全局 Hard Failures

最重要的两个全局 confidence/support gates：

| gate | value | threshold | status |
| --- | ---: | ---: | --- |
| pointwise_high_value_low_confidence | 0.000145 | <= 0.02 | pass |
| low_support_high_value | 0.003502 | <= 0.02 | pass |

这和 `v2` 的情况完全不同。`v2` 的大问题是 section-level
`low_confidence_high_value` 到了 `0.5646` feature-sections / section，而且解释
混乱。`v3` 拆分后，真正 pointwise high-value low-confidence 只有 `0.0145%`
feature-section rate。这已经很低。

所以：`v3` 的 confidence audit 已经可以支撑训练，不再是 blocker。

但要注意，global feature-section denominator 会稀释单个 feature 的问题，所以
还要看 per-feature summary。

## Feature Readiness

### Green: 可以直接进入 control encoder

这些 feature 可以视为 ready。

#### density_level

| metric | value |
| --- | ---: |
| high_value_section_rate | 0.594062 |
| pointwise_high_value_low_conf_rate | 0 |
| low_support_high_value_rate | 0 |
| confidence_at_peak_p10 | 0.999937 |
| confidence_at_peak_median | 1.000000 |

`density_level` 很稳。它覆盖面大、confidence 高、没有 pointwise/support 问题。

结论：ready。

#### hold_occupancy

从 saturation report 看：

| metric | value |
| --- | ---: |
| p95 | 0.450625 |
| p99 | 0.612771 |
| max | 1.000000 |

`hold_occupancy` 没有明显 corpus-level saturation。它是 interpretable LN pressure
feature。

结论：ready。

#### chord_ratio

| metric | value |
| --- | ---: |
| high_value_section_rate | 0.023456 |
| pointwise_high_value_low_conf_rate | 0 |
| low_support_high_value_rate | 0 |
| confidence_at_peak_p10 | 0.862278 |
| confidence_at_peak_median | 0.979397 |
| near_high_rate | 0.000010 |

`chord_ratio` 现在很干净。`v2` 里 top review queue 给人的错觉是 `chord_ratio`
很危险；`v3` summary 证明它只是 rare severe cases，不是 global issue。

结论：ready。

#### jack_excess

| metric | value |
| --- | ---: |
| high_value_section_rate | 0.002464 |
| pointwise_high_value_low_conf_rate | 0 |
| low_support_high_value_rate | 0 |
| p95 | 0.058731 |
| p99 | 0.278366 |
| max | 0.903495 |

`jack_excess` 是 sparse tail feature。它不会经常激活，但激活时很有意义。没有
pointwise confidence failure，也没有 support failure。

结论：ready，但训练时要按 sparse feature 处理。不要期待它像 `density_level`
一样提供 dense gradient。可以给 positive sections 更高 sampling probability，
或者用 feature-balanced loss。

#### jack_streak_exposure

| metric | value |
| --- | ---: |
| high_value_section_rate | 0.162548 |
| high_value_unique_map_rate | 0.338708 |
| high_value_difficulty_mean | 5.112609 |
| pointwise_high_value_low_conf_rate | 0 |
| low_support_high_value_rate | 0 |
| window_only_low_conf_rate | 0.047248 |

这是一个很有价值的 difficulty/technical pressure feature。它 high-value
sections 的 mean difficulty 是 `5.11`，说明它确实捕捉到高难 technical strain。

`window_only_low_conf_rate = 4.72%` 不应视为 blocker，因为 peak 本身不是
low-confidence；只是 section 里其他地方 confidence 低。

结论：ready，但训练 loss 应乘 `jack_streak_confidence`。

#### hand_imbalance_abs

| metric | value |
| --- | ---: |
| high_value_section_rate | 0.001178 |
| pointwise_high_value_low_conf_rate | 0 |
| low_support_high_value_rate | 0 |
| confidence_at_peak_p10 | 0.603239 |
| confidence_at_peak_median | 0.762777 |
| near_high_rate | 0 |

`hand_imbalance_abs` 是 rare feature，但没有 audit failure。它适合作为 hand
imbalance magnitude signal。

结论：ready，但 sparse。

#### repeat_exact, repeat_shift, repeat_motion

| feature | high_value_section_rate | confidence_at_peak_p10 | confidence_at_peak_median | near_high_rate | near_zero_rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| repeat_exact | 0.002218 | 0.964407 | 0.998589 | 0.000037 | 0.002834 |
| repeat_shift | 0.008010 | 0.965376 | 0.998737 | 0.000044 | 0.002790 |
| repeat_motion | 0.003404 | 0.964547 | 0.998582 | 0.000037 | 0.002805 |

这些是 `v3` 里最干净的一组：

- 没有 pointwise failure。
- 没有 low-support failure。
- 没有 near-high saturation。
- 也没有 structurally zero。

删除 `repeat_rhythm` 后，repeat family 现在合理了。

结论：ready。

### Yellow: 可以用，但训练时要小心

#### ln_change_rate_gated

这是 `v3` 里最重要的 yellow feature。

| metric | value |
| --- | ---: |
| high_value_section_rate | 0.627690 |
| high_value_unique_map_rate | 0.893231 |
| pointwise_high_value_low_conf_rate | 0 |
| window_only_low_conf_rate | 0.393506 |
| low_support_high_value_rate | 0.035024 |
| low_support_unique_map_rate | 0.401294 |
| confidence_at_peak_median | 0.805172 |
| section_confidence_p20_median | 0.000000 |

Interpretation:

- `pointwise_high_value_low_conf_rate = 0` 是好消息，说明 high-value peak 本身
  没有低 confidence 问题。
- `low_support_high_value_rate = 3.5024%` 是剩下的主要 issue。Top low-support
  examples 的 `n_eff_at_peak` 很多在 `2.96-2.99`，threshold 是 `3.0`，属于贴着
  threshold 的 failure，不是灾难性低 support。
- `window_only_low_conf_rate = 39.35%` 很高，但这不是 hard failure。它说明
  section 内 LN-change support 很局部：peak 位置可以可信，但 section 其他区域
  没有 LN-change evidence。

所以 `ln_change_rate_gated` 不应该再被视为 `v2` 那种 blocker。但它也不是完全
green。

结论：`ln_change_rate_gated` 可以训练，但必须 confidence/support-aware。

训练建议：

```text
loss_weight_for_ln_change = ln_change_confidence
```

或者更保守：

```text
mask out ln_change_rate_gated targets where ln_change_n_eff < 3.0
```

如果不想直接丢样本，可以做 soft weighting：

```text
ln_change_loss_weight = clip((ln_change_n_eff - 2.0) / 1.0, 0, 1) * ln_change_confidence
```

不建议把 `ln_change_rate_gated` 当成和 `density_level` 一样干净的 dense control
axis。

#### density_burst

这是另一个 yellow feature。

| metric | value |
| --- | ---: |
| high_value_section_rate | 0.975882 |
| high_value_unique_map_rate | 1.000000 |
| pointwise_high_value_low_conf_rate | 0.001453 |
| window_only_low_conf_rate | 0.006668 |
| confidence_at_peak_p10 | 0.999646 |
| confidence_at_peak_median | 0.999999 |
| p95 | 0.999954 |
| p99 | 0.999999 |
| max | 1.000000 |

This is not a confidence problem. Confidence is excellent.

The issue is semantic/calibration: `density_burst_p95` is almost always near `1`
at section level. That means `density_burst` is probably not a useful
section-level continuous axis. It may still be useful as a local time-series
transient feature, but its section-level p95 is almost binary/saturated.

The review queue also shows `density_burst` as the top pointwise bucket. Some
examples have `confidence_at_peak = 0` or very low `confidence_top_value_mean`,
but the overall rate is only `0.1453%` sections, so this is not a blocker.

结论：`density_burst` can be used, but do not give it high control weight
initially.

For encoder training:

- keep it as a local auxiliary channel;
- use lower loss weight, for example `0.25-0.5x` of `density_level`;
- do not use section-level `p95(density_burst)` as a primary control target;
- consider replacing/augmenting it later with a less saturated burst metric,
  e.g. burst-over-baseline or short/medium density ratio.

#### hand_balance_signed

`hand_balance_signed` is direction, not magnitude. The audit summary focuses on
`hand_imbalance_abs`, which is correct.

Use `hand_balance_signed` only together with:

- `hand_imbalance_abs`
- `hand_confidence`

It should not be optimized as a standalone “pressure magnitude” feature. Also
make sure mirror augmentation flips the sign.

结论：usable, but direction-only.

#### control_confidence

This is useful as a summary confidence channel, but it should not be the only
weighting signal.

Use per-family confidence for feature losses:

- `density_confidence`
- `ln_change_confidence`
- `chord_confidence`
- `jack_confidence`
- `jack_streak_confidence`
- `hand_confidence`
- `repeat_confidence`

Implementation note: the current pinned checkout includes `jack_streak_confidence`
when computing `control_confidence`. That resolves the earlier caveat that this
signal might be missing from the global confidence average. Even so,
`control_confidence` should still be treated as a rough summary only.

结论：useful, but do not rely on it alone.

## Diagnostic-Only Feature

### ln_change_rate_raw

This should not be used as a training target for the control encoder.

Its contract is:

```text
raw_value_with_side_confidence
```

Scope is diagnostic. That is correct.

Use it for inspection, calibration, and debugging. Train on:

```text
ln_change_rate_gated
```

not:

```text
ln_change_rate_raw
```

## What The Two Failed Gates Mean

The saved notebook display truncates the middle of the 103-row gate table, so
the exact two failed row names are not quoted from visible output. But from the
visible feature audit summary, the only metrics large enough to plausibly
explain the two failures are:

1. `ln_change_rate_gated` per-feature low-support high-value
   - `low_support_high_value_rate = 0.035024`
   - affects `40.13%` of maps at least once
   - top failures are very close to threshold, usually `n_eff_at_peak ~= 2.96-2.99`
     vs threshold `3.0`
2. `density_burst` pointwise / calibration issue
   - `pointwise_high_value_low_conf_rate = 0.001453`
   - `density_burst_p95 ~= 0.999954`
   - `high_value_section_rate = 0.975882`

Neither looks like “do not train” failure.

They mean:

- `ln_change_rate_gated`: train with support/confidence weighting.
- `density_burst`: train as low-weight local auxiliary, not as major global
  control axis.

## Is This Ready For A Control Encoder?

Yes, with feature-specific loss handling.

Train the encoder now, but not with a flat unweighted MSE over all 20 channels.

### Value Feature Loss

Use feature-specific confidence weighting:

| value feature | suggested loss weight |
| --- | --- |
| density_level | 1.0 |
| hold_occupancy | 1.0 |
| chord_ratio | 1.0 |
| repeat_exact | 1.0, but sparse-balanced |
| repeat_shift | 1.0, but sparse-balanced |
| repeat_motion | 1.0, but sparse-balanced |
| jack_excess | 0.75-1.0, sparse-balanced |
| jack_streak_exposure | 1.0 with jack_streak_confidence |
| hand_imbalance_abs | 0.75-1.0, sparse-balanced |
| hand_balance_signed | 0.5-0.75, only with hand confidence |
| ln_change_rate_gated | 0.5-0.75 initially |
| density_burst | 0.25-0.5 initially |

### Confidence Weighting

Use per-feature confidence, not only `control_confidence`:

| value feature | confidence/support signal |
| --- | --- |
| density_level, density_burst | density_confidence |
| ln_change_rate_gated | ln_change_confidence + support weight |
| chord_ratio | chord_confidence |
| jack_excess | jack_confidence |
| jack_streak_exposure | jack_streak_confidence |
| hand_balance_signed/abs | hand_confidence |
| repeat_exact/shift/motion | repeat_confidence |

### Sparse Features

For sparse/high-value rare features, use either positive-section oversampling or
feature-balanced loss.

Otherwise the encoder can minimize loss by learning mostly zeros for
`jack_excess`, `hand_imbalance_abs`, and repeat features.

## Feature-By-Feature Readiness Table

| feature | readiness | reason |
| --- | --- | --- |
| density_level | Ready | finite, high confidence, no pointwise/support failure |
| density_burst | Use with caution | confidence good, but section p95 almost saturated |
| hold_occupancy | Ready | stable LN occupancy signal |
| ln_change_rate_gated | Use with caution | pointwise confidence fixed, but 3.5% low-support high-value |
| chord_ratio | Ready | rare high values, strong confidence, no failure |
| jack_excess | Ready, sparse | tail feature, no support/conf failure |
| jack_streak_exposure | Ready with confidence weighting | strong high-difficulty axis, no pointwise failure |
| hand_balance_signed | Usable direction channel | use with hand_imbalance_abs and hand_confidence |
| hand_imbalance_abs | Ready, sparse | no saturation/failure |
| repeat_exact | Ready | clean after removing repeat_rhythm |
| repeat_shift | Ready | clean, no saturation |
| repeat_motion | Ready | clean, no saturation |
| confidence channels | Ready as loss weights / auxiliary inputs | use per-family confidence, not just global confidence |
| ln_change_rate_raw | Diagnostic only | do not train as model-facing control |

## Final Recommendation

Start training the control encoder with `v3`.

But treat it as:

```text
ready-for-training
```

not yet:

```text
final frozen feature contract
```

The only things to change before or during the first training run:

1. Use confidence-weighted loss, especially for `ln_change_rate_gated`,
   `jack_streak_exposure`, hand, and repeat features.
2. Downweight `density_burst` because section-level p95 is saturated.
3. Downweight or mask low-support `ln_change_rate_gated` samples, especially
   where `ln_change_n_eff < 3`.
4. Do not train on `ln_change_rate_raw`.
5. Use sparse-feature balancing for `jack_excess`, `hand_imbalance_abs`, and
   repeat features.
6. Do not let `control_confidence` replace per-feature confidence.

The `v3` audit result is good enough to move forward. The remaining issues are
training-weighting issues, not extractor blockers.
