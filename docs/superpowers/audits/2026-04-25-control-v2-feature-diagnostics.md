---
date: 2026-04-25
drafted_on: 2026-04-25
effective_on: 2026-04-25
pinned_commit: a8045ec59c65884d8cb6304ca54289891a661ab9
source_links:
  - "[control_feature_visualization_v2_full_eligible_executed.ipynb](../../../train/artifacts/features/control_v2_audit/control_feature_visualization_v2_full_eligible_executed.ipynb)"
  - "[control_v2.py](../../../train/stage1_oracle/features/control_v2.py)"
  - "[control_v2_artifact.py](../../../train/stage1_oracle/features/control_v2_artifact.py)"
source_note: "原始 diagnostics 提到 control_v2(1).py；当前 checkout 未找到该文件，本文链接的是仓库内对应的 control_v2.py。"
---

# Control V2 Feature Diagnostics

以下结论基于已执行的
`control_feature_visualization_v2_full_eligible_executed.ipynb` 输出，以及
`control_v2.py`、`control_v2_artifact.py` 中的特征定义与 artifact 生成逻辑。
没有重新跑全量 parquet 原始数据，所以结论严格限于 notebook 已记录的执行结果。

## 总结判断

`control_v2` 的大多数 feature 在 synthetic tests、perturbation tests、全量
section audit 上表现可用：没有运行错误，数值有限，section 覆盖率高，核心
operator 没有明显退化成 density proxy。

但这次审计不能算完全通过。最主要的两个问题是：

1. `repeat_rhythm` 明显过饱和：`repeat_rhythm_p95 >= 0.95` 的 section 比例达到
   22.83%，超过审计阈值 5%。
2. `low_confidence_high_value` gate 失败：高 feature value 同时低 confidence 的
   feature-section 命中率是 0.5646 / section，远高于阈值 0.10。这说明
   confidence gating 或审计口径需要重新看。

所以结论不是 “V2 已经完全健康”，而是：feature extractor 的方向性基本正确，
但 corpus-level calibration 仍有两个硬问题，尤其是 `repeat_rhythm` 与
confidence/value mismatch。

## 数据范围与运行状态

这次全量 audit 使用的是 eligible 4K mania corpus：

| 项目 | 结果 |
| --- | ---: |
| maps | 10,977 |
| difficulty range | 2.0-6.0 |
| timeseries rows | 16,858,417 |
| section audit rows | 407,491 |
| section size / stride | 8s window, 4s stride |
| artifact error_count | 0 |
| section audit error_df | empty |
| model channels | 21 |

21 个 `MODEL_FEATURE_NAMES` 包括 13 个 value features 和 8 个 confidence
features：

```text
density_level, density_burst, hold_occupancy, ln_change_rate, chord_ratio,
jack_excess, jack_streak_exposure, hand_balance_signed, hand_imbalance_abs,
repeat_exact, repeat_shift, repeat_motion, repeat_rhythm, plus confidence channels.
```

运行层面没有失败：artifact metadata 里 `error_count = 0`，section audit 的
`error_df.head(20)` 是 empty。

## Synthetic Probes 结论

Synthetic expectations 23/23 全部通过。

这说明 feature 的基本语义方向是对的：

- rest 保持中性：`density_level = 0`，`repeat_exact = 0`。
- `normal_jack` 能触发 `jack_excess` 和 `repeat_exact`：
  - `jack_excess_p95 = 0.6168`
  - `repeat_exact_p95 = 0.9924`
- `single_stair` / `double_stair` 不会误判成 jack：
  - `single_stair jack_excess_p95 = 0`
  - `double_stair jack_excess_p95 = 0`
- `double_stair` 能触发 `repeat_shift`：
  - `repeat_shift_p95 = 0.5242`
- LN fixture 能触发对应信号：
  - `ln_hold_section hold_occupancy_p95 = 0.75`
  - `ln_release_heavy_section ln_change_rate_p95 = 1.9095`
- `mixed_ln_chord_section` 同时保留 LN pressure 和 chord pressure：
  - `hold_occupancy_p95 = 0.5`
  - `chord_ratio_p95 = 0.2752`

一个需要注意的点：`repeat_rhythm` 在很多规律 timing 的 fixture 上都很高，例如
`random_stream repeat_rhythm_p95 = 0.9997`。这不是 synthetic test 失败，但它
预示了后面 corpus audit 里的过饱和问题。

## Perturbation Tests 结论

Perturbation expectations 26/26 全部通过。

主要行为符合设计：

- `mirror_columns` 保持大多数 feature invariant，`hand_balance_signed` 只做 sign
  flip。
- `time_stretch_1p1` 会降低 density：
  - `single_stream density_level_p95 delta = -0.0750`
- `column_shuffle` 保持 onset density 和 chord size，但降低 same-column / exact
  recurrence：
  - `normal_jack jack_excess delta = -0.3673`
  - `normal_jack repeat_exact delta = -0.7848`
- `singles_to_doubles` 会提高 density 和 chord ratio，但不制造 jack：
  - `density_level delta = +0.6807`
  - `chord_ratio delta = +0.2986`
  - `jack_excess delta = 0`
- `force_same_column` 会明显提高 jack 和 exact repeat：
  - `jack_excess delta = +0.7226`
  - `repeat_exact delta = +0.7717`
- LN tail 操作是局部的：
  - removing LN tails: `hold_occupancy delta = -0.2494`
  - extending LN tails: `hold_occupancy delta = +0.0469`
  - onset density / chord ratio 基本不变。

这里的结论是：operator 的局部敏感性是合理的，互相污染不严重。

但有一个风险信号：`normal_jack_time_stretch_1p1` 让 `jack_excess_p95` 从
0.6168 掉到 0。虽然 expectation 只要求 “not increase”，所以测试通过，但这说明
`jack_excess` 对 timing gap / tempo scaling 非常敏感。这个行为是否符合 mapper
intuition，需要人工确认。

## Corpus Distribution 结论

Section-level distribution 显示大部分 feature 没有系统性 NaN 或爆炸。

### Density / Chord / LN

- `density_level_mean` 平均 2.418，p95 3.088，分布正常。
- `density_burst_mean` 平均很低，只有 0.004，median 接近 0；但
  `density_burst_p95` 的 saturation report 显示 p95 接近 0.99995。这说明 burst
  是短时峰值型 feature：section 平均不高，但窗口内 p95 很容易接近上限。
- `hold_occupancy_mean` 平均 0.101，p95 0.328。
- `ln_change_rate_mean` 平均 0.917，p95 2.572，波动较大。
- `chord_ratio_mean` 平均 0.157，p95 0.343；`chord_ratio_p95 >= 0.95` 的比例
  只有 0.001%，没有系统性高饱和。

### Jack

`jack_excess` 是稀疏 tail feature：

- `jack_excess_mean` 平均 0.00264
- median 0
- p95 0.00807
- p99 0.0665
- max 0.7567

但看 section 的 `jack_excess_p95`：

- p95 = 0.0587
- p99 = 0.2784
- max = 0.9035
- tail_heaviness = 4.74

结论：`jack_excess` 在大多数 section 很低，但 tail 很尖。它适合作为 outlier /
stress signal，不适合作为均匀分布的连续控制旋钮。

### Repeat

- `repeat_exact_mean` 平均 0.0717
- `repeat_shift_mean` 平均 0.1154
- `repeat_motion_mean` 平均 0.0834
- `repeat_rhythm_mean` 平均 0.5574

前三个 repeat channels 没有明显 near-high saturation：

- `repeat_exact_p95 >= 0.95`: 0.0037%
- `repeat_shift_p95 >= 0.95`: 0.0044%
- `repeat_motion_p95 >= 0.95`: 0.0037%

但 `repeat_rhythm` 严重不同：

- `repeat_rhythm_mean p95 = 0.9716`
- `repeat_rhythm_p95 >= 0.95`: 22.83%
- audit gate failed。

这说明 `repeat_rhythm` 更像是在捕捉 “regular timing / rhythmic regularity”，而
不是足够稀疏的 pattern recurrence。作为 rhythm regularity 指标它可能有用，但
作为 bounded discriminative control feature，它现在太容易打满。

### Hand Balance

- `hand_balance_signed_mean = -0.000106`，非常接近 0，说明 corpus 上左右手没有
  整体偏置。
- `hand_imbalance_abs_mean = 0.0364`，p95 0.0834，整体不高。
- 但 max 到 0.7538，说明局部极端 imbalance 存在。

## Confidence 结论

Confidence 分布不是完全一致：

| confidence feature | mean | p50 | p95 |
| --- | ---: | ---: | ---: |
| density_confidence | 0.993 | ~1.000 | 1.000 |
| chord_confidence | 0.938 | 0.981 | 0.9998 |
| repeat_confidence | 0.974 | 0.998 | ~1.000 |
| jack_confidence | 0.691 | 0.797 | 0.9998 |
| ln_change_confidence | 0.390 | 0.302 | 0.987 |
| jack_streak_confidence | 0.311 | 0.172 | 0.940 |
| control_confidence | 0.789 | 0.804 | 0.949 |

结论：density / chord / repeat 的 confidence 很高；LN change、jack streak 的
confidence 更依赖局部支持量，低 confidence 情况很多。

这和失败的 `low_confidence_high_value` gate 是一致的：有不少 high-value feature
出现在局部 support 不够稳的窗口里。

## Audit Gates 结论

`evaluate_control_v2_audit(section_df)` 一共输出 17 个 gates，其中 15 个通过，
2 个失败。

通过的关键项：

- `finite_model_stats`: `finite_rate = 1.0`，通过。
- `valid_fraction`: `mean = 0.993986`，通过。
- `chord_ratio`、`repeat_exact`、`repeat_shift`、`repeat_motion` 的 near-high
  saturation 都通过。
- repeat channels 的 near-zero rate 都很低，没有 “结构性全零”。
- 主要 partial correlation gates 都通过，说明多数 feature 没有塌缩成简单 proxy。

失败项 1：`low_confidence_high_value`

- value = 0.564628
- threshold = `<= 0.10`
- failed。

这不是小偏差，是大幅失败。

需要注意：这个指标的实现是把每个 feature 的命中行 concat 起来，然后除以 section
数量。因此它更准确地说是 feature-section hit rate，不是 unique section rate。
一个 section 如果多个 feature 同时 high-value low-confidence，会被重复计数。

但即使考虑这个口径，0.5646 也太高。建议后续同时报告：

- unique affected section rate
- per-feature hit rate
- high value at peak confidence vs window min confidence 的分离版本

失败项 2：`repeat_rhythm_near_high`

- near_high_rate = 0.228300
- threshold = `<= 0.05`
- failed。

这个是最明确的 calibration 问题：`repeat_rhythm` 太容易接近上限。

## Correlation / Redundancy 结论

几个重要相关性：

| pair | Pearson | Spearman | partial |
| --- | ---: | ---: | ---: |
| `jack_excess_mean` vs `density_level_mean`, control `chord_ratio_mean` | -0.016 | -0.005 | -0.008 |
| `jack_excess_mean` vs `density_level_mean`, control `density_raw_med_mean` | -0.016 | -0.005 | 0.025 |
| `repeat_motion_mean` vs `density_level_mean`, control `chord_ratio_mean` | -0.081 | -0.110 | 0.027 |
| `repeat_exact_mean` vs `jack_excess_mean`, control `density_level_mean` | 0.205 | -0.002 | 0.205 |
| `ln_change_rate_mean` vs `hold_occupancy_mean`, control `density_level_mean` | 0.837 | 0.929 | 0.852 |
| `chord_ratio_mean` vs `density_level_mean` | 0.534 | 0.527 | n/a |
| `repeat_shift_mean` vs `repeat_motion_mean`, control `density_level_mean` | 0.737 | 0.708 | 0.735 |
| `hand_imbalance_abs_mean` vs `density_level_mean` | -0.307 | -0.376 | n/a |

Interpretation：

- `jack_excess` 没有塌缩成 density proxy，这是好信号。
- `repeat_motion` 也没有塌缩成 density proxy，这是好信号。
- `chord_ratio` 和 `density_level` 中等正相关，合理：dense sections 更可能有
  chords。
- `hand_imbalance_abs` 和 `density_level` 中等负相关，说明更 dense 的 section
  通常更平衡。
- `ln_change_rate` 和 `hold_occupancy` 高度相关，partial 仍然 0.852，虽然 gate
  阈值是 0.90 所以通过，但已经很接近。它们不是独立信号。
- `repeat_shift` 和 `repeat_motion` 也高度相关，partial 0.735。这两个 repeat
  family 可能有明显 redundancy。

## Outlier / Low-Confidence Examples

`low_confidence_high_value_report(section_df, n=25)` 的前 25 行全部是
`chord_ratio`，value 大约从 0.665 到 0.802。其中不少 section 的：

- `confidence_min = 0`
- `confidence_p20` 很低
- 但 `confidence_at_feature_peak` 又可能很高

这说明窗口内部 confidence 分布不均：某些时间点 high-value 是可信的，但整个
section 内有低 confidence 区域，导致 gate 被打中。

所以这里不能简单说 `chord_ratio` 错了。更准确的结论是：当前 gate 把
window-level low confidence 和 feature peak confidence 混在一起，可能过于严厉，
但它确实暴露了 confidence aggregation 口径问题。

## Clustering 没有结论

`RUN_CLUSTERING_EXAMPLE = False`。

虽然 notebook 里写了 section embedding / KMeans / HDBSCAN 相关代码，但这次执行
没有跑 clustering，所以不能从这个 notebook 得出 cluster profile、cluster
stability、representative sections 等结论。

## 最终结论

`control_v2` 的 feature extractor 方向上是成立的：

- synthetic probes 通过；
- perturbation tests 通过；
- 全量 corpus 没有 error；
- 数值 finite；
- 大多数 bounded features 没有系统性 near-high saturation；
- `jack_excess`、`repeat_motion` 没有退化成 density proxy。

但现在还不应该把这次结果理解成 “production-ready fully validated”。

必须优先处理：

1. 重新校准 `repeat_rhythm`

   当前它在 corpus 上过饱和，22.83% sections 接近上限。要么降低其权重，要么改
   normalization，要么把它重新定义成 rhythm regularity，而不是和
   `repeat_exact` / `repeat_shift` / `repeat_motion` 同级的 bounded recurrence
   signal。

2. 重做 `low_confidence_high_value` audit 口径

   当前失败很大，但指标本身是 feature-section hit rate，不是 unique-section
   rate。需要按 feature 分解，并区分 `confidence_min`、`confidence_p20`、
   `confidence_at_feature_peak`。

3. 人工 review high-value low-confidence sections

   尤其是 `chord_ratio` 和可能的 `repeat_rhythm` outliers。现在的前 25 个
   `chord_ratio` outliers 不一定是错，但它们证明 confidence aggregation 需要更细。

4. 关注 feature redundancy

   `ln_change_rate` vs `hold_occupancy`、`repeat_shift` vs `repeat_motion` 的相关性
   很高。如果后续模型把它们当独立 controls，可能会重复计权。

一句话结论：V2 extractor 的机制测试过关，corpus 数值健康度大体可接受，但
`repeat_rhythm` calibration 和 confidence gating 是明确 blocker。
