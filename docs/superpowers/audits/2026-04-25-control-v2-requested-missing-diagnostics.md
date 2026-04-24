---
date: 2026-04-25
drafted_on: 2026-04-25
effective_on: 2026-04-25
pinned_commit: 0983f49dc86d493ba6acb8b46e1348048b8e82f9
updates:
  - "[2026-04-25-control-v2-feature-diagnostics.md](2026-04-25-control-v2-feature-diagnostics.md)"
source_links:
  - "[control_v2_requested_missing_diagnostics.ipynb](../../../train/notebooks/control_v2_requested_missing_diagnostics.ipynb)"
  - "[control_v2_requested_missing_diagnostics_manifest.json](../../../train/artifacts/features/control_v2_audit/control_v2_requested_missing_diagnostics_manifest.json)"
  - "[control_v2.py](../../../train/stage1_oracle/features/control_v2.py)"
  - "[control_v2_artifact.py](../../../train/stage1_oracle/features/control_v2_artifact.py)"
artifact_links:
  - "[control_v2_section_audit_8s_stride4.parquet](../../../train/artifacts/features/control_v2_audit/control_v2_section_audit_8s_stride4.parquet)"
  - "[control_v2_low_confidence_high_value_feature_breakdown.parquet](../../../train/artifacts/features/control_v2_audit/control_v2_low_confidence_high_value_feature_breakdown.parquet)"
  - "[control_v2_low_confidence_high_value_feature_sections.parquet](../../../train/artifacts/features/control_v2_audit/control_v2_low_confidence_high_value_feature_sections.parquet)"
  - "[control_v2_low_confidence_high_value_unique_sections.parquet](../../../train/artifacts/features/control_v2_audit/control_v2_low_confidence_high_value_unique_sections.parquet)"
  - "[control_v2_low_confidence_high_value_review_queue.parquet](../../../train/artifacts/features/control_v2_audit/control_v2_low_confidence_high_value_review_queue.parquet)"
  - "[control_v2_repeat_rhythm_near_high_sections.parquet](../../../train/artifacts/features/control_v2_audit/control_v2_repeat_rhythm_near_high_sections.parquet)"
  - "[control_v2_kmeans8_axis_dominance.parquet](../../../train/artifacts/features/control_v2_audit/control_v2_kmeans8_axis_dominance.parquet)"
  - "[control_v2_kmeans8_profile.parquet](../../../train/artifacts/features/control_v2_audit/control_v2_kmeans8_profile.parquet)"
  - "[control_v2_kmeans8_representatives.parquet](../../../train/artifacts/features/control_v2_audit/control_v2_kmeans8_representatives.parquet)"
  - "[control_v2_kmeans8_sections.parquet](../../../train/artifacts/features/control_v2_audit/control_v2_kmeans8_sections.parquet)"
  - "[control_v2_kmeans8_stability.parquet](../../../train/artifacts/features/control_v2_audit/control_v2_kmeans8_stability.parquet)"
analysis_scope: "基于已执行 notebook 的 summary/head tables 和 manifest counts；本文不重新读取完整 parquet artifacts。"
artifact_manifest_code_commit: fa64069d66bfc2037292e2ee2718884d40345e12
artifact_manifest_code_dirty: true
source_note: "原 diagnostic 提到 control_v2(1).py；当前 checkout 中对应实现文件是 train/stage1_oracle/features/control_v2.py。"
---

# Control V2 Requested Missing Diagnostics Update

本文基于 `control_v2_requested_missing_diagnostics.ipynb` 中已经嵌入的执行输出，
修正前一版 `control_v2` feature diagnostics 的结论。最重要的变化是：
`low_confidence_high_value` 失败的真实主因现在清楚了，不是 `chord_ratio`，
也不主要是多 feature 重复计数，而是 `ln_change_rate` 与 section-level
confidence aggregation 的口径冲突。

本文只使用 notebook 已显示的 summary tables、head tables 和 manifest counts。
frontmatter 中链接了相关 parquet artifacts 以便追溯，但本文没有重新读取完整
parquet。

## 更新后的总判断

`control_v2` 不是整体坏掉。新的 diagnostics 反而说明 feature family 有清晰
结构，clustering 也能分出有意义的 section 类型。

但现在有两个明确问题：

1. `repeat_rhythm` 仍然是硬 calibration blocker。
   它不是低 confidence 噪声，而是在高 confidence 下真实打满。也就是说，问题
   来自 feature definition 本身。
2. `low_confidence_high_value` 失败需要重新解释。
   原 gate 显示失败很严重：
   `feature_sections_per_input_section = 0.564628`。新数据说明其中 86.31% 的
   feature-section hits 来自 `ln_change_rate`。更关键的是，大多数是
   `window_only_low_confidence`，不是 high-value peak 本身 low-confidence。

更准确的结论是：`repeat_rhythm` 是 feature 定义和归一化问题；
`low_confidence_high_value` 主要是 `ln_change_rate` 加 section aggregation audit
口径问题，而不是全体 feature unreliable。

## Low-Confidence / High-Value 的真实来源

manifest counts：

| Metric | Value |
| --- | ---: |
| total sections | 407,491 |
| feature-section hits | 230,081 |
| feature_sections_per_input_section | 0.564628 |
| unique affected sections | 206,890 |
| unique affected section rate | 0.507717 |
| hits per affected section | 1.112 |

这个结果说明，之前不能简单说 “只是重复计数造成的假高”。`unique affected
sections` 已经是 50.77%，确实有大量 sections 被这个 audit 命中。

但新的 per-feature breakdown 改变了重点：

| Feature | Hit count | Section rate | Share of all hits | Peak-low count | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| `ln_change_rate` | 198,594 | 48.7358% | 86.31% | 10,422 | 主要来源 |
| `jack_streak_exposure` | 19,253 | 4.7248% | 8.37% | 0 | window aggregation issue |
| `repeat_rhythm` | 4,525 | 1.1105% | 1.97% | 0 | 不是主要 low-conf issue |
| `density_level` | 3,475 | 0.8528% | 1.51% | 136 | minor |
| `density_burst` | 3,309 | 0.8120% | 1.44% | 527 | minor but worth checking |
| `jack_excess` | 366 | 0.0898% | 0.16% | 0 | negligible |
| `chord_ratio` | 219 | 0.0537% | 0.10% | 0 | rare extreme cases |
| repeat / hand others | < 0.05% each | tiny | tiny | 0 | not priority |

结论很明确：`low_confidence_high_value` 不是 `chord_ratio` 的 corpus-level
问题。

前一个 notebook 的 review queue 前 25 行几乎都是 `chord_ratio`，这会误导
优先级判断。`chord_ratio` 的 top cases 确实看起来极端，例如 peak value 大约
0.67 到 0.81，但它总共只有 219 个 feature-section hits，占全体 sections 的
0.0537%。这是 edge-case review，不是主故障源。

## 严重性要降级，但不能忽略

最关键的新信息是 `window_only_low_confidence_count` 与
`peak_low_confidence_count` 的分离：

| Category | Count | Rate vs all sections |
| --- | ---: | ---: |
| window-only low-confidence hits | 218,996 | 53.74% |
| peak-low-confidence hits | 11,085 | 2.72% |

这说明当前 audit 很可能把两件事混在一起了：

- section 里某个时间点 confidence 很低；
- section 里另一个时间点 feature value 很高。

如果 audit 是用 section-level `p95(value)` 搭配 section-level
`min(confidence)` 或 `p20(confidence)`，它会制造大量 “同一个 8s window 内 value
高且 confidence 低” 的命中，但 value peak 和 low confidence 不一定发生在同一
时间点。

这不是说没问题。更准确地说：

- 如果 downstream 使用 section-level summary，并且把一个 section 的低
  confidence 视为整段不可信，那么这个 gate failure 是严重的。
- 如果 downstream 使用 time-series 或 peak-paired confidence，那么真实 hard
  failure 大概接近 2.72% feature-section rate，明显没有之前 56.46% 那么糟。

`ln_change_rate` 本身也解释了为什么它会主导失败。从 `control_v2.py` 看，
`ln_change_rate = log1p(ln_change_raw)`，没有被 `ln_change_confidence` gate
掉；confidence 是单独输出的 channel。相比之下，`chord_ratio`、`jack_excess`、
`jack_streak_exposure`、`hand_balance_signed`、repeat concentration features
里有更多内置 confidence gating。

所以这里不是简单 bug，而是 design/audit mismatch：

- 要么承认 `ln_change_rate` 是 ungated intensity feature，audit 不该用同一套
  gate 判它；
- 要么把 `ln_change_rate` 改成 confidence-gated value；
- 要么同时输出 `ln_change_rate_raw` 和 `ln_change_rate_gated`，让 downstream
  明确选择。

## `repeat_rhythm` 问题被确认

`repeat_rhythm` near-high summary：

| Metric | Value |
| --- | ---: |
| threshold | 0.95 |
| near_high_sections | 93,030 |
| near_high_rate | 22.83% |
| near_high_unique_maps | 7,699 |
| near_high_difficulty_mean | 4.067865 |
| `repeat_rhythm_p95` corpus p50 | 0.725428 |
| `repeat_rhythm_p95` corpus p95 | 0.999973 |
| `repeat_rhythm_p95` corpus p99 | 1.000000 |

这个结果很强：22.83% sections 的 `repeat_rhythm_p95 >= 0.95`，而且覆盖 7,699
maps。如果总 eligible maps 是 10,977，这差不多是 70.1% maps 都至少出现过
near-high `repeat_rhythm` section。

更关键的是，top near-high rows 不是低 confidence：

- `repeat_rhythm_confidence_at_peak = 1.0`
- `repeat_rhythm_peak_value = 1.0`
- `repeat_rhythm_n_eff_at_peak` 大约 73 到 84
- `repeat_rhythm_top1_freq_at_peak = 1.0`
- `repeat_rhythm_pattern_variety_at_peak = 0.0`

这说明它不是 noisy estimate。它是在大量有效 support 下判断 rhythm token 完全
集中。

从实现看，`repeat_rhythm` token 是 `(q,)` 或 `(prev_q, q)`，其中 `q` 是 timing
gap 的 rhythm bucket。对于大量 regular stream / constant timing pattern，这个
token 很容易高度集中。mania 里 regular timing 很常见，所以它自然会饱和。

因此，当前 `repeat_rhythm` 更像 rhythmic regularity，不是和 `repeat_exact`、
`repeat_shift`、`repeat_motion` 同级的 discriminative recurrence feature。

这不是调 threshold 就能彻底解决的问题。建议方向：

- 把当前 `repeat_rhythm` 重命名为 `rhythm_regularity`。
- 新建真正的 `repeat_rhythm_motif`，使用 3-gap 或 4-gap motifs，而不是单 gap
  或两 gap token。
- 对 `top1_freq = 1.0` 且 `pattern_variety = 0.0` 的 trivial constant rhythm 做
  discount。
- 将 `repeat_rhythm` 从 model conditioning target 中降权或移除，只作为
  diagnostic side channel。

## Clustering 现在有结论

之前 clustering 没跑；新 notebook 里 KMeans8 artifact 已经有结果。

| Metric | Value |
| --- | ---: |
| clustered sections | 377,548 |
| total sections | 407,491 |
| clustered coverage | 92.65% |
| clusters | 8 |
| stability seeds | 1, 3, 5, 7, 11 |

clustering 的最强区分轴：

| Rank | Axis | eta² |
| ---: | --- | ---: |
| 1 | `jack_streak_exposure_p90` | 0.677446 |
| 2 | `density_level_p90` | 0.525895 |
| 3 | `density_level_mean` | 0.513084 |
| 4 | `repeat_shift_mean` | 0.456892 |
| 5 | `repeat_motion_mean` | 0.440862 |
| 6 | `repeat_exact_mean` | 0.411140 |
| 7 | `hold_occupancy_mean` | 0.410518 |
| 8 | `ln_change_rate_mean` | 0.402320 |
| 9 | `hold_occupancy_p90` | 0.356896 |
| 10 | `jack_excess_p90` | 0.348857 |

Interpretation：

- `jack_streak_exposure`、density、repeat family、LN family 都确实在 corpus 中
  形成结构。
- `hand_balance_signed_mean` 的 eta² 只有 0.000370，几乎不参与 cluster
  separation。这是合理的，因为 signed hand balance 在 mirror / left-right
  symmetry 下会抵消。真正有用的是 `hand_imbalance_abs`，不是 signed mean。
- `repeat_rhythm_mean` eta² 只有 0.133836。这进一步支持前面的判断：它虽然经常
  很高，但区分力不强。一个到处都高的 feature，不是好 control axis。

## Cluster Profile 解释

cluster label 是 KMeans 的 arbitrary ID，不代表天然顺序。按 section 数量排序
后，大致可以这样解释：

| Cluster | Sections | Total share | Difficulty mean | Signature | Interpretation |
| ---: | ---: | ---: | ---: | --- | --- |
| 0 | 84,538 | 20.75% | 3.74 | mid density, low LN, low jack, balanced | 主流 mid-density pattern |
| 5 | 63,085 | 15.48% | 2.97 | lowest density, moderate hold/LN, lower confidence | easier sparse/LN-ish sections |
| 2 | 57,582 | 14.13% | 3.24 | elevated repeat exact/shift/motion | repeat-pattern sections |
| 4 | 57,421 | 14.09% | 4.04 | highest hold occupancy and `ln_change_rate` | LN-heavy / LN-change-heavy sections |
| 1 | 49,950 | 12.26% | 5.12 | highest density, huge `jack_streak_exposure_p90` | high-density jack-streak pressure; hardest cluster |
| 7 | 29,499 | 7.24% | 4.02 | highest `jack_excess_p90`, higher hand imbalance | actual jack-excess / localized jack pressure |
| 6 | 23,678 | 5.81% | 3.85 | mixed LN plus jack streak | mixed technical sections |
| 3 | 11,795 | 2.89% | 3.52 | strongest repeat exact/shift/motion/rhythm, highest hand imbalance | small repeat/imbalance cluster |

重要 observations：

- Cluster 1 是最难的：difficulty mean 5.119，density mean 3.005，
  `jack_streak_exposure_p90 = 0.745`。这说明 `jack_streak_exposure` 是 strong
  difficulty-correlated axis。
- Cluster 4 证明 LN family 是有用的：`hold_occupancy_mean = 0.255`，
  `ln_change_rate_mean = 2.194`，而且 control confidence 最高，约 0.891。所以
  不要因为 `ln_change_rate` audit 问题就删 LN features。
- Cluster 3 虽小，但 repeat family 很突出：`repeat_exact_mean = 0.165`，
  `repeat_shift_mean = 0.271`，`repeat_motion_mean = 0.198`，
  `repeat_rhythm_mean = 0.697`。这说明 `repeat_exact`、`repeat_shift`、
  `repeat_motion` 有真实结构，不是无用噪声。
- Cluster 7 和 Cluster 1 的区别很有价值：Cluster 1 有高
  `jack_streak_exposure`，但 `jack_excess_p90` 接近 0；Cluster 7 有明显
  `jack_excess_p90 = 0.0976`。这说明 `jack_streak_exposure` 和 `jack_excess`
  不是完全重复的东西。

## 对前一版结论的修正

前一版结论里说：`low_confidence_high_value` 是 blocker，可能需要看 confidence
aggregation；top examples 是 `chord_ratio`。

现在要改成：

- `low_confidence_high_value` 的 corpus-level mass 几乎全是 `ln_change_rate`，
  不是 `chord_ratio`。
- `chord_ratio` 是 rare severe examples，不是 global failure。
- 当前 audit 的 section-level aggregation 过于悲观。如果改成 peak-paired
  confidence，hard failure 上限大约降到 2.72% feature-section hits。
- 但 `ln_change_rate` ungated design 必须被明确处理，否则 audit 和 downstream
  training 会继续语义不一致。

前一版说 `repeat_rhythm` 是 blocker，这一点现在更确定：

- `repeat_rhythm` near-high 不是 confidence 问题，而是高 support、高
  confidence、低 variety 的真实饱和。
- 这个必须改定义或降权。

前一版说 clustering 无结论，现在可以补上：

- KMeans8 clustering 显示 control features 能形成稳定、可解释的 section
  regimes。
- 最有区分力的是 `jack_streak_exposure`、density、repeat
  exact/shift/motion、LN family。
- `repeat_rhythm` 和 `hand_balance_signed_mean` 的 cluster utility 较弱。

## 优先级建议

### P0：重新定义或降权 `repeat_rhythm`

这是最明确的 hard issue。

当前 `repeat_rhythm` 太容易把普通 regular stream 判成 near-max recurrence。作为
conditioning control，它会造成两个风险：

- 训练目标过饱和，模型学不到细分差异；
- 生成时过度鼓励 mechanical regular timing。

建议把当前 channel 改名为 `rhythm_regularity`，或者改 tokenization，让它捕捉
non-trivial rhythm motif recurrence。

### P0：重写 `low_confidence_high_value` audit

不要再用单一 section-level rule 混合 `p95(value)` 和 `min(confidence)` 或
`p20(confidence)`。

至少拆成三类：

1. `pointwise_high_value_low_confidence_rate`
   同一时间点 value 高且 confidence 低。
2. `section_has_high_value_and_low_confidence_elsewhere_rate`
   section 内 value 高，但低 confidence 发生在别处。
3. `unique_affected_section_rate_by_feature`
   防止 feature-section hit count 掩盖 unique section 情况。

这样才能判断是真 feature 问题，还是 aggregation issue。

### P0/P1：明确 `ln_change_rate` 的 design contract

现在 `ln_change_rate` 是 ungated intensity feature，`ln_change_confidence` 是
旁路 channel。这个设计可以成立，但 audit 和 downstream 必须知道。

可选方案：

- 保留 `ln_change_rate_raw`。
- 保留 `ln_change_confidence`。
- 新增 `ln_change_rate_gated = confidence_gate(log1p(raw), ln_change_confidence, 0)`。
- downstream 使用 gated 版做 conditioning，raw 版做 diagnostic。

这比直接删 `ln_change_rate` 好，因为 clustering 显示 LN-change axis 很有信息量。

### P1：review queue 改成 stratified queue

当前 review queue 的 top rows 被 `chord_ratio` extreme cases 占据，但
corpus-level 问题主要是 `ln_change_rate`。这说明 review queue 排序有偏。

建议每个 review batch 固定包含：

- top `ln_change_rate` peak-low cases；
- top `density_burst` peak-low cases；
- rare severe `chord_ratio` cases；
- `repeat_rhythm` near-high cases；
- 每个 KMeans cluster 的 medoid 和 extreme representatives。

否则人工 review 会一直看 rare extremes，错过真实 mass failure。

## 最终更新结论

`control_v2` 的总体机制仍然可用，而且新 clustering 证明它不是一堆随机
feature：density、jack、LN、repeat family 都能分出稳定 section regimes。

但当前还不能称为 fully validated。现在的真实问题排序是：

1. `repeat_rhythm` definition/calibration 必须改。
   22.83% sections near-high，且 top cases 是 `confidence = 1.0` 的真实饱和，
   不是噪声。
2. `ln_change_rate` 是 `low_confidence_high_value` 的主因。
   占 86.31% hits。问题不是 feature 无用，而是 ungated value 加 confidence
   side channel 与 audit 口径不一致。
3. `chord_ratio` 不是 global blocker。
   它有 rare severe examples，但总命中率只有 0.0537%。
4. clustering 支持保留大多数 feature family。
   尤其是 `jack_streak_exposure`、density、repeat exact/shift/motion、LN
   features。但 `repeat_rhythm` 和 `hand_balance_signed_mean` 的 global
   discriminative value 较弱。

一句话：`control_v2` 的 feature system 有结构、有用，但 validation gate 现在
混淆了 section aggregation 与 pointwise confidence；真正需要马上修的是
`repeat_rhythm` 饱和和 `ln_change_rate` confidence contract。
