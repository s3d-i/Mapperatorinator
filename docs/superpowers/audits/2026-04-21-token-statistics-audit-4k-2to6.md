# Token Statistics Audit: 4K 2.0-6.0 Stage 1

Date: 2026-04-21

Spec: `docs/superpowers/specs/2026-04-21-oracle-timing-4k-mapper-2to6-design.md`

Status: **PASS for Token Statistics Audit scope only**

This audit does **not** satisfy the separate Quantization Audit or Window Boundary Audit gates.

## Artifact

Full machine-readable artifact:

- Path: `train/artifacts/reports/audits/token_statistics_4k_2to6_2026-04-21.json`
- SHA-256: `179d9f5b905da97e82f36a20b43f2f60f8e885070da6915465c1854eedd016fc`
- Schema version: `1`

The JSON artifact contains the full `ts_counts` and `ts_distribution` dictionaries for every difficulty bin. The markdown below summarizes the gate-relevant parts only.

## Command

```bash
uv run python -m train.stage1_oracle.audits.token_statistics_artifact --index-path train/artifacts/indexes/beatmap_index_4k.parquet --dataset-root mania-dataset --output-json train/artifacts/reports/audits/token_statistics_4k_2to6_2026-04-21.json --max-decode-len 512 --empty-window-cap-ratio 0.05
```

## Provenance

```text
index_path train/artifacts/indexes/beatmap_index_4k.parquet
index_sha256 b494941bcac9c12ca1b1cb56bdc0807ad0dabbe2f266a6b3a4edc61825b6a578
dataset_root mania-dataset
eligible_map_count 11047
unique_audio_count 4766
difficulty_column difficulty
difficulty_source train.stage1_oracle.data.dataset.build_4k_index:calculate_mania_difficulties(speed=1.0)->calculate_mania_difficulty->compute_mania_star_rating_20241007
difficulty_code_commit f99233fc86ef8fcd836c163baac805a98d22b638
code_commit f99233fc86ef8fcd836c163baac805a98d22b638
code_dirty true
audio_duration_source ffprobe
audio_duration_failure_count 0
```

The audit records the difficulty source and index hash. It does not recompute difficulty from `.osu` during this pass; it audits the indexed `difficulty` column and records the code path that produced that index.

## Scope

Input corpus:

- Index: `train/artifacts/indexes/beatmap_index_4k.parquet`
- Beatmap root: `mania-dataset/`
- Difficulty filter: `2.0 <= difficulty <= 6.0`
- Key count: 4K only

Canonical preprocessing:

1. Parse osu!mania hitobjects.
2. Require at least one valid red timing point.
3. Reject negative-time hitobjects.
4. Quantize hitobject times with deterministic 10ms half-up rounding.
5. Normalize zero-length holds after quantization into taps.
6. Count unsupported same-lane compounds before filtering maps.
7. Merge same-timestamp lane actions into canonical timepoint events.
8. Filter maps containing frozen 4-state unsupported `END_TAP` or `END_START` actions.
9. Compute `generation_end = max(ceil_10ms(audio_duration_ms), max_quantized_event_time + 10)`.
10. Assign canonical events to 8s half-open write windows.

Hard assertions added for this gate:

- each canonical timepoint has exactly four lane actions
- no canonical timepoint encodes an all-empty EV
- every lane action is a known `LaneAction`
- post-filter audited maps contain only `NONE`, `TAP`, `HOLD_START`, `HOLD_END`
- post-filter audited maps have strict hold state transitions
- if both `generation_end_ms` and `audio_duration_ms` are supplied, `generation_end_ms` must equal the spec formula

Token lengths count decoder target tokens only: canonical `TS+EV` groups plus `EOS`. They exclude `BOS` and the forced `DIFF_x OPEN_MASK_xxxx` condition prefix.

## Corpus Result

```text
total_map_count 11047
audited_map_count 11043
missing_red_timing_map_count 0
negative_time_hitobject_map_count 0
out_of_range_map_count 0
unsupported_compound_map_count 3
unsupported_compound_event_count 41
unsupported_compound_lane_action_count 82
four_state_unsupported_map_count 1
four_state_unsupported_event_count 1
four_state_unsupported_lane_action_count 1
filtered_event_count 42
filtered_lane_action_count 83
zero_length_hold_normalized_count 12
```

## Per-Bin Summary

| Bin | Maps | Windows | Tokens mean | p95 | p99 | Max | Event TP/s | Notes/s | LN ratio | Empty windows | Hold-cross windows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2-3 | 3,646 | 62,483 | 77.01 | 121 | 141 | 193 | 4.89 | 6.46 | 16.66% | 4.16% | 44.62% |
| 3-4 | 3,405 | 65,143 | 103.69 | 167 | 197 | 289 | 6.59 | 9.26 | 17.74% | 4.17% | 46.42% |
| 4-5 | 2,726 | 64,318 | 130.99 | 219 | 263 | 361 | 8.29 | 11.97 | 15.51% | 4.16% | 40.84% |
| 5-6 | 1,266 | 33,769 | 163.54 | 281 | 331 | 455 | 10.34 | 14.93 | 13.68% | 4.79% | 33.69% |

## Chord-Size Distribution

| Bin | 1-note | 2-note | 3-note | 4-note |
| --- | ---: | ---: | ---: | ---: |
| 2-3 | 66.12% | 32.01% | 1.85% | 0.02% |
| 3-4 | 59.65% | 33.25% | 6.91% | 0.20% |
| 4-5 | 58.01% | 32.22% | 9.21% | 0.56% |
| 5-6 | 58.99% | 30.64% | 9.20% | 1.16% |

## Time-Shift Distribution

The full TS distribution is in the JSON artifact at:

```text
report.bins.<bin>.ts_counts
report.bins.<bin>.ts_distribution
```

Summary:

| Bin | Top-10 TS coverage | Other TS share | TS_0 share | TS_1000 share | Multi-TS windows | Max event delta | Max TS tokens per delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2-3 | 60.01% | 39.99% | 0.133% | 0.587% | 11.20% | 7,980ms | 8 |
| 3-4 | 72.14% | 27.86% | 0.126% | 0.374% | 9.40% | 7,980ms | 8 |
| 4-5 | 76.55% | 23.45% | 0.126% | 0.293% | 8.89% | 7,990ms | 8 |
| 5-6 | 82.58% | 17.42% | 0.126% | 0.233% | 8.64% | 7,950ms | 8 |

Top TS tokens by bin:

```text
2-3: TS_170, TS_160, TS_150, TS_180, TS_90, TS_100, TS_110, TS_190, TS_120, TS_80
3-4: TS_80, TS_90, TS_170, TS_160, TS_100, TS_150, TS_110, TS_70, TS_120, TS_180
4-5: TS_80, TS_90, TS_70, TS_60, TS_100, TS_150, TS_160, TS_50, TS_110, TS_170
5-6: TS_70, TS_60, TS_80, TS_50, TS_90, TS_40, TS_100, TS_110, TS_120, TS_130
```

`TS_1000` is sufficient for Stage 1's 8s write windows under canonical greedy decomposition. This does not mean multi-TS support is optional: 8.64-11.20% of windows require repeated TS tokens, and the largest observed delta uses up to 8 TS tokens.

## Gate Decision

```text
status PASS
configured_max_decode_len 512
max_decode_len_applies_to target_tokens_excluding_bos_and_condition_prefix
observed_max_target_tokens 455
max_decode_len_headroom_tokens 57
empty_window_cap_policy per_difficulty_bin_per_epoch
empty_window_cap_ratio 0.05
empty_window_cap_by_bin 2-3=0.05, 3-4=0.05, 4-5=0.05, 5-6=0.05
ts_1000_sufficient true
covers_quantization_audit false
covers_window_boundary_audit false
```

Training config implication:

- `max_decode_len = 512` applies to generated target tokens only: `TS`, `EV`, and `EOS`.
- `max_decode_len = 512` excludes `BOS`, `DIFF_x`, and `OPEN_MASK_xxxx`.
- Dataset/collate code must reject or fail on over-length targets; it must not silently truncate.
- Empty-window cap is exactly `5.0%` per difficulty bin per epoch.

This gate passes for token statistics. It remains separate from:

- Quantization Audit: quantization error and post-quantization collision rate
- Window Boundary Audit: boundary event density, stitch duplicate/collision risk, and boundary state propagation
