# Window Boundary Audit: 4K 2.0-6.0 Stage 1

Date: 2026-04-21

Spec: `docs/superpowers/specs/2026-04-21-oracle-timing-4k-mapper-2to6-design.md`

Status: **PASS**

## Artifact

Full machine-readable artifact:

- Path: `train/artifacts/reports/audits/window_boundary_4k_2to6_2026-04-21.json`
- SHA-256: `b0f0a29dc00107e8bf162bbab7ce21f072dc31337a777fa883c873b61501f078`
- Schema version: `1`

## Command

```bash
uv run python -m train.stage1_oracle.audits.window_boundary_artifact --index-path train/artifacts/indexes/beatmap_index_4k.parquet --dataset-root mania-dataset --output-json train/artifacts/reports/audits/window_boundary_4k_2to6_2026-04-21.json
```

## Scope

Input corpus:

- Index: `train/artifacts/indexes/beatmap_index_4k.parquet`
- Beatmap root: `mania-dataset/`
- Difficulty filter: `2.0 <= difficulty <= 6.0`
- Key count: 4K only

Canonical preprocessing used by the audit:

1. Parse osu!mania hitobjects.
2. Require at least one valid red timing point.
3. Reject negative-time hitobjects.
4. Quantize hitobject times with deterministic 10ms half-up rounding.
5. Normalize zero-length holds after quantization into taps.
6. Filter unsupported same-lane compounds and frozen 4-state unsupported `END_TAP` / `END_START` maps.
7. Compute `generation_end = max(ceil_10ms(audio_duration_ms), max_quantized_event_time + 10)`.
8. Assign events to 8s half-open write windows and stitch with `t_abs = write_start + t_rel`.

## Provenance

```text
index_path train/artifacts/indexes/beatmap_index_4k.parquet
index_sha256 b494941bcac9c12ca1b1cb56bdc0807ad0dabbe2f266a6b3a4edc61825b6a578
dataset_root mania-dataset
eligible_map_count 11047
unique_audio_count 4766
difficulty_column difficulty
difficulty_source train.stage1_oracle.data.dataset.build_4k_index:calculate_mania_difficulties(speed=1.0)->calculate_mania_difficulty->compute_mania_star_rating_20241007
code_commit c42e672a96113a9900cb8283a765c843a1e525db
code_dirty true
audio_duration_source ffprobe
audio_duration_failure_count 0
```

## Corpus Result

```text
total_map_count 11047
audited_map_count 11043
missing_red_timing_map_count 0
negative_time_hitobject_map_count 0
out_of_range_map_count 0
unsupported_compound_map_count 3
four_state_unsupported_map_count 1
boundary_count 214670
boundary_event_count 16022
boundary_lane_action_count 27034
boundary_event_density 0.07463548702659896
hold_crossing_boundary_count 63396
hold_crossing_boundary_rate 0.29531839567708573
hold_crossing_lane_count 83990
hold_crossing_lane_rate 0.0978129221595938
stitch_duplicate_timepoint_count 0
stitch_collision_timepoint_count 0
stitch_roundtrip_mismatch_count 0
```

## Per-Bin Summary

| Bin | Boundaries | Boundary events | Event density | Boundary lane actions | Hold-cross boundaries | Hold-cross rate | Hold-cross lane rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2-3 | 58,837 | 3,127 | 5.315% | 5,007 | 18,031 | 30.646% | 8.759% |
| 3-4 | 61,738 | 4,207 | 6.814% | 7,204 | 20,113 | 32.578% | 10.541% |
| 4-5 | 61,592 | 5,228 | 8.488% | 8,992 | 17,516 | 28.439% | 10.180% |
| 5-6 | 32,503 | 3,460 | 10.645% | 5,831 | 7,736 | 23.801% | 9.435% |

Stitch dry-run result by bin:

| Bin | Duplicate timepoints | Collision timepoints | Round-trip mismatches |
| --- | ---: | ---: | ---: |
| 2-3 | 0 | 0 | 0 |
| 3-4 | 0 | 0 | 0 |
| 4-5 | 0 | 0 | 0 |
| 5-6 | 0 | 0 | 0 |

## Gate Decision

```text
status PASS
window_ownership half_open_write_intervals
write_window_ms 8000
stitch_duplicate_timepoint_count 0
stitch_collision_timepoint_count 0
stitch_roundtrip_mismatch_count 0
boundary_event_density 0.07463548702659896
hold_crossing_boundary_rate 0.29531839567708573
```

Boundary events are owned by the next window because write regions are half-open. Open-hold state at a write boundary is computed from canonical events with `q_t < write_start`, so events exactly on the boundary are not folded into the boundary mask.
