# Quantization Audit: 4K 2.0-6.0 Stage 1

Date: 2026-04-21

Spec: `docs/superpowers/specs/2026-04-21-oracle-timing-4k-mapper-2to6-design.md`

Quantization gate status: **PASS**

Dataset train-ready status: **not proven by this artifact**. This audit reports post-quantization collisions for downstream Event Space filtering; it does not assert that the raw eligible corpus is directly train-ready.

## Artifact

Full machine-readable artifact:

- Path: `train/artifacts/reports/audits/quantization_4k_2to6_2026-04-21.json`
- SHA-256: `c4f7093d5ac9a30e51ba334905318292495dc74ccca1343d4f6bab1c662c467b`
- Schema version: `2`

## Command

```bash
uv run python -m train.stage1_oracle.audits.quantization_artifact --index-path train/artifacts/indexes/beatmap_index_4k.parquet --dataset-root mania-dataset --output-json train/artifacts/reports/audits/quantization_4k_2to6_2026-04-21.json
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
4. Quantize start and hold-end timestamps with deterministic 10ms half-up rounding: `floor((t_ms + 5) / 10) * 10`.
5. Normalize quantized zero-length holds (`q_end <= q_start`) into `TAP` at `q_start`.
6. Count post-quantization same-lane primitive action collisions after normalization.

## Provenance

```text
index_path train/artifacts/indexes/beatmap_index_4k.parquet
index_sha256 b494941bcac9c12ca1b1cb56bdc0807ad0dabbe2f266a6b3a4edc61825b6a578
dataset_root mania-dataset
eligible_map_count 11047
difficulty_column difficulty
difficulty_source train.stage1_oracle.data.dataset.build_4k_index:calculate_mania_difficulties(speed=1.0)->calculate_mania_difficulty->compute_mania_star_rating_20241007
code_commit fe3975efb84f68860d92a3c8d0455bb2231cddcd
code_dirty true
dirty_patch_sha256 27e85c343ac63e1ed75ddd62e379159d686dc0acf4faac7caacdc195e9b345a8
dirty_patch_file_count 2
```

The artifact was produced from a dirty tree, so reproducibility is recorded as `PATCH_HASH_RECORDED`, not `CLEAN`.

## Result

```text
total_map_count 11047
audited_map_count 11047
missing_red_timing_map_count 0
negative_time_hitobject_map_count 0
out_of_range_map_count 0
key_count 4
quantizer 10ms_half_up
quantizer_grid_ms 10
quantizer_tie_break floor((t_ms + 5) / 10)
timestamp_count 20635804
primitive_lane_action_count 20635792
zero_length_hold_normalized_count 12
zero_length_hold_normalized_map_count 3
mean_quantization_error_ms 2.4758836631710595
p95_quantization_error_ms 5
max_quantization_error_ms 5
post_quantization_collision_timepoint_count 42
post_quantization_collision_lane_time_cell_count 42
post_quantization_collision_lane_action_count 84
post_quantization_collision_affected_map_count 4
post_quantization_collision_rate 4.070597338837298e-06
post_quantization_collision_rate_denominator primitive_lane_action_count
```

## Per-Bin Summary

| Bin | Maps | Timestamps | Mean error | P95 | Max | Zero-length holds | Collision timepoints | Collision lane-time cells | Collision lane actions | Affected maps | Collision rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2-3 | 3,649 | 3,654,284 | 2.4743ms | 5ms | 5ms | 0 | 3 | 3 | 6 | 3 | 0.000164% |
| 3-4 | 3,405 | 5,526,223 | 2.4620ms | 5ms | 5ms | 10 | 0 | 0 | 0 | 0 | 0.000000% |
| 4-5 | 2,727 | 6,957,632 | 2.4892ms | 5ms | 5ms | 0 | 39 | 39 | 78 | 1 | 0.001121% |
| 5-6 | 1,266 | 4,497,665 | 2.4737ms | 5ms | 5ms | 2 | 0 | 0 | 0 | 0 | 0.000000% |

## Gate Decision

```text
status PASS
coverage_status PASS
key_count_status PASS
quantizer_status PASS
quantization_error_status PASS
zero_length_hold_metric_status PASS
collision_metric_status PASS
difficulty_source_status PASS
reproducibility_status PATCH_HASH_RECORDED
post_quantization_collision_threshold_status NOT_APPLICABLE_NO_DESIGN_THRESHOLD
failure_reasons []
```

The design doc does not define a collision-rate failure threshold for this audit, so the gate requires collision metrics with explicit units and leaves nonzero collision handling to the Event Space filtering gate.
