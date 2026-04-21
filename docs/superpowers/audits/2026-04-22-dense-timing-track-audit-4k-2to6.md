# Dense Timing Track Audit: 4K 2.0-6.0 Stage 1

Date: 2026-04-22

Spec: `docs/superpowers/specs/2026-04-21-oracle-timing-4k-mapper-2to6-design.md`

Status: **PASS**

This is the rerun after enabling the parser/renderer red timing validity gate and the explicit timing
anomaly gate. The old renderer-only numeric smoke test was a false positive. With invalid source
timing rejected before rendering, the accepted subset has bounded raw BPM/beat-length extrema. The
overall gate now passes because the 70 rejected maps are classified timing anomalies, account for
only `0.6337%` of the eligible corpus, contain no nonfinite timing points, and stay under the
configured `1.0%` anomaly-map cap.

## Result

```text
Render Numerics: PASS
Valid Timing Subset: PASS
Timing Anomaly Gate: PASS
Coverage: PASS
Overall Gate: PASS
```

The renderer still consumes only semantically valid red timing points. Classified invalid authored
timing is filtered before rendering and guarded as a bounded anomaly class, not treated as valid BPM
conditioning.

## Artifact

Machine-readable artifact:

- Path: `train/artifacts/reports/audits/dense_timing_track_4k_2to6_2026-04-22.json`
- SHA-256: `761016e793cccfc85334ba4c6c0cd5e66baf27495ce405f845ad9165b1309b97`
- Schema version: `2`

Debug plots:

- Directory: `train/artifacts/reports/audits/dense_timing_debug_2026-04-22/`
- Count: `8`

## Command

```bash
uv run python -m train.stage1_oracle.audits.dense_timing_artifact --index-path train/artifacts/indexes/beatmap_index_4k.parquet --dataset-root mania-dataset --output-json train/artifacts/reports/audits/dense_timing_track_4k_2to6_2026-04-22.json --debug-plot-dir train/artifacts/reports/audits/dense_timing_debug_2026-04-22 --debug-plot-count 8
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
code_commit f6bcaccbe606b3ef2e95afb0fb775b8c4e1a19a4
code_dirty true
audio_duration_source ffprobe
audio_duration_failure_count 0
```

## Root Cause

The abnormal timing is not produced by normalization, clipping, or a parser fallback. It is literal
map-authored data in `[TimingPoints]`.

Evidence from the eligible 4K 2.0-6.0 scan:

```text
eligible_map_count 11047
valid_map_count 10977
invalid_red_timing_map_count 70
invalid_red_timing_point_count 60396
nonfinite_red_timing_point_count 0
nonpositive_red_timing_point_count 20
too_fast_red_timing_point_count 32446
too_slow_red_timing_point_count 27930
invalid_uninherited_flag_counts {1: 60396}
invalid_timing_line_field_counts {8: 60396}
```

All invalid timing lines have the full osu! timing schema and `uninherited = 1`, so they are red
timing points according to the file flag. The values are invalid for this renderer because they do
not represent physically meaningful BPM.

Representative lines:

```text
mania-dataset/0/1670387/... [SV Memorize Level Hard].osu
56563,1e-100,4,1,0,70,1,0
56564,1e+100,4,1,0,70,1,0
56734,2.0000000000000004,4,1,0,70,1,0

mania-dataset/0/697153/... [H0w2Ch3xL1k3J4k4d5's 4K MXM].osu
1098,1E+308,4,1,0,0,1,0
1106.9,0.01,4,1,0,0,1,0

mania-dataset/0/501530/... [Artificial Mind].osu
29706,1000000,4,2,0,0,1,0
29707,1000000,4,2,0,0,1,0
```

Top invalid beat-length literals:

```text
1000000: 11563
0.06: 6639
10000000: 4885
0.6: 4561
999999999: 3768
60000000: 1878
1E+308: 402
1e-100: 144
```

The affected maps are concentrated in SV/gimmick/ASPIRE-style content. A metadata/path token scan
found `sv` in 41 of the 70 invalid maps, with additional hits for `speed`, `gimmick`, `jack`, `sdvx`,
and `j4k`. These tokens are diagnostic only; the gate criterion is numeric validity, not metadata.

Conclusion: the real cause is that some mania maps encode visual or gimmick timing behavior as
syntactically red timing points with impossible beat lengths. The old audit treated those sections
as valid because they were finite and positive, then log-normalization/clipping hid the source
contract violation.

## Validity Policy

For `timing_track_20ms_v1`, a valid red timing point must satisfy:

```text
offset_ms is finite
beat_length_ms is finite
60 <= beat_length_ms <= 3000
20 <= bpm <= 1000
```

This range is derived from the anomaly analysis: the false-positive run showed normal central
percentiles (`p01 ~= 88`, `p50 ~= 178`, `p99 = 300`) while the invalid tails were orders of magnitude
outside physical BPM use (`1e-100`, `1e+308`, `1E+308`, `1000000`, `0.06`).

## Rerun Metrics

```text
total_map_count 11047
audited_map_count 10977
out_of_range_map_count 0
missing_red_timing_map_count 0
invalid_red_timing_map_count 70
invalid_red_timing_point_count 60396
nonfinite_red_timing_point_count 0
nonpositive_red_timing_point_count 20
implausible_red_timing_point_count 60376
negative_time_hitobject_map_count 0
window_count 223885
frame_count 134331000
timing_track_nan_count 0
timing_track_inf_count 0
phase_unit_norm_error_mean 1.448667724428816e-08
phase_unit_norm_error_max 4.2026173652232046e-08
beat_pulse_nonzero_ratio 0.23867241366475347
local_bpm_log_norm_mean 0.0009880436439266194
local_bpm_log_norm_std 0.986902082052936
local_bpm_log_norm_min -4.0
local_bpm_log_norm_max 4.0
raw_bpm_min 20.0
raw_bpm_p01 88.69999999999996
raw_bpm_p50 179.0
raw_bpm_p99 300.0
raw_bpm_max 896.0000000000002
raw_beat_length_min 66.9642857142857
raw_beat_length_max 3000.0
bpm_norm_clipped_low_count 237829
bpm_norm_clipped_high_count 134948
bpm_norm_clipped_ratio 0.0027750630904258885
bpm_log_mean 5.160359804030509
bpm_log_std 0.23196550602850757
```

The bounded raw extrema confirm the validity gate is now active. The `bpm_log_std` also returns to a
normal scale after removing the extreme red timing sections.

## Per-Bin Summary

| Bin | Maps | Windows | Frames | Beat pulse nonzero | BPM norm mean | BPM norm std |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2-3 | 3,638 | 62,322 | 37,393,200 | 22.612% | -0.2363 | 1.0196 |
| 3-4 | 3,376 | 64,357 | 38,614,200 | 23.302% | -0.0889 | 0.9177 |
| 4-5 | 2,705 | 63,707 | 38,224,200 | 24.358% | 0.1058 | 0.9046 |
| 5-6 | 1,258 | 33,499 | 20,099,400 | 26.355% | 0.4159 | 1.0429 |

Invalid-map distribution by bin, relative to the false-positive run:

```text
2-3: 11 maps filtered
3-4: 29 maps filtered
4-5: 22 maps filtered
5-6: 8 maps filtered
```

## Debug Plots

Generated fixed-sample debug plots contain:

- audio energy summary
- `beat_pulse`
- beat phase sin/cos
- `local_bpm_log_norm`
- ground-truth hitobject start markers

Files:

```text
train/artifacts/reports/audits/dense_timing_debug_2026-04-22/dense_timing_debug_000.png
train/artifacts/reports/audits/dense_timing_debug_2026-04-22/dense_timing_debug_001.png
train/artifacts/reports/audits/dense_timing_debug_2026-04-22/dense_timing_debug_002.png
train/artifacts/reports/audits/dense_timing_debug_2026-04-22/dense_timing_debug_003.png
train/artifacts/reports/audits/dense_timing_debug_2026-04-22/dense_timing_debug_004.png
train/artifacts/reports/audits/dense_timing_debug_2026-04-22/dense_timing_debug_005.png
train/artifacts/reports/audits/dense_timing_debug_2026-04-22/dense_timing_debug_006.png
train/artifacts/reports/audits/dense_timing_debug_2026-04-22/dense_timing_debug_007.png
```

## Gate Decision

```text
status PASS
renderer_numerics_status PASS
valid_timing_subset_status PASS
timing_anomaly_status PASS
coverage_status PASS
timing_anomaly_policy classified_invalid_red_timing_filtered_with_1pct_map_cap
max_timing_anomaly_map_ratio 0.01
timing_anomaly_map_ratio 0.006336561962523762
accounted_map_count 11047
timing_track_version timing_track_20ms_v1
timing_frame_hop_ms 20
timing_frame_center_offset_ms 10
timing_frame_count_per_window 600
timing_channels beat_pulse,beat_phase_sin,beat_phase_cos,local_bpm_log_norm,timing_confidence
pulse_shape triangular
pulse_width_ms 40
missing_red_timing_map_count 0
invalid_red_timing_map_count 70
invalid_red_timing_point_count 60396
nonfinite_red_timing_point_count 0
nonpositive_red_timing_point_count 20
implausible_red_timing_point_count 60376
timing_track_nan_count 0
timing_track_inf_count 0
phase_unit_norm_error_max 4.2026173652232046e-08
local_bpm_log_norm_min -4.0
local_bpm_log_norm_max 4.0
raw_bpm_min 20.0
raw_bpm_p01 88.69999999999996
raw_bpm_p50 179.0
raw_bpm_p99 300.0
raw_bpm_max 896.0000000000002
raw_beat_length_min 66.9642857142857
raw_beat_length_max 3000.0
bpm_norm_clipped_low_count 237829
bpm_norm_clipped_high_count 134948
bpm_norm_clipped_ratio 0.0027750630904258885
bpm_log_mean 5.160359804030509
bpm_log_std 0.23196550602850757
failure_reasons []
```

## Suggestions

- Keep the validity gate at the parser boundary and renderer boundary. Parser validation prevents
  dirty `.osu` timing from silently entering corpus-level stats; renderer validation protects direct
  callers and tests.
- Do not address this with robust normalization first. The source contract is the failing layer; once
  invalid timing is filtered, normalization statistics become coherent.
- Treat the 70 affected maps as excluded timing anomalies for this oracle dense timing track unless a
  separate design decision defines how to model gimmick/SV red timing. They are syntactically red in
  `.osu`, but they are not valid BPM conditioning for `timing_track_20ms_v1`.
- Consider adding structured invalid timing examples to future JSON artifacts so gate failures can be
  diagnosed without rerunning an ad hoc root-cause scan.
