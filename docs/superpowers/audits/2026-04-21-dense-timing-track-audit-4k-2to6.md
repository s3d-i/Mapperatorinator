# Dense Timing Track Audit: 4K 2.0-6.0 Stage 1

Date: 2026-04-21

Spec: `docs/superpowers/specs/2026-04-21-oracle-timing-4k-mapper-2to6-design.md`

Status: **FAIL**

Follow-up rerun with the validity gate enabled:
`docs/superpowers/audits/2026-04-22-dense-timing-track-audit-4k-2to6.md`.

This audit run is retained as evidence of a renderer-numerics false positive. The rendered dense
track contained no NaN/Inf values, but the source red timing set contained positive timing points
with impossible beat lengths. Overall pre-training gate status is therefore FAIL until the audit is
rerun with the input timing validity gate enabled.

## Artifact

Superseded machine-readable artifact from the false-positive run:

- Path: `train/artifacts/reports/audits/dense_timing_track_4k_2to6_2026-04-21.json`
- SHA-256: `38de0ad17b603ce7c5386a1d67002be1f2f6683e877f53b3c4c38ddef3fd1cfb`
- Schema version: `1`

Debug plots:

- Directory: `train/artifacts/reports/audits/dense_timing_debug_2026-04-21/`
- Count: `8`

## Command

```bash
uv run python -m train.stage1_oracle.audits.dense_timing_artifact --index-path train/artifacts/indexes/beatmap_index_4k.parquet --dataset-root mania-dataset --output-json train/artifacts/reports/audits/dense_timing_track_4k_2to6_2026-04-21.json --debug-plot-dir train/artifacts/reports/audits/dense_timing_debug_2026-04-21 --debug-plot-count 8
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
code_commit ae829df698473dff6f90e7b18535174616b701cb
code_dirty true
audio_duration_source ffprobe
audio_duration_failure_count 0
```

## Scope

Input corpus:

- Index: `train/artifacts/indexes/beatmap_index_4k.parquet`
- Beatmap root: `mania-dataset/`
- Difficulty filter: `2.0 <= difficulty <= 6.0`
- Key count: 4K only

Dense timing renderer:

- Version: `timing_track_20ms_v1`
- Shape per window: `[600, 5]`
- Frame hop: `20ms`
- Frame center offset: `10ms`
- Input duration: `12000ms`
- Beat pulse: triangular, `pulse_width_ms = 40`
- Timing source: valid reference `.osu` red timing points only
- Red timing lookup: last valid red timing point with `offset <= frame_time`, with first/last red timing extrapolation outside the authored range
- Local BPM normalization stats are frame-weighted over audited Stage 1 timing frames and recorded as `bpm_log_mean` / `bpm_log_std`
- Raw BPM and raw beat-length diagnostics are frame-weighted over the rendered dense timing frames
- BPM normalization clip counts are computed before clipping `local_bpm_log_norm` to `[-4, 4]`
- Valid red timing requires finite offsets and `60 <= beat_length_ms <= 3000`
  (`20 <= bpm <= 1000`)

Channels:

```text
0 beat_pulse
1 beat_phase_sin
2 beat_phase_cos
3 local_bpm_log_norm
4 timing_confidence
```

## Superseded Renderer-Numerics Result

```text
total_map_count 11047
audited_map_count 11047
out_of_range_map_count 0
invalid_red_timing_map_count 0
negative_time_hitobject_map_count 0
window_count 225798
frame_count 135478800
timing_track_nan_count 0
timing_track_inf_count 0
phase_unit_norm_error_mean 1.448169360881535e-08
phase_unit_norm_error_max 4.2026173652232046e-08
beat_pulse_nonzero_ratio 0.23871570312107873
local_bpm_log_norm_mean 0.0049946515285887345
local_bpm_log_norm_std 0.12509017432990863
local_bpm_log_norm_min -4.0
local_bpm_log_norm_max 4.0
raw_bpm_min 6e-304
raw_bpm_p01 87.99999999999997
raw_bpm_p50 178.0000000000001
raw_bpm_p99 300.0
raw_bpm_max 6e+104
raw_beat_length_min 1e-100
raw_beat_length_max 1e+308
bpm_norm_clipped_low_count 46545
bpm_norm_clipped_high_count 73
bpm_norm_clipped_ratio 0.00034409811719619604
bpm_log_mean 5.142640674620268
bpm_log_std 2.5119782183782853
```

The raw timing extrema above are the failing evidence:

- `raw_beat_length_min = 1e-100`
- `raw_beat_length_max = 1e+308`
- `raw_bpm_min = 6e-304`
- `raw_bpm_max = 6e+104`

These values are finite after rendering and clipping, which is why the old gate reported PASS, but
they are invalid source timing for `timing_track_20ms_v1`.

## Input Timing Validity Scan

Fast parser-level scan over the same eligible index with the validity policy:

```text
eligible_map_count 11047
valid_map_count 10977
missing_red_timing_map_count 0
invalid_red_timing_map_count 70
invalid_red_timing_point_count 60396
nonfinite_red_timing_point_count 0
nonpositive_red_timing_point_count 20
implausible_red_timing_point_count 60376
```

## Per-Bin Summary

| Bin | Maps | Windows | Frames | Beat pulse nonzero | BPM norm mean | BPM norm std |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2-3 | 3,649 | 62,549 | 37,529,400 | 22.607% | -0.0154 | 0.1037 |
| 3-4 | 3,405 | 65,143 | 39,085,800 | 23.298% | -0.0049 | 0.1412 |
| 4-5 | 2,727 | 64,337 | 38,602,200 | 24.379% | 0.0144 | 0.1274 |
| 5-6 | 1,266 | 33,769 | 20,261,400 | 26.353% | 0.0440 | 0.1126 |

## Debug Plots

Generated fixed-sample debug plots contain:

- audio energy summary
- `beat_pulse`
- beat phase sin/cos
- `local_bpm_log_norm`
- ground-truth hitobject start markers

Files:

```text
train/artifacts/reports/audits/dense_timing_debug_2026-04-21/dense_timing_debug_000.png
train/artifacts/reports/audits/dense_timing_debug_2026-04-21/dense_timing_debug_001.png
train/artifacts/reports/audits/dense_timing_debug_2026-04-21/dense_timing_debug_002.png
train/artifacts/reports/audits/dense_timing_debug_2026-04-21/dense_timing_debug_003.png
train/artifacts/reports/audits/dense_timing_debug_2026-04-21/dense_timing_debug_004.png
train/artifacts/reports/audits/dense_timing_debug_2026-04-21/dense_timing_debug_005.png
train/artifacts/reports/audits/dense_timing_debug_2026-04-21/dense_timing_debug_006.png
train/artifacts/reports/audits/dense_timing_debug_2026-04-21/dense_timing_debug_007.png
```

## Gate Decision

```text
status FAIL
timing_track_version timing_track_20ms_v1
timing_frame_hop_ms 20
timing_frame_center_offset_ms 10
timing_frame_count_per_window 600
timing_channels beat_pulse,beat_phase_sin,beat_phase_cos,local_bpm_log_norm,timing_confidence
pulse_shape triangular
pulse_width_ms 40
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
raw_bpm_min 6e-304
raw_bpm_p01 87.99999999999997
raw_bpm_p50 178.0000000000001
raw_bpm_p99 300.0
raw_bpm_max 6e+104
raw_beat_length_min 1e-100
raw_beat_length_max 1e+308
bpm_norm_clipped_low_count 46545
bpm_norm_clipped_high_count 73
bpm_norm_clipped_ratio 0.00034409811719619604
bpm_log_mean 5.142640674620268
bpm_log_std 2.5119782183782853
```

The renderer numerics smoke test passed, but input timing validity failed. This gate verifies both
conditions: the oracle dense timing representation must render into finite 5-channel tensors, and
the renderer must only consume valid red timing points.
