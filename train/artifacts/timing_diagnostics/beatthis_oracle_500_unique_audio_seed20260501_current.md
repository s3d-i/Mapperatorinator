---
pinned_commit: 309d2b2cb3af9936ce8ebeb3390d6f1334e66327
audit_kind: stage_2_timing_beatthis_oracle_unique_audio
sample_seed: 20260501
sample_size: 500
---

# Stage 2 Timing Performance and Accuracy Audit

Audited 500 successful unique audio paths from `train/artifacts/indexes/beatmap_index_4k_no_timing_anomalies_2to6.parquet` using `mania-dataset` on `cpu`. The driver attempted 501 unique audio paths and recorded 1 failures separately.

## Headline

- `total_seconds`: mean 3.47, p95 6.998, max 78.9 seconds. End-to-end wall-clock time for prediction, grid fitting, oracle comparison, and row bookkeeping for one audio. In this run the mean is 3.47025 seconds, p95 is 6.99792, and max is 78.9023; lower values are better for this metric.
- `prediction_seconds`: mean 2.473, p95 4.76, max 53.94 seconds. Wall-clock time spent loading audio and running BeatThis to produce beat/downbeat frame probabilities. In this run the mean is 2.47256 seconds, p95 is 4.76003, and max is 53.939; lower values are better for this metric.
- `fit_seconds`: mean 0.9927, p95 2.219, max 24.92 seconds. Wall-clock time spent by the current GridFitter converting frame probabilities into timing segments. In this run the mean is 0.992749 seconds, p95 is 2.21886, and max is 24.9247; lower values are better for this metric.
- `fit_score`: mean 0.7292, p95 0.8899, max 0.9372 normalized correlation. GridFitter internal normalized score for how well the fitted pulse grid matches the beat probability signal. In this run the mean is 0.72916 normalized correlation, p95 is 0.889859, and max is 0.937182; higher values are better for this metric.
- `mean_phase_error_ms`: mean 48.55, p95 87.27, max 161.9 milliseconds. Mean wrapped phase error converted to milliseconds using the oracle beat length active at each frame. In this run the mean is 48.5534 milliseconds, p95 is 87.2669, and max is 161.896; lower values are better for this metric.
- `first_bpm_alias_error`: mean 3.789, p95 34.01, max 138.5 BPM after alias normalization. Minimum first-segment BPM error after allowing common tempo aliases: quarter, half, exact, double, and quadruple tempo. In this run the mean is 3.78932 BPM after alias normalization, p95 is 34.0125, and max is 138.5; lower values are better for this metric.
- `local_bpm_mae`: mean 35.15, p95 115, max 160.5 BPM. Mean absolute frame-wise BPM difference between the fitted dense timing track and oracle dense timing track. In this run the mean is 35.1481 BPM, p95 is 115, and max is 160.465; lower values are better for this metric.

## Metrics

### `audio_duration_seconds`

- What it is: Audio duration represented by the BeatThis frame sequence, computed from frame_count / frame_rate_hz.
- What this data means: Use this as context for runtime and fitting complexity. Long-tail durations can explain slower prediction or fitting times. In this run the mean is 179.409 seconds, p95 is 330.67, and max is 4358.12; interpret this as context rather than a pass/fail score.
- Summary: min 31.5, p50 146.47, p90 295.94, p95 330.67, p99 400.162, max 4358.12, mean 179.409, std 204.314 seconds.

### `frame_count`

- What it is: Number of timing-probability frames emitted by BeatThis for the audio.
- What this data means: Higher frame counts mean longer songs and more data for GridFitter to search across. In this run the mean is 8970.45 frames at 50 Hz, p95 is 16533.5, and max is 217906; interpret this as context rather than a pass/fail score.
- Summary: min 1575, p50 7323.5, p90 14797, p95 16533.5, p99 20008.1, max 217906, mean 8970.45, std 10215.7 frames at 50 Hz.

### `prediction_seconds`

- What it is: Wall-clock time spent loading audio and running BeatThis to produce beat/downbeat frame probabilities.
- What this data means: This is the model/provider cost before GridFitter starts. It dominates end-to-end latency when much larger than fit_seconds. In this run the mean is 2.47256 seconds, p95 is 4.76003, and max is 53.939; lower values are better for this metric.
- Summary: min 0.638678, p50 1.93141, p90 4.03793, p95 4.76003, p99 6.64771, max 53.939, mean 2.47256, std 2.60172 seconds.

### `fit_seconds`

- What it is: Wall-clock time spent by the current GridFitter converting frame probabilities into timing segments.
- What this data means: This isolates the current timing module performance. p95 and max show whether long or complex tracks create unacceptable fitter latency. In this run the mean is 0.992749 seconds, p95 is 2.21886, and max is 24.9247; lower values are better for this metric.
- Summary: min 0.0564731, p50 0.760152, p90 1.93478, p95 2.21886, p99 2.81153, max 24.9247, mean 0.992749, std 1.25494 seconds.

### `total_seconds`

- What it is: End-to-end wall-clock time for prediction, grid fitting, oracle comparison, and row bookkeeping for one audio.
- What this data means: This approximates user-visible per-audio timing extraction cost on the audited device. In this run the mean is 3.47025 seconds, p95 is 6.99792, and max is 78.9023; lower values are better for this metric.
- Summary: min 0.728158, p50 2.72397, p90 5.86584, p95 6.99792, p99 9.08286, max 78.9023, mean 3.47025, std 3.82551 seconds.

### `fit_score`

- What it is: GridFitter internal normalized score for how well the fitted pulse grid matches the beat probability signal.
- What this data means: Higher values indicate stronger agreement with the model probabilities, but this is not an oracle accuracy metric by itself. In this run the mean is 0.72916 normalized correlation, p95 is 0.889859, and max is 0.937182; higher values are better for this metric.
- Summary: min 0.161367, p50 0.759219, p90 0.868055, p95 0.889859, p99 0.914799, max 0.937182, mean 0.72916, std 0.132267 normalized correlation.

### `candidate_count`

- What it is: Number of BPM/offset grid candidates evaluated across all fitted segments.
- What this data means: This is a workload proxy. High values usually explain slower fit_seconds and indicate harder or more segmented searches. In this run the mean is 3105.41 grid candidates, p95 is 10988.4, and max is 19963; lower values are better for this metric.
- Summary: min 993, p50 1000, p90 7988, p95 10988.4, p99 15978, max 19963, mean 3105.41, std 3451.45 grid candidates.

### `beat_pulse_mae`

- What it is: Mean absolute error between the dense beat-pulse channel rendered from the fitted grid and from the osu red-timing oracle.
- What this data means: Lower values mean fitted beats land closer to oracle beat pulses over the full track. Values near zero mean near-identical pulse placement. In this run the mean is 0.137116 mean absolute pulse-channel error, p95 is 0.204921, and max is 0.318179; lower values are better for this metric.
- Summary: min 0.00563735, p50 0.139012, p90 0.190272, p95 0.204921, p99 0.227539, max 0.318179, mean 0.137116, std 0.0444941 mean absolute pulse-channel error.

### `local_bpm_mae`

- What it is: Mean absolute frame-wise BPM difference between the fitted dense timing track and oracle dense timing track.
- What this data means: This captures tempo accuracy across the song. It is sensitive to half/double tempo aliases and to missing inherited timing changes. In this run the mean is 35.1481 BPM, p95 is 115, and max is 160.465; lower values are better for this metric.
- Summary: min 0, p50 5.79611, p90 100, p95 115, p99 143.103, max 160.465, mean 35.1481, std 43.6431 BPM.

### `mean_phase_error_beats`

- What it is: Mean wrapped phase distance between fitted and oracle beat phase, expressed as fractions of a beat.
- What this data means: This measures average beat alignment independent of absolute offset wrapping. 0.10 means the fitted grid is off by about one tenth of a beat on average. In this run the mean is 0.144501 beats, p95 is 0.250138, and max is 0.34446; lower values are better for this metric.
- Summary: min 0.00283687, p50 0.131756, p90 0.249992, p95 0.250138, p99 0.268622, max 0.34446, mean 0.144501, std 0.076845 beats.

### `max_phase_error_beats`

- What it is: Worst wrapped phase distance between fitted and oracle beat phase across frames, expressed as fractions of a beat.
- What this data means: This highlights severe local timing disagreement even when the mean looks acceptable. In this run the mean is 0.330264 beats, p95 is 0.5, and max is 0.5; lower values are better for this metric.
- Summary: min 0.00291668, p50 0.496, p90 0.499967, p95 0.5, p99 0.5, max 0.5, mean 0.330264, std 0.202121 beats.

### `mean_phase_error_ms`

- What it is: Mean wrapped phase error converted to milliseconds using the oracle beat length active at each frame.
- What this data means: This is the most directly interpretable average alignment metric. Lower values mean beats are closer in real time. In this run the mean is 48.5534 milliseconds, p95 is 87.2669, and max is 161.896; lower values are better for this metric.
- Summary: min 1.17388, p50 45.6704, p90 78.9635, p95 87.2669, p99 107.435, max 161.896, mean 48.5534, std 23.3393 milliseconds.

### `max_phase_error_ms`

- What it is: Worst wrapped phase error in milliseconds across frames.
- What this data means: This shows the worst local beat-alignment miss. Large values usually indicate tempo aliasing, segment mismatch, or an oracle timing change the fitter did not model. In this run the mean is 129.717 milliseconds, p95 is 286.152, and max is 1474; lower values are better for this metric.
- Summary: min 1.2069, p50 136.295, p90 219.059, p95 286.152, p99 675.1, max 1474, mean 129.717, std 127.289 milliseconds.

### `first_bpm_abs_error`

- What it is: Absolute BPM difference between the first fitted segment and the first oracle red timing segment.
- What this data means: This is a strict first-segment tempo check. It intentionally penalizes half/double tempo aliases. In this run the mean is 38.6444 BPM, p95 is 133.5, and max is 213; lower values are better for this metric.
- Summary: min 0, p50 0.05, p90 110.1, p95 133.5, p99 177.565, max 213, mean 38.6444, std 52.6754 BPM.

### `first_bpm_alias_error`

- What it is: Minimum first-segment BPM error after allowing common tempo aliases: quarter, half, exact, double, and quadruple tempo.
- What this data means: This separates real tempo misses from musically equivalent alias choices. Low alias error with high absolute error usually means the grid is at half/double tempo. In this run the mean is 3.78932 BPM after alias normalization, p95 is 34.0125, and max is 138.5; lower values are better for this metric.
- Summary: min 0, p50 1.98952e-13, p90 1.541, p95 34.0125, p99 76.02, max 138.5, mean 3.78932, std 14.9585 BPM after alias normalization.

### `first_offset_phase_error_ms`

- What it is: Wrapped phase distance from the first fitted offset to the first oracle offset using the oracle first beat length.
- What this data means: This measures whether the first beat phase is aligned, regardless of whole-beat offset shifts. In this run the mean is 41.1081 milliseconds, p95 is 133.947, and max is 462.033; lower values are better for this metric.
- Summary: min 0, p50 31.4664, p90 67.1561, p95 133.947, p99 230.032, max 462.033, mean 41.1081, std 44.4534 milliseconds.

### `predicted_segment_count`

- What it is: Number of timing segments emitted by the current fitter.
- What this data means: Compare with oracle_segment_count to see whether the fitter is simplifying or over-splitting timing structure. In this run the mean is 2.99 segments, p95 is 10.05, and max is 20; interpret this as context rather than a pass/fail score.
- Summary: min 1, p50 1, p90 8, p95 10.05, p99 16, max 20, mean 2.99, std 3.26709 segments.

### `oracle_segment_count`

- What it is: Number of red timing segments parsed from the selected osu beatmap.
- What this data means: This is the reference timing complexity for the selected chart. Very large counts can include noisy or unusually detailed timing maps. In this run the mean is 8.412 segments, p95 is 15.25, and max is 687; interpret this as context rather than a pass/fail score.
- Summary: min 1, p50 1, p90 8, p95 15.25, p99 130.15, max 687, mean 8.412, std 49.9759 segments.

### `segment_count_delta`

- What it is: predicted_segment_count minus oracle_segment_count.
- What this data means: Zero means matching segment count. Positive values indicate over-splitting; negative values indicate the fitter used fewer timing changes than the oracle. In this run the mean is -5.422 segments, p95 is 7.05, and max is 15; interpret this as context rather than a pass/fail score.
- Summary: min -671, p50 0, p90 5, p95 7.05, p99 12.01, max 15, mean -5.422, std 49.0384 segments.

## Counts

- Successful rows: 500
- Failed rows: 1
- Rows with at least one anomaly flag: 395
- Flag counts: `{"beat_pulse_mae_gt_0_20": 31, "bpm_mae_gt_5": 251, "first_bpm_alias_error_gt_5": 41, "first_offset_phase_error_gt_120ms": 30, "fit_over_1s": 179, "mean_phase_gt_50ms": 218, "multi_oracle_fitted_one_segment": 58, "segment_count_mismatch": 285}`

## Artifacts

- JSON: `train/artifacts/timing_diagnostics/beatthis_oracle_500_unique_audio_seed20260501_current.json`
- Markdown: `train/artifacts/timing_diagnostics/beatthis_oracle_500_unique_audio_seed20260501_current.md`
