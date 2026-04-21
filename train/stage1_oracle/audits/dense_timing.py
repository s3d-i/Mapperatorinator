from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..events.canonical import NegativeHitObjectTimeError, ceil_10ms, quantize_10ms_half_up
from ..features.audio import load_audio_file
from ..features.timing import DEFAULT_TIMING_TRACK_CONFIG
from ..features.timing import TIMING_TRACK_CHANNELS
from ..features.timing import TIMING_TRACK_VERSION
from ..features.timing import TimingTrackConfig
from ..features.timing import render_local_bpm_log_20ms_v1
from ..features.timing import render_raw_beat_lengths_20ms_v1
from ..features.timing import render_timing_track_20ms_v1
from ..osu.hitobjects import ManiaHitObject, ManiaHitObjectKind, parse_mania_hit_objects
from ..osu.timing import InvalidRedTimingError
from ..osu.timing import MAX_VALID_RED_BEAT_LENGTH_MS
from ..osu.timing import MAX_VALID_RED_BPM
from ..osu.timing import MIN_VALID_RED_BEAT_LENGTH_MS
from ..osu.timing import MIN_VALID_RED_BPM
from ..osu.timing import MissingRedTimingError
from ..osu.timing import RedTimingPoint
from ..osu.timing import require_red_timing_points
from .token_statistics import DIFFICULTY_BIN_LABELS, WRITE_WINDOW_MS, difficulty_bin_label


# The 2026-04-22 audit found a small SV/gimmick subset with syntactically red
# timing points whose beat lengths are physically impossible. Those maps are
# excluded from oracle timing statistics, but this cap keeps the exclusion an
# explicit anomaly gate instead of letting broad timing corruption pass silently.
MAX_TIMING_ANOMALY_MAP_RATIO = 0.01
TIMING_ANOMALY_POLICY = "classified_invalid_red_timing_filtered_with_1pct_map_cap"


@dataclass(frozen=True)
class DenseTimingMapInput:
    beatmap_path: str | Path
    difficulty: float
    audio_duration_ms: float
    audio_path: str | Path | None = None


@dataclass(frozen=True)
class DenseTimingBinReport:
    label: str
    map_count: int
    window_count: int
    frame_count: int
    beat_pulse_nonzero_ratio: float
    local_bpm_log_norm_mean: float
    local_bpm_log_norm_std: float


@dataclass(frozen=True)
class DenseTimingAuditReport:
    total_map_count: int
    audited_map_count: int
    out_of_range_map_count: int
    missing_red_timing_map_count: int
    invalid_red_timing_map_count: int
    invalid_red_timing_point_count: int
    nonfinite_red_timing_point_count: int
    nonpositive_red_timing_point_count: int
    implausible_red_timing_point_count: int
    negative_time_hitobject_map_count: int
    audio_duration_failure_count: int
    window_count: int
    frame_count: int
    timing_track_nan_count: int
    timing_track_inf_count: int
    phase_unit_norm_error_mean: float
    phase_unit_norm_error_max: float
    beat_pulse_nonzero_ratio: float
    local_bpm_log_norm_mean: float
    local_bpm_log_norm_std: float
    local_bpm_log_norm_min: float
    local_bpm_log_norm_max: float
    raw_bpm_min: float
    raw_bpm_p01: float
    raw_bpm_p50: float
    raw_bpm_p99: float
    raw_bpm_max: float
    raw_beat_length_min: float
    raw_beat_length_max: float
    bpm_norm_clipped_low_count: int
    bpm_norm_clipped_high_count: int
    bpm_norm_clipped_ratio: float
    bpm_log_mean: float
    bpm_log_std: float
    bins: dict[str, DenseTimingBinReport]
    debug_plot_paths: list[str]


@dataclass(frozen=True)
class DenseTimingGateDecision:
    status: str
    renderer_numerics_status: str
    valid_timing_subset_status: str
    timing_anomaly_status: str
    coverage_status: str
    timing_anomaly_policy: str
    max_timing_anomaly_map_ratio: float
    timing_anomaly_map_ratio: float
    accounted_map_count: int
    timing_track_version: str
    timing_frame_hop_ms: int
    timing_frame_center_offset_ms: int
    timing_frame_count_per_window: int
    timing_channels: tuple[str, ...]
    pulse_shape: str
    pulse_width_ms: int
    missing_red_timing_map_count: int
    invalid_red_timing_map_count: int
    invalid_red_timing_point_count: int
    nonfinite_red_timing_point_count: int
    nonpositive_red_timing_point_count: int
    implausible_red_timing_point_count: int
    timing_track_nan_count: int
    timing_track_inf_count: int
    phase_unit_norm_error_max: float
    local_bpm_log_norm_min: float
    local_bpm_log_norm_max: float
    raw_bpm_min: float
    raw_bpm_p01: float
    raw_bpm_p50: float
    raw_bpm_p99: float
    raw_bpm_max: float
    raw_beat_length_min: float
    raw_beat_length_max: float
    bpm_norm_clipped_low_count: int
    bpm_norm_clipped_high_count: int
    bpm_norm_clipped_ratio: float
    bpm_log_mean: float
    bpm_log_std: float
    failure_reasons: list[str]


@dataclass(frozen=True)
class _PreparedDenseTimingMap:
    beatmap_path: Path
    difficulty: float
    generation_end_ms: int
    red_timing_points: Sequence[RedTimingPoint]
    hitobjects: Sequence[ManiaHitObject]
    audio_path: Path | None


@dataclass
class _RunningStats:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    min: float = math.inf
    max: float = -math.inf

    def update_array(self, values: np.ndarray) -> None:
        flattened = np.asarray(values, dtype=np.float64).reshape(-1)
        if flattened.size == 0:
            return
        batch_count = int(flattened.size)
        batch_mean = float(np.mean(flattened))
        batch_m2 = float(np.sum((flattened - batch_mean) ** 2))
        self.update_batch(
            count=batch_count,
            mean=batch_mean,
            m2=batch_m2,
            minimum=float(np.min(flattened)),
            maximum=float(np.max(flattened)),
        )

    def update_batch(self, *, count: int, mean: float, m2: float, minimum: float, maximum: float) -> None:
        if count == 0:
            return
        if self.count == 0:
            self.count = count
            self.mean = mean
            self.m2 = m2
            self.min = minimum
            self.max = maximum
            return

        delta = mean - self.mean
        total_count = self.count + count
        self.mean = self.mean + delta * count / total_count
        self.m2 = self.m2 + m2 + delta * delta * self.count * count / total_count
        self.count = total_count
        self.min = min(self.min, minimum)
        self.max = max(self.max, maximum)

    @property
    def std(self) -> float:
        if self.count == 0:
            return 0.0
        return math.sqrt(max(0.0, self.m2 / self.count))


@dataclass
class _WeightedValueDistribution:
    counts: dict[float, int] | None = None
    total_count: int = 0
    min: float = math.inf
    max: float = -math.inf

    def __post_init__(self) -> None:
        self.counts = {} if self.counts is None else self.counts

    def update_array(self, values: np.ndarray) -> None:
        flattened = np.asarray(values, dtype=np.float64).reshape(-1)
        if flattened.size == 0:
            return
        unique_values, unique_counts = np.unique(flattened, return_counts=True)
        assert self.counts is not None
        for value, count in zip(unique_values, unique_counts):
            value_float = float(value)
            count_int = int(count)
            self.counts[value_float] = self.counts.get(value_float, 0) + count_int
            self.total_count += count_int
        self.min = min(self.min, float(unique_values[0]))
        self.max = max(self.max, float(unique_values[-1]))

    def percentile(self, percentile: float) -> float:
        if self.total_count == 0:
            return 0.0
        if percentile < 0 or percentile > 100:
            raise ValueError(f"percentile must be in [0, 100]: {percentile}")

        # Nearest-rank percentile over the frame-weighted rendered dense timing values.
        rank = max(1, math.ceil((percentile / 100.0) * self.total_count))
        cumulative = 0
        assert self.counts is not None
        for value in sorted(self.counts):
            cumulative += self.counts[value]
            if cumulative >= rank:
                return value
        return self.max


@dataclass
class _DenseTimingBinAccumulator:
    label: str
    map_count: int = 0
    window_count: int = 0
    frame_count: int = 0
    beat_pulse_nonzero_count: int = 0
    bpm_norm_stats: _RunningStats | None = None

    def __post_init__(self) -> None:
        self.bpm_norm_stats = _RunningStats() if self.bpm_norm_stats is None else self.bpm_norm_stats

    def to_report(self) -> DenseTimingBinReport:
        assert self.bpm_norm_stats is not None
        return DenseTimingBinReport(
            label=self.label,
            map_count=self.map_count,
            window_count=self.window_count,
            frame_count=self.frame_count,
            beat_pulse_nonzero_ratio=_rate(self.beat_pulse_nonzero_count, self.frame_count),
            local_bpm_log_norm_mean=self.bpm_norm_stats.mean if self.bpm_norm_stats.count else 0.0,
            local_bpm_log_norm_std=self.bpm_norm_stats.std,
        )


def audit_dense_timing_tracks(
    map_inputs: Iterable[DenseTimingMapInput],
    *,
    config: TimingTrackConfig = DEFAULT_TIMING_TRACK_CONFIG,
    debug_plot_dir: str | Path | None = None,
    debug_plot_count: int = 0,
) -> DenseTimingAuditReport:
    prepared_maps: list[_PreparedDenseTimingMap] = []
    total_map_count = 0
    out_of_range_map_count = 0
    missing_red_timing_map_count = 0
    invalid_red_timing_map_count = 0
    invalid_red_timing_point_count = 0
    nonfinite_red_timing_point_count = 0
    nonpositive_red_timing_point_count = 0
    implausible_red_timing_point_count = 0
    negative_time_hitobject_map_count = 0
    audio_duration_failure_count = 0
    bpm_log_stats = _RunningStats()
    raw_bpm_distribution = _WeightedValueDistribution()
    raw_beat_length_distribution = _WeightedValueDistribution()

    for map_input in map_inputs:
        total_map_count += 1
        label = difficulty_bin_label(map_input.difficulty)
        if label is None:
            out_of_range_map_count += 1
            continue
        if map_input.audio_duration_ms < 0:
            audio_duration_failure_count += 1
            continue

        beatmap_path = Path(map_input.beatmap_path)
        try:
            red_timing_points = require_red_timing_points(beatmap_path)
        except InvalidRedTimingError as exc:
            invalid_red_timing_map_count += 1
            invalid_red_timing_point_count += exc.counts.total
            nonfinite_red_timing_point_count += exc.counts.nonfinite
            nonpositive_red_timing_point_count += exc.counts.nonpositive
            implausible_red_timing_point_count += exc.counts.implausible
            continue
        except MissingRedTimingError:
            missing_red_timing_map_count += 1
            continue

        hitobjects = parse_mania_hit_objects(beatmap_path, expected_key_count=4)
        try:
            generation_end_ms = _generation_end_ms(map_input.audio_duration_ms, hitobjects)
        except NegativeHitObjectTimeError:
            negative_time_hitobject_map_count += 1
            continue

        prepared_map = _PreparedDenseTimingMap(
            beatmap_path=beatmap_path,
            difficulty=map_input.difficulty,
            generation_end_ms=generation_end_ms,
            red_timing_points=red_timing_points,
            hitobjects=hitobjects,
            audio_path=Path(map_input.audio_path) if map_input.audio_path is not None else None,
        )
        prepared_maps.append(prepared_map)

        for input_start_ms in _input_starts(generation_end_ms):
            raw_beat_lengths = render_raw_beat_lengths_20ms_v1(
                red_timing_points,
                input_start_ms=input_start_ms,
                config=config,
            )
            raw_bpm_values = 60000.0 / raw_beat_lengths
            raw_beat_length_distribution.update_array(raw_beat_lengths)
            raw_bpm_distribution.update_array(raw_bpm_values)
            bpm_log_values = np.log(raw_bpm_values)
            bpm_log_stats.update_array(bpm_log_values)

    bpm_log_mean = bpm_log_stats.mean if bpm_log_stats.count else 0.0
    bpm_log_std = bpm_log_stats.std if bpm_log_stats.std > 0 else 1.0

    phase_error_stats = _RunningStats()
    bpm_norm_stats = _RunningStats()
    bin_accumulators = {label: _DenseTimingBinAccumulator(label) for label in DIFFICULTY_BIN_LABELS}
    timing_track_nan_count = 0
    timing_track_inf_count = 0
    beat_pulse_nonzero_count = 0
    bpm_norm_clipped_low_count = 0
    bpm_norm_clipped_high_count = 0
    window_count = 0
    frame_count = 0
    debug_plot_paths: list[str] = []
    debug_dir = Path(debug_plot_dir) if debug_plot_dir is not None else None
    if debug_dir is not None and debug_plot_count > 0:
        debug_dir.mkdir(parents=True, exist_ok=True)

    for prepared_map in prepared_maps:
        label = difficulty_bin_label(prepared_map.difficulty)
        if label is None:
            raise ValueError(f"prepared map outside supported difficulty range: {prepared_map.difficulty}")
        accumulator = bin_accumulators[label]
        accumulator.map_count += 1

        first_window_track: np.ndarray | None = None
        for input_start_ms in _input_starts(prepared_map.generation_end_ms):
            bpm_log_values = render_local_bpm_log_20ms_v1(
                prepared_map.red_timing_points,
                input_start_ms=input_start_ms,
                config=config,
            )
            pre_clip_bpm_norm = (bpm_log_values - bpm_log_mean) / bpm_log_std
            bpm_norm_clipped_low_count += int(np.count_nonzero(pre_clip_bpm_norm < -4.0))
            bpm_norm_clipped_high_count += int(np.count_nonzero(pre_clip_bpm_norm > 4.0))
            track = render_timing_track_20ms_v1(
                prepared_map.red_timing_points,
                input_start_ms=input_start_ms,
                bpm_log_mean=bpm_log_mean,
                bpm_log_std=bpm_log_std,
                config=config,
            )
            if first_window_track is None:
                first_window_track = track

            window_count += 1
            frame_count += int(track.shape[0])
            timing_track_nan_count += int(np.isnan(track).sum())
            timing_track_inf_count += int(np.isinf(track).sum())

            phase_norm = np.sqrt(track[:, 1].astype(np.float64) ** 2 + track[:, 2].astype(np.float64) ** 2)
            phase_error = np.abs(phase_norm - 1.0)
            phase_error_stats.update_array(phase_error)

            beat_pulse_nonzero = int(np.count_nonzero(track[:, 0] > 0.0))
            beat_pulse_nonzero_count += beat_pulse_nonzero
            bpm_norm_values = track[:, 3].astype(np.float64)
            bpm_norm_stats.update_array(bpm_norm_values)

            accumulator.window_count += 1
            accumulator.frame_count += int(track.shape[0])
            accumulator.beat_pulse_nonzero_count += beat_pulse_nonzero
            assert accumulator.bpm_norm_stats is not None
            accumulator.bpm_norm_stats.update_array(bpm_norm_values)

        if (
            debug_dir is not None
            and len(debug_plot_paths) < debug_plot_count
            and first_window_track is not None
        ):
            debug_path = debug_dir / f"dense_timing_debug_{len(debug_plot_paths):03d}.png"
            _write_debug_plot(
                debug_path,
                prepared_map=prepared_map,
                track=first_window_track,
                config=config,
            )
            debug_plot_paths.append(debug_path.as_posix())

    return DenseTimingAuditReport(
        total_map_count=total_map_count,
        audited_map_count=len(prepared_maps),
        out_of_range_map_count=out_of_range_map_count,
        missing_red_timing_map_count=missing_red_timing_map_count,
        invalid_red_timing_map_count=invalid_red_timing_map_count,
        invalid_red_timing_point_count=invalid_red_timing_point_count,
        nonfinite_red_timing_point_count=nonfinite_red_timing_point_count,
        nonpositive_red_timing_point_count=nonpositive_red_timing_point_count,
        implausible_red_timing_point_count=implausible_red_timing_point_count,
        negative_time_hitobject_map_count=negative_time_hitobject_map_count,
        audio_duration_failure_count=audio_duration_failure_count,
        window_count=window_count,
        frame_count=frame_count,
        timing_track_nan_count=timing_track_nan_count,
        timing_track_inf_count=timing_track_inf_count,
        phase_unit_norm_error_mean=phase_error_stats.mean if phase_error_stats.count else 0.0,
        phase_unit_norm_error_max=phase_error_stats.max if phase_error_stats.count else 0.0,
        beat_pulse_nonzero_ratio=_rate(beat_pulse_nonzero_count, frame_count),
        local_bpm_log_norm_mean=bpm_norm_stats.mean if bpm_norm_stats.count else 0.0,
        local_bpm_log_norm_std=bpm_norm_stats.std,
        local_bpm_log_norm_min=bpm_norm_stats.min if bpm_norm_stats.count else 0.0,
        local_bpm_log_norm_max=bpm_norm_stats.max if bpm_norm_stats.count else 0.0,
        raw_bpm_min=raw_bpm_distribution.min if raw_bpm_distribution.total_count else 0.0,
        raw_bpm_p01=raw_bpm_distribution.percentile(1),
        raw_bpm_p50=raw_bpm_distribution.percentile(50),
        raw_bpm_p99=raw_bpm_distribution.percentile(99),
        raw_bpm_max=raw_bpm_distribution.max if raw_bpm_distribution.total_count else 0.0,
        raw_beat_length_min=(
            raw_beat_length_distribution.min if raw_beat_length_distribution.total_count else 0.0
        ),
        raw_beat_length_max=(
            raw_beat_length_distribution.max if raw_beat_length_distribution.total_count else 0.0
        ),
        bpm_norm_clipped_low_count=bpm_norm_clipped_low_count,
        bpm_norm_clipped_high_count=bpm_norm_clipped_high_count,
        bpm_norm_clipped_ratio=_rate(bpm_norm_clipped_low_count + bpm_norm_clipped_high_count, frame_count),
        bpm_log_mean=bpm_log_mean,
        bpm_log_std=bpm_log_std,
        bins={label: bin_accumulators[label].to_report() for label in DIFFICULTY_BIN_LABELS},
        debug_plot_paths=debug_plot_paths,
    )


def build_dense_timing_gate_decision(
    report: DenseTimingAuditReport,
    *,
    max_timing_anomaly_map_ratio: float = MAX_TIMING_ANOMALY_MAP_RATIO,
) -> DenseTimingGateDecision:
    failure_reasons: list[str] = []

    accounted_map_count = (
        report.audited_map_count
        + report.out_of_range_map_count
        + report.missing_red_timing_map_count
        + report.invalid_red_timing_map_count
        + report.negative_time_hitobject_map_count
        + report.audio_duration_failure_count
    )
    coverage_status = "PASS" if accounted_map_count == report.total_map_count else "FAIL"
    if coverage_status == "FAIL":
        failure_reasons.append("audited and filtered map counts do not account for total_map_count")

    renderer_numerics_status = (
        "PASS"
        if (
            report.audited_map_count > 0
            and report.window_count > 0
            and report.frame_count > 0
            and report.timing_track_nan_count == 0
            and report.timing_track_inf_count == 0
            and report.phase_unit_norm_error_max <= 1e-5
            and report.bpm_log_std > 0
            and -4.000001 <= report.local_bpm_log_norm_min <= 4.000001
            and -4.000001 <= report.local_bpm_log_norm_max <= 4.000001
        )
        else "FAIL"
    )
    if renderer_numerics_status == "FAIL":
        failure_reasons.append("rendered dense timing numerics violate finite/unit/clip constraints")

    valid_timing_subset_status = (
        "PASS"
        if (
            report.audited_map_count > 0
            and report.raw_bpm_min >= MIN_VALID_RED_BPM - 1e-6
            and report.raw_bpm_max <= MAX_VALID_RED_BPM + 1e-6
            and report.raw_beat_length_min >= MIN_VALID_RED_BEAT_LENGTH_MS - 1e-6
            and report.raw_beat_length_max <= MAX_VALID_RED_BEAT_LENGTH_MS + 1e-6
        )
        else "FAIL"
    )
    if valid_timing_subset_status == "FAIL":
        failure_reasons.append("filtered timing subset raw BPM/beat-length extrema violate policy")

    timing_anomaly_map_ratio = _rate(report.invalid_red_timing_map_count, report.total_map_count)
    invalid_reason_total = (
        report.nonfinite_red_timing_point_count
        + report.nonpositive_red_timing_point_count
        + report.implausible_red_timing_point_count
    )
    timing_anomaly_status = (
        "PASS"
        if (
            report.missing_red_timing_map_count == 0
            and report.nonfinite_red_timing_point_count == 0
            and report.invalid_red_timing_point_count == invalid_reason_total
            and (
                report.invalid_red_timing_map_count == 0
                or report.invalid_red_timing_point_count > 0
            )
            and timing_anomaly_map_ratio <= max_timing_anomaly_map_ratio
        )
        else "FAIL"
    )
    if timing_anomaly_status == "FAIL":
        if report.missing_red_timing_map_count:
            failure_reasons.append("missing red timing maps are not accepted by the anomaly gate")
        if report.nonfinite_red_timing_point_count:
            failure_reasons.append("nonfinite red timing points are not accepted by the anomaly gate")
        if report.invalid_red_timing_point_count != invalid_reason_total:
            failure_reasons.append("invalid red timing point counts do not match classified reasons")
        if report.invalid_red_timing_map_count and report.invalid_red_timing_point_count == 0:
            failure_reasons.append("invalid red timing maps have no classified invalid timing points")
        if timing_anomaly_map_ratio > max_timing_anomaly_map_ratio:
            failure_reasons.append("timing anomaly map ratio exceeds configured cap")

    return DenseTimingGateDecision(
        status="PASS" if not failure_reasons else "FAIL",
        renderer_numerics_status=renderer_numerics_status,
        valid_timing_subset_status=valid_timing_subset_status,
        timing_anomaly_status=timing_anomaly_status,
        coverage_status=coverage_status,
        timing_anomaly_policy=TIMING_ANOMALY_POLICY,
        max_timing_anomaly_map_ratio=max_timing_anomaly_map_ratio,
        timing_anomaly_map_ratio=timing_anomaly_map_ratio,
        accounted_map_count=accounted_map_count,
        timing_track_version=TIMING_TRACK_VERSION,
        timing_frame_hop_ms=int(DEFAULT_TIMING_TRACK_CONFIG.frame_hop_ms),
        timing_frame_center_offset_ms=int(DEFAULT_TIMING_TRACK_CONFIG.frame_center_offset_ms),
        timing_frame_count_per_window=DEFAULT_TIMING_TRACK_CONFIG.frame_count,
        timing_channels=TIMING_TRACK_CHANNELS,
        pulse_shape="triangular",
        pulse_width_ms=int(DEFAULT_TIMING_TRACK_CONFIG.pulse_width_ms),
        missing_red_timing_map_count=report.missing_red_timing_map_count,
        invalid_red_timing_map_count=report.invalid_red_timing_map_count,
        invalid_red_timing_point_count=report.invalid_red_timing_point_count,
        nonfinite_red_timing_point_count=report.nonfinite_red_timing_point_count,
        nonpositive_red_timing_point_count=report.nonpositive_red_timing_point_count,
        implausible_red_timing_point_count=report.implausible_red_timing_point_count,
        timing_track_nan_count=report.timing_track_nan_count,
        timing_track_inf_count=report.timing_track_inf_count,
        phase_unit_norm_error_max=report.phase_unit_norm_error_max,
        local_bpm_log_norm_min=report.local_bpm_log_norm_min,
        local_bpm_log_norm_max=report.local_bpm_log_norm_max,
        raw_bpm_min=report.raw_bpm_min,
        raw_bpm_p01=report.raw_bpm_p01,
        raw_bpm_p50=report.raw_bpm_p50,
        raw_bpm_p99=report.raw_bpm_p99,
        raw_bpm_max=report.raw_bpm_max,
        raw_beat_length_min=report.raw_beat_length_min,
        raw_beat_length_max=report.raw_beat_length_max,
        bpm_norm_clipped_low_count=report.bpm_norm_clipped_low_count,
        bpm_norm_clipped_high_count=report.bpm_norm_clipped_high_count,
        bpm_norm_clipped_ratio=report.bpm_norm_clipped_ratio,
        bpm_log_mean=report.bpm_log_mean,
        bpm_log_std=report.bpm_log_std,
        failure_reasons=failure_reasons,
    )


def _generation_end_ms(audio_duration_ms: float, hitobjects: Sequence[ManiaHitObject]) -> int:
    max_quantized_event_time = 0
    for hitobject in hitobjects:
        q_start = quantize_10ms_half_up(hitobject.start_time_ms)
        max_quantized_event_time = max(max_quantized_event_time, q_start)
        if hitobject.kind == ManiaHitObjectKind.HOLD:
            q_end = quantize_10ms_half_up(hitobject.end_time_ms)
            if q_end > q_start:
                max_quantized_event_time = max(max_quantized_event_time, q_end)
    return max(ceil_10ms(audio_duration_ms), max_quantized_event_time + 10)


def _input_starts(generation_end_ms: int) -> Iterable[int]:
    write_start = 0
    while write_start < generation_end_ms:
        yield write_start - 2000
        write_start += WRITE_WINDOW_MS


def _write_debug_plot(
    output_path: Path,
    *,
    prepared_map: _PreparedDenseTimingMap,
    track: np.ndarray,
    config: TimingTrackConfig,
) -> None:
    input_start_ms = -2000
    frame_times = input_start_ms + config.frame_hop_ms * np.arange(track.shape[0]) + config.frame_center_offset_ms
    audio_energy = _audio_energy_for_debug_plot(prepared_map.audio_path, frame_count=track.shape[0])
    hitobject_times = [
        hitobject.start_time_ms
        for hitobject in prepared_map.hitobjects
        if input_start_ms <= hitobject.start_time_ms < input_start_ms + 12000
    ]

    fig, axes = plt.subplots(5, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(frame_times, audio_energy, linewidth=0.8)
    axes[0].set_ylabel("audio")
    axes[1].plot(frame_times, track[:, 0], linewidth=0.8)
    axes[1].set_ylabel("pulse")
    axes[2].plot(frame_times, track[:, 1], label="sin", linewidth=0.8)
    axes[2].plot(frame_times, track[:, 2], label="cos", linewidth=0.8)
    axes[2].set_ylabel("phase")
    axes[2].legend(loc="upper right")
    axes[3].plot(frame_times, track[:, 3], linewidth=0.8)
    axes[3].set_ylabel("bpm")
    axes[4].eventplot(hitobject_times, lineoffsets=0.5, linelengths=0.8)
    axes[4].set_ylabel("objects")
    axes[4].set_xlabel("time_ms")
    fig.suptitle(prepared_map.beatmap_path.name)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def _audio_energy_for_debug_plot(audio_path: Path | None, *, frame_count: int) -> np.ndarray:
    if audio_path is None or not audio_path.exists():
        return np.zeros(frame_count, dtype=np.float32)
    try:
        waveform = load_audio_file(audio_path, sample_rate=16000)
    except Exception:
        return np.zeros(frame_count, dtype=np.float32)
    samples_per_frame = int(16000 * 0.02)
    values = np.zeros(frame_count, dtype=np.float32)
    for frame_index in range(frame_count):
        # First debug window input is [-2000, 10000), so the first 2s are silence.
        start = (frame_index * samples_per_frame) - int(16000 * 2.0)
        end = start + samples_per_frame
        if end <= 0:
            continue
        if start >= waveform.shape[0]:
            break
        frame = waveform[max(0, start) : end]
        if frame.size:
            values[frame_index] = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))
    return values


def _rate(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator
