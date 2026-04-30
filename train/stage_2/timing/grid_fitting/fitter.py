from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from train.stage_2.timing.schema import FittedTimingGrid, FrameTimingPrediction, TimingSegment


@dataclass(frozen=True)
class GridFitterConfig:
    min_bpm: float = 20.0
    max_bpm: float = 1000.0
    bpm_step: float = 0.5
    offset_step_ms: float = 20.0
    pulse_width_ms: float = 40.0
    double_tempo_score_ratio_threshold: float = 0.95
    max_segments: int = 16
    min_segment_duration_ms: float = 8000.0
    split_step_ms: float = 4000.0
    split_score_improvement_threshold: float = 0.02
    split_phase_change_threshold_ms: float = 10.0
    autocorrelation_candidate_count: int = 16
    bpm_search_window_ratio: float = 0.08
    bpm_search_window_min_bpm: float = 2.0
    max_grid_candidates_per_segment: int = 1000
    max_split_candidates_per_segment: int = 4
    initial_batch_split_candidate_count: int = 16
    initial_batch_split_min_candidate_count: int = 8
    initial_batch_split_max_parent_score: float = 0.75
    long_prediction_duration_seconds: float = 600.0
    long_max_segments: int = 20
    long_max_grid_candidates_per_segment: int = 1000
    long_max_split_candidates_per_segment: int = 5
    long_downbeat_refine_candidate_count: int = 20
    downbeat_period_beats: int = 4
    downbeat_tie_score_margin: float = 0.05
    downbeat_split_score_bonus: float = 0.5
    downbeat_refine_candidate_count: int = 16
    merge_similar_segments: bool = True
    merge_bpm_tolerance: float = 1.5
    merge_relative_bpm_tolerance: float = 0.005
    merge_phase_tolerance_ms: float = 35.0
    merge_many_similar_min_segments: int = 4
    merge_many_similar_bpm_tolerance: float = 2.0
    merge_alias_min_segments: int = 4
    merge_alias_requires_downbeat_signal: bool = True
    merge_alias_bpm_tolerance: float = 2.0
    merge_alias_phase_tolerance_ms: float = 60.0
    merge_alias_max_fit_score: float = 0.92

    def __post_init__(self) -> None:
        if not np.isfinite(self.min_bpm) or self.min_bpm <= 0.0:
            raise ValueError(f"min_bpm must be positive and finite, got {self.min_bpm!r}")
        if not np.isfinite(self.max_bpm) or self.max_bpm <= self.min_bpm:
            raise ValueError(f"max_bpm must be finite and greater than min_bpm, got {self.max_bpm!r}")
        if not np.isfinite(self.bpm_step) or self.bpm_step <= 0.0:
            raise ValueError(f"bpm_step must be positive and finite, got {self.bpm_step!r}")
        if not np.isfinite(self.offset_step_ms) or self.offset_step_ms <= 0.0:
            raise ValueError(f"offset_step_ms must be positive and finite, got {self.offset_step_ms!r}")
        if not np.isfinite(self.pulse_width_ms) or self.pulse_width_ms <= 0.0:
            raise ValueError(f"pulse_width_ms must be positive and finite, got {self.pulse_width_ms!r}")
        if (
            not np.isfinite(self.double_tempo_score_ratio_threshold)
            or self.double_tempo_score_ratio_threshold < 0.0
        ):
            raise ValueError(
                "double_tempo_score_ratio_threshold must be non-negative and finite, "
                f"got {self.double_tempo_score_ratio_threshold!r}",
            )
        if self.max_segments <= 0:
            raise ValueError(f"max_segments must be positive, got {self.max_segments!r}")
        if not np.isfinite(self.min_segment_duration_ms) or self.min_segment_duration_ms <= 0.0:
            raise ValueError(
                "min_segment_duration_ms must be positive and finite, "
                f"got {self.min_segment_duration_ms!r}",
            )
        if not np.isfinite(self.split_step_ms) or self.split_step_ms <= 0.0:
            raise ValueError(f"split_step_ms must be positive and finite, got {self.split_step_ms!r}")
        if (
            not np.isfinite(self.split_score_improvement_threshold)
            or self.split_score_improvement_threshold < 0.0
        ):
            raise ValueError(
                "split_score_improvement_threshold must be non-negative and finite, "
                f"got {self.split_score_improvement_threshold!r}",
            )
        if not np.isfinite(self.split_phase_change_threshold_ms) or self.split_phase_change_threshold_ms < 0.0:
            raise ValueError(
                "split_phase_change_threshold_ms must be non-negative and finite, "
                f"got {self.split_phase_change_threshold_ms!r}",
            )
        if self.autocorrelation_candidate_count <= 0:
            raise ValueError(
                "autocorrelation_candidate_count must be positive, "
                f"got {self.autocorrelation_candidate_count!r}",
            )
        if not np.isfinite(self.bpm_search_window_ratio) or self.bpm_search_window_ratio < 0.0:
            raise ValueError(
                "bpm_search_window_ratio must be non-negative and finite, "
                f"got {self.bpm_search_window_ratio!r}",
            )
        if not np.isfinite(self.bpm_search_window_min_bpm) or self.bpm_search_window_min_bpm < 0.0:
            raise ValueError(
                "bpm_search_window_min_bpm must be non-negative and finite, "
                f"got {self.bpm_search_window_min_bpm!r}",
            )
        if self.max_grid_candidates_per_segment <= 0:
            raise ValueError(
                "max_grid_candidates_per_segment must be positive, "
                f"got {self.max_grid_candidates_per_segment!r}",
            )
        if self.max_split_candidates_per_segment <= 0:
            raise ValueError(
                "max_split_candidates_per_segment must be positive, "
                f"got {self.max_split_candidates_per_segment!r}",
            )
        if self.initial_batch_split_candidate_count < 0:
            raise ValueError(
                "initial_batch_split_candidate_count must be non-negative, "
                f"got {self.initial_batch_split_candidate_count!r}",
            )
        if self.initial_batch_split_min_candidate_count < 0:
            raise ValueError(
                "initial_batch_split_min_candidate_count must be non-negative, "
                f"got {self.initial_batch_split_min_candidate_count!r}",
            )
        if (
            not np.isfinite(self.initial_batch_split_max_parent_score)
            or self.initial_batch_split_max_parent_score < 0.0
        ):
            raise ValueError(
                "initial_batch_split_max_parent_score must be non-negative and finite, "
                f"got {self.initial_batch_split_max_parent_score!r}",
            )
        if (
            not np.isfinite(self.long_prediction_duration_seconds)
            or self.long_prediction_duration_seconds < 0.0
        ):
            raise ValueError(
                "long_prediction_duration_seconds must be non-negative and finite, "
                f"got {self.long_prediction_duration_seconds!r}",
            )
        if self.long_max_segments <= 0:
            raise ValueError(f"long_max_segments must be positive, got {self.long_max_segments!r}")
        if self.long_max_grid_candidates_per_segment <= 0:
            raise ValueError(
                "long_max_grid_candidates_per_segment must be positive, "
                f"got {self.long_max_grid_candidates_per_segment!r}",
            )
        if self.long_max_split_candidates_per_segment <= 0:
            raise ValueError(
                "long_max_split_candidates_per_segment must be positive, "
                f"got {self.long_max_split_candidates_per_segment!r}",
            )
        if self.long_downbeat_refine_candidate_count <= 0:
            raise ValueError(
                "long_downbeat_refine_candidate_count must be positive, "
                f"got {self.long_downbeat_refine_candidate_count!r}",
            )
        if self.downbeat_period_beats <= 0:
            raise ValueError(f"downbeat_period_beats must be positive, got {self.downbeat_period_beats!r}")
        if not np.isfinite(self.downbeat_tie_score_margin) or self.downbeat_tie_score_margin < 0.0:
            raise ValueError(
                "downbeat_tie_score_margin must be non-negative and finite, "
                f"got {self.downbeat_tie_score_margin!r}",
            )
        if not np.isfinite(self.downbeat_split_score_bonus) or self.downbeat_split_score_bonus < 0.0:
            raise ValueError(
                "downbeat_split_score_bonus must be non-negative and finite, "
                f"got {self.downbeat_split_score_bonus!r}",
            )
        if self.downbeat_refine_candidate_count <= 0:
            raise ValueError(
                "downbeat_refine_candidate_count must be positive, "
                f"got {self.downbeat_refine_candidate_count!r}",
            )
        if not np.isfinite(self.merge_bpm_tolerance) or self.merge_bpm_tolerance < 0.0:
            raise ValueError(
                "merge_bpm_tolerance must be non-negative and finite, "
                f"got {self.merge_bpm_tolerance!r}",
            )
        if not np.isfinite(self.merge_relative_bpm_tolerance) or self.merge_relative_bpm_tolerance < 0.0:
            raise ValueError(
                "merge_relative_bpm_tolerance must be non-negative and finite, "
                f"got {self.merge_relative_bpm_tolerance!r}",
            )
        if not np.isfinite(self.merge_phase_tolerance_ms) or self.merge_phase_tolerance_ms < 0.0:
            raise ValueError(
                "merge_phase_tolerance_ms must be non-negative and finite, "
                f"got {self.merge_phase_tolerance_ms!r}",
            )
        if self.merge_many_similar_min_segments <= 0:
            raise ValueError(
                "merge_many_similar_min_segments must be positive, "
                f"got {self.merge_many_similar_min_segments!r}",
            )
        if not np.isfinite(self.merge_many_similar_bpm_tolerance) or self.merge_many_similar_bpm_tolerance < 0.0:
            raise ValueError(
                "merge_many_similar_bpm_tolerance must be non-negative and finite, "
                f"got {self.merge_many_similar_bpm_tolerance!r}",
            )
        if self.merge_alias_min_segments <= 0:
            raise ValueError(
                "merge_alias_min_segments must be positive, "
                f"got {self.merge_alias_min_segments!r}",
            )
        if not np.isfinite(self.merge_alias_bpm_tolerance) or self.merge_alias_bpm_tolerance < 0.0:
            raise ValueError(
                "merge_alias_bpm_tolerance must be non-negative and finite, "
                f"got {self.merge_alias_bpm_tolerance!r}",
            )
        if not np.isfinite(self.merge_alias_phase_tolerance_ms) or self.merge_alias_phase_tolerance_ms < 0.0:
            raise ValueError(
                "merge_alias_phase_tolerance_ms must be non-negative and finite, "
                f"got {self.merge_alias_phase_tolerance_ms!r}",
            )
        if not np.isfinite(self.merge_alias_max_fit_score) or self.merge_alias_max_fit_score < 0.0:
            raise ValueError(
                "merge_alias_max_fit_score must be non-negative and finite, "
                f"got {self.merge_alias_max_fit_score!r}",
            )


@dataclass(frozen=True)
class TimingFitDiagnostics:
    selected_period_frames: float
    selected_offset_frames: float
    selected_bpm: float
    candidate_count: int
    half_tempo_score: float
    double_tempo_score: float
    raw_selected_bpm: float
    raw_score: float
    tempo_multiplier: float


@dataclass(frozen=True)
class TimingFitResult:
    grid: FittedTimingGrid
    score: float
    diagnostics: TimingFitDiagnostics


@dataclass(frozen=True)
class _SegmentFit:
    start_frame: int
    end_frame: int
    score: float
    beat_length_ms: float
    offset_ms: float
    half_tempo_score: float
    double_tempo_score: float
    raw_bpm: float
    raw_score: float
    tempo_multiplier: float
    candidate_count: int

    @property
    def frame_count(self) -> int:
        return self.end_frame - self.start_frame

    @property
    def bpm(self) -> float:
        return 60000.0 / self.beat_length_ms


@dataclass(frozen=True)
class _SplitCandidate:
    frame: int
    score: float


@dataclass(frozen=True)
class _ChangeTimeCandidate:
    time_ms: float
    score: float


@dataclass(frozen=True)
class _EvaluatedSplit:
    segment_index: int
    candidate: _SplitCandidate
    left_fit: _SegmentFit
    right_fit: _SegmentFit
    improvement: float


@dataclass(frozen=True)
class _GridCandidate:
    score: float
    bpm: float
    beat_length_ms: float
    offset_ms: float


class GridFitter:
    def __init__(self, config: GridFitterConfig = GridFitterConfig()) -> None:
        self.config = config

    def fit(self, prediction: FrameTimingPrediction) -> TimingFitResult:
        return fit_timing_grid(prediction, config=self.config)


def fit_timing_grid(
    prediction: FrameTimingPrediction,
    *,
    config: GridFitterConfig = GridFitterConfig(),
) -> TimingFitResult:
    config = _effective_config_for_prediction(
        prediction.frame_count,
        frame_rate_hz=prediction.frame_rate_hz,
        config=config,
    )
    min_period_frames, max_period_frames = _candidate_period_frame_bounds(
        prediction.frame_rate_hz,
        config=config,
    )
    if prediction.frame_count < max_period_frames:
        raise ValueError(
            "prediction is too short to fit the configured tempo range: "
            f"frame_count={prediction.frame_count}, required>={max_period_frames}",
        )

    signal = prediction.beat_prob.astype(np.float64, copy=False)
    downbeat_signal = prediction.downbeat_prob.astype(np.float64, copy=False)
    if float(np.linalg.norm(signal - float(np.mean(signal)))) == 0.0:
        raise ValueError("beat_prob contains no beat signal")

    frame_times_ms = np.arange(prediction.frame_count, dtype=np.float64) / prediction.frame_rate_hz * 1000.0
    initial_fit = _fit_segment_range(
        signal,
        frame_times_ms=frame_times_ms,
        downbeat_signal=downbeat_signal,
        start_frame=0,
        end_frame=prediction.frame_count,
        config=config,
    )
    segment_fits = _split_segment_range(
        signal,
        frame_times_ms=frame_times_ms,
        downbeat_signal=downbeat_signal,
        fit=initial_fit,
        config=config,
        remaining_splits=config.max_segments - 1,
    )
    candidate_count = sum(fit.candidate_count for fit in segment_fits)
    best_score = _weighted_score(segment_fits)
    first_fit = segment_fits[0]
    grid = FittedTimingGrid(segments=_timing_segments_from_fits(segment_fits, frame_times_ms, config=config))

    return TimingFitResult(
        grid=grid,
        score=float(best_score),
        diagnostics=TimingFitDiagnostics(
            selected_period_frames=float(first_fit.beat_length_ms / 1000.0 * prediction.frame_rate_hz),
            selected_offset_frames=float(first_fit.offset_ms / 1000.0 * prediction.frame_rate_hz),
            selected_bpm=float(first_fit.bpm),
            candidate_count=candidate_count,
            half_tempo_score=float(first_fit.half_tempo_score),
            double_tempo_score=float(first_fit.double_tempo_score),
            raw_selected_bpm=float(first_fit.raw_bpm),
            raw_score=float(first_fit.raw_score),
            tempo_multiplier=first_fit.tempo_multiplier,
        ),
    )


def _effective_config_for_prediction(
    frame_count: int,
    *,
    frame_rate_hz: float,
    config: GridFitterConfig,
) -> GridFitterConfig:
    duration_seconds = float(frame_count) / frame_rate_hz
    if duration_seconds < config.long_prediction_duration_seconds:
        return config
    return replace(
        config,
        max_segments=max(config.max_segments, config.long_max_segments),
        max_grid_candidates_per_segment=max(
            config.max_grid_candidates_per_segment,
            config.long_max_grid_candidates_per_segment,
        ),
        max_split_candidates_per_segment=max(
            config.max_split_candidates_per_segment,
            config.long_max_split_candidates_per_segment,
        ),
        downbeat_refine_candidate_count=max(
            config.downbeat_refine_candidate_count,
            config.long_downbeat_refine_candidate_count,
        ),
    )


def _fit_segment_range(
    signal: NDArray[np.float64],
    *,
    frame_times_ms: NDArray[np.float64],
    downbeat_signal: NDArray[np.float64] | None = None,
    start_frame: int,
    end_frame: int,
    config: GridFitterConfig,
) -> _SegmentFit:
    segment_signal = signal[start_frame:end_frame]
    centered_signal = segment_signal - float(np.mean(segment_signal))
    signal_norm = float(np.linalg.norm(centered_signal))
    if signal_norm == 0.0:
        return _SegmentFit(
            start_frame=start_frame,
            end_frame=end_frame,
            score=-np.inf,
            beat_length_ms=60000.0 / config.min_bpm,
            offset_ms=0.0,
            half_tempo_score=-np.inf,
            double_tempo_score=-np.inf,
            raw_bpm=config.min_bpm,
            raw_score=-np.inf,
            tempo_multiplier=1.0,
            candidate_count=0,
        )

    segment_frame_times_ms = frame_times_ms[start_frame:end_frame]
    segment_downbeat_signal = None if downbeat_signal is None else downbeat_signal[start_frame:end_frame]
    downbeat_centered_signal, downbeat_signal_norm = _centered_signal_and_norm(segment_downbeat_signal)
    candidate_count = 0
    candidates: list[_GridCandidate] = []

    frame_rate_hz = _frame_rate_hz_from_times(segment_frame_times_ms)
    for bpm in _candidate_bpms(
        centered_signal,
        frame_rate_hz=frame_rate_hz,
        config=config,
    ):
        beat_length_ms = 60000.0 / bpm
        for offset_ms in _candidate_offsets_ms(beat_length_ms, config=config):
            score = _score_grid(
                centered_signal,
                signal_norm=signal_norm,
                frame_times_ms=segment_frame_times_ms,
                beat_length_ms=beat_length_ms,
                offset_ms=offset_ms,
                pulse_width_ms=config.pulse_width_ms,
            )
            candidate_count += 1
            candidates.append(
                _GridCandidate(
                    score=float(score),
                    bpm=float(bpm),
                    beat_length_ms=float(beat_length_ms),
                    offset_ms=float(offset_ms),
                )
            )

    best_score, best_downbeat_score, best_bpm, best_beat_length_ms, best_offset_ms = _best_grid_candidate(
        candidates,
        downbeat_centered_signal=downbeat_centered_signal,
        downbeat_signal_norm=downbeat_signal_norm,
        frame_times_ms=segment_frame_times_ms,
        config=config,
    )

    raw_bpm = best_bpm
    raw_score = best_score
    raw_downbeat_score = best_downbeat_score
    half_tempo_score, _, _ = _best_bpm_fit(
        centered_signal,
        signal_norm=signal_norm,
        frame_times_ms=segment_frame_times_ms,
        bpm=raw_bpm / 2.0,
        pulse_width_ms=config.pulse_width_ms,
        downbeat_centered_signal=downbeat_centered_signal,
        downbeat_signal_norm=downbeat_signal_norm,
        config=config,
    )
    double_tempo_score, double_tempo_offset_ms, double_tempo_downbeat_score = _best_bpm_fit(
        centered_signal,
        signal_norm=signal_norm,
        frame_times_ms=segment_frame_times_ms,
        bpm=raw_bpm * 2.0,
        pulse_width_ms=config.pulse_width_ms,
        downbeat_centered_signal=downbeat_centered_signal,
        downbeat_signal_norm=downbeat_signal_norm,
        config=config,
    )

    tempo_multiplier = 1.0
    if (
        double_tempo_score != -np.inf
        and double_tempo_score >= raw_score * config.double_tempo_score_ratio_threshold
        and not _downbeat_rejects_close_tempo_alias(
            raw_score,
            raw_downbeat_score,
            double_tempo_score,
            double_tempo_downbeat_score,
            config=config,
        )
    ):
        best_bpm = raw_bpm * 2.0
        best_beat_length_ms = 60000.0 / best_bpm
        best_offset_ms = double_tempo_offset_ms
        best_score = double_tempo_score
        tempo_multiplier = 2.0

    return _SegmentFit(
        start_frame=start_frame,
        end_frame=end_frame,
        score=float(best_score),
        beat_length_ms=float(best_beat_length_ms),
        offset_ms=float(best_offset_ms),
        half_tempo_score=float(half_tempo_score),
        double_tempo_score=float(double_tempo_score),
        raw_bpm=float(raw_bpm),
        raw_score=float(raw_score),
        tempo_multiplier=tempo_multiplier,
        candidate_count=candidate_count,
    )


def _split_segment_range(
    signal: NDArray[np.float64],
    *,
    frame_times_ms: NDArray[np.float64],
    downbeat_signal: NDArray[np.float64] | None,
    fit: _SegmentFit,
    config: GridFitterConfig,
    remaining_splits: int,
) -> list[_SegmentFit]:
    if remaining_splits <= 0:
        return [fit]

    fit_cache: dict[tuple[int, int], _SegmentFit] = {(fit.start_frame, fit.end_frame): fit}
    segment_fits: list[_SegmentFit]
    if config.initial_batch_split_candidate_count > 0:
        segment_fits = _batch_split_for_fit(
            signal,
            frame_times_ms=frame_times_ms,
            downbeat_signal=downbeat_signal,
            fit=fit,
            config=config,
            remaining_splits=remaining_splits,
            fit_cache=fit_cache,
        ) or [fit]
        remaining_splits -= len(segment_fits) - 1
    else:
        segment_fits = [fit]

    split_cache: dict[tuple[int, int], _EvaluatedSplit | None] = {}

    while remaining_splits > 0:
        best_split: _EvaluatedSplit | None = None
        for segment_index, segment_fit in enumerate(segment_fits):
            cache_key = (segment_fit.start_frame, segment_fit.end_frame)
            if cache_key not in split_cache:
                split_cache[cache_key] = _best_split_for_fit(
                    signal,
                    frame_times_ms=frame_times_ms,
                    downbeat_signal=downbeat_signal,
                    fit=segment_fit,
                    segment_index=segment_index,
                    config=config,
                    fit_cache=fit_cache,
                )
            cached_split = split_cache[cache_key]
            if cached_split is None:
                continue
            evaluated_split = _EvaluatedSplit(
                segment_index=segment_index,
                candidate=cached_split.candidate,
                left_fit=cached_split.left_fit,
                right_fit=cached_split.right_fit,
                improvement=cached_split.improvement,
            )
            if best_split is None or _split_is_better(evaluated_split, best_split):
                best_split = evaluated_split

        if best_split is None or best_split.improvement < config.split_score_improvement_threshold:
            break

        segment_fits[best_split.segment_index : best_split.segment_index + 1] = [
            best_split.left_fit,
            best_split.right_fit,
        ]
        remaining_splits -= 1

    return segment_fits


def _batch_split_for_fit(
    signal: NDArray[np.float64],
    *,
    frame_times_ms: NDArray[np.float64],
    downbeat_signal: NDArray[np.float64] | None,
    fit: _SegmentFit,
    config: GridFitterConfig,
    remaining_splits: int,
    fit_cache: dict[tuple[int, int], _SegmentFit],
) -> list[_SegmentFit] | None:
    candidates = _detect_change_split_candidates(
        signal,
        frame_times_ms=frame_times_ms,
        downbeat_signal=downbeat_signal,
        fit=fit,
        config=config,
    )
    if len(candidates) < 2:
        return None
    if (
        len(candidates) < config.initial_batch_split_min_candidate_count
        and fit.score > config.initial_batch_split_max_parent_score
    ):
        return None

    selected_count = min(remaining_splits, config.initial_batch_split_candidate_count)
    selected_candidates = sorted(candidates[:selected_count], key=lambda candidate: candidate.frame)
    region_edges = [fit.start_frame, *(candidate.frame for candidate in selected_candidates), fit.end_frame]
    segment_fits = [
        _cached_fit_segment_range(
            signal,
            frame_times_ms=frame_times_ms,
            downbeat_signal=downbeat_signal,
            start_frame=start_frame,
            end_frame=end_frame,
            config=config,
            fit_cache=fit_cache,
        )
        for start_frame, end_frame in zip(region_edges, region_edges[1:])
    ]
    segment_fits = _merge_adjacent_segment_fits(
        segment_fits,
        signal,
        frame_times_ms=frame_times_ms,
        downbeat_signal=downbeat_signal,
        config=config,
        fit_cache=fit_cache,
    )
    if len(segment_fits) < 2:
        return None
    if _weighted_score(segment_fits) - fit.score < config.split_score_improvement_threshold:
        return None
    return segment_fits


def _merge_adjacent_segment_fits(
    segment_fits: Sequence[_SegmentFit],
    signal: NDArray[np.float64],
    *,
    frame_times_ms: NDArray[np.float64],
    downbeat_signal: NDArray[np.float64] | None,
    config: GridFitterConfig,
    fit_cache: dict[tuple[int, int], _SegmentFit],
) -> list[_SegmentFit]:
    merged_fits: list[_SegmentFit] = []
    has_downbeat_signal = (
        downbeat_signal is not None
        and float(np.linalg.norm(downbeat_signal - float(np.mean(downbeat_signal)))) > 0.0
    )
    allow_alias_merge = (
        len(segment_fits) >= config.merge_alias_min_segments
        and (has_downbeat_signal or not config.merge_alias_requires_downbeat_signal)
    )
    allow_loose_similar_merge = len(segment_fits) >= config.merge_many_similar_min_segments
    for fit in segment_fits:
        if not merged_fits:
            merged_fits.append(fit)
            continue
        previous_fit = merged_fits[-1]
        pair_allows_alias_merge = allow_alias_merge and fit.score <= config.merge_alias_max_fit_score
        if not _segment_fits_are_mergeable(
            previous_fit,
            fit,
            frame_times_ms=frame_times_ms,
            config=config,
            allow_alias=pair_allows_alias_merge,
            allow_loose_similar=allow_loose_similar_merge,
        ):
            merged_fits.append(fit)
            continue
        merged_fits[-1] = _cached_fit_segment_range(
            signal,
            frame_times_ms=frame_times_ms,
            downbeat_signal=downbeat_signal,
            start_frame=previous_fit.start_frame,
            end_frame=fit.end_frame,
            config=config,
            fit_cache=fit_cache,
        )
    return merged_fits


def _best_split_for_fit(
    signal: NDArray[np.float64],
    *,
    frame_times_ms: NDArray[np.float64],
    downbeat_signal: NDArray[np.float64] | None,
    fit: _SegmentFit,
    segment_index: int,
    config: GridFitterConfig,
    fit_cache: dict[tuple[int, int], _SegmentFit],
) -> _EvaluatedSplit | None:
    candidates = _detect_change_split_candidates(
        signal,
        frame_times_ms=frame_times_ms,
        downbeat_signal=downbeat_signal,
        fit=fit,
        config=config,
    )
    best_split: _EvaluatedSplit | None = None
    for candidate in candidates[: config.max_split_candidates_per_segment]:
        left_fit = _cached_fit_segment_range(
            signal,
            frame_times_ms=frame_times_ms,
            downbeat_signal=downbeat_signal,
            start_frame=fit.start_frame,
            end_frame=candidate.frame,
            config=config,
            fit_cache=fit_cache,
        )
        right_fit = _cached_fit_segment_range(
            signal,
            frame_times_ms=frame_times_ms,
            downbeat_signal=downbeat_signal,
            start_frame=candidate.frame,
            end_frame=fit.end_frame,
            config=config,
            fit_cache=fit_cache,
        )
        if _segment_fits_are_mergeable(left_fit, right_fit, frame_times_ms=frame_times_ms, config=config):
            continue
        improvement = _weighted_score((left_fit, right_fit)) - fit.score
        evaluated_split = _EvaluatedSplit(
            segment_index=segment_index,
            candidate=candidate,
            left_fit=left_fit,
            right_fit=right_fit,
            improvement=float(improvement),
        )
        if best_split is None or _split_is_better(evaluated_split, best_split):
            best_split = evaluated_split

    return best_split


def _cached_fit_segment_range(
    signal: NDArray[np.float64],
    *,
    frame_times_ms: NDArray[np.float64],
    downbeat_signal: NDArray[np.float64] | None,
    start_frame: int,
    end_frame: int,
    config: GridFitterConfig,
    fit_cache: dict[tuple[int, int], _SegmentFit],
) -> _SegmentFit:
    cache_key = (start_frame, end_frame)
    if cache_key not in fit_cache:
        fit_cache[cache_key] = _fit_segment_range(
            signal,
            frame_times_ms=frame_times_ms,
            downbeat_signal=downbeat_signal,
            start_frame=start_frame,
            end_frame=end_frame,
            config=config,
        )
    return fit_cache[cache_key]


def _detect_change_split_candidates(
    signal: NDArray[np.float64],
    *,
    frame_times_ms: NDArray[np.float64],
    downbeat_signal: NDArray[np.float64] | None,
    fit: _SegmentFit,
    config: GridFitterConfig,
) -> list[_SplitCandidate]:
    segment_signal = signal[fit.start_frame : fit.end_frame]
    segment_frame_times_ms = frame_times_ms[fit.start_frame : fit.end_frame]
    peak_times_ms = _beat_peak_times_ms(segment_signal, frame_times_ms=segment_frame_times_ms, config=config)
    if peak_times_ms.shape[0] < 4:
        return []

    frame_step_ms = _frame_step_ms(frame_times_ms)
    min_segment_frames = max(1, int(round(config.min_segment_duration_ms / frame_step_ms)))
    downbeat_peak_times_ms = _segment_downbeat_peak_times_ms(
        downbeat_signal,
        frame_times_ms=frame_times_ms,
        fit=fit,
        config=config,
    )
    candidate_times = _candidate_change_times_from_peaks(peak_times_ms, fit=fit, config=config)

    candidates_by_frame: dict[int, _SplitCandidate] = {}
    for candidate in candidate_times:
        split_frame = int(np.searchsorted(frame_times_ms, candidate.time_ms, side="left"))
        if not _split_respects_minimum_duration(split_frame, fit=fit, min_segment_frames=min_segment_frames):
            continue
        candidate_score = candidate.score + _downbeat_boundary_bonus(
            candidate.time_ms,
            downbeat_peak_times_ms=downbeat_peak_times_ms,
            fit=fit,
            config=config,
        )
        existing = candidates_by_frame.get(split_frame)
        if existing is None or candidate_score > existing.score:
            candidates_by_frame[split_frame] = _SplitCandidate(frame=split_frame, score=float(candidate_score))

    return sorted(candidates_by_frame.values(), key=lambda candidate: candidate.score, reverse=True)


def _segment_downbeat_peak_times_ms(
    downbeat_signal: NDArray[np.float64] | None,
    *,
    frame_times_ms: NDArray[np.float64],
    fit: _SegmentFit,
    config: GridFitterConfig,
) -> NDArray[np.float64]:
    if downbeat_signal is None:
        return np.asarray([], dtype=np.float64)
    segment_downbeat_signal = downbeat_signal[fit.start_frame : fit.end_frame]
    if float(np.linalg.norm(segment_downbeat_signal - float(np.mean(segment_downbeat_signal)))) == 0.0:
        return np.asarray([], dtype=np.float64)
    return _beat_peak_times_ms(
        segment_downbeat_signal,
        frame_times_ms=frame_times_ms[fit.start_frame : fit.end_frame],
        config=config,
    )


def _downbeat_boundary_bonus(
    candidate_time_ms: float,
    *,
    downbeat_peak_times_ms: NDArray[np.float64],
    fit: _SegmentFit,
    config: GridFitterConfig,
) -> float:
    if downbeat_peak_times_ms.shape[0] == 0:
        return 0.0
    nearest_distance_ms = float(np.min(np.abs(downbeat_peak_times_ms - candidate_time_ms)))
    tolerance_ms = min(fit.beat_length_ms * 0.5, config.min_segment_duration_ms * 0.125)
    if nearest_distance_ms > tolerance_ms:
        return 0.0
    closeness = 1.0 - nearest_distance_ms / max(tolerance_ms, 1e-6)
    return float(config.downbeat_split_score_bonus * closeness)


def _split_is_better(candidate: _EvaluatedSplit, incumbent: _EvaluatedSplit) -> bool:
    if candidate.improvement > incumbent.improvement:
        return True
    if candidate.improvement < incumbent.improvement:
        return False
    return candidate.candidate.score > incumbent.candidate.score


def _beat_peak_times_ms(
    signal: NDArray[np.float64],
    *,
    frame_times_ms: NDArray[np.float64],
    config: GridFitterConfig,
) -> NDArray[np.float64]:
    if signal.shape[0] < 3:
        return np.asarray([], dtype=np.float64)

    threshold = max(float(np.mean(signal) + np.std(signal)), float(np.max(signal) * 0.35))
    candidate_indices = np.flatnonzero(
        (signal[1:-1] >= signal[:-2])
        & (signal[1:-1] > signal[2:])
        & (signal[1:-1] >= threshold)
    ) + 1
    if candidate_indices.shape[0] == 0:
        return np.asarray([], dtype=np.float64)

    min_peak_distance_ms = config.offset_step_ms
    selected_indices: list[int] = []
    last_selected_time_ms = -np.inf
    for index in candidate_indices:
        time_ms = float(frame_times_ms[index])
        if time_ms - last_selected_time_ms >= min_peak_distance_ms:
            selected_indices.append(int(index))
            last_selected_time_ms = time_ms
        elif selected_indices and signal[index] > signal[selected_indices[-1]]:
            selected_indices[-1] = int(index)
            last_selected_time_ms = time_ms

    return frame_times_ms[np.asarray(selected_indices, dtype=np.int64)]


def _candidate_change_times_from_peaks(
    peak_times_ms: NDArray[np.float64],
    *,
    fit: _SegmentFit,
    config: GridFitterConfig,
) -> list[_ChangeTimeCandidate]:
    intervals_ms = np.diff(peak_times_ms)
    if intervals_ms.shape[0] < 3:
        return []

    interval_change_threshold_ms = max(config.offset_step_ms, config.split_phase_change_threshold_ms * 2.0)
    local_interval_count = 4
    candidates: list[_ChangeTimeCandidate] = []
    phase_errors_ms = _peak_phase_errors_ms(peak_times_ms, fit=fit)

    for split_index in range(local_interval_count, intervals_ms.shape[0] - local_interval_count + 1):
        before_intervals = intervals_ms[split_index - local_interval_count : split_index]
        after_intervals = intervals_ms[split_index : split_index + local_interval_count]
        if before_intervals.shape[0] < 3 or after_intervals.shape[0] < 3:
            continue
        interval_change_ms = abs(float(np.median(after_intervals)) - float(np.median(before_intervals)))
        before_phase_error_ms = float(np.median(phase_errors_ms[split_index - local_interval_count : split_index]))
        after_phase_error_ms = float(np.median(phase_errors_ms[split_index : split_index + local_interval_count]))
        phase_change_ms = abs(after_phase_error_ms - before_phase_error_ms)
        phase_residual_ms = max(before_phase_error_ms, after_phase_error_ms)
        score = (
            interval_change_ms / interval_change_threshold_ms
            + phase_change_ms / max(config.split_phase_change_threshold_ms, 1e-6)
            + phase_residual_ms / max(fit.beat_length_ms, 1e-6)
        )
        if (
            interval_change_ms + 1e-9 >= interval_change_threshold_ms
            or phase_change_ms >= config.split_phase_change_threshold_ms
            or phase_residual_ms >= config.split_phase_change_threshold_ms * 2.0
        ):
            candidates.append(_ChangeTimeCandidate(time_ms=float(peak_times_ms[split_index]), score=float(score)))

    return _merge_candidate_change_times(candidates, min_distance_ms=config.min_segment_duration_ms)


def _merge_candidate_change_times(
    candidates: Sequence[_ChangeTimeCandidate],
    *,
    min_distance_ms: float,
) -> list[_ChangeTimeCandidate]:
    merged_candidates: list[_ChangeTimeCandidate] = []
    for candidate in sorted(candidates, key=lambda value: value.time_ms):
        if not merged_candidates or candidate.time_ms - merged_candidates[-1].time_ms >= min_distance_ms:
            merged_candidates.append(candidate)
            continue
        if candidate.score > merged_candidates[-1].score:
            merged_candidates[-1] = candidate
    return merged_candidates


def _peak_phase_errors_ms(
    peak_times_ms: NDArray[np.float64],
    *,
    fit: _SegmentFit,
) -> NDArray[np.float64]:
    phase_ms = np.mod(peak_times_ms - fit.offset_ms, fit.beat_length_ms)
    return np.minimum(phase_ms, fit.beat_length_ms - phase_ms)


def _split_respects_minimum_duration(
    split_frame: int,
    *,
    fit: _SegmentFit,
    min_segment_frames: int,
) -> bool:
    return (
        split_frame - fit.start_frame >= min_segment_frames
        and fit.end_frame - split_frame >= min_segment_frames
    )


def _frame_step_ms(frame_times_ms: NDArray[np.float64]) -> float:
    if frame_times_ms.shape[0] < 2:
        return 20.0
    return float(np.median(np.diff(frame_times_ms)))


def _frame_rate_hz_from_times(frame_times_ms: NDArray[np.float64]) -> float:
    frame_step_ms = _frame_step_ms(frame_times_ms)
    if frame_step_ms <= 0.0:
        return 50.0
    return 1000.0 / frame_step_ms


def _weighted_score(segment_fits: Sequence[_SegmentFit]) -> float:
    total_frames = sum(fit.frame_count for fit in segment_fits)
    if total_frames <= 0:
        return -np.inf
    return float(sum(fit.score * fit.frame_count for fit in segment_fits) / total_frames)


def _timing_segments_from_fits(
    segment_fits: Sequence[_SegmentFit],
    frame_times_ms: NDArray[np.float64],
    *,
    config: GridFitterConfig,
) -> tuple[TimingSegment, ...]:
    return _merge_similar_timing_segments(
        _raw_timing_segments_from_fits(segment_fits, frame_times_ms),
        config=config,
    )


def _raw_timing_segments_from_fits(
    segment_fits: Sequence[_SegmentFit],
    frame_times_ms: NDArray[np.float64],
) -> tuple[TimingSegment, ...]:
    segments: list[TimingSegment] = []
    for index, fit in enumerate(segment_fits):
        offset_ms = fit.offset_ms
        if index > 0:
            boundary_time_ms = frame_times_ms[fit.start_frame]
            offset_ms = _nearest_congruent_offset(
                offset_ms,
                beat_length_ms=fit.beat_length_ms,
                target_time_ms=boundary_time_ms,
            )
            while offset_ms <= segments[-1].offset_ms:
                offset_ms += fit.beat_length_ms
        segments.append(TimingSegment(offset_ms=float(offset_ms), beat_length_ms=fit.beat_length_ms))
    return tuple(segments)


def _segment_fits_are_mergeable(
    left_fit: _SegmentFit,
    right_fit: _SegmentFit,
    *,
    frame_times_ms: NDArray[np.float64],
    config: GridFitterConfig,
    allow_alias: bool = False,
    allow_loose_similar: bool = False,
) -> bool:
    segments = _raw_timing_segments_from_fits((left_fit, right_fit), frame_times_ms)
    return len(segments) == 2 and _timing_segments_are_mergeable(
        segments[0],
        segments[1],
        config=config,
        allow_alias=allow_alias,
        allow_loose_similar=allow_loose_similar,
    )


def _merge_similar_timing_segments(
    segments: Sequence[TimingSegment],
    *,
    config: GridFitterConfig,
) -> tuple[TimingSegment, ...]:
    if not config.merge_similar_segments or not segments:
        return tuple(segments)

    merged_segments: list[TimingSegment] = [segments[0]]
    for segment in segments[1:]:
        previous_segment = merged_segments[-1]
        if _timing_segments_are_mergeable(
            previous_segment,
            segment,
            config=config,
        ):
            continue
        merged_segments.append(segment)
    return tuple(merged_segments)


def _timing_segments_are_mergeable(
    previous_segment: TimingSegment,
    segment: TimingSegment,
    *,
    config: GridFitterConfig,
    allow_alias: bool = False,
    allow_loose_similar: bool = False,
) -> bool:
    bpm_tolerance = max(
        config.merge_bpm_tolerance,
        min(previous_segment.local_bpm, segment.local_bpm) * config.merge_relative_bpm_tolerance,
    )
    if allow_loose_similar:
        bpm_tolerance = max(bpm_tolerance, config.merge_many_similar_bpm_tolerance)
    if abs(previous_segment.local_bpm - segment.local_bpm) <= bpm_tolerance:
        if allow_loose_similar:
            return True
        return (
            _phase_error_ms(
                segment.offset_ms,
                offset_ms=previous_segment.offset_ms,
                beat_length_ms=previous_segment.beat_length_ms,
            )
            <= config.merge_phase_tolerance_ms
        )
    if not allow_alias or not _bpms_are_alias_compatible(
        previous_segment.local_bpm,
        segment.local_bpm,
        tolerance_bpm=config.merge_alias_bpm_tolerance,
    ):
        return False
    alias_beat_length_ms = min(previous_segment.beat_length_ms, segment.beat_length_ms)
    return (
        _phase_error_ms(
            segment.offset_ms,
            offset_ms=previous_segment.offset_ms,
            beat_length_ms=alias_beat_length_ms,
        )
        <= config.merge_alias_phase_tolerance_ms
    )


def _bpms_are_alias_compatible(
    first_bpm: float,
    second_bpm: float,
    *,
    tolerance_bpm: float,
) -> bool:
    if first_bpm <= 0.0 or second_bpm <= 0.0:
        return False
    for multiplier in (0.25, 0.5, 2.0, 4.0):
        if abs(first_bpm * multiplier - second_bpm) <= tolerance_bpm:
            return True
    return False


def _phase_error_ms(
    time_ms: float,
    *,
    offset_ms: float,
    beat_length_ms: float,
) -> float:
    phase_ms = float(np.mod(time_ms - offset_ms, beat_length_ms))
    return float(min(phase_ms, beat_length_ms - phase_ms))


def _nearest_congruent_offset(
    offset_ms: float,
    *,
    beat_length_ms: float,
    target_time_ms: float,
) -> float:
    period_count = round((target_time_ms - offset_ms) / beat_length_ms)
    return float(offset_ms + period_count * beat_length_ms)


def _candidate_period_frame_bounds(
    frame_rate_hz: float,
    *,
    config: GridFitterConfig,
) -> tuple[int, int]:
    min_period_frames = int(np.ceil(frame_rate_hz * 60.0 / config.max_bpm))
    max_period_frames = int(np.floor(frame_rate_hz * 60.0 / config.min_bpm))
    if min_period_frames <= 0 or max_period_frames < min_period_frames:
        raise ValueError(
            "configured tempo range does not produce valid frame periods: "
            f"frame_rate_hz={frame_rate_hz}, min_bpm={config.min_bpm}, max_bpm={config.max_bpm}",
        )
    return min_period_frames, max_period_frames


def _candidate_bpms(
    centered_signal: NDArray[np.float64],
    *,
    frame_rate_hz: float,
    config: GridFitterConfig,
) -> NDArray[np.float64]:
    candidate_bpms = _autocorrelation_candidate_bpms(
        centered_signal,
        frame_rate_hz=frame_rate_hz,
        config=config,
    )
    if candidate_bpms.shape[0] == 0:
        candidate_bpms = np.arange(
            config.min_bpm,
            config.max_bpm + config.bpm_step * 0.5,
            config.bpm_step,
            dtype=np.float64,
        )
    return _limit_bpm_candidates_by_grid_count(candidate_bpms, config=config)


def _autocorrelation_candidate_bpms(
    centered_signal: NDArray[np.float64],
    *,
    frame_rate_hz: float,
    config: GridFitterConfig,
) -> NDArray[np.float64]:
    min_period_frames, max_period_frames = _candidate_period_frame_bounds(frame_rate_hz, config=config)
    max_period_frames = min(max_period_frames, centered_signal.shape[0] - 1)
    if max_period_frames < min_period_frames:
        return np.asarray([], dtype=np.float64)

    lag_scores: list[tuple[float, int]] = []
    for lag in range(min_period_frames, max_period_frames + 1):
        left = centered_signal[:-lag]
        right = centered_signal[lag:]
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        if denominator == 0.0:
            continue
        lag_scores.append((float(np.dot(left, right) / denominator), lag))
    if not lag_scores:
        return np.asarray([], dtype=np.float64)

    lag_scores.sort(reverse=True)
    selected_lags = [lag for _, lag in lag_scores[: config.autocorrelation_candidate_count]]

    exact_candidates: list[float] = []
    expanded_candidates: list[float] = []
    for lag in selected_lags:
        base_bpm = 60.0 * frame_rate_hz / lag
        for multiplier in (1.0, 2.0, 0.5, 3.0, 4.0, 1.0 / 3.0, 0.25):
            candidate_bpm = base_bpm * multiplier
            exact_candidates.append(candidate_bpm)
            expanded_candidates.extend(_expanded_bpm_window(candidate_bpm, config=config))

    return _ordered_unique_bpms([*exact_candidates, *expanded_candidates], config=config)


def _expanded_bpm_window(bpm: float, *, config: GridFitterConfig) -> list[float]:
    if not np.isfinite(bpm) or bpm < config.min_bpm or bpm > config.max_bpm:
        return []

    window_radius_bpm = max(config.bpm_search_window_min_bpm, bpm * config.bpm_search_window_ratio)
    start_bpm = max(config.min_bpm, _quantize_bpm(bpm - window_radius_bpm, config=config))
    end_bpm = min(config.max_bpm, _quantize_bpm(bpm + window_radius_bpm, config=config))
    return [
        float(value)
        for value in np.arange(
            start_bpm,
            end_bpm + config.bpm_step * 0.5,
            config.bpm_step,
            dtype=np.float64,
        )
    ]


def _quantize_bpm(bpm: float, *, config: GridFitterConfig) -> float:
    steps = round((bpm - config.min_bpm) / config.bpm_step)
    return float(config.min_bpm + steps * config.bpm_step)


def _ordered_unique_bpms(bpms: Sequence[float], *, config: GridFitterConfig) -> NDArray[np.float64]:
    seen: set[float] = set()
    ordered: list[float] = []
    for bpm in bpms:
        quantized_bpm = _quantize_bpm(float(bpm), config=config)
        if quantized_bpm < config.min_bpm or quantized_bpm > config.max_bpm:
            continue
        if quantized_bpm in seen:
            continue
        seen.add(quantized_bpm)
        ordered.append(quantized_bpm)
    return np.asarray(ordered, dtype=np.float64)


def _limit_bpm_candidates_by_grid_count(
    candidate_bpms: NDArray[np.float64],
    *,
    config: GridFitterConfig,
) -> NDArray[np.float64]:
    selected: list[float] = []
    grid_candidate_count = 0
    for bpm in candidate_bpms:
        beat_length_ms = 60000.0 / float(bpm)
        offset_count = max(1, int(np.ceil(beat_length_ms / config.offset_step_ms)))
        if selected and grid_candidate_count + offset_count > config.max_grid_candidates_per_segment:
            continue
        selected.append(float(bpm))
        grid_candidate_count += offset_count
    if not selected:
        return candidate_bpms[:1]
    return np.asarray(selected, dtype=np.float64)


def _candidate_offsets_ms(beat_length_ms: float, *, config: GridFitterConfig) -> NDArray[np.float64]:
    offsets = np.arange(0.0, beat_length_ms, config.offset_step_ms, dtype=np.float64)
    if offsets.shape[0] == 0:
        return np.asarray([0.0], dtype=np.float64)
    return offsets


def _best_grid_candidate(
    candidates: Sequence[_GridCandidate],
    *,
    downbeat_centered_signal: NDArray[np.float64] | None,
    downbeat_signal_norm: float,
    frame_times_ms: NDArray[np.float64],
    config: GridFitterConfig,
) -> tuple[float, float, float, float, float]:
    if not candidates:
        return -np.inf, -np.inf, config.min_bpm, 60000.0 / config.min_bpm, 0.0

    sorted_candidates = sorted(candidates, key=lambda candidate: candidate.score, reverse=True)
    best_candidate = sorted_candidates[0]
    if downbeat_centered_signal is None or downbeat_signal_norm == 0.0:
        return (
            float(best_candidate.score),
            -np.inf,
            float(best_candidate.bpm),
            float(best_candidate.beat_length_ms),
            float(best_candidate.offset_ms),
        )

    best_score = -np.inf
    best_downbeat_score = -np.inf
    best_bpm = best_candidate.bpm
    best_beat_length_ms = best_candidate.beat_length_ms
    best_offset_ms = best_candidate.offset_ms
    for candidate in sorted_candidates[: config.downbeat_refine_candidate_count]:
        downbeat_score, downbeat_offset_ms = _best_downbeat_grid_fit(
            downbeat_centered_signal,
            downbeat_signal_norm=downbeat_signal_norm,
            frame_times_ms=frame_times_ms,
            beat_length_ms=candidate.beat_length_ms,
            offset_ms=candidate.offset_ms,
            config=config,
        )
        if _grid_candidate_is_better(
            candidate.score,
            downbeat_score,
            best_score,
            best_downbeat_score,
            config=config,
        ):
            best_score = candidate.score
            best_downbeat_score = downbeat_score
            best_bpm = candidate.bpm
            best_beat_length_ms = candidate.beat_length_ms
            best_offset_ms = downbeat_offset_ms

    return (
        float(best_score),
        float(best_downbeat_score),
        float(best_bpm),
        float(best_beat_length_ms),
        float(best_offset_ms),
    )


def _best_bpm_fit(
    centered_signal: NDArray[np.float64],
    *,
    signal_norm: float,
    frame_times_ms: NDArray[np.float64],
    bpm: float,
    pulse_width_ms: float,
    downbeat_centered_signal: NDArray[np.float64] | None = None,
    downbeat_signal_norm: float = 0.0,
    config: GridFitterConfig,
) -> tuple[float, float, float]:
    if bpm < config.min_bpm or bpm > config.max_bpm:
        return -np.inf, 0.0, -np.inf

    beat_length_ms = 60000.0 / bpm
    candidates: list[_GridCandidate] = []
    for offset_ms in _candidate_offsets_ms(beat_length_ms, config=config):
        score = _score_grid(
            centered_signal,
            signal_norm=signal_norm,
            frame_times_ms=frame_times_ms,
            beat_length_ms=beat_length_ms,
            offset_ms=offset_ms,
            pulse_width_ms=pulse_width_ms,
        )
        candidates.append(
            _GridCandidate(
                score=float(score),
                bpm=float(bpm),
                beat_length_ms=float(beat_length_ms),
                offset_ms=float(offset_ms),
            )
        )
    best_score, best_downbeat_score, _, _, best_offset_ms = _best_grid_candidate(
        candidates,
        downbeat_centered_signal=downbeat_centered_signal,
        downbeat_signal_norm=downbeat_signal_norm,
        frame_times_ms=frame_times_ms,
        config=config,
    )
    return float(best_score), best_offset_ms, float(best_downbeat_score)


def _centered_signal_and_norm(
    signal: NDArray[np.float64] | None,
) -> tuple[NDArray[np.float64] | None, float]:
    if signal is None:
        return None, 0.0
    centered_signal = signal - float(np.mean(signal))
    signal_norm = float(np.linalg.norm(centered_signal))
    if signal_norm == 0.0:
        return None, 0.0
    return centered_signal, signal_norm


def _best_downbeat_grid_fit(
    downbeat_centered_signal: NDArray[np.float64] | None,
    *,
    downbeat_signal_norm: float,
    frame_times_ms: NDArray[np.float64],
    beat_length_ms: float,
    offset_ms: float,
    config: GridFitterConfig,
) -> tuple[float, float]:
    if downbeat_centered_signal is None or downbeat_signal_norm == 0.0:
        return -np.inf, float(offset_ms)

    downbeat_period_ms = beat_length_ms * config.downbeat_period_beats
    best_score = -np.inf
    best_offset_ms = float(offset_ms)
    for beat_index in range(config.downbeat_period_beats):
        candidate_offset_ms = float(offset_ms + beat_index * beat_length_ms)
        score = _score_grid(
            downbeat_centered_signal,
            signal_norm=downbeat_signal_norm,
            frame_times_ms=frame_times_ms,
            beat_length_ms=downbeat_period_ms,
            offset_ms=candidate_offset_ms,
            pulse_width_ms=config.pulse_width_ms,
        )
        if score > best_score:
            best_score = score
            best_offset_ms = candidate_offset_ms
    return float(best_score), best_offset_ms


def _grid_candidate_is_better(
    score: float,
    downbeat_score: float,
    best_score: float,
    best_downbeat_score: float,
    *,
    config: GridFitterConfig,
) -> bool:
    if score > best_score + config.downbeat_tie_score_margin:
        return True
    if score < best_score - config.downbeat_tie_score_margin:
        return False
    if np.isfinite(downbeat_score) or np.isfinite(best_downbeat_score):
        if downbeat_score > best_downbeat_score + config.downbeat_tie_score_margin:
            return True
        if downbeat_score < best_downbeat_score - config.downbeat_tie_score_margin:
            return False
    return score > best_score


def _downbeat_rejects_close_tempo_alias(
    raw_score: float,
    raw_downbeat_score: float,
    alias_score: float,
    alias_downbeat_score: float,
    *,
    config: GridFitterConfig,
) -> bool:
    if not np.isfinite(raw_downbeat_score) or not np.isfinite(alias_downbeat_score):
        return False
    if alias_score > raw_score + config.downbeat_tie_score_margin:
        return False
    return raw_downbeat_score > alias_downbeat_score + config.downbeat_tie_score_margin


def _score_grid(
    centered_signal: NDArray[np.float64],
    *,
    signal_norm: float,
    frame_times_ms: NDArray[np.float64],
    beat_length_ms: float,
    offset_ms: float,
    pulse_width_ms: float,
) -> float:
    template = _pulse_template(
        frame_times_ms,
        beat_length_ms=beat_length_ms,
        offset_ms=offset_ms,
        pulse_width_ms=pulse_width_ms,
    )
    centered_template = template - float(np.mean(template))
    template_norm = float(np.linalg.norm(centered_template))
    if template_norm == 0.0:
        return -np.inf
    return float(np.dot(centered_signal, centered_template) / (signal_norm * template_norm))


def _pulse_template(
    frame_times_ms: NDArray[np.float64],
    *,
    beat_length_ms: float,
    offset_ms: float,
    pulse_width_ms: float,
) -> NDArray[np.float64]:
    phase_ms = np.mod(frame_times_ms - offset_ms, beat_length_ms)
    distance_ms = np.minimum(phase_ms, beat_length_ms - phase_ms)
    return np.maximum(0.0, 1.0 - distance_ms / pulse_width_ms)
