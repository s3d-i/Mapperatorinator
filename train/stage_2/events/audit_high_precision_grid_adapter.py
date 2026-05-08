from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from array import array
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from train.stage1_oracle.osu.hitobjects import ManiaHitObjectKind, parse_mania_hit_objects
from train.stage_2.osu_core.timing import InvalidRedTimingError, MissingRedTimingError, require_red_timing_points
from train.stage_2.timing.grid_fitting import GridFitterConfig
from train.stage_2.timing.providers.oracle import fitted_timing_grid_from_red_points
from train.stage_2.timing.schema import FittedTimingGrid


DEFAULT_INDEX_PATH = Path(
    "train/artifacts/indexes/"
    "beatmap_index_4k_no_timing_anomalies_2to6_dense_local_bpm_norm_unique_le3.parquet"
)
DEFAULT_DATASET_ROOT = Path("mania-dataset")
DEFAULT_GRID_FITTER_CACHE_PATH = Path("train/artifacts/features/stage2_grid_fitter_cache.parquet")
DEFAULT_GRID_FITTER_CACHE_META_PATH = Path("train/artifacts/features/stage2_grid_fitter_cache_meta.json")
DEFAULT_ADAPTER_CACHE_PATH = Path("train/artifacts/features/stage2_high_precision_grid_adapter_cache.parquet")
DEFAULT_OUTPUT_JSON_PATH = Path("train/artifacts/reports/events/high_precision_grid_adapter_audit.json")
DEFAULT_OUTPUT_MD_PATH = Path("train/artifacts/reports/events/high_precision_grid_adapter_audit.md")
GRID_SOURCE_RED_TIMING_FALLBACK = "red_timing_fallback"
GRID_SOURCE_STAGE2_TIMING_MODULE = "stage2_timing_module_beatthis"

ACTION_HOLD_END = "HOLD_END"
ACTION_TAP = "TAP"
ACTION_HOLD_START = "HOLD_START"
ACTION_ORDER = {ACTION_HOLD_END: 0, ACTION_TAP: 1, ACTION_HOLD_START: 2}

D_UNIVERSE_2TO6 = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)
D_DIAGNOSTIC_TAIL = (96, 128, 192)
DEFAULT_K_MAX_VALUES = (16, 32, 64, 128)
DEFAULT_OFFSET_RANGES = ((-1, 1), (-2, 2), (-3, 3), (-5, 5))
DEFAULT_RESIDUAL_TOLERANCES_MS = (0.5, 0.75, 1.0, 1.5, 2.0)
DEFAULT_SAFE_KNOT_POLICIES = (
    "policy_0_soft",
    "policy_1_section_quantile",
    "policy_2_sparse_only",
    "policy_3_boundary_plus_sparse",
)
MONOTONICITY_STRICT = "strict"
MONOTONICITY_WEAK = "weak"
MONOTONICITY_RELAXED = "relaxed"
MONOTONICITY_MODES = (MONOTONICITY_STRICT, MONOTONICITY_WEAK, MONOTONICITY_RELAXED)

WINDOW_LENGTH_MS = 8000
EXACT_EPSILON_MS = 1.0e-6
SPECIAL_VOCAB_SIZE = 3
END_VOCAB_SIZE = 1
EVENT_VOCAB_SIZE = 4 * 3
WORST_EXAMPLE_LIMIT = 24
MAX_CANDIDATES_PER_TIME = 64
PRIMARY_CONFIG_NAME = "D_universe_offset_m2_p2_tol_0_5_policy_3_strict"
WEAK_CONFIG_NAME = "D_universe_offset_search_m20_p20_tol_0_5_policy_3_weak_delta3"


@dataclass(frozen=True)
class TimingSection:
    offset_ms: float
    beat_length_ms: float
    meter: int = 4
    section_id: int = 0
    end_ms: float | None = None

    @property
    def start_ms_hp(self) -> float:
        return float(self.offset_ms)

    @property
    def beat_zero_ms_hp(self) -> float:
        return float(self.offset_ms)

    @property
    def beat_length_ms_hp(self) -> float:
        return float(self.beat_length_ms)

    @property
    def bpm_hp(self) -> float:
        return 60000.0 / self.beat_length_ms_hp


@dataclass(frozen=True)
class FittedTimingSection:
    beatmap_key: str
    beatmap_path: str
    section_id: int
    start_ms_hp: float
    end_ms_hp: float | None
    beat_length_ms_hp: float
    bpm_hp: float
    meter: int
    beat_zero_ms_hp: float
    source: str
    confidence: float | None
    fit_residual_p50_ms: float | None
    fit_residual_p95_ms: float | None
    fit_residual_max_ms: float | None

    @property
    def offset_ms(self) -> float:
        return self.beat_zero_ms_hp

    @property
    def beat_length_ms(self) -> float:
        return self.beat_length_ms_hp

    @property
    def end_ms(self) -> float | None:
        return self.end_ms_hp


@dataclass(frozen=True)
class PrimitiveEvent:
    object_id: int
    lane: int
    action: str
    original_time_ms: int

    @property
    def sort_key(self) -> tuple[int, int, int, int]:
        return (self.original_time_ms, ACTION_ORDER[self.action], self.lane, self.object_id)


@dataclass(frozen=True)
class RawEvent:
    raw_time_ms: int
    lane: int
    action: str


@dataclass(frozen=True)
class TickCandidate:
    raw_time_ms: int
    offset_ms: int
    adapted_time_ms: float
    section_id: int
    divisor: int
    tick_index: int
    tick_time_ms_hp: float
    residual_ms: float
    canonical_rank: tuple[float, int, int, int, int]
    raw_section_id: int
    cross_section: bool = False


@dataclass(frozen=True)
class AssignedTick:
    raw_time_ms: int
    offset_ms: int
    adapted_time_ms: float
    section_id: int
    divisor: int
    tick_index: int
    tick_time_ms_hp: float
    residual_ms: float
    cross_section: bool = False


@dataclass(frozen=True)
class GapFeatures:
    left_time_ms: int
    right_time_ms: int
    raw_gap_ms: int
    normalized_gap_beats: float
    local_event_density_left: float
    local_event_density_right: float
    left_density_percentile: float
    right_density_percentile: float
    gap_percentile_within_section: float
    density_percentile: float
    section_id: int
    near_section_boundary: bool
    appears_dense: bool
    dense_pattern_edge: bool
    dense_core: bool

    @property
    def salience(self) -> float:
        boundary_bonus = 0.35 if self.near_section_boundary else 0.0
        beat_component = min(self.normalized_gap_beats, 4.0) / 4.0
        density_penalty = 0.5 * self.density_percentile
        return max(0.0, self.gap_percentile_within_section + beat_component + boundary_bonus - density_penalty)


@dataclass(frozen=True)
class CarryLNState:
    open_mask: int = 0
    open_age_ms_by_lane: tuple[int, int, int, int] = (0, 0, 0, 0)


@dataclass(frozen=True)
class AdapterConfig:
    name: str
    divisors: tuple[int, ...] = D_UNIVERSE_2TO6
    offset_min_ms: int = -2
    offset_max_ms: int = 2
    residual_tolerance_ms: float = 0.5
    safe_knot_policy: str = "policy_3_boundary_plus_sparse"
    strict_monotonicity: bool = True
    monotonicity_mode: str | None = None
    beam_width: int = 48
    max_offset_delta_ms: int = 3
    collapse_penalty: float = 4.0

    @property
    def offsets(self) -> tuple[int, ...]:
        return tuple(range(self.offset_min_ms, self.offset_max_ms + 1))

    @property
    def effective_monotonicity_mode(self) -> str:
        if self.monotonicity_mode is not None:
            mode = str(self.monotonicity_mode)
        else:
            mode = MONOTONICITY_STRICT if self.strict_monotonicity else MONOTONICITY_RELAXED
        if mode not in MONOTONICITY_MODES:
            raise ValueError(f"unknown monotonicity mode: {mode}")
        return mode


@dataclass(frozen=True)
class GridAdapterConfig:
    divisors: tuple[int, ...] = D_UNIVERSE_2TO6
    k_max: int = 64
    offset_min_ms: int = -2
    offset_max_ms: int = 2
    residual_tolerance_ms: float = 0.5
    safe_knot_policy: str = "policy_3_boundary_plus_sparse"
    strict_monotonicity: bool = True
    monotonicity_mode: str | None = None
    max_offset_delta_ms: int = 3

    def to_adapter_config(self) -> AdapterConfig:
        return AdapterConfig(
            name="test_grid_adapter_config",
            divisors=canonicalize_divisors(self.divisors),
            offset_min_ms=int(self.offset_min_ms),
            offset_max_ms=int(self.offset_max_ms),
            residual_tolerance_ms=float(self.residual_tolerance_ms),
            safe_knot_policy=self.safe_knot_policy,
            strict_monotonicity=bool(self.strict_monotonicity),
            monotonicity_mode=self.monotonicity_mode,
            max_offset_delta_ms=int(self.max_offset_delta_ms),
        )


@dataclass(frozen=True)
class AdapterAssignmentResult:
    ok: bool
    assignments: tuple[AssignedTick, ...] = ()
    failure_type: str | None = None
    failure_index: int | None = None
    knot_count: int = 0
    knots: tuple[dict[str, Any], ...] = ()
    collapsed_distinct_time_count: int = 0
    cross_section_assignment_count: int = 0
    no_candidate_count: int = 0
    monotonicity_failure_count: int = 0
    safe_knot_failure_count: int = 0
    offset_jump_failure_count: int = 0
    adapted_monotonicity_failure_count: int = 0
    sequence_identity_failure_count: int = 0
    transition_failure_counts: Mapping[str, int] = field(default_factory=dict)
    candidate_count: int = 0
    wanted_dense_knot_count: int = 0


@dataclass(frozen=True)
class SequenceIdentityResult:
    ok: bool
    mismatch_count: int = 0
    failure_examples: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class LegalityResult:
    ok: bool
    ln_invalid_count: int = 0
    same_lane_collision_count: int = 0
    hold_end_before_or_equal_start_count: int = 0
    failure_examples: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class GridToken:
    section_index: int
    divisor: int
    k: int
    start_ms: float
    end_ms: float


@dataclass(frozen=True)
class DecompositionResult:
    ok: bool
    tokens: tuple[GridToken, ...] = ()
    failure_type: str | None = None
    section_boundary_split_count: int = 0


@dataclass(frozen=True)
class RawTimeGridMatch:
    raw_time_ms: int
    grid_time_ms: float | None
    integer_ms_exact: bool
    section_index: int | None
    divisor: int | None
    tick_index: int | None
    residual_ms: float | None


@dataclass(frozen=True)
class AdaptedTimepoint:
    raw_time_ms: int
    grid_time_ms: float
    integer_ms_correction_ms: float
    section_index: int
    divisor: int
    tick_index: int
    lane_actions: tuple[str | None, str | None, str | None, str | None]


@dataclass(frozen=True)
class AdaptedEventResult:
    ok: bool
    timepoints: tuple[AdaptedTimepoint, ...] = ()
    tokens: tuple[GridToken, ...] = ()
    failure_type: str | None = None


@dataclass
class _BeamState:
    candidate: TickCandidate
    cost: float
    knot_count: int
    collapsed_count: int
    prev_index: int | None
    transition_knot: dict[str, Any] | None


@dataclass(frozen=True)
class _TransitionReject:
    reason: str
    wanted_dense_knot: bool = False


def parse_primitive_events(beatmap_path: str | Path) -> list[PrimitiveEvent]:
    hitobjects = parse_mania_hit_objects(beatmap_path, expected_key_count=4)
    events: list[PrimitiveEvent] = []
    for object_id, hitobject in enumerate(hitobjects):
        start_ms = _integer_osu_time(hitobject.start_time_ms)
        if start_ms is None:
            raise ValueError(f"{beatmap_path} has non-integer hitobject start time {hitobject.start_time_ms!r}")
        lane = int(hitobject.lane)
        if lane < 0 or lane >= 4:
            raise ValueError(f"{beatmap_path} has non-4K lane {lane!r}")

        if hitobject.kind == ManiaHitObjectKind.TAP:
            events.append(PrimitiveEvent(object_id=object_id, lane=lane, action=ACTION_TAP, original_time_ms=start_ms))
            continue

        end_ms = _integer_osu_time(hitobject.end_time_ms)
        if end_ms is None:
            raise ValueError(f"{beatmap_path} has non-integer LN end time {hitobject.end_time_ms!r}")
        events.append(
            PrimitiveEvent(object_id=object_id, lane=lane, action=ACTION_HOLD_START, original_time_ms=start_ms)
        )
        events.append(PrimitiveEvent(object_id=object_id, lane=lane, action=ACTION_HOLD_END, original_time_ms=end_ms))

    events.sort(key=lambda event: event.sort_key)
    return events


def sections_from_fitted_grid(
    *,
    beatmap_key: str,
    beatmap_path: str | Path,
    grid: FittedTimingGrid,
    source: str,
    confidence: float | None = None,
    fit_residual_p50_ms: float | None = None,
    fit_residual_p95_ms: float | None = None,
    fit_residual_max_ms: float | None = None,
) -> tuple[FittedTimingSection, ...]:
    sections: list[FittedTimingSection] = []
    segments = tuple(grid.segments)
    for index, segment in enumerate(segments):
        end_ms = None if index + 1 >= len(segments) else float(segments[index + 1].offset_ms)
        beat_length = float(segment.beat_length_ms)
        sections.append(
            FittedTimingSection(
                beatmap_key=beatmap_key,
                beatmap_path=Path(beatmap_path).as_posix(),
                section_id=index,
                start_ms_hp=float(segment.offset_ms),
                end_ms_hp=end_ms,
                beat_length_ms_hp=beat_length,
                bpm_hp=60000.0 / beat_length,
                meter=int(getattr(segment, "meter", 4)),
                beat_zero_ms_hp=float(segment.offset_ms),
                source=source,
                confidence=confidence,
                fit_residual_p50_ms=fit_residual_p50_ms,
                fit_residual_p95_ms=fit_residual_p95_ms,
                fit_residual_max_ms=fit_residual_max_ms,
            )
        )
    return tuple(sections)


@dataclass(frozen=True)
class _CachedTimingModuleGrid:
    grid: FittedTimingGrid
    score: float
    prediction_seconds: float
    fit_seconds: float
    frame_count: int
    candidate_count: int
    alias_candidate_count: int


class _Stage2TimingModuleGridCache:
    def __init__(
        self,
        *,
        checkpoint_path: str,
        device: str,
        float16: bool,
        fitter_config: GridFitterConfig = GridFitterConfig(),
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.float16 = bool(float16)
        self.fitter_config = fitter_config
        self._provider: Any | None = None
        self._fitter: Any | None = None
        self._cache: dict[str, _CachedTimingModuleGrid] = {}
        self._hits = 0
        self._misses = 0
        self._prediction_seconds: list[float] = []
        self._fit_seconds: list[float] = []
        self._scores: list[float] = []
        self._frame_counts: list[int] = []
        self._candidate_counts: list[int] = []
        self._alias_candidate_counts: list[int] = []

    def grid_for_audio(self, audio_path: Path) -> _CachedTimingModuleGrid:
        key = audio_path.as_posix()
        cached = self._cache.get(key)
        if cached is not None:
            self._hits += 1
            return cached

        self._misses += 1
        prediction_start = time.perf_counter()
        prediction = self._get_provider().predict_file(audio_path)
        prediction_seconds = time.perf_counter() - prediction_start
        fit_start = time.perf_counter()
        fit_result = self._get_fitter().fit(prediction)
        fit_seconds = time.perf_counter() - fit_start
        cached = _CachedTimingModuleGrid(
            grid=fit_result.grid,
            score=float(fit_result.score),
            prediction_seconds=float(prediction_seconds),
            fit_seconds=float(fit_seconds),
            frame_count=int(prediction.frame_count),
            candidate_count=int(fit_result.diagnostics.candidate_count),
            alias_candidate_count=int(fit_result.diagnostics.alias_candidate_count),
        )
        self._cache[key] = cached
        self._prediction_seconds.append(float(prediction_seconds))
        self._fit_seconds.append(float(fit_seconds))
        self._scores.append(float(fit_result.score))
        self._frame_counts.append(int(prediction.frame_count))
        self._candidate_counts.append(int(fit_result.diagnostics.candidate_count))
        self._alias_candidate_counts.append(int(fit_result.diagnostics.alias_candidate_count))
        return cached

    def stats(self) -> dict[str, Any]:
        return {
            "provider": "beat-this",
            "checkpoint_path": self.checkpoint_path,
            "device": self.device,
            "float16": self.float16,
            "model_cached_in_process": True,
            "grid_cache_key": "shard/audio_path",
            "unique_audio_fit_count": len(self._cache),
            "audio_cache_hits": self._hits,
            "audio_cache_misses": self._misses,
            "prediction_seconds": _number_list_stats(self._prediction_seconds),
            "fit_seconds": _number_list_stats(self._fit_seconds),
            "fit_score": _number_list_stats(self._scores),
            "frame_count": _number_list_stats(self._frame_counts),
            "candidate_count": _number_list_stats(self._candidate_counts),
            "alias_candidate_count": _number_list_stats(self._alias_candidate_counts),
        }

    def _get_provider(self) -> Any:
        if self._provider is None:
            from train.stage_2.timing.providers.beatthis import BeatThisTimingProvider

            self._provider = BeatThisTimingProvider(
                checkpoint_path=self.checkpoint_path,
                device=self.device,
                float16=self.float16,
            )
        return self._provider

    def _get_fitter(self) -> Any:
        if self._fitter is None:
            from train.stage_2.timing.grid_fitting import GridFitter

            self._fitter = GridFitter(self.fitter_config)
        return self._fitter


def _grid_for_audit_row(
    *,
    beatmap_path: Path,
    audio_path: Path,
    grid_source: str,
    timing_module_cache: _Stage2TimingModuleGridCache | None,
) -> tuple[FittedTimingGrid, str, float | None]:
    if grid_source == GRID_SOURCE_RED_TIMING_FALLBACK:
        red_points = require_red_timing_points(beatmap_path)
        return fitted_timing_grid_from_red_points(red_points), GRID_SOURCE_RED_TIMING_FALLBACK, None
    if grid_source == GRID_SOURCE_STAGE2_TIMING_MODULE:
        if timing_module_cache is None:
            raise RuntimeError("timing module cache is required for stage2 timing module grid source")
        cached = timing_module_cache.grid_for_audio(audio_path)
        return cached.grid, GRID_SOURCE_STAGE2_TIMING_MODULE, cached.score
    raise ValueError(f"unknown grid source: {grid_source}")


def assign_primitive_events(
    events: Sequence[PrimitiveEvent],
    sections: Sequence[FittedTimingSection | TimingSection],
    config: AdapterConfig,
) -> tuple[AdapterAssignmentResult, LegalityResult]:
    unique_times = sorted({event.original_time_ms for event in events})
    assignment = assign_monotone_adapter(unique_times, sections, config)
    if not assignment.ok:
        return assignment, LegalityResult(ok=False)
    legality = validate_lane_ln_legality(events, assignment.assignments)
    if not legality.ok:
        return (
            AdapterAssignmentResult(
                ok=False,
                assignments=assignment.assignments,
                failure_type="lane_ln_legality",
                knot_count=assignment.knot_count,
                knots=assignment.knots,
                collapsed_distinct_time_count=assignment.collapsed_distinct_time_count,
                cross_section_assignment_count=assignment.cross_section_assignment_count,
                candidate_count=assignment.candidate_count,
            ),
            legality,
        )
    sequence_identity = validate_sequence_identity(events, assignment.assignments)
    if not sequence_identity.ok:
        return (
            AdapterAssignmentResult(
                ok=False,
                assignments=assignment.assignments,
                failure_type="sequence_identity",
                knot_count=assignment.knot_count,
                knots=assignment.knots,
                collapsed_distinct_time_count=assignment.collapsed_distinct_time_count,
                cross_section_assignment_count=assignment.cross_section_assignment_count,
                candidate_count=assignment.candidate_count,
                sequence_identity_failure_count=sequence_identity.mismatch_count,
            ),
            legality,
        )
    return assignment, legality


def match_raw_time_to_grid(
    raw_time_ms: int,
    sections: Sequence[FittedTimingSection | TimingSection],
    *,
    divisors: Sequence[int],
    residual_tolerance_ms: float = 0.5,
) -> RawTimeGridMatch:
    candidates = generate_tick_candidates(
        int(raw_time_ms),
        validate_timing_sections(sections),
        canonicalize_divisors(divisors),
        offsets=(0,),
        residual_tolerance_ms=residual_tolerance_ms,
    )
    if candidates:
        candidate = candidates[0]
        return RawTimeGridMatch(
            raw_time_ms=int(raw_time_ms),
            grid_time_ms=candidate.tick_time_ms_hp,
            integer_ms_exact=abs(candidate.residual_ms) <= residual_tolerance_ms + EXACT_EPSILON_MS,
            section_index=candidate.section_id,
            divisor=candidate.divisor,
            tick_index=candidate.tick_index,
            residual_ms=candidate.residual_ms,
        )
    return RawTimeGridMatch(
        raw_time_ms=int(raw_time_ms),
        grid_time_ms=None,
        integer_ms_exact=False,
        section_index=None,
        divisor=None,
        tick_index=None,
        residual_ms=None,
    )


def adapt_events_to_high_precision_grid(
    events: Sequence[RawEvent],
    sections: Sequence[FittedTimingSection | TimingSection],
    *,
    config: GridAdapterConfig = GridAdapterConfig(),
) -> AdaptedEventResult:
    sections = validate_timing_sections(sections)
    primitive_events = tuple(
        PrimitiveEvent(
            object_id=index,
            lane=int(event.lane),
            action=str(event.action),
            original_time_ms=int(event.raw_time_ms),
        )
        for index, event in enumerate(events)
    )
    assignment, legality = assign_primitive_events(primitive_events, sections, config.to_adapter_config())
    if not legality.ok:
        _raise_legality_error(legality)
    if not assignment.ok:
        return AdaptedEventResult(ok=False, failure_type=assignment.failure_type)
    return AdaptedEventResult(
        ok=True,
        timepoints=_adapted_timepoints(primitive_events, assignment.assignments),
    )


def validate_timing_sections(
    sections: Sequence[FittedTimingSection | TimingSection],
    *,
    min_section_span_ms: float = 0.0,
) -> tuple[FittedTimingSection | TimingSection, ...]:
    normalized = tuple(sections)
    previous_start: float | None = None
    for section in normalized:
        start = _section_start(section)
        if previous_start is not None:
            delta = start - previous_start
            if delta <= 0.0:
                raise ValueError("timing sections must be strictly increasing")
            if min_section_span_ms > 0.0 and delta < min_section_span_ms:
                raise ValueError("timing section boundary creates a dense staircase")
        previous_start = start
    return normalized


def canonicalize_divisors(divisors: Sequence[int]) -> tuple[int, ...]:
    cleaned: set[int] = set()
    for value in divisors:
        if isinstance(value, bool):
            raise ValueError("divisors must be positive integer values")
        if isinstance(value, int):
            divisor = value
        elif isinstance(value, float) and value.is_integer():
            divisor = int(value)
        else:
            raise ValueError("divisors must be positive integer values")
        if divisor <= 0:
            raise ValueError("divisors must be positive integer values")
        cleaned.add(divisor)
    if not cleaned:
        raise ValueError("divisors must contain at least one positive integer")
    return tuple(sorted(cleaned))


def decompose_time_shift(
    start_ms: float,
    end_ms: float,
    sections: Sequence[FittedTimingSection | TimingSection],
    *,
    config: GridAdapterConfig,
) -> DecompositionResult:
    return decompose_interval_to_grid_tokens(
        start_ms,
        end_ms,
        validate_timing_sections(sections),
        canonicalize_divisors(config.divisors),
        k_max=int(config.k_max),
    )


def assign_monotone_adapter(
    unique_raw_times_ms: Sequence[int],
    sections: Sequence[FittedTimingSection | TimingSection],
    config: AdapterConfig,
) -> AdapterAssignmentResult:
    unique_times = tuple(sorted({int(value) for value in unique_raw_times_ms}))
    if not unique_times:
        return AdapterAssignmentResult(ok=True)
    sections = tuple(sections)
    if not sections:
        return AdapterAssignmentResult(ok=False, failure_type="no_timing_sections", no_candidate_count=len(unique_times))

    gap_features = compute_gap_features(unique_times, sections)
    zero_result = _assign_zero_offset_fast_path(unique_times, sections, config)
    if zero_result.ok:
        return zero_result

    candidates_by_time: list[tuple[TickCandidate, ...]] = []
    candidate_count = 0
    for time_ms in unique_times:
        candidates = generate_tick_candidates(
            time_ms,
            sections,
            config.divisors,
            offsets=config.offsets,
            residual_tolerance_ms=config.residual_tolerance_ms,
        )
        candidates = tuple(sorted(candidates, key=lambda candidate: candidate.canonical_rank)[:MAX_CANDIDATES_PER_TIME])
        candidate_count += len(candidates)
        if not candidates:
            return AdapterAssignmentResult(
                ok=False,
                failure_type="no_candidate",
                failure_index=len(candidates_by_time),
                no_candidate_count=1,
                candidate_count=candidate_count,
            )
        candidates_by_time.append(candidates)

    beams: list[list[_BeamState]] = []
    first_beam = [
        _BeamState(
            candidate=candidate,
            cost=_candidate_cost(candidate),
            knot_count=0,
            collapsed_count=0,
            prev_index=None,
            transition_knot=None,
        )
        for candidate in candidates_by_time[0]
    ]
    first_beam.sort(key=lambda state: (state.cost, state.candidate.canonical_rank))
    beams.append(first_beam[: config.beam_width])

    transition_failures: Counter[str] = Counter()
    wanted_dense_knot_count = 0
    for index in range(1, len(unique_times)):
        gap = gap_features[index - 1]
        current_states: list[_BeamState] = []
        best_by_key: dict[tuple[int, int, int, int], _BeamState] = {}
        for prev_index, prev in enumerate(beams[-1]):
            for candidate in candidates_by_time[index]:
                transition = _transition_cost_and_knot(prev.candidate, candidate, gap, config)
                if isinstance(transition, _TransitionReject):
                    transition_failures[transition.reason] += 1
                    wanted_dense_knot_count += int(transition.wanted_dense_knot)
                    continue
                transition_cost, knot, collapsed = transition
                cost = prev.cost + _candidate_cost(candidate) + transition_cost
                state = _BeamState(
                    candidate=candidate,
                    cost=cost,
                    knot_count=prev.knot_count + (1 if knot is not None else 0),
                    collapsed_count=prev.collapsed_count + (1 if collapsed else 0),
                    prev_index=prev_index,
                    transition_knot=knot,
                )
                key = (candidate.offset_ms, candidate.section_id, candidate.tick_index, candidate.divisor)
                old = best_by_key.get(key)
                if old is None or (state.cost, state.knot_count) < (old.cost, old.knot_count):
                    best_by_key[key] = state

        current_states = sorted(best_by_key.values(), key=lambda state: (state.cost, state.knot_count))[
            : config.beam_width
        ]
        if not current_states:
            failure_type = _dominant_transition_failure_type(transition_failures)
            return AdapterAssignmentResult(
                ok=False,
                failure_type=failure_type,
                failure_index=index,
                monotonicity_failure_count=int(transition_failures["monotonicity"]),
                safe_knot_failure_count=int(transition_failures["safe_knot"]),
                offset_jump_failure_count=int(transition_failures["offset_jump"]),
                adapted_monotonicity_failure_count=int(transition_failures["adapted_monotonicity"]),
                transition_failure_counts=dict(transition_failures),
                candidate_count=candidate_count,
                wanted_dense_knot_count=wanted_dense_knot_count,
            )
        beams.append(current_states)

    best_final_index, best_final = min(
        enumerate(beams[-1]),
        key=lambda item: (item[1].cost, item[1].knot_count, item[1].candidate.canonical_rank),
    )
    raw_states: list[_BeamState] = []
    state_index: int | None = best_final_index
    for beam_index in range(len(beams) - 1, -1, -1):
        assert state_index is not None
        state = beams[beam_index][state_index]
        raw_states.append(state)
        state_index = state.prev_index
    raw_states.reverse()

    assignments = tuple(_assigned_tick_from_candidate(state.candidate) for state in raw_states)
    knots = tuple(state.transition_knot for state in raw_states[1:] if state.transition_knot is not None)
    return AdapterAssignmentResult(
        ok=True,
        assignments=assignments,
        knot_count=len(knots),
        knots=knots,
        collapsed_distinct_time_count=best_final.collapsed_count,
        cross_section_assignment_count=sum(1 for assignment in assignments if assignment.cross_section),
        candidate_count=candidate_count,
        wanted_dense_knot_count=wanted_dense_knot_count,
    )


def generate_tick_candidates(
    raw_time_ms: int,
    sections: Sequence[FittedTimingSection | TimingSection],
    divisors: Sequence[int],
    *,
    offsets: Sequence[int],
    residual_tolerance_ms: float,
) -> tuple[TickCandidate, ...]:
    sections = tuple(sections)
    offsets_by_section = [_section_start(section) for section in sections]
    raw_section_id = _section_index_at_offsets(offsets_by_section, raw_time_ms)
    if raw_section_id < 0:
        raw_section_id = 0

    candidates: list[TickCandidate] = []
    seen: set[tuple[int, int, int, int]] = set()
    for offset in offsets:
        adapted_time = float(raw_time_ms + offset)
        adapted_section_id = _section_index_at_offsets(offsets_by_section, adapted_time)
        if adapted_section_id < 0:
            adapted_section_id = 0
        candidate_section_ids = {raw_section_id, adapted_section_id, raw_section_id - 1, raw_section_id + 1}
        for section_id in sorted(candidate_section_ids):
            if section_id < 0 or section_id >= len(sections):
                continue
            section = sections[section_id]
            cross_section = section_id != raw_section_id
            if cross_section and not _near_section_boundary(
                raw_time_ms,
                adapted_time,
                sections,
                max(abs(offset) + residual_tolerance_ms + 1.0, 5.0),
            ):
                continue
            start = _section_start(section)
            if section_id == 0 and min(float(raw_time_ms), adapted_time) < start:
                start = -math.inf
            end = _section_end(section)
            if end is None and section_id + 1 < len(sections):
                end = _section_start(sections[section_id + 1])
            for divisor in divisors:
                step_ms = _section_beat_length(section) / int(divisor)
                tick_index = int(round((adapted_time - _section_beat_zero(section)) / step_ms))
                tick_time = _section_beat_zero(section) + tick_index * step_ms
                if tick_time + EXACT_EPSILON_MS < start:
                    continue
                if end is not None and tick_time >= end - EXACT_EPSILON_MS:
                    continue
                residual = adapted_time - tick_time
                if abs(residual) > residual_tolerance_ms + EXACT_EPSILON_MS:
                    continue
                key = (offset, section_id, int(divisor), tick_index)
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(
                    TickCandidate(
                        raw_time_ms=int(raw_time_ms),
                        offset_ms=int(offset),
                        adapted_time_ms=float(adapted_time),
                        section_id=int(section_id),
                        divisor=int(divisor),
                        tick_index=int(tick_index),
                        tick_time_ms_hp=float(tick_time),
                        residual_ms=float(residual),
                        canonical_rank=(
                            round(abs(float(residual)), 9),
                            int(divisor),
                            abs(int(offset)),
                            int(section_id),
                            int(tick_index),
                        ),
                        raw_section_id=int(raw_section_id),
                        cross_section=bool(cross_section),
                    )
                )
    return tuple(sorted(candidates, key=lambda candidate: candidate.canonical_rank))


def compute_gap_features(
    unique_raw_times_ms: Sequence[int],
    sections: Sequence[FittedTimingSection | TimingSection],
    *,
    density_window_beats: float = 4.0,
) -> tuple[GapFeatures, ...]:
    unique_times = tuple(sorted({int(value) for value in unique_raw_times_ms}))
    if len(unique_times) < 2:
        return ()
    offsets = [_section_start(section) for section in sections]
    gap_records: list[dict[str, Any]] = []
    gaps_by_section: dict[int, list[int]] = defaultdict(list)
    density_values: list[float] = []
    left_density_values: list[float] = []
    right_density_values: list[float] = []

    for left, right in zip(unique_times[:-1], unique_times[1:], strict=True):
        midpoint = (left + right) / 2.0
        section_id = max(0, _section_index_at_offsets(offsets, midpoint))
        section = sections[section_id]
        beat_length = _section_beat_length(section)
        window_ms = max(beat_length * density_window_beats, 1.0)
        left_window_start = left - window_ms
        right_window_end = right + window_ms
        left_density = _count_times_in_range(unique_times, left_window_start, left) / density_window_beats
        right_density = _count_times_in_range(unique_times, right, right_window_end) / density_window_beats
        raw_gap = int(right - left)
        density = left_density + right_density
        gap_records.append(
            {
                "left": left,
                "right": right,
                "raw_gap": raw_gap,
                "section_id": section_id,
                "beat_length": beat_length,
                "left_density": left_density,
                "right_density": right_density,
                "density": density,
                "near_boundary": _gap_crosses_or_touches_section_boundary(left, right, sections),
            }
        )
        gaps_by_section[section_id].append(raw_gap)
        density_values.append(density)
        left_density_values.append(left_density)
        right_density_values.append(right_density)

    density_sorted = sorted(density_values)
    left_density_sorted = sorted(left_density_values)
    right_density_sorted = sorted(right_density_values)
    density_p75 = _percentile(density_values, 75.0)
    features: list[GapFeatures] = []
    for record in gap_records:
        section_gaps = sorted(gaps_by_section[int(record["section_id"])])
        gap_percentile = _rank_percentile(section_gaps, int(record["raw_gap"]))
        density_percentile = _rank_percentile(density_sorted, float(record["density"]))
        left_density_percentile = _rank_percentile(left_density_sorted, float(record["left_density"]))
        right_density_percentile = _rank_percentile(right_density_sorted, float(record["right_density"]))
        appears_dense = density_percentile >= 0.75 and float(record["density"]) >= density_p75
        left_dense = left_density_percentile >= 0.75
        right_dense = right_density_percentile >= 0.75
        left_edge = left_dense and right_density_percentile <= 0.55
        right_edge = right_dense and left_density_percentile <= 0.55
        gap_salient_for_edge = gap_percentile >= 0.60
        dense_pattern_edge = gap_salient_for_edge and (left_edge or right_edge)
        dense_core = left_dense and right_dense and gap_percentile < 0.85
        features.append(
            GapFeatures(
                left_time_ms=int(record["left"]),
                right_time_ms=int(record["right"]),
                raw_gap_ms=int(record["raw_gap"]),
                normalized_gap_beats=float(record["raw_gap"]) / float(record["beat_length"]),
                local_event_density_left=float(record["left_density"]),
                local_event_density_right=float(record["right_density"]),
                left_density_percentile=float(left_density_percentile),
                right_density_percentile=float(right_density_percentile),
                gap_percentile_within_section=float(gap_percentile),
                density_percentile=float(density_percentile),
                section_id=int(record["section_id"]),
                near_section_boundary=bool(record["near_boundary"]),
                appears_dense=bool(appears_dense),
                dense_pattern_edge=bool(dense_pattern_edge),
                dense_core=bool(dense_core),
            )
        )
    return tuple(features)


def validate_lane_ln_legality(
    events: Sequence[PrimitiveEvent],
    assignments: Sequence[AssignedTick],
) -> LegalityResult:
    assignment_by_time = {assignment.raw_time_ms: assignment for assignment in assignments}
    open_start_by_lane: dict[int, float] = {}
    same_tick_taps: set[tuple[float, int]] = set()
    ln_invalid = 0
    same_lane_collision = 0
    hold_end_before_or_equal = 0
    examples: list[dict[str, Any]] = []

    for event in sorted(events, key=lambda item: item.sort_key):
        assignment = assignment_by_time.get(event.original_time_ms)
        if assignment is None:
            ln_invalid += 1
            _append_example(examples, event, "missing_assignment")
            continue
        tick_time = assignment.tick_time_ms_hp
        lane = int(event.lane)
        if event.action == ACTION_HOLD_END:
            if lane not in open_start_by_lane:
                ln_invalid += 1
                _append_example(examples, event, "hold_end_without_open")
                continue
            start_time = open_start_by_lane.pop(lane)
            if tick_time <= start_time + EXACT_EPSILON_MS:
                hold_end_before_or_equal += 1
                _append_example(examples, event, "hold_end_before_or_equal_start")
            continue

        if event.action == ACTION_TAP:
            if lane in open_start_by_lane:
                same_lane_collision += 1
                _append_example(examples, event, "tap_while_lane_open")
            key = (round(tick_time, 9), lane)
            if key in same_tick_taps:
                same_lane_collision += 1
                _append_example(examples, event, "duplicate_same_lane_tap")
            same_tick_taps.add(key)
            continue

        if event.action == ACTION_HOLD_START:
            if lane in open_start_by_lane:
                same_lane_collision += 1
                _append_example(examples, event, "hold_start_while_lane_open")
            else:
                open_start_by_lane[lane] = tick_time
            continue

        ln_invalid += 1
        _append_example(examples, event, "unknown_action")

    if open_start_by_lane:
        ln_invalid += len(open_start_by_lane)
        for lane, tick_time in sorted(open_start_by_lane.items()):
            if len(examples) < WORST_EXAMPLE_LIMIT:
                examples.append({"lane": lane, "tick_time_ms_hp": tick_time, "reason": "unclosed_hold"})

    return LegalityResult(
        ok=(ln_invalid == 0 and same_lane_collision == 0 and hold_end_before_or_equal == 0),
        ln_invalid_count=ln_invalid,
        same_lane_collision_count=same_lane_collision,
        hold_end_before_or_equal_start_count=hold_end_before_or_equal,
        failure_examples=tuple(examples),
    )


def validate_sequence_identity(
    events: Sequence[PrimitiveEvent],
    assignments: Sequence[AssignedTick],
) -> SequenceIdentityResult:
    assignment_by_time = {assignment.raw_time_ms: assignment for assignment in assignments}
    original_sequence = tuple(
        (event.object_id, event.lane, event.action)
        for event in sorted(events, key=lambda item: item.sort_key)
    )
    mapped_events: list[tuple[tuple[float, int, int, int], PrimitiveEvent]] = []
    for event in events:
        assignment = assignment_by_time.get(event.original_time_ms)
        if assignment is None:
            return SequenceIdentityResult(
                ok=False,
                mismatch_count=1,
                failure_examples=(
                    {
                        "reason": "missing_assignment",
                        "raw_time_ms": event.original_time_ms,
                        "object_id": event.object_id,
                        "lane": event.lane,
                        "action": event.action,
                    },
                ),
            )
        mapped_events.append(
            (
                (
                    assignment.tick_time_ms_hp,
                    ACTION_ORDER[event.action],
                    event.lane,
                    event.object_id,
                ),
                event,
            )
        )

    mapped_sequence = tuple(
        (event.object_id, event.lane, event.action)
        for _, event in sorted(mapped_events, key=lambda item: item[0])
    )
    if original_sequence == mapped_sequence:
        return SequenceIdentityResult(ok=True)

    mismatch_count = 0
    examples: list[dict[str, Any]] = []
    for index, (original, mapped) in enumerate(zip(original_sequence, mapped_sequence, strict=False)):
        if original == mapped:
            continue
        mismatch_count += 1
        if len(examples) < WORST_EXAMPLE_LIMIT:
            examples.append(
                {
                    "index": index,
                    "original": {"object_id": original[0], "lane": original[1], "action": original[2]},
                    "mapped": {"object_id": mapped[0], "lane": mapped[1], "action": mapped[2]},
                }
            )
    mismatch_count += abs(len(original_sequence) - len(mapped_sequence))
    return SequenceIdentityResult(ok=False, mismatch_count=mismatch_count, failure_examples=tuple(examples))


def canonical_divisor_for_tick(
    time_ms: float,
    sections: Sequence[FittedTimingSection | TimingSection],
    divisors: Sequence[int],
    *,
    exact_epsilon_ms: float = EXACT_EPSILON_MS,
) -> tuple[int, int, int] | None:
    section_id = _section_index_at_offsets([_section_start(section) for section in sections], time_ms)
    if section_id < 0:
        section_id = 0
    section = sections[section_id]
    for divisor in sorted({int(value) for value in divisors}):
        step_ms = _section_beat_length(section) / divisor
        tick_index = int(round((time_ms - _section_beat_zero(section)) / step_ms))
        tick_time = _section_beat_zero(section) + tick_index * step_ms
        if abs(time_ms - tick_time) <= exact_epsilon_ms:
            return section_id, divisor, tick_index
    return None


def decompose_interval_to_grid_tokens(
    start_ms: float,
    end_ms: float,
    sections: Sequence[FittedTimingSection | TimingSection],
    divisors: Sequence[int],
    *,
    k_max: int,
    exact_epsilon_ms: float = EXACT_EPSILON_MS,
) -> DecompositionResult:
    if end_ms < start_ms - exact_epsilon_ms:
        return DecompositionResult(ok=False, failure_type="negative_delta")
    if abs(end_ms - start_ms) <= exact_epsilon_ms:
        return DecompositionResult(ok=True)
    sections = tuple(sections)
    if not sections:
        return DecompositionResult(ok=False, failure_type="no_timing_sections")

    split_points = [float(start_ms)]
    split_points.extend(_section_start(section) for section in sections if start_ms < _section_start(section) < end_ms)
    split_points.append(float(end_ms))
    tokens: list[GridToken] = []
    section_splits = max(0, len(split_points) - 2)
    for segment_start, segment_end in zip(split_points[:-1], split_points[1:], strict=True):
        section_id = _section_index_at_offsets([_section_start(section) for section in sections], segment_start)
        if section_id < 0:
            section_id = 0
        result = _decompose_single_section_interval(
            segment_start,
            segment_end,
            section_id,
            sections[section_id],
            divisors,
            k_max=k_max,
            exact_epsilon_ms=exact_epsilon_ms,
        )
        if not result.ok:
            return DecompositionResult(
                ok=False,
                failure_type=result.failure_type,
                section_boundary_split_count=section_splits,
            )
        tokens.extend(result.tokens)
    return DecompositionResult(ok=True, tokens=tuple(tokens), section_boundary_split_count=section_splits)


def derive_divisor_tiers(
    divisor_counts: Mapping[int, int],
    *,
    ordered_divisors: Sequence[int] = (*D_UNIVERSE_2TO6, *D_DIAGNOSTIC_TAIL),
) -> dict[str, Any]:
    total = sum(int(value) for value in divisor_counts.values())
    ordered = [int(divisor) for divisor in ordered_divisors if int(divisor) in divisor_counts]
    cumulative_rows: list[dict[str, Any]] = []
    running = 0
    previous = 0
    for divisor in ordered:
        count = int(divisor_counts.get(divisor, 0))
        running += count
        cumulative = _rate(running, total)
        cumulative_rows.append(
            {
                "divisor": divisor,
                "count": count,
                "frequency": _rate(count, total),
                "cumulative_count": running,
                "cumulative_rate": cumulative,
                "marginal_gain": _rate(running - previous, total),
            }
        )
        previous = running

    return {
        "selected_divisor_frequency": {str(key): int(value) for key, value in sorted(divisor_counts.items())},
        "cumulative_coverage": cumulative_rows,
        "D_core_2to6": _smallest_prefix_for_threshold(cumulative_rows, 0.99),
        "D_main_2to6": _smallest_prefix_for_threshold(cumulative_rows, 0.999),
        "D_safe_2to6": _smallest_prefix_for_threshold(cumulative_rows, 0.9999),
        "diagnostic_tail_used": any(int(divisor_counts.get(divisor, 0)) > 0 for divisor in D_DIAGNOSTIC_TAIL),
    }


def audit_high_precision_grid_adapter(
    *,
    index_path: str | Path = DEFAULT_INDEX_PATH,
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    grid_source: str = GRID_SOURCE_RED_TIMING_FALLBACK,
    beatthis_checkpoint: str = "final0",
    beatthis_device: str = "cpu",
    beatthis_float16: bool = False,
    grid_fitter_cache_path: str | Path = DEFAULT_GRID_FITTER_CACHE_PATH,
    grid_fitter_cache_meta_path: str | Path = DEFAULT_GRID_FITTER_CACHE_META_PATH,
    adapter_cache_path: str | Path = DEFAULT_ADAPTER_CACHE_PATH,
    output_json_path: str | Path = DEFAULT_OUTPUT_JSON_PATH,
    output_md_path: str | Path = DEFAULT_OUTPUT_MD_PATH,
    max_maps: int | None = None,
    config_preset: str = "primary",
    progress_every: int = 250,
    audit_command: str | None = None,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    index_path = Path(index_path)
    dataset_root = Path(dataset_root)
    if grid_source not in {GRID_SOURCE_RED_TIMING_FALLBACK, GRID_SOURCE_STAGE2_TIMING_MODULE}:
        raise ValueError(f"unknown grid source: {grid_source}")
    grid_fitter_cache_path = Path(grid_fitter_cache_path)
    grid_fitter_cache_meta_path = Path(grid_fitter_cache_meta_path)
    adapter_cache_path = Path(adapter_cache_path)
    output_json_path = Path(output_json_path)
    output_md_path = Path(output_md_path)

    index_df = _load_unique_beatmap_index(index_path)
    if max_maps is not None:
        index_df = index_df.head(max_maps).copy()

    configs = _adapter_configs_for_preset(config_preset)
    primary_config = configs[0]
    grid_cache_rows: list[dict[str, Any]] = []
    adapter_writer = _ParquetChunkWriter(adapter_cache_path)

    dataset_counts = Counter()
    parse_failures = Counter()
    action_counts = Counter()
    section_counts: list[int] = []
    fitter_sources = Counter()
    config_metrics = {config.name: _new_config_metrics(config) for config in configs}
    primary_residuals_after = array("f")
    primary_residuals_before = array("f")
    primary_divisor_counts = Counter()
    primary_token_metrics: dict[str, dict[str, Any]] | None = None
    primary_success_maps: list[dict[str, Any]] = []
    knot_examples: list[dict[str, Any]] = []
    failure_examples: list[dict[str, Any]] = []
    cross_section_assignment_count = 0
    cross_section_rejected_count = 0
    timing_module_cache = (
        _Stage2TimingModuleGridCache(
            checkpoint_path=beatthis_checkpoint,
            device=beatthis_device,
            float16=beatthis_float16,
        )
        if grid_source == GRID_SOURCE_STAGE2_TIMING_MODULE
        else None
    )

    for row_number, row in enumerate(index_df.itertuples(index=False), start=1):
        beatmap_key = _beatmap_key(row)
        beatmap_path = dataset_root / str(row.shard) / str(row.beatmap_path)
        audio_path = dataset_root / str(row.shard) / str(getattr(row, "audio_path"))
        try:
            grid, source, confidence = _grid_for_audit_row(
                beatmap_path=beatmap_path,
                audio_path=audio_path,
                grid_source=grid_source,
                timing_module_cache=timing_module_cache,
            )
            events = parse_primitive_events(beatmap_path)
        except MissingRedTimingError:
            parse_failures["no_red_timing_points"] += 1
            _append_failure_example(failure_examples, beatmap_path, "no_red_timing_points")
            continue
        except InvalidRedTimingError as exc:
            parse_failures["invalid_red_timing_points"] += 1
            _append_failure_example(failure_examples, beatmap_path, str(exc))
            continue
        except (OSError, RuntimeError) as exc:
            parse_failures["timing_module_failure"] += 1
            _append_failure_example(failure_examples, beatmap_path, str(exc))
            continue
        except ValueError as exc:
            parse_failures[_parser_failure_type(str(exc))] += 1
            _append_failure_example(failure_examples, beatmap_path, str(exc))
            continue

        sections = sections_from_fitted_grid(
            beatmap_key=beatmap_key,
            beatmap_path=beatmap_path,
            grid=grid,
            source=source,
            confidence=confidence,
        )
        grid_cache_rows.extend(asdict(section) for section in sections)
        fitter_sources[source] += 1
        section_counts.append(len(sections))

        dataset_counts["beatmaps_audited"] += 1
        dataset_counts["primitive_events"] += len(events)
        unique_times = sorted({event.original_time_ms for event in events})
        dataset_counts["unique_raw_event_times"] += len(unique_times)
        for event in events:
            action_counts[event.action] += 1

        primary_assignment: AdapterAssignmentResult | None = None
        primary_legality: LegalityResult | None = None
        for config in configs:
            metrics = config_metrics[config.name]
            assignment, legality = assign_primitive_events(events, sections, config)
            _update_config_metrics(metrics, events, unique_times, assignment, legality)
            if config.name == primary_config.name:
                primary_assignment = assignment
                primary_legality = legality

        assert primary_assignment is not None
        assert primary_legality is not None
        if primary_assignment.ok and primary_legality.ok:
            assignments = primary_assignment.assignments
            adapter_writer.write_rows(
                _adapter_cache_rows(
                    beatmap_key=beatmap_key,
                    beatmap_path=beatmap_path,
                    config=primary_config,
                    assignments=assignments,
                )
            )
            primary_success_maps.append(
                {
                    "beatmap_key": beatmap_key,
                    "beatmap_path": beatmap_path.as_posix(),
                    "event_count": len(events),
                    "unique_time_count": len(unique_times),
                    "assignments": assignments,
                    "sections": sections,
                }
            )
            for assignment in assignments:
                primary_residuals_after.append(abs(float(assignment.residual_ms)))
                primary_divisor_counts[int(assignment.divisor)] += 1
                before = generate_tick_candidates(
                    assignment.raw_time_ms,
                    sections,
                    primary_config.divisors,
                    offsets=(0,),
                    residual_tolerance_ms=10.0,
                )
                if before:
                    primary_residuals_before.append(abs(float(before[0].residual_ms)))
            for knot in primary_assignment.knots[:3]:
                if len(knot_examples) < WORST_EXAMPLE_LIMIT:
                    payload = dict(knot)
                    payload["beatmap_path"] = beatmap_path.as_posix()
                    knot_examples.append(payload)
            cross_section_assignment_count += primary_assignment.cross_section_assignment_count
        else:
            cross_section_rejected_count += 1
            _append_failure_example(
                failure_examples,
                beatmap_path,
                primary_assignment.failure_type or "primary_adapter_failed",
                unique_times=unique_times,
            )

        if progress_every > 0 and (row_number == 1 or row_number % progress_every == 0):
            print(
                f"high_precision_grid_adapter_audit progress maps={row_number}/{len(index_df)} "
                f"audited={dataset_counts['beatmaps_audited']} "
                f"events={dataset_counts['primitive_events']}",
                file=sys.stderr,
                flush=True,
            )

    adapter_writer.close()

    grid_fitter_cache_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(grid_cache_rows).to_parquet(grid_fitter_cache_path, index=False)
    timing_module_stats = timing_module_cache.stats() if timing_module_cache is not None else None
    source_note = _grid_source_note(grid_source, timing_module_stats)
    grid_meta = {
        "schema_version": 1,
        "commit": _git_rev_parse("HEAD"),
        "index_path": index_path.as_posix(),
        "grid_source": grid_source,
        "source_note": source_note,
        "grid_fitter_implementation_found": True,
        "maps_with_fitted_grid": int(dataset_counts["beatmaps_audited"]),
        "maps_using_red_timing_fallback": int(fitter_sources[GRID_SOURCE_RED_TIMING_FALLBACK]),
        "maps_using_stage2_timing_module": int(fitter_sources[GRID_SOURCE_STAGE2_TIMING_MODULE]),
        "stage2_timing_module": timing_module_stats,
        "section_count": _number_list_stats(section_counts),
    }
    grid_fitter_cache_meta_path.parent.mkdir(parents=True, exist_ok=True)
    grid_fitter_cache_meta_path.write_text(json.dumps(_json_ready(grid_meta), indent=2, sort_keys=True) + "\n")

    divisor_report = derive_divisor_tiers(primary_divisor_counts)
    primary_token_metrics = _build_tokenization_metrics(
        primary_success_maps,
        divisor_report,
        k_max_values=DEFAULT_K_MAX_VALUES,
    )
    config_report = {
        name: _finalize_config_metrics(metrics, dataset_counts["beatmaps_audited"])
        for name, metrics in sorted(config_metrics.items())
    }
    recommendation = _build_recommendation(config_report, divisor_report, primary_token_metrics)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "audit_name": "stage2_high_precision_grid_adapter_audit",
        "framing": (
            "This audit measures whether raw .osu integer-ms beatmaps can be lifted to a fitted high-precision "
            "timing grid through a monotone adapter. It does not require raw object times to be exact grid ticks."
        ),
        "provenance": {
            "commit": _git_rev_parse("HEAD"),
            "baseline_commit_context": "090286a251427f5d52504ea9f98687120caf5ab1",
            "index_path": index_path.as_posix(),
            "index_sha256": _sha256_file(index_path),
            "dataset_root": dataset_root.as_posix(),
            "audit_command": audit_command or " ".join(sys.argv),
            "started_at_unix": started_at,
            "elapsed_s": time.perf_counter() - started_at,
        },
        "dataset": {
            "index_row_count_after_dedupe": int(len(index_df)),
            "beatmaps_audited": int(dataset_counts["beatmaps_audited"]),
            "primitive_events_audited": int(dataset_counts["primitive_events"]),
            "unique_raw_event_times_audited": int(dataset_counts["unique_raw_event_times"]),
            "tap_count": int(action_counts[ACTION_TAP]),
            "hold_start_count": int(action_counts[ACTION_HOLD_START]),
            "hold_end_count": int(action_counts[ACTION_HOLD_END]),
            "parse_failures": dict(sorted(parse_failures.items())),
        },
        "grid_fitter_cache": grid_meta,
        "adapter_configs": {config.name: _config_json(config) for config in configs},
        "adapter_coverage": config_report,
        "residuals": {
            "primary_before_adapter_ms": _array_stats(primary_residuals_before),
            "primary_after_adapter_ms": _array_stats(primary_residuals_after),
        },
        "adapter_knots": {
            "examples": knot_examples,
            "cross_section_assignment_count": cross_section_assignment_count,
            "cross_section_rejected_count": cross_section_rejected_count,
        },
        "divisor_usage": divisor_report,
        "grid_only_tokenization": primary_token_metrics,
        "examples": {
            "worst_failures": failure_examples,
            "suspicious_knots": [
                example for example in knot_examples if bool(example.get("appears_dense"))
            ][:WORST_EXAMPLE_LIMIT],
        },
        "recommendation": recommendation,
        "artifacts": {
            "grid_fitter_cache_path": grid_fitter_cache_path.as_posix(),
            "grid_fitter_cache_meta_path": grid_fitter_cache_meta_path.as_posix(),
            "adapter_cache_path": adapter_cache_path.as_posix(),
            "output_json_path": output_json_path.as_posix(),
            "output_md_path": output_md_path.as_posix(),
        },
    }

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_md_path.parent.mkdir(parents=True, exist_ok=True)
    output_md_path.write_text(_render_markdown(payload), encoding="utf-8")
    return payload


def _assign_zero_offset_fast_path(
    unique_times: tuple[int, ...],
    sections: Sequence[FittedTimingSection | TimingSection],
    config: AdapterConfig,
) -> AdapterAssignmentResult:
    assignments: list[AssignedTick] = []
    candidate_count = 0
    previous_tick: float | None = None
    previous_adapted: float | None = None
    collapsed = 0
    monotonicity_mode = config.effective_monotonicity_mode
    for index, time_ms in enumerate(unique_times):
        candidates = generate_tick_candidates(
            time_ms,
            sections,
            config.divisors,
            offsets=(0,),
            residual_tolerance_ms=config.residual_tolerance_ms,
        )
        candidate_count += len(candidates)
        if not candidates:
            return AdapterAssignmentResult(ok=False, failure_type="no_candidate", failure_index=index)
        candidate = candidates[0]
        if previous_tick is not None:
            assert previous_adapted is not None
            if monotonicity_mode == MONOTONICITY_STRICT and (
                candidate.adapted_time_ms <= previous_adapted + EXACT_EPSILON_MS
                or candidate.tick_time_ms_hp <= previous_tick + EXACT_EPSILON_MS
            ):
                return AdapterAssignmentResult(ok=False, failure_type="monotonicity", failure_index=index)
            if monotonicity_mode != MONOTONICITY_STRICT and (
                candidate.adapted_time_ms < previous_adapted - EXACT_EPSILON_MS
                or candidate.tick_time_ms_hp < previous_tick - EXACT_EPSILON_MS
            ):
                return AdapterAssignmentResult(ok=False, failure_type="monotonicity", failure_index=index)
            if monotonicity_mode != MONOTONICITY_STRICT and (
                abs(candidate.tick_time_ms_hp - previous_tick) <= EXACT_EPSILON_MS
                or abs(candidate.adapted_time_ms - previous_adapted) <= EXACT_EPSILON_MS
            ):
                collapsed += 1
        previous_tick = candidate.tick_time_ms_hp
        previous_adapted = candidate.adapted_time_ms
        assignments.append(_assigned_tick_from_candidate(candidate))
    return AdapterAssignmentResult(
        ok=True,
        assignments=tuple(assignments),
        collapsed_distinct_time_count=collapsed,
        cross_section_assignment_count=sum(1 for assignment in assignments if assignment.cross_section),
        candidate_count=candidate_count,
    )


def _adapted_timepoints(
    events: Sequence[PrimitiveEvent],
    assignments: Sequence[AssignedTick],
) -> tuple[AdaptedTimepoint, ...]:
    events_by_time: dict[int, list[PrimitiveEvent]] = defaultdict(list)
    for event in sorted(events, key=lambda item: item.sort_key):
        events_by_time[event.original_time_ms].append(event)
    timepoints: list[AdaptedTimepoint] = []
    for assignment in assignments:
        lane_actions: list[str | None] = [None, None, None, None]
        for event in events_by_time.get(assignment.raw_time_ms, []):
            lane_actions[event.lane] = event.action
        timepoints.append(
            AdaptedTimepoint(
                raw_time_ms=assignment.raw_time_ms,
                grid_time_ms=assignment.tick_time_ms_hp,
                integer_ms_correction_ms=assignment.residual_ms,
                section_index=assignment.section_id,
                divisor=assignment.divisor,
                tick_index=assignment.tick_index,
                lane_actions=tuple(lane_actions),  # type: ignore[arg-type]
            )
        )
    return tuple(timepoints)


def _raise_legality_error(legality: LegalityResult) -> None:
    reason = legality.failure_examples[0]["reason"] if legality.failure_examples else "lane_ln_legality"
    if reason == "hold_end_without_open":
        raise ValueError("HOLD_END without open hold")
    if reason == "hold_start_while_lane_open":
        raise ValueError("HOLD_START while hold is open")
    if reason == "tap_while_lane_open":
        raise ValueError("TAP while hold is open")
    if reason == "hold_end_before_or_equal_start":
        raise ValueError("HOLD_END before or equal HOLD_START")
    if reason == "duplicate_same_lane_tap":
        raise ValueError("duplicate same-lane TAP at adapted tick")
    raise ValueError(f"lane/LN legality failure: {reason}")


def _transition_cost_and_knot(
    previous: TickCandidate,
    current: TickCandidate,
    gap: GapFeatures,
    config: AdapterConfig,
) -> tuple[float, dict[str, Any] | None, bool] | _TransitionReject:
    if previous.raw_time_ms >= current.raw_time_ms:
        return _TransitionReject("raw_order")

    monotonicity_mode = config.effective_monotonicity_mode
    collapsed = False
    if monotonicity_mode == MONOTONICITY_STRICT:
        if current.adapted_time_ms <= previous.adapted_time_ms + EXACT_EPSILON_MS:
            return _TransitionReject("adapted_monotonicity")
        if current.tick_time_ms_hp <= previous.tick_time_ms_hp + EXACT_EPSILON_MS:
            return _TransitionReject("monotonicity")
    else:
        if current.adapted_time_ms < previous.adapted_time_ms - EXACT_EPSILON_MS:
            return _TransitionReject("adapted_monotonicity")
        if current.tick_time_ms_hp < previous.tick_time_ms_hp - EXACT_EPSILON_MS:
            return _TransitionReject("monotonicity")
        collapsed = (
            abs(current.tick_time_ms_hp - previous.tick_time_ms_hp) <= EXACT_EPSILON_MS
            or abs(current.adapted_time_ms - previous.adapted_time_ms) <= EXACT_EPSILON_MS
        )

    offset_delta = current.offset_ms - previous.offset_ms
    if offset_delta == 0:
        return ((config.collapse_penalty if collapsed else 0.0), None, collapsed)
    if abs(offset_delta) > config.max_offset_delta_ms:
        return _TransitionReject("offset_jump", wanted_dense_knot=gap.dense_core)
    knot_kind = _safe_knot_kind(gap, config.safe_knot_policy)
    if knot_kind is None:
        return _TransitionReject("safe_knot", wanted_dense_knot=gap.dense_core)

    penalty = (
        0.2
        + abs(offset_delta) * 0.2
        + _gap_penalty(gap, config.safe_knot_policy)
        + (config.collapse_penalty if collapsed else 0.0)
    )
    knot = {
        "left_time_ms": gap.left_time_ms,
        "right_time_ms": gap.right_time_ms,
        "raw_gap_ms": gap.raw_gap_ms,
        "normalized_gap_beats": gap.normalized_gap_beats,
        "local_event_density_left": gap.local_event_density_left,
        "local_event_density_right": gap.local_event_density_right,
        "left_density_percentile": gap.left_density_percentile,
        "right_density_percentile": gap.right_density_percentile,
        "gap_percentile_within_section": gap.gap_percentile_within_section,
        "density_percentile": gap.density_percentile,
        "section_id": gap.section_id,
        "offset_before_ms": previous.offset_ms,
        "offset_after_ms": current.offset_ms,
        "offset_delta_ms": offset_delta,
        "near_section_boundary": gap.near_section_boundary,
        "appears_dense": gap.appears_dense,
        "dense_pattern_edge": gap.dense_pattern_edge,
        "dense_core": gap.dense_core,
        "knot_kind": knot_kind,
        "safe_knot_policy": config.safe_knot_policy,
    }
    return penalty, knot, collapsed


def _candidate_cost(candidate: TickCandidate) -> float:
    return (
        abs(candidate.residual_ms)
        + 0.001 * math.log2(max(candidate.divisor, 1))
        + 0.01 * abs(candidate.offset_ms)
        + (0.005 if candidate.cross_section else 0.0)
    )


def _dominant_transition_failure_type(counts: Mapping[str, int]) -> str:
    if not counts:
        return "monotone_or_safe_knot"
    priority = {
        "no_candidate": 0,
        "monotonicity": 1,
        "adapted_monotonicity": 2,
        "safe_knot": 3,
        "offset_jump": 4,
        "raw_order": 5,
    }
    reason, _ = max(counts.items(), key=lambda item: (item[1], -priority.get(item[0], 99), item[0]))
    return str(reason)


def _safe_knot_kind(gap: GapFeatures, policy: str) -> str | None:
    if policy == "policy_0_soft":
        return "soft"
    if policy == "policy_1_section_quantile":
        if gap.near_section_boundary:
            return "section_boundary"
        if gap.gap_percentile_within_section >= 0.75:
            return "section_quantile"
        return None
    if policy == "policy_2_sparse_only":
        if gap.gap_percentile_within_section >= 0.85 and gap.density_percentile <= 0.5:
            return "sparse_gap"
        return None
    if policy == "policy_3_boundary_plus_sparse":
        if gap.near_section_boundary:
            return "section_boundary"
        if gap.gap_percentile_within_section >= 0.85 and gap.density_percentile <= 0.5:
            return "sparse_gap"
        if gap.dense_pattern_edge:
            return "dense_pattern_edge"
        return None
    raise ValueError(f"unknown safe knot policy: {policy}")


def _safe_knot_allowed(gap: GapFeatures, policy: str) -> bool:
    return _safe_knot_kind(gap, policy) is not None


def _gap_penalty(gap: GapFeatures, policy: str) -> float:
    if policy == "policy_0_soft":
        return 0.05 / max(gap.salience, 0.05)
    return 0.02 / max(gap.salience, 0.05)


def _assigned_tick_from_candidate(candidate: TickCandidate) -> AssignedTick:
    return AssignedTick(
        raw_time_ms=candidate.raw_time_ms,
        offset_ms=candidate.offset_ms,
        adapted_time_ms=candidate.adapted_time_ms,
        section_id=candidate.section_id,
        divisor=candidate.divisor,
        tick_index=candidate.tick_index,
        tick_time_ms_hp=candidate.tick_time_ms_hp,
        residual_ms=candidate.residual_ms,
        cross_section=candidate.cross_section,
    )


def _decompose_single_section_interval(
    start_ms: float,
    end_ms: float,
    section_index: int,
    section: FittedTimingSection | TimingSection,
    divisors: Sequence[int],
    *,
    k_max: int,
    exact_epsilon_ms: float,
) -> DecompositionResult:
    divisors = tuple(sorted({int(divisor) for divisor in divisors}))
    lcm = _lcm_many(divisors)
    start_units = _time_to_common_units(start_ms, section, lcm, exact_epsilon_ms=exact_epsilon_ms)
    end_units = _time_to_common_units(end_ms, section, lcm, exact_epsilon_ms=exact_epsilon_ms)
    if start_units is None or end_units is None:
        return DecompositionResult(ok=False, failure_type="event_not_on_grid_for_divisor_set")
    if end_units < start_units:
        return DecompositionResult(ok=False, failure_type="negative_delta")

    tokens: list[GridToken] = []
    current_units = start_units
    guard = 0
    while current_units < end_units:
        final_token = _token_to_exact_target(current_units, end_units, lcm, divisors, k_max)
        if final_token is not None:
            divisor, k, landing_units = final_token
        else:
            progress_token = _progress_token(current_units, end_units, lcm, divisors, k_max)
            if progress_token is None:
                return DecompositionResult(ok=False, failure_type="canonical_decomposition_exceeded_k_max")
            divisor, k, landing_units = progress_token

        token_start_ms = _section_beat_zero(section) + current_units * _section_beat_length(section) / lcm
        token_end_ms = _section_beat_zero(section) + landing_units * _section_beat_length(section) / lcm
        tokens.append(
            GridToken(
                section_index=section_index,
                divisor=int(divisor),
                k=int(k),
                start_ms=float(token_start_ms),
                end_ms=float(token_end_ms),
            )
        )
        current_units = landing_units
        guard += 1
        if guard > 100000:
            return DecompositionResult(ok=False, failure_type="canonical_decomposition_exceeded_k_max")
    return DecompositionResult(ok=True, tokens=tuple(tokens))


def _token_to_exact_target(
    current_units: int,
    target_units: int,
    lcm: int,
    divisors: Sequence[int],
    k_max: int,
) -> tuple[int, int, int] | None:
    candidates: list[tuple[int, int, int, int]] = []
    for divisor in divisors:
        step_units = lcm // divisor
        if current_units % step_units != 0 or target_units % step_units != 0:
            continue
        k = target_units // step_units - current_units // step_units
        if 1 <= k <= k_max:
            candidates.append((1, divisor, -k, target_units))
    if not candidates:
        return None
    _, divisor, negative_k, landing_units = min(candidates)
    return divisor, -negative_k, landing_units


def _progress_token(
    current_units: int,
    target_units: int,
    lcm: int,
    divisors: Sequence[int],
    k_max: int,
) -> tuple[int, int, int] | None:
    candidates: list[tuple[int, int, int, int]] = []
    for divisor in divisors:
        step_units = lcm // divisor
        floor_index = current_units // step_units
        max_index_before_target = (target_units - 1) // step_units
        k = min(k_max, max_index_before_target - floor_index)
        if k < 1:
            continue
        landing_units = (floor_index + k) * step_units
        if current_units < landing_units < target_units:
            candidates.append((-landing_units, divisor, -k, k))
    if not candidates:
        return None
    negative_landing_units, divisor, negative_k, k = min(candidates)
    return divisor, k if negative_k < 0 else -negative_k, -negative_landing_units


def _time_to_common_units(
    time_ms: float,
    section: FittedTimingSection | TimingSection,
    lcm: int,
    *,
    exact_epsilon_ms: float,
) -> int | None:
    raw_units = (float(time_ms) - _section_beat_zero(section)) * lcm / _section_beat_length(section)
    units = int(round(raw_units))
    reconstructed = _section_beat_zero(section) + units * _section_beat_length(section) / lcm
    if abs(float(time_ms) - reconstructed) > exact_epsilon_ms:
        return None
    return units


def _build_tokenization_metrics(
    success_maps: Sequence[Mapping[str, Any]],
    divisor_report: Mapping[str, Any],
    *,
    k_max_values: Sequence[int],
) -> dict[str, Any]:
    tiers = {
        "D_core_2to6": tuple(divisor_report["D_core_2to6"]),
        "D_main_2to6": tuple(divisor_report["D_main_2to6"]),
        "D_safe_2to6": tuple(divisor_report["D_safe_2to6"]),
    }
    report: dict[str, Any] = {}
    for tier_name, divisors in tiers.items():
        for k_max in k_max_values:
            label = f"{tier_name}_k{k_max}"
            map_counts: list[int] = []
            window_counts: list[int] = []
            failures = Counter()
            max_delta_tokens = 0
            section_boundary_splits = 0
            for map_info in success_maps:
                assignments = tuple(map_info["assignments"])
                sections = tuple(map_info["sections"])
                result = _time_shift_token_count(assignments, sections, divisors, k_max=k_max)
                if result["ok"]:
                    map_counts.append(int(result["token_count"]))
                    max_delta_tokens = max(max_delta_tokens, int(result["max_delta_tokens"]))
                    section_boundary_splits += int(result["section_boundary_split_count"])
                else:
                    failures[str(result["failure_type"])] += 1
                for window_assignments in _assignments_by_window(assignments).values():
                    window_result = _time_shift_token_count(window_assignments, sections, divisors, k_max=k_max)
                    if window_result["ok"]:
                        window_counts.append(int(window_result["token_count"]))
                        max_delta_tokens = max(max_delta_tokens, int(window_result["max_delta_tokens"]))
                    else:
                        failures["window_" + str(window_result["failure_type"])] += 1
            report[label] = {
                "divisors": list(divisors),
                "k_max": int(k_max),
                "vocab_size": _grid_vocab_size(len(divisors), int(k_max)),
                "successful_full_map_count": len(map_counts),
                "successful_8s_window_count": len(window_counts),
                "full_map_time_shift_tokens": _number_list_stats(map_counts),
                "window_8s_time_shift_tokens": _number_list_stats(window_counts),
                "max_time_shift_tokens_for_one_event_delta": max_delta_tokens,
                "section_boundary_split_count": section_boundary_splits,
                "decomposition_failures": dict(sorted(failures.items())),
            }
    return report


def _time_shift_token_count(
    assignments: Sequence[AssignedTick],
    sections: Sequence[FittedTimingSection | TimingSection],
    divisors: Sequence[int],
    *,
    k_max: int,
) -> dict[str, Any]:
    total = 0
    max_delta = 0
    split_count = 0
    ordered = sorted(assignments, key=lambda assignment: assignment.raw_time_ms)
    for left, right in zip(ordered[:-1], ordered[1:], strict=True):
        result = decompose_interval_to_grid_tokens(
            left.tick_time_ms_hp,
            right.tick_time_ms_hp,
            sections,
            divisors,
            k_max=k_max,
        )
        if not result.ok:
            return {"ok": False, "failure_type": result.failure_type}
        token_count = len(result.tokens)
        total += token_count
        max_delta = max(max_delta, token_count)
        split_count += result.section_boundary_split_count
    return {
        "ok": True,
        "token_count": total,
        "max_delta_tokens": max_delta,
        "section_boundary_split_count": split_count,
    }


def _assignments_by_window(assignments: Sequence[AssignedTick]) -> dict[int, list[AssignedTick]]:
    windows: dict[int, list[AssignedTick]] = defaultdict(list)
    for assignment in assignments:
        windows[int(assignment.raw_time_ms) // WINDOW_LENGTH_MS].append(assignment)
    return dict(windows)


def _new_config_metrics(config: AdapterConfig) -> dict[str, Any]:
    return {
        "config": config,
        "beatmap_success_count": 0,
        "beatmap_failed_count": 0,
        "unique_time_success_count": 0,
        "unique_time_failed_count": 0,
        "primitive_event_success_count": 0,
        "primitive_event_failed_count": 0,
        "window_success_count": 0,
        "window_failed_count": 0,
        "failure_types": Counter(),
        "ln_invalid_count": 0,
        "same_lane_collision_count": 0,
        "hold_end_before_or_equal_start_count": 0,
        "no_candidate_failures": 0,
        "monotonicity_failures": 0,
        "safe_knot_failures": 0,
        "offset_jump_failures": 0,
        "adapted_monotonicity_failures": 0,
        "sequence_identity_failures": 0,
        "transition_failure_types": Counter(),
        "collapsed_distinct_time_count": 0,
        "cross_section_assignment_count": 0,
        "knots_per_beatmap": [],
        "offset_magnitude": Counter(),
        "offset_transition": Counter(),
        "knot_gap_ms": [],
        "knot_normalized_gap_beats": [],
        "knot_density": [],
        "dense_knot_count": 0,
        "dense_core_knot_count": 0,
        "dense_pattern_edge_knot_count": 0,
        "knot_kind": Counter(),
    }


def _update_config_metrics(
    metrics: dict[str, Any],
    events: Sequence[PrimitiveEvent],
    unique_times: Sequence[int],
    assignment: AdapterAssignmentResult,
    legality: LegalityResult,
) -> None:
    windows = {int(time_ms) // WINDOW_LENGTH_MS for time_ms in unique_times}
    if assignment.ok and legality.ok:
        metrics["beatmap_success_count"] += 1
        metrics["unique_time_success_count"] += len(unique_times)
        metrics["primitive_event_success_count"] += len(events)
        metrics["window_success_count"] += len(windows)
    else:
        metrics["beatmap_failed_count"] += 1
        metrics["unique_time_failed_count"] += len(unique_times)
        metrics["primitive_event_failed_count"] += len(events)
        metrics["window_failed_count"] += len(windows)
        metrics["failure_types"][assignment.failure_type or "unknown"] += 1
    metrics["ln_invalid_count"] += legality.ln_invalid_count
    metrics["same_lane_collision_count"] += legality.same_lane_collision_count
    metrics["hold_end_before_or_equal_start_count"] += legality.hold_end_before_or_equal_start_count
    metrics["no_candidate_failures"] += assignment.no_candidate_count
    metrics["monotonicity_failures"] += assignment.monotonicity_failure_count
    metrics["safe_knot_failures"] += assignment.safe_knot_failure_count
    metrics["offset_jump_failures"] += assignment.offset_jump_failure_count
    metrics["adapted_monotonicity_failures"] += assignment.adapted_monotonicity_failure_count
    metrics["sequence_identity_failures"] += assignment.sequence_identity_failure_count
    metrics["transition_failure_types"].update(assignment.transition_failure_counts)
    metrics["collapsed_distinct_time_count"] += assignment.collapsed_distinct_time_count
    metrics["cross_section_assignment_count"] += assignment.cross_section_assignment_count
    metrics["knots_per_beatmap"].append(assignment.knot_count)
    for assignment_tick in assignment.assignments:
        metrics["offset_magnitude"][abs(assignment_tick.offset_ms)] += 1
    previous_offset: int | None = None
    for assignment_tick in assignment.assignments:
        if previous_offset is not None:
            metrics["offset_transition"][assignment_tick.offset_ms - previous_offset] += 1
        previous_offset = assignment_tick.offset_ms
    for knot in assignment.knots:
        metrics["knot_gap_ms"].append(float(knot["raw_gap_ms"]))
        metrics["knot_normalized_gap_beats"].append(float(knot["normalized_gap_beats"]))
        metrics["knot_density"].append(
            float(knot["local_event_density_left"]) + float(knot["local_event_density_right"])
        )
        metrics["dense_knot_count"] += int(bool(knot.get("appears_dense")))
        metrics["dense_core_knot_count"] += int(bool(knot.get("dense_core")))
        metrics["dense_pattern_edge_knot_count"] += int(bool(knot.get("dense_pattern_edge")))
        metrics["knot_kind"][str(knot.get("knot_kind", "unknown"))] += 1


def _finalize_config_metrics(metrics: Mapping[str, Any], beatmap_count: int) -> dict[str, Any]:
    primitive_success = int(metrics["primitive_event_success_count"])
    primitive_failed = int(metrics["primitive_event_failed_count"])
    unique_success = int(metrics["unique_time_success_count"])
    unique_failed = int(metrics["unique_time_failed_count"])
    window_success = int(metrics["window_success_count"])
    window_failed = int(metrics["window_failed_count"])
    return {
        "divisor_tier": metrics["config"].name.split("_offset", 1)[0],
        "divisors": list(metrics["config"].divisors),
        "offset_range": [metrics["config"].offset_min_ms, metrics["config"].offset_max_ms],
        "residual_tolerance_ms": metrics["config"].residual_tolerance_ms,
        "safe_knot_policy": metrics["config"].safe_knot_policy,
        "strict_monotonicity": metrics["config"].strict_monotonicity,
        "monotonicity_mode": metrics["config"].effective_monotonicity_mode,
        "max_offset_delta_ms": metrics["config"].max_offset_delta_ms,
        "full_beatmap_adapter_success_rate": _rate(metrics["beatmap_success_count"], beatmap_count),
        "full_beatmap_success_count": int(metrics["beatmap_success_count"]),
        "failed_beatmap_count": int(metrics["beatmap_failed_count"]),
        "unique_raw_time_assignment_success_rate": _rate(unique_success, unique_success + unique_failed),
        "primitive_event_assignment_success_rate": _rate(primitive_success, primitive_success + primitive_failed),
        "window_8s_adapter_success_rate": _rate(window_success, window_success + window_failed),
        "failed_window_count": window_failed,
        "failed_event_count": primitive_failed,
        "no_candidate_failures": int(metrics["no_candidate_failures"]),
        "monotonicity_failures": int(metrics["monotonicity_failures"]),
        "safe_knot_failures": int(metrics["safe_knot_failures"]),
        "offset_jump_failures": int(metrics["offset_jump_failures"]),
        "adapted_monotonicity_failures": int(metrics["adapted_monotonicity_failures"]),
        "sequence_identity_failures": int(metrics["sequence_identity_failures"]),
        "transition_failure_types": dict(sorted(metrics["transition_failure_types"].items())),
        "ln_lane_legality_failures": {
            "ln_invalid_count": int(metrics["ln_invalid_count"]),
            "same_lane_collision_count": int(metrics["same_lane_collision_count"]),
            "hold_end_before_or_equal_start_count": int(metrics["hold_end_before_or_equal_start_count"]),
        },
        "section_boundary_failures": 0,
        "collapsed_distinct_time_count": int(metrics["collapsed_distinct_time_count"]),
        "cross_section_assignment_count": int(metrics["cross_section_assignment_count"]),
        "failure_types": dict(sorted(metrics["failure_types"].items())),
        "adapter_knots": {
            "knots_per_beatmap": _number_list_stats(metrics["knots_per_beatmap"]),
            "offset_magnitude_distribution": dict(sorted(metrics["offset_magnitude"].items())),
            "offset_transition_distribution": dict(sorted(metrics["offset_transition"].items())),
            "selected_knot_gap_size_ms": _number_list_stats(metrics["knot_gap_ms"]),
            "selected_knot_normalized_beat_gap": _number_list_stats(metrics["knot_normalized_gap_beats"]),
            "selected_knot_local_density": _number_list_stats(metrics["knot_density"]),
            "dense_knot_count": int(metrics["dense_knot_count"]),
            "dense_core_knot_count": int(metrics["dense_core_knot_count"]),
            "dense_pattern_edge_knot_count": int(metrics["dense_pattern_edge_knot_count"]),
            "knot_kind_distribution": dict(sorted(metrics["knot_kind"].items())),
        },
    }


def _adapter_configs_for_preset(preset: str) -> tuple[AdapterConfig, ...]:
    primary = AdapterConfig(
        name=PRIMARY_CONFIG_NAME,
        divisors=D_UNIVERSE_2TO6,
        offset_min_ms=-2,
        offset_max_ms=2,
        residual_tolerance_ms=0.5,
        safe_knot_policy="policy_3_boundary_plus_sparse",
        strict_monotonicity=True,
    )
    if preset == "primary":
        return (primary,)
    weak = AdapterConfig(
        name=WEAK_CONFIG_NAME,
        divisors=D_UNIVERSE_2TO6,
        offset_min_ms=-20,
        offset_max_ms=20,
        residual_tolerance_ms=0.5,
        safe_knot_policy="policy_3_boundary_plus_sparse",
        strict_monotonicity=False,
        monotonicity_mode=MONOTONICITY_WEAK,
        beam_width=96,
        max_offset_delta_ms=3,
        collapse_penalty=8.0,
    )
    if preset == "weak":
        return (weak,)
    if preset == "primary_plus_weak":
        return (primary, weak)
    if preset != "standard":
        raise ValueError(f"unknown config preset: {preset}")

    configs: list[AdapterConfig] = [primary]
    for offset_min, offset_max in DEFAULT_OFFSET_RANGES:
        name = f"D_universe_offset_{offset_min}_{offset_max}_tol_0_5_policy_3_strict"
        if name != primary.name:
            configs.append(
                AdapterConfig(
                    name=name,
                    divisors=D_UNIVERSE_2TO6,
                    offset_min_ms=offset_min,
                    offset_max_ms=offset_max,
                    residual_tolerance_ms=0.5,
                    safe_knot_policy="policy_3_boundary_plus_sparse",
                    strict_monotonicity=True,
                )
            )
    for tolerance in DEFAULT_RESIDUAL_TOLERANCES_MS:
        name = f"D_universe_offset_m2_p2_tol_{str(tolerance).replace('.', '_')}_policy_3_strict"
        if name != primary.name:
            configs.append(
                AdapterConfig(
                    name=name,
                    divisors=D_UNIVERSE_2TO6,
                    offset_min_ms=-2,
                    offset_max_ms=2,
                    residual_tolerance_ms=float(tolerance),
                    safe_knot_policy="policy_3_boundary_plus_sparse",
                    strict_monotonicity=True,
                )
            )
    for policy in DEFAULT_SAFE_KNOT_POLICIES:
        name = f"D_universe_offset_m2_p2_tol_0_5_{policy}_strict"
        if name != primary.name:
            configs.append(
                AdapterConfig(
                    name=name,
                    divisors=D_UNIVERSE_2TO6,
                    offset_min_ms=-2,
                    offset_max_ms=2,
                    residual_tolerance_ms=0.5,
                    safe_knot_policy=policy,
                    strict_monotonicity=True,
                )
            )
    configs.append(
        AdapterConfig(
            name="D_universe_offset_m2_p2_tol_0_5_policy_3_relaxed",
            divisors=D_UNIVERSE_2TO6,
            offset_min_ms=-2,
            offset_max_ms=2,
            residual_tolerance_ms=0.5,
            safe_knot_policy="policy_3_boundary_plus_sparse",
            strict_monotonicity=False,
        )
    )
    configs.append(weak)
    return tuple(configs)


def _config_json(config: AdapterConfig) -> dict[str, Any]:
    return {
        "divisors": list(config.divisors),
        "offset_range": [config.offset_min_ms, config.offset_max_ms],
        "residual_tolerance_ms": config.residual_tolerance_ms,
        "safe_knot_policy": config.safe_knot_policy,
        "strict_monotonicity": config.strict_monotonicity,
        "monotonicity_mode": config.effective_monotonicity_mode,
        "beam_width": config.beam_width,
        "max_offset_delta_ms": config.max_offset_delta_ms,
        "collapse_penalty": config.collapse_penalty,
    }


def _build_recommendation(
    config_report: Mapping[str, Mapping[str, Any]],
    divisor_report: Mapping[str, Any],
    token_report: Mapping[str, Any],
) -> dict[str, Any]:
    primary = config_report.get(PRIMARY_CONFIG_NAME, next(iter(config_report.values())))
    success_rate = float(primary["window_8s_adapter_success_rate"])
    strict_failures_low = float(primary["unique_raw_time_assignment_success_rate"]) >= 0.999
    dense_knot_count = int(primary["adapter_knots"].get("dense_core_knot_count", primary["adapter_knots"]["dense_knot_count"]))
    main_divisors = list(divisor_report["D_main_2to6"])
    safe_divisors = list(divisor_report["D_safe_2to6"])
    k_max_choice = _choose_k_max(token_report, "D_main_2to6")
    plausible = success_rate >= 0.999 and strict_failures_low and dense_knot_count == 0
    if plausible:
        summary = (
            "Stage 2 Mapper v1 can plausibly use grid-only TIME_SHIFT after high-precision monotone adaptation. "
            "Keep raw integer-ms reconstruction as an export projection."
        )
        fallback = None
    else:
        summary = (
            "Grid-only TIME_SHIFT is not proven by this run. Use the smallest reported fallback around the failing "
            "axis before treating the adapter as lossless."
        )
        fallback = "rare_residual_or_boundary_fallback"
    return {
        "can_stage2_mapper_v1_use_grid_only_time_shift": plausible,
        "recommended_D_core_2to6": list(divisor_report["D_core_2to6"]),
        "recommended_D_main_2to6": main_divisors,
        "recommended_D_safe_2to6": safe_divisors,
        "recommended_k_max": k_max_choice,
        "divisors_beyond_D_main_useful_on_2to6": safe_divisors != main_divisors,
        "smallest_fallback_if_not": fallback,
        "summary": summary,
    }


def _choose_k_max(token_report: Mapping[str, Any], tier_name: str) -> int:
    for k_max in (32, 64, 128, 16):
        report = token_report.get(f"{tier_name}_k{k_max}")
        if not report:
            continue
        p99 = report["window_8s_time_shift_tokens"].get("p99")
        if p99 is not None and float(p99) <= 256.0:
            return k_max
    return 128


def _render_markdown(payload: Mapping[str, Any]) -> str:
    provenance = payload["provenance"]
    dataset = payload["dataset"]
    recommendation = payload["recommendation"]
    primary = payload["adapter_coverage"].get(PRIMARY_CONFIG_NAME, next(iter(payload["adapter_coverage"].values())))
    lines = [
        "---",
        f"pinned_commit: {provenance['commit']}",
        "audit: stage2_high_precision_grid_adapter_audit",
        "---",
        "",
        "# Stage 2 High-Precision Grid Adapter Audit",
        "",
        payload["framing"],
        "",
        "## Scope",
        "",
        f"- Commit: `{provenance['commit']}`",
        f"- Baseline commit context: `{provenance['baseline_commit_context']}`",
        f"- Dataset/index path: `{provenance['index_path']}`",
        f"- Beatmaps audited: {dataset['beatmaps_audited']:,}",
        f"- Primitive events audited: {dataset['primitive_events_audited']:,}",
        f"- Unique raw event times audited: {dataset['unique_raw_event_times_audited']:,}",
        f"- TAP / HOLD_START / HOLD_END: {dataset['tap_count']:,} / "
        f"{dataset['hold_start_count']:,} / {dataset['hold_end_count']:,}",
        "",
        "## Grid Fitter Cache",
        "",
        f"- GridFitter implementation found: `{payload['grid_fitter_cache']['grid_fitter_implementation_found']}`",
        f"- Grid source: `{payload['grid_fitter_cache']['grid_source']}`",
        f"- Maps with fitted grid cache rows: {payload['grid_fitter_cache']['maps_with_fitted_grid']:,}",
        f"- Red timing fallback maps: {payload['grid_fitter_cache']['maps_using_red_timing_fallback']:,}",
        f"- Stage 2 timing module maps: {payload['grid_fitter_cache']['maps_using_stage2_timing_module']:,}",
        f"- Note: {payload['grid_fitter_cache']['source_note']}",
        "",
        "## Primary Adapter Coverage",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| full beatmap success rate | {_pct(primary['full_beatmap_adapter_success_rate'])} |",
        f"| 8s window success rate | {_pct(primary['window_8s_adapter_success_rate'])} |",
        f"| unique raw time success rate | {_pct(primary['unique_raw_time_assignment_success_rate'])} |",
        f"| primitive event success rate | {_pct(primary['primitive_event_assignment_success_rate'])} |",
        f"| failed beatmaps | {primary['failed_beatmap_count']:,} |",
        f"| failed windows | {primary['failed_window_count']:,} |",
        f"| collapsed distinct raw times | {primary['collapsed_distinct_time_count']:,} |",
        f"| dense selected knots | {primary['adapter_knots']['dense_knot_count']:,} |",
        "",
        "## Divisor Recommendation",
        "",
        f"- D_core_2to6: `{payload['divisor_usage']['D_core_2to6']}`",
        f"- D_main_2to6: `{payload['divisor_usage']['D_main_2to6']}`",
        f"- D_safe_2to6: `{payload['divisor_usage']['D_safe_2to6']}`",
        f"- Diagnostic tail used: `{payload['divisor_usage']['diagnostic_tail_used']}`",
        "",
        "## Grid-Only TIME_SHIFT",
        "",
        "| config | vocab size | 8s p50 | 8s p95 | 8s p99 | 8s max | max TS per delta | failures |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, report in payload["grid_only_tokenization"].items():
        stats = report["window_8s_time_shift_tokens"]
        failure_count = sum(int(value) for value in report["decomposition_failures"].values())
        lines.append(
            f"| {name} | {report['vocab_size']:,} | {_num(stats['p50'])} | {_num(stats['p95'])} | "
            f"{_num(stats['p99'])} | {_num(stats['max'])} | "
            f"{report['max_time_shift_tokens_for_one_event_delta']:,} | {failure_count:,} |"
        )

    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            f"- Can Stage 2 Mapper v1 use grid-only TIME_SHIFT: "
            f"`{recommendation['can_stage2_mapper_v1_use_grid_only_time_shift']}`",
            f"- Recommended D_main_2to6: `{recommendation['recommended_D_main_2to6']}`",
            f"- Recommended k_max: `{recommendation['recommended_k_max']}`",
            f"- Smallest fallback if not: `{recommendation['smallest_fallback_if_not']}`",
            "",
            recommendation["summary"],
            "",
        ]
    )
    return "\n".join(lines)


class _ParquetChunkWriter:
    def __init__(self, path: Path, *, chunk_size: int = 100_000) -> None:
        self.path = path
        self.chunk_size = chunk_size
        self.rows: list[dict[str, Any]] = []
        self.writer: Any = None
        self.pa: Any = None
        self.pq: Any = None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self.path.unlink()

    def write_rows(self, rows: Iterable[dict[str, Any]]) -> None:
        for row in rows:
            self.rows.append(row)
            if len(self.rows) >= self.chunk_size:
                self.flush()

    def flush(self) -> None:
        if not self.rows:
            return
        if self.pa is None or self.pq is None:
            import pyarrow as pa
            import pyarrow.parquet as pq

            self.pa = pa
            self.pq = pq
        table = self.pa.Table.from_pylist(self.rows)
        if self.writer is None:
            self.writer = self.pq.ParquetWriter(self.path, table.schema)
        self.writer.write_table(table)
        self.rows = []

    def close(self) -> None:
        self.flush()
        if self.writer is not None:
            self.writer.close()
        elif not self.path.exists():
            pd.DataFrame().to_parquet(self.path, index=False)


def _adapter_cache_rows(
    *,
    beatmap_key: str,
    beatmap_path: Path,
    config: AdapterConfig,
    assignments: Sequence[AssignedTick],
) -> Iterable[dict[str, Any]]:
    for assignment in assignments:
        yield {
            "beatmap_key": beatmap_key,
            "beatmap_path": beatmap_path.as_posix(),
            "config_name": config.name,
            "raw_time_ms": assignment.raw_time_ms,
            "offset_ms": assignment.offset_ms,
            "adapted_time_ms": assignment.adapted_time_ms,
            "section_id": assignment.section_id,
            "divisor": assignment.divisor,
            "tick_index": assignment.tick_index,
            "tick_time_ms_hp": assignment.tick_time_ms_hp,
            "residual_ms": assignment.residual_ms,
            "cross_section": assignment.cross_section,
        }


def _section_start(section: FittedTimingSection | TimingSection) -> float:
    return float(getattr(section, "start_ms_hp", getattr(section, "offset_ms")))


def _section_end(section: FittedTimingSection | TimingSection) -> float | None:
    return getattr(section, "end_ms_hp", getattr(section, "end_ms", None))


def _section_beat_zero(section: FittedTimingSection | TimingSection) -> float:
    return float(getattr(section, "beat_zero_ms_hp", getattr(section, "offset_ms")))


def _section_beat_length(section: FittedTimingSection | TimingSection) -> float:
    return float(getattr(section, "beat_length_ms_hp", getattr(section, "beat_length_ms")))


def _section_index_at_offsets(offsets: Sequence[float], time_ms: float) -> int:
    return bisect_right(offsets, time_ms) - 1


def _near_section_boundary(
    raw_time_ms: float,
    adapted_time_ms: float,
    sections: Sequence[FittedTimingSection | TimingSection],
    tolerance_ms: float,
) -> bool:
    for section in sections[1:]:
        boundary = _section_start(section)
        if abs(raw_time_ms - boundary) <= tolerance_ms or abs(adapted_time_ms - boundary) <= tolerance_ms:
            return True
    return False


def _gap_crosses_or_touches_section_boundary(
    left_time_ms: int,
    right_time_ms: int,
    sections: Sequence[FittedTimingSection | TimingSection],
) -> bool:
    return any(left_time_ms < _section_start(section) < right_time_ms for section in sections[1:])


def _count_times_in_range(times: Sequence[int], lower: float, upper: float) -> int:
    return bisect_right(times, upper) - bisect_left(times, lower)


def _rank_percentile(sorted_values: Sequence[float], value: float) -> float:
    if not sorted_values:
        return 0.0
    return bisect_right(sorted_values, value) / len(sorted_values)


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _integer_osu_time(value: float) -> int | None:
    rounded = int(round(float(value)))
    if abs(float(value) - rounded) > 1.0e-6:
        return None
    return rounded


def _load_unique_beatmap_index(index_path: Path) -> pd.DataFrame:
    index_df = pd.read_parquet(index_path)
    required_columns = {"shard", "beatmap_path", "difficulty", "mode", "key_count"}
    missing = sorted(required_columns.difference(index_df.columns))
    if missing:
        raise ValueError(f"{index_path} is missing required column(s): {missing}")
    filtered = index_df[
        (index_df["mode"] == 3)
        & (index_df["key_count"] == 4)
        & (index_df["difficulty"] >= 2.0)
        & (index_df["difficulty"] <= 6.0)
    ].copy()
    return filtered.drop_duplicates(["shard", "beatmap_path"]).reset_index(drop=True)


def _beatmap_key(row: object) -> str:
    beatmap_id = getattr(row, "beatmap_id", None)
    if beatmap_id is not None and not pd.isna(beatmap_id):
        return str(int(beatmap_id))
    return f"{getattr(row, 'shard')}:{getattr(row, 'beatmap_path')}"


def _parser_failure_type(message: str) -> str:
    lowered = message.lower()
    if "not a 4k" in lowered or "not an osu!mania" in lowered:
        return "unsupported_key_count_or_mode"
    if "hold" in lowered:
        return "malformed_hold"
    if "non-integer" in lowered:
        return "non_integer_hitobject_time"
    return "unknown_parser_issue"


def _append_example(examples: list[dict[str, Any]], event: PrimitiveEvent, reason: str) -> None:
    if len(examples) >= WORST_EXAMPLE_LIMIT:
        return
    examples.append(
        {
            "object_id": event.object_id,
            "lane": event.lane,
            "action": event.action,
            "original_time_ms": event.original_time_ms,
            "reason": reason,
        }
    )


def _append_failure_example(
    examples: list[dict[str, Any]],
    beatmap_path: Path,
    reason: str,
    *,
    unique_times: Sequence[int] | None = None,
) -> None:
    if len(examples) >= WORST_EXAMPLE_LIMIT:
        return
    payload: dict[str, Any] = {"beatmap_path": beatmap_path.as_posix(), "reason": reason}
    if unique_times:
        payload["raw_times_excerpt"] = list(unique_times[:12])
    examples.append(payload)


def _smallest_prefix_for_threshold(cumulative_rows: Sequence[Mapping[str, Any]], threshold: float) -> list[int]:
    selected: list[int] = []
    for row in cumulative_rows:
        selected.append(int(row["divisor"]))
        if float(row["cumulative_rate"]) >= threshold:
            return selected
    return selected


def _array_stats(values: array) -> dict[str, float | int | None]:
    if not values:
        return _empty_stats()
    data = np.frombuffer(values, dtype=np.float32)
    return _stats_from_numpy(data)


def _number_list_stats(values: Sequence[float | int]) -> dict[str, float | int | None]:
    if not values:
        return _empty_stats()
    data = np.asarray(values, dtype=np.float64)
    return _stats_from_numpy(data)


def _stats_from_numpy(data: np.ndarray) -> dict[str, float | int]:
    return {
        "count": int(data.size),
        "mean": float(np.mean(data)),
        "p50": float(np.percentile(data, 50)),
        "p90": float(np.percentile(data, 90)),
        "p95": float(np.percentile(data, 95)),
        "p99": float(np.percentile(data, 99)),
        "max": float(np.max(data)),
    }


def _empty_stats() -> dict[str, None | int]:
    return {"count": 0, "mean": None, "p50": None, "p90": None, "p95": None, "p99": None, "max": None}


def _grid_vocab_size(divisor_count: int, k_max: int) -> int:
    return SPECIAL_VOCAB_SIZE + END_VOCAB_SIZE + EVENT_VOCAB_SIZE + divisor_count * int(k_max)


def _lcm_many(values: Iterable[int]) -> int:
    result = 1
    for value in values:
        result = math.lcm(result, int(value))
    return result


def _rate(numerator: int | float, denominator: int | float) -> float:
    return 0.0 if denominator == 0 else float(numerator) / float(denominator)


def _pct(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100:.6f}%"


def _num(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.6g}"


def _grid_source_note(grid_source: str, timing_module_stats: Mapping[str, Any] | None) -> str:
    if grid_source == GRID_SOURCE_RED_TIMING_FALLBACK:
        return (
            "This audit uses high-precision red timing as red_timing_fallback. No BeatThis/GridFitter timing "
            "module inference was run for this cache."
        )
    if grid_source == GRID_SOURCE_STAGE2_TIMING_MODULE:
        unique_audio_count = (
            int(timing_module_stats.get("unique_audio_fit_count", 0)) if timing_module_stats is not None else 0
        )
        return (
            "This audit uses the Stage 2 timing module: BeatThis audio prediction plus GridFitter timing grid "
            f"fit. The BeatThis provider/model is cached in-process, and fitted grids are memoized by audio path "
            f"for {unique_audio_count} unique audio file(s)."
        )
    return f"This audit uses grid_source={grid_source!r}."


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_rev_parse(revision: str) -> str:
    try:
        result = subprocess.run(["git", "rev-parse", revision], check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, Counter):
        return dict(value)
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit high-precision grid adapter feasibility for Stage 2 Mapper.")
    parser.add_argument("--index-path", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument(
        "--grid-source",
        choices=(GRID_SOURCE_RED_TIMING_FALLBACK, GRID_SOURCE_STAGE2_TIMING_MODULE),
        default=GRID_SOURCE_RED_TIMING_FALLBACK,
    )
    parser.add_argument("--beatthis-checkpoint", default="final0")
    parser.add_argument("--beatthis-device", default="cpu")
    parser.add_argument("--beatthis-float16", action="store_true")
    parser.add_argument("--grid-fitter-cache", type=Path, default=DEFAULT_GRID_FITTER_CACHE_PATH)
    parser.add_argument("--grid-fitter-cache-meta", type=Path, default=DEFAULT_GRID_FITTER_CACHE_META_PATH)
    parser.add_argument("--adapter-cache", type=Path, default=DEFAULT_ADAPTER_CACHE_PATH)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON_PATH)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD_PATH)
    parser.add_argument("--max-maps", type=int, default=None)
    parser.add_argument("--config-preset", choices=("primary", "primary_plus_weak", "weak", "standard"), default="primary")
    parser.add_argument("--progress-every", type=int, default=250)
    args = parser.parse_args(argv)

    command = "uv run python -m train.stage_2.events.audit_high_precision_grid_adapter " + " ".join(
        sys.argv[1:] if argv is None else argv
    )
    payload = audit_high_precision_grid_adapter(
        index_path=args.index_path,
        dataset_root=args.dataset_root,
        grid_source=args.grid_source,
        beatthis_checkpoint=args.beatthis_checkpoint,
        beatthis_device=args.beatthis_device,
        beatthis_float16=args.beatthis_float16,
        grid_fitter_cache_path=args.grid_fitter_cache,
        grid_fitter_cache_meta_path=args.grid_fitter_cache_meta,
        adapter_cache_path=args.adapter_cache,
        output_json_path=args.output_json,
        output_md_path=args.output_md,
        max_maps=args.max_maps,
        config_preset=args.config_preset,
        progress_every=args.progress_every,
        audit_command=command,
    )
    primary_key = PRIMARY_CONFIG_NAME if PRIMARY_CONFIG_NAME in payload["adapter_coverage"] else next(iter(payload["adapter_coverage"]))
    primary = payload["adapter_coverage"][primary_key]
    print(f"audited_beatmap_count {payload['dataset']['beatmaps_audited']}")
    print(f"primitive_events_audited {payload['dataset']['primitive_events_audited']}")
    print(f"unique_raw_event_times_audited {payload['dataset']['unique_raw_event_times_audited']}")
    print(f"primary_window_success_rate {primary['window_8s_adapter_success_rate']:.9f}")
    print(
        "can_stage2_mapper_v1_use_grid_only_time_shift "
        f"{payload['recommendation']['can_stage2_mapper_v1_use_grid_only_time_shift']}"
    )
    print(f"recommended_D_main_2to6 {payload['recommendation']['recommended_D_main_2to6']}")
    print(f"recommended_k_max {payload['recommendation']['recommended_k_max']}")
    print(f"output_json {args.output_json}")
    print(f"output_md {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
