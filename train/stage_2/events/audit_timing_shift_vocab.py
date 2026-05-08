from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import subprocess
import sys
import time
from array import array
from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from train.stage1_oracle.osu.hitobjects import ManiaHitObjectKind, parse_mania_hit_objects
from train.stage_2.osu_core.timing import InvalidRedTimingError, MissingRedTimingError, require_red_timing_points


DEFAULT_INDEX_PATH = Path(
    "train/artifacts/indexes/"
    "beatmap_index_4k_no_timing_anomalies_2to6_dense_local_bpm_norm_unique_le3.parquet"
)
DEFAULT_DATASET_ROOT = Path("mania-dataset")
DEFAULT_OUTPUT_JSON_PATH = Path("train/artifacts/reports/events/timing_shift_vocab_audit.json")
DEFAULT_OUTPUT_MD_PATH = Path("train/artifacts/reports/events/timing_shift_vocab_audit.md")

ACTION_HOLD_END = "HOLD_END"
ACTION_TAP = "TAP"
ACTION_HOLD_START = "HOLD_START"
ACTION_ORDER = {ACTION_HOLD_END: 0, ACTION_TAP: 1, ACTION_HOLD_START: 2}

CANDIDATE_DIVISOR_SETS: dict[str, tuple[int, ...]] = {
    "A_conservative": (1, 2, 3, 4, 6, 8, 12, 16),
    "B_medium": (1, 2, 3, 4, 6, 8, 12, 16, 24, 32),
    "C_wide": (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64),
    "D_very_wide": (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192),
}
K_MAX_VALUES = (16, 32, 64, 128)
SNAP_TOLERANCES_MS = (0.5, 1.0, 2.0, 5.0)
EXACT_EPSILON_MS = 1.0e-7
FLOAT_AMBIGUITY_EPSILON_MS = 1.0e-6
WINDOW_LENGTH_MS = 8000
SPECIAL_VOCAB_SIZE = 2
END_VOCAB_SIZE = 1
EVENT_VOCAB_SIZE = 4 * 3
WORST_EXAMPLE_LIMIT = 12


@dataclass(frozen=True)
class TimingSection:
    offset_ms: float
    beat_length_ms: float
    meter: int = 4

    @property
    def bpm(self) -> float:
        return 60000.0 / self.beat_length_ms


@dataclass(frozen=True)
class BeatmapEvent:
    time_ms: int
    lane: int
    action: str

    @property
    def sort_key(self) -> tuple[int, int, int]:
        return (self.time_ms, ACTION_ORDER[self.action], self.lane)


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


@dataclass(frozen=True)
class GridMatch:
    before_first_red: bool
    residual_ms: float
    signed_residual_ms: float
    nearest_tick_ms: float | None
    divisor: int | None
    section_index: int | None
    tick_index: int | None
    float_precision_ambiguous: bool

    @property
    def exact(self) -> bool:
        return (not self.before_first_red) and self.residual_ms <= EXACT_EPSILON_MS

    @property
    def integer_ms_roundtrip(self) -> bool:
        if self.nearest_tick_ms is None:
            return False
        original_time_ms = self.nearest_tick_ms + self.signed_residual_ms
        return _round_half_up(self.nearest_tick_ms) == _round_half_up(original_time_ms)


@dataclass
class CoverageAccumulator:
    config_name: str
    divisors: tuple[int, ...]
    event_count: int = 0
    exact_count: int = 0
    integer_roundtrip_count: int = 0
    before_first_red_count: int = 0
    float_precision_ambiguous_count: int = 0
    failed_beatmap_count: int = 0
    residual_sum_ms: float = 0.0
    max_residual_ms: float = 0.0
    residuals_ms: array = None  # type: ignore[assignment]
    tolerance_counts: Counter[str] = None  # type: ignore[assignment]
    action_counts: Counter[str] = None  # type: ignore[assignment]
    action_exact_counts: Counter[str] = None  # type: ignore[assignment]
    worst_examples: list[tuple[float, int, dict[str, Any]]] = None  # type: ignore[assignment]
    _example_counter: int = 0

    def __post_init__(self) -> None:
        if self.residuals_ms is None:
            self.residuals_ms = array("f")
        if self.tolerance_counts is None:
            self.tolerance_counts = Counter()
        if self.action_counts is None:
            self.action_counts = Counter()
        if self.action_exact_counts is None:
            self.action_exact_counts = Counter()
        if self.worst_examples is None:
            self.worst_examples = []

    def add(
        self,
        *,
        event: BeatmapEvent,
        match: GridMatch,
        beatmap_path: Path,
        section: TimingSection | None,
    ) -> bool:
        self.event_count += 1
        self.action_counts[event.action] += 1
        residual = float(match.residual_ms)
        if math.isfinite(residual):
            self.residual_sum_ms += residual
            self.max_residual_ms = max(self.max_residual_ms, residual)
            self.residuals_ms.append(residual)
        if match.before_first_red:
            self.before_first_red_count += 1
        if match.float_precision_ambiguous:
            self.float_precision_ambiguous_count += 1
        if match.exact:
            self.exact_count += 1
            self.action_exact_counts[event.action] += 1
        if match.integer_ms_roundtrip:
            self.integer_roundtrip_count += 1
        for tolerance in SNAP_TOLERANCES_MS:
            if residual <= tolerance:
                self.tolerance_counts[_tolerance_key(tolerance)] += 1

        if not match.exact and not match.before_first_red:
            self._push_worst_example(
                {
                    "beatmap_path": beatmap_path.as_posix(),
                    "time_ms": event.time_ms,
                    "lane": event.lane,
                    "action": event.action,
                    "nearest_tick_ms": match.nearest_tick_ms,
                    "residual_ms": match.signed_residual_ms,
                    "abs_residual_ms": match.residual_ms,
                    "divisor": match.divisor,
                    "tick_index": match.tick_index,
                    "section_index": match.section_index,
                    "section_offset_ms": None if section is None else section.offset_ms,
                    "beat_length_ms": None if section is None else section.beat_length_ms,
                    "bpm": None if section is None else section.bpm,
                }
            )
        return match.exact

    def add_failed_beatmap(self) -> None:
        self.failed_beatmap_count += 1

    def _push_worst_example(self, example: dict[str, Any]) -> None:
        self._example_counter += 1
        residual = float(example["abs_residual_ms"])
        item = (residual, self._example_counter, example)
        if len(self.worst_examples) < WORST_EXAMPLE_LIMIT:
            heapq.heappush(self.worst_examples, item)
        elif residual > self.worst_examples[0][0]:
            heapq.heapreplace(self.worst_examples, item)

    def to_json(self, beatmap_count: int) -> dict[str, Any]:
        residual_stats = _array_stats(self.residuals_ms)
        exact_by_action = {}
        for action, count in sorted(self.action_counts.items()):
            exact = self.action_exact_counts[action]
            exact_by_action[action] = {
                "event_count": count,
                "exact_count": exact,
                "exact_rate": _rate(exact, count),
            }
        return {
            "divisors": list(self.divisors),
            "event_count": self.event_count,
            "exact_real_count": self.exact_count,
            "exact_real_rate": _rate(self.exact_count, self.event_count),
            "failed_event_count": self.event_count - self.exact_count,
            "failed_beatmap_count": self.failed_beatmap_count,
            "failed_beatmap_rate": _rate(self.failed_beatmap_count, beatmap_count),
            "integer_ms_roundtrip_count": self.integer_roundtrip_count,
            "integer_ms_roundtrip_rate": _rate(self.integer_roundtrip_count, self.event_count),
            "before_first_red_count": self.before_first_red_count,
            "float_precision_ambiguous_count": self.float_precision_ambiguous_count,
            "tolerance_counts": dict(sorted(self.tolerance_counts.items())),
            "tolerance_rates": {
                key: _rate(value, self.event_count) for key, value in sorted(self.tolerance_counts.items())
            },
            "residual_ms": residual_stats,
            "exact_by_action": exact_by_action,
            "worst_examples": [
                item[2] for item in sorted(self.worst_examples, key=lambda entry: (-entry[0], entry[1]))
            ],
        }


@dataclass
class TokenCostAccumulator:
    label: str
    map_sequence_tokens: list[int] = None  # type: ignore[assignment]
    map_time_shift_tokens: list[int] = None  # type: ignore[assignment]
    window_sequence_tokens: list[int] = None  # type: ignore[assignment]
    window_time_shift_tokens: list[int] = None  # type: ignore[assignment]
    failed_event_count: int = 0
    failed_beatmap_count: int = 0
    failed_window_count: int = 0
    failed_delta_count: int = 0
    crossing_delta_count: int = 0
    max_time_shift_tokens_for_delta: int = 0
    failure_types: Counter[str] = None  # type: ignore[assignment]
    worst_deltas: list[tuple[int, int, dict[str, Any]]] = None  # type: ignore[assignment]
    _example_counter: int = 0

    def __post_init__(self) -> None:
        if self.map_sequence_tokens is None:
            self.map_sequence_tokens = []
        if self.map_time_shift_tokens is None:
            self.map_time_shift_tokens = []
        if self.window_sequence_tokens is None:
            self.window_sequence_tokens = []
        if self.window_time_shift_tokens is None:
            self.window_time_shift_tokens = []
        if self.failure_types is None:
            self.failure_types = Counter()
        if self.worst_deltas is None:
            self.worst_deltas = []

    def add_map_success(self, event_count: int, time_shift_tokens: int) -> None:
        self.map_time_shift_tokens.append(time_shift_tokens)
        self.map_sequence_tokens.append(event_count + time_shift_tokens + 2)

    def add_window_success(self, event_count: int, time_shift_tokens: int) -> None:
        self.window_time_shift_tokens.append(time_shift_tokens)
        self.window_sequence_tokens.append(event_count + time_shift_tokens + 2)

    def add_failure(self, failure_type: str, *, event_count: int = 0, beatmap: bool = False, window: bool = False) -> None:
        self.failure_types[failure_type] += 1
        if not window:
            self.failed_event_count += event_count
        if beatmap:
            self.failed_beatmap_count += 1
        if window:
            self.failed_window_count += 1

    def observe_delta(self, token_count: int, example: dict[str, Any]) -> None:
        self.max_time_shift_tokens_for_delta = max(self.max_time_shift_tokens_for_delta, token_count)
        self._example_counter += 1
        item = (token_count, self._example_counter, example)
        if len(self.worst_deltas) < WORST_EXAMPLE_LIMIT:
            heapq.heappush(self.worst_deltas, item)
        elif token_count > self.worst_deltas[0][0]:
            heapq.heapreplace(self.worst_deltas, item)

    def to_json(self, *, event_count: int, beatmap_count: int, window_count: int, vocab_size: int) -> dict[str, Any]:
        encoded_event_count = max(0, event_count - self.failed_event_count)
        return {
            "vocab_size": vocab_size,
            "coverage_event_count": encoded_event_count,
            "coverage_rate": _rate(encoded_event_count, event_count),
            "failed_event_count": self.failed_event_count,
            "failed_beatmap_count": self.failed_beatmap_count,
            "failed_beatmap_rate": _rate(self.failed_beatmap_count, beatmap_count),
            "failed_window_count": self.failed_window_count,
            "failed_window_rate": _rate(self.failed_window_count, window_count),
            "failed_delta_count": self.failed_delta_count,
            "crossing_delta_count": self.crossing_delta_count,
            "successful_map_count": len(self.map_sequence_tokens),
            "successful_window_count": len(self.window_sequence_tokens),
            "map_sequence_tokens": _number_list_stats(self.map_sequence_tokens),
            "map_time_shift_tokens": _number_list_stats(self.map_time_shift_tokens),
            "window_8s_sequence_tokens": _number_list_stats(self.window_sequence_tokens),
            "window_8s_time_shift_tokens": _number_list_stats(self.window_time_shift_tokens),
            "max_time_shift_tokens_for_single_delta": self.max_time_shift_tokens_for_delta,
            "failure_types": dict(sorted(self.failure_types.items())),
            "worst_deltas": [
                item[2] for item in sorted(self.worst_deltas, key=lambda entry: (-entry[0], entry[1]))
            ],
        }


def best_grid_match(
    time_ms: float,
    sections: Sequence[TimingSection],
    divisors: Sequence[int],
    *,
    exact_epsilon_ms: float = EXACT_EPSILON_MS,
    float_ambiguity_epsilon_ms: float = FLOAT_AMBIGUITY_EPSILON_MS,
) -> GridMatch:
    if not sections:
        raise ValueError("sections must be non-empty")
    section_index = _section_index_at(sections, time_ms)
    if section_index < 0:
        return GridMatch(
            before_first_red=True,
            residual_ms=math.inf,
            signed_residual_ms=math.inf,
            nearest_tick_ms=None,
            divisor=None,
            section_index=None,
            tick_index=None,
            float_precision_ambiguous=False,
        )

    section = sections[section_index]
    best: tuple[float, int, float, int] | None = None
    for divisor in divisors:
        step_ms = section.beat_length_ms / divisor
        tick_index = int(round((time_ms - section.offset_ms) / step_ms))
        nearest_tick_ms = section.offset_ms + tick_index * step_ms
        signed_residual_ms = time_ms - nearest_tick_ms
        residual_ms = abs(signed_residual_ms)
        candidate = (residual_ms, divisor, nearest_tick_ms, tick_index)
        if best is None or candidate < best:
            best = candidate

    assert best is not None
    residual_ms, divisor, nearest_tick_ms, tick_index = best
    signed_residual_ms = time_ms - nearest_tick_ms
    return GridMatch(
        before_first_red=False,
        residual_ms=float(residual_ms),
        signed_residual_ms=float(signed_residual_ms),
        nearest_tick_ms=float(nearest_tick_ms),
        divisor=int(divisor),
        section_index=section_index,
        tick_index=int(tick_index),
        float_precision_ambiguous=exact_epsilon_ms < residual_ms <= float_ambiguity_epsilon_ms,
    )


def _divisor_grid_matches(
    time_ms: float,
    section: TimingSection | None,
    divisors: Sequence[int],
) -> dict[int, tuple[float, float, float, int]]:
    if section is None:
        return {}
    matches: dict[int, tuple[float, float, float, int]] = {}
    for divisor in divisors:
        step_ms = section.beat_length_ms / divisor
        tick_index = int(round((time_ms - section.offset_ms) / step_ms))
        nearest_tick_ms = section.offset_ms + tick_index * step_ms
        signed_residual_ms = time_ms - nearest_tick_ms
        residual_ms = abs(signed_residual_ms)
        matches[divisor] = (residual_ms, signed_residual_ms, nearest_tick_ms, tick_index)
    return matches


def _best_grid_match_from_divisor_matches(
    section_index: int,
    divisor_matches: Mapping[int, tuple[float, float, float, int]],
    divisors: Sequence[int],
) -> GridMatch:
    if section_index < 0:
        return GridMatch(
            before_first_red=True,
            residual_ms=math.inf,
            signed_residual_ms=math.inf,
            nearest_tick_ms=None,
            divisor=None,
            section_index=None,
            tick_index=None,
            float_precision_ambiguous=False,
        )
    best: tuple[float, int, float, int, float] | None = None
    for divisor in divisors:
        residual_ms, signed_residual_ms, nearest_tick_ms, tick_index = divisor_matches[divisor]
        candidate = (residual_ms, divisor, nearest_tick_ms, tick_index, signed_residual_ms)
        if best is None or candidate < best:
            best = candidate
    assert best is not None
    residual_ms, divisor, nearest_tick_ms, tick_index, signed_residual_ms = best
    return GridMatch(
        before_first_red=False,
        residual_ms=float(residual_ms),
        signed_residual_ms=float(signed_residual_ms),
        nearest_tick_ms=float(nearest_tick_ms),
        divisor=int(divisor),
        section_index=section_index,
        tick_index=int(tick_index),
        float_precision_ambiguous=EXACT_EPSILON_MS < residual_ms <= FLOAT_AMBIGUITY_EPSILON_MS,
    )


def decompose_interval_to_grid_tokens(
    start_ms: float,
    end_ms: float,
    sections: Sequence[TimingSection],
    divisors: Sequence[int],
    *,
    k_max: int,
    exact_epsilon_ms: float = EXACT_EPSILON_MS,
) -> DecompositionResult:
    if end_ms < start_ms:
        return DecompositionResult(ok=False, failure_type="negative_delta")
    if abs(end_ms - start_ms) <= exact_epsilon_ms:
        return DecompositionResult(ok=True)
    if not sections:
        return DecompositionResult(ok=False, failure_type="no_red_timing_points")

    tokens: list[GridToken] = []
    points = [start_ms]
    points.extend(section.offset_ms for section in sections if start_ms < section.offset_ms < end_ms)
    points.append(end_ms)

    for segment_start, segment_end in zip(points[:-1], points[1:], strict=True):
        section_index = _section_index_at(sections, segment_start)
        if section_index < 0:
            return DecompositionResult(ok=False, failure_type="event_before_first_red")
        section = sections[section_index]
        segment = _decompose_single_section_interval(
            segment_start,
            segment_end,
            section_index,
            section,
            divisors,
            k_max=k_max,
            exact_epsilon_ms=exact_epsilon_ms,
        )
        if not segment.ok:
            failure_type = segment.failure_type or "unknown_decomposition_failure"
            if segment_end in [point.offset_ms for point in sections[1:]]:
                failure_type = "crossing_timing_section"
            return DecompositionResult(ok=False, failure_type=failure_type)
        tokens.extend(segment.tokens)
    return DecompositionResult(ok=True, tokens=tuple(tokens))


def audit_timing_shift_vocab(
    *,
    index_path: str | Path = DEFAULT_INDEX_PATH,
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    output_json_path: str | Path = DEFAULT_OUTPUT_JSON_PATH,
    output_md_path: str | Path = DEFAULT_OUTPUT_MD_PATH,
    candidate_divisor_sets: Mapping[str, Sequence[int]] = CANDIDATE_DIVISOR_SETS,
    k_max_values: Sequence[int] = K_MAX_VALUES,
    max_maps: int | None = None,
    progress_every: int = 250,
    audit_command: str | None = None,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    index_path = Path(index_path)
    dataset_root = Path(dataset_root)
    output_json_path = Path(output_json_path)
    output_md_path = Path(output_md_path)
    candidate_divisor_sets = {
        name: tuple(_validate_divisors(divisors)) for name, divisors in candidate_divisor_sets.items()
    }
    all_candidate_divisors = tuple(sorted({divisor for divisors in candidate_divisor_sets.values() for divisor in divisors}))
    k_max_values = tuple(int(value) for value in k_max_values)

    index_df = _load_unique_beatmap_index(index_path)
    if max_maps is not None:
        index_df = index_df.head(max_maps).copy()

    coverage = {
        name: CoverageAccumulator(config_name=name, divisors=tuple(divisors))
        for name, divisors in candidate_divisor_sets.items()
    }
    grid_token_cost = {
        f"{name}_k{k_max}": TokenCostAccumulator(label=f"{name}_k{k_max}")
        for name in candidate_divisor_sets
        for k_max in k_max_values
    }
    fallback_token_cost = {
        f"1ms_k{k_max}": TokenCostAccumulator(label=f"1ms_k{k_max}") for k_max in k_max_values
    }
    baseline = _new_baseline_counters()
    failure_taxonomy = Counter()
    parse_failure_examples: list[dict[str, Any]] = []

    audited_beatmap_count = 0
    total_hitobject_count = 0
    total_event_count = 0
    total_window_count = 0
    total_unique_event_time_count = 0
    event_to_event_delta_count = 0
    event_to_event_crossing_delta_count = 0
    duplicate_same_lane_event_count = 0
    same_time_same_lane_mixed_action_count = 0
    hold_end_before_start_count = 0
    zero_length_hold_count = 0

    for row_number, row in enumerate(index_df.itertuples(index=False), start=1):
        beatmap_path = dataset_root / str(row.shard) / str(row.beatmap_path)
        try:
            raw_sections = require_red_timing_points(beatmap_path)
            sections = tuple(
                TimingSection(
                    offset_ms=float(point.offset_ms),
                    beat_length_ms=float(point.beat_length_ms),
                    meter=int(getattr(point, "meter", 4)),
                )
                for point in raw_sections
            )
            section_offsets = tuple(section.offset_ms for section in sections)
            hitobjects = parse_mania_hit_objects(beatmap_path, expected_key_count=4)
        except MissingRedTimingError:
            failure_taxonomy["no_red_timing_points"] += 1
            _append_parse_failure_example(parse_failure_examples, beatmap_path, "no_red_timing_points")
            continue
        except InvalidRedTimingError as exc:
            failure_taxonomy["invalid_red_timing_points"] += 1
            _append_parse_failure_example(parse_failure_examples, beatmap_path, str(exc))
            continue
        except ValueError as exc:
            message = str(exc)
            failure_type = _parser_failure_type(message)
            failure_taxonomy[failure_type] += 1
            _append_parse_failure_example(parse_failure_examples, beatmap_path, message)
            continue

        audited_beatmap_count += 1
        total_hitobject_count += len(hitobjects)
        events: list[BeatmapEvent] = []
        for hitobject in hitobjects:
            start_ms = _integer_osu_time(hitobject.start_time_ms)
            if start_ms is None:
                failure_taxonomy["unknown_parser_issue"] += 1
                continue
            lane = int(hitobject.lane)
            if lane < 0 or lane >= 4:
                failure_taxonomy["unsupported_key_count"] += 1
                continue
            if hitobject.kind == ManiaHitObjectKind.TAP:
                events.append(BeatmapEvent(time_ms=start_ms, lane=lane, action=ACTION_TAP))
                continue

            end_ms = _integer_osu_time(hitobject.end_time_ms)
            if end_ms is None:
                failure_taxonomy["unknown_parser_issue"] += 1
                continue
            if end_ms < start_ms:
                hold_end_before_start_count += 1
            if end_ms == start_ms:
                zero_length_hold_count += 1
            events.append(BeatmapEvent(time_ms=start_ms, lane=lane, action=ACTION_HOLD_START))
            events.append(BeatmapEvent(time_ms=end_ms, lane=lane, action=ACTION_HOLD_END))

        events.sort(key=lambda event: event.sort_key)
        total_event_count += len(events)
        if not events:
            continue

        duplicate_same_lane_event_count += _duplicate_same_lane_event_count(events)
        same_time_same_lane_mixed_action_count += _same_time_same_lane_mixed_action_count(events)

        unique_event_times = _unique_sorted_times(events)
        total_unique_event_time_count += len(unique_event_times)
        event_to_event_delta_count += max(0, len(unique_event_times) - 1)
        crossing_delta_count = _crossing_delta_count(unique_event_times, sections)
        event_to_event_crossing_delta_count += crossing_delta_count

        window_events = _events_by_window(events)
        total_window_count += len(window_events)

        per_config_event_exact: dict[str, list[bool]] = {name: [] for name in candidate_divisor_sets}
        per_config_map_exact = {name: True for name in candidate_divisor_sets}

        for event in events:
            section_index = _section_index_at_offsets(section_offsets, event.time_ms)
            section = None if section_index < 0 else sections[section_index]
            divisor_matches = _divisor_grid_matches(event.time_ms, section, all_candidate_divisors)
            for name, divisors in candidate_divisor_sets.items():
                match = _best_grid_match_from_divisor_matches(section_index, divisor_matches, divisors)
                exact = coverage[name].add(
                    event=event,
                    match=match,
                    beatmap_path=beatmap_path,
                    section=section,
                )
                per_config_event_exact[name].append(exact)
                if not exact:
                    per_config_map_exact[name] = False

            _update_baselines(baseline, event.time_ms)

        for name, is_exact in per_config_map_exact.items():
            if not is_exact:
                coverage[name].add_failed_beatmap()

        for k_max, accumulator in fallback_token_cost.items():
            numeric_k_max = int(k_max.rsplit("k", 1)[1])
            shift_tokens, max_delta_tokens, worst_delta = _fallback_time_shift_tokens(unique_event_times, numeric_k_max)
            accumulator.add_map_success(len(events), shift_tokens)
            if worst_delta is not None:
                accumulator.observe_delta(max_delta_tokens, _delta_example(beatmap_path, worst_delta, max_delta_tokens))
            for window_index, window_event_list in window_events.items():
                window_times = _unique_sorted_times(window_event_list)
                window_shift_tokens, window_max_delta_tokens, window_worst_delta = _fallback_time_shift_tokens(
                    window_times,
                    numeric_k_max,
                )
                accumulator.add_window_success(len(window_event_list), window_shift_tokens)
                if window_worst_delta is not None:
                    accumulator.observe_delta(
                        window_max_delta_tokens,
                        _delta_example(beatmap_path, window_worst_delta, window_max_delta_tokens, window_index),
                    )

        for config_name, divisors in candidate_divisor_sets.items():
            exact_by_event = per_config_event_exact[config_name]
            for k_max in k_max_values:
                label = f"{config_name}_k{k_max}"
                accumulator = grid_token_cost[label]
                accumulator.crossing_delta_count += crossing_delta_count
                if not per_config_map_exact[config_name]:
                    failed_events = len(events) - sum(1 for value in exact_by_event if value)
                    accumulator.add_failure("event_not_on_grid_for_divisor_set", event_count=failed_events, beatmap=True)
                else:
                    result = _grid_time_shift_tokens(
                        unique_event_times,
                        sections,
                        divisors,
                        k_max=k_max,
                        beatmap_path=beatmap_path,
                    )
                    if result["ok"]:
                        accumulator.add_map_success(len(events), int(result["time_shift_tokens"]))
                        worst = result.get("worst_delta")
                        if worst is not None:
                            accumulator.observe_delta(int(worst["token_count"]), worst)
                    else:
                        accumulator.failed_delta_count += 1
                        accumulator.add_failure(str(result["failure_type"]), event_count=len(events), beatmap=True)

                _audit_grid_windows(
                    accumulator,
                    beatmap_path=beatmap_path,
                    window_events=window_events,
                    all_events=events,
                    exact_by_event=exact_by_event,
                    sections=sections,
                    divisors=divisors,
                    k_max=k_max,
                )

        if progress_every > 0 and (row_number == 1 or row_number % progress_every == 0):
            print(
                f"timing_shift_vocab_audit progress maps={row_number}/{len(index_df)} "
                f"audited={audited_beatmap_count} events={total_event_count}",
                file=sys.stderr,
                flush=True,
            )

    baseline_report = _baseline_report(baseline, total_event_count)
    coverage_report = {
        name: accumulator.to_json(audited_beatmap_count) for name, accumulator in coverage.items()
    }
    token_report = {
        label: accumulator.to_json(
            event_count=total_event_count,
            beatmap_count=audited_beatmap_count,
            window_count=total_window_count,
            vocab_size=_grid_vocab_size(
                len(candidate_divisor_sets[label.rsplit("_k", 1)[0]]),
                int(label.rsplit("_k", 1)[1]),
            ),
        )
        for label, accumulator in sorted(grid_token_cost.items())
    }
    fallback_token_report = {
        label: accumulator.to_json(
            event_count=total_event_count,
            beatmap_count=audited_beatmap_count,
            window_count=total_window_count,
            vocab_size=_fallback_vocab_size(int(label.rsplit("k", 1)[1])),
        )
        for label, accumulator in sorted(fallback_token_cost.items())
    }

    failure_taxonomy["duplicate_same_lane_event_at_same_timestamp"] = duplicate_same_lane_event_count
    failure_taxonomy["same_time_invalid_lane_action_ordering"] = same_time_same_lane_mixed_action_count
    failure_taxonomy["event_to_event_crossing_timing_section_delta"] = event_to_event_crossing_delta_count
    failure_taxonomy["hold_end_before_hold_start"] += hold_end_before_start_count
    failure_taxonomy["malformed_zero_length_hold"] += zero_length_hold_count

    payload: dict[str, Any] = {
        "schema_version": 1,
        "audit_name": "stage2_timing_shift_vocab_audit",
        "provenance": {
            "commit": _git_rev_parse("HEAD"),
            "index_path": index_path.as_posix(),
            "index_sha256": _sha256_file(index_path),
            "dataset_root": dataset_root.as_posix(),
            "audit_command": audit_command or " ".join(sys.argv),
            "started_at_unix": started_at,
            "elapsed_s": time.perf_counter() - started_at,
            "exact_epsilon_ms": EXACT_EPSILON_MS,
            "float_ambiguity_epsilon_ms": FLOAT_AMBIGUITY_EPSILON_MS,
            "window_length_ms": WINDOW_LENGTH_MS,
            "note": (
                "exact_real_count requires the integer osu timestamp to equal a red-timing-grid tick "
                "within exact_epsilon_ms. integer_ms_roundtrip_count only means the nearest tick rounds "
                "back to the same integer millisecond and is not treated as lossless grid exactness."
            ),
        },
        "dataset": {
            "index_row_count_after_dedupe": len(index_df),
            "audited_beatmap_count": audited_beatmap_count,
            "total_hitobject_count": total_hitobject_count,
            "total_event_count": total_event_count,
            "total_unique_event_time_count": total_unique_event_time_count,
            "total_8s_window_count": total_window_count,
            "event_to_event_delta_count": event_to_event_delta_count,
            "event_to_event_crossing_timing_section_delta_count": event_to_event_crossing_delta_count,
        },
        "candidate_divisor_sets": {name: list(divisors) for name, divisors in candidate_divisor_sets.items()},
        "k_max_values": list(k_max_values),
        "event_time_coverage": coverage_report,
        "baseline_coverage": baseline_report,
        "grid_token_configs": token_report,
        "fallback_1ms_token_configs": fallback_token_report,
        "failure_taxonomy": dict(sorted(failure_taxonomy.items())),
        "parse_failure_examples": parse_failure_examples,
        "recommendation": _build_recommendation(
            coverage_report=coverage_report,
            token_report=token_report,
            fallback_token_report=fallback_token_report,
            total_event_count=total_event_count,
        ),
    }

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_md_path.parent.mkdir(parents=True, exist_ok=True)
    output_md_path.write_text(_render_markdown(payload), encoding="utf-8")
    return payload


def _decompose_single_section_interval(
    start_ms: float,
    end_ms: float,
    section_index: int,
    section: TimingSection,
    divisors: Sequence[int],
    *,
    k_max: int,
    exact_epsilon_ms: float,
) -> DecompositionResult:
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

        token_start_ms = section.offset_ms + current_units * section.beat_length_ms / lcm
        token_end_ms = section.offset_ms + landing_units * section.beat_length_ms / lcm
        tokens.append(
            GridToken(
                section_index=section_index,
                divisor=divisor,
                k=k,
                start_ms=token_start_ms,
                end_ms=token_end_ms,
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
        if target_units % step_units != 0:
            continue
        floor_index = current_units // step_units
        target_index = target_units // step_units
        k = target_index - floor_index
        if 1 <= k <= k_max and target_units > current_units:
            candidates.append((divisor, -k, k, target_units))
    if not candidates:
        return None
    divisor, _, k, landing_units = min(candidates)
    return divisor, k, landing_units


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
    neg_landing, divisor, _, k = min(candidates)
    return divisor, k, -neg_landing


def _time_to_common_units(
    time_ms: float,
    section: TimingSection,
    lcm: int,
    *,
    exact_epsilon_ms: float,
) -> int | None:
    raw_units = (time_ms - section.offset_ms) * lcm / section.beat_length_ms
    units = int(round(raw_units))
    reconstructed = section.offset_ms + units * section.beat_length_ms / lcm
    if abs(time_ms - reconstructed) > exact_epsilon_ms:
        return None
    return units


def _load_unique_beatmap_index(index_path: Path) -> pd.DataFrame:
    index_df = pd.read_parquet(index_path)
    required_columns = {"shard", "beatmap_path", "difficulty"}
    missing = sorted(required_columns.difference(index_df.columns))
    if missing:
        raise ValueError(f"{index_path} is missing required column(s): {missing}")
    filtered = index_df[(index_df["difficulty"] >= 2.0) & (index_df["difficulty"] <= 6.0)].copy()
    return filtered.drop_duplicates(["shard", "beatmap_path"]).reset_index(drop=True)


def _validate_divisors(divisors: Sequence[int]) -> tuple[int, ...]:
    cleaned = tuple(sorted({int(divisor) for divisor in divisors}))
    if not cleaned or any(divisor <= 0 for divisor in cleaned):
        raise ValueError(f"divisors must be positive integers: {divisors!r}")
    return cleaned


def _section_index_at(sections: Sequence[TimingSection], time_ms: float) -> int:
    offsets = [section.offset_ms for section in sections]
    return _section_index_at_offsets(offsets, time_ms)


def _section_index_at_offsets(offsets: Sequence[float], time_ms: float) -> int:
    return bisect_right(offsets, time_ms) - 1


def _integer_osu_time(value: float) -> int | None:
    rounded = int(round(float(value)))
    if abs(float(value) - rounded) > 1.0e-6:
        return None
    return rounded


def _unique_sorted_times(events: Sequence[BeatmapEvent]) -> list[int]:
    return sorted({event.time_ms for event in events})


def _events_by_window(events: Sequence[BeatmapEvent]) -> dict[int, list[BeatmapEvent]]:
    windows: dict[int, list[BeatmapEvent]] = defaultdict(list)
    for event in events:
        windows[event.time_ms // WINDOW_LENGTH_MS].append(event)
    return dict(windows)


def _crossing_delta_count(unique_times: Sequence[int], sections: Sequence[TimingSection]) -> int:
    if len(unique_times) < 2:
        return 0
    offsets = [section.offset_ms for section in sections]
    count = 0
    for start, end in zip(unique_times[:-1], unique_times[1:], strict=True):
        left = bisect_right(offsets, start)
        right = bisect_right(offsets, end - EXACT_EPSILON_MS)
        if right > left:
            count += 1
    return count


def _duplicate_same_lane_event_count(events: Sequence[BeatmapEvent]) -> int:
    counts = Counter((event.time_ms, event.lane, event.action) for event in events)
    return sum(count - 1 for count in counts.values() if count > 1)


def _same_time_same_lane_mixed_action_count(events: Sequence[BeatmapEvent]) -> int:
    actions_by_cell: dict[tuple[int, int], set[str]] = defaultdict(set)
    for event in events:
        actions_by_cell[(event.time_ms, event.lane)].add(event.action)
    return sum(1 for actions in actions_by_cell.values() if len(actions) > 1)


def _new_baseline_counters() -> dict[str, Counter[str]]:
    return {
        "pure_ms": Counter(),
        "10ms": Counter(),
        "5ms": Counter(),
        "1ms": Counter(),
    }


def _update_baselines(baseline: dict[str, Counter[str]], time_ms: int) -> None:
    baseline["pure_ms"]["exact"] += 1
    baseline["1ms"]["exact"] += 1
    if time_ms % 10 == 0:
        baseline["10ms"]["exact"] += 1
    if time_ms % 5 == 0:
        baseline["5ms"]["exact"] += 1


def _baseline_report(baseline: dict[str, Counter[str]], event_count: int) -> dict[str, Any]:
    return {
        "E_pure_ms_integer_delta": {
            "event_count": event_count,
            "exact_count": baseline["pure_ms"]["exact"],
            "exact_rate": _rate(baseline["pure_ms"]["exact"], event_count),
            "note": "All parsed osu event timestamps are integer milliseconds; delta vocabulary size depends on max allowed K.",
        },
        "F_10ms_grid": {
            "event_count": event_count,
            "exact_count": baseline["10ms"]["exact"],
            "exact_rate": _rate(baseline["10ms"]["exact"], event_count),
            "failed_event_count": event_count - baseline["10ms"]["exact"],
        },
        "G_5ms_grid": {
            "event_count": event_count,
            "exact_count": baseline["5ms"]["exact"],
            "exact_rate": _rate(baseline["5ms"]["exact"], event_count),
            "failed_event_count": event_count - baseline["5ms"]["exact"],
        },
        "H_1ms_fallback": {
            "event_count": event_count,
            "exact_count": baseline["1ms"]["exact"],
            "exact_rate": _rate(baseline["1ms"]["exact"], event_count),
            "failed_event_count": 0,
        },
    }


def _fallback_time_shift_tokens(unique_times: Sequence[int], k_max: int) -> tuple[int, int, tuple[int, int] | None]:
    total = 0
    max_for_delta = 0
    worst_delta: tuple[int, int] | None = None
    for start, end in zip(unique_times[:-1], unique_times[1:], strict=True):
        delta = end - start
        if delta < 0:
            continue
        tokens = int(math.ceil(delta / k_max)) if delta else 0
        total += tokens
        if tokens > max_for_delta:
            max_for_delta = tokens
            worst_delta = (start, end)
    return total, max_for_delta, worst_delta


def _grid_time_shift_tokens(
    unique_times: Sequence[int],
    sections: Sequence[TimingSection],
    divisors: Sequence[int],
    *,
    k_max: int,
    beatmap_path: Path,
) -> dict[str, Any]:
    total = 0
    worst_delta: dict[str, Any] | None = None
    max_token_count = 0
    for start, end in zip(unique_times[:-1], unique_times[1:], strict=True):
        result = decompose_interval_to_grid_tokens(start, end, sections, divisors, k_max=k_max)
        if not result.ok:
            return {
                "ok": False,
                "failure_type": result.failure_type,
                "start_ms": start,
                "end_ms": end,
            }
        token_count = len(result.tokens)
        total += token_count
        if token_count > max_token_count:
            max_token_count = token_count
            worst_delta = _delta_example(beatmap_path, (start, end), token_count)
            worst_delta["tokens"] = [_token_json(token) for token in result.tokens[:24]]
    return {"ok": True, "time_shift_tokens": total, "worst_delta": worst_delta}


def _audit_grid_windows(
    accumulator: TokenCostAccumulator,
    *,
    beatmap_path: Path,
    window_events: Mapping[int, Sequence[BeatmapEvent]],
    all_events: Sequence[BeatmapEvent],
    exact_by_event: Sequence[bool],
    sections: Sequence[TimingSection],
    divisors: Sequence[int],
    k_max: int,
) -> None:
    exact_by_sort_key: dict[tuple[int, int, int], list[bool]] = defaultdict(list)
    for event, exact in zip(all_events, exact_by_event, strict=True):
        exact_by_sort_key[event.sort_key].append(exact)

    for window_index, events in window_events.items():
        failed_events = 0
        for event in events:
            values = exact_by_sort_key[event.sort_key]
            exact = values.pop(0)
            if not exact:
                failed_events += 1
        if failed_events:
            accumulator.add_failure("event_not_on_grid_for_divisor_set", event_count=failed_events, window=True)
            continue
        times = _unique_sorted_times(events)
        result = _grid_time_shift_tokens(times, sections, divisors, k_max=k_max, beatmap_path=beatmap_path)
        if result["ok"]:
            accumulator.add_window_success(len(events), int(result["time_shift_tokens"]))
            worst = result.get("worst_delta")
            if worst is not None:
                worst["window_index"] = window_index
                accumulator.observe_delta(int(worst["token_count"]), worst)
        else:
            accumulator.failed_delta_count += 1
            accumulator.add_failure(str(result["failure_type"]), event_count=len(events), window=True)


def _delta_example(
    beatmap_path: Path,
    delta: tuple[int, int],
    token_count: int,
    window_index: int | None = None,
) -> dict[str, Any]:
    start, end = delta
    payload = {
        "beatmap_path": beatmap_path.as_posix(),
        "start_ms": start,
        "end_ms": end,
        "delta_ms": end - start,
        "token_count": token_count,
    }
    if window_index is not None:
        payload["window_index"] = window_index
    return payload


def _token_json(token: GridToken) -> dict[str, Any]:
    return {
        "section_index": token.section_index,
        "divisor": token.divisor,
        "k": token.k,
        "start_ms": token.start_ms,
        "end_ms": token.end_ms,
    }


def _round_half_up(value: float) -> int:
    return int(math.floor(value + 0.5))


def _tolerance_key(tolerance: float) -> str:
    return f"le_{str(tolerance).replace('.', '_')}ms"


def _array_stats(values: array) -> dict[str, float | int | None]:
    if not values:
        return _empty_stats()
    data = np.frombuffer(values, dtype=np.float32)
    return _stats_from_numpy(data)


def _number_list_stats(values: Sequence[int]) -> dict[str, float | int | None]:
    if not values:
        return _empty_stats()
    data = np.asarray(values, dtype=np.float64)
    return _stats_from_numpy(data)


def _stats_from_numpy(data: np.ndarray) -> dict[str, float | int]:
    return {
        "count": int(data.size),
        "mean": float(np.mean(data)),
        "p50": float(np.percentile(data, 50)),
        "p95": float(np.percentile(data, 95)),
        "p99": float(np.percentile(data, 99)),
        "max": float(np.max(data)),
    }


def _empty_stats() -> dict[str, None | int]:
    return {"count": 0, "mean": None, "p50": None, "p95": None, "p99": None, "max": None}


def _rate(numerator: int | float, denominator: int | float) -> float:
    return 0.0 if denominator == 0 else float(numerator) / float(denominator)


def _grid_vocab_size(divisor_count: int, k_max: int) -> int:
    return SPECIAL_VOCAB_SIZE + END_VOCAB_SIZE + EVENT_VOCAB_SIZE + divisor_count * k_max


def _fallback_vocab_size(k_max: int) -> int:
    return SPECIAL_VOCAB_SIZE + END_VOCAB_SIZE + EVENT_VOCAB_SIZE + k_max


def _lcm_many(values: Iterable[int]) -> int:
    result = 1
    for value in values:
        result = math.lcm(result, int(value))
    return result


def _parser_failure_type(message: str) -> str:
    if "not a 4K" in message or "not a 4K" in message:
        return "unsupported_key_count"
    if "hold" in message.lower():
        return "malformed_hold"
    return "unknown_parser_issue"


def _append_parse_failure_example(examples: list[dict[str, Any]], beatmap_path: Path, reason: str) -> None:
    if len(examples) >= WORST_EXAMPLE_LIMIT:
        return
    examples.append({"beatmap_path": beatmap_path.as_posix(), "reason": reason})


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


def _build_recommendation(
    *,
    coverage_report: Mapping[str, Mapping[str, Any]],
    token_report: Mapping[str, Mapping[str, Any]],
    fallback_token_report: Mapping[str, Mapping[str, Any]],
    total_event_count: int,
) -> dict[str, Any]:
    best_exact_name, best_exact = max(
        coverage_report.items(),
        key=lambda item: (item[1]["exact_real_count"], item[1]["integer_ms_roundtrip_count"]),
    )
    grid_alone_lossless = best_exact["exact_real_count"] == total_event_count
    best_approx_name, best_approx = max(
        coverage_report.items(),
        key=lambda item: item[1]["tolerance_counts"].get("le_1_0ms", 0),
    )
    fallback_128 = fallback_token_report.get("1ms_k128", {})
    return {
        "grid_relative_time_shift_alone_lossless": grid_alone_lossless,
        "fine_ms_fallback_required": not grid_alone_lossless,
        "best_exact_divisor_set": best_exact_name,
        "best_exact_rate": best_exact["exact_real_rate"],
        "best_integer_ms_roundtrip_rate": best_exact["integer_ms_roundtrip_rate"],
        "best_le_1ms_approx_divisor_set": best_approx_name,
        "best_le_1ms_approx_rate": best_approx["tolerance_rates"].get("le_1_0ms", 0.0),
        "recommended_divisor_set_if_using_grid_component": "D_very_wide",
        "recommended_k_max": 128,
        "fallback_1ms_k128_window_p99_sequence_tokens": fallback_128.get("window_8s_sequence_tokens", {}).get("p99"),
        "summary": (
            "Use grid-relative TIME_SHIFT only as a component unless exact_real_rate reaches 1.0. "
            "The 1ms fallback is exact for integer osu timestamps and gives the conservative token-cost bound."
        ),
    }


def _render_markdown(payload: Mapping[str, Any]) -> str:
    provenance = payload["provenance"]
    dataset = payload["dataset"]
    coverage = payload["event_time_coverage"]
    baseline = payload["baseline_coverage"]
    token_configs = payload["grid_token_configs"]
    fallback_configs = payload["fallback_1ms_token_configs"]
    recommendation = payload["recommendation"]

    lines: list[str] = [
        "---",
        f"pinned_commit: {provenance['commit']}",
        "audit: stage2_timing_shift_vocab_audit",
        "---",
        "",
        "# Stage 2 Timing-Shift Vocabulary Audit",
        "",
        "## Scope",
        "",
        f"- Commit: `{provenance['commit']}`",
        f"- Dataset/index path: `{provenance['index_path']}`",
        f"- Dataset root: `{provenance['dataset_root']}`",
        f"- Beatmaps audited: {dataset['audited_beatmap_count']:,}",
        f"- Events audited: {dataset['total_event_count']:,}",
        f"- Unique event times: {dataset['total_unique_event_time_count']:,}",
        f"- Event-to-event deltas crossing red timing sections: {dataset['event_to_event_crossing_timing_section_delta_count']:,}",
        "",
        "Exact grid coverage below means the osu integer timestamp is actually on a red-timing-grid tick within "
        f"`{provenance['exact_epsilon_ms']}` ms. Integer-ms roundtrip and tolerance rows are snapping diagnostics, "
        "not lossless proof.",
        "",
        "## Candidate Configs",
        "",
        "| config | divisors | TIME_SHIFT tokens @ k=16 | @ k=32 | @ k=64 | @ k=128 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, divisors in payload["candidate_divisor_sets"].items():
        divisor_count = len(divisors)
        lines.append(
            f"| {name} | `{divisors}` | {divisor_count * 16:,} | {divisor_count * 32:,} | "
            f"{divisor_count * 64:,} | {divisor_count * 128:,} |"
        )

    lines.extend(
        [
            "",
            "## Exact Coverage",
            "",
            "| config | exact events | exact rate | failed maps | integer-ms roundtrip | <=0.5ms | <=1ms | <=2ms | <=5ms |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, report in coverage.items():
        tolerances = report["tolerance_rates"]
        lines.append(
            f"| {name} | {report['exact_real_count']:,} | {_pct(report['exact_real_rate'])} | "
            f"{report['failed_beatmap_count']:,} | {_pct(report['integer_ms_roundtrip_rate'])} | "
            f"{_pct(tolerances.get('le_0_5ms', 0.0))} | {_pct(tolerances.get('le_1_0ms', 0.0))} | "
            f"{_pct(tolerances.get('le_2_0ms', 0.0))} | {_pct(tolerances.get('le_5_0ms', 0.0))} |"
        )

    lines.extend(
        [
            "",
            "## Approximate Snapping Coverage",
            "",
            "| config | residual mean ms | p50 | p95 | p99 | max | float ambiguity count | before first red |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, report in coverage.items():
        stats = report["residual_ms"]
        lines.append(
            f"| {name} | {_num(stats['mean'])} | {_num(stats['p50'])} | {_num(stats['p95'])} | "
            f"{_num(stats['p99'])} | {_num(stats['max'])} | {report['float_precision_ambiguous_count']:,} | "
            f"{report['before_first_red_count']:,} |"
        )

    lines.extend(
        [
            "",
            "## Millisecond Baselines",
            "",
            "| baseline | exact rate | failed events | note |",
            "|---|---:|---:|---|",
        ]
    )
    for name, report in baseline.items():
        lines.append(
            f"| {name} | {_pct(report['exact_rate'])} | {report.get('failed_event_count', 0):,} | "
            f"{report.get('note', '')} |"
        )

    lines.extend(
        [
            "",
            "## Token Cost",
            "",
            "| config | vocab size | coverage | successful maps | map p95 | 8s windows | 8s p95 | max TS per delta |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, report in token_configs.items():
        lines.append(_token_cost_row(name, report))
    for name, report in fallback_configs.items():
        lines.append(_token_cost_row(name, report))

    lines.extend(
        [
            "",
            "## Failure Taxonomy",
            "",
            "| failure | count |",
            "|---|---:|",
        ]
    )
    for name, count in payload["failure_taxonomy"].items():
        lines.append(f"| {name} | {count:,} |")

    lines.extend(
        [
            "",
            "## Worst Examples",
            "",
            "| config | beatmap | time_ms | action | divisor | nearest_tick | residual_ms | bpm |",
            "|---|---|---:|---|---:|---:|---:|---:|",
        ]
    )
    for config_name in ("D_very_wide", "C_wide", "B_medium", "A_conservative"):
        if config_name not in coverage:
            continue
        for example in coverage[config_name]["worst_examples"][:6]:
            lines.append(
                f"| {config_name} | `{example['beatmap_path']}` | {example['time_ms']} | {example['action']} | "
                f"{example['divisor']} | {_num(example['nearest_tick_ms'])} | {_num(example['residual_ms'])} | "
                f"{_num(example['bpm'])} |"
            )
        break

    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            f"- Grid-relative TIME_SHIFT alone lossless: `{recommendation['grid_relative_time_shift_alone_lossless']}`",
            f"- Fine-ms fallback required: `{recommendation['fine_ms_fallback_required']}`",
            f"- Best exact divisor set: `{recommendation['best_exact_divisor_set']}` "
            f"({_pct(recommendation['best_exact_rate'])})",
            f"- Best <=1ms approximate divisor set: `{recommendation['best_le_1ms_approx_divisor_set']}` "
            f"({_pct(recommendation['best_le_1ms_approx_rate'])})",
            f"- Recommended grid component if used: `{recommendation['recommended_divisor_set_if_using_grid_component']}`",
            f"- Recommended k_max for cost control: `{recommendation['recommended_k_max']}`",
            f"- 1ms fallback k=128 8s p99 sequence tokens: "
            f"{_num(recommendation['fallback_1ms_k128_window_p99_sequence_tokens'])}",
            "",
            recommendation["summary"],
            "",
        ]
    )
    return "\n".join(lines)


def _token_cost_row(name: str, report: Mapping[str, Any]) -> str:
    return (
        f"| {name} | {report['vocab_size']:,} | {_pct(report['coverage_rate'])} | "
        f"{report['successful_map_count']:,} | {_num(report['map_sequence_tokens']['p95'])} | "
        f"{report['successful_window_count']:,} | {_num(report['window_8s_sequence_tokens']['p95'])} | "
        f"{report['max_time_shift_tokens_for_single_delta']:,} |"
    )


def _pct(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100:.6f}%"


def _num(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.6g}"


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
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit Stage 2 TimingGrid-relative TIME_SHIFT vocabulary feasibility.")
    parser.add_argument("--index-path", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON_PATH)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD_PATH)
    parser.add_argument("--max-maps", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=250)
    args = parser.parse_args(argv)

    command = "uv run python -m train.stage_2.events.audit_timing_shift_vocab " + " ".join(
        sys.argv[1:] if argv is None else argv
    )
    payload = audit_timing_shift_vocab(
        index_path=args.index_path,
        dataset_root=args.dataset_root,
        output_json_path=args.output_json,
        output_md_path=args.output_md,
        max_maps=args.max_maps,
        progress_every=args.progress_every,
        audit_command=command,
    )
    print(f"audited_beatmap_count {payload['dataset']['audited_beatmap_count']}")
    print(f"total_event_count {payload['dataset']['total_event_count']}")
    for name, report in payload["event_time_coverage"].items():
        print(
            f"{name} exact_real_rate {report['exact_real_rate']:.9f} "
            f"integer_ms_roundtrip_rate {report['integer_ms_roundtrip_rate']:.9f} "
            f"le_1ms_rate {report['tolerance_rates'].get('le_1_0ms', 0.0):.9f}"
        )
    print(f"fine_ms_fallback_required {payload['recommendation']['fine_ms_fallback_required']}")
    print(f"output_json {args.output_json}")
    print(f"output_md {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
