from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from ..events.canonical import CanonicalTimepoint, LaneAction
from ..events.canonical import NegativeHitObjectTimeError
from ..events.canonical import UnsupportedCompoundLaneActionError
from ..events.canonical import build_canonical_quantized_events
from ..events.tokens import decompose_ts_delta
from ..events.windowing import WRITE_WINDOW_MS, compute_generation_end_ms
from ..osu.hitobjects import ManiaHitObject, ManiaHitObjectKind, parse_mania_hit_objects
from ..osu.timing import MissingRedTimingError, require_red_timing_points


DIFFICULTY_BIN_LABELS = ("2-3", "3-4", "4-5", "5-6")
FOUR_STATE_UNSUPPORTED_ACTIONS = frozenset({LaneAction.END_TAP, LaneAction.END_START})
FROZEN_FOUR_STATE_ACTIONS = frozenset(
    {
        LaneAction.NONE,
        LaneAction.TAP,
        LaneAction.HOLD_START,
        LaneAction.HOLD_END,
    },
)


@dataclass(frozen=True)
class TokenStatisticsMap:
    difficulty: float
    generation_end_ms: int
    timepoints: Sequence[CanonicalTimepoint]


@dataclass(frozen=True)
class OsuTokenStatisticsMapInput:
    beatmap_path: str | Path
    difficulty: float
    generation_end_ms: int | None = None
    audio_duration_ms: float | None = None


@dataclass(frozen=True)
class TokenLengthStats:
    mean: float
    p95: int
    p99: int
    max: int


@dataclass(frozen=True)
class DifficultyBinTokenStatistics:
    label: str
    map_count: int
    window_count: int
    tokens_per_window: TokenLengthStats
    event_timepoints_per_second: float
    note_events_per_second: float
    ln_ratio: float
    chord_size_counts: dict[int, int]
    chord_size_distribution: dict[int, float]
    ts_counts: dict[int, int]
    ts_distribution: dict[int, float]
    empty_window_count: int
    empty_window_ratio: float
    hold_crossing_window_count: int
    hold_crossing_window_ratio: float
    max_decode_len: int
    max_event_delta_ms: int
    max_ts_tokens_per_delta: int
    windows_requiring_multi_ts_count: int
    windows_requiring_multi_ts_ratio: float


@dataclass(frozen=True)
class CompoundIssueCounts:
    event_count: int
    lane_action_count: int


@dataclass(frozen=True)
class FourStateUnsupportedCounts:
    event_count: int
    lane_action_count: int


@dataclass(frozen=True)
class TokenStatisticsAuditReport:
    total_map_count: int
    audited_map_count: int
    out_of_range_map_count: int
    missing_red_timing_map_count: int
    negative_time_hitobject_map_count: int
    unsupported_compound_map_count: int
    unsupported_compound_event_count: int
    unsupported_compound_lane_action_count: int
    four_state_unsupported_map_count: int
    four_state_unsupported_event_count: int
    four_state_unsupported_lane_action_count: int
    filtered_event_count: int
    filtered_lane_action_count: int
    zero_length_hold_normalized_count: int
    bins: dict[str, DifficultyBinTokenStatistics]


@dataclass(frozen=True)
class TokenStatisticsGateDecision:
    status: str
    configured_max_decode_len: int
    max_decode_len_applies_to: str
    observed_max_target_tokens: int
    max_decode_len_headroom_tokens: int
    empty_window_cap_policy: str
    empty_window_cap_ratio: float
    empty_window_cap_by_bin: dict[str, float]
    ts_1000_sufficient: bool
    max_event_delta_ms: int
    max_ts_tokens_per_delta: int
    covers_quantization_audit: bool
    covers_window_boundary_audit: bool


@dataclass
class _BinAccumulator:
    label: str
    map_count: int = 0
    window_count: int = 0
    total_write_duration_ms: int = 0
    token_counts: list[int] | None = None
    event_timepoint_count: int = 0
    note_event_count: int = 0
    hold_start_count: int = 0
    chord_size_counts: Counter[int] | None = None
    ts_counts: Counter[int] | None = None
    empty_window_count: int = 0
    hold_crossing_window_count: int = 0
    max_event_delta_ms: int = 0
    max_ts_tokens_per_delta: int = 0
    windows_requiring_multi_ts_count: int = 0

    def __post_init__(self) -> None:
        self.token_counts = [] if self.token_counts is None else self.token_counts
        self.chord_size_counts = Counter() if self.chord_size_counts is None else self.chord_size_counts
        self.ts_counts = Counter() if self.ts_counts is None else self.ts_counts

    def to_report(self) -> DifficultyBinTokenStatistics:
        token_counts = self.token_counts or []
        chord_counts = dict(sorted((self.chord_size_counts or Counter()).items()))
        ts_counts = dict(sorted((self.ts_counts or Counter()).items()))
        write_seconds = self.total_write_duration_ms / 1000
        return DifficultyBinTokenStatistics(
            label=self.label,
            map_count=self.map_count,
            window_count=self.window_count,
            tokens_per_window=_token_length_stats(token_counts),
            event_timepoints_per_second=_rate(self.event_timepoint_count, write_seconds),
            note_events_per_second=_rate(self.note_event_count, write_seconds),
            ln_ratio=_rate(self.hold_start_count, self.note_event_count),
            chord_size_counts=chord_counts,
            chord_size_distribution=_counter_distribution(chord_counts),
            ts_counts=ts_counts,
            ts_distribution=_counter_distribution(ts_counts),
            empty_window_count=self.empty_window_count,
            empty_window_ratio=_rate(self.empty_window_count, self.window_count),
            hold_crossing_window_count=self.hold_crossing_window_count,
            hold_crossing_window_ratio=_rate(self.hold_crossing_window_count, self.window_count),
            max_decode_len=max(token_counts, default=0),
            max_event_delta_ms=self.max_event_delta_ms,
            max_ts_tokens_per_delta=self.max_ts_tokens_per_delta,
            windows_requiring_multi_ts_count=self.windows_requiring_multi_ts_count,
            windows_requiring_multi_ts_ratio=_rate(self.windows_requiring_multi_ts_count, self.window_count),
        )


@dataclass(frozen=True)
class _WindowTokenSummary:
    token_count: int
    event_timepoint_count: int
    note_event_count: int
    hold_start_count: int
    chord_size_counts: Counter[int]
    ts_counts: Counter[int]
    max_event_delta_ms: int
    max_ts_tokens_per_delta: int
    requires_multi_ts: bool


def audit_token_statistics(event_maps: Iterable[TokenStatisticsMap]) -> TokenStatisticsAuditReport:
    total_map_count = 0
    out_of_range_map_count = 0
    four_state_unsupported_map_count = 0
    four_state_unsupported_event_count = 0
    four_state_unsupported_lane_action_count = 0
    audited_maps: list[TokenStatisticsMap] = []

    for event_map in event_maps:
        total_map_count += 1
        if difficulty_bin_label(event_map.difficulty) is None:
            out_of_range_map_count += 1
            continue

        _validate_timepoints_structure(
            event_map.timepoints,
            generation_end_ms=event_map.generation_end_ms,
            key_count=4,
        )
        unsupported_counts = _four_state_unsupported_counts(event_map.timepoints)
        if unsupported_counts.lane_action_count:
            four_state_unsupported_map_count += 1
            four_state_unsupported_event_count += unsupported_counts.event_count
            four_state_unsupported_lane_action_count += unsupported_counts.lane_action_count
            continue

        _validate_frozen_four_state_and_hold_legality(event_map.timepoints)
        audited_maps.append(event_map)

    return _build_report(
        total_map_count=total_map_count,
        audited_maps=audited_maps,
        out_of_range_map_count=out_of_range_map_count,
        missing_red_timing_map_count=0,
        negative_time_hitobject_map_count=0,
        unsupported_compound_map_count=0,
        unsupported_compound_event_count=0,
        unsupported_compound_lane_action_count=0,
        four_state_unsupported_map_count=four_state_unsupported_map_count,
        four_state_unsupported_event_count=four_state_unsupported_event_count,
        four_state_unsupported_lane_action_count=four_state_unsupported_lane_action_count,
        zero_length_hold_normalized_count=0,
    )


def audit_osu_token_statistics(
    map_inputs: Iterable[OsuTokenStatisticsMapInput],
    *,
    key_count: int = 4,
) -> TokenStatisticsAuditReport:
    total_map_count = 0
    out_of_range_map_count = 0
    missing_red_timing_map_count = 0
    negative_time_hitobject_map_count = 0
    unsupported_compound_map_count = 0
    unsupported_compound_event_count = 0
    unsupported_compound_lane_action_count = 0
    four_state_unsupported_map_count = 0
    four_state_unsupported_event_count = 0
    four_state_unsupported_lane_action_count = 0
    zero_length_hold_normalized_count = 0
    audited_maps: list[TokenStatisticsMap] = []

    for map_input in map_inputs:
        total_map_count += 1
        if difficulty_bin_label(map_input.difficulty) is None:
            out_of_range_map_count += 1
            continue

        beatmap_path = Path(map_input.beatmap_path)
        try:
            require_red_timing_points(beatmap_path)
        except MissingRedTimingError:
            missing_red_timing_map_count += 1
            continue

        hitobjects = parse_mania_hit_objects(beatmap_path, expected_key_count=key_count)
        try:
            compound_counts = _count_unsupported_compounds(hitobjects, key_count=key_count)
        except NegativeHitObjectTimeError:
            negative_time_hitobject_map_count += 1
            continue

        if compound_counts.event_count:
            unsupported_compound_map_count += 1
            unsupported_compound_event_count += compound_counts.event_count
            unsupported_compound_lane_action_count += compound_counts.lane_action_count
            continue

        try:
            build_result = build_canonical_quantized_events(hitobjects, key_count=key_count)
        except NegativeHitObjectTimeError:
            negative_time_hitobject_map_count += 1
            continue
        except UnsupportedCompoundLaneActionError:
            unsupported_compound_map_count += 1
            unsupported_compound_event_count += 1
            continue

        zero_length_hold_normalized_count += build_result.zero_length_hold_normalized_count
        generation_end_ms = _resolve_generation_end_ms(map_input, build_result.timepoints)
        _validate_timepoints_structure(
            build_result.timepoints,
            generation_end_ms=generation_end_ms,
            key_count=key_count,
        )
        unsupported_counts = _four_state_unsupported_counts(build_result.timepoints)
        if unsupported_counts.lane_action_count:
            four_state_unsupported_map_count += 1
            four_state_unsupported_event_count += unsupported_counts.event_count
            four_state_unsupported_lane_action_count += unsupported_counts.lane_action_count
            continue

        _validate_frozen_four_state_and_hold_legality(build_result.timepoints)
        audited_maps.append(
            TokenStatisticsMap(
                difficulty=map_input.difficulty,
                generation_end_ms=generation_end_ms,
                timepoints=build_result.timepoints,
            ),
        )

    return _build_report(
        total_map_count=total_map_count,
        audited_maps=audited_maps,
        out_of_range_map_count=out_of_range_map_count,
        missing_red_timing_map_count=missing_red_timing_map_count,
        negative_time_hitobject_map_count=negative_time_hitobject_map_count,
        unsupported_compound_map_count=unsupported_compound_map_count,
        unsupported_compound_event_count=unsupported_compound_event_count,
        unsupported_compound_lane_action_count=unsupported_compound_lane_action_count,
        four_state_unsupported_map_count=four_state_unsupported_map_count,
        four_state_unsupported_event_count=four_state_unsupported_event_count,
        four_state_unsupported_lane_action_count=four_state_unsupported_lane_action_count,
        zero_length_hold_normalized_count=zero_length_hold_normalized_count,
    )


def difficulty_bin_label(stars: float) -> str | None:
    if 2.0 <= stars < 3.0:
        return "2-3"
    if 3.0 <= stars < 4.0:
        return "3-4"
    if 4.0 <= stars < 5.0:
        return "4-5"
    if 5.0 <= stars <= 6.0:
        return "5-6"
    return None


def build_token_statistics_gate_decision(
    report: TokenStatisticsAuditReport,
    *,
    configured_max_decode_len: int,
    empty_window_cap_ratio: float,
) -> TokenStatisticsGateDecision:
    observed_max_target_tokens = max(
        (bin_report.tokens_per_window.max for bin_report in report.bins.values()),
        default=0,
    )
    max_event_delta_ms = max((bin_report.max_event_delta_ms for bin_report in report.bins.values()), default=0)
    max_ts_tokens_per_delta = max(
        (bin_report.max_ts_tokens_per_delta for bin_report in report.bins.values()),
        default=0,
    )
    headroom = configured_max_decode_len - observed_max_target_tokens

    return TokenStatisticsGateDecision(
        status="PASS" if headroom >= 0 else "FAIL",
        configured_max_decode_len=configured_max_decode_len,
        max_decode_len_applies_to="target_tokens_excluding_bos_and_condition_prefix",
        observed_max_target_tokens=observed_max_target_tokens,
        max_decode_len_headroom_tokens=headroom,
        empty_window_cap_policy="per_difficulty_bin_per_epoch",
        empty_window_cap_ratio=empty_window_cap_ratio,
        empty_window_cap_by_bin={label: empty_window_cap_ratio for label in DIFFICULTY_BIN_LABELS},
        ts_1000_sufficient=max_ts_tokens_per_delta <= 8 and max_event_delta_ms < WRITE_WINDOW_MS,
        max_event_delta_ms=max_event_delta_ms,
        max_ts_tokens_per_delta=max_ts_tokens_per_delta,
        covers_quantization_audit=False,
        covers_window_boundary_audit=False,
    )


def _build_report(
    *,
    total_map_count: int,
    audited_maps: Sequence[TokenStatisticsMap],
    out_of_range_map_count: int,
    missing_red_timing_map_count: int,
    negative_time_hitobject_map_count: int,
    unsupported_compound_map_count: int,
    unsupported_compound_event_count: int,
    unsupported_compound_lane_action_count: int,
    four_state_unsupported_map_count: int,
    four_state_unsupported_event_count: int,
    four_state_unsupported_lane_action_count: int,
    zero_length_hold_normalized_count: int,
) -> TokenStatisticsAuditReport:
    accumulators = {label: _BinAccumulator(label) for label in DIFFICULTY_BIN_LABELS}
    for event_map in audited_maps:
        label = difficulty_bin_label(event_map.difficulty)
        if label is None:
            raise ValueError(f"audited map outside supported difficulty range: {event_map.difficulty}")
        accumulator = accumulators[label]
        accumulator.map_count += 1
        _accumulate_map(accumulator, event_map)

    return TokenStatisticsAuditReport(
        total_map_count=total_map_count,
        audited_map_count=len(audited_maps),
        out_of_range_map_count=out_of_range_map_count,
        missing_red_timing_map_count=missing_red_timing_map_count,
        negative_time_hitobject_map_count=negative_time_hitobject_map_count,
        unsupported_compound_map_count=unsupported_compound_map_count,
        unsupported_compound_event_count=unsupported_compound_event_count,
        unsupported_compound_lane_action_count=unsupported_compound_lane_action_count,
        four_state_unsupported_map_count=four_state_unsupported_map_count,
        four_state_unsupported_event_count=four_state_unsupported_event_count,
        four_state_unsupported_lane_action_count=four_state_unsupported_lane_action_count,
        filtered_event_count=unsupported_compound_event_count + four_state_unsupported_event_count,
        filtered_lane_action_count=unsupported_compound_lane_action_count + four_state_unsupported_lane_action_count,
        zero_length_hold_normalized_count=zero_length_hold_normalized_count,
        bins={label: accumulators[label].to_report() for label in DIFFICULTY_BIN_LABELS},
    )


def _accumulate_map(accumulator: _BinAccumulator, event_map: TokenStatisticsMap) -> None:
    timepoints = sorted(event_map.timepoints, key=lambda timepoint: timepoint.time_ms)
    event_index = 0
    open_hold_mask = 0
    write_start = 0

    while write_start < event_map.generation_end_ms:
        write_end = min(write_start + WRITE_WINDOW_MS, event_map.generation_end_ms)

        while event_index < len(timepoints) and timepoints[event_index].time_ms < write_start:
            open_hold_mask = _apply_open_hold_mask(open_hold_mask, timepoints[event_index])
            event_index += 1

        open_hold_mask_at_start = open_hold_mask
        end_event_index = event_index
        open_hold_mask_at_end = open_hold_mask
        window_timepoints: list[CanonicalTimepoint] = []

        while end_event_index < len(timepoints) and timepoints[end_event_index].time_ms < write_end:
            timepoint = timepoints[end_event_index]
            window_timepoints.append(timepoint)
            open_hold_mask_at_end = _apply_open_hold_mask(open_hold_mask_at_end, timepoint)
            end_event_index += 1

        summary = _summarize_window_tokens(window_timepoints, write_start_ms=write_start)
        accumulator.window_count += 1
        accumulator.total_write_duration_ms += write_end - write_start
        assert accumulator.token_counts is not None
        accumulator.token_counts.append(summary.token_count)
        accumulator.event_timepoint_count += summary.event_timepoint_count
        accumulator.note_event_count += summary.note_event_count
        accumulator.hold_start_count += summary.hold_start_count
        assert accumulator.chord_size_counts is not None
        accumulator.chord_size_counts.update(summary.chord_size_counts)
        assert accumulator.ts_counts is not None
        accumulator.ts_counts.update(summary.ts_counts)
        if not window_timepoints:
            accumulator.empty_window_count += 1
        if open_hold_mask_at_start or open_hold_mask_at_end:
            accumulator.hold_crossing_window_count += 1
        accumulator.max_event_delta_ms = max(accumulator.max_event_delta_ms, summary.max_event_delta_ms)
        accumulator.max_ts_tokens_per_delta = max(
            accumulator.max_ts_tokens_per_delta,
            summary.max_ts_tokens_per_delta,
        )
        if summary.requires_multi_ts:
            accumulator.windows_requiring_multi_ts_count += 1

        open_hold_mask = open_hold_mask_at_end
        event_index = end_event_index
        write_start += WRITE_WINDOW_MS


def _summarize_window_tokens(
    timepoints: Sequence[CanonicalTimepoint],
    *,
    write_start_ms: int,
) -> _WindowTokenSummary:
    token_count = 1
    event_timepoint_count = 0
    note_event_count = 0
    hold_start_count = 0
    chord_size_counts: Counter[int] = Counter()
    ts_counts: Counter[int] = Counter()
    max_event_delta_ms = 0
    max_ts_tokens_per_delta = 0
    requires_multi_ts = False
    previous_time_rel: int | None = None

    for timepoint in timepoints:
        time_rel = timepoint.time_ms - write_start_ms
        if previous_time_rel is None:
            delta_ms = time_rel
        else:
            delta_ms = time_rel - previous_time_rel
            if delta_ms <= 0:
                raise ValueError(f"canonical timepoints must strictly increase within a window: {timepoints}")

        ts_values = decompose_ts_delta(delta_ms)
        token_count += len(ts_values) + 1
        ts_counts.update(ts_values)
        event_timepoint_count += 1

        chord_size = _note_start_count(timepoint.lane_actions)
        if chord_size:
            chord_size_counts[chord_size] += 1
            note_event_count += chord_size
        hold_start_count += _hold_start_count(timepoint.lane_actions)

        max_event_delta_ms = max(max_event_delta_ms, delta_ms)
        max_ts_tokens_per_delta = max(max_ts_tokens_per_delta, len(ts_values))
        if len(ts_values) > 1:
            requires_multi_ts = True
        previous_time_rel = time_rel

    return _WindowTokenSummary(
        token_count=token_count,
        event_timepoint_count=event_timepoint_count,
        note_event_count=note_event_count,
        hold_start_count=hold_start_count,
        chord_size_counts=chord_size_counts,
        ts_counts=ts_counts,
        max_event_delta_ms=max_event_delta_ms,
        max_ts_tokens_per_delta=max_ts_tokens_per_delta,
        requires_multi_ts=requires_multi_ts,
    )


def _resolve_generation_end_ms(
    map_input: OsuTokenStatisticsMapInput,
    timepoints: Sequence[CanonicalTimepoint],
) -> int:
    if map_input.generation_end_ms is not None:
        if map_input.audio_duration_ms is not None:
            expected_generation_end_ms = compute_generation_end_ms(map_input.audio_duration_ms, timepoints)
            if map_input.generation_end_ms != expected_generation_end_ms:
                raise ValueError(
                    "generation_end_ms does not match spec rule: "
                    f"provided={map_input.generation_end_ms}, expected={expected_generation_end_ms}",
                )
        return map_input.generation_end_ms
    if map_input.audio_duration_ms is not None:
        return compute_generation_end_ms(map_input.audio_duration_ms, timepoints)
    raise ValueError(
        "OsuTokenStatisticsMapInput requires generation_end_ms or audio_duration_ms "
        f"for {map_input.beatmap_path}",
    )

def _validate_timepoints_structure(
    timepoints: Sequence[CanonicalTimepoint],
    *,
    generation_end_ms: int,
    key_count: int,
) -> None:
    if generation_end_ms < 0:
        raise ValueError(f"generation_end_ms must be non-negative: {generation_end_ms}")
    if generation_end_ms % 10 != 0:
        raise ValueError(f"generation_end_ms must be on the 10ms grid: {generation_end_ms}")

    max_event_time_ms = max((timepoint.time_ms for timepoint in timepoints), default=-1)
    if max_event_time_ms >= generation_end_ms:
        raise ValueError(
            "generation_end_ms must be greater than the last canonical event time: "
            f"generation_end_ms={generation_end_ms}, last_event={max_event_time_ms}",
        )

    valid_lane_actions = set(LaneAction)
    seen_times: set[int] = set()
    for timepoint in timepoints:
        if timepoint.time_ms < 0:
            raise NegativeHitObjectTimeError(f"canonical timepoint has negative time: {timepoint}")
        if timepoint.time_ms % 10 != 0:
            raise ValueError(f"canonical timepoint must be on the 10ms grid: {timepoint}")
        if timepoint.time_ms in seen_times:
            raise ValueError(f"duplicate canonical timepoint time: {timepoint.time_ms}")
        if len(timepoint.lane_actions) != key_count:
            raise ValueError(
                f"canonical timepoint must have {key_count} lane actions: {timepoint}",
            )
        if any(action not in valid_lane_actions for action in timepoint.lane_actions):
            raise ValueError(f"canonical timepoint contains unknown lane action: {timepoint}")
        if all(action == LaneAction.NONE for action in timepoint.lane_actions):
            raise ValueError(f"canonical timepoint must not encode an all-empty EV: {timepoint}")
        seen_times.add(timepoint.time_ms)


def _validate_frozen_four_state_and_hold_legality(timepoints: Sequence[CanonicalTimepoint]) -> None:
    open_hold_mask = 0
    for timepoint in sorted(timepoints, key=lambda item: item.time_ms):
        for lane, action in enumerate(timepoint.lane_actions):
            lane_bit = 1 << lane
            if action not in FROZEN_FOUR_STATE_ACTIONS:
                raise ValueError(f"post-filter token statistics input contains non-4-state action: {timepoint}")
            if action == LaneAction.HOLD_START:
                if open_hold_mask & lane_bit:
                    raise ValueError(f"HOLD_START while lane already open at {timepoint.time_ms}ms lane {lane}")
                open_hold_mask |= lane_bit
            elif action == LaneAction.HOLD_END:
                if not open_hold_mask & lane_bit:
                    raise ValueError(f"HOLD_END without open hold at {timepoint.time_ms}ms lane {lane}")
                open_hold_mask &= ~lane_bit


def _apply_open_hold_mask(open_hold_mask: int, timepoint: CanonicalTimepoint) -> int:
    for lane, action in enumerate(timepoint.lane_actions):
        lane_bit = 1 << lane
        if action == LaneAction.HOLD_START:
            open_hold_mask |= lane_bit
        elif action == LaneAction.HOLD_END:
            open_hold_mask &= ~lane_bit
        elif action == LaneAction.END_TAP:
            open_hold_mask &= ~lane_bit
        elif action == LaneAction.END_START:
            open_hold_mask |= lane_bit
    return open_hold_mask


def _note_start_count(lane_actions: Sequence[LaneAction]) -> int:
    note_start_actions = {
        LaneAction.TAP,
        LaneAction.HOLD_START,
        LaneAction.END_TAP,
        LaneAction.END_START,
    }
    return sum(1 for action in lane_actions if action in note_start_actions)


def _hold_start_count(lane_actions: Sequence[LaneAction]) -> int:
    return sum(1 for action in lane_actions if action in {LaneAction.HOLD_START, LaneAction.END_START})


def _four_state_unsupported_counts(timepoints: Sequence[CanonicalTimepoint]) -> FourStateUnsupportedCounts:
    event_count = 0
    lane_action_count = 0
    for timepoint in timepoints:
        timepoint_lane_action_count = sum(
            1 for action in timepoint.lane_actions if action in FOUR_STATE_UNSUPPORTED_ACTIONS
        )
        if timepoint_lane_action_count:
            event_count += 1
            lane_action_count += timepoint_lane_action_count

    return FourStateUnsupportedCounts(
        event_count=event_count,
        lane_action_count=lane_action_count,
    )


def _count_unsupported_compounds(
    hitobjects: Sequence[ManiaHitObject],
    *,
    key_count: int,
) -> CompoundIssueCounts:
    primitive_actions: dict[int, dict[int, list[LaneAction]]] = defaultdict(lambda: defaultdict(list))

    for hitobject in hitobjects:
        _validate_hitobject_for_compound_count(hitobject, key_count)
        q_start = _quantize_for_compound_count(hitobject.start_time_ms)

        if hitobject.kind == ManiaHitObjectKind.TAP:
            primitive_actions[q_start][hitobject.lane].append(LaneAction.TAP)
            continue

        q_end = _quantize_for_compound_count(hitobject.end_time_ms)
        if q_end <= q_start:
            primitive_actions[q_start][hitobject.lane].append(LaneAction.TAP)
            continue

        primitive_actions[q_start][hitobject.lane].append(LaneAction.HOLD_START)
        primitive_actions[q_end][hitobject.lane].append(LaneAction.HOLD_END)

    event_count = 0
    lane_action_count = 0
    for lane_actions_by_time in primitive_actions.values():
        for actions in lane_actions_by_time.values():
            if _is_supported_compound_action_list(actions):
                continue
            event_count += 1
            lane_action_count += len(actions)

    return CompoundIssueCounts(event_count=event_count, lane_action_count=lane_action_count)


def _is_supported_compound_action_list(actions: Sequence[LaneAction]) -> bool:
    if len(actions) <= 1:
        return True
    action_set = set(actions)
    return len(actions) == 2 and action_set in (
        {LaneAction.HOLD_END, LaneAction.TAP},
        {LaneAction.HOLD_END, LaneAction.HOLD_START},
    )


def _validate_hitobject_for_compound_count(hitobject: ManiaHitObject, key_count: int) -> None:
    if hitobject.start_time_ms < 0 or hitobject.end_time_ms < 0:
        raise NegativeHitObjectTimeError(f"hit object has negative time: {hitobject}")
    if not 0 <= hitobject.lane < key_count:
        raise ValueError(f"hit object lane {hitobject.lane} outside 0..{key_count - 1}: {hitobject}")


def _quantize_for_compound_count(time_ms: float) -> int:
    if time_ms < 0:
        raise NegativeHitObjectTimeError(f"cannot quantize negative time: {time_ms}")
    return int(10 * math.floor((time_ms + 5) / 10))


def _token_length_stats(token_counts: Sequence[int]) -> TokenLengthStats:
    if not token_counts:
        return TokenLengthStats(mean=0.0, p95=0, p99=0, max=0)
    return TokenLengthStats(
        mean=sum(token_counts) / len(token_counts),
        p95=_nearest_rank_percentile(token_counts, 95),
        p99=_nearest_rank_percentile(token_counts, 99),
        max=max(token_counts),
    )


def _nearest_rank_percentile(values: Sequence[int], percentile: int) -> int:
    if not values:
        return 0
    sorted_values = sorted(values)
    index = math.ceil((percentile / 100) * len(sorted_values)) - 1
    return sorted_values[max(0, min(index, len(sorted_values) - 1))]


def _counter_distribution(counts: dict[int, int]) -> dict[int, float]:
    total = sum(counts.values())
    if total == 0:
        return {}
    return {key: value / total for key, value in counts.items()}


def _rate(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator
