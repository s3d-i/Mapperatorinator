from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from ..events.canonical import CanonicalTimepoint, LaneAction
from ..events.canonical import NegativeHitObjectTimeError
from ..events.canonical import UnsupportedCompoundLaneActionError
from ..events.canonical import build_canonical_quantized_events
from ..events.windowing import WRITE_WINDOW_MS, compute_generation_end_ms
from ..osu.hitobjects import parse_mania_hit_objects
from ..osu.timing import MissingRedTimingError, require_red_timing_points
from .token_statistics import DIFFICULTY_BIN_LABELS, difficulty_bin_label
from .token_statistics import _count_unsupported_compounds
from .token_statistics import _four_state_unsupported_counts
from .token_statistics import _validate_frozen_four_state_and_hold_legality
from .token_statistics import _validate_timepoints_structure


@dataclass(frozen=True)
class WindowBoundaryMap:
    difficulty: float
    generation_end_ms: int
    timepoints: Sequence[CanonicalTimepoint]


@dataclass(frozen=True)
class OsuWindowBoundaryMapInput:
    beatmap_path: str | Path
    difficulty: float
    generation_end_ms: int | None = None
    audio_duration_ms: float | None = None


@dataclass(frozen=True)
class WindowBoundaryBinReport:
    label: str
    map_count: int
    boundary_count: int
    boundary_event_count: int
    boundary_with_event_count: int
    boundary_lane_action_count: int
    boundary_event_density: float
    boundary_with_event_rate: float
    boundary_lane_action_density: float
    hold_crossing_boundary_count: int
    hold_crossing_boundary_rate: float
    hold_crossing_lane_count: int
    hold_crossing_lane_rate: float
    stitch_duplicate_timepoint_count: int
    stitch_duplicate_timepoint_rate: float
    stitch_collision_timepoint_count: int
    stitch_collision_lane_action_count: int
    stitch_collision_timepoint_rate: float
    stitch_roundtrip_mismatch_count: int


@dataclass(frozen=True)
class WindowBoundaryAuditReport:
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
    boundary_count: int
    boundary_event_count: int
    boundary_with_event_count: int
    boundary_lane_action_count: int
    boundary_event_density: float
    boundary_with_event_rate: float
    boundary_lane_action_density: float
    hold_crossing_boundary_count: int
    hold_crossing_boundary_rate: float
    hold_crossing_lane_count: int
    hold_crossing_lane_rate: float
    stitch_duplicate_timepoint_count: int
    stitch_duplicate_timepoint_rate: float
    stitch_collision_timepoint_count: int
    stitch_collision_lane_action_count: int
    stitch_collision_timepoint_rate: float
    stitch_roundtrip_mismatch_count: int
    bins: dict[str, WindowBoundaryBinReport]


@dataclass(frozen=True)
class WindowBoundaryGateDecision:
    status: str
    window_ownership: str
    write_window_ms: int
    stitch_duplicate_timepoint_count: int
    stitch_collision_timepoint_count: int
    stitch_roundtrip_mismatch_count: int
    boundary_event_density: float
    hold_crossing_boundary_rate: float


@dataclass
class _WindowBoundaryAccumulator:
    label: str
    map_count: int = 0
    boundary_count: int = 0
    boundary_event_count: int = 0
    boundary_with_event_count: int = 0
    boundary_lane_action_count: int = 0
    hold_crossing_boundary_count: int = 0
    hold_crossing_lane_count: int = 0
    stitch_duplicate_timepoint_count: int = 0
    stitch_collision_timepoint_count: int = 0
    stitch_collision_lane_action_count: int = 0
    stitch_roundtrip_mismatch_count: int = 0

    def add_summary(self, summary: "_WindowBoundaryMapSummary") -> None:
        self.map_count += 1
        self.boundary_count += summary.boundary_count
        self.boundary_event_count += summary.boundary_event_count
        self.boundary_with_event_count += summary.boundary_with_event_count
        self.boundary_lane_action_count += summary.boundary_lane_action_count
        self.hold_crossing_boundary_count += summary.hold_crossing_boundary_count
        self.hold_crossing_lane_count += summary.hold_crossing_lane_count
        self.stitch_duplicate_timepoint_count += summary.stitch_duplicate_timepoint_count
        self.stitch_collision_timepoint_count += summary.stitch_collision_timepoint_count
        self.stitch_collision_lane_action_count += summary.stitch_collision_lane_action_count
        self.stitch_roundtrip_mismatch_count += summary.stitch_roundtrip_mismatch_count

    def to_report(self) -> WindowBoundaryBinReport:
        return WindowBoundaryBinReport(
            label=self.label,
            map_count=self.map_count,
            boundary_count=self.boundary_count,
            boundary_event_count=self.boundary_event_count,
            boundary_with_event_count=self.boundary_with_event_count,
            boundary_lane_action_count=self.boundary_lane_action_count,
            boundary_event_density=_rate(self.boundary_event_count, self.boundary_count),
            boundary_with_event_rate=_rate(self.boundary_with_event_count, self.boundary_count),
            boundary_lane_action_density=_rate(self.boundary_lane_action_count, self.boundary_count),
            hold_crossing_boundary_count=self.hold_crossing_boundary_count,
            hold_crossing_boundary_rate=_rate(self.hold_crossing_boundary_count, self.boundary_count),
            hold_crossing_lane_count=self.hold_crossing_lane_count,
            hold_crossing_lane_rate=_rate(self.hold_crossing_lane_count, self.boundary_count * 4),
            stitch_duplicate_timepoint_count=self.stitch_duplicate_timepoint_count,
            stitch_duplicate_timepoint_rate=_rate(self.stitch_duplicate_timepoint_count, self.boundary_count),
            stitch_collision_timepoint_count=self.stitch_collision_timepoint_count,
            stitch_collision_lane_action_count=self.stitch_collision_lane_action_count,
            stitch_collision_timepoint_rate=_rate(self.stitch_collision_timepoint_count, self.boundary_count),
            stitch_roundtrip_mismatch_count=self.stitch_roundtrip_mismatch_count,
        )


@dataclass(frozen=True)
class _WindowBoundaryMapSummary:
    boundary_count: int
    boundary_event_count: int
    boundary_with_event_count: int
    boundary_lane_action_count: int
    hold_crossing_boundary_count: int
    hold_crossing_lane_count: int
    stitch_duplicate_timepoint_count: int
    stitch_collision_timepoint_count: int
    stitch_collision_lane_action_count: int
    stitch_roundtrip_mismatch_count: int


def audit_window_boundaries(
    event_maps: Iterable[WindowBoundaryMap],
    *,
    key_count: int = 4,
) -> WindowBoundaryAuditReport:
    total_map_count = 0
    out_of_range_map_count = 0
    four_state_unsupported_map_count = 0
    four_state_unsupported_event_count = 0
    four_state_unsupported_lane_action_count = 0
    accumulators = {label: _WindowBoundaryAccumulator(label) for label in DIFFICULTY_BIN_LABELS}

    for event_map in event_maps:
        total_map_count += 1
        label = difficulty_bin_label(event_map.difficulty)
        if label is None:
            out_of_range_map_count += 1
            continue

        _validate_timepoints_structure(
            event_map.timepoints,
            generation_end_ms=event_map.generation_end_ms,
            key_count=key_count,
        )
        unsupported_counts = _four_state_unsupported_counts(event_map.timepoints)
        if unsupported_counts.lane_action_count:
            four_state_unsupported_map_count += 1
            four_state_unsupported_event_count += unsupported_counts.event_count
            four_state_unsupported_lane_action_count += unsupported_counts.lane_action_count
            continue
        _validate_frozen_four_state_and_hold_legality(event_map.timepoints)
        summary = _summarize_window_boundaries(
            event_map.timepoints,
            generation_end_ms=event_map.generation_end_ms,
            key_count=key_count,
        )
        accumulators[label].add_summary(summary)

    return _build_report(
        total_map_count=total_map_count,
        out_of_range_map_count=out_of_range_map_count,
        missing_red_timing_map_count=0,
        negative_time_hitobject_map_count=0,
        unsupported_compound_map_count=0,
        unsupported_compound_event_count=0,
        unsupported_compound_lane_action_count=0,
        four_state_unsupported_map_count=four_state_unsupported_map_count,
        four_state_unsupported_event_count=four_state_unsupported_event_count,
        four_state_unsupported_lane_action_count=four_state_unsupported_lane_action_count,
        accumulators=accumulators,
    )


def audit_osu_window_boundaries(
    map_inputs: Iterable[OsuWindowBoundaryMapInput],
    *,
    key_count: int = 4,
) -> WindowBoundaryAuditReport:
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
    accumulators = {label: _WindowBoundaryAccumulator(label) for label in DIFFICULTY_BIN_LABELS}

    for map_input in map_inputs:
        total_map_count += 1
        label = difficulty_bin_label(map_input.difficulty)
        if label is None:
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
        summary = _summarize_window_boundaries(
            build_result.timepoints,
            generation_end_ms=generation_end_ms,
            key_count=key_count,
        )
        accumulators[label].add_summary(summary)

    return _build_report(
        total_map_count=total_map_count,
        out_of_range_map_count=out_of_range_map_count,
        missing_red_timing_map_count=missing_red_timing_map_count,
        negative_time_hitobject_map_count=negative_time_hitobject_map_count,
        unsupported_compound_map_count=unsupported_compound_map_count,
        unsupported_compound_event_count=unsupported_compound_event_count,
        unsupported_compound_lane_action_count=unsupported_compound_lane_action_count,
        four_state_unsupported_map_count=four_state_unsupported_map_count,
        four_state_unsupported_event_count=four_state_unsupported_event_count,
        four_state_unsupported_lane_action_count=four_state_unsupported_lane_action_count,
        accumulators=accumulators,
    )


def build_window_boundary_gate_decision(report: WindowBoundaryAuditReport) -> WindowBoundaryGateDecision:
    status = (
        "PASS"
        if (
            report.stitch_duplicate_timepoint_count == 0
            and report.stitch_collision_timepoint_count == 0
            and report.stitch_roundtrip_mismatch_count == 0
        )
        else "FAIL"
    )
    return WindowBoundaryGateDecision(
        status=status,
        window_ownership="half_open_write_intervals",
        write_window_ms=WRITE_WINDOW_MS,
        stitch_duplicate_timepoint_count=report.stitch_duplicate_timepoint_count,
        stitch_collision_timepoint_count=report.stitch_collision_timepoint_count,
        stitch_roundtrip_mismatch_count=report.stitch_roundtrip_mismatch_count,
        boundary_event_density=report.boundary_event_density,
        hold_crossing_boundary_rate=report.hold_crossing_boundary_rate,
    )


def _summarize_window_boundaries(
    timepoints: Sequence[CanonicalTimepoint],
    *,
    generation_end_ms: int,
    key_count: int,
) -> _WindowBoundaryMapSummary:
    sorted_timepoints = sorted(timepoints, key=lambda timepoint: timepoint.time_ms)
    timepoint_by_time = {timepoint.time_ms: timepoint for timepoint in sorted_timepoints}
    boundaries = list(range(WRITE_WINDOW_MS, generation_end_ms, WRITE_WINDOW_MS))

    boundary_event_count = 0
    boundary_with_event_count = 0
    boundary_lane_action_count = 0
    hold_crossing_boundary_count = 0
    hold_crossing_lane_count = 0
    event_index = 0
    open_hold_mask = 0

    for boundary_ms in boundaries:
        while event_index < len(sorted_timepoints) and sorted_timepoints[event_index].time_ms < boundary_ms:
            open_hold_mask = _apply_open_hold_mask(open_hold_mask, sorted_timepoints[event_index])
            event_index += 1

        boundary_timepoint = timepoint_by_time.get(boundary_ms)
        if boundary_timepoint is not None:
            boundary_event_count += 1
            boundary_with_event_count += 1
            boundary_lane_action_count += _non_empty_lane_action_count(boundary_timepoint.lane_actions)

        open_lane_count = open_hold_mask.bit_count()
        if open_lane_count:
            hold_crossing_boundary_count += 1
            hold_crossing_lane_count += open_lane_count

    stitched_timepoints = _stitch_half_open_windows(sorted_timepoints, generation_end_ms=generation_end_ms)
    duplicate_timepoint_count, collision_timepoint_count, collision_lane_action_count = _stitch_collision_counts(
        stitched_timepoints,
        key_count=key_count,
    )

    return _WindowBoundaryMapSummary(
        boundary_count=len(boundaries),
        boundary_event_count=boundary_event_count,
        boundary_with_event_count=boundary_with_event_count,
        boundary_lane_action_count=boundary_lane_action_count,
        hold_crossing_boundary_count=hold_crossing_boundary_count,
        hold_crossing_lane_count=hold_crossing_lane_count,
        stitch_duplicate_timepoint_count=duplicate_timepoint_count,
        stitch_collision_timepoint_count=collision_timepoint_count,
        stitch_collision_lane_action_count=collision_lane_action_count,
        stitch_roundtrip_mismatch_count=_roundtrip_mismatch_count(sorted_timepoints, stitched_timepoints),
    )


def _stitch_half_open_windows(
    timepoints: Sequence[CanonicalTimepoint],
    *,
    generation_end_ms: int,
) -> list[CanonicalTimepoint]:
    stitched: list[CanonicalTimepoint] = []
    event_index = 0
    write_start = 0

    while write_start < generation_end_ms:
        write_end = min(write_start + WRITE_WINDOW_MS, generation_end_ms)
        while event_index < len(timepoints) and timepoints[event_index].time_ms < write_start:
            event_index += 1

        end_event_index = event_index
        while end_event_index < len(timepoints) and timepoints[end_event_index].time_ms < write_end:
            timepoint = timepoints[end_event_index]
            time_rel = timepoint.time_ms - write_start
            if time_rel < 0 or time_rel >= write_end - write_start:
                raise ValueError(f"timepoint outside half-open write region: {timepoint}")
            stitched.append(CanonicalTimepoint(write_start + time_rel, timepoint.lane_actions))
            end_event_index += 1

        event_index = end_event_index
        write_start += WRITE_WINDOW_MS

    return stitched


def _stitch_collision_counts(
    stitched_timepoints: Sequence[CanonicalTimepoint],
    *,
    key_count: int,
) -> tuple[int, int, int]:
    groups: dict[int, list[CanonicalTimepoint]] = defaultdict(list)
    for timepoint in stitched_timepoints:
        groups[timepoint.time_ms].append(timepoint)

    duplicate_timepoint_count = 0
    collision_timepoint_count = 0
    collision_lane_action_count = 0
    for group in groups.values():
        if len(group) <= 1:
            continue
        duplicate_timepoint_count += len(group) - 1
        collided_lanes = 0
        for lane in range(key_count):
            non_empty_count = sum(1 for timepoint in group if timepoint.lane_actions[lane] != LaneAction.NONE)
            if non_empty_count > 1:
                collided_lanes += 1
                collision_lane_action_count += non_empty_count
        if collided_lanes:
            collision_timepoint_count += 1

    return duplicate_timepoint_count, collision_timepoint_count, collision_lane_action_count


def _roundtrip_mismatch_count(
    original_timepoints: Sequence[CanonicalTimepoint],
    stitched_timepoints: Sequence[CanonicalTimepoint],
) -> int:
    original_counts = Counter(original_timepoints)
    stitched_counts = Counter(stitched_timepoints)
    all_timepoints = set(original_counts) | set(stitched_counts)
    return sum(abs(original_counts[timepoint] - stitched_counts[timepoint]) for timepoint in all_timepoints)


def _build_report(
    *,
    total_map_count: int,
    out_of_range_map_count: int,
    missing_red_timing_map_count: int,
    negative_time_hitobject_map_count: int,
    unsupported_compound_map_count: int,
    unsupported_compound_event_count: int,
    unsupported_compound_lane_action_count: int,
    four_state_unsupported_map_count: int,
    four_state_unsupported_event_count: int,
    four_state_unsupported_lane_action_count: int,
    accumulators: dict[str, _WindowBoundaryAccumulator],
) -> WindowBoundaryAuditReport:
    bin_reports = {label: accumulators[label].to_report() for label in DIFFICULTY_BIN_LABELS}
    boundary_count = sum(report.boundary_count for report in bin_reports.values())
    boundary_event_count = sum(report.boundary_event_count for report in bin_reports.values())
    boundary_with_event_count = sum(report.boundary_with_event_count for report in bin_reports.values())
    boundary_lane_action_count = sum(report.boundary_lane_action_count for report in bin_reports.values())
    hold_crossing_boundary_count = sum(report.hold_crossing_boundary_count for report in bin_reports.values())
    hold_crossing_lane_count = sum(report.hold_crossing_lane_count for report in bin_reports.values())
    stitch_duplicate_timepoint_count = sum(report.stitch_duplicate_timepoint_count for report in bin_reports.values())
    stitch_collision_timepoint_count = sum(report.stitch_collision_timepoint_count for report in bin_reports.values())
    stitch_collision_lane_action_count = sum(
        report.stitch_collision_lane_action_count for report in bin_reports.values()
    )
    stitch_roundtrip_mismatch_count = sum(report.stitch_roundtrip_mismatch_count for report in bin_reports.values())

    return WindowBoundaryAuditReport(
        total_map_count=total_map_count,
        audited_map_count=sum(report.map_count for report in bin_reports.values()),
        out_of_range_map_count=out_of_range_map_count,
        missing_red_timing_map_count=missing_red_timing_map_count,
        negative_time_hitobject_map_count=negative_time_hitobject_map_count,
        unsupported_compound_map_count=unsupported_compound_map_count,
        unsupported_compound_event_count=unsupported_compound_event_count,
        unsupported_compound_lane_action_count=unsupported_compound_lane_action_count,
        four_state_unsupported_map_count=four_state_unsupported_map_count,
        four_state_unsupported_event_count=four_state_unsupported_event_count,
        four_state_unsupported_lane_action_count=four_state_unsupported_lane_action_count,
        boundary_count=boundary_count,
        boundary_event_count=boundary_event_count,
        boundary_with_event_count=boundary_with_event_count,
        boundary_lane_action_count=boundary_lane_action_count,
        boundary_event_density=_rate(boundary_event_count, boundary_count),
        boundary_with_event_rate=_rate(boundary_with_event_count, boundary_count),
        boundary_lane_action_density=_rate(boundary_lane_action_count, boundary_count),
        hold_crossing_boundary_count=hold_crossing_boundary_count,
        hold_crossing_boundary_rate=_rate(hold_crossing_boundary_count, boundary_count),
        hold_crossing_lane_count=hold_crossing_lane_count,
        hold_crossing_lane_rate=_rate(hold_crossing_lane_count, boundary_count * 4),
        stitch_duplicate_timepoint_count=stitch_duplicate_timepoint_count,
        stitch_duplicate_timepoint_rate=_rate(stitch_duplicate_timepoint_count, boundary_count),
        stitch_collision_timepoint_count=stitch_collision_timepoint_count,
        stitch_collision_lane_action_count=stitch_collision_lane_action_count,
        stitch_collision_timepoint_rate=_rate(stitch_collision_timepoint_count, boundary_count),
        stitch_roundtrip_mismatch_count=stitch_roundtrip_mismatch_count,
        bins=bin_reports,
    )


def _resolve_generation_end_ms(
    map_input: OsuWindowBoundaryMapInput,
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
        "OsuWindowBoundaryMapInput requires generation_end_ms or audio_duration_ms "
        f"for {map_input.beatmap_path}",
    )


def _apply_open_hold_mask(open_hold_mask: int, timepoint: CanonicalTimepoint) -> int:
    for lane, action in enumerate(timepoint.lane_actions):
        lane_bit = 1 << lane
        if action == LaneAction.HOLD_START:
            open_hold_mask |= lane_bit
        elif action == LaneAction.HOLD_END:
            open_hold_mask &= ~lane_bit
    return open_hold_mask


def _non_empty_lane_action_count(lane_actions: Sequence[LaneAction]) -> int:
    return sum(1 for action in lane_actions if action != LaneAction.NONE)


def _rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator
