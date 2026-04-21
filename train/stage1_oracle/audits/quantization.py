from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from ..events.canonical import LaneAction, NegativeHitObjectTimeError, quantize_10ms_half_up
from ..osu.hitobjects import ManiaHitObject, ManiaHitObjectKind, parse_mania_hit_objects
from ..osu.timing import MissingRedTimingError, require_red_timing_points
from .token_statistics import DIFFICULTY_BIN_LABELS, difficulty_bin_label


QUANTIZER = "10ms_half_up"
QUANTIZER_GRID_MS = 10
QUANTIZER_TIE_BREAK = "floor((t_ms + 5) / 10)"
POST_QUANTIZATION_COLLISION_RATE_DENOMINATOR = "primitive_lane_action_count"
POST_QUANTIZATION_COLLISION_THRESHOLD_STATUS = "NOT_APPLICABLE_NO_DESIGN_THRESHOLD"


@dataclass(frozen=True)
class QuantizationMap:
    difficulty: float
    hitobjects: Sequence[ManiaHitObject]


@dataclass(frozen=True)
class OsuQuantizationMapInput:
    beatmap_path: str | Path
    difficulty: float


@dataclass(frozen=True)
class QuantizationErrorStats:
    mean: float
    p95: float
    max: float


@dataclass(frozen=True)
class QuantizationBinReport:
    label: str
    map_count: int
    key_count: int
    quantizer: str
    quantizer_grid_ms: int
    quantizer_tie_break: str
    timestamp_count: int
    quantization_error_ms: QuantizationErrorStats
    primitive_lane_action_count: int
    zero_length_hold_normalized_count: int
    zero_length_hold_normalized_map_count: int
    post_quantization_collision_timepoint_count: int
    post_quantization_collision_lane_time_cell_count: int
    post_quantization_collision_lane_action_count: int
    post_quantization_collision_affected_map_count: int
    post_quantization_collision_rate: float
    post_quantization_collision_rate_denominator: str


@dataclass(frozen=True)
class QuantizationAuditReport:
    total_map_count: int
    audited_map_count: int
    out_of_range_map_count: int
    missing_red_timing_map_count: int
    negative_time_hitobject_map_count: int
    key_count: int
    quantizer: str
    quantizer_grid_ms: int
    quantizer_tie_break: str
    timestamp_count: int
    quantization_error_ms: QuantizationErrorStats
    primitive_lane_action_count: int
    zero_length_hold_normalized_count: int
    zero_length_hold_normalized_map_count: int
    post_quantization_collision_timepoint_count: int
    post_quantization_collision_lane_time_cell_count: int
    post_quantization_collision_lane_action_count: int
    post_quantization_collision_affected_map_count: int
    post_quantization_collision_rate: float
    post_quantization_collision_rate_denominator: str
    bins: dict[str, QuantizationBinReport]


@dataclass(frozen=True)
class QuantizationGateDecision:
    status: str
    deterministic_quantizer: str
    quantizer_grid_ms: int
    quantizer_tie_break: str
    quantizer_status: str
    max_allowed_quantization_error_ms: int
    observed_max_quantization_error_ms: float
    quantization_error_status: str
    expected_key_count: int
    observed_key_count: int
    key_count_status: str
    eligible_map_count: int
    accounted_map_count: int
    coverage_status: str
    zero_length_hold_metric_status: str
    collision_metric_status: str
    post_quantization_collision_threshold_status: str
    post_quantization_collision_rate: float
    difficulty_source_status: str
    reproducibility_status: str
    failure_reasons: list[str]


@dataclass
class _QuantizationAccumulator:
    label: str
    key_count: int
    map_count: int = 0
    errors_ms: list[float] | None = None
    primitive_lane_action_count: int = 0
    zero_length_hold_normalized_count: int = 0
    zero_length_hold_normalized_map_count: int = 0
    post_quantization_collision_timepoint_count: int = 0
    post_quantization_collision_lane_time_cell_count: int = 0
    post_quantization_collision_lane_action_count: int = 0
    post_quantization_collision_affected_map_count: int = 0

    def __post_init__(self) -> None:
        self.errors_ms = [] if self.errors_ms is None else self.errors_ms

    def add_summary(self, summary: "_QuantizationMapSummary") -> None:
        self.map_count += 1
        assert self.errors_ms is not None
        self.errors_ms.extend(summary.errors_ms)
        self.primitive_lane_action_count += summary.primitive_lane_action_count
        self.zero_length_hold_normalized_count += summary.zero_length_hold_normalized_count
        if summary.zero_length_hold_normalized_count:
            self.zero_length_hold_normalized_map_count += 1
        self.post_quantization_collision_timepoint_count += summary.post_quantization_collision_timepoint_count
        self.post_quantization_collision_lane_time_cell_count += (
            summary.post_quantization_collision_lane_time_cell_count
        )
        self.post_quantization_collision_lane_action_count += summary.post_quantization_collision_lane_action_count
        if summary.post_quantization_collision_lane_time_cell_count:
            self.post_quantization_collision_affected_map_count += 1

    def to_report(self) -> QuantizationBinReport:
        errors = self.errors_ms or []
        return QuantizationBinReport(
            label=self.label,
            map_count=self.map_count,
            key_count=self.key_count,
            quantizer=QUANTIZER,
            quantizer_grid_ms=QUANTIZER_GRID_MS,
            quantizer_tie_break=QUANTIZER_TIE_BREAK,
            timestamp_count=len(errors),
            quantization_error_ms=_quantization_error_stats(errors),
            primitive_lane_action_count=self.primitive_lane_action_count,
            zero_length_hold_normalized_count=self.zero_length_hold_normalized_count,
            zero_length_hold_normalized_map_count=self.zero_length_hold_normalized_map_count,
            post_quantization_collision_timepoint_count=self.post_quantization_collision_timepoint_count,
            post_quantization_collision_lane_time_cell_count=self.post_quantization_collision_lane_time_cell_count,
            post_quantization_collision_lane_action_count=self.post_quantization_collision_lane_action_count,
            post_quantization_collision_affected_map_count=self.post_quantization_collision_affected_map_count,
            post_quantization_collision_rate=_rate(
                self.post_quantization_collision_lane_action_count,
                self.primitive_lane_action_count,
            ),
            post_quantization_collision_rate_denominator=POST_QUANTIZATION_COLLISION_RATE_DENOMINATOR,
        )


@dataclass(frozen=True)
class _QuantizationMapSummary:
    errors_ms: list[float]
    primitive_lane_action_count: int
    zero_length_hold_normalized_count: int
    post_quantization_collision_timepoint_count: int
    post_quantization_collision_lane_time_cell_count: int
    post_quantization_collision_lane_action_count: int


def audit_quantization(
    event_maps: Iterable[QuantizationMap],
    *,
    key_count: int = 4,
) -> QuantizationAuditReport:
    total_map_count = 0
    out_of_range_map_count = 0
    accumulators = {
        label: _QuantizationAccumulator(label=label, key_count=key_count) for label in DIFFICULTY_BIN_LABELS
    }

    for event_map in event_maps:
        total_map_count += 1
        label = difficulty_bin_label(event_map.difficulty)
        if label is None:
            out_of_range_map_count += 1
            continue

        summary = _summarize_quantization(event_map.hitobjects, key_count=key_count)
        accumulators[label].add_summary(summary)

    return _build_report(
        total_map_count=total_map_count,
        out_of_range_map_count=out_of_range_map_count,
        missing_red_timing_map_count=0,
        negative_time_hitobject_map_count=0,
        key_count=key_count,
        accumulators=accumulators,
    )


def audit_osu_quantization(
    map_inputs: Iterable[OsuQuantizationMapInput],
    *,
    key_count: int = 4,
) -> QuantizationAuditReport:
    total_map_count = 0
    out_of_range_map_count = 0
    missing_red_timing_map_count = 0
    negative_time_hitobject_map_count = 0
    accumulators = {
        label: _QuantizationAccumulator(label=label, key_count=key_count) for label in DIFFICULTY_BIN_LABELS
    }

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
            summary = _summarize_quantization(hitobjects, key_count=key_count)
        except NegativeHitObjectTimeError:
            negative_time_hitobject_map_count += 1
            continue
        accumulators[label].add_summary(summary)

    return _build_report(
        total_map_count=total_map_count,
        out_of_range_map_count=out_of_range_map_count,
        missing_red_timing_map_count=missing_red_timing_map_count,
        negative_time_hitobject_map_count=negative_time_hitobject_map_count,
        key_count=key_count,
        accumulators=accumulators,
    )


def build_quantization_gate_decision(
    report: QuantizationAuditReport,
    *,
    eligible_map_count: int,
    difficulty_source: str,
    expected_difficulty_source: str,
    code_dirty: bool,
    dirty_patch_sha256: str | None,
    expected_key_count: int = 4,
) -> QuantizationGateDecision:
    failure_reasons: list[str] = []

    accounted_map_count = (
        report.audited_map_count
        + report.out_of_range_map_count
        + report.missing_red_timing_map_count
        + report.negative_time_hitobject_map_count
    )
    coverage_status = "PASS"
    if report.total_map_count != eligible_map_count:
        coverage_status = "FAIL"
        failure_reasons.append("report total_map_count does not match eligible_map_count")
    if accounted_map_count != report.total_map_count:
        coverage_status = "FAIL"
        failure_reasons.append("audited plus filtered map counts do not account for total_map_count")

    key_count_status = "PASS" if report.key_count == expected_key_count else "FAIL"
    if key_count_status == "FAIL":
        failure_reasons.append(f"key_count {report.key_count} does not match expected {expected_key_count}")

    quantizer_status = (
        "PASS"
        if (
            report.quantizer == QUANTIZER
            and report.quantizer_grid_ms == QUANTIZER_GRID_MS
            and report.quantizer_tie_break == QUANTIZER_TIE_BREAK
        )
        else "FAIL"
    )
    if quantizer_status == "FAIL":
        failure_reasons.append("report quantizer metadata does not match the design-doc quantizer")

    observed_max_error = report.quantization_error_ms.max
    quantization_error_status = "PASS" if observed_max_error <= 5 else "FAIL"
    if quantization_error_status == "FAIL":
        failure_reasons.append("observed max quantization error exceeds 5ms")

    zero_length_hold_metric_status = (
        "PASS"
        if (
            isinstance(report.zero_length_hold_normalized_count, int)
            and isinstance(report.zero_length_hold_normalized_map_count, int)
        )
        else "FAIL"
    )
    if zero_length_hold_metric_status == "FAIL":
        failure_reasons.append("zero-length hold normalization metrics are not reported")

    collision_metric_status = (
        "PASS"
        if (
            isinstance(report.post_quantization_collision_timepoint_count, int)
            and isinstance(report.post_quantization_collision_lane_time_cell_count, int)
            and isinstance(report.post_quantization_collision_lane_action_count, int)
            and isinstance(report.post_quantization_collision_affected_map_count, int)
            and report.post_quantization_collision_rate_denominator
            == POST_QUANTIZATION_COLLISION_RATE_DENOMINATOR
        )
        else "FAIL"
    )
    if collision_metric_status == "FAIL":
        failure_reasons.append("post-quantization collision metrics are not reported with defined units")

    difficulty_source_status = "PASS" if difficulty_source == expected_difficulty_source else "FAIL"
    if difficulty_source_status == "FAIL":
        failure_reasons.append("difficulty source does not match approved source")

    if code_dirty:
        reproducibility_status = "PATCH_HASH_RECORDED" if dirty_patch_sha256 else "FAIL"
    else:
        reproducibility_status = "CLEAN"
    if reproducibility_status == "FAIL":
        failure_reasons.append("code dirty without recorded dirty patch sha256")

    return QuantizationGateDecision(
        status="PASS" if not failure_reasons else "FAIL",
        deterministic_quantizer=QUANTIZER,
        quantizer_grid_ms=QUANTIZER_GRID_MS,
        quantizer_tie_break=QUANTIZER_TIE_BREAK,
        quantizer_status=quantizer_status,
        max_allowed_quantization_error_ms=5,
        observed_max_quantization_error_ms=observed_max_error,
        quantization_error_status=quantization_error_status,
        expected_key_count=expected_key_count,
        observed_key_count=report.key_count,
        key_count_status=key_count_status,
        eligible_map_count=eligible_map_count,
        accounted_map_count=accounted_map_count,
        coverage_status=coverage_status,
        zero_length_hold_metric_status=zero_length_hold_metric_status,
        collision_metric_status=collision_metric_status,
        post_quantization_collision_threshold_status=POST_QUANTIZATION_COLLISION_THRESHOLD_STATUS,
        post_quantization_collision_rate=report.post_quantization_collision_rate,
        difficulty_source_status=difficulty_source_status,
        reproducibility_status=reproducibility_status,
        failure_reasons=failure_reasons,
    )


def _summarize_quantization(
    hitobjects: Sequence[ManiaHitObject],
    *,
    key_count: int,
) -> _QuantizationMapSummary:
    errors_ms: list[float] = []
    primitive_actions: dict[int, dict[int, list[LaneAction]]] = defaultdict(lambda: defaultdict(list))
    zero_length_hold_normalized_count = 0

    for hitobject in hitobjects:
        _validate_hitobject(hitobject, key_count)
        q_start = quantize_10ms_half_up(hitobject.start_time_ms)
        errors_ms.append(_quantization_error_ms(hitobject.start_time_ms, q_start))

        if hitobject.kind == ManiaHitObjectKind.TAP:
            primitive_actions[q_start][hitobject.lane].append(LaneAction.TAP)
            continue

        q_end = quantize_10ms_half_up(hitobject.end_time_ms)
        errors_ms.append(_quantization_error_ms(hitobject.end_time_ms, q_end))
        if q_end <= q_start:
            zero_length_hold_normalized_count += 1
            primitive_actions[q_start][hitobject.lane].append(LaneAction.TAP)
            continue

        primitive_actions[q_start][hitobject.lane].append(LaneAction.HOLD_START)
        primitive_actions[q_end][hitobject.lane].append(LaneAction.HOLD_END)

    primitive_lane_action_count = sum(
        len(actions)
        for lane_actions_by_time in primitive_actions.values()
        for actions in lane_actions_by_time.values()
    )
    collision_timepoints: set[int] = set()
    collision_lane_time_cell_count = 0
    collision_lane_action_count = 0
    for q_time, lane_actions_by_time in primitive_actions.items():
        for actions in lane_actions_by_time.values():
            if len(actions) <= 1:
                continue
            collision_timepoints.add(q_time)
            collision_lane_time_cell_count += 1
            collision_lane_action_count += len(actions)

    return _QuantizationMapSummary(
        errors_ms=errors_ms,
        primitive_lane_action_count=primitive_lane_action_count,
        zero_length_hold_normalized_count=zero_length_hold_normalized_count,
        post_quantization_collision_timepoint_count=len(collision_timepoints),
        post_quantization_collision_lane_time_cell_count=collision_lane_time_cell_count,
        post_quantization_collision_lane_action_count=collision_lane_action_count,
    )


def _build_report(
    *,
    total_map_count: int,
    out_of_range_map_count: int,
    missing_red_timing_map_count: int,
    negative_time_hitobject_map_count: int,
    key_count: int,
    accumulators: dict[str, _QuantizationAccumulator],
) -> QuantizationAuditReport:
    bin_reports = {label: accumulators[label].to_report() for label in DIFFICULTY_BIN_LABELS}
    all_errors: list[float] = []
    for accumulator in accumulators.values():
        all_errors.extend(accumulator.errors_ms or [])
    primitive_lane_action_count = sum(report.primitive_lane_action_count for report in bin_reports.values())
    zero_length_hold_normalized_count = sum(
        report.zero_length_hold_normalized_count for report in bin_reports.values()
    )
    zero_length_hold_normalized_map_count = sum(
        report.zero_length_hold_normalized_map_count for report in bin_reports.values()
    )
    collision_timepoint_count = sum(
        report.post_quantization_collision_timepoint_count for report in bin_reports.values()
    )
    collision_lane_time_cell_count = sum(
        report.post_quantization_collision_lane_time_cell_count for report in bin_reports.values()
    )
    collision_lane_action_count = sum(
        report.post_quantization_collision_lane_action_count for report in bin_reports.values()
    )
    collision_affected_map_count = sum(
        report.post_quantization_collision_affected_map_count for report in bin_reports.values()
    )

    return QuantizationAuditReport(
        total_map_count=total_map_count,
        audited_map_count=sum(report.map_count for report in bin_reports.values()),
        out_of_range_map_count=out_of_range_map_count,
        missing_red_timing_map_count=missing_red_timing_map_count,
        negative_time_hitobject_map_count=negative_time_hitobject_map_count,
        key_count=key_count,
        quantizer=QUANTIZER,
        quantizer_grid_ms=QUANTIZER_GRID_MS,
        quantizer_tie_break=QUANTIZER_TIE_BREAK,
        timestamp_count=len(all_errors),
        quantization_error_ms=_quantization_error_stats(all_errors),
        primitive_lane_action_count=primitive_lane_action_count,
        zero_length_hold_normalized_count=zero_length_hold_normalized_count,
        zero_length_hold_normalized_map_count=zero_length_hold_normalized_map_count,
        post_quantization_collision_timepoint_count=collision_timepoint_count,
        post_quantization_collision_lane_time_cell_count=collision_lane_time_cell_count,
        post_quantization_collision_lane_action_count=collision_lane_action_count,
        post_quantization_collision_affected_map_count=collision_affected_map_count,
        post_quantization_collision_rate=_rate(collision_lane_action_count, primitive_lane_action_count),
        post_quantization_collision_rate_denominator=POST_QUANTIZATION_COLLISION_RATE_DENOMINATOR,
        bins=bin_reports,
    )


def _validate_hitobject(hitobject: ManiaHitObject, key_count: int) -> None:
    if hitobject.start_time_ms < 0 or hitobject.end_time_ms < 0:
        raise NegativeHitObjectTimeError(f"hit object has negative time: {hitobject}")
    if not 0 <= hitobject.lane < key_count:
        raise ValueError(f"hit object lane {hitobject.lane} outside 0..{key_count - 1}: {hitobject}")


def _quantization_error_ms(raw_time_ms: float, quantized_time_ms: int) -> float:
    error_ms = abs(raw_time_ms - quantized_time_ms)
    return int(error_ms) if error_ms.is_integer() else error_ms


def _quantization_error_stats(errors_ms: Sequence[float]) -> QuantizationErrorStats:
    if not errors_ms:
        return QuantizationErrorStats(mean=0.0, p95=0, max=0)
    return QuantizationErrorStats(
        mean=sum(errors_ms) / len(errors_ms),
        p95=_nearest_rank_percentile(errors_ms, 95),
        max=max(errors_ms),
    )


def _nearest_rank_percentile(values: Sequence[float], percentile: int) -> float:
    if not values:
        return 0
    sorted_values = sorted(values)
    index = math.ceil((percentile / 100) * len(sorted_values)) - 1
    return sorted_values[max(0, min(index, len(sorted_values) - 1))]


def _rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator
