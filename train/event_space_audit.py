from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .canonical_events import CanonicalTimepoint, LaneAction
from .canonical_events import NegativeHitObjectTimeError
from .canonical_events import UnsupportedCompoundLaneActionError
from .canonical_events import build_canonical_quantized_events
from .osu_hitobjects import parse_mania_hit_objects
from .osu_timing import MissingRedTimingError, require_red_timing_points


@dataclass(frozen=True)
class EventSpaceAuditReport:
    map_count: int
    total_timepoints: int
    total_non_empty_lane_actions: int
    action_counts: dict[LaneAction, int]
    end_tap_frequency: float
    end_start_frequency: float
    same_lane_compound_event_count: int
    same_lane_compound_event_frequency: float
    four_state_unsupported_map_count: int
    four_state_unsupported_lane_action_count: int
    four_state_unsupported_lane_action_rate: float
    top_k_event_counts: list[tuple[tuple[LaneAction, ...], int]]
    top_k_event_coverage: float
    rare_event_count: int


@dataclass(frozen=True)
class OsuEventSpaceAuditReport:
    total_map_count: int
    audited_map_count: int
    missing_red_timing_map_count: int
    negative_time_hitobject_map_count: int
    unsupported_compound_map_count: int
    zero_length_hold_normalized_count: int
    event_space: EventSpaceAuditReport


FOUR_STATE_UNSUPPORTED_ACTIONS = frozenset({LaneAction.END_TAP, LaneAction.END_START})


def audit_event_space(
    event_maps: Iterable[Sequence[CanonicalTimepoint]],
    *,
    top_k: int = 50,
    rare_event_threshold: int = 1,
) -> EventSpaceAuditReport:
    action_counts = {action: 0 for action in LaneAction}
    event_signature_counts: Counter[tuple[LaneAction, ...]] = Counter()
    map_count = 0
    total_timepoints = 0
    total_non_empty_lane_actions = 0
    same_lane_compound_event_count = 0
    four_state_unsupported_map_count = 0
    four_state_unsupported_lane_action_count = 0

    for timepoints in event_maps:
        map_count += 1
        map_has_four_state_unsupported = False

        for timepoint in timepoints:
            total_timepoints += 1
            event_signature_counts[timepoint.lane_actions] += 1

            for action in timepoint.lane_actions:
                if action == LaneAction.NONE:
                    continue

                action_counts[action] += 1
                total_non_empty_lane_actions += 1

                if action in FOUR_STATE_UNSUPPORTED_ACTIONS:
                    same_lane_compound_event_count += 1
                    four_state_unsupported_lane_action_count += 1
                    map_has_four_state_unsupported = True

        if map_has_four_state_unsupported:
            four_state_unsupported_map_count += 1

    top_k_event_counts = event_signature_counts.most_common(top_k)
    top_k_covered_count = sum(count for _, count in top_k_event_counts)

    return EventSpaceAuditReport(
        map_count=map_count,
        total_timepoints=total_timepoints,
        total_non_empty_lane_actions=total_non_empty_lane_actions,
        action_counts=action_counts,
        end_tap_frequency=_rate(action_counts[LaneAction.END_TAP], total_non_empty_lane_actions),
        end_start_frequency=_rate(action_counts[LaneAction.END_START], total_non_empty_lane_actions),
        same_lane_compound_event_count=same_lane_compound_event_count,
        same_lane_compound_event_frequency=_rate(same_lane_compound_event_count, total_non_empty_lane_actions),
        four_state_unsupported_map_count=four_state_unsupported_map_count,
        four_state_unsupported_lane_action_count=four_state_unsupported_lane_action_count,
        four_state_unsupported_lane_action_rate=_rate(
            four_state_unsupported_lane_action_count,
            total_non_empty_lane_actions,
        ),
        top_k_event_counts=top_k_event_counts,
        top_k_event_coverage=_rate(top_k_covered_count, total_timepoints),
        rare_event_count=sum(1 for count in event_signature_counts.values() if count <= rare_event_threshold),
    )


def audit_osu_event_space(
    beatmap_paths: Iterable[str | Path],
    *,
    key_count: int = 4,
    top_k: int = 50,
    rare_event_threshold: int = 1,
) -> OsuEventSpaceAuditReport:
    event_maps: list[list[CanonicalTimepoint]] = []
    total_map_count = 0
    missing_red_timing_map_count = 0
    negative_time_hitobject_map_count = 0
    unsupported_compound_map_count = 0
    zero_length_hold_normalized_count = 0

    for beatmap_path in beatmap_paths:
        total_map_count += 1
        try:
            require_red_timing_points(beatmap_path)
        except MissingRedTimingError:
            missing_red_timing_map_count += 1
            continue

        hitobjects = parse_mania_hit_objects(beatmap_path, expected_key_count=key_count)
        try:
            build_result = build_canonical_quantized_events(hitobjects, key_count=key_count)
        except NegativeHitObjectTimeError:
            negative_time_hitobject_map_count += 1
            continue
        except UnsupportedCompoundLaneActionError:
            unsupported_compound_map_count += 1
            continue

        event_maps.append(build_result.timepoints)
        zero_length_hold_normalized_count += build_result.zero_length_hold_normalized_count

    return OsuEventSpaceAuditReport(
        total_map_count=total_map_count,
        audited_map_count=len(event_maps),
        missing_red_timing_map_count=missing_red_timing_map_count,
        negative_time_hitobject_map_count=negative_time_hitobject_map_count,
        unsupported_compound_map_count=unsupported_compound_map_count,
        zero_length_hold_normalized_count=zero_length_hold_normalized_count,
        event_space=audit_event_space(
            event_maps,
            top_k=top_k,
            rare_event_threshold=rare_event_threshold,
        ),
    )


def _rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator
