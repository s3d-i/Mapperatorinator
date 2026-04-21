from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .canonical import CanonicalTimepoint, LaneAction


@dataclass(frozen=True)
class DecodedWindowEvents:
    write_start_ms: int
    timepoints: Sequence[CanonicalTimepoint]


@dataclass(frozen=True)
class StitchedDecodeResult:
    timepoints: list[CanonicalTimepoint]
    boundary_open_masks: dict[int, int]
    final_open_hold_mask: int
    invalid_hold_end_count: int
    same_lane_collision_count: int

    @property
    def invalid_hold_transition_count(self) -> int:
        return self.invalid_hold_end_count + self.same_lane_collision_count


def stitch_decoded_windows(
    windows: Sequence[DecodedWindowEvents],
    *,
    initial_open_hold_mask: int = 0,
) -> StitchedDecodeResult:
    stitched: list[CanonicalTimepoint] = []
    boundary_open_masks: dict[int, int] = {}
    open_hold_mask = initial_open_hold_mask
    invalid_hold_end_count = 0
    same_lane_collision_count = 0

    for window in sorted(windows, key=lambda item: item.write_start_ms):
        boundary_open_masks[window.write_start_ms] = open_hold_mask
        for timepoint in window.timepoints:
            if timepoint.time_ms < 0:
                raise ValueError(f"decoded relative time must be non-negative: {timepoint}")
            absolute_timepoint = CanonicalTimepoint(
                time_ms=window.write_start_ms + timepoint.time_ms,
                lane_actions=timepoint.lane_actions,
            )
            stitched.append(absolute_timepoint)
            open_hold_mask, invalid_end_count, collision_count = _apply_open_hold_mask(open_hold_mask, absolute_timepoint)
            invalid_hold_end_count += invalid_end_count
            same_lane_collision_count += collision_count

    return StitchedDecodeResult(
        timepoints=stitched,
        boundary_open_masks=boundary_open_masks,
        final_open_hold_mask=open_hold_mask,
        invalid_hold_end_count=invalid_hold_end_count,
        same_lane_collision_count=same_lane_collision_count,
    )


def _apply_open_hold_mask(open_hold_mask: int, timepoint: CanonicalTimepoint) -> tuple[int, int, int]:
    invalid_hold_end_count = 0
    same_lane_collision_count = 0
    for lane, action in enumerate(timepoint.lane_actions):
        lane_bit = 1 << lane
        is_open = (open_hold_mask & lane_bit) != 0
        if action == LaneAction.TAP:
            if is_open:
                same_lane_collision_count += 1
        elif action == LaneAction.HOLD_START:
            if is_open:
                same_lane_collision_count += 1
            else:
                open_hold_mask |= lane_bit
        elif action == LaneAction.HOLD_END:
            if not is_open:
                invalid_hold_end_count += 1
            open_hold_mask &= ~lane_bit
    return open_hold_mask, invalid_hold_end_count, same_lane_collision_count
