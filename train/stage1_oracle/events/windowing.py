from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .canonical import CanonicalTimepoint, LaneAction, ceil_10ms


WRITE_WINDOW_MS = 8000
INPUT_LEFT_CONTEXT_MS = 2000
INPUT_DURATION_MS = 12000
INPUT_RIGHT_CONTEXT_MS = INPUT_DURATION_MS - INPUT_LEFT_CONTEXT_MS
MAX_TS_TOKEN_MS = 1000
KEY_COUNT = 4


@dataclass(frozen=True)
class WindowSpec:
    write_start_ms: int
    write_end_ms: int
    input_start_ms: int
    input_end_ms: int

    @property
    def write_duration_ms(self) -> int:
        return self.write_end_ms - self.write_start_ms

    @property
    def input_duration_ms(self) -> int:
        return self.input_end_ms - self.input_start_ms


def compute_generation_end_ms(
    audio_duration_ms: float,
    timepoints: Sequence[CanonicalTimepoint],
) -> int:
    max_event_time_plus_grid = max((timepoint.time_ms + 10 for timepoint in timepoints), default=0)
    return max(ceil_10ms(audio_duration_ms), max_event_time_plus_grid)


def iter_window_specs(generation_end_ms: int) -> list[WindowSpec]:
    if generation_end_ms < 0:
        raise ValueError(f"generation_end_ms must be non-negative: {generation_end_ms}")

    windows: list[WindowSpec] = []
    write_start = 0
    while write_start < generation_end_ms:
        write_end = min(write_start + WRITE_WINDOW_MS, generation_end_ms)
        input_start = write_start - INPUT_LEFT_CONTEXT_MS
        windows.append(
            WindowSpec(
                write_start_ms=write_start,
                write_end_ms=write_end,
                input_start_ms=input_start,
                input_end_ms=write_start + INPUT_RIGHT_CONTEXT_MS,
            ),
        )
        write_start += WRITE_WINDOW_MS
    return windows


def window_timepoints(
    timepoints: Sequence[CanonicalTimepoint],
    *,
    write_start_ms: int,
    write_end_ms: int,
) -> list[CanonicalTimepoint]:
    return [
        timepoint
        for timepoint in sorted(timepoints, key=lambda item: item.time_ms)
        if write_start_ms <= timepoint.time_ms < write_end_ms
    ]


def open_hold_mask_at_write_start(
    timepoints: Sequence[CanonicalTimepoint],
    write_start_ms: int,
) -> int:
    open_hold_mask = 0
    for timepoint in sorted(timepoints, key=lambda item: item.time_ms):
        if timepoint.time_ms >= write_start_ms:
            break
        open_hold_mask = apply_open_hold_mask(open_hold_mask, timepoint)
    return open_hold_mask


def apply_open_hold_mask(open_hold_mask: int, timepoint: CanonicalTimepoint) -> int:
    for lane, action in enumerate(timepoint.lane_actions):
        lane_bit = 1 << lane
        if action == LaneAction.HOLD_START:
            open_hold_mask |= lane_bit
        elif action == LaneAction.HOLD_END:
            open_hold_mask &= ~lane_bit
    return open_hold_mask
