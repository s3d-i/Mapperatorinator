from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Sequence

import torch

from .replay import ReplayError, close_labels_from_tokens, replay_state_tensors
from .vocab import KEY_COUNT, LaneAction, MapperV1Vocab, coerce_lane_action


MAPPER_WRITE_MS = 8000
MAPPER_DENSITY_FRAMES = 400
MAPPER_DENSITY_FRAME_MS = 20


class MapperTokenizationError(ValueError):
    pass


class CrossWindowLongNoteError(MapperTokenizationError):
    pass


class UnsupportedMapperActionError(MapperTokenizationError):
    pass


@dataclass(frozen=True)
class MapperTimepoint:
    time_ms: int
    lane_actions: tuple[LaneAction, LaneAction, LaneAction, LaneAction]


@dataclass(frozen=True)
class TokenizedMapperWindow:
    target_ids: list[int]
    write_start_ms: int
    write_end_ms: int
    teacher_current_ms: torch.Tensor
    teacher_open_mask: torch.Tensor
    teacher_open_age_ms: torch.Tensor
    close_labels: torch.Tensor
    close_label_mask: torch.Tensor

    @property
    def seq_len(self) -> int:
        return len(self.target_ids)

    def target_tensor(self) -> torch.Tensor:
        return torch.tensor(self.target_ids, dtype=torch.long)


def encode_mapper_window(
    timepoints: Sequence[MapperTimepoint | Any],
    *,
    vocab: MapperV1Vocab,
    write_start_ms: int,
    write_end_ms: int,
) -> TokenizedMapperWindow:
    write_start_ms = int(write_start_ms)
    write_end_ms = int(write_end_ms)
    if write_end_ms <= write_start_ms:
        raise ValueError(f"write_end_ms must be after write_start_ms: {write_start_ms}..{write_end_ms}")
    if (write_end_ms - write_start_ms) % 10 != 0:
        raise ValueError("mapper write window must align to the 10ms grid")

    grouped = _group_timepoints(timepoints)
    target_ids = [vocab.bos_id]
    current_ms = write_start_ms
    for timepoint in grouped:
        if not write_start_ms <= timepoint.time_ms < write_end_ms:
            raise MapperTokenizationError(f"timepoint outside write window: {timepoint}")
        if timepoint.time_ms % 10 != 0:
            raise MapperTokenizationError(f"timepoint must be on the 10ms grid: {timepoint.time_ms}")
        delta_ms = timepoint.time_ms - current_ms
        if delta_ms < 0:
            raise MapperTokenizationError(f"timepoints must be nondecreasing after grouping: {grouped}")
        target_ids.extend(vocab.time_shift_token_id(value) for value in vocab.decompose_time_shift_delta(delta_ms))
        target_ids.append(vocab.encode_event(timepoint.lane_actions))
        current_ms = timepoint.time_ms

    target_ids.extend(
        vocab.time_shift_token_id(value)
        for value in vocab.decompose_time_shift_delta(write_end_ms - current_ms)
    )
    target_ids.append(vocab.eos_id)

    try:
        state_tensors = replay_state_tensors(
            target_ids,
            vocab=vocab,
            write_start_ms=write_start_ms,
            write_end_ms=write_end_ms,
        )
    except ReplayError as exc:
        message = str(exc)
        if "HOLD_END is illegal on closed lane" in message:
            raise CrossWindowLongNoteError(f"window requires carry-in LN state: {message}") from exc
        if "EOS requires all lanes closed" in message or "TIME_SHIFT to write_end_ms while an LN is open" in message:
            raise CrossWindowLongNoteError(f"window requires carry-out LN state: {message}") from exc
        raise MapperTokenizationError(message) from exc

    close_labels, close_label_mask = close_labels_from_tokens(
        target_ids,
        vocab=vocab,
        write_start_ms=write_start_ms,
        write_end_ms=write_end_ms,
    )
    return TokenizedMapperWindow(
        target_ids=target_ids,
        write_start_ms=write_start_ms,
        write_end_ms=write_end_ms,
        teacher_current_ms=state_tensors["current_ms"],
        teacher_open_mask=state_tensors["open_mask"],
        teacher_open_age_ms=state_tensors["open_age_ms"],
        close_labels=close_labels,
        close_label_mask=close_label_mask,
    )


def window_timepoints(
    timepoints: Sequence[MapperTimepoint | Any],
    *,
    write_start_ms: int,
    write_end_ms: int,
) -> list[MapperTimepoint]:
    return [
        timepoint
        for timepoint in _group_timepoints(timepoints)
        if int(write_start_ms) <= timepoint.time_ms < int(write_end_ms)
    ]


def hitobjects_to_mapper_timepoints(hitobjects: Sequence[Any], *, key_count: int = KEY_COUNT) -> list[MapperTimepoint]:
    if key_count != KEY_COUNT:
        raise ValueError(f"mapper v1 supports only {KEY_COUNT}K, got {key_count}")

    primitive_actions: dict[int, dict[int, list[LaneAction]]] = defaultdict(lambda: defaultdict(list))
    for hitobject in hitobjects:
        lane = int(hitobject.lane)
        if not 0 <= lane < KEY_COUNT:
            raise ValueError(f"hit object lane outside 4K range: {lane}")
        start_ms = quantize_10ms_half_up(float(hitobject.start_time_ms))
        kind = getattr(getattr(hitobject, "kind", None), "value", getattr(hitobject, "kind", None))
        if kind == "TAP":
            primitive_actions[start_ms][lane].append(LaneAction.TAP)
            continue
        if kind != "HOLD":
            raise ValueError(f"unsupported mapper v1 hit object kind: {kind}")
        end_ms = quantize_10ms_half_up(float(hitobject.end_time_ms))
        if end_ms <= start_ms:
            primitive_actions[start_ms][lane].append(LaneAction.TAP)
            continue
        primitive_actions[start_ms][lane].append(LaneAction.HOLD_START)
        primitive_actions[end_ms][lane].append(LaneAction.HOLD_END)

    timepoints: list[MapperTimepoint] = []
    for time_ms in sorted(primitive_actions):
        lane_actions = tuple(
            _merge_lane_primitive_actions(primitive_actions[time_ms].get(lane, []), time_ms=time_ms, lane=lane)
            for lane in range(KEY_COUNT)
        )
        if any(action != LaneAction.NONE for action in lane_actions):
            timepoints.append(MapperTimepoint(time_ms=time_ms, lane_actions=lane_actions))  # type: ignore[arg-type]
    return timepoints


def tokenize_hitobjects_window(
    hitobjects: Sequence[Any],
    *,
    vocab: MapperV1Vocab,
    write_start_ms: int,
    write_end_ms: int,
) -> TokenizedMapperWindow:
    timepoints = hitobjects_to_mapper_timepoints(hitobjects)
    reason = cross_window_ln_state_reason(
        timepoints,
        write_start_ms=write_start_ms,
        write_end_ms=write_end_ms,
    )
    if reason is not None:
        raise CrossWindowLongNoteError(f"window requires {reason} LN state")
    return encode_mapper_window(
        window_timepoints(
            timepoints,
            write_start_ms=write_start_ms,
            write_end_ms=write_end_ms,
        ),
        vocab=vocab,
        write_start_ms=write_start_ms,
        write_end_ms=write_end_ms,
    )


def cross_window_ln_state_reason(
    timepoints: Sequence[MapperTimepoint | Any],
    *,
    write_start_ms: int,
    write_end_ms: int,
) -> str | None:
    """Return the V1 carry-state reason that makes a mapper window ineligible."""

    grouped = _group_timepoints(timepoints)
    open_mask = [False] * KEY_COUNT
    for timepoint in grouped:
        if timepoint.time_ms >= int(write_start_ms):
            break
        _apply_timepoint_to_open_mask(open_mask, timepoint)
    if any(open_mask):
        return "carry-in"

    for timepoint in grouped:
        if timepoint.time_ms < int(write_start_ms):
            continue
        if timepoint.time_ms >= int(write_end_ms):
            break
        _apply_timepoint_to_open_mask(open_mask, timepoint)
    if any(open_mask):
        return "carry-out"
    return None


def quantize_10ms_half_up(time_ms: float) -> int:
    if time_ms < 0:
        raise ValueError(f"cannot quantize negative time: {time_ms}")
    return int(10 * math.floor((time_ms + 5) / 10))


def _group_timepoints(timepoints: Sequence[MapperTimepoint | Any]) -> list[MapperTimepoint]:
    grouped: dict[int, list[LaneAction]] = {}
    for raw_timepoint in sorted((_coerce_timepoint(item) for item in timepoints), key=lambda item: item.time_ms):
        actions = grouped.setdefault(raw_timepoint.time_ms, [LaneAction.NONE] * KEY_COUNT)
        for lane, action in enumerate(raw_timepoint.lane_actions):
            if action == LaneAction.NONE:
                continue
            if actions[lane] != LaneAction.NONE:
                raise UnsupportedMapperActionError(
                    f"multiple same-lane actions at {raw_timepoint.time_ms}ms lane {lane}",
                )
            actions[lane] = action
    return [
        MapperTimepoint(time_ms=time_ms, lane_actions=tuple(actions))  # type: ignore[arg-type]
        for time_ms, actions in sorted(grouped.items())
        if any(action != LaneAction.NONE for action in actions)
    ]


def _coerce_timepoint(timepoint: MapperTimepoint | Any) -> MapperTimepoint:
    if isinstance(timepoint, MapperTimepoint):
        return timepoint
    if not hasattr(timepoint, "time_ms") or not hasattr(timepoint, "lane_actions"):
        raise TypeError(f"mapper timepoint must expose time_ms and lane_actions: {timepoint!r}")
    actions = tuple(coerce_lane_action(action) for action in timepoint.lane_actions)
    if len(actions) != KEY_COUNT:
        raise ValueError(f"mapper timepoint must contain {KEY_COUNT} lane actions: {actions}")
    return MapperTimepoint(time_ms=int(timepoint.time_ms), lane_actions=actions)  # type: ignore[arg-type]


def _merge_lane_primitive_actions(actions: Sequence[LaneAction], *, time_ms: int, lane: int) -> LaneAction:
    if not actions:
        return LaneAction.NONE
    if len(actions) == 1:
        return actions[0]
    raise UnsupportedMapperActionError(
        f"mapper v1 cannot represent same-lane compound actions at {time_ms}ms lane {lane}: {list(actions)}",
    )


def _apply_timepoint_to_open_mask(open_mask: list[bool], timepoint: MapperTimepoint) -> None:
    for lane, action in enumerate(timepoint.lane_actions):
        if action == LaneAction.HOLD_START:
            if open_mask[lane]:
                raise MapperTokenizationError(f"HOLD_START on open lane {lane} at {timepoint.time_ms}ms")
            open_mask[lane] = True
        elif action == LaneAction.HOLD_END:
            if not open_mask[lane]:
                raise MapperTokenizationError(f"HOLD_END on closed lane {lane} at {timepoint.time_ms}ms")
            open_mask[lane] = False
        elif action == LaneAction.TAP and open_mask[lane]:
            raise MapperTokenizationError(f"TAP on open lane {lane} at {timepoint.time_ms}ms")
