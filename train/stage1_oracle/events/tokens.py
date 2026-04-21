from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from .canonical import CanonicalTimepoint, LaneAction
from .windowing import KEY_COUNT, MAX_TS_TOKEN_MS


FROZEN_EVENT_ACTIONS = (
    LaneAction.NONE,
    LaneAction.TAP,
    LaneAction.HOLD_START,
    LaneAction.HOLD_END,
)
ACTION_TO_CODE = {action: code for code, action in enumerate(FROZEN_EVENT_ACTIONS)}
CODE_TO_ACTION = {code: action for action, code in ACTION_TO_CODE.items()}


@dataclass(frozen=True)
class TokenizedWindow:
    condition_ids: list[int]
    target_ids: list[int]


class Stage1Vocab:
    def __init__(self) -> None:
        token_names: list[str] = ["PAD", "BOS", "EOS"]
        token_names.extend(f"DIFF_{2.0 + bucket * 0.25:.2f}" for bucket in range(17))
        token_names.extend(f"OPEN_{mask:04b}" for mask in range(16))
        token_names.extend(f"TS_{value}" for value in range(0, MAX_TS_TOKEN_MS + 10, 10))

        event_start = len(token_names)
        self._event_actions_by_id: dict[int, tuple[LaneAction, ...]] = {}
        self._event_id_by_actions: dict[tuple[LaneAction, ...], int] = {}
        for offset, code in enumerate(range(1, 4**KEY_COUNT)):
            actions = self._decode_event_code(code)
            token_names.append(f"EV_{''.join(str(ACTION_TO_CODE[action]) for action in actions)}")
            event_id = event_start + offset
            self._event_actions_by_id[event_id] = actions
            self._event_id_by_actions[actions] = event_id

        self.id_to_token = tuple(token_names)
        self.token_to_id = {token: token_id for token_id, token in enumerate(self.id_to_token)}

        self.pad_id = self.token_to_id["PAD"]
        self.bos_id = self.token_to_id["BOS"]
        self.eos_id = self.token_to_id["EOS"]
        self.diff_token_ids = tuple(self.token_to_id[f"DIFF_{2.0 + bucket * 0.25:.2f}"] for bucket in range(17))
        self.open_token_ids = tuple(self.token_to_id[f"OPEN_{mask:04b}"] for mask in range(16))
        self.ts_token_ids = tuple(self.token_to_id[f"TS_{value}"] for value in range(0, MAX_TS_TOKEN_MS + 10, 10))
        self.event_token_ids = tuple(range(event_start, len(self.id_to_token)))

    @property
    def size(self) -> int:
        return len(self.id_to_token)

    def token_name(self, token_id: int) -> str:
        return self.id_to_token[token_id]

    def diff_token_id(self, bucket_id: int) -> int:
        if not 0 <= bucket_id < len(self.diff_token_ids):
            raise ValueError(f"difficulty bucket outside supported range: {bucket_id}")
        return self.diff_token_ids[bucket_id]

    def open_token_id(self, open_hold_mask: int) -> int:
        if not 0 <= open_hold_mask < len(self.open_token_ids):
            raise ValueError(f"open hold mask outside 4K range: {open_hold_mask}")
        return self.open_token_ids[open_hold_mask]

    def ts_token_id(self, delta_ms: int) -> int:
        if delta_ms % 10 != 0 or not 0 <= delta_ms <= MAX_TS_TOKEN_MS:
            raise ValueError(f"TS value outside vocabulary: {delta_ms}")
        return self.ts_token_ids[delta_ms // 10]

    def ts_value(self, token_id: int) -> int:
        if token_id not in self.ts_token_ids:
            raise ValueError(f"not a TS token: {token_id}")
        return (token_id - self.ts_token_ids[0]) * 10

    def is_ts_token(self, token_id: int) -> bool:
        return self.ts_token_ids[0] <= token_id <= self.ts_token_ids[-1]

    def is_event_token(self, token_id: int) -> bool:
        return token_id in self._event_actions_by_id

    def encode_timepoint_event(self, lane_actions: Sequence[LaneAction]) -> int:
        actions = tuple(lane_actions)
        if len(actions) != KEY_COUNT:
            raise ValueError(f"event must contain exactly {KEY_COUNT} lane actions: {actions}")
        if all(action == LaneAction.NONE for action in actions):
            raise ValueError("EV token cannot represent an all-empty timepoint")
        for action in actions:
            if action not in ACTION_TO_CODE:
                raise ValueError(f"unsupported Stage 1 lane action: {action}")
        return self._event_id_by_actions[actions]

    def decode_event_token(self, token_id: int) -> tuple[LaneAction, ...]:
        try:
            return self._event_actions_by_id[token_id]
        except KeyError as exc:
            raise ValueError(f"not an event token: {token_id}") from exc

    def difficulty_bucket_id(self, stars: float) -> int:
        if not 2.0 <= stars <= 6.0:
            raise ValueError(f"difficulty outside supported 2.0*..6.0* range: {stars}")
        bucket_id = int(math.floor(((stars - 2.0) / 0.25) + 0.5))
        if not 0 <= bucket_id <= 16:
            raise ValueError(f"difficulty outside supported 2.0*..6.0* range: {stars}")
        return bucket_id

    def _decode_event_code(self, code: int) -> tuple[LaneAction, ...]:
        actions: list[LaneAction] = []
        value = code
        for _ in range(KEY_COUNT):
            actions.append(CODE_TO_ACTION[value % 4])
            value //= 4
        return tuple(actions)


def decompose_ts_delta(delta_ms: int) -> list[int]:
    if delta_ms < 0:
        raise ValueError(f"TS delta must be non-negative: {delta_ms}")
    if delta_ms % 10 != 0:
        raise ValueError(f"TS delta must be on the 10ms grid: {delta_ms}")
    if delta_ms <= MAX_TS_TOKEN_MS:
        return [delta_ms]

    values = [MAX_TS_TOKEN_MS] * (delta_ms // MAX_TS_TOKEN_MS)
    remainder = delta_ms % MAX_TS_TOKEN_MS
    if remainder:
        values.append(remainder)
    return values


def encode_window_tokens(
    timepoints: Sequence[CanonicalTimepoint],
    *,
    vocab: Stage1Vocab,
    write_start_ms: int,
    write_end_ms: int,
    difficulty: float,
    open_hold_mask: int,
) -> TokenizedWindow:
    write_duration_ms = write_end_ms - write_start_ms
    if write_duration_ms <= 0:
        raise ValueError(f"write duration must be positive: {write_duration_ms}")

    condition_ids = [
        vocab.bos_id,
        vocab.diff_token_id(vocab.difficulty_bucket_id(difficulty)),
        vocab.open_token_id(open_hold_mask),
    ]
    target_ids: list[int] = []
    previous_time_rel: int | None = None
    for timepoint in sorted(timepoints, key=lambda item: item.time_ms):
        time_rel = timepoint.time_ms - write_start_ms
        if not 0 <= time_rel < write_duration_ms:
            raise ValueError(f"timepoint outside write region: {timepoint}")
        delta_ms = time_rel if previous_time_rel is None else time_rel - previous_time_rel
        if previous_time_rel is not None and delta_ms <= 0:
            raise ValueError(f"window timepoints must strictly increase: {timepoints}")
        target_ids.extend(vocab.ts_token_id(value) for value in decompose_ts_delta(delta_ms))
        target_ids.append(vocab.encode_timepoint_event(timepoint.lane_actions))
        previous_time_rel = time_rel

    target_ids.append(vocab.eos_id)
    return TokenizedWindow(
        condition_ids=condition_ids,
        target_ids=target_ids,
    )


def decode_target_tokens(
    token_ids: Sequence[int],
    *,
    vocab: Stage1Vocab,
    write_duration_ms: int,
) -> list[CanonicalTimepoint]:
    timepoints: list[CanonicalTimepoint] = []
    current_time_rel = 0
    pending_delta = 0
    for token_id in token_ids:
        if token_id == vocab.eos_id:
            break
        if vocab.is_ts_token(token_id):
            pending_delta += vocab.ts_value(token_id)
            continue
        if not vocab.is_event_token(token_id):
            raise ValueError(f"unexpected target token: {token_id}")
        event_time_rel = current_time_rel + pending_delta if timepoints else pending_delta
        if not 0 <= event_time_rel < write_duration_ms:
            raise ValueError(f"decoded event outside write duration: {event_time_rel}")
        timepoints.append(
            CanonicalTimepoint(
                time_ms=event_time_rel,
                lane_actions=vocab.decode_event_token(token_id),
            ),
        )
        current_time_rel = event_time_rel
        pending_delta = 0
    return timepoints
