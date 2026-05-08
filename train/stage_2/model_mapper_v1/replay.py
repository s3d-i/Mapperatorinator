from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .vocab import KEY_COUNT, LaneAction, MapperV1Vocab


class ReplayError(ValueError):
    pass


@dataclass(frozen=True)
class MapperReplayState:
    position: int
    current_ms: int
    open_mask: tuple[bool, bool, bool, bool]
    open_age_ms: tuple[int, int, int, int]

    @property
    def open_mask_bits(self) -> int:
        return open_mask_tuple_to_bits(self.open_mask)


def initial_replay_state(write_start_ms: int) -> MapperReplayState:
    return MapperReplayState(
        position=-1,
        current_ms=int(write_start_ms),
        open_mask=(False, False, False, False),
        open_age_ms=(0, 0, 0, 0),
    )


def replay_tokens(
    token_ids: Sequence[int],
    *,
    vocab: MapperV1Vocab,
    write_start_ms: int,
    write_end_ms: int,
    validate_final: bool = False,
) -> list[MapperReplayState]:
    if int(write_end_ms) <= int(write_start_ms):
        raise ValueError(f"write_end_ms must be after write_start_ms: {write_start_ms}..{write_end_ms}")

    state = initial_replay_state(write_start_ms)
    states: list[MapperReplayState] = []
    for position, token_id in enumerate(token_ids):
        state = transition_replay_state(
            state,
            int(token_id),
            position=position,
            vocab=vocab,
            write_start_ms=int(write_start_ms),
            write_end_ms=int(write_end_ms),
        )
        states.append(state)

    if validate_final:
        if not token_ids:
            raise ReplayError("token sequence is empty")
        if int(token_ids[-1]) != vocab.eos_id:
            raise ReplayError("token sequence must end with EOS")
    return states


def transition_replay_state(
    state: MapperReplayState,
    token_id: int,
    *,
    position: int,
    vocab: MapperV1Vocab,
    write_start_ms: int,
    write_end_ms: int,
) -> MapperReplayState:
    if token_id == vocab.pad_id:
        raise ReplayError("PAD is not legal in replay")

    if token_id == vocab.bos_id:
        if position != 0 or state.position != -1:
            raise ReplayError("BOS is legal only at token position 0")
        if state.current_ms != int(write_start_ms) or any(state.open_mask) or any(state.open_age_ms):
            raise ReplayError("BOS must start from the empty write-start state")
        return MapperReplayState(position=position, current_ms=state.current_ms, open_mask=state.open_mask, open_age_ms=state.open_age_ms)

    if state.position < 0:
        raise ReplayError("token sequence must start with BOS")

    if token_id == vocab.eos_id:
        if state.current_ms != int(write_end_ms):
            raise ReplayError(f"EOS requires current_ms == write_end_ms: {state.current_ms} != {write_end_ms}")
        if any(state.open_mask):
            raise ReplayError(f"EOS requires all lanes closed, got open_mask={state.open_mask_bits:04b}")
        return MapperReplayState(position=position, current_ms=state.current_ms, open_mask=state.open_mask, open_age_ms=state.open_age_ms)

    if vocab.is_time_shift_token(token_id):
        delta_ms = vocab.time_shift_value(token_id)
        next_ms = state.current_ms + delta_ms
        if next_ms > int(write_end_ms):
            raise ReplayError(f"TIME_SHIFT moves past write_end_ms: {next_ms} > {write_end_ms}")
        if any(state.open_mask) and next_ms == int(write_end_ms):
            raise ReplayError("TIME_SHIFT to write_end_ms while an LN is open creates a dead-end state")
        next_age = tuple(
            int(age + delta_ms) if is_open else 0
            for is_open, age in zip(state.open_mask, state.open_age_ms, strict=True)
        )
        return MapperReplayState(position=position, current_ms=next_ms, open_mask=state.open_mask, open_age_ms=next_age)

    if vocab.is_event_token(token_id):
        if state.current_ms >= int(write_end_ms):
            raise ReplayError("EVENT is illegal at or after write_end_ms")
        next_open = list(state.open_mask)
        next_age = list(state.open_age_ms)
        for lane, action in enumerate(vocab.decode_event(token_id)):
            is_open = state.open_mask[lane]
            if is_open and action not in {LaneAction.NONE, LaneAction.HOLD_END}:
                raise ReplayError(f"{action.value} is illegal on open lane {lane}")
            if not is_open and action == LaneAction.HOLD_END:
                raise ReplayError(f"HOLD_END is illegal on closed lane {lane}")

            if action == LaneAction.HOLD_START:
                next_open[lane] = True
                next_age[lane] = 0
            elif action == LaneAction.HOLD_END:
                next_open[lane] = False
                next_age[lane] = 0
            elif action == LaneAction.TAP:
                next_age[lane] = 0
        return MapperReplayState(
            position=position,
            current_ms=state.current_ms,
            open_mask=tuple(next_open),  # type: ignore[arg-type]
            open_age_ms=tuple(int(value) for value in next_age),  # type: ignore[arg-type]
        )

    raise ReplayError(f"unknown mapper v1 token id: {token_id}")


def replay_state_tensors(
    token_ids: Sequence[int],
    *,
    vocab: MapperV1Vocab,
    write_start_ms: int,
    write_end_ms: int,
) -> dict[str, torch.Tensor]:
    states = replay_tokens(
        token_ids,
        vocab=vocab,
        write_start_ms=write_start_ms,
        write_end_ms=write_end_ms,
        validate_final=True,
    )
    return {
        "current_ms": torch.tensor([state.current_ms for state in states], dtype=torch.long),
        "open_mask": torch.tensor([state.open_mask for state in states], dtype=torch.bool),
        "open_age_ms": torch.tensor([state.open_age_ms for state in states], dtype=torch.long),
    }


def close_labels_from_tokens(
    token_ids: Sequence[int],
    *,
    vocab: MapperV1Vocab,
    write_start_ms: int,
    write_end_ms: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    states = replay_tokens(
        token_ids,
        vocab=vocab,
        write_start_ms=write_start_ms,
        write_end_ms=write_end_ms,
        validate_final=True,
    )
    labels = torch.zeros((len(token_ids), KEY_COUNT), dtype=torch.bool)
    mask = torch.zeros((len(token_ids), KEY_COUNT), dtype=torch.bool)
    for index, state in enumerate(states[:-1]):
        for lane, is_open in enumerate(state.open_mask):
            mask[index, lane] = is_open
        next_token = int(token_ids[index + 1])
        if not vocab.is_event_token(next_token):
            continue
        for lane, action in enumerate(vocab.decode_event(next_token)):
            labels[index, lane] = state.open_mask[lane] and action == LaneAction.HOLD_END
    return labels, mask


def open_mask_bits_to_tuple(mask_bits: int) -> tuple[bool, bool, bool, bool]:
    if not 0 <= int(mask_bits) < 2**KEY_COUNT:
        raise ValueError(f"open mask outside 4K range: {mask_bits}")
    return tuple(bool(int(mask_bits) & (1 << lane)) for lane in range(KEY_COUNT))  # type: ignore[return-value]


def open_mask_tuple_to_bits(open_mask: Sequence[bool]) -> int:
    if len(open_mask) != KEY_COUNT:
        raise ValueError(f"open mask must contain {KEY_COUNT} lanes: {open_mask}")
    bits = 0
    for lane, is_open in enumerate(open_mask):
        if bool(is_open):
            bits |= 1 << lane
    return bits
