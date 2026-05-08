from __future__ import annotations

from collections.abc import Sequence

import torch

from .tokenizer import MAPPER_WRITE_MS
from .vocab import KEY_COUNT, LaneAction, MapperV1Vocab

_TOKEN_GRID_MS = 10
_OPEN_MASK_COUNT = 2**KEY_COUNT
_GRAMMAR_TABLE_CACHE: dict[tuple[tuple[str, ...], int | None, int], torch.Tensor] = {}
_GRAMMAR_DEVICE_TABLE_CACHE: dict[tuple[tuple[str, ...], int | None, int, str], torch.Tensor] = {}


def valid_token_mask(
    *,
    position: int,
    current_ms: int,
    open_mask: int | Sequence[bool] | torch.Tensor,
    write_start_ms: int,
    write_end_ms: int,
    vocab: MapperV1Vocab,
    min_ln_duration_ms: int | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    del write_start_ms
    resolved_device = torch.device("cpu") if device is None else device
    mask = torch.zeros(vocab.size, dtype=torch.bool, device=resolved_device)
    if int(position) < 0:
        mask[vocab.bos_id] = True
        return mask

    current_ms = int(current_ms)
    write_end_ms = int(write_end_ms)
    open_tuple = _normalize_open_mask(open_mask)
    any_open = any(open_tuple)

    if int(position) > 0 and current_ms == write_end_ms and not any_open:
        mask[vocab.eos_id] = True

    for token_id in vocab.time_shift_token_ids:
        delta_ms = vocab.time_shift_value(token_id)
        next_ms = current_ms + delta_ms
        if next_ms > write_end_ms:
            continue
        if any_open and next_ms >= write_end_ms:
            continue
        mask[token_id] = True

    if current_ms < write_end_ms:
        remaining_ms = write_end_ms - current_ms
        for token_id in vocab.event_token_ids:
            if _event_is_legal(
                vocab.decode_event(token_id),
                open_tuple,
                remaining_ms=remaining_ms,
                min_ln_duration_ms=min_ln_duration_ms,
            ):
                mask[token_id] = True
    return mask


def build_grammar_mask(
    *,
    current_ms: torch.Tensor,
    open_mask: torch.Tensor,
    write_start_ms: torch.Tensor | int,
    write_end_ms: torch.Tensor | int,
    vocab: MapperV1Vocab,
    positions: torch.Tensor | None = None,
    min_ln_duration_ms: int | None = None,
    invalid_value: float = -torch.inf,
) -> torch.Tensor:
    if current_ms.ndim == 1:
        current_ms = current_ms.unsqueeze(0)
    if open_mask.ndim == 2:
        open_mask = open_mask.unsqueeze(0)
    if current_ms.ndim != 2:
        raise ValueError(f"current_ms must have shape [B,T] or [T], got {tuple(current_ms.shape)}")
    if open_mask.shape[:2] != current_ms.shape or int(open_mask.shape[-1]) != KEY_COUNT:
        raise ValueError(
            f"open_mask must have shape {tuple(current_ms.shape)}x{KEY_COUNT}, got {tuple(open_mask.shape)}",
        )
    batch_size, steps = current_ms.shape
    device = current_ms.device
    if positions is None:
        positions = torch.arange(steps, dtype=torch.long, device=device).expand(batch_size, steps)
    elif positions.ndim == 1:
        positions = positions.unsqueeze(0).expand(batch_size, steps)
    if tuple(positions.shape) != (batch_size, steps):
        raise ValueError(f"positions must have shape {(batch_size, steps)}, got {tuple(positions.shape)}")

    write_start_values = _broadcast_window_tensor(write_start_ms, batch_size=batch_size, device=device)
    write_end_values = _broadcast_window_tensor(write_end_ms, batch_size=batch_size, device=device)
    remaining_ms = write_end_values.reshape(batch_size, 1) - current_ms.to(dtype=torch.long)
    if (
        bool((remaining_ms < 0).any())
        or bool((remaining_ms > MAPPER_WRITE_MS).any())
        or bool((remaining_ms % _TOKEN_GRID_MS != 0).any())
    ):
        valid = _build_grammar_valid_slow(
            current_ms=current_ms,
            open_mask=open_mask,
            write_start_values=write_start_values,
            write_end_values=write_end_values,
            vocab=vocab,
            positions=positions,
            min_ln_duration_ms=min_ln_duration_ms,
            device=device,
        )
        return torch.zeros_like(valid, dtype=torch.float32).masked_fill(~valid, invalid_value)

    table = _grammar_lookup_table(vocab=vocab, min_ln_duration_ms=min_ln_duration_ms, device=device)
    remaining_index = (remaining_ms // _TOKEN_GRID_MS).to(dtype=torch.long)
    lane_bits = torch.tensor([1, 2, 4, 8], dtype=torch.long, device=device).reshape(1, 1, KEY_COUNT)
    open_bits = (open_mask.to(dtype=torch.long) * lane_bits).sum(dim=-1)
    position_positive = (positions > 0).to(dtype=torch.long)
    valid = table[remaining_index, open_bits, position_positive]
    negative_position = positions < 0
    if bool(negative_position.any()):
        valid = valid.clone()
        valid[negative_position] = False
        valid[:, :, vocab.bos_id] = valid[:, :, vocab.bos_id] | negative_position
    return torch.zeros_like(valid, dtype=torch.float32).masked_fill(~valid, invalid_value)


def _build_grammar_valid_slow(
    *,
    current_ms: torch.Tensor,
    open_mask: torch.Tensor,
    write_start_values: torch.Tensor,
    write_end_values: torch.Tensor,
    vocab: MapperV1Vocab,
    positions: torch.Tensor,
    min_ln_duration_ms: int | None,
    device: torch.device,
) -> torch.Tensor:
    batch_size, steps = current_ms.shape
    valid = torch.zeros((batch_size, steps, vocab.size), dtype=torch.bool, device=device)
    for batch_index in range(batch_size):
        for step in range(steps):
            valid[batch_index, step] = valid_token_mask(
                position=int(positions[batch_index, step].item()),
                current_ms=int(current_ms[batch_index, step].item()),
                open_mask=open_mask[batch_index, step],
                write_start_ms=int(write_start_values[batch_index].item()),
                write_end_ms=int(write_end_values[batch_index].item()),
                vocab=vocab,
                min_ln_duration_ms=min_ln_duration_ms,
                device=device,
            )
    return valid


def _grammar_lookup_table(
    *,
    vocab: MapperV1Vocab,
    min_ln_duration_ms: int | None,
    device: torch.device,
) -> torch.Tensor:
    max_remaining_units = MAPPER_WRITE_MS // _TOKEN_GRID_MS
    cache_key = (tuple(vocab.id_to_token), min_ln_duration_ms, max_remaining_units)
    table = _GRAMMAR_TABLE_CACHE.get(cache_key)
    if table is None:
        table = _build_grammar_lookup_table(
            vocab=vocab,
            min_ln_duration_ms=min_ln_duration_ms,
            max_remaining_units=max_remaining_units,
        )
        _GRAMMAR_TABLE_CACHE[cache_key] = table
    if device.type == "cpu":
        return table

    device_key = (*cache_key, str(device))
    device_table = _GRAMMAR_DEVICE_TABLE_CACHE.get(device_key)
    if device_table is None or device_table.device != device:
        device_table = table.to(device=device)
        _GRAMMAR_DEVICE_TABLE_CACHE[device_key] = device_table
    return device_table


def _build_grammar_lookup_table(
    *,
    vocab: MapperV1Vocab,
    min_ln_duration_ms: int | None,
    max_remaining_units: int,
) -> torch.Tensor:
    table = torch.zeros((max_remaining_units + 1, _OPEN_MASK_COUNT, 2, vocab.size), dtype=torch.bool)
    time_shift_values = tuple(vocab.time_shift_value(token_id) for token_id in vocab.time_shift_token_ids)
    event_actions = tuple(vocab.decode_event(token_id) for token_id in vocab.event_token_ids)
    for remaining_units in range(max_remaining_units + 1):
        remaining_ms = remaining_units * _TOKEN_GRID_MS
        for open_bits in range(_OPEN_MASK_COUNT):
            open_tuple = tuple(bool(open_bits & (1 << lane)) for lane in range(KEY_COUNT))
            any_open = any(open_tuple)

            if remaining_ms == 0 and not any_open:
                table[remaining_units, open_bits, 1, vocab.eos_id] = True

            for token_id, delta_ms in zip(vocab.time_shift_token_ids, time_shift_values, strict=True):
                if delta_ms <= remaining_ms and (not any_open or delta_ms < remaining_ms):
                    table[remaining_units, open_bits, :, token_id] = True

            if remaining_ms > 0:
                for token_id, lane_actions in zip(vocab.event_token_ids, event_actions, strict=True):
                    if _event_is_legal(
                        lane_actions,
                        open_tuple,
                        remaining_ms=remaining_ms,
                        min_ln_duration_ms=min_ln_duration_ms,
                    ):
                        table[remaining_units, open_bits, :, token_id] = True
    return table


def _event_is_legal(
    lane_actions: Sequence[LaneAction],
    open_mask: Sequence[bool],
    *,
    remaining_ms: int,
    min_ln_duration_ms: int | None,
) -> bool:
    for lane, action in enumerate(lane_actions):
        is_open = bool(open_mask[lane])
        if is_open and action not in {LaneAction.NONE, LaneAction.HOLD_END}:
            return False
        if not is_open and action == LaneAction.HOLD_END:
            return False
        if (
            min_ln_duration_ms is not None
            and not is_open
            and action == LaneAction.HOLD_START
            and remaining_ms < int(min_ln_duration_ms)
        ):
            return False
    return True


def _normalize_open_mask(open_mask: int | Sequence[bool] | torch.Tensor) -> tuple[bool, bool, bool, bool]:
    if isinstance(open_mask, int):
        if not 0 <= open_mask < 2**KEY_COUNT:
            raise ValueError(f"open_mask outside 4K range: {open_mask}")
        return tuple(bool(open_mask & (1 << lane)) for lane in range(KEY_COUNT))  # type: ignore[return-value]
    if isinstance(open_mask, torch.Tensor):
        values = open_mask.detach().cpu().reshape(-1).tolist()
    else:
        values = list(open_mask)
    if len(values) != KEY_COUNT:
        raise ValueError(f"open_mask must contain {KEY_COUNT} lanes: {open_mask}")
    return tuple(bool(value) for value in values)  # type: ignore[return-value]


def _broadcast_window_tensor(value: torch.Tensor | int, *, batch_size: int, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.to(device=device, dtype=torch.long).reshape(-1)
        if tensor.numel() == 1:
            return tensor.expand(batch_size)
        if tensor.numel() == batch_size:
            return tensor
        raise ValueError(f"window tensor must have 1 or {batch_size} values, got {tensor.numel()}")
    return torch.full((batch_size,), int(value), dtype=torch.long, device=device)
