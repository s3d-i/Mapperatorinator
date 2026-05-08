from __future__ import annotations

from collections.abc import Sequence

import torch

from .vocab import KEY_COUNT, LaneAction, MapperV1Vocab


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
    return torch.zeros_like(valid, dtype=torch.float32).masked_fill(~valid, invalid_value)


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
