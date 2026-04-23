from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Sequence

import torch

from .canonical import LaneAction
from .tokens import Stage1Vocab


class DecodePhase(str, Enum):
    EXPECT_TS_OR_EOS = "EXPECT_TS_OR_EOS"
    EXPECT_EVENT_OR_TS_REMAINDER = "EXPECT_EVENT_OR_TS_REMAINDER"
    EXPECT_EVENT_ONLY = "EXPECT_EVENT_ONLY"
    FINISHED = "FINISHED"


def _event_actions_compatible_with_open_hold_mask(
    open_hold_mask: int,
    lane_actions: Sequence[LaneAction],
) -> bool:
    for lane, action in enumerate(lane_actions):
        is_open = (open_hold_mask & (1 << lane)) != 0
        if is_open and action not in {LaneAction.NONE, LaneAction.HOLD_END}:
            return False
        if not is_open and action == LaneAction.HOLD_END:
            return False
    return True


@dataclass(frozen=True)
class ConstrainedDecodeState:
    current_time_rel: int
    pending_delta: int
    has_emitted_event: bool
    open_hold_mask: int
    write_duration_ms: int
    phase: DecodePhase

    @classmethod
    def after_prefix(cls, *, open_hold_mask: int, write_duration_ms: int) -> "ConstrainedDecodeState":
        return cls(
            current_time_rel=0,
            pending_delta=0,
            has_emitted_event=False,
            open_hold_mask=open_hold_mask,
            write_duration_ms=write_duration_ms,
            phase=DecodePhase.EXPECT_TS_OR_EOS,
        )

    def is_legal(self, token_id: int, vocab: Stage1Vocab) -> bool:
        if self.phase == DecodePhase.FINISHED:
            return False
        if token_id == vocab.eos_id:
            return self.phase == DecodePhase.EXPECT_TS_OR_EOS
        if vocab.is_ts_token(token_id):
            return self._is_legal_ts(vocab.ts_value(token_id))
        if vocab.is_event_token(token_id):
            return self.phase in {
                DecodePhase.EXPECT_EVENT_ONLY,
                DecodePhase.EXPECT_EVENT_OR_TS_REMAINDER,
            } and self._event_time_in_bounds() and self._is_legal_event(vocab.decode_event_token(token_id))
        return False

    def legal_token_mask(
        self,
        vocab: Stage1Vocab,
        *,
        device: torch.device | None = None,
        mask_cache: "_LegalityMaskCache | None" = None,
    ) -> torch.Tensor:
        if mask_cache is None:
            resolved_device = torch.device("cpu") if device is None else device
            mask_cache = _LegalityMaskCache(vocab=vocab, device=resolved_device)
        return mask_cache.mask_for_state(self)

    def transition(self, token_id: int, vocab: Stage1Vocab) -> "ConstrainedDecodeState":
        if not self.is_legal(token_id, vocab):
            raise ValueError(f"illegal token {vocab.token_name(token_id)} for decode state {self}")

        if token_id == vocab.eos_id:
            return replace(self, phase=DecodePhase.FINISHED)

        if vocab.is_ts_token(token_id):
            value = vocab.ts_value(token_id)
            phase = DecodePhase.EXPECT_EVENT_OR_TS_REMAINDER if value == 1000 else DecodePhase.EXPECT_EVENT_ONLY
            return replace(self, pending_delta=self.pending_delta + value, phase=phase)

        event_time = self._next_event_time_rel()
        open_hold_mask = self.open_hold_mask
        for lane, action in enumerate(vocab.decode_event_token(token_id)):
            lane_bit = 1 << lane
            if action == LaneAction.HOLD_START:
                open_hold_mask |= lane_bit
            elif action == LaneAction.HOLD_END:
                open_hold_mask &= ~lane_bit

        return ConstrainedDecodeState(
            current_time_rel=event_time,
            pending_delta=0,
            has_emitted_event=True,
            open_hold_mask=open_hold_mask,
            write_duration_ms=self.write_duration_ms,
            phase=DecodePhase.EXPECT_TS_OR_EOS,
        )

    def _is_legal_ts(self, value: int) -> bool:
        if self.phase not in {
            DecodePhase.EXPECT_TS_OR_EOS,
            DecodePhase.EXPECT_EVENT_OR_TS_REMAINDER,
        }:
            return False
        if self.phase == DecodePhase.EXPECT_TS_OR_EOS:
            if self.has_emitted_event and value == 0:
                return False
        elif value == 0:
            return False

        return self._next_event_time_rel(extra_delta=value) < self.write_duration_ms

    def _event_time_in_bounds(self) -> bool:
        return self._next_event_time_rel() < self.write_duration_ms

    def _next_event_time_rel(self, *, extra_delta: int = 0) -> int:
        base = self.current_time_rel if self.has_emitted_event else 0
        return base + self.pending_delta + extra_delta

    def _is_legal_event(self, lane_actions: Sequence[LaneAction]) -> bool:
        return _event_actions_compatible_with_open_hold_mask(self.open_hold_mask, lane_actions)


@dataclass
class _LegalityMaskCache:
    vocab: Stage1Vocab
    device: torch.device

    def __post_init__(self) -> None:
        self._empty_mask = torch.zeros(self.vocab.size, dtype=torch.bool, device=self.device)
        self._max_ts_index = len(self.vocab.ts_token_ids) - 1
        self._event_masks_by_open_hold = {
            open_hold_mask: self._build_event_mask(open_hold_mask)
            for open_hold_mask in range(len(self.vocab.open_token_ids))
        }
        self._ts_masks_by_key: dict[tuple[DecodePhase, bool, int], torch.Tensor] = {}

    def mask_for_state(self, state: ConstrainedDecodeState) -> torch.Tensor:
        mask = self._ts_mask_for_state(state).clone()
        if state.phase == DecodePhase.EXPECT_TS_OR_EOS:
            mask[self.vocab.eos_id] = True
        if state.phase in {
            DecodePhase.EXPECT_EVENT_ONLY,
            DecodePhase.EXPECT_EVENT_OR_TS_REMAINDER,
        } and state._event_time_in_bounds():
            mask |= self._event_masks_by_open_hold[state.open_hold_mask]
        return mask

    def _build_event_mask(self, open_hold_mask: int) -> torch.Tensor:
        mask = self._empty_mask.clone()
        for token_id in self.vocab.event_token_ids:
            if _event_actions_compatible_with_open_hold_mask(
                open_hold_mask,
                self.vocab.decode_event_token(token_id),
            ):
                mask[token_id] = True
        return mask

    def _ts_mask_for_state(self, state: ConstrainedDecodeState) -> torch.Tensor:
        if state.phase not in {
            DecodePhase.EXPECT_TS_OR_EOS,
            DecodePhase.EXPECT_EVENT_OR_TS_REMAINDER,
        }:
            return self._empty_mask

        min_ts_index = 1 if state.phase == DecodePhase.EXPECT_EVENT_OR_TS_REMAINDER or state.has_emitted_event else 0
        remaining_time_ms = state.write_duration_ms - state._next_event_time_rel()
        if remaining_time_ms <= 0:
            return self._empty_mask

        max_ts_index = min(self._max_ts_index, (remaining_time_ms - 1) // 10)
        if max_ts_index < min_ts_index:
            return self._empty_mask

        key = (state.phase, state.has_emitted_event, max_ts_index)
        cached_mask = self._ts_masks_by_key.get(key)
        if cached_mask is not None:
            return cached_mask

        mask = self._empty_mask.clone()
        start_token_id = self.vocab.ts_token_ids[min_ts_index]
        stop_token_id = self.vocab.ts_token_ids[max_ts_index]
        mask[start_token_id : stop_token_id + 1] = True
        self._ts_masks_by_key[key] = mask
        return mask


@dataclass(frozen=True)
class ConstrainedDecodeResult:
    token_ids: list[int]
    max_decode_len_reached: bool
    eos_emitted_by_model: bool
    eos_forced_after_pending_ts: bool


def force_eos_after_pending_ts(token_ids: Sequence[int], vocab: Stage1Vocab) -> list[int]:
    trimmed = list(token_ids)
    while trimmed and vocab.is_ts_token(trimmed[-1]):
        trimmed.pop()
    if trimmed and trimmed[-1] == vocab.eos_id:
        return trimmed
    trimmed.append(vocab.eos_id)
    return trimmed


def constrained_greedy_decode(
    model,
    *,
    packed_audio: torch.Tensor,
    timing_track: torch.Tensor,
    difficulty_bucket: torch.Tensor,
    condition_ids: Sequence[int],
    open_hold_mask: int,
    write_duration_ms: int,
    vocab: Stage1Vocab,
    max_decode_len: int = 512,
) -> ConstrainedDecodeResult:
    if max_decode_len <= 0:
        raise ValueError(f"max_decode_len must be positive: {max_decode_len}")

    device = packed_audio.device
    split_decode = callable(getattr(model, "encode_context", None)) and callable(
        getattr(model, "decode_from_memory", None),
    )
    memory = None
    if split_decode:
        memory = model.encode_context(
            packed_audio=packed_audio,
            timing_track=timing_track,
            difficulty_bucket=difficulty_bucket,
        )
    generated = list(condition_ids)
    decoder_input_ids = torch.empty(
        (1, len(condition_ids) + max_decode_len),
        dtype=torch.long,
        device=device,
    )
    if condition_ids:
        decoder_input_ids[0, : len(condition_ids)] = torch.tensor(condition_ids, dtype=torch.long, device=device)
    generated_length = len(condition_ids)
    state = ConstrainedDecodeState.after_prefix(
        open_hold_mask=open_hold_mask,
        write_duration_ms=write_duration_ms,
    )
    legality_mask_cache = _LegalityMaskCache(vocab=vocab, device=device)
    target_count = 0

    while target_count < max_decode_len:
        remaining_target_slots = max_decode_len - target_count
        if remaining_target_slots == 1:
            if state.phase == DecodePhase.EXPECT_TS_OR_EOS:
                generated.append(vocab.eos_id)
                return ConstrainedDecodeResult(
                    token_ids=generated,
                    max_decode_len_reached=True,
                    eos_emitted_by_model=False,
                    eos_forced_after_pending_ts=False,
                )
            return ConstrainedDecodeResult(
                token_ids=force_eos_after_pending_ts(generated, vocab),
                max_decode_len_reached=True,
                eos_emitted_by_model=False,
                eos_forced_after_pending_ts=True,
            )

        active_decoder_input_ids = decoder_input_ids[:, :generated_length]
        if memory is None:
            logits = model(
                packed_audio=packed_audio,
                timing_track=timing_track,
                difficulty_bucket=difficulty_bucket,
                decoder_input_ids=active_decoder_input_ids,
            )[0, -1]
        else:
            logits = model.decode_from_memory(
                memory=memory,
                decoder_input_ids=active_decoder_input_ids,
            )[0, -1]
        legal_mask = state.legal_token_mask(vocab, device=device, mask_cache=legality_mask_cache)
        masked_logits = logits.masked_fill(~legal_mask, -torch.inf)
        next_token_id = int(torch.argmax(masked_logits).item())
        generated.append(next_token_id)
        decoder_input_ids[0, generated_length] = next_token_id
        generated_length += 1
        target_count += 1
        state = state.transition(next_token_id, vocab)
        if next_token_id == vocab.eos_id:
            return ConstrainedDecodeResult(
                token_ids=generated,
                max_decode_len_reached=False,
                eos_emitted_by_model=True,
                eos_forced_after_pending_ts=False,
            )

    if state.phase == DecodePhase.EXPECT_TS_OR_EOS:
        generated.append(vocab.eos_id)
        return ConstrainedDecodeResult(
            token_ids=generated,
            max_decode_len_reached=True,
            eos_emitted_by_model=False,
            eos_forced_after_pending_ts=False,
        )
    return ConstrainedDecodeResult(
        token_ids=force_eos_after_pending_ts(generated, vocab),
        max_decode_len_reached=True,
        eos_emitted_by_model=False,
        eos_forced_after_pending_ts=True,
    )
