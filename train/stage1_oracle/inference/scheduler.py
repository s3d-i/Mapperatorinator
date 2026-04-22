from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DecodedTokenWindow:
    write_start_ms: int
    write_end_ms: int
    token_ids: list[int]
    max_decode_len_reached: bool = False
    eos_emitted_by_model: bool = True
    eos_forced_after_pending_ts: bool = False

    @property
    def generated_through_ms(self) -> int:
        return self.write_end_ms


@dataclass(frozen=True)
class BufferStatus:
    state: str
    future_ms: int
    ready: bool
    buffering: bool


class BufferPolicy:
    def __init__(self, *, target_buffer_ms: int = 2000, rebuffer_floor_ms: int = 750) -> None:
        if target_buffer_ms <= 0:
            raise ValueError(f"target_buffer_ms must be positive: {target_buffer_ms}")
        if rebuffer_floor_ms < 0:
            raise ValueError(f"rebuffer_floor_ms must be non-negative: {rebuffer_floor_ms}")
        if rebuffer_floor_ms >= target_buffer_ms:
            raise ValueError("rebuffer_floor_ms must be smaller than target_buffer_ms")
        self.target_buffer_ms = int(target_buffer_ms)
        self.rebuffer_floor_ms = int(rebuffer_floor_ms)
        self._started = False
        self._buffering = True

    def update(self, *, playhead_ms: int, committed_through_ms: int, done: bool = False) -> BufferStatus:
        future_ms = max(0, int(committed_through_ms) - int(playhead_ms))
        if done:
            self._started = True
            self._buffering = False
            return BufferStatus(state="done", future_ms=future_ms, ready=True, buffering=False)
        if not self._started:
            if future_ms >= self.target_buffer_ms:
                self._started = True
                self._buffering = False
                return BufferStatus(state="ready", future_ms=future_ms, ready=True, buffering=False)
            return BufferStatus(state="startup", future_ms=future_ms, ready=False, buffering=True)
        if future_ms < self.rebuffer_floor_ms:
            self._buffering = True
            return BufferStatus(state="buffering", future_ms=future_ms, ready=True, buffering=True)
        if self._buffering and future_ms < self.target_buffer_ms:
            return BufferStatus(state="buffering", future_ms=future_ms, ready=True, buffering=True)
        self._buffering = False
        return BufferStatus(state="ready", future_ms=future_ms, ready=True, buffering=False)
