from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..events.canonical import LaneAction
from ..events.grammar import ConstrainedDecodeState
from ..events.tokens import Stage1Vocab, decode_target_tokens
from .scheduler import DecodedTokenWindow


NoteKind = Literal["tap", "hold"]


@dataclass(frozen=True)
class PreviewNote:
    time_ms: int
    lane: int
    kind: NoteKind
    end_time_ms: int | None = None

    def to_json(self) -> dict[str, int | str]:
        payload: dict[str, int | str] = {
            "time_ms": self.time_ms,
            "lane": self.lane,
            "kind": self.kind,
        }
        if self.kind == "hold":
            if self.end_time_ms is None:
                raise ValueError(f"hold note missing end_time_ms: {self}")
            payload["end_time_ms"] = self.end_time_ms
        return payload


@dataclass(frozen=True)
class NoteBatch:
    start_ms: int
    end_ms: int
    generated_through_ms: int
    notes: list[PreviewNote]

    def to_json(self) -> dict[str, object]:
        return {
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "generated_through_ms": self.generated_through_ms,
            "notes": [note.to_json() for note in sorted(self.notes, key=lambda item: (item.time_ms, item.lane))],
        }


class StreamingAssembler:
    def __init__(self, *, vocab: Stage1Vocab | None = None, batch_ms: int = 1000) -> None:
        if batch_ms <= 0:
            raise ValueError(f"batch_ms must be positive: {batch_ms}")
        self.vocab = Stage1Vocab() if vocab is None else vocab
        self.batch_ms = batch_ms
        self.generated_through_ms = 0
        self.committed_through_ms = 0
        self._started = False
        self._open_hold_start_ms_by_lane: dict[int, int] = {}
        self._notes_by_batch_start: dict[int, list[PreviewNote]] = {}

    @property
    def open_hold_mask(self) -> int:
        mask = 0
        for lane in self._open_hold_start_ms_by_lane:
            mask |= 1 << lane
        return mask

    def process_window(self, window: DecodedTokenWindow) -> list[NoteBatch]:
        self._validate_complete_window(window)
        if not self._started:
            self._started = True
            self.generated_through_ms = window.write_start_ms
            self.committed_through_ms = window.write_start_ms
        if window.write_start_ms != self.generated_through_ms:
            raise ValueError(
                f"decoded windows must be processed contiguously: expected {self.generated_through_ms}, "
                f"got {window.write_start_ms}",
            )
        if window.write_end_ms <= window.write_start_ms:
            raise ValueError(f"decoded window end must be after start: {window}")

        write_duration_ms = window.write_end_ms - window.write_start_ms
        self._validate_target_token_grammar(
            window.token_ids,
            write_duration_ms=write_duration_ms,
            open_hold_mask=self.open_hold_mask,
        )
        relative_timepoints = decode_target_tokens(
            window.token_ids,
            vocab=self.vocab,
            write_duration_ms=write_duration_ms,
        )
        for timepoint in relative_timepoints:
            absolute_time_ms = window.write_start_ms + timepoint.time_ms
            for lane, action in enumerate(timepoint.lane_actions):
                self._apply_lane_action(absolute_time_ms=absolute_time_ms, lane=lane, action=action)

        self.generated_through_ms = window.write_end_ms
        return self._commit_ready_batches()

    def _validate_complete_window(self, window: DecodedTokenWindow) -> None:
        reasons: list[str] = []
        if not window.token_ids or window.token_ids[-1] != self.vocab.eos_id:
            reasons.append("EOS token missing from window end")
        if reasons:
            raise ValueError(
                f"incomplete decoded window {window.write_start_ms}-{window.write_end_ms}ms: {', '.join(reasons)}",
            )

    def _validate_target_token_grammar(
        self,
        token_ids: list[int],
        *,
        write_duration_ms: int,
        open_hold_mask: int,
    ) -> None:
        state = ConstrainedDecodeState.after_prefix(
            open_hold_mask=open_hold_mask,
            write_duration_ms=write_duration_ms,
        )
        for position, token_id in enumerate(token_ids):
            if not state.is_legal(token_id, self.vocab):
                raise ValueError(
                    f"illegal target token at position {position}: {_token_name(self.vocab, token_id)}",
                )
            state = state.transition(token_id, self.vocab)

    def finish(self) -> list[NoteBatch]:
        if self._open_hold_start_ms_by_lane:
            raise ValueError(f"generation ended with unclosed holds: {sorted(self._open_hold_start_ms_by_lane)}")
        return self._commit_ready_batches(force_partial=True)

    def _apply_lane_action(self, *, absolute_time_ms: int, lane: int, action: LaneAction) -> None:
        if action == LaneAction.NONE:
            return
        if action == LaneAction.TAP:
            if lane in self._open_hold_start_ms_by_lane:
                raise ValueError(f"TAP while hold is open at {absolute_time_ms}ms lane {lane}")
            self._queue_note(PreviewNote(time_ms=absolute_time_ms, lane=lane, kind="tap"))
            return
        if action == LaneAction.HOLD_START:
            if lane in self._open_hold_start_ms_by_lane:
                raise ValueError(f"HOLD_START while hold is open at {absolute_time_ms}ms lane {lane}")
            self._open_hold_start_ms_by_lane[lane] = absolute_time_ms
            return
        if action == LaneAction.HOLD_END:
            start_ms = self._open_hold_start_ms_by_lane.pop(lane, None)
            if start_ms is None:
                raise ValueError(f"HOLD_END without open hold at {absolute_time_ms}ms lane {lane}")
            if absolute_time_ms <= start_ms:
                raise ValueError(f"hold end must be after start at {absolute_time_ms}ms lane {lane}")
            self._queue_note(PreviewNote(time_ms=start_ms, lane=lane, kind="hold", end_time_ms=absolute_time_ms))
            return
        raise ValueError(f"unsupported decoded lane action for preview: {action}")

    def _queue_note(self, note: PreviewNote) -> None:
        batch_start = _batch_start(note.time_ms, self.batch_ms)
        self._notes_by_batch_start.setdefault(batch_start, []).append(note)

    def _commit_ready_batches(self, *, force_partial: bool = False) -> list[NoteBatch]:
        batches: list[NoteBatch] = []
        safe_through_ms = self._safe_commit_through(force_partial=force_partial)
        while self.committed_through_ms < safe_through_ms:
            batch_start = self.committed_through_ms
            batch_end = min(batch_start + self.batch_ms, safe_through_ms) if force_partial else batch_start + self.batch_ms
            if batch_end > safe_through_ms:
                break
            notes = self._notes_by_batch_start.pop(batch_start, [])
            batches.append(
                NoteBatch(
                    start_ms=batch_start,
                    end_ms=batch_end,
                    generated_through_ms=self.generated_through_ms,
                    notes=notes,
                ),
            )
            self.committed_through_ms = batch_end
        return batches

    def _safe_commit_through(self, *, force_partial: bool) -> int:
        safe_through_ms = self.generated_through_ms
        if self._open_hold_start_ms_by_lane:
            earliest_open_start = min(self._open_hold_start_ms_by_lane.values())
            safe_through_ms = min(safe_through_ms, _batch_start(earliest_open_start, self.batch_ms))
        if force_partial:
            return safe_through_ms
        return _batch_start(safe_through_ms, self.batch_ms)


def _batch_start(time_ms: int, batch_ms: int) -> int:
    return (time_ms // batch_ms) * batch_ms


def _token_name(vocab: Stage1Vocab, token_id: int) -> str:
    if 0 <= token_id < vocab.size:
        return vocab.token_name(token_id)
    return f"<out-of-range:{token_id}>"
