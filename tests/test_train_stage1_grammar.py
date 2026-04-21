import unittest

import torch

from train.stage1_oracle.events.canonical import LaneAction
from train.stage1_oracle.events.grammar import (
    ConstrainedDecodeState,
    constrained_greedy_decode,
    force_eos_after_pending_ts,
)
from train.stage1_oracle.events.tokens import Stage1Vocab


class _TokenSequenceModel:
    def __init__(self, token_ids: list[int], *, vocab_size: int) -> None:
        self.token_ids = token_ids
        self.vocab_size = vocab_size

    def __call__(
        self,
        *,
        packed_audio: torch.Tensor,
        timing_track: torch.Tensor,
        difficulty_bucket: torch.Tensor,
        decoder_input_ids: torch.Tensor,
    ) -> torch.Tensor:
        step = decoder_input_ids.shape[1] - 3
        next_token_id = self.token_ids[min(step, len(self.token_ids) - 1)]
        logits = torch.full(
            (decoder_input_ids.shape[0], decoder_input_ids.shape[1], self.vocab_size),
            -1000.0,
            dtype=torch.float32,
        )
        logits[:, -1, next_token_id] = 1000.0
        return logits


class Stage1GrammarTests(unittest.TestCase):
    def test_eos_is_legal_only_after_prefix_or_completed_event(self) -> None:
        vocab = Stage1Vocab()
        state = ConstrainedDecodeState.after_prefix(open_hold_mask=0, write_duration_ms=8000)

        self.assertTrue(state.is_legal(vocab.eos_id, vocab))
        state = state.transition(vocab.ts_token_id(100), vocab)
        self.assertFalse(state.is_legal(vocab.eos_id, vocab))
        event_id = vocab.encode_timepoint_event((LaneAction.TAP, LaneAction.NONE, LaneAction.NONE, LaneAction.NONE))
        state = state.transition(event_id, vocab)
        self.assertTrue(state.is_legal(vocab.eos_id, vocab))

    def test_ts_zero_is_legal_only_for_first_event(self) -> None:
        vocab = Stage1Vocab()
        event_id = vocab.encode_timepoint_event((LaneAction.TAP, LaneAction.NONE, LaneAction.NONE, LaneAction.NONE))
        state = ConstrainedDecodeState.after_prefix(open_hold_mask=0, write_duration_ms=8000)

        self.assertTrue(state.is_legal(vocab.ts_token_id(0), vocab))
        state = state.transition(vocab.ts_token_id(0), vocab).transition(event_id, vocab)

        self.assertFalse(state.is_legal(vocab.ts_token_id(0), vocab))

    def test_ts_mask_enforces_canonical_decomposition_and_time_bound(self) -> None:
        vocab = Stage1Vocab()
        state = ConstrainedDecodeState.after_prefix(open_hold_mask=0, write_duration_ms=1500)

        state = state.transition(vocab.ts_token_id(500), vocab)
        self.assertFalse(state.is_legal(vocab.ts_token_id(500), vocab))

        state = ConstrainedDecodeState.after_prefix(open_hold_mask=0, write_duration_ms=1500)
        state = state.transition(vocab.ts_token_id(1000), vocab)
        self.assertTrue(state.is_legal(vocab.ts_token_id(490), vocab))
        self.assertFalse(state.is_legal(vocab.ts_token_id(500), vocab))

    def test_event_legality_tracks_open_hold_state_per_lane(self) -> None:
        vocab = Stage1Vocab()
        state = ConstrainedDecodeState.after_prefix(open_hold_mask=0b0001, write_duration_ms=8000)
        tap_on_open_lane = vocab.encode_timepoint_event(
            (LaneAction.TAP, LaneAction.NONE, LaneAction.NONE, LaneAction.NONE),
        )
        end_open_lane = vocab.encode_timepoint_event(
            (LaneAction.HOLD_END, LaneAction.NONE, LaneAction.NONE, LaneAction.NONE),
        )

        state = state.transition(vocab.ts_token_id(100), vocab)

        self.assertFalse(state.is_legal(tap_on_open_lane, vocab))
        self.assertTrue(state.is_legal(end_open_lane, vocab))
        self.assertEqual(state.transition(end_open_lane, vocab).open_hold_mask, 0)

    def test_force_eos_rolls_back_only_pending_ts_suffix(self) -> None:
        vocab = Stage1Vocab()
        event_id = vocab.encode_timepoint_event((LaneAction.TAP, LaneAction.NONE, LaneAction.NONE, LaneAction.NONE))
        tokens = [
            vocab.bos_id,
            vocab.diff_token_id(0),
            vocab.open_token_id(0),
            vocab.ts_token_id(0),
            event_id,
            vocab.ts_token_id(100),
            vocab.ts_token_id(1000),
        ]

        self.assertEqual(
            force_eos_after_pending_ts(tokens, vocab),
            [
                vocab.bos_id,
                vocab.diff_token_id(0),
                vocab.open_token_id(0),
                vocab.ts_token_id(0),
                event_id,
                vocab.eos_id,
            ],
        )

    def test_constrained_decode_forces_eos_within_max_target_budget_after_event(self) -> None:
        vocab = Stage1Vocab()
        event_id = vocab.encode_timepoint_event((LaneAction.TAP, LaneAction.NONE, LaneAction.NONE, LaneAction.NONE))
        condition_ids = [
            vocab.bos_id,
            vocab.diff_token_id(0),
            vocab.open_token_id(0),
        ]
        model = _TokenSequenceModel(
            [vocab.ts_token_id(0), event_id, vocab.ts_token_id(100)],
            vocab_size=vocab.size,
        )

        result = constrained_greedy_decode(
            model,
            packed_audio=torch.zeros(1, 600, 160),
            timing_track=torch.zeros(1, 600, 5),
            difficulty_bucket=torch.tensor([0]),
            condition_ids=condition_ids,
            open_hold_mask=0,
            write_duration_ms=8000,
            vocab=vocab,
            max_decode_len=3,
        )

        self.assertEqual(
            result.token_ids,
            [
                vocab.bos_id,
                vocab.diff_token_id(0),
                vocab.open_token_id(0),
                vocab.ts_token_id(0),
                event_id,
                vocab.eos_id,
            ],
        )
        self.assertEqual(len(result.token_ids[3:]), 3)
        self.assertTrue(result.max_decode_len_reached)
        self.assertFalse(result.eos_emitted_by_model)
        self.assertFalse(result.eos_forced_after_pending_ts)

    def test_constrained_decode_rolls_back_pending_ts_within_max_target_budget(self) -> None:
        vocab = Stage1Vocab()
        event_id = vocab.encode_timepoint_event((LaneAction.TAP, LaneAction.NONE, LaneAction.NONE, LaneAction.NONE))
        condition_ids = [
            vocab.bos_id,
            vocab.diff_token_id(0),
            vocab.open_token_id(0),
        ]
        model = _TokenSequenceModel(
            [vocab.ts_token_id(0), event_id, vocab.ts_token_id(100)],
            vocab_size=vocab.size,
        )

        result = constrained_greedy_decode(
            model,
            packed_audio=torch.zeros(1, 600, 160),
            timing_track=torch.zeros(1, 600, 5),
            difficulty_bucket=torch.tensor([0]),
            condition_ids=condition_ids,
            open_hold_mask=0,
            write_duration_ms=8000,
            vocab=vocab,
            max_decode_len=4,
        )

        self.assertEqual(
            result.token_ids,
            [
                vocab.bos_id,
                vocab.diff_token_id(0),
                vocab.open_token_id(0),
                vocab.ts_token_id(0),
                event_id,
                vocab.eos_id,
            ],
        )
        self.assertLessEqual(len(result.token_ids[3:]), 4)
        self.assertTrue(result.max_decode_len_reached)
        self.assertFalse(result.eos_emitted_by_model)
        self.assertTrue(result.eos_forced_after_pending_ts)

    def test_constrained_decode_marks_model_emitted_eos_as_natural(self) -> None:
        vocab = Stage1Vocab()
        condition_ids = [
            vocab.bos_id,
            vocab.diff_token_id(0),
            vocab.open_token_id(0),
        ]
        model = _TokenSequenceModel([vocab.eos_id], vocab_size=vocab.size)

        result = constrained_greedy_decode(
            model,
            packed_audio=torch.zeros(1, 600, 160),
            timing_track=torch.zeros(1, 600, 5),
            difficulty_bucket=torch.tensor([0]),
            condition_ids=condition_ids,
            open_hold_mask=0,
            write_duration_ms=8000,
            vocab=vocab,
            max_decode_len=3,
        )

        self.assertEqual(result.token_ids, condition_ids + [vocab.eos_id])
        self.assertFalse(result.max_decode_len_reached)
        self.assertTrue(result.eos_emitted_by_model)
        self.assertFalse(result.eos_forced_after_pending_ts)


if __name__ == "__main__":
    unittest.main()
