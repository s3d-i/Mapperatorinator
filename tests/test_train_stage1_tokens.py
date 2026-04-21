import unittest

from train.stage1_oracle.events.canonical import CanonicalTimepoint, LaneAction
from train.stage1_oracle.events.tokens import (
    Stage1Vocab,
    TokenizedWindow,
    decompose_ts_delta,
    decode_target_tokens,
    encode_window_tokens,
)


def _lane_actions(*actions: LaneAction) -> tuple[LaneAction, ...]:
    padded = list(actions)
    while len(padded) < 4:
        padded.append(LaneAction.NONE)
    return tuple(padded)


class Stage1TokenTests(unittest.TestCase):
    def test_vocab_has_deterministic_frozen_sizes_and_bucket_ids(self) -> None:
        vocab = Stage1Vocab()

        self.assertEqual(vocab.pad_id, 0)
        self.assertEqual(vocab.bos_id, 1)
        self.assertEqual(vocab.eos_id, 2)
        self.assertEqual(vocab.size, 392)
        self.assertEqual(len(vocab.diff_token_ids), 17)
        self.assertEqual(len(vocab.open_token_ids), 16)
        self.assertEqual(len(vocab.ts_token_ids), 101)
        self.assertEqual(len(vocab.event_token_ids), 255)
        self.assertEqual(vocab.difficulty_bucket_id(2.0), 0)
        self.assertEqual(vocab.difficulty_bucket_id(2.12), 0)
        self.assertEqual(vocab.difficulty_bucket_id(2.13), 1)
        self.assertEqual(vocab.difficulty_bucket_id(6.0), 16)

        with self.assertRaisesRegex(ValueError, "outside supported"):
            vocab.difficulty_bucket_id(6.01)

    def test_event_token_roundtrip_rejects_all_empty_event(self) -> None:
        vocab = Stage1Vocab()
        actions = _lane_actions(
            LaneAction.TAP,
            LaneAction.NONE,
            LaneAction.HOLD_START,
            LaneAction.HOLD_END,
        )

        token_id = vocab.encode_timepoint_event(actions)

        self.assertEqual(vocab.decode_event_token(token_id), actions)
        with self.assertRaisesRegex(ValueError, "all-empty"):
            vocab.encode_timepoint_event(_lane_actions())

    def test_ts_decomposition_is_greedy_and_rejects_invalid_grid_values(self) -> None:
        self.assertEqual(decompose_ts_delta(0), [0])
        self.assertEqual(decompose_ts_delta(990), [990])
        self.assertEqual(decompose_ts_delta(1000), [1000])
        self.assertEqual(decompose_ts_delta(1500), [1000, 500])
        self.assertEqual(decompose_ts_delta(2700), [1000, 1000, 700])

        with self.assertRaisesRegex(ValueError, "10ms grid"):
            decompose_ts_delta(15)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            decompose_ts_delta(-10)

    def test_window_tokenization_uses_forced_prefix_and_relative_deltas(self) -> None:
        vocab = Stage1Vocab()
        tokenized = encode_window_tokens(
            [
                CanonicalTimepoint(8000, _lane_actions(LaneAction.TAP)),
                CanonicalTimepoint(9500, _lane_actions(LaneAction.NONE, LaneAction.HOLD_START)),
            ],
            vocab=vocab,
            write_start_ms=8000,
            write_end_ms=12000,
            difficulty=4.25,
            open_hold_mask=0b0010,
        )

        self.assertIsInstance(tokenized, TokenizedWindow)
        self.assertEqual(
            tokenized.condition_ids,
            [
                vocab.bos_id,
                vocab.diff_token_id(vocab.difficulty_bucket_id(4.25)),
                vocab.open_token_id(0b0010),
            ],
        )
        self.assertEqual(
            [vocab.token_name(token_id) for token_id in tokenized.target_ids],
            [
                "TS_0",
                vocab.token_name(vocab.encode_timepoint_event(_lane_actions(LaneAction.TAP))),
                "TS_1000",
                "TS_500",
                vocab.token_name(vocab.encode_timepoint_event(_lane_actions(LaneAction.NONE, LaneAction.HOLD_START))),
                "EOS",
            ],
        )

    def test_empty_window_target_is_eos_only(self) -> None:
        vocab = Stage1Vocab()
        tokenized = encode_window_tokens(
            [],
            vocab=vocab,
            write_start_ms=0,
            write_end_ms=8000,
            difficulty=2.5,
            open_hold_mask=0,
        )

        self.assertEqual(tokenized.target_ids, [vocab.eos_id])

    def test_decode_target_tokens_returns_window_relative_timepoints(self) -> None:
        vocab = Stage1Vocab()
        first_event = vocab.encode_timepoint_event(_lane_actions(LaneAction.TAP))
        second_event = vocab.encode_timepoint_event(_lane_actions(LaneAction.NONE, LaneAction.HOLD_START))

        decoded = decode_target_tokens(
            [
                vocab.ts_token_id(0),
                first_event,
                vocab.ts_token_id(1000),
                vocab.ts_token_id(500),
                second_event,
                vocab.eos_id,
            ],
            vocab=vocab,
            write_duration_ms=4000,
        )

        self.assertEqual([timepoint.time_ms for timepoint in decoded], [0, 1500])


if __name__ == "__main__":
    unittest.main()
