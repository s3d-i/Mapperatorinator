import unittest

import torch

from train.stage_2.model_mapper_v1.grammar import build_grammar_mask, valid_token_mask
from train.stage_2.model_mapper_v1.replay import replay_tokens
from train.stage_2.model_mapper_v1.tokenizer import (
    CrossWindowLongNoteError,
    MapperTimepoint,
    cross_window_ln_state_reason,
    encode_mapper_window,
)
from train.stage_2.model_mapper_v1.vocab import LaneAction, MapperV1Vocab


def _actions(*actions: LaneAction) -> tuple[LaneAction, ...]:
    padded = list(actions)
    while len(padded) < 4:
        padded.append(LaneAction.NONE)
    return tuple(padded)


class MapperV1TokenizerReplayGrammarTests(unittest.TestCase):
    def test_empty_window_shifts_to_write_end_before_eos(self) -> None:
        vocab = MapperV1Vocab()

        tokenized = encode_mapper_window([], vocab=vocab, write_start_ms=0, write_end_ms=8000)

        self.assertEqual(
            [vocab.token_name(token_id) for token_id in tokenized.target_ids],
            ["BOS", "TS_4000", "TS_4000", "EOS"],
        )
        self.assertEqual(tokenized.teacher_current_ms.tolist(), [0, 4000, 8000, 8000])
        self.assertFalse(tokenized.teacher_open_mask.any().item())

    def test_window_tokenization_records_after_consuming_teacher_state(self) -> None:
        vocab = MapperV1Vocab()
        tokenized = encode_mapper_window(
            [
                MapperTimepoint(1000, _actions(LaneAction.HOLD_START)),
                MapperTimepoint(1300, _actions(LaneAction.HOLD_END)),
            ],
            vocab=vocab,
            write_start_ms=0,
            write_end_ms=8000,
        )

        self.assertEqual(tokenized.teacher_current_ms[:4].tolist(), [0, 1000, 1000, 1300])
        self.assertFalse(tokenized.teacher_open_mask[1, 0].item())
        self.assertTrue(tokenized.teacher_open_mask[2, 0].item())
        self.assertEqual(int(tokenized.teacher_open_age_ms[3, 0].item()), 300)
        self.assertTrue(tokenized.close_label_mask[2, 0].item())
        self.assertFalse(tokenized.close_labels[2, 0].item())
        self.assertTrue(tokenized.close_labels[3, 0].item())

    def test_cross_window_lns_are_rejected(self) -> None:
        vocab = MapperV1Vocab()

        with self.assertRaises(CrossWindowLongNoteError):
            encode_mapper_window(
                [MapperTimepoint(1000, _actions(LaneAction.HOLD_END))],
                vocab=vocab,
                write_start_ms=0,
                write_end_ms=8000,
            )
        with self.assertRaises(CrossWindowLongNoteError):
            encode_mapper_window(
                [MapperTimepoint(1000, _actions(LaneAction.HOLD_START))],
                vocab=vocab,
                write_start_ms=0,
                write_end_ms=8000,
            )

    def test_cross_window_ln_spanning_entire_window_is_detected_from_full_timepoints(self) -> None:
        self.assertEqual(
            cross_window_ln_state_reason(
                [
                    MapperTimepoint(500, _actions(LaneAction.HOLD_START)),
                    MapperTimepoint(9000, _actions(LaneAction.HOLD_END)),
                ],
                write_start_ms=1000,
                write_end_ms=9000,
            ),
            "carry-in",
        )

    def test_grammar_enforces_write_end_eos_and_open_ln_dead_end_guard(self) -> None:
        vocab = MapperV1Vocab()
        closed_at_start = valid_token_mask(
            position=0,
            current_ms=0,
            open_mask=0,
            write_start_ms=0,
            write_end_ms=8000,
            vocab=vocab,
        )
        self.assertFalse(closed_at_start[vocab.eos_id].item())
        self.assertTrue(closed_at_start[vocab.time_shift_token_id(4000)].item())

        open_near_end = valid_token_mask(
            position=3,
            current_ms=7000,
            open_mask=0b0001,
            write_start_ms=0,
            write_end_ms=8000,
            vocab=vocab,
        )
        self.assertFalse(open_near_end[vocab.time_shift_token_id(1000)].item())
        self.assertFalse(open_near_end[vocab.eos_id].item())
        close_event = vocab.encode_event(_actions(LaneAction.HOLD_END))
        self.assertTrue(open_near_end[close_event].item())

    def test_build_grammar_mask_returns_negative_inf_for_invalid_tokens(self) -> None:
        vocab = MapperV1Vocab()
        mask = build_grammar_mask(
            current_ms=torch.tensor([[8000]]),
            open_mask=torch.zeros((1, 1, 4), dtype=torch.bool),
            write_start_ms=torch.tensor([0]),
            write_end_ms=torch.tensor([8000]),
            vocab=vocab,
            positions=torch.tensor([[3]]),
        )

        self.assertEqual(mask.shape, (1, 1, vocab.size))
        self.assertEqual(float(mask[0, 0, vocab.eos_id].item()), 0.0)
        self.assertTrue(torch.isneginf(mask[0, 0, vocab.pad_id]))

    def test_replay_rejects_bos_after_position_zero(self) -> None:
        vocab = MapperV1Vocab()

        with self.assertRaisesRegex(ValueError, "BOS"):
            replay_tokens(
                [vocab.bos_id, vocab.bos_id],
                vocab=vocab,
                write_start_ms=0,
                write_end_ms=8000,
            )


if __name__ == "__main__":
    unittest.main()
