import unittest

import torch

from train.stage_2.model_mapper_v1.generation import reconstruct_ln_carry_states
from train.stage_2.model_mapper_v1.grammar import build_grammar_mask, valid_token_mask
from train.stage_2.model_mapper_v1.replay import (
    LNCarryState,
    ReplayError,
    empty_ln_carry_state,
    ln_carry_state_tensors,
    replay_tokens,
)
from train.stage_2.model_mapper_v1.tokenizer import (
    MapperTimepoint,
    encode_full_chart_tokens,
    encode_mapper_window,
)
from train.stage_2.model_mapper_v1.vocab import LaneAction, MapperV1Vocab


def _actions(*actions: LaneAction) -> tuple[LaneAction, ...]:
    padded = list(actions)
    while len(padded) < 4:
        padded.append(LaneAction.NONE)
    return tuple(padded)


class MapperV1TokenizerReplayGrammarTests(unittest.TestCase):
    def test_empty_window_fragment_has_no_synthetic_bos_or_eos(self) -> None:
        vocab = MapperV1Vocab()

        tokenized = encode_mapper_window([], vocab=vocab, write_start_ms=0, write_end_ms=8000)

        self.assertEqual(
            [vocab.token_name(token_id) for token_id in tokenized.target_fragment_ids],
            ["TS_4000", "TS_4000"],
        )
        self.assertEqual(
            [vocab.token_name(token_id) for token_id in tokenized.decoder_input_ids],
            ["BOS", "TS_4000"],
        )
        self.assertEqual(tokenized.target_fragment_current_ms.tolist(), [0, 4000])
        self.assertFalse(tokenized.target_fragment_open_mask.any().item())
        self.assertFalse(tokenized.is_full_chart_end)

    def test_full_chart_tokens_use_bos_and_eos_only_at_chart_boundaries(self) -> None:
        vocab = MapperV1Vocab()

        token_ids = encode_full_chart_tokens(
            [MapperTimepoint(1000, _actions(LaneAction.TAP))],
            vocab=vocab,
            chart_start_ms=0,
            chart_end_ms=8000,
        )

        self.assertEqual(token_ids[0], vocab.bos_id)
        self.assertEqual(token_ids[-1], vocab.eos_id)
        self.assertNotIn(vocab.bos_id, token_ids[1:])
        self.assertNotIn(vocab.eos_id, token_ids[:-1])

    def test_final_chart_window_targets_eos_only_at_chart_end(self) -> None:
        vocab = MapperV1Vocab()

        tokenized = encode_mapper_window(
            [],
            vocab=vocab,
            write_start_ms=0,
            write_end_ms=8000,
            chart_end_ms=8000,
        )

        self.assertTrue(tokenized.is_full_chart_end)
        self.assertEqual(
            [vocab.token_name(token_id) for token_id in tokenized.target_fragment_ids],
            ["TS_4000", "TS_4000", "EOS"],
        )
        self.assertEqual(
            [vocab.token_name(token_id) for token_id in tokenized.decoder_input_ids],
            ["BOS", "TS_4000", "TS_4000"],
        )

    def test_window_tokenization_records_pre_target_replay_state(self) -> None:
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

        self.assertEqual(tokenized.target_fragment_current_ms[:4].tolist(), [0, 1000, 1000, 1300])
        self.assertFalse(tokenized.target_fragment_open_mask[1, 0].item())
        self.assertTrue(tokenized.target_fragment_open_mask[2, 0].item())
        self.assertEqual(int(tokenized.target_fragment_open_age_ms[3, 0].item()), 300)
        self.assertTrue(tokenized.close_label_mask[3, 0].item())
        self.assertTrue(tokenized.close_labels[3, 0].item())

    def test_cross_window_lns_are_represented_by_carry_state(self) -> None:
        vocab = MapperV1Vocab()

        tokenized = encode_mapper_window(
            [
                MapperTimepoint(500, _actions(LaneAction.HOLD_START)),
                MapperTimepoint(9500, _actions(LaneAction.HOLD_END)),
            ],
            vocab=vocab,
            write_start_ms=1000,
            write_end_ms=9000,
        )

        self.assertEqual(tokenized.ln_carry_in.current_ms, 1000)
        self.assertTrue(tokenized.ln_carry_in.open_mask[0])
        self.assertEqual(tokenized.ln_carry_in.open_start_ms[0], 500)
        self.assertEqual(tokenized.ln_carry_in.open_age_ms[0], 500)
        self.assertTrue(tokenized.ln_carry_out.open_mask[0])
        self.assertEqual(tokenized.ln_carry_out.open_start_ms[0], 500)
        self.assertEqual(tokenized.ln_carry_out.open_age_ms[0], 8500)
        self.assertEqual(
            [vocab.token_name(token_id) for token_id in tokenized.target_fragment_ids],
            ["TS_4000", "TS_4000"],
        )

    def test_carry_in_close_at_window_start_is_predicted_from_open_state(self) -> None:
        vocab = MapperV1Vocab()

        tokenized = encode_mapper_window(
            [
                MapperTimepoint(500, _actions(LaneAction.HOLD_START)),
                MapperTimepoint(1000, _actions(LaneAction.HOLD_END)),
            ],
            vocab=vocab,
            write_start_ms=1000,
            write_end_ms=9000,
            chart_end_ms=9000,
        )

        self.assertTrue(tokenized.ln_carry_in.open_mask[0])
        self.assertEqual(tokenized.ln_carry_in.open_start_ms[0], 500)
        self.assertFalse(tokenized.ln_carry_out.open_mask[0])
        self.assertEqual(tokenized.target_fragment_current_ms.tolist(), [1000, 1000, 5000, 9000])
        self.assertEqual(vocab.decode_event(tokenized.target_fragment_ids[0])[0], LaneAction.HOLD_END)
        self.assertTrue(tokenized.target_fragment_open_mask[0, 0].item())
        self.assertTrue(tokenized.close_label_mask[0, 0].item())
        self.assertTrue(tokenized.close_labels[0, 0].item())
        self.assertEqual(vocab.token_name(tokenized.target_fragment_ids[-1]), "EOS")

    def test_write_end_close_remains_for_next_window_start(self) -> None:
        vocab = MapperV1Vocab()
        timepoints = [
            MapperTimepoint(500, _actions(LaneAction.HOLD_START)),
            MapperTimepoint(9000, _actions(LaneAction.HOLD_END)),
        ]

        first = encode_mapper_window(
            timepoints,
            vocab=vocab,
            write_start_ms=1000,
            write_end_ms=9000,
            chart_end_ms=17000,
        )
        second = encode_mapper_window(
            timepoints,
            vocab=vocab,
            write_start_ms=9000,
            write_end_ms=17000,
            chart_end_ms=17000,
        )

        self.assertTrue(first.ln_carry_out.open_mask[0])
        self.assertEqual(first.ln_carry_out.open_start_ms[0], 500)
        self.assertEqual(
            [vocab.token_name(token_id) for token_id in first.target_fragment_ids],
            ["TS_4000", "TS_4000"],
        )
        self.assertTrue(second.ln_carry_in.open_mask[0])
        self.assertEqual(vocab.decode_event(second.target_fragment_ids[0])[0], LaneAction.HOLD_END)
        self.assertFalse(second.ln_carry_out.open_mask[0])

        reconstructed_in, reconstructed_out = reconstruct_ln_carry_states(
            timepoints,
            write_start_ms=1000,
            write_end_ms=9000,
        )
        self.assertEqual(reconstructed_in, first.ln_carry_in)
        self.assertEqual(reconstructed_out, first.ln_carry_out)

    def test_decoder_input_uses_final_full_chart_token_before_window(self) -> None:
        vocab = MapperV1Vocab()

        tokenized = encode_mapper_window(
            [MapperTimepoint(1000, _actions(LaneAction.TAP))],
            vocab=vocab,
            write_start_ms=8000,
            write_end_ms=16000,
        )

        self.assertEqual(vocab.token_name(tokenized.decoder_input_ids[0]), "TS_3000")
        self.assertEqual(vocab.token_name(tokenized.target_fragment_ids[0]), "TS_4000")

    def test_grammar_is_carry_aware_at_write_end(self) -> None:
        vocab = MapperV1Vocab()
        carry_in = empty_ln_carry_state(0)
        matching_carry_out = LNCarryState(
            current_ms=8000,
            open_mask=(True, False, False, False),
            open_start_ms=(6000, None, None, None),
            open_age_ms=(2000, 0, 0, 0),
        )
        open_near_end = valid_token_mask(
            position=3,
            current_ms=7000,
            open_mask=0b0001,
            open_start_ms=(6000, None, None, None),
            open_age_ms=(1000, 0, 0, 0),
            write_start_ms=0,
            write_end_ms=8000,
            ln_carry_in=carry_in,
            ln_carry_out=matching_carry_out,
            is_full_chart_start=False,
            is_full_chart_end=False,
            vocab=vocab,
        )
        self.assertTrue(open_near_end[vocab.time_shift_token_id(1000)].item())
        self.assertFalse(open_near_end[vocab.eos_id].item())

        mismatched_carry_out = empty_ln_carry_state(8000)
        mismatched = valid_token_mask(
            position=3,
            current_ms=7000,
            open_mask=0b0001,
            open_start_ms=(6000, None, None, None),
            open_age_ms=(1000, 0, 0, 0),
            write_start_ms=0,
            write_end_ms=8000,
            ln_carry_in=carry_in,
            ln_carry_out=mismatched_carry_out,
            is_full_chart_start=False,
            is_full_chart_end=False,
            vocab=vocab,
        )
        self.assertFalse(mismatched[vocab.time_shift_token_id(1000)].item())

    def test_eos_is_full_chart_end_only(self) -> None:
        vocab = MapperV1Vocab()
        carry_in = empty_ln_carry_state(0)
        carry_out = empty_ln_carry_state(8000)

        ordinary = valid_token_mask(
            position=2,
            current_ms=8000,
            open_mask=0,
            open_start_ms=(None, None, None, None),
            open_age_ms=(0, 0, 0, 0),
            write_start_ms=0,
            write_end_ms=8000,
            ln_carry_in=carry_in,
            ln_carry_out=carry_out,
            is_full_chart_start=True,
            is_full_chart_end=False,
            vocab=vocab,
        )
        final = valid_token_mask(
            position=2,
            current_ms=8000,
            open_mask=0,
            open_start_ms=(None, None, None, None),
            open_age_ms=(0, 0, 0, 0),
            write_start_ms=0,
            write_end_ms=8000,
            ln_carry_in=carry_in,
            ln_carry_out=carry_out,
            is_full_chart_start=True,
            is_full_chart_end=True,
            vocab=vocab,
        )

        self.assertFalse(ordinary[vocab.eos_id].item())
        self.assertTrue(final[vocab.eos_id].item())

    def test_build_grammar_mask_returns_negative_inf_for_invalid_tokens(self) -> None:
        vocab = MapperV1Vocab()
        tokenized = encode_mapper_window([], vocab=vocab, write_start_ms=0, write_end_ms=8000)
        mask = build_grammar_mask(
            current_ms=tokenized.target_fragment_current_ms.unsqueeze(0),
            open_mask=tokenized.target_fragment_open_mask.unsqueeze(0),
            open_start_ms=tokenized.target_fragment_open_start_ms.unsqueeze(0),
            open_age_ms=tokenized.target_fragment_open_age_ms.unsqueeze(0),
            write_start_ms=torch.tensor([0]),
            write_end_ms=torch.tensor([8000]),
            ln_carry_in=ln_carry_state_tensors(tokenized.ln_carry_in),
            ln_carry_out=ln_carry_state_tensors(tokenized.ln_carry_out),
            is_full_chart_start=torch.tensor([True]),
            is_full_chart_end=torch.tensor([False]),
            vocab=vocab,
        )

        self.assertEqual(mask.shape, (1, 2, vocab.size))
        self.assertEqual(float(mask[0, 0, vocab.time_shift_token_id(4000)].item()), 0.0)
        self.assertTrue(torch.isneginf(mask[0, 0, vocab.pad_id]))
        self.assertTrue(torch.isneginf(mask[0, 0, vocab.bos_id]))

    def test_replay_rejects_bos_inside_window_fragment(self) -> None:
        vocab = MapperV1Vocab()

        with self.assertRaisesRegex(ReplayError, "BOS"):
            replay_tokens(
                [vocab.bos_id],
                vocab=vocab,
                write_start_ms=0,
                write_end_ms=8000,
                ln_carry_in=empty_ln_carry_state(0),
                ln_carry_out=empty_ln_carry_state(8000),
            )


if __name__ == "__main__":
    unittest.main()
