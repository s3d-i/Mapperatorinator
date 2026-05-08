import unittest

from train.stage_2.model_mapper_v1.vocab import LaneAction, MapperV1Vocab


class MapperV1VocabTests(unittest.TestCase):
    def test_vocab_has_design_size_and_time_shift_values(self) -> None:
        vocab = MapperV1Vocab()

        self.assertEqual(vocab.pad_id, 0)
        self.assertEqual(vocab.bos_id, 1)
        self.assertEqual(vocab.eos_id, 2)
        self.assertEqual(vocab.size, 280)
        self.assertEqual(len(vocab.time_shift_token_ids), 22)
        self.assertEqual(len(vocab.event_token_ids), 255)
        self.assertNotIn("TS_0", vocab.token_to_id)
        self.assertEqual(vocab.time_shift_values_ms[-3:], (2000, 3000, 4000))

    def test_time_shift_decomposition_is_canonical_largest_first(self) -> None:
        vocab = MapperV1Vocab()

        self.assertEqual(vocab.decompose_time_shift_delta(0), [])
        self.assertEqual(vocab.decompose_time_shift_delta(3270), [3000, 200, 70])
        self.assertEqual(vocab.decompose_time_shift_delta(8000), [4000, 4000])

        with self.assertRaisesRegex(ValueError, "10ms grid"):
            vocab.decompose_time_shift_delta(15)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            vocab.decompose_time_shift_delta(-10)

    def test_event_roundtrip_rejects_all_none_and_counts_onsets(self) -> None:
        vocab = MapperV1Vocab()
        token_id = vocab.encode_event(
            (
                LaneAction.TAP,
                LaneAction.NONE,
                LaneAction.HOLD_START,
                LaneAction.HOLD_END,
            )
        )

        self.assertEqual(
            vocab.decode_event(token_id),
            (
                LaneAction.TAP,
                LaneAction.NONE,
                LaneAction.HOLD_START,
                LaneAction.HOLD_END,
            ),
        )
        self.assertEqual(vocab.event_onset_weight(token_id), 2)
        with self.assertRaisesRegex(ValueError, "all-NONE"):
            vocab.encode_event((LaneAction.NONE,) * 4)


if __name__ == "__main__":
    unittest.main()
