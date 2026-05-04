import unittest

import torch

from train.stage_2.model_control.context import (
    CONTEXT_LENGTH_FRAMES,
    TARGET_OFFSET_IN_CONTEXT,
    prepare_control_context_batch,
)


class Stage2ControlContextTests(unittest.TestCase):
    def test_context_slice_at_song_start_zero_pads_left_side_and_regenerates_target_mask(self) -> None:
        batch = _batch(frame_count=400, target_start_frame=0)

        prepared = prepare_control_context_batch(batch)

        self.assertEqual(prepared["context_mel"].shape, (1, CONTEXT_LENGTH_FRAMES, 160))
        self.assertTrue(prepared["context_padding_mask"][0, :TARGET_OFFSET_IN_CONTEXT].all())
        self.assertFalse(prepared["context_padding_mask"][0, TARGET_OFFSET_IN_CONTEXT:650].any())
        self.assertTrue(torch.equal(prepared["context_mel"][0, TARGET_OFFSET_IN_CONTEXT, :,], batch["full_mel"][0, 0]))
        self.assertTrue(prepared["target_valid_mask"].all())

    def test_context_slice_in_song_middle_has_no_context_padding(self) -> None:
        batch = _batch(frame_count=900, target_start_frame=300)

        prepared = prepare_control_context_batch(batch)

        self.assertFalse(prepared["context_padding_mask"].any())
        self.assertEqual(int(prepared["context_start_frame"][0].item()), 50)
        self.assertTrue(torch.equal(prepared["context_dense_timing_v2"][0, 0], batch["full_dense_timing_v2"][0, 50]))
        self.assertTrue(torch.equal(prepared["context_dense_timing_v2"][0, 250], batch["full_dense_timing_v2"][0, 300]))

    def test_context_slice_at_song_end_pads_right_side_and_masks_target_tail(self) -> None:
        batch = _batch(frame_count=340, target_start_frame=300)

        prepared = prepare_control_context_batch(batch)

        self.assertFalse(prepared["context_padding_mask"][0, :290].any())
        self.assertTrue(prepared["context_padding_mask"][0, 290:].all())
        self.assertEqual(int(prepared["target_valid_mask"].sum().item()), 40)
        self.assertTrue(prepared["target_valid_mask"][0, :40].all())
        self.assertFalse(prepared["target_valid_mask"][0, 40:].any())

    def test_target_valid_mask_uses_frame_count_formula_not_internal_padding(self) -> None:
        batch = _batch(frame_count=600, target_start_frame=250)
        batch["padding_mask"][0, 260] = True
        batch["full_mel"][0, 260, 0] = float("nan")
        batch["full_dense_timing_v2"][0, 260, 0] = float("nan")

        prepared = prepare_control_context_batch(batch)

        self.assertTrue(prepared["context_padding_mask"][0, 260].item())
        self.assertTrue(prepared["target_valid_mask"][0, 10].item())


def _batch(*, frame_count: int, target_start_frame: int) -> dict[str, torch.Tensor]:
    full_mel = torch.arange(frame_count * 160, dtype=torch.float32).reshape(1, frame_count, 160)
    full_dense_timing_v2 = torch.arange(frame_count * 4, dtype=torch.float32).reshape(1, frame_count, 4)
    return {
        "full_mel": full_mel,
        "full_dense_timing_v2": full_dense_timing_v2,
        "padding_mask": torch.zeros(1, frame_count, dtype=torch.bool),
        "frame_count": torch.tensor([frame_count], dtype=torch.long),
        "target_start_frame": torch.tensor([target_start_frame], dtype=torch.long),
        "control_v3_target": torch.zeros(1, 100, 20),
        "target_valid_mask": torch.ones(1, 100, dtype=torch.bool),
        "ln_change_n_eff_target": torch.full((1, 100), 3.0),
        "normalized_difficulty": torch.zeros(1),
    }


if __name__ == "__main__":
    unittest.main()
