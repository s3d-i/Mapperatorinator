import unittest

import torch

from train.stage_2.model_control.context import (
    CONTEXT_LENGTH_FRAMES,
    TARGET_OFFSET_IN_CONTEXT,
    prepare_control_context_batch,
)


class Stage2ControlContextTests(unittest.TestCase):
    def test_slices_start_middle_and_end_contexts_with_regenerated_target_masks(self) -> None:
        batch = _full_song_batch(frame_counts=[300, 700, 275], target_starts=[0, 250, 200])
        batch["target_valid_mask"] = torch.zeros(3, 100, dtype=torch.bool)

        out = prepare_control_context_batch(batch)

        self.assertEqual(out["context_mel"].shape, (3, CONTEXT_LENGTH_FRAMES, 160))
        self.assertEqual(out["context_dense_timing_v2"].shape, (3, CONTEXT_LENGTH_FRAMES, 4))
        self.assertEqual(out["context_padding_mask"].shape, (3, CONTEXT_LENGTH_FRAMES))

        self.assertTrue(out["context_padding_mask"][0, :TARGET_OFFSET_IN_CONTEXT].all())
        self.assertFalse(out["context_padding_mask"][0, 250:550].any())
        self.assertTrue(out["context_padding_mask"][0, 550:].all())
        self.assertEqual(float(out["context_mel"][0, 250, 0].item()), 1.0)
        self.assertEqual(float(out["context_mel"][0, 549, 0].item()), 300.0)
        self.assertEqual(float(out["context_mel"][0, 249, 0].item()), 0.0)

        self.assertFalse(out["context_padding_mask"][1].any())
        self.assertEqual(float(out["context_mel"][1, 0, 0].item()), 1.0)
        self.assertEqual(float(out["context_mel"][1, -1, 0].item()), 600.0)

        self.assertTrue(out["context_padding_mask"][2, :50].all())
        self.assertFalse(out["context_padding_mask"][2, 50:325].any())
        self.assertTrue(out["context_padding_mask"][2, 325:].all())
        self.assertEqual(float(out["context_mel"][2, 50, 0].item()), 1.0)
        self.assertEqual(float(out["context_mel"][2, 324, 0].item()), 275.0)

        self.assertTrue(out["target_valid_mask"][0].all())
        self.assertTrue(out["target_valid_mask"][1].all())
        self.assertTrue(out["target_valid_mask"][2, :75].all())
        self.assertFalse(out["target_valid_mask"][2, 75:].any())

    def test_rejects_target_start_outside_song(self) -> None:
        batch = _full_song_batch(frame_counts=[100], target_starts=[100])

        with self.assertRaisesRegex(ValueError, "target_start_frame"):
            prepare_control_context_batch(batch)

    def test_context_honors_padding_mask_but_target_mask_uses_frame_count_formula(self) -> None:
        batch = _full_song_batch(frame_counts=[600], target_starts=[250])
        batch["padding_mask"][0, 10] = True
        batch["padding_mask"][0, 260] = True
        batch["full_mel"][0, 10, 0] = float("nan")
        batch["full_mel"][0, 260, 0] = float("nan")
        batch["full_dense_timing_v2"][0, 10, 0] = float("nan")
        batch["full_dense_timing_v2"][0, 260, 0] = float("nan")

        out = prepare_control_context_batch(batch)

        self.assertTrue(out["context_padding_mask"][0, 10].item())
        self.assertEqual(float(out["context_mel"][0, 10, 0].item()), 0.0)
        self.assertTrue(out["target_valid_mask"][0, 10].item())

    def test_rejects_fractional_frame_index_tensors(self) -> None:
        batch = _full_song_batch(frame_counts=[600], target_starts=[250])
        batch["target_start_frame"] = torch.tensor([250.9], dtype=torch.float32)

        with self.assertRaisesRegex(ValueError, "integer frame"):
            prepare_control_context_batch(batch)


def _full_song_batch(*, frame_counts: list[int], target_starts: list[int]) -> dict[str, torch.Tensor]:
    batch_size = len(frame_counts)
    padded_frames = max(frame_counts)
    full_mel = torch.zeros(batch_size, padded_frames, 160, dtype=torch.float32)
    full_dense_timing_v2 = torch.zeros(batch_size, padded_frames, 4, dtype=torch.float32)
    padding_mask = torch.ones(batch_size, padded_frames, dtype=torch.bool)
    for batch_index, frame_count in enumerate(frame_counts):
        frame_values = torch.arange(1, frame_count + 1, dtype=torch.float32)
        full_mel[batch_index, :frame_count] = frame_values.reshape(-1, 1)
        full_dense_timing_v2[batch_index, :frame_count] = (frame_values * 0.01).reshape(-1, 1)
        padding_mask[batch_index, :frame_count] = False
    return {
        "full_mel": full_mel,
        "full_dense_timing_v2": full_dense_timing_v2,
        "padding_mask": padding_mask,
        "frame_count": torch.tensor(frame_counts, dtype=torch.long),
        "target_start_frame": torch.tensor(target_starts, dtype=torch.long),
    }


if __name__ == "__main__":
    unittest.main()
