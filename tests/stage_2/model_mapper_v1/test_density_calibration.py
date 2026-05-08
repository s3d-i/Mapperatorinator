import unittest

import torch

from train.stage_2.model_mapper_v1.density_calibration import (
    fit_monotonic_sigmoid_calibration,
    scatter_tokenized_gold_onset_mass,
    smooth_density_mass,
)
from train.stage_2.model_mapper_v1.tokenizer import MapperTimepoint, encode_mapper_window
from train.stage_2.model_mapper_v1.vocab import LaneAction, MapperV1Vocab


def _actions(*actions: LaneAction) -> tuple[LaneAction, ...]:
    padded = list(actions)
    while len(padded) < 4:
        padded.append(LaneAction.NONE)
    return tuple(padded)


class MapperV1DensityCalibrationTests(unittest.TestCase):
    def test_gold_onset_mass_scatters_tap_and_hold_start_only(self) -> None:
        vocab = MapperV1Vocab()
        tokenized = encode_mapper_window(
            [
                MapperTimepoint(40, _actions(LaneAction.TAP, LaneAction.HOLD_START)),
                MapperTimepoint(80, _actions(LaneAction.NONE, LaneAction.HOLD_END)),
            ],
            vocab=vocab,
            write_start_ms=0,
            write_end_ms=8000,
        )

        mass = scatter_tokenized_gold_onset_mass(tokenized, vocab=vocab)

        self.assertEqual(float(mass[2].item()), 2.0)
        self.assertEqual(float(mass[4].item()), 0.0)
        self.assertEqual(float(mass.sum().item()), 2.0)

    def test_smoothing_preserves_total_mass_away_from_edges(self) -> None:
        mass = torch.zeros(400)
        mass[200] = 3.0

        smoothed = smooth_density_mass(mass, radius=5)

        self.assertAlmostEqual(float(smoothed.sum().item()), 3.0, places=5)
        self.assertGreater(float(smoothed[200].item()), float(smoothed[195].item()))

    def test_monotonic_calibration_fit_predicts_increasing_values(self) -> None:
        mass = torch.linspace(0.0, 4.0, 400)
        target = torch.sigmoid(1.5 * mass - 2.0)

        calibration = fit_monotonic_sigmoid_calibration(mass, target)
        pred = calibration.predict(mass)

        self.assertGreaterEqual(calibration.scale, 0.0)
        self.assertLess(float(pred[10].item()), float(pred[-10].item()))

    def test_monotonic_calibration_rejects_nonfinite_targets_before_clamp(self) -> None:
        with self.assertRaisesRegex(ValueError, "density_target"):
            fit_monotonic_sigmoid_calibration(
                torch.arange(4, dtype=torch.float32),
                torch.tensor([0.1, float("inf"), 0.8, 0.9]),
            )


if __name__ == "__main__":
    unittest.main()
