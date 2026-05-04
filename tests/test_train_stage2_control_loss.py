import unittest

import torch

from train.stage_2.features.control_v3_targets import CONFIDENCE_FEATURE_NAMES, MODEL_FEATURE_NAMES, VALUE_FEATURE_NAMES
from train.stage_2.model_control.loss import ControlLossConfig, ControlModelLoss
from train.stage_2.model_control.model import ControlEncoderOutput


class Stage2ControlLossTests(unittest.TestCase):
    def test_loss_uses_confidence_channel_mapping_and_hold_ignores_confidence(self) -> None:
        loss_fn = ControlModelLoss()
        target = _target()
        output = _output(value=0.0, confidence=0.5)
        target[:, :, VALUE_FEATURE_NAMES.index("density_level")] = 1.0
        target[:, :, VALUE_FEATURE_NAMES.index("hold_occupancy")] = 1.0
        target[:, :, MODEL_FEATURE_NAMES.index("density_confidence")] = 0.0
        target[:, :, MODEL_FEATURE_NAMES.index("control_confidence")] = 0.5

        weights = loss_fn.value_weights(
            control_v3_target=target,
            target_valid_mask=torch.ones(1, 100, dtype=torch.bool),
            ln_change_n_eff_target=torch.full((1, 100), 3.0),
        )

        self.assertTrue(torch.equal(weights[..., VALUE_FEATURE_NAMES.index("density_level")], torch.zeros(1, 100)))
        self.assertTrue(torch.all(weights[..., VALUE_FEATURE_NAMES.index("hold_occupancy")] > 0.0))
        result = loss_fn(
            output,
            control_v3_target=target,
            target_valid_mask=torch.ones(1, 100, dtype=torch.bool),
            ln_change_n_eff_target=torch.full((1, 100), 3.0),
        )
        self.assertGreater(float(result.value_loss.item()), 0.0)
        self.assertIn("value/hold_occupancy/weighted_smooth_l1", result.metrics)

    def test_target_mask_removes_invalid_frames_from_loss(self) -> None:
        loss_fn = ControlModelLoss()
        target = _target()
        output = _output(value=0.0, confidence=0.5)
        target[:, 50:, VALUE_FEATURE_NAMES.index("chord_ratio")] = 1.0
        valid = torch.zeros(1, 100, dtype=torch.bool)
        valid[:, :50] = True

        result = loss_fn(
            output,
            control_v3_target=target,
            target_valid_mask=valid,
            ln_change_n_eff_target=torch.full((1, 100), 3.0),
        )

        self.assertEqual(float(result.value_loss.item()), 0.0)
        self.assertEqual(result.metrics["target/masked_frame_count"], 50)

    def test_sparse_balancing_boosts_positive_targets(self) -> None:
        loss_fn = ControlModelLoss(ControlLossConfig(sparse_boost=4.0))
        target = _target(batch=1)
        feature_index = VALUE_FEATURE_NAMES.index("repeat_exact")
        target[:, :50, feature_index] = 0.0
        target[:, 50:, feature_index] = 0.5

        weights = loss_fn.value_weights(
            control_v3_target=target,
            target_valid_mask=torch.ones(1, 100, dtype=torch.bool),
            ln_change_n_eff_target=torch.full((1, 100), 3.0),
        )

        low_weight = weights[0, 0, feature_index]
        high_weight = weights[0, 50, feature_index]
        self.assertGreater(float(high_weight.item()), float(low_weight.item()))
        self.assertAlmostEqual(float(high_weight.item()), 5.0, places=5)

    def test_hand_balance_and_ln_change_support_gates_weights(self) -> None:
        loss_fn = ControlModelLoss()
        target = _target(batch=1)
        hand_balance_index = VALUE_FEATURE_NAMES.index("hand_balance_signed")
        hand_imbalance_index = VALUE_FEATURE_NAMES.index("hand_imbalance_abs")
        ln_change_index = VALUE_FEATURE_NAMES.index("ln_change_rate_gated")
        target[:, :, hand_imbalance_index] = torch.linspace(0.0, 0.25, 100)
        n_eff = torch.tensor([[1.0, 2.0, 2.5, 3.0] + [3.0] * 96], dtype=torch.float32)

        weights = loss_fn.value_weights(
            control_v3_target=target,
            target_valid_mask=torch.ones(1, 100, dtype=torch.bool),
            ln_change_n_eff_target=n_eff,
        )

        self.assertEqual(float(weights[0, 0, hand_balance_index].item()), 0.0)
        self.assertGreater(float(weights[0, -1, hand_balance_index].item()), 0.0)
        self.assertEqual(float(weights[0, 0, ln_change_index].item()), 0.0)
        self.assertEqual(float(weights[0, 1, ln_change_index].item()), 0.0)
        self.assertAlmostEqual(float(weights[0, 2, ln_change_index].item()), 0.3, places=5)
        self.assertAlmostEqual(float(weights[0, 3, ln_change_index].item()), 0.6, places=5)

    def test_confidence_loss_and_compound_output_are_supervised(self) -> None:
        loss_fn = ControlModelLoss()
        target = _target()
        control_index = CONFIDENCE_FEATURE_NAMES.index("control_confidence")
        target[..., len(VALUE_FEATURE_NAMES) + control_index] = 1.0
        output = _output(value=0.0, confidence=0.0)

        result = loss_fn(
            output,
            control_v3_target=target,
            target_valid_mask=torch.ones(1, 100, dtype=torch.bool),
            ln_change_n_eff_target=torch.full((1, 100), 3.0),
        )

        self.assertGreater(float(result.confidence_loss.item()), 0.0)
        self.assertGreater(result.metrics["confidence/compound_control_confidence/mae"], 0.0)


def _target(*, batch: int = 1) -> torch.Tensor:
    target = torch.zeros(batch, 100, len(MODEL_FEATURE_NAMES), dtype=torch.float32)
    for name in CONFIDENCE_FEATURE_NAMES:
        target[..., MODEL_FEATURE_NAMES.index(name)] = 1.0
    return target


def _output(*, batch: int = 1, value: float, confidence: float) -> ControlEncoderOutput:
    confidence_pred = torch.full((batch, 100, len(CONFIDENCE_FEATURE_NAMES)), confidence, dtype=torch.float32)
    control_index = CONFIDENCE_FEATURE_NAMES.index("control_confidence")
    return ControlEncoderOutput(
        value_pred=torch.full((batch, 100, len(VALUE_FEATURE_NAMES)), value, dtype=torch.float32),
        confidence_pred=confidence_pred,
        compound_confidence_pred=confidence_pred[..., control_index : control_index + 1],
        control_memory=torch.zeros(batch, 600, 8),
        memory_padding_mask=torch.zeros(batch, 600, dtype=torch.bool),
    )


if __name__ == "__main__":
    unittest.main()
