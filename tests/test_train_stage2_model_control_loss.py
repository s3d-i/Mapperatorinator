import math
import unittest

import torch

from train.stage_2.features.control_v3_targets import CONFIDENCE_FEATURE_NAMES, MODEL_FEATURE_NAMES, VALUE_FEATURE_NAMES
from train.stage_2.model_control.loss import ControlLossConfig, ControlModelLoss
from train.stage_2.model_control.model import ControlEncoderOutput


class Stage2ControlLossTests(unittest.TestCase):
    def test_confidence_weighting_uses_feature_family_and_hold_ignores_confidence(self) -> None:
        output, target, mask = _loss_case()
        target[:, :, MODEL_FEATURE_NAMES.index("density_confidence")] = 0.0
        target[:, :, MODEL_FEATURE_NAMES.index("control_confidence")] = 1.0
        output.value_pred[:, :, VALUE_FEATURE_NAMES.index("density_level")] = 10.0
        output.value_pred[:, :, VALUE_FEATURE_NAMES.index("hold_occupancy")] = 1.0
        output.confidence_pred.copy_(target[:, :, [MODEL_FEATURE_NAMES.index(name) for name in CONFIDENCE_FEATURE_NAMES]])

        loss_fn = ControlModelLoss()
        result = loss_fn(output, control_v3_target=target, target_valid_mask=mask, ln_change_n_eff_target=_n_eff())
        weights = loss_fn.value_weights(control_v3_target=target, target_valid_mask=mask, ln_change_n_eff_target=_n_eff())

        self.assertEqual(float(weights[..., VALUE_FEATURE_NAMES.index("density_level")].sum().item()), 0.0)
        self.assertEqual(float(weights[..., VALUE_FEATURE_NAMES.index("hold_occupancy")].sum().item()), 100.0)
        self.assertAlmostEqual(float(result.value_loss.item()), 0.95, places=5)
        self.assertAlmostEqual(float(result.confidence_loss.item()), 0.0, places=6)

    def test_target_valid_mask_removes_invalid_tail_from_value_and_confidence_loss(self) -> None:
        output, target, mask = _loss_case()
        mask[:, :50] = True
        mask[:, 50:] = False
        output.value_pred[:, 50:, VALUE_FEATURE_NAMES.index("hold_occupancy")] = 1.0
        output.confidence_pred[:, 50:, CONFIDENCE_FEATURE_NAMES.index("control_confidence")] = 1.0

        result = ControlModelLoss()(output, control_v3_target=target, target_valid_mask=mask, ln_change_n_eff_target=_n_eff())

        self.assertEqual(float(result.total_loss.item()), 0.0)
        self.assertEqual(result.metrics["target/masked_frame_count"], 50)

    def test_masked_nonfinite_target_channels_do_not_affect_losses_or_metrics(self) -> None:
        output, target, mask = _loss_case()
        mask[:, :50] = True
        mask[:, 50:] = False
        target[:, 50:, :] = float("nan")
        target[:, 50:, MODEL_FEATURE_NAMES.index("control_confidence")] = float("inf")
        output.value_pred[:, 50:, :] = float("nan")
        output.confidence_pred[:, 50:, :] = float("nan")
        n_eff = _n_eff()
        n_eff[:, 50:] = float("nan")

        result = ControlModelLoss()(output, control_v3_target=target, target_valid_mask=mask, ln_change_n_eff_target=n_eff)

        self.assertTrue(torch.isfinite(result.total_loss).item())
        self.assertTrue(torch.isfinite(result.value_loss).item())
        self.assertTrue(torch.isfinite(result.confidence_loss).item())
        self.assertEqual(float(result.total_loss.item()), 0.0)
        self.assertEqual(float(result.value_loss.item()), 0.0)
        self.assertEqual(float(result.confidence_loss.item()), 0.0)
        for key in (
            "loss/total",
            "loss/value",
            "loss/confidence",
            "value/hold_occupancy/weighted_smooth_l1",
            "value/hold_occupancy/mae",
            "confidence/control_confidence/mae",
            "confidence/compound_control_confidence/mae",
        ):
            self.assertTrue(math.isfinite(result.metrics[key]), key)
            self.assertEqual(result.metrics[key], 0.0)
        for collection_name, collection in (
            ("metrics", result.metrics),
            ("metric_numerators", result.metric_numerators),
            ("metric_denominators", result.metric_denominators),
        ):
            for key, value in collection.items():
                self.assertTrue(math.isfinite(value), f"{collection_name}[{key}]={value}")

    def test_valid_nonfinite_targets_and_sidecar_raise(self) -> None:
        output, target, mask = _loss_case()
        target[:, 0, MODEL_FEATURE_NAMES.index("hold_occupancy")] = float("nan")
        with self.assertRaisesRegex(ValueError, "control_v3_target"):
            ControlModelLoss()(output, control_v3_target=target, target_valid_mask=mask, ln_change_n_eff_target=_n_eff())

        output, target, mask = _loss_case()
        n_eff = _n_eff()
        n_eff[:, 0] = float("inf")
        with self.assertRaisesRegex(ValueError, "ln_change_n_eff_target"):
            ControlModelLoss()(output, control_v3_target=target, target_valid_mask=mask, ln_change_n_eff_target=n_eff)

    def test_valid_nonfinite_predictions_raise(self) -> None:
        output, target, mask = _loss_case()
        output.value_pred[:, 0, VALUE_FEATURE_NAMES.index("hold_occupancy")] = float("nan")
        with self.assertRaisesRegex(ValueError, "value_pred"):
            ControlModelLoss()(output, control_v3_target=target, target_valid_mask=mask, ln_change_n_eff_target=_n_eff())

        output, target, mask = _loss_case()
        output.confidence_pred[:, 0, CONFIDENCE_FEATURE_NAMES.index("control_confidence")] = float("inf")
        with self.assertRaisesRegex(ValueError, "confidence_pred"):
            ControlModelLoss()(output, control_v3_target=target, target_valid_mask=mask, ln_change_n_eff_target=_n_eff())

        output, target, mask = _loss_case()
        compound_confidence_pred = torch.zeros_like(output.compound_confidence_pred)
        compound_confidence_pred[:, 0, 0] = float("nan")
        output = ControlEncoderOutput(
            value_pred=output.value_pred,
            confidence_pred=output.confidence_pred,
            compound_confidence_pred=compound_confidence_pred,
            control_memory=output.control_memory,
            memory_padding_mask=output.memory_padding_mask,
        )
        with self.assertRaisesRegex(ValueError, "compound_confidence_pred"):
            ControlModelLoss()(output, control_v3_target=target, target_valid_mask=mask, ln_change_n_eff_target=_n_eff())

    def test_sparse_balancing_increases_positive_sparse_feature_weight(self) -> None:
        boosted_output, boosted_target, mask = _loss_case()
        boosted_target[:, :, MODEL_FEATURE_NAMES.index("jack_confidence")] = 1.0
        boosted_target[:, :, MODEL_FEATURE_NAMES.index("chord_confidence")] = 1.0
        boosted_target[:, :, MODEL_FEATURE_NAMES.index("jack_excess")] = 1.0
        boosted_output.value_pred[:, :, VALUE_FEATURE_NAMES.index("jack_excess")] = 0.0

        no_boost_loss = ControlModelLoss(ControlLossConfig(sparse_boost=0.0))
        boosted_loss = ControlModelLoss(ControlLossConfig(sparse_boost=4.0))
        no_boost = no_boost_loss(
            boosted_output,
            control_v3_target=boosted_target,
            target_valid_mask=mask,
            ln_change_n_eff_target=_n_eff(),
        )
        boosted = boosted_loss(
            boosted_output,
            control_v3_target=boosted_target,
            target_valid_mask=mask,
            ln_change_n_eff_target=_n_eff(),
        )
        no_boost_weights = no_boost_loss.value_weights(
            control_v3_target=boosted_target,
            target_valid_mask=mask,
            ln_change_n_eff_target=_n_eff(),
        )
        boosted_weights = boosted_loss.value_weights(
            control_v3_target=boosted_target,
            target_valid_mask=mask,
            ln_change_n_eff_target=_n_eff(),
        )
        jack_index = VALUE_FEATURE_NAMES.index("jack_excess")

        self.assertAlmostEqual(
            float(boosted_weights[..., jack_index].sum().item()),
            float(no_boost_weights[..., jack_index].sum().item()) * 5.0,
            places=4,
        )
        self.assertGreater(float(boosted.value_loss.item()), float(no_boost.value_loss.item()))

    def test_hand_balance_signed_is_gated_by_target_hand_imbalance(self) -> None:
        output, target, mask = _loss_case()
        target[:, :, MODEL_FEATURE_NAMES.index("hand_confidence")] = 1.0
        output.value_pred[:, :, VALUE_FEATURE_NAMES.index("hand_balance_signed")] = 1.0
        target[:, :, MODEL_FEATURE_NAMES.index("hand_balance_signed")] = -1.0
        target[:, :, MODEL_FEATURE_NAMES.index("hand_imbalance_abs")] = 0.0

        loss_fn = ControlModelLoss()
        gated_off = loss_fn(output, control_v3_target=target, target_valid_mask=mask, ln_change_n_eff_target=_n_eff())
        gated_off_weights = loss_fn.value_weights(control_v3_target=target, target_valid_mask=mask, ln_change_n_eff_target=_n_eff())
        target[:, :, MODEL_FEATURE_NAMES.index("hand_imbalance_abs")] = 0.25
        output.value_pred[:, :, VALUE_FEATURE_NAMES.index("hand_imbalance_abs")] = 0.25
        gated_on = loss_fn(output, control_v3_target=target, target_valid_mask=mask, ln_change_n_eff_target=_n_eff())
        gated_on_weights = loss_fn.value_weights(control_v3_target=target, target_valid_mask=mask, ln_change_n_eff_target=_n_eff())
        hand_balance_index = VALUE_FEATURE_NAMES.index("hand_balance_signed")

        self.assertEqual(float(gated_off_weights[..., hand_balance_index].sum().item()), 0.0)
        self.assertGreater(float(gated_on_weights[..., hand_balance_index].sum().item()), 0.0)
        self.assertGreater(float(gated_on.value_loss.item()), float(gated_off.value_loss.item()))

    def test_ln_change_support_weight_uses_n_eff_sidecar(self) -> None:
        output, target, mask = _loss_case()
        target[:, :, MODEL_FEATURE_NAMES.index("ln_change_confidence")] = 1.0
        target[:, :, MODEL_FEATURE_NAMES.index("ln_change_rate_gated")] = 1.0
        n_eff = torch.full((1, 100), 2.0)

        loss_fn = ControlModelLoss()
        unsupported = loss_fn(
            output,
            control_v3_target=target,
            target_valid_mask=mask,
            ln_change_n_eff_target=n_eff,
        )
        unsupported_weights = loss_fn.value_weights(
            control_v3_target=target,
            target_valid_mask=mask,
            ln_change_n_eff_target=n_eff,
        )
        n_eff.fill_(3.0)
        supported = loss_fn(
            output,
            control_v3_target=target,
            target_valid_mask=mask,
            ln_change_n_eff_target=n_eff,
        )
        supported_weights = loss_fn.value_weights(
            control_v3_target=target,
            target_valid_mask=mask,
            ln_change_n_eff_target=n_eff,
        )
        ln_change_index = VALUE_FEATURE_NAMES.index("ln_change_rate_gated")

        self.assertEqual(float(unsupported_weights[..., ln_change_index].sum().item()), 0.0)
        self.assertGreater(float(supported_weights[..., ln_change_index].sum().item()), 0.0)
        self.assertGreater(float(supported.value_loss.item()), float(unsupported.value_loss.item()))

    def test_total_loss_applies_confidence_loss_weight(self) -> None:
        output, target, mask = _loss_case()
        output.confidence_pred[:, :, CONFIDENCE_FEATURE_NAMES.index("control_confidence")] = 1.0
        target[:, :, MODEL_FEATURE_NAMES.index("control_confidence")] = 0.0

        result = ControlModelLoss(ControlLossConfig(confidence_loss_weight=0.5))(
            output,
            control_v3_target=target,
            target_valid_mask=mask,
            ln_change_n_eff_target=_n_eff(),
        )

        self.assertAlmostEqual(
            float(result.total_loss.item()),
            float(result.value_loss.item()) + 0.5 * float(result.confidence_loss.item()),
            places=6,
        )
        self.assertGreater(float(result.confidence_loss.item()), 0.0)

    def test_requires_ln_change_n_eff_sidecar_and_rejects_negative_feature_weights(self) -> None:
        output, target, mask = _loss_case()
        with self.assertRaisesRegex(ValueError, "ln_change_n_eff_target"):
            ControlModelLoss()(output, control_v3_target=target, target_valid_mask=mask)

        feature_weights = {name: 1.0 for name in VALUE_FEATURE_NAMES}
        feature_weights["density_level"] = -1.0
        with self.assertRaisesRegex(ValueError, "feature_weights"):
            ControlModelLoss(ControlLossConfig(feature_weights=feature_weights))


def _loss_case() -> tuple[ControlEncoderOutput, torch.Tensor, torch.Tensor]:
    value_pred = torch.zeros(1, 100, len(VALUE_FEATURE_NAMES), dtype=torch.float32)
    confidence_pred = torch.zeros(1, 100, len(CONFIDENCE_FEATURE_NAMES), dtype=torch.float32)
    target = torch.zeros(1, 100, len(MODEL_FEATURE_NAMES), dtype=torch.float32)
    mask = torch.ones(1, 100, dtype=torch.bool)
    output = ControlEncoderOutput(
        value_pred=value_pred,
        confidence_pred=confidence_pred,
        compound_confidence_pred=confidence_pred[
            :,
            :,
            CONFIDENCE_FEATURE_NAMES.index("control_confidence") : CONFIDENCE_FEATURE_NAMES.index("control_confidence") + 1,
        ],
        control_memory=torch.zeros(1, 600, 8),
        memory_padding_mask=torch.zeros(1, 600, dtype=torch.bool),
    )
    return output, target, mask


def _n_eff() -> torch.Tensor:
    return torch.full((1, 100), 3.0, dtype=torch.float32)


if __name__ == "__main__":
    unittest.main()
