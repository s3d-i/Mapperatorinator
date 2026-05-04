import unittest

import torch
from torch import nn

from train.stage_2.features.control_v3_targets import CONFIDENCE_FEATURE_NAMES, CONTROL_CONFIDENCE_FEATURE_NAME
from train.stage_2.model_control.model import (
    BOUNDED_01_VALUE_FEATURES,
    ControlEncoder,
    ControlEncoderConfig,
    VALUE_FEATURE_NAMES,
)


class Stage2ControlEncoderTests(unittest.TestCase):
    def test_forward_shapes_ranges_masks_and_compound_confidence_alias(self) -> None:
        torch.manual_seed(3)
        model = ControlEncoder(
            ControlEncoderConfig(
                d_model=32,
                heads=4,
                layers=1,
                ffn_dim=64,
                dropout=0.0,
            )
        )
        model.eval()
        context_padding_mask = torch.zeros(2, 600, dtype=torch.bool)
        context_padding_mask[1, 500:] = True

        out = model(
            context_mel=torch.randn(2, 600, 160),
            context_dense_timing_v2=torch.randn(2, 600, 4),
            normalized_difficulty=torch.tensor([-1.0, 1.0]),
            context_padding_mask=context_padding_mask,
        )

        self.assertEqual(out.value_pred.shape, (2, 100, 12))
        self.assertEqual(out.confidence_pred.shape, (2, 100, 8))
        self.assertEqual(out.compound_confidence_pred.shape, (2, 100, 1))
        self.assertEqual(out.control_memory.shape, (2, 600, 32))
        self.assertTrue(torch.equal(out.memory_padding_mask, context_padding_mask))
        self.assertEqual(float(out.control_memory[1, 500:].abs().max().item()), 0.0)
        self.assertTrue(torch.all((out.confidence_pred >= 0.0) & (out.confidence_pred <= 1.0)))

        bounded_indexes = [VALUE_FEATURE_NAMES.index(name) for name in BOUNDED_01_VALUE_FEATURES]
        bounded_values = out.value_pred[..., bounded_indexes]
        self.assertTrue(torch.all((bounded_values >= 0.0) & (bounded_values <= 1.0)))

        control_index = CONFIDENCE_FEATURE_NAMES.index(CONTROL_CONFIDENCE_FEATURE_NAME)
        self.assertTrue(
            torch.allclose(
                out.compound_confidence_pred,
                out.confidence_pred[..., control_index : control_index + 1],
            )
        )
        self.assertLess(model.parameter_count(), 25_000_000)

    def test_film_identity_initialization_makes_initial_outputs_difficulty_invariant(self) -> None:
        torch.manual_seed(5)
        model = ControlEncoder(
            ControlEncoderConfig(
                d_model=24,
                heads=4,
                layers=2,
                ffn_dim=48,
                dropout=0.0,
            )
        )
        model.eval()
        context_mel = torch.randn(1, 600, 160).expand(2, -1, -1).contiguous()
        context_dense_timing_v2 = torch.randn(1, 600, 4).expand(2, -1, -1).contiguous()

        for film in [model.stem_film, *model.block_films]:
            final = film.net[-1]
            self.assertIsInstance(final, nn.Linear)
            self.assertEqual(float(final.weight.abs().max().item()), 0.0)
            self.assertEqual(float(final.bias.abs().max().item()), 0.0)

        out = model(
            context_mel=context_mel,
            context_dense_timing_v2=context_dense_timing_v2,
            normalized_difficulty=torch.tensor([-1.0, 1.0]),
        )

        self.assertTrue(torch.allclose(out.value_pred[0], out.value_pred[1], atol=1e-6))
        self.assertTrue(torch.allclose(out.confidence_pred[0], out.confidence_pred[1], atol=1e-6))
        self.assertTrue(torch.allclose(out.control_memory[0], out.control_memory[1], atol=1e-6))

    def test_masked_context_values_do_not_change_predictions(self) -> None:
        torch.manual_seed(7)
        model = ControlEncoder(
            ControlEncoderConfig(
                d_model=32,
                heads=4,
                layers=1,
                ffn_dim=64,
                dropout=0.0,
            )
        )
        model.eval()
        context_padding_mask = torch.zeros(1, 600, dtype=torch.bool)
        context_padding_mask[:, :250] = True
        context_mel = torch.randn(1, 600, 160)
        context_dense_timing_v2 = torch.randn(1, 600, 4)
        changed_mel = context_mel.clone()
        changed_timing = context_dense_timing_v2.clone()
        changed_mel[:, :250] = 1000.0
        changed_timing[:, :250] = -1000.0

        baseline = model(
            context_mel=context_mel,
            context_dense_timing_v2=context_dense_timing_v2,
            normalized_difficulty=torch.zeros(1),
            context_padding_mask=context_padding_mask,
        )
        changed = model(
            context_mel=changed_mel,
            context_dense_timing_v2=changed_timing,
            normalized_difficulty=torch.zeros(1),
            context_padding_mask=context_padding_mask,
        )

        self.assertTrue(torch.allclose(baseline.value_pred, changed.value_pred, atol=1e-6))
        self.assertTrue(torch.allclose(baseline.confidence_pred, changed.confidence_pred, atol=1e-6))
        self.assertTrue(torch.allclose(baseline.control_memory, changed.control_memory, atol=1e-6))

    def test_rejects_wrong_context_length(self) -> None:
        model = ControlEncoder(ControlEncoderConfig(d_model=16, heads=4, layers=1, ffn_dim=32))

        with self.assertRaisesRegex(ValueError, "600"):
            model(
                context_mel=torch.zeros(1, 599, 160),
                context_dense_timing_v2=torch.zeros(1, 599, 4),
                normalized_difficulty=torch.zeros(1),
            )


if __name__ == "__main__":
    unittest.main()
