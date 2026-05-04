import unittest

import torch

from train.stage_2.features.control_v3_targets import CONFIDENCE_FEATURE_NAMES, VALUE_FEATURE_NAMES
from train.stage_2.model_control.model import BOUNDED_01_VALUE_FEATURES, ControlEncoder, ControlEncoderConfig


class Stage2ControlModelTests(unittest.TestCase):
    def test_control_encoder_outputs_design_shapes_and_derived_compound_confidence(self) -> None:
        torch.manual_seed(0)
        model = ControlEncoder(_small_config())
        model.eval()

        output = model(
            context_mel=torch.zeros(2, 600, 160),
            context_dense_timing_v2=torch.zeros(2, 600, 4),
            normalized_difficulty=torch.tensor([-1.0, 1.0]),
            context_padding_mask=torch.tensor([[False] * 600, [False] * 300 + [True] * 300]),
        )

        self.assertEqual(output.value_pred.shape, (2, 100, len(VALUE_FEATURE_NAMES)))
        self.assertEqual(output.confidence_pred.shape, (2, 100, len(CONFIDENCE_FEATURE_NAMES)))
        self.assertEqual(output.compound_confidence_pred.shape, (2, 100, 1))
        self.assertEqual(output.control_memory.shape, (2, 600, 32))
        self.assertEqual(output.memory_padding_mask.shape, (2, 600))
        control_index = CONFIDENCE_FEATURE_NAMES.index("control_confidence")
        self.assertTrue(torch.equal(output.compound_confidence_pred, output.confidence_pred[..., control_index : control_index + 1]))

    def test_value_ranges_are_feature_name_based(self) -> None:
        torch.manual_seed(1)
        model = ControlEncoder(_small_config())
        density_burst_index = VALUE_FEATURE_NAMES.index("density_burst")
        with torch.no_grad():
            model.value_head.weight[density_burst_index].zero_()
            model.value_head.bias[density_burst_index].fill_(-1.0)
        output = model(
            context_mel=torch.randn(1, 600, 160),
            context_dense_timing_v2=torch.randn(1, 600, 4),
            normalized_difficulty=torch.zeros(1),
        )

        bounded_indexes = [VALUE_FEATURE_NAMES.index(name) for name in BOUNDED_01_VALUE_FEATURES]
        self.assertTrue(torch.all(output.value_pred[..., bounded_indexes] >= 0.0))
        self.assertTrue(torch.all(output.value_pred[..., bounded_indexes] <= 1.0))
        self.assertFalse(torch.all(output.value_pred[..., density_burst_index] >= 0.0))
        self.assertTrue(torch.all(output.confidence_pred >= 0.0))
        self.assertTrue(torch.all(output.confidence_pred <= 1.0))

    def test_film_final_projections_start_at_identity(self) -> None:
        model = ControlEncoder(_small_config())

        film_modules = [model.stem_film, *model.block_films]
        for film in film_modules:
            final = film.net[-1]
            self.assertTrue(torch.equal(final.weight, torch.zeros_like(final.weight)))
            self.assertTrue(torch.equal(final.bias, torch.zeros_like(final.bias)))

    def test_masked_context_values_do_not_change_predictions(self) -> None:
        torch.manual_seed(2)
        model = ControlEncoder(_small_config())
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

    def test_parameter_budget_is_small_for_smoke_config_and_larger_for_default(self) -> None:
        small = ControlEncoder(_small_config())
        default = ControlEncoder()

        self.assertLess(small.parameter_count(), 250_000)
        self.assertGreater(default.parameter_count(), small.parameter_count())


def _small_config() -> ControlEncoderConfig:
    return ControlEncoderConfig(d_model=32, heads=4, layers=1, ffn_dim=64, dropout=0.0, conv_blocks=1)


if __name__ == "__main__":
    unittest.main()
