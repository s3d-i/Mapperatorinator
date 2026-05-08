import tempfile
import unittest
from importlib import import_module
from pathlib import Path

import torch

from train.stage_2.model_mapper_v1 import MapperV1Config, MapperV1Model
from train.stage_2.training.mapper_v1 import (
    MapperV1PhaseBLossConfig,
    _collate_synthetic_mapper_samples,
    _loss_for_raw_batch,
    _synthetic_mapper_samples,
    load_run_config,
)


class MapperV1PhaseBTrainingTests(unittest.TestCase):
    def test_data_and_training_import_without_mapper_package_preimport(self) -> None:
        self.assertIsNotNone(import_module("train.stage_2.data.mapper_v1_windows"))
        self.assertIsNotNone(import_module("train.stage_2.training.mapper_v1"))

    def test_phase_b_config_loads_with_density_disabled_and_skip_scale_zero(self) -> None:
        config = load_run_config("train/stage_2/training/configs/stage2_mapper_v1_phase_b_mps.yaml")

        self.assertEqual(config["loss"]["lambda_density"], 0.0)
        self.assertEqual(config["loss"]["lambda_density_teacher"], 0.0)
        self.assertEqual(config["model"]["skip_scale"], 0.0)
        self.assertEqual(config["model"]["state_prior_adapter_scale"], 0.03)
        self.assertIn("stage2_control_demo_global", config["init_from_control_checkpoint"])

    def test_phase_b_loss_config_rejects_density_enablement(self) -> None:
        with self.assertRaisesRegex(ValueError, "density loss is disabled"):
            MapperV1PhaseBLossConfig(lambda_density=0.01)
        with self.assertRaisesRegex(ValueError, "density loss is disabled"):
            MapperV1PhaseBLossConfig(lambda_density_teacher=0.01)

    def test_synthetic_loss_path_reports_density_zero(self) -> None:
        torch.manual_seed(23)
        model_config = MapperV1Config(
            control_dim=16,
            d_model=16,
            heads=4,
            layers=1,
            ffn_dim=32,
            dropout=0.0,
            max_seq_len=16,
            state_hidden_dim=16,
            ln_close_hidden_dim=16,
        )
        model = MapperV1Model(model_config)
        batch = _collate_synthetic_mapper_samples(_synthetic_mapper_samples(model_config=model_config)[:2])

        loss_output = _loss_for_raw_batch(
            model,
            batch,
            device=torch.device("cpu"),
            loss_config=MapperV1PhaseBLossConfig(),
        )

        self.assertTrue(torch.isfinite(loss_output.total_loss))
        self.assertEqual(loss_output.metrics["loss/density"], 0.0)
        self.assertGreater(loss_output.metrics["target/token_count"], 0.0)

    def test_masked_pad_column_with_invalid_state_does_not_affect_loss(self) -> None:
        torch.manual_seed(29)
        model_config = MapperV1Config(
            control_dim=16,
            d_model=16,
            heads=4,
            layers=1,
            ffn_dim=32,
            dropout=0.0,
            max_seq_len=16,
            state_hidden_dim=16,
            ln_close_hidden_dim=16,
        )
        model = MapperV1Model(model_config)
        model.eval()
        batch = _collate_synthetic_mapper_samples(_synthetic_mapper_samples(model_config=model_config)[:2])
        batch["target_token_mask"] = torch.ones_like(batch["target_tokens"], dtype=torch.bool)
        base = _loss_for_raw_batch(
            model,
            batch,
            device=torch.device("cpu"),
            loss_config=MapperV1PhaseBLossConfig(),
        )
        padded = dict(batch)
        padded["target_tokens"] = torch.cat(
            [batch["target_tokens"], torch.zeros((2, 1), dtype=torch.long)],
            dim=1,
        )
        padded["target_token_mask"] = torch.cat(
            [batch["target_token_mask"], torch.zeros((2, 1), dtype=torch.bool)],
            dim=1,
        )
        padded["teacher_current_ms"] = torch.cat(
            [batch["teacher_current_ms"], torch.full((2, 1), 99_999, dtype=torch.long)],
            dim=1,
        )
        padded["teacher_open_mask"] = torch.cat(
            [batch["teacher_open_mask"], torch.ones((2, 1, 4), dtype=torch.bool)],
            dim=1,
        )
        padded["teacher_open_age_ms"] = torch.cat(
            [batch["teacher_open_age_ms"], torch.full((2, 1, 4), 99_999, dtype=torch.long)],
            dim=1,
        )
        padded["close_labels"] = torch.cat(
            [batch["close_labels"], torch.ones((2, 1, 4), dtype=torch.bool)],
            dim=1,
        )
        padded["close_label_mask"] = torch.cat(
            [batch["close_label_mask"], torch.ones((2, 1, 4), dtype=torch.bool)],
            dim=1,
        )

        masked = _loss_for_raw_batch(
            model,
            padded,
            device=torch.device("cpu"),
            loss_config=MapperV1PhaseBLossConfig(),
        )

        self.assertTrue(torch.allclose(masked.total_loss, base.total_loss, atol=1e-6, rtol=1e-6))

    def test_phase_b_rejects_control_memory_padding_mask(self) -> None:
        model_config = MapperV1Config(
            control_dim=16,
            d_model=16,
            heads=4,
            layers=1,
            ffn_dim=32,
            dropout=0.0,
            max_seq_len=16,
            state_hidden_dim=16,
            ln_close_hidden_dim=16,
        )
        model = MapperV1Model(model_config)
        batch = _collate_synthetic_mapper_samples(_synthetic_mapper_samples(model_config=model_config)[:1])
        batch["control_memory_padding_mask_8s"] = torch.zeros((1, 400), dtype=torch.bool)

        with self.assertRaisesRegex(ValueError, "control_memory_padding_mask_8s is not supported"):
            _loss_for_raw_batch(
                model,
                batch,
                device=torch.device("cpu"),
                loss_config=MapperV1PhaseBLossConfig(),
            )

    def test_unknown_config_keys_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "bad.yaml"
            path.write_text("model:\n  does_not_exist: 1\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "unknown model config keys"):
                load_run_config(path)


if __name__ == "__main__":
    unittest.main()
