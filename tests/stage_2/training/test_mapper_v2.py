import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from train.stage_2.model_mapper_v2 import MapperV2Config, MapperV2Model
from train.stage_2.training import mapper_v2 as mapper_v2_training
from train.stage_2.training.mapper_v2 import (
    initialize_mapper_v2_from_mapper_checkpoint,
    load_run_config,
    run_synthetic_smoke,
)


class MapperV2PhaseBTrainingTests(unittest.TestCase):
    def test_phase_b_global_config_loads_v2_fields(self) -> None:
        config = load_run_config("train/stage_2/training/configs/stage2_mapper_v2_phase_b_global_mps.yaml")

        self.assertTrue(config["include_full_song_context"])
        self.assertTrue(config["skip_first_eval_pass"])
        self.assertTrue(config["model"]["use_global_context"])
        self.assertEqual(config["model"]["global_stride"], 16)
        self.assertEqual(config["model"]["global_layers"], 1)
        self.assertEqual(config["batch_size"], 2)
        MapperV2Config(**config["model"])

    def test_phase_b_large_global_config_loads_v2_fields(self) -> None:
        config = load_run_config("train/stage_2/training/configs/stage2_mapper_v2_phase_b_global_d768_l8_mps.yaml")

        self.assertTrue(config["include_full_song_context"])
        self.assertTrue(config["skip_first_eval_pass"])
        self.assertEqual(config["model"]["d_model"], 768)
        self.assertEqual(config["model"]["heads"], 12)
        self.assertEqual(config["model"]["layers"], 8)
        self.assertEqual(config["model"]["ffn_dim"], 3072)
        self.assertEqual(config["mps_cleanup_every"], 20)
        self.assertTrue(config["resume_from"].endswith("checkpoint.pt"))
        MapperV2Config(**config["model"])

    def test_main_forwards_v2_training_options(self) -> None:
        train_result = SimpleNamespace(
            report_path=Path("report.json"),
            checkpoint_path=Path("checkpoint.pt"),
            final_loss=0.0,
            completed_steps=0,
        )
        with patch.object(
            mapper_v2_training,
            "run_mapper_v2_phase_b_training",
            return_value=train_result,
            autospec=True,
        ) as train:
            mapper_v2_training.main(
                [
                    "--config",
                    "train/stage_2/training/configs/stage2_mapper_v2_phase_b_global_mps.yaml",
                    "--max-steps",
                    "1",
                ]
            )

        train.assert_called_once()
        kwargs = train.call_args.kwargs
        self.assertTrue(kwargs["include_full_song_context"])
        self.assertTrue(kwargs["skip_first_eval_pass"])
        self.assertEqual(kwargs["batch_size"], 2)
        self.assertTrue(kwargs["model_config_overrides"]["use_global_context"])
        self.assertEqual(kwargs["model_config_overrides"]["global_gate_init"], -2.94)

    def test_synthetic_smoke_can_skip_initial_eval_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_synthetic_smoke(
                output_dir=Path(temp_dir),
                max_steps=2,
                eval_every=100,
                save_every=2,
                batch_size=1,
                learning_rate=1e-3,
                device_name="cpu",
                final_train_eval_size=1,
                skip_first_eval_pass=True,
            )

            report = json.loads(result.report_path.read_text(encoding="utf-8"))

        self.assertEqual([entry["step"] for entry in report["history"]], [2])
        self.assertTrue(report["skip_first_eval_pass"])

    def test_cache_only_cli_runs_shared_control_teacher_precompute(self) -> None:
        precompute_result = SimpleNamespace(
            reports=[
                {
                    "split": "source",
                    "total_entries": 1,
                    "computed_entries": 1,
                    "skipped_entries": 0,
                    "elapsed_s": 0.0,
                }
            ]
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(
                mapper_v2_training,
                "precompute_mapper_v1_phase_b_control_teacher_cache",
                return_value=precompute_result,
                autospec=True,
            ) as precompute:
                with patch.object(mapper_v2_training, "run_mapper_v2_phase_b_training", autospec=True) as train:
                    mapper_v2_training.main(
                        [
                            "--precompute-control-teacher-cache-only",
                            "--control-teacher-cache-dir",
                            str(Path(temp_dir) / "cache"),
                        ]
                    )

        precompute.assert_called_once()
        train.assert_not_called()
        self.assertIn("control_model_config_overrides", precompute.call_args.kwargs)

    def test_synthetic_smoke_runs_v2_global_training_loop(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_synthetic_smoke(
                output_dir=Path(temp_dir),
                max_steps=1,
                eval_every=1,
                save_every=1,
                batch_size=1,
                learning_rate=1e-3,
                device_name="cpu",
                final_train_eval_size=1,
            )

            self.assertEqual(result.completed_steps, 1)
            self.assertTrue(result.report_path.exists())
            self.assertTrue(result.checkpoint_path.exists())
            self.assertTrue(torch.isfinite(torch.tensor(result.final_loss)))

    def test_initialize_mapper_v2_checkpoint_loads_state_only(self) -> None:
        config = MapperV2Config(
            control_dim=16,
            d_model=16,
            heads=4,
            layers=1,
            ffn_dim=32,
            dropout=0.0,
            max_seq_len=16,
            state_hidden_dim=16,
            ln_close_hidden_dim=16,
            global_stride=16,
            global_layers=1,
            global_ffn_dim=32,
            global_conv_blocks=0,
        )
        source = MapperV2Model(config)
        with torch.no_grad():
            for index, parameter in enumerate(source.parameters()):
                parameter.fill_(0.01 * (index + 1))
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "mapper_v2.pt"
            torch.save(
                {
                    "checkpoint_schema_version": mapper_v2_training.CHECKPOINT_SCHEMA_VERSION,
                    "model_state_dict": source.state_dict(),
                    "optimizer_state_dict": {"state": {"ignored": torch.ones(1)}},
                    "model_config": config.__dict__,
                    "control_model_config": None,
                    "training_state": {"step": 17},
                },
                checkpoint_path,
            )
            target = MapperV2Model(config)

            report = initialize_mapper_v2_from_mapper_checkpoint(
                target,
                checkpoint_path,
                expected_model_config=config,
                expected_control_model_config=None,
            )

        self.assertEqual(report["kind"], "mapper_v2_model_state")
        self.assertEqual(report["checkpoint_step"], 17)
        self.assertFalse(report["optimizer_state_loaded"])
        for key, value in source.state_dict().items():
            self.assertTrue(torch.equal(target.state_dict()[key], value), key)


if __name__ == "__main__":
    unittest.main()
