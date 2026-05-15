import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from train.stage_2.model_control_demo_global import ControlDemoGlobalEncoderConfig
from train.stage_2.model_mapper_v2_1 import MapperV21Config, MapperV21LossConfig
from train.stage_2.training import mapper_v2_1 as mapper_v2_1_training
from train.stage_2.training.mapper_v2_1 import load_run_config


class MapperV21PhaseBTrainingTests(unittest.TestCase):
    def test_phase_b_sparse_global_config_loads_v2_1_fields(self) -> None:
        config = load_run_config(
            "train/stage_2/training/configs/stage2_mapper_v2_1_phase_b_sparse_global_mps.yaml",
        )

        self.assertTrue(config["include_full_song_context"])
        self.assertTrue(config["skip_first_eval_pass"])
        self.assertEqual(config["mps_cleanup_every"], 20)
        self.assertEqual(config["batch_size"], 2)
        self.assertEqual(config["model"]["max_seq_len"], 1024)
        self.assertTrue(config["model"]["use_global_context"])
        self.assertEqual(config["loss"]["lambda_density"], 0.05)
        self.assertIn("plus_end.parquet", config["index_path"])
        self.assertIn("stage2_mapper_v2_1/window_records", config["mapper_record_cache_path"])
        self.assertNotIn("stage2_mapper_v1/window_records", config["mapper_record_cache_path"])

        MapperV21Config(**config["model"])
        ControlDemoGlobalEncoderConfig(**config["control_model"])
        MapperV21LossConfig(**config["loss"])

    def test_main_forwards_v2_1_training_options(self) -> None:
        train_result = SimpleNamespace(
            report_path=Path("report.json"),
            checkpoint_path=Path("checkpoint.pt"),
            final_loss=0.0,
            completed_steps=0,
        )
        with patch.object(
            mapper_v2_1_training,
            "run_mapper_v2_1_phase_b_training",
            return_value=train_result,
            autospec=True,
        ) as train:
            mapper_v2_1_training.main(
                [
                    "--config",
                    "train/stage_2/training/configs/stage2_mapper_v2_1_phase_b_sparse_global_mps.yaml",
                    "--max-steps",
                    "1",
                ],
            )

        train.assert_called_once()
        kwargs = train.call_args.kwargs
        self.assertTrue(kwargs["include_full_song_context"])
        self.assertTrue(kwargs["skip_first_eval_pass"])
        self.assertEqual(kwargs["mps_cleanup_every"], 20)
        self.assertTrue(kwargs["require_control_teacher_cache"])
        self.assertEqual(kwargs["batch_size"], 2)
        self.assertEqual(kwargs["model_config_overrides"]["max_seq_len"], 1024)
        self.assertEqual(kwargs["loss_config_overrides"]["lambda_density"], 0.05)


if __name__ == "__main__":
    unittest.main()
