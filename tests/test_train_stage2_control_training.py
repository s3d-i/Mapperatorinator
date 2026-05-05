import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import torch

from train.stage_2.data.control_windows import DEFAULT_MAX_CACHED_MAPS
from train.stage_2.features.control_v3_targets import CONFIDENCE_FEATURE_NAMES, MODEL_FEATURE_NAMES, VALUE_FEATURE_NAMES
from train.stage_2.model_control import ControlModelLoss
from train.stage_2.model_control.model import ControlEncoderOutput
from train.stage_2.training.control import (
    ControlTrainingResult,
    _resume_training_config,
    load_run_config,
    main,
    metrics_for_loader,
    run_control_training,
    run_synthetic_smoke,
    split_train_eval_dataset,
)


class Stage2ControlTrainingTests(unittest.TestCase):
    def test_load_run_config_rejects_unknown_keys_and_normalizes_sections(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "control.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "output-dir: out",
                        "max_steps: 3",
                        "log-every: 10",
                        "model:",
                        "  d-model: 32",
                        "  conv-blocks: 1",
                        "loss:",
                        "  confidence-loss-weight: 0.5",
                    ]
                ),
                encoding="utf-8",
            )

            config = load_run_config(config_path)
            self.assertEqual(config["output_dir"], "out")
            self.assertEqual(config["log_every"], 10)
            self.assertEqual(config["model"]["d_model"], 32)
            self.assertEqual(config["model"]["conv_blocks"], 1)
            self.assertEqual(config["loss"]["confidence_loss_weight"], 0.5)

            config_path.write_text("unknown_key: true\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unknown run config"):
                load_run_config(config_path)

            config_path.write_text("model:\n  unknown: 1\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unknown model config"):
                load_run_config(config_path)

    def test_synthetic_smoke_writes_report_checkpoint_and_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                result = run_synthetic_smoke(
                    output_dir=Path(tmpdir),
                    max_steps=2,
                    save_every=1,
                    device_name="cpu",
                    seed=7,
                )

            self.assertEqual(result.completed_steps, 2)
            self.assertTrue(result.report_path.is_file())
            self.assertTrue(result.checkpoint_path.is_file())
            self.assertIn("train_progress step=1/2", stdout.getvalue())
            report = json.loads(result.report_path.read_text(encoding="utf-8"))
            checkpoint = torch.load(result.checkpoint_path, map_location="cpu", weights_only=True)
            self.assertEqual(report["completed_steps"], 2)
            self.assertTrue(report["is_complete"])
            self.assertEqual(report["dataset"]["status"], "synthetic_smoke")
            self.assertIn("loss/total", report["final_eval_metrics"])
            self.assertEqual(checkpoint["checkpoint_schema_version"], 1)
            self.assertEqual(checkpoint["training_state"]["step"], 2)
            self.assertIn("optimizer_state_dict", checkpoint)
            self.assertIn("rng_state", checkpoint["training_state"])

    def test_synthetic_smoke_logs_at_requested_step_interval(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                run_synthetic_smoke(
                    output_dir=Path(tmpdir),
                    max_steps=4,
                    eval_every=4,
                    save_every=4,
                    log_every=2,
                    device_name="cpu",
                    seed=7,
                )

            output = stdout.getvalue()
            self.assertIn("train_progress step=1/4", output)
            self.assertIn("train_progress step=2/4", output)
            self.assertNotIn("train_progress step=3/4", output)
            self.assertIn("train_progress step=4/4", output)
            self.assertIn("steps_per_s=", output)

    def test_metrics_for_loader_aggregates_with_loss_specific_denominators(self) -> None:
        model = _MarkerControlModel()
        loss_fn = ControlModelLoss()
        split_metrics = metrics_for_loader(
            model,
            loss_fn,
            [
                _control_eval_batch(markers=[0.0], frame_counts=[600]),
                _control_eval_batch(markers=[1.0], frame_counts=[300]),
            ],
            device=torch.device("cpu"),
        )
        combined_metrics = metrics_for_loader(
            model,
            loss_fn,
            [_control_eval_batch(markers=[0.0, 1.0], frame_counts=[600, 300])],
            device=torch.device("cpu"),
        )

        self.assertAlmostEqual(split_metrics["loss/value"], combined_metrics["loss/value"], places=6)
        self.assertAlmostEqual(split_metrics["loss/confidence"], combined_metrics["loss/confidence"], places=6)
        self.assertAlmostEqual(split_metrics["loss/total"], combined_metrics["loss/total"], places=6)
        self.assertAlmostEqual(split_metrics["target/valid_frame_rate"], 0.75, places=6)
        self.assertLess(split_metrics["loss/value"], 0.20)

    def test_synthetic_smoke_resume_continues_from_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            with redirect_stdout(io.StringIO()):
                first = run_synthetic_smoke(
                    output_dir=output_dir,
                    max_steps=1,
                    eval_every=1,
                    save_every=1,
                    device_name="cpu",
                    seed=11,
                )

            stdout = io.StringIO()
            original_torch_load = torch.load
            with patch("train.stage_2.training.control.torch.load", wraps=original_torch_load) as torch_load:
                with redirect_stdout(stdout):
                    resumed = run_synthetic_smoke(
                        output_dir=output_dir,
                        max_steps=3,
                        eval_every=1,
                        save_every=1,
                        device_name="cpu",
                        seed=11,
                        resume_from=first.checkpoint_path,
                    )

            self.assertTrue(torch_load.call_args_list)
            self.assertTrue(all(call.kwargs.get("weights_only") is True for call in torch_load.call_args_list))
            self.assertIn("resume_progress", stdout.getvalue())
            self.assertNotIn("train_progress step=1/3", stdout.getvalue())
            self.assertIn("train_progress step=2/3", stdout.getvalue())
            checkpoint = original_torch_load(resumed.checkpoint_path, map_location="cpu", weights_only=True)
            self.assertEqual(checkpoint["training_state"]["step"], 3)
            self.assertEqual(resumed.completed_steps, 3)

    def test_synthetic_smoke_resume_restores_weights_only_safe_rng_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            with redirect_stdout(io.StringIO()):
                first = run_synthetic_smoke(
                    output_dir=output_dir,
                    max_steps=1,
                    eval_every=1,
                    save_every=1,
                    device_name="cpu",
                    seed=19,
                )

            checkpoint = torch.load(first.checkpoint_path, map_location="cpu", weights_only=True)
            python_rng_state = checkpoint["training_state"]["rng_state"]["python_random"]
            numpy_rng_state = checkpoint["training_state"]["rng_state"]["numpy_random"]
            self.assertIsInstance(python_rng_state, dict)
            self.assertIsInstance(python_rng_state["state"], list)
            self.assertIsInstance(numpy_rng_state, dict)
            self.assertIsInstance(numpy_rng_state["state"], list)

            with redirect_stdout(io.StringIO()):
                resumed = run_synthetic_smoke(
                    output_dir=output_dir,
                    max_steps=2,
                    eval_every=1,
                    save_every=1,
                    device_name="cpu",
                    seed=19,
                    resume_from=first.checkpoint_path,
                )

            self.assertEqual(resumed.completed_steps, 2)

    def test_resume_ignores_runtime_dataset_metadata_from_checkpoint_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            with redirect_stdout(io.StringIO()):
                first = run_synthetic_smoke(
                    output_dir=output_dir,
                    max_steps=1,
                    eval_every=1,
                    save_every=1,
                    device_name="cpu",
                    seed=23,
                )

            checkpoint = torch.load(first.checkpoint_path, map_location="cpu", weights_only=True)
            checkpoint["training_config"]["dataset"]["max_cached_maps"] = 16
            checkpoint["training_config"]["dataset"]["num_workers"] = 2
            torch.save(checkpoint, first.checkpoint_path)

            with redirect_stdout(io.StringIO()):
                resumed = run_synthetic_smoke(
                    output_dir=output_dir,
                    max_steps=2,
                    eval_every=1,
                    save_every=1,
                    device_name="cpu",
                    seed=23,
                    resume_from=first.checkpoint_path,
                )

            self.assertEqual(resumed.completed_steps, 2)

    def test_resume_rejects_training_config_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            with redirect_stdout(io.StringIO()):
                first = run_synthetic_smoke(
                    output_dir=output_dir,
                    max_steps=1,
                    save_every=1,
                    learning_rate=1e-2,
                    device_name="cpu",
                    seed=11,
                )

            with self.assertRaisesRegex(ValueError, "training_config"):
                with redirect_stdout(io.StringIO()):
                    run_synthetic_smoke(
                        output_dir=output_dir,
                        max_steps=2,
                        save_every=1,
                        learning_rate=1e-4,
                        device_name="cpu",
                        seed=11,
                        resume_from=first.checkpoint_path,
                    )

    def test_resume_rejects_checkpoint_that_requires_unsafe_unpickling(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            resume_path = Path(tmpdir) / "unsafe.pt"
            torch.save({"payload": _UnsafeTorchLoadPayload()}, resume_path)

            original_torch_load = torch.load
            with patch("train.stage_2.training.control.torch.load", wraps=original_torch_load) as torch_load:
                with self.assertRaisesRegex(ValueError, "weights_only=True"):
                    with redirect_stdout(io.StringIO()):
                        run_synthetic_smoke(
                            output_dir=Path(tmpdir) / "out",
                            max_steps=1,
                            device_name="cpu",
                            resume_from=resume_path,
                        )

            self.assertTrue(torch_load.call_args_list)
            self.assertTrue(all(call.kwargs.get("weights_only") is True for call in torch_load.call_args_list))

    def test_rejects_non_finite_training_numbers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(ValueError, "learning_rate"):
                run_synthetic_smoke(
                    output_dir=Path(tmpdir),
                    max_steps=1,
                    learning_rate=float("nan"),
                    device_name="cpu",
                )

    def test_split_train_eval_dataset_is_deterministic_and_keeps_training_nonempty(self) -> None:
        dataset = list(range(10))

        train_a, eval_a = split_train_eval_dataset(dataset, eval_fraction=0.2, eval_size=None, seed=123)
        train_b, eval_b = split_train_eval_dataset(dataset, eval_fraction=0.2, eval_size=None, seed=123)

        self.assertEqual(train_a.indices, train_b.indices)
        self.assertEqual(eval_a.indices, eval_b.indices)
        self.assertEqual(len(train_a), 8)
        self.assertEqual(len(eval_a), 2)

        train_one, eval_one = split_train_eval_dataset([0], eval_fraction=0.5, eval_size=None, seed=1)
        self.assertEqual(len(train_one), 1)
        self.assertEqual(len(eval_one), 0)

    def test_split_train_eval_dataset_keeps_windows_from_same_map_together(self) -> None:
        dataset = _MetadataOnlyDataset(
            [
                {"beatmap_path": "maps/a.osu", "target_start_frame": 0},
                {"beatmap_path": "maps/a.osu", "target_start_frame": 100},
                {"beatmap_path": "maps/a.osu", "target_start_frame": 200},
                {"beatmap_path": "maps/b.osu", "target_start_frame": 0},
                {"beatmap_path": "maps/b.osu", "target_start_frame": 100},
                {"beatmap_path": "maps/b.osu", "target_start_frame": 200},
            ]
        )

        train_a, eval_a = split_train_eval_dataset(dataset, eval_fraction=0.0, eval_size=2, seed=0)
        train_b, eval_b = split_train_eval_dataset(dataset, eval_fraction=0.0, eval_size=2, seed=0)

        self.assertEqual(train_a.indices, train_b.indices)
        self.assertEqual(eval_a.indices, eval_b.indices)
        self.assertGreater(len(train_a), 0)
        self.assertGreater(len(eval_a), 0)
        train_maps = _map_paths_for_indices(dataset, train_a.indices)
        eval_maps = _map_paths_for_indices(dataset, eval_a.indices)
        self.assertTrue(train_maps.isdisjoint(eval_maps))
        for map_path in eval_maps:
            map_indices = {index for index, record in enumerate(dataset.records) if record["beatmap_path"] == map_path}
            self.assertTrue(map_indices.issubset(set(eval_a.indices)))

    def test_split_train_eval_dataset_groups_by_beatmap_path_before_row_ids(self) -> None:
        dataset = _MetadataOnlyDataset(
            [
                {"beatmap_path": "maps/a.osu", "filtered_index": 10, "source_index": 100},
                {"beatmap_path": "maps/a.osu", "filtered_index": 11, "source_index": 101},
                {"beatmap_path": "maps/a.osu", "filtered_index": 12, "source_index": 102},
                {"beatmap_path": "maps/b.osu", "filtered_index": 20, "source_index": 200},
                {"beatmap_path": "maps/b.osu", "filtered_index": 21, "source_index": 201},
                {"beatmap_path": "maps/b.osu", "filtered_index": 22, "source_index": 202},
            ]
        )

        train_split, eval_split = split_train_eval_dataset(dataset, eval_fraction=0.0, eval_size=2, seed=0)

        train_maps = _map_paths_for_indices(dataset, train_split.indices)
        eval_maps = _map_paths_for_indices(dataset, eval_split.indices)
        self.assertTrue(train_maps.isdisjoint(eval_maps))
        for map_path in eval_maps:
            map_indices = {index for index, record in enumerate(dataset.records) if record["beatmap_path"] == map_path}
            self.assertTrue(map_indices.issubset(set(eval_split.indices)))

    def test_main_loads_yaml_config_and_cli_overrides_it_for_synthetic_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "run.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "output_dir: configured-out",
                        "max_steps: 4",
                        "eval_every: 2",
                        "log_every: 3",
                        "batch_size: 4",
                        "learning_rate: 0.01",
                        "seed: 2026",
                        "device: cpu",
                        "resume_from: checkpoint.pt",
                        "final-train-eval-size: 5",
                        "synthetic_smoke: true",
                        "model:",
                        "  d_model: 24",
                        "  heads: 4",
                        "loss:",
                        "  sparse_boost: 2.0",
                    ]
                ),
                encoding="utf-8",
            )
            with patch(
                "train.stage_2.training.control.run_synthetic_smoke",
                return_value=ControlTrainingResult(
                    report_path=Path("report.json"),
                    checkpoint_path=Path("checkpoint.pt"),
                    final_loss=0.0,
                    final_value_loss=0.0,
                    final_confidence_loss=0.0,
                    completed_steps=1,
                ),
            ) as run_smoke:
                with redirect_stdout(io.StringIO()):
                    main(
                        [
                            "--config",
                            str(config_path),
                            "--max-steps",
                            "6",
                            "--log-every",
                            "5",
                            "--d-model",
                            "32",
                            "--final-train-eval-size",
                            "3",
                        ]
                    )

        kwargs = run_smoke.call_args.kwargs
        self.assertEqual(kwargs["output_dir"], Path("configured-out"))
        self.assertEqual(kwargs["max_steps"], 6)
        self.assertEqual(kwargs["eval_every"], 2)
        self.assertEqual(kwargs["log_every"], 5)
        self.assertEqual(kwargs["batch_size"], 4)
        self.assertEqual(kwargs["learning_rate"], 0.01)
        self.assertEqual(kwargs["seed"], 2026)
        self.assertEqual(kwargs["device_name"], "cpu")
        self.assertEqual(kwargs["resume_from"], Path("checkpoint.pt"))
        self.assertEqual(kwargs["final_train_eval_size"], 3)
        self.assertEqual(kwargs["model_config_overrides"]["d_model"], 32)
        self.assertEqual(kwargs["model_config_overrides"]["heads"], 4)
        self.assertEqual(kwargs["loss_config_overrides"]["sparse_boost"], 2.0)

    def test_main_passes_max_cached_maps_for_real_training(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "run.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "output_dir: configured-out",
                        "max_cached_maps: 16",
                        "model:",
                        "  d_model: 24",
                    ]
                ),
                encoding="utf-8",
            )
            with patch(
                "train.stage_2.training.control.run_control_training",
                return_value=ControlTrainingResult(
                    report_path=Path("report.json"),
                    checkpoint_path=Path("checkpoint.pt"),
                    final_loss=0.0,
                    final_value_loss=0.0,
                    final_confidence_loss=0.0,
                    completed_steps=1,
                ),
            ) as run_training:
                with redirect_stdout(io.StringIO()):
                    main(
                        [
                            "--config",
                            str(config_path),
                            "--max-cached-maps",
                            "5",
                        ]
                    )

        kwargs = run_training.call_args.kwargs
        self.assertEqual(kwargs["max_cached_maps"], 5)
        self.assertEqual(kwargs["model_config_overrides"]["d_model"], 24)

    def test_run_control_training_limits_final_train_eval_loader(self) -> None:
        dataset = _MetadataOnlyDataset(
            [{"beatmap_path": f"maps/{index}.osu"} for index in range(10)]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("train.stage_2.training.control.ControlWindowDataset", return_value=dataset) as dataset_cls:
                with patch(
                    "train.stage_2.training.control._run_training",
                    return_value=ControlTrainingResult(
                        report_path=Path("report.json"),
                        checkpoint_path=Path("checkpoint.pt"),
                        final_loss=0.0,
                        final_value_loss=0.0,
                        final_confidence_loss=0.0,
                        completed_steps=1,
                    ),
                ) as run_training:
                    run_control_training(
                        output_dir=Path(tmpdir),
                        max_steps=1,
                        batch_size=2,
                        device_name="cpu",
                        eval_fraction=0.0,
                        eval_size=0,
                        final_train_eval_size=3,
                        max_cached_maps=5,
                    )

        kwargs = run_training.call_args.kwargs
        self.assertEqual(dataset_cls.call_args.kwargs["max_cached_maps"], 5)
        self.assertEqual(len(kwargs["train_eval_loader"].dataset), 3)
        self.assertEqual(kwargs["dataset_report"]["train_window_count"], 10)
        self.assertEqual(kwargs["dataset_report"]["final_train_eval_size"], 3)
        self.assertEqual(kwargs["dataset_report"]["final_train_eval_window_count"], 3)
        self.assertEqual(kwargs["dataset_report"]["max_cached_maps"], 5)

    def test_run_control_training_reports_effective_default_cache_limit(self) -> None:
        dataset = _MetadataOnlyDataset(
            [{"beatmap_path": f"maps/{index}.osu"} for index in range(10)]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("train.stage_2.training.control.ControlWindowDataset", return_value=dataset) as dataset_cls:
                with patch(
                    "train.stage_2.training.control._run_training",
                    return_value=ControlTrainingResult(
                        report_path=Path("report.json"),
                        checkpoint_path=Path("checkpoint.pt"),
                        final_loss=0.0,
                        final_value_loss=0.0,
                        final_confidence_loss=0.0,
                        completed_steps=1,
                    ),
                ) as run_training:
                    run_control_training(
                        output_dir=Path(tmpdir),
                        max_steps=1,
                        batch_size=2,
                        device_name="cpu",
                        eval_fraction=0.0,
                        eval_size=0,
                        final_train_eval_size=3,
                    )

        kwargs = run_training.call_args.kwargs
        self.assertEqual(dataset_cls.call_args.kwargs["max_cached_maps"], DEFAULT_MAX_CACHED_MAPS)
        self.assertEqual(kwargs["dataset_report"]["max_cached_maps"], DEFAULT_MAX_CACHED_MAPS)

    def test_resume_training_config_excludes_runtime_dataset_knobs(self) -> None:
        config = _resume_training_config(
            seed=7,
            run_name="control",
            batch_size=2,
            learning_rate=1e-3,
            weight_decay=0.01,
            eval_every=10,
            save_every=10,
            dataset_report={
                "train_window_count": 100,
                "eval_window_count": 20,
                "max_cached_maps": 5,
                "num_workers": 2,
            },
        )

        self.assertEqual(
            config["dataset"],
            {
                "train_window_count": 100,
                "eval_window_count": 20,
            },
        )


class _MetadataOnlyDataset:
    def __init__(self, records: list[dict[str, object]]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        raise AssertionError("split_train_eval_dataset should infer map identity from records")


class _UnsafeTorchLoadPayload:
    def __reduce__(self) -> tuple[object, tuple[str]]:
        return (eval, ("1 + 1",))


class _MarkerControlModel(torch.nn.Module):
    def forward(
        self,
        *,
        context_mel: torch.Tensor,
        context_dense_timing_v2: torch.Tensor,
        normalized_difficulty: torch.Tensor,
        context_padding_mask: torch.Tensor,
    ) -> ControlEncoderOutput:
        del context_dense_timing_v2, normalized_difficulty
        batch_size = int(context_mel.shape[0])
        marker = context_mel[:, 0, 0].reshape(batch_size, 1, 1)
        confidence_pred = marker.expand(batch_size, 100, len(CONFIDENCE_FEATURE_NAMES)).contiguous()
        control_index = CONFIDENCE_FEATURE_NAMES.index("control_confidence")
        return ControlEncoderOutput(
            value_pred=torch.zeros(batch_size, 100, len(VALUE_FEATURE_NAMES), dtype=context_mel.dtype, device=context_mel.device),
            confidence_pred=confidence_pred,
            compound_confidence_pred=confidence_pred[..., control_index : control_index + 1],
            control_memory=torch.zeros(batch_size, 600, 8, dtype=context_mel.dtype, device=context_mel.device),
            memory_padding_mask=context_padding_mask,
        )


def _control_eval_batch(*, markers: list[float], frame_counts: list[int]) -> dict[str, torch.Tensor]:
    batch_size = len(markers)
    max_frame_count = max(frame_counts)
    full_mel = torch.zeros(batch_size, max_frame_count, 160, dtype=torch.float32)
    full_dense_timing_v2 = torch.zeros(batch_size, max_frame_count, 4, dtype=torch.float32)
    padding_mask = torch.ones(batch_size, max_frame_count, dtype=torch.bool)
    control_v3_target = torch.zeros(batch_size, 100, len(MODEL_FEATURE_NAMES), dtype=torch.float32)

    for batch_index, (marker, frame_count) in enumerate(zip(markers, frame_counts, strict=True)):
        full_mel[batch_index, 0, 0] = marker
        padding_mask[batch_index, :frame_count] = False
        if marker == 0.0:
            control_v3_target[batch_index, :, MODEL_FEATURE_NAMES.index("hold_occupancy")] = 1.0
        else:
            for name in CONFIDENCE_FEATURE_NAMES:
                control_v3_target[batch_index, :, MODEL_FEATURE_NAMES.index(name)] = 1.0

    return {
        "full_mel": full_mel,
        "full_dense_timing_v2": full_dense_timing_v2,
        "padding_mask": padding_mask,
        "control_v3_target": control_v3_target,
        "ln_change_n_eff_target": torch.full((batch_size, 100), 3.0, dtype=torch.float32),
        "target_valid_mask": torch.ones(batch_size, 100, dtype=torch.bool),
        "normalized_difficulty": torch.zeros(batch_size, dtype=torch.float32),
        "target_start_frame": torch.full((batch_size,), 250, dtype=torch.long),
        "frame_count": torch.tensor(frame_counts, dtype=torch.long),
    }


def _map_paths_for_indices(dataset: _MetadataOnlyDataset, indices: list[int]) -> set[str]:
    return {str(dataset.records[index]["beatmap_path"]) for index in indices}


if __name__ == "__main__":
    unittest.main()
