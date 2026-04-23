import io
import tempfile
import unittest
import json
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from train.stage1_oracle.data.windows import OracleWindowFilterReport, collate_oracle_windows
from train.stage1_oracle.events.canonical import CanonicalTimepoint, LaneAction
from train.stage1_oracle.events.tokens import Stage1Vocab, decompose_ts_delta
from train.stage1_oracle.training.overfit_32 import (
    _DecodeWindowInput,
    _clone_tensor_to_cpu,
    _decode_window_input_to_device,
    _move_rollout_batch_inputs,
    REQUIRED_PRETRAINING_GATE_NAMES,
    PretrainingGateValidationError,
    OverfitRunResult,
    build_balanced_epoch_sampling_plan,
    greedy_decode_metrics_for_loader,
    load_run_config,
    main,
    run_overfit_32,
    run_synthetic_smoke,
    summarize_overfit_coverage,
    training_config_from_pretraining_gates,
    timing_training_stats_from_pretraining_gates,
    validate_pretraining_gate_manifest,
)


@dataclass(frozen=True)
class _CoverageRecord:
    beatmap_path: Path
    difficulty: float


class _SequenceModel:
    def __init__(
        self,
        sequences: dict[int, list[int]],
        *,
        key_source: str,
        vocab_size: int,
        max_decode_len: int = 8,
    ) -> None:
        self.sequences = sequences
        self.key_source = key_source
        self.config = type("_Config", (), {"vocab_size": vocab_size, "max_decode_len": max_decode_len})()

    def eval(self) -> None:
        pass

    def __call__(
        self,
        *,
        packed_audio: torch.Tensor,
        timing_track: torch.Tensor,
        difficulty_bucket: torch.Tensor,
        decoder_input_ids: torch.Tensor,
    ) -> torch.Tensor:
        sequence_key = int(difficulty_bucket[0].item())
        if self.key_source == "open_token":
            sequence_key = int(decoder_input_ids[0, 2].item())
        step = decoder_input_ids.shape[1] - 3
        sequence = self.sequences[sequence_key]
        next_token_id = sequence[min(step, len(sequence) - 1)]
        logits = torch.full(
            (decoder_input_ids.shape[0], decoder_input_ids.shape[1], self.config.vocab_size),
            -1000.0,
            dtype=torch.float32,
            device=decoder_input_ids.device,
        )
        logits[:, -1, next_token_id] = 1000.0
        return logits


class _DatasetStub:
    def __init__(self, records: list[object]) -> None:
        self.records = records
        self.bpm_log_mean = 5.0
        self.bpm_log_std = 0.25
        self.filter_report = OracleWindowFilterReport(
            source_map_count=len(records),
            difficulty_filtered_map_count=0,
            candidate_map_count=len(records),
            missing_red_timing_map_count=0,
            invalid_red_timing_map_count=0,
            negative_time_hitobject_map_count=0,
            unsupported_compound_map_count=0,
            unsupported_compound_event_count=0,
            four_state_unsupported_map_count=0,
            four_state_unsupported_lane_action_count=0,
            zero_length_hold_normalized_count=0,
            retained_map_count=len(records),
            generated_window_count=len(records),
        )

    def __len__(self) -> int:
        return len(self.records)


class Stage1OverfitSmokeTests(unittest.TestCase):
    def test_synthetic_smoke_writes_report_and_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                result = run_synthetic_smoke(
                    output_dir=Path(tmpdir),
                    max_steps=2,
                    seed=1337,
                )

            self.assertTrue(result.report_path.is_file())
            self.assertTrue(result.checkpoint_path.is_file())
            self.assertIn("train_progress step=1/2", stdout.getvalue())
            self.assertIn("train_progress step=2/2", stdout.getvalue())
            self.assertGreaterEqual(result.final_token_accuracy, 0.0)
            self.assertLessEqual(result.final_token_accuracy, 1.0)
            report = json.loads(result.report_path.read_text(encoding="utf-8"))
            checkpoint = torch.load(result.checkpoint_path, map_location="cpu", weights_only=False)
            self.assertEqual(report["timing_track"]["bpm_log_mean"], checkpoint["timing_track"]["bpm_log_mean"])
            self.assertEqual(report["timing_track"]["bpm_log_std"], checkpoint["timing_track"]["bpm_log_std"])
            self.assertEqual(report["timing_track"]["timing_track_version"], "timing_track_20ms_v1")
            self.assertEqual(report["timing_track"]["timing_frame_hop_ms"], 20.0)
            self.assertEqual(report["timing_track"]["timing_frame_center_offset_ms"], 10.0)
            self.assertEqual(report["timing_track"]["pulse_shape"], "triangular")
            self.assertEqual(report["timing_track"]["pulse_width_ms"], 40.0)
            self.assertEqual(report["timing_track"]["red_timing_source"], "reference_osu_red_points_for_oracle_phase")
            self.assertEqual(report["pretraining_gates"]["status"], "SKIPPED_SYNTHETIC_SMOKE")
            self.assertIsNone(report["dataset_filter_report"])
            self.assertIn("loss", report["final_train_teacher_forced"])
            self.assertIn("token_accuracy", report["final_train_teacher_forced"])
            self.assertIn("loss", report["final_val_teacher_forced"])
            self.assertIn("token_accuracy", report["final_val_teacher_forced"])
            self.assertIn("decode_eos_failure_rate", report["final_rollout_probe"])
            self.assertIn("decode_empty_output_rate", report["final_rollout_probe"])
            self.assertIn("decode_eos_forced_after_pending_ts_rate", report["final_rollout_probe"])
            self.assertEqual(report["final_rollout_probe"]["decode_evaluated_window_count"], 2)
            self.assertNotIn("final", report)

    def test_synthetic_smoke_writes_periodic_training_state_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                result = run_synthetic_smoke(
                    output_dir=Path(tmpdir),
                    max_steps=3,
                    save_every=2,
                    seed=1337,
                    device_name="cpu",
                )

            self.assertEqual(result.checkpoint_path, Path(tmpdir) / "checkpoint.pt")
            checkpoints = sorted((Path(tmpdir) / "checkpoints").glob("checkpoint_step_*.pt"))
            self.assertEqual(
                [path.name for path in checkpoints],
                [
                    "checkpoint_step_000001.pt",
                    "checkpoint_step_000002.pt",
                    "checkpoint_step_000003.pt",
                ],
            )
            self.assertIn("checkpoint_progress step=2/3", stdout.getvalue())

            checkpoint = torch.load(result.checkpoint_path, map_location="cpu", weights_only=False)
            self.assertEqual(checkpoint["checkpoint_schema_version"], 1)
            self.assertEqual(checkpoint["training_state"]["step"], 3)
            self.assertEqual(checkpoint["training_state"]["save_every"], 2)
            self.assertIn("optimizer_state_dict", checkpoint)
            self.assertIn("rng_state", checkpoint["training_state"])

            report = json.loads(result.report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["completed_steps"], 3)
            self.assertTrue(report["is_complete"])

    def test_synthetic_smoke_resume_continues_from_saved_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            with redirect_stdout(io.StringIO()):
                first = run_synthetic_smoke(
                    output_dir=output_dir,
                    max_steps=1,
                    save_every=1,
                    seed=1337,
                    device_name="cpu",
                )

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                resumed = run_synthetic_smoke(
                    output_dir=output_dir,
                    max_steps=3,
                    save_every=1,
                    seed=1337,
                    device_name="cpu",
                    resume_from=first.checkpoint_path,
                )

            output = stdout.getvalue()
            self.assertIn("resume_progress", output)
            self.assertNotIn("train_progress step=1/3", output)
            self.assertIn("train_progress step=2/3", output)
            self.assertIn("train_progress step=3/3", output)

            checkpoint = torch.load(resumed.checkpoint_path, map_location="cpu", weights_only=False)
            self.assertEqual(checkpoint["training_state"]["step"], 3)
            self.assertEqual([entry["step"] for entry in checkpoint["history"]], [1, 3])
            self.assertIn("final_train_teacher_forced", checkpoint["training_state"])
            self.assertIn("final_val_teacher_forced", checkpoint["training_state"])
            self.assertIn("final_rollout_probe", checkpoint["training_state"])

    def test_training_progress_prints_before_checkpoint_eval(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            stdout = io.StringIO()

            def metrics_probe(*args: object, **kwargs: object) -> dict[str, float]:
                self.assertIn("train_progress step=1/1", stdout.getvalue())
                self.assertNotIn("eval_progress step=1/1", stdout.getvalue())
                return {"loss": 0.0, "token_accuracy": 1.0}

            with patch(
                "train.stage1_oracle.training.overfit_32.teacher_forced_metrics_for_loader",
                side_effect=metrics_probe,
            ):
                with redirect_stdout(stdout):
                    run_synthetic_smoke(
                        output_dir=Path(tmpdir),
                        max_steps=1,
                        seed=1337,
                    )

    def test_overfit_coverage_reports_unique_maps_and_missing_bins(self) -> None:
        coverage = summarize_overfit_coverage(
            [
                _CoverageRecord(Path("a.osu"), 2.5),
                _CoverageRecord(Path("a.osu"), 2.5),
                _CoverageRecord(Path("b.osu"), 4.5),
            ],
        )

        self.assertEqual(coverage["unique_map_count"], 2)
        self.assertEqual(coverage["map_count_by_bin"], {"2-3": 1, "3-4": 0, "4-5": 1, "5-6": 0})
        self.assertEqual(coverage["window_count_by_bin"], {"2-3": 2, "3-4": 0, "4-5": 1, "5-6": 0})
        self.assertEqual(coverage["missing_bins"], ["3-4", "5-6"])

    def test_overfit_treats_maps_per_bin_as_cap_not_exact_quota(self) -> None:
        records = []
        for difficulty, count in ((2.5, 8), (3.5, 7), (4.5, 8), (5.5, 8)):
            for index in range(count):
                records.append(
                    _window_record(
                        difficulty=difficulty,
                        has_event=True,
                        name=f"{difficulty}-{index}",
                    ),
                )

        pretraining_gates = {
            "training": {
                "max_decode_len": 16,
                "empty_window_cap_ratio": 0.05,
            },
            "gates": {
                "dense_timing_track": {
                    "bpm_log_mean": 5.0,
                    "bpm_log_std": 0.25,
                },
            },
        }
        with patch(
            "train.stage1_oracle.training.overfit_32.validate_pretraining_gate_manifest",
            return_value=pretraining_gates,
        ):
            with patch("train.stage1_oracle.training.overfit_32.OracleWindowDataset", return_value=_DatasetStub(records)):
                with patch(
                    "train.stage1_oracle.training.overfit_32._run_training",
                    return_value=OverfitRunResult(
                        report_path=Path("report.json"),
                        checkpoint_path=Path("checkpoint.pt"),
                        final_loss=0.0,
                        final_token_accuracy=1.0,
                    ),
                ) as run_training:
                    with redirect_stdout(io.StringIO()):
                        run_overfit_32(
                            dataset_root=Path("mania-dataset"),
                            index_path=None,
                            gate_manifest_path=Path("gates.json"),
                            output_dir=Path("out"),
                        )

        call_kwargs = run_training.call_args.kwargs
        self.assertEqual(call_kwargs["overfit_coverage"]["map_count_by_bin"]["3-4"], 7)

    def test_overfit_allows_custom_maps_per_bin_dropout_and_run_name(self) -> None:
        records = []
        for difficulty in (2.5, 3.5, 4.5, 5.5):
            records.append(_window_record(difficulty=difficulty, has_event=True, name=f"{difficulty}"))

        pretraining_gates = {
            "training": {
                "max_decode_len": 16,
                "empty_window_cap_ratio": 0.05,
            },
            "gates": {
                "dense_timing_track": {
                    "bpm_log_mean": 5.0,
                    "bpm_log_std": 0.25,
                },
            },
        }

        def dataset_probe(*args: object, **kwargs: object) -> _DatasetStub:
            self.assertEqual(kwargs["max_maps_per_bin"], 1)
            return _DatasetStub(records)

        with patch(
            "train.stage1_oracle.training.overfit_32.validate_pretraining_gate_manifest",
            return_value=pretraining_gates,
        ):
            with patch("train.stage1_oracle.training.overfit_32.OracleWindowDataset", side_effect=dataset_probe):
                with patch(
                    "train.stage1_oracle.training.overfit_32._run_training",
                    return_value=OverfitRunResult(
                        report_path=Path("report.json"),
                        checkpoint_path=Path("checkpoint.pt"),
                        final_loss=0.0,
                        final_token_accuracy=1.0,
                    ),
                ) as run_training:
                    with redirect_stdout(io.StringIO()):
                        run_overfit_32(
                            dataset_root=Path("mania-dataset"),
                            index_path=None,
                            gate_manifest_path=Path("gates.json"),
                            output_dir=Path("out"),
                            maps_per_bin=1,
                            dropout=0.0,
                            device_name="mps",
                            run_name="overfit_4_dropout0",
                        )

        call_kwargs = run_training.call_args.kwargs
        self.assertEqual(call_kwargs["run_name"], "overfit_4_dropout0")
        self.assertEqual(call_kwargs["device_name"], "mps")
        self.assertEqual(call_kwargs["config"].dropout, 0.0)
        self.assertEqual(call_kwargs["overfit_coverage"]["unique_map_count"], 4)

    def test_overfit_allows_per_bin_map_caps(self) -> None:
        records = []
        for difficulty, count in ((2.5, 3), (3.5, 2), (4.5, 1), (5.5, 1)):
            for index in range(count):
                records.append(_window_record(difficulty=difficulty, has_event=True, name=f"{difficulty}-{index}"))

        pretraining_gates = {
            "training": {
                "max_decode_len": 16,
                "empty_window_cap_ratio": 0.05,
            },
            "gates": {
                "dense_timing_track": {
                    "bpm_log_mean": 5.0,
                    "bpm_log_std": 0.25,
                },
            },
        }
        caps = {
            "2-3": 3,
            "3-4": 2,
            "4-5": 1,
            "5-6": 1,
        }

        def dataset_probe(*args: object, **kwargs: object) -> _DatasetStub:
            self.assertEqual(kwargs["max_maps_per_bin"], caps)
            return _DatasetStub(records)

        with patch(
            "train.stage1_oracle.training.overfit_32.validate_pretraining_gate_manifest",
            return_value=pretraining_gates,
        ):
            with patch("train.stage1_oracle.training.overfit_32.OracleWindowDataset", side_effect=dataset_probe):
                with patch(
                    "train.stage1_oracle.training.overfit_32._run_training",
                    return_value=OverfitRunResult(
                        report_path=Path("report.json"),
                        checkpoint_path=Path("checkpoint.pt"),
                        final_loss=0.0,
                        final_token_accuracy=1.0,
                    ),
                ) as run_training:
                    with redirect_stdout(io.StringIO()):
                        run_overfit_32(
                            dataset_root=Path("mania-dataset"),
                            index_path=None,
                            gate_manifest_path=Path("gates.json"),
                            output_dir=Path("out"),
                            maps_per_bin=caps,
                        )

        self.assertEqual(run_training.call_args.kwargs["overfit_coverage"]["map_count_by_bin"], caps)

    def test_overfit_allows_model_size_overrides(self) -> None:
        records = []
        for difficulty in (2.5, 3.5, 4.5, 5.5):
            records.append(_window_record(difficulty=difficulty, has_event=True, name=f"{difficulty}"))

        pretraining_gates = {
            "training": {
                "max_decode_len": 16,
                "empty_window_cap_ratio": 0.05,
            },
            "gates": {
                "dense_timing_track": {
                    "bpm_log_mean": 5.0,
                    "bpm_log_std": 0.25,
                },
            },
        }

        with patch(
            "train.stage1_oracle.training.overfit_32.validate_pretraining_gate_manifest",
            return_value=pretraining_gates,
        ):
            with patch("train.stage1_oracle.training.overfit_32.OracleWindowDataset", return_value=_DatasetStub(records)):
                with patch(
                    "train.stage1_oracle.training.overfit_32._run_training",
                    return_value=OverfitRunResult(
                        report_path=Path("report.json"),
                        checkpoint_path=Path("checkpoint.pt"),
                        final_loss=0.0,
                        final_token_accuracy=1.0,
                    ),
                ) as run_training:
                    with redirect_stdout(io.StringIO()):
                        run_overfit_32(
                            dataset_root=Path("mania-dataset"),
                            index_path=None,
                            gate_manifest_path=Path("gates.json"),
                            output_dir=Path("out"),
                            maps_per_bin=1,
                            model_config_overrides={
                                "d_model": 320,
                                "heads": 5,
                                "encoder_layers": 5,
                                "decoder_layers": 7,
                                "ffn_dim": 1280,
                            },
                        )

        config = run_training.call_args.kwargs["config"]
        self.assertEqual(config.d_model, 320)
        self.assertEqual(config.heads, 5)
        self.assertEqual(config.encoder_layers, 5)
        self.assertEqual(config.decoder_layers, 7)
        self.assertEqual(config.ffn_dim, 1280)

    def test_main_loads_yaml_config_and_cli_overrides_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "run.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "dataset_root: custom-dataset",
                        "index_path: custom-index.parquet",
                        "gate_manifest: gates.json",
                        "output_dir: configured-out",
                        "maps_per_bin:",
                        "  2-3: 3638",
                        "  3-4: 3376",
                        "  4-5: 2705",
                        "  5-6: 1258",
                        "max_steps: 10000",
                        "eval_every: 500",
                        "batch_size: 4",
                        "learning_rate: 0.0002",
                        "dropout: 0.1",
                        "seed: 2026",
                        "device: mps",
                        "run_name: configured-run",
                        "save_every: 250",
                        "resume_from: configured-checkpoint.pt",
                        "model:",
                        "  d_model: 320",
                        "  heads: 5",
                        "  encoder_layers: 5",
                        "  decoder_layers: 7",
                        "  ffn_dim: 1280",
                    ],
                ),
                encoding="utf-8",
            )

            with patch(
                "train.stage1_oracle.training.overfit_32.run_overfit_32",
                return_value=OverfitRunResult(
                    report_path=Path("report.json"),
                    checkpoint_path=Path("checkpoint.pt"),
                    final_loss=0.0,
                    final_token_accuracy=1.0,
                ),
            ) as run_overfit:
                with redirect_stdout(io.StringIO()):
                    main(["--config", str(config_path), "--max-steps", "12000", "--run-name", "cli-run"])

        call_kwargs = run_overfit.call_args.kwargs
        self.assertEqual(call_kwargs["dataset_root"], Path("custom-dataset"))
        self.assertEqual(call_kwargs["index_path"], Path("custom-index.parquet"))
        self.assertEqual(call_kwargs["gate_manifest_path"], Path("gates.json"))
        self.assertEqual(call_kwargs["output_dir"], Path("configured-out"))
        self.assertEqual(
            call_kwargs["maps_per_bin"],
            {
                "2-3": 3638,
                "3-4": 3376,
                "4-5": 2705,
                "5-6": 1258,
            },
        )
        self.assertEqual(call_kwargs["max_steps"], 12000)
        self.assertEqual(call_kwargs["eval_every"], 500)
        self.assertEqual(call_kwargs["batch_size"], 4)
        self.assertEqual(call_kwargs["learning_rate"], 0.0002)
        self.assertEqual(call_kwargs["dropout"], 0.1)
        self.assertEqual(call_kwargs["seed"], 2026)
        self.assertEqual(call_kwargs["device_name"], "mps")
        self.assertEqual(call_kwargs["run_name"], "cli-run")
        self.assertEqual(call_kwargs["save_every"], 250)
        self.assertEqual(call_kwargs["resume_from"], Path("configured-checkpoint.pt"))
        self.assertEqual(
            call_kwargs["model_config_overrides"],
            {
                "d_model": 320,
                "heads": 5,
                "encoder_layers": 5,
                "decoder_layers": 7,
                "ffn_dim": 1280,
            },
        )

    def test_run_config_accepts_split_and_probe_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "run.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "train_manifest: splits/train.json",
                        "eval_manifest: splits/eval.json",
                        "rollout_probe_manifest: splits/probe.json",
                        "rollout_eval_every: 2500",
                    ],
                ),
                encoding="utf-8",
            )

            config = load_run_config(config_path)

        self.assertEqual(config["train_manifest"], "splits/train.json")
        self.assertEqual(config["eval_manifest"], "splits/eval.json")
        self.assertEqual(config["rollout_probe_manifest"], "splits/probe.json")
        self.assertEqual(config["rollout_eval_every"], 2500)

    def test_main_forwards_split_and_probe_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "run.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "dataset_root: custom-dataset",
                        "gate_manifest: gates.json",
                        "output_dir: out",
                        "train_manifest: splits/train.json",
                        "eval_manifest: splits/eval.json",
                        "rollout_probe_manifest: splits/probe.json",
                        "rollout_eval_every: 2500",
                    ],
                ),
                encoding="utf-8",
            )

            with patch(
                "train.stage1_oracle.training.overfit_32.run_overfit_32",
                return_value=OverfitRunResult(
                    report_path=Path("report.json"),
                    checkpoint_path=Path("checkpoint.pt"),
                    final_loss=0.0,
                    final_token_accuracy=1.0,
                ),
            ) as run_overfit:
                with redirect_stdout(io.StringIO()):
                    main(["--config", str(config_path)])

        call_kwargs = run_overfit.call_args.kwargs
        self.assertEqual(call_kwargs["train_manifest_path"], Path("splits/train.json"))
        self.assertEqual(call_kwargs["eval_manifest_path"], Path("splits/eval.json"))
        self.assertEqual(call_kwargs["rollout_probe_manifest_path"], Path("splits/probe.json"))
        self.assertEqual(call_kwargs["rollout_eval_every"], 2500)

    def test_split_manifests_build_separate_datasets_and_loaders(self) -> None:
        train_records = [_window_record(difficulty=2.5, has_event=True, name="train")]
        eval_records = [_window_record(difficulty=3.5, has_event=True, name="eval")]
        probe_records = [_window_record(difficulty=4.5, has_event=True, name="probe")]
        datasets = {
            Path("splits/train.json"): _DatasetStub(train_records),
            Path("splits/eval.json"): _DatasetStub(eval_records),
            Path("splits/probe.json"): _DatasetStub(probe_records),
        }
        seen_manifest_paths: list[Path | None] = []
        pretraining_gates = {
            "training": {
                "max_decode_len": 16,
                "empty_window_cap_ratio": 0.05,
            },
            "gates": {
                "dense_timing_track": {
                    "bpm_log_mean": 5.0,
                    "bpm_log_std": 0.25,
                },
            },
        }

        def dataset_probe(*args: object, **kwargs: object) -> _DatasetStub:
            manifest_path = kwargs.get("manifest_path")
            seen_manifest_paths.append(manifest_path)
            self.assertIn(manifest_path, datasets)
            if manifest_path == Path("splits/train.json"):
                self.assertEqual(kwargs["max_maps_per_bin"], 1)
            else:
                self.assertIsNone(kwargs["max_maps_per_bin"])
            return datasets[manifest_path]

        with patch(
            "train.stage1_oracle.training.overfit_32.validate_pretraining_gate_manifest",
            return_value=pretraining_gates,
        ):
            with patch("train.stage1_oracle.training.overfit_32.OracleWindowDataset", side_effect=dataset_probe):
                with patch(
                    "train.stage1_oracle.training.overfit_32._run_training",
                    return_value=OverfitRunResult(
                        report_path=Path("report.json"),
                        checkpoint_path=Path("checkpoint.pt"),
                        final_loss=0.0,
                        final_token_accuracy=1.0,
                    ),
                ) as run_training:
                    with redirect_stdout(io.StringIO()):
                        run_overfit_32(
                            dataset_root=Path("mania-dataset"),
                            index_path=None,
                            gate_manifest_path=Path("gates.json"),
                            output_dir=Path("out"),
                            maps_per_bin=1,
                            train_manifest_path=Path("splits/train.json"),
                            eval_manifest_path=Path("splits/eval.json"),
                            rollout_probe_manifest_path=Path("splits/probe.json"),
                            rollout_eval_every=2500,
                        )

        call_kwargs = run_training.call_args.kwargs
        self.assertEqual(seen_manifest_paths, [Path("splits/train.json"), Path("splits/eval.json"), Path("splits/probe.json")])
        self.assertIs(call_kwargs["loader"].dataset, datasets[Path("splits/train.json")])
        self.assertIs(call_kwargs["train_eval_loader"].dataset, datasets[Path("splits/train.json")])
        self.assertIs(call_kwargs["eval_loader"].dataset, datasets[Path("splits/eval.json")])
        self.assertIs(call_kwargs["rollout_probe_loader"].dataset, datasets[Path("splits/probe.json")])
        self.assertEqual(call_kwargs["rollout_eval_every"], 2500)

    def test_overfit_prints_dataset_progress_before_dataset_build(self) -> None:
        records = []
        for difficulty in (2.5, 3.5, 4.5, 5.5):
            for index in range(8):
                records.append(_window_record(difficulty=difficulty, has_event=True, name=f"{difficulty}-{index}"))

        pretraining_gates = {
            "training": {
                "max_decode_len": 16,
                "empty_window_cap_ratio": 0.05,
            },
            "gates": {
                "dense_timing_track": {
                    "bpm_log_mean": 5.0,
                    "bpm_log_std": 0.25,
                },
            },
        }
        stdout = io.StringIO()

        def dataset_probe(*args: object, **kwargs: object) -> _DatasetStub:
            self.assertIn("dataset_progress phase=build_windows status=start", stdout.getvalue())
            self.assertTrue(kwargs["progress"])
            return _DatasetStub(records)

        with patch(
            "train.stage1_oracle.training.overfit_32.validate_pretraining_gate_manifest",
            return_value=pretraining_gates,
        ):
            with patch("train.stage1_oracle.training.overfit_32.OracleWindowDataset", side_effect=dataset_probe):
                with patch(
                    "train.stage1_oracle.training.overfit_32._run_training",
                    return_value=OverfitRunResult(
                        report_path=Path("report.json"),
                        checkpoint_path=Path("checkpoint.pt"),
                        final_loss=0.0,
                        final_token_accuracy=1.0,
                    ),
                ):
                    with redirect_stdout(stdout):
                        run_overfit_32(
                            dataset_root=Path("mania-dataset"),
                            index_path=None,
                            gate_manifest_path=Path("gates.json"),
                            output_dir=Path("out"),
                        )

        self.assertIn("dataset_progress phase=build_windows status=done", stdout.getvalue())

    def test_decode_metrics_allow_holds_that_close_in_a_later_window(self) -> None:
        vocab = Stage1Vocab()
        hold_start = vocab.encode_timepoint_event(
            (LaneAction.HOLD_START, LaneAction.NONE, LaneAction.NONE, LaneAction.NONE),
        )
        hold_end = vocab.encode_timepoint_event(
            (LaneAction.HOLD_END, LaneAction.NONE, LaneAction.NONE, LaneAction.NONE),
        )
        first_target = [vocab.ts_token_id(value) for value in decompose_ts_delta(7000)] + [
            hold_start,
            vocab.eos_id,
        ]
        second_target = [vocab.ts_token_id(100), hold_end, vocab.eos_id]
        first_condition = [vocab.bos_id, vocab.diff_token_id(0), vocab.open_token_id(0)]
        second_condition = [vocab.bos_id, vocab.diff_token_id(1), vocab.open_token_id(0b0001)]
        samples = [
            _sample_for_decode_metrics(
                vocab,
                condition=first_condition,
                target=first_target,
                difficulty_bucket=0,
                open_hold_mask=0,
                write_start_ms=0,
            ),
            _sample_for_decode_metrics(
                vocab,
                condition=second_condition,
                target=second_target,
                difficulty_bucket=1,
                open_hold_mask=0b0001,
                write_start_ms=8000,
            ),
        ]
        batch = collate_oracle_windows(samples, pad_id=vocab.pad_id)
        model = _SequenceModel(
            {
                0: first_target,
                1: second_target,
            },
            key_source="difficulty_bucket",
            vocab_size=vocab.size,
            max_decode_len=16,
        )

        metrics = greedy_decode_metrics_for_loader(model, [batch], vocab=vocab, device=torch.device("cpu"))

        self.assertEqual(metrics["oracle_boundary_invalid_hold_end_rate"], 0.0)
        self.assertEqual(metrics["stitched_boundary_invalid_hold_end_rate"], 0.0)
        self.assertEqual(metrics["stitched_boundary_unclosed_hold_rate"], 0.0)
        self.assertEqual(metrics["active_boundary_exact_match"], 1.0)
        self.assertEqual(metrics["active_boundary_evaluated_boundary_count"], 1)

    def test_decode_metrics_report_stitched_boundary_open_mask_error(self) -> None:
        vocab = Stage1Vocab()
        hold_end = vocab.encode_timepoint_event(
            (LaneAction.HOLD_END, LaneAction.NONE, LaneAction.NONE, LaneAction.NONE),
        )
        first_condition = [vocab.bos_id, vocab.diff_token_id(0), vocab.open_token_id(0)]
        second_condition = [vocab.bos_id, vocab.diff_token_id(1), vocab.open_token_id(0b0001)]
        first_target = [vocab.eos_id]
        second_target = [vocab.ts_token_id(100), hold_end, vocab.eos_id]
        samples = [
            _sample_for_decode_metrics(
                vocab,
                condition=first_condition,
                target=first_target,
                difficulty_bucket=0,
                open_hold_mask=0,
                write_start_ms=0,
            ),
            _sample_for_decode_metrics(
                vocab,
                condition=second_condition,
                target=second_target,
                difficulty_bucket=1,
                open_hold_mask=0b0001,
                write_start_ms=8000,
            ),
        ]
        batch = collate_oracle_windows(samples, pad_id=vocab.pad_id)
        model = _SequenceModel(
            {
                0: [vocab.eos_id],
                1: second_target,
            },
            key_source="difficulty_bucket",
            vocab_size=vocab.size,
        )

        metrics = greedy_decode_metrics_for_loader(model, [batch], vocab=vocab, device=torch.device("cpu"))

        self.assertEqual(metrics["stitched_boundary_evaluated_boundary_count"], 1)
        self.assertEqual(metrics["stitched_boundary_open_mask_error_rate"], 1.0)
        self.assertEqual(metrics["oracle_boundary_invalid_hold_end_rate"], 0.0)
        self.assertEqual(metrics["stitched_boundary_invalid_hold_end_rate"], 0.0)
        self.assertNotIn("decode_invalid_hold_end_rate", metrics)
        self.assertEqual(metrics["active_boundary_exact_match"], 0.0)
        self.assertEqual(metrics["active_boundary_evaluated_boundary_count"], 1)
        self.assertEqual(metrics["boundary_error_by_bin"]["2-3"], 1.0)
        self.assertEqual(metrics["boundary_error_by_bin"]["3-4"], 0.0)

    def test_decode_density_error_counts_lane_note_events_not_timepoints(self) -> None:
        vocab = Stage1Vocab()
        one_lane_tap = vocab.encode_timepoint_event(
            (LaneAction.TAP, LaneAction.NONE, LaneAction.NONE, LaneAction.NONE),
        )
        four_lane_chord = vocab.encode_timepoint_event(
            (LaneAction.TAP, LaneAction.TAP, LaneAction.TAP, LaneAction.TAP),
        )
        condition = [vocab.bos_id, vocab.diff_token_id(0), vocab.open_token_id(0)]
        generated_target = [vocab.ts_token_id(100), one_lane_tap, vocab.eos_id]
        reference_target = [vocab.ts_token_id(100), four_lane_chord, vocab.eos_id]
        sample = _sample_for_decode_metrics(
            vocab,
            condition=condition,
            target=reference_target,
            difficulty_bucket=0,
            open_hold_mask=0,
            write_start_ms=0,
        )
        batch = collate_oracle_windows([sample], pad_id=vocab.pad_id)
        model = _SequenceModel(
            {0: generated_target},
            key_source="difficulty_bucket",
            vocab_size=vocab.size,
        )

        metrics = greedy_decode_metrics_for_loader(model, [batch], vocab=vocab, device=torch.device("cpu"))

        self.assertEqual(metrics["decode_density_error"], 0.75)

    def test_decode_metrics_count_max_length_forced_eos_as_eos_failure(self) -> None:
        vocab = Stage1Vocab()
        tap = vocab.encode_timepoint_event(
            (LaneAction.TAP, LaneAction.NONE, LaneAction.NONE, LaneAction.NONE),
        )
        condition = [vocab.bos_id, vocab.diff_token_id(0), vocab.open_token_id(0)]
        reference_target = [vocab.ts_token_id(0), tap, vocab.eos_id]
        sample = _sample_for_decode_metrics(
            vocab,
            condition=condition,
            target=reference_target,
            difficulty_bucket=0,
            open_hold_mask=0,
            write_start_ms=0,
        )
        batch = collate_oracle_windows([sample], pad_id=vocab.pad_id)
        model = _SequenceModel(
            {0: [vocab.ts_token_id(0), tap, vocab.ts_token_id(100)]},
            key_source="difficulty_bucket",
            vocab_size=vocab.size,
            max_decode_len=3,
        )

        metrics = greedy_decode_metrics_for_loader(model, [batch], vocab=vocab, device=torch.device("cpu"))

        self.assertEqual(metrics["decode_max_decode_len_reached_rate"], 1.0)
        self.assertEqual(metrics["decode_eos_failure_rate"], 1.0)

    def test_decode_metrics_report_per_difficulty_bin(self) -> None:
        vocab = Stage1Vocab()
        tap = vocab.encode_timepoint_event(
            (LaneAction.TAP, LaneAction.NONE, LaneAction.NONE, LaneAction.NONE),
        )
        low_condition = [vocab.bos_id, vocab.diff_token_id(0), vocab.open_token_id(0)]
        high_condition = [vocab.bos_id, vocab.diff_token_id(12), vocab.open_token_id(0)]
        reference_target = [vocab.ts_token_id(100), tap, vocab.eos_id]
        high_generated_target = [vocab.ts_token_id(100), tap, vocab.eos_id]
        samples = [
            _sample_for_decode_metrics(
                vocab,
                condition=low_condition,
                target=reference_target,
                difficulty_bucket=0,
                open_hold_mask=0,
                write_start_ms=0,
            ),
            _sample_for_decode_metrics(
                vocab,
                condition=high_condition,
                target=reference_target,
                difficulty_bucket=12,
                open_hold_mask=0,
                write_start_ms=0,
            ),
        ]
        batch = collate_oracle_windows(samples, pad_id=vocab.pad_id)
        model = _SequenceModel(
            {
                0: [vocab.eos_id],
                12: high_generated_target,
            },
            key_source="difficulty_bucket",
            vocab_size=vocab.size,
        )

        metrics = greedy_decode_metrics_for_loader(model, [batch], vocab=vocab, device=torch.device("cpu"))
        per_bin = metrics["decode_per_difficulty_bin"]

        self.assertEqual(metrics["decode_empty_output_rate"], 0.5)
        self.assertEqual(per_bin["2-3"]["decode_empty_output_rate"], 1.0)
        self.assertEqual(per_bin["2-3"]["decode_density_error"], 1.0)
        self.assertEqual(per_bin["5-6"]["decode_empty_output_rate"], 0.0)
        self.assertEqual(per_bin["5-6"]["decode_density_error"], 0.0)
        self.assertEqual(per_bin["3-4"]["decode_evaluated_window_count"], 0)

    def test_decode_metrics_use_true_difficulty_for_coarse_bins(self) -> None:
        vocab = Stage1Vocab()
        boundary_difficulty = 3.88
        rounded_bucket = vocab.difficulty_bucket_id(boundary_difficulty)
        condition = [vocab.bos_id, vocab.diff_token_id(rounded_bucket), vocab.open_token_id(0)]
        target = [vocab.eos_id]
        sample = _sample_for_decode_metrics(
            vocab,
            condition=condition,
            target=target,
            difficulty_bucket=rounded_bucket,
            difficulty=boundary_difficulty,
            open_hold_mask=0,
            write_start_ms=0,
        )
        batch = collate_oracle_windows([sample], pad_id=vocab.pad_id)
        model = _SequenceModel(
            {rounded_bucket: target},
            key_source="difficulty_bucket",
            vocab_size=vocab.size,
        )

        metrics = greedy_decode_metrics_for_loader(model, [batch], vocab=vocab, device=torch.device("cpu"))
        per_bin = metrics["decode_per_difficulty_bin"]

        self.assertEqual(per_bin["3-4"]["decode_evaluated_window_count"], 1)
        self.assertEqual(per_bin["4-5"]["decode_evaluated_window_count"], 0)

    def test_clone_tensor_to_cpu_detaches_and_copies_slice_storage(self) -> None:
        source = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
        slice_view = source[:, 1:2, :]

        cloned = _clone_tensor_to_cpu(slice_view)
        source[:, 1:2, :].fill_(-1)

        self.assertEqual(cloned.device.type, "cpu")
        self.assertEqual(cloned.tolist(), [[[4.0, 5.0, 6.0, 7.0]], [[16.0, 17.0, 18.0, 19.0]]])
        self.assertNotEqual(cloned.untyped_storage().data_ptr(), slice_view.untyped_storage().data_ptr())

    def test_move_rollout_batch_inputs_moves_only_model_inputs(self) -> None:
        batch = {
            "packed_audio": torch.arange(12, dtype=torch.float32).reshape(1, 3, 4),
            "timing_track": torch.arange(15, dtype=torch.float32).reshape(1, 3, 5),
            "difficulty_bucket": torch.tensor([2], dtype=torch.long),
            "decoder_input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "labels": torch.tensor([[4, 5, 6]], dtype=torch.long),
        }

        moved = _move_rollout_batch_inputs(batch, torch.device("cpu"))

        self.assertEqual(set(moved), {"packed_audio", "timing_track", "difficulty_bucket"})
        self.assertTrue(torch.equal(moved["packed_audio"], batch["packed_audio"]))
        self.assertTrue(torch.equal(moved["timing_track"], batch["timing_track"]))
        self.assertTrue(torch.equal(moved["difficulty_bucket"], batch["difficulty_bucket"]))

    def test_decode_window_input_to_device_restores_cpu_cached_tensors(self) -> None:
        decode_input = _DecodeWindowInput(
            write_start_ms=120,
            packed_audio=torch.arange(12, dtype=torch.float32).reshape(1, 3, 4),
            timing_track=torch.arange(15, dtype=torch.float32).reshape(1, 3, 5),
            difficulty_bucket=torch.tensor([2], dtype=torch.long),
            condition_ids=[1, 2, 3],
            oracle_open_hold_mask=1,
            write_duration_ms=8000,
            difficulty_bin_label="2-3",
        )

        packed_audio, timing_track, difficulty_bucket = _decode_window_input_to_device(
            decode_input,
            device=torch.device("cpu"),
        )

        self.assertEqual(packed_audio.device.type, "cpu")
        self.assertEqual(timing_track.device.type, "cpu")
        self.assertEqual(difficulty_bucket.device.type, "cpu")
        self.assertTrue(torch.equal(packed_audio, decode_input.packed_audio))
        self.assertTrue(torch.equal(timing_track, decode_input.timing_track))
        self.assertTrue(torch.equal(difficulty_bucket, decode_input.difficulty_bucket))

    def test_balanced_sampling_caps_empty_windows_per_bin(self) -> None:
        records = []
        for difficulty in (2.5, 3.5):
            for index in range(4):
                records.append(_window_record(difficulty=difficulty, has_event=True, name=f"{difficulty}-hit-{index}"))
            for index in range(10):
                records.append(_window_record(difficulty=difficulty, has_event=False, name=f"{difficulty}-empty-{index}"))

        plan = build_balanced_epoch_sampling_plan(records, empty_window_cap_ratio=0.25, seed=7)

        self.assertEqual(plan.sample_count_by_bin["2-3"], plan.sample_count_by_bin["3-4"])
        self.assertLessEqual(plan.empty_sample_count_by_bin["2-3"] / plan.sample_count_by_bin["2-3"], 0.25)
        self.assertLessEqual(plan.empty_sample_count_by_bin["3-4"] / plan.sample_count_by_bin["3-4"], 0.25)

    def test_balanced_sampling_uses_empty_window_cap_by_bin(self) -> None:
        records = []
        for difficulty in (2.5, 3.5):
            for index in range(2):
                records.append(_window_record(difficulty=difficulty, has_event=True, name=f"{difficulty}-hit-{index}"))
            for index in range(10):
                records.append(_window_record(difficulty=difficulty, has_event=False, name=f"{difficulty}-empty-{index}"))

        plan = build_balanced_epoch_sampling_plan(
            records,
            empty_window_cap_ratio=0.5,
            empty_window_cap_by_bin={
                "2-3": 0.0,
                "3-4": 0.5,
                "4-5": 0.0,
                "5-6": 0.0,
            },
            seed=7,
        )

        self.assertEqual(plan.empty_sample_count_by_bin["2-3"], 0)
        self.assertGreater(plan.empty_sample_count_by_bin["3-4"], 0)
        self.assertEqual(plan.empty_window_cap_by_bin["2-3"], 0.0)
        self.assertEqual(plan.empty_window_cap_by_bin["3-4"], 0.5)

    def test_gate_manifest_validator_rejects_missing_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(PretrainingGateValidationError, "not found"):
                validate_pretraining_gate_manifest(Path(tmpdir) / "missing.json", repo_root=Path(tmpdir))

    def test_gate_manifest_validator_rejects_missing_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            debug_dir = root / "debug"
            debug_dir.mkdir()
            (debug_dir / "plot.png").write_bytes(b"png")
            _write_gate_artifacts(root, debug_dir=debug_dir)
            manifest = root / "gates.json"
            gates = _gate_manifest_entries()
            del gates["event_space"]
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "training": _training_manifest_config(),
                        "gates": gates,
                    },
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(PretrainingGateValidationError, "event_space"):
                validate_pretraining_gate_manifest(manifest, repo_root=root)

    def test_gate_manifest_validator_rejects_failing_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            debug_dir = root / "debug"
            debug_dir.mkdir()
            (debug_dir / "plot.png").write_bytes(b"png")
            _write_gate_artifacts(root, debug_dir=debug_dir)
            artifact = root / "token_statistics.json"
            artifact.write_text(json.dumps({"gate_decision": {"status": "FAIL"}}), encoding="utf-8")
            manifest = _write_gate_manifest(
                root,
                {
                    "token_statistics": {"artifact_path": "token_statistics.json"},
                },
            )

            with self.assertRaisesRegex(PretrainingGateValidationError, "token_statistics"):
                validate_pretraining_gate_manifest(manifest, repo_root=root)

    def test_gate_manifest_validator_rejects_inline_gate_status_without_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            debug_dir = root / "debug"
            debug_dir.mkdir()
            (debug_dir / "plot.png").write_bytes(b"png")
            _write_gate_artifacts(root, debug_dir=debug_dir)
            gates = _gate_manifest_entries()
            gates["round_trip"] = {"status": "PASS"}
            manifest = root / "gates.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "training": _training_manifest_config(),
                        "gates": gates,
                    },
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(PretrainingGateValidationError, "round_trip.*artifact_path"):
                validate_pretraining_gate_manifest(manifest, repo_root=root)

    def test_gate_manifest_validator_accepts_structured_v2_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            debug_dir = root / "debug"
            debug_dir.mkdir()
            (debug_dir / "plot.png").write_bytes(b"png")
            _write_gate_artifacts(root, debug_dir=debug_dir)
            manifest = _write_gate_manifest(root)

            result = validate_pretraining_gate_manifest(manifest, repo_root=root)
            config = training_config_from_pretraining_gates(result)

        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["schema_version"], 2)
        self.assertEqual(result["training"]["max_decode_len"], 640)
        self.assertEqual(result["training"]["empty_window_cap_ratio"], 0.05)
        self.assertEqual(result["gates"]["token_statistics"]["status"], "PASS")
        self.assertEqual(result["gates"]["token_statistics"]["artifact_path"], str(root / "token_statistics.json"))
        self.assertEqual(config["max_decode_len"], 640)
        self.assertEqual(config["empty_window_cap_ratio"], 0.05)

    def test_gate_manifest_validator_rejects_training_config_mismatched_with_token_statistics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            debug_dir = root / "debug"
            debug_dir.mkdir()
            (debug_dir / "plot.png").write_bytes(b"png")
            _write_gate_artifacts(root, debug_dir=debug_dir)
            (root / "token_statistics.json").write_text(
                json.dumps(
                    {
                        "gate_decision": {
                            "status": "PASS",
                            "configured_max_decode_len": 640,
                            "empty_window_cap_ratio": 0.05,
                            "empty_window_cap_by_bin": {
                                "2-3": 0.05,
                                "3-4": 0.05,
                                "4-5": 0.05,
                                "5-6": 0.05,
                            },
                        },
                    },
                ),
                encoding="utf-8",
            )
            manifest = _write_gate_manifest(
                root,
                training={
                    "max_decode_len": 128,
                    "empty_window_cap_ratio": 0.05,
                },
            )

            with self.assertRaisesRegex(PretrainingGateValidationError, "token_statistics.*max_decode_len"):
                validate_pretraining_gate_manifest(manifest, repo_root=root)

    def test_training_config_uses_token_statistics_empty_cap_by_bin(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            debug_dir = root / "debug"
            debug_dir.mkdir()
            (debug_dir / "plot.png").write_bytes(b"png")
            _write_gate_artifacts(root, debug_dir=debug_dir)
            (root / "token_statistics.json").write_text(
                json.dumps(
                    {
                        "gate_decision": {
                            "status": "PASS",
                            "configured_max_decode_len": 640,
                            "empty_window_cap_ratio": 0.05,
                            "empty_window_cap_by_bin": {
                                "2-3": 0.02,
                                "3-4": 0.03,
                                "4-5": 0.04,
                                "5-6": 0.05,
                            },
                        },
                    },
                ),
                encoding="utf-8",
            )
            manifest = _write_gate_manifest(root)

            result = validate_pretraining_gate_manifest(manifest, repo_root=root)
            config = training_config_from_pretraining_gates(result)

        self.assertEqual(
            config["empty_window_cap_by_bin"],
            {
                "2-3": 0.02,
                "3-4": 0.03,
                "4-5": 0.04,
                "5-6": 0.05,
            },
        )

    def test_gate_manifest_exposes_dense_timing_training_stats(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            debug_dir = root / "debug"
            debug_dir.mkdir()
            (debug_dir / "plot.png").write_bytes(b"png")
            _write_gate_artifacts(root, debug_dir=debug_dir)
            artifact = root / "dense_timing.json"
            artifact.write_text(
                json.dumps(
                    {
                        "gate_decision": {
                            "status": "PASS",
                            "bpm_log_mean": 4.75,
                            "bpm_log_std": 0.33,
                        },
                    },
                ),
                encoding="utf-8",
            )
            manifest = _write_gate_manifest(
                root,
                {
                    "dense_timing_track": {"artifact_path": "dense_timing.json"},
                },
            )

            result = validate_pretraining_gate_manifest(manifest, repo_root=root)
            stats = timing_training_stats_from_pretraining_gates(result)

        self.assertEqual(result["gates"]["dense_timing_track"]["bpm_log_mean"], 4.75)
        self.assertEqual(result["gates"]["dense_timing_track"]["bpm_log_std"], 0.33)
        self.assertEqual(stats, {"bpm_log_mean": 4.75, "bpm_log_std": 0.33})

    def test_gate_manifest_exposes_top_level_training_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            debug_dir = root / "debug"
            debug_dir.mkdir()
            (debug_dir / "plot.png").write_bytes(b"png")
            _write_gate_artifacts(root, debug_dir=debug_dir)
            manifest = _write_gate_manifest(
                root,
                training={
                    "max_decode_len": 768,
                    "empty_window_cap_ratio": 0.125,
                },
            )

            result = validate_pretraining_gate_manifest(manifest, repo_root=root)
            config = training_config_from_pretraining_gates(result)

        self.assertEqual(config["max_decode_len"], 768)
        self.assertEqual(config["empty_window_cap_ratio"], 0.125)

    def test_gate_manifest_validator_requires_debug_plot_png(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "debug").mkdir()
            _write_gate_artifacts(root, debug_dir=root / "debug", write_debug_plot=False)
            manifest = _write_gate_manifest(
                root,
                {"dense_timing_debug_plots": {"artifact_path": "dense_timing_debug_plots.json", "debug_plot_dir": "debug"}},
            )

            with self.assertRaisesRegex(PretrainingGateValidationError, "debug"):
                validate_pretraining_gate_manifest(manifest, repo_root=root)


def _write_gate_manifest(
    root: Path,
    overrides: dict[str, dict[str, str]] | None = None,
    *,
    training: dict[str, object] | None = None,
) -> Path:
    gates = _gate_manifest_entries()
    if overrides:
        gates.update(overrides)
    manifest = root / "gates.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "training": _training_manifest_config(training),
                "gates": gates,
            },
        ),
        encoding="utf-8",
    )
    return manifest


def _gate_manifest_entries() -> dict[str, dict[str, str]]:
    return {
        name: {"artifact_path": f"{name}.json"}
        for name in REQUIRED_PRETRAINING_GATE_NAMES
    } | {
        "dense_timing_debug_plots": {
            "artifact_path": "dense_timing_debug_plots.json",
            "debug_plot_dir": "debug",
        },
        "token_statistics": {"artifact_path": "token_statistics.json"},
        "dense_timing_track": {"artifact_path": "dense_timing.json"},
    }


def _training_manifest_config(overrides: dict[str, object] | None = None) -> dict[str, object]:
    config = {
        "max_decode_len": 640,
        "empty_window_cap_ratio": 0.05,
    }
    if overrides:
        config.update(overrides)
    return config


def _write_gate_artifacts(root: Path, *, debug_dir: Path, write_debug_plot: bool = True) -> None:
    for gate_name in REQUIRED_PRETRAINING_GATE_NAMES:
        artifact_name = "dense_timing.json" if gate_name == "dense_timing_track" else f"{gate_name}.json"
        payload: dict[str, object] = {"gate_decision": {"status": "PASS"}}
        if gate_name == "dense_timing_track":
            payload = {
                "gate_decision": {
                    "status": "PASS",
                    "bpm_log_mean": 4.75,
                    "bpm_log_std": 0.33,
                },
            }
        (root / artifact_name).write_text(json.dumps(payload), encoding="utf-8")
    if write_debug_plot and debug_dir.is_dir():
        (debug_dir / "plot.png").write_bytes(b"png")


def _sample_for_decode_metrics(
    vocab: Stage1Vocab,
    *,
    condition: list[int],
    target: list[int],
    difficulty_bucket: int,
    difficulty: float | None = None,
    open_hold_mask: int,
    write_start_ms: int,
) -> dict[str, object]:
    return {
        "packed_audio": torch.zeros(600, 160),
        "timing_track": torch.zeros(600, 5),
        "difficulty_bucket": torch.tensor(difficulty_bucket, dtype=torch.long),
        "open_hold_mask": torch.tensor(open_hold_mask, dtype=torch.long),
        "write_duration_ms": torch.tensor(8000, dtype=torch.long),
        "decoder_input_ids": torch.tensor(condition + target[:-1], dtype=torch.long),
        "labels": torch.tensor([-100, -100] + target, dtype=torch.long),
        "beatmap_path": "crossing_hold.osu",
        "audio_path": "crossing_hold.mp3",
        "difficulty": float(difficulty if difficulty is not None else 2.0 + difficulty_bucket * 0.25),
        "write_start_ms": write_start_ms,
    }


def _window_record(*, difficulty: float, has_event: bool, name: str) -> object:
    timepoints = (
        (CanonicalTimepoint(100, (LaneAction.TAP, LaneAction.NONE, LaneAction.NONE, LaneAction.NONE)),)
        if has_event
        else ()
    )
    return SimpleNamespace(
        beatmap_path=Path(f"{name}.osu"),
        difficulty=difficulty,
        timepoints=timepoints,
        window=SimpleNamespace(write_start_ms=0, write_end_ms=8000),
    )


if __name__ == "__main__":
    unittest.main()
