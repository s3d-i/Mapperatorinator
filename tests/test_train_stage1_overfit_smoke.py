import tempfile
import unittest
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from train.stage1_oracle.data.windows import OracleWindowFilterReport, collate_oracle_windows
from train.stage1_oracle.events.canonical import CanonicalTimepoint, LaneAction
from train.stage1_oracle.events.tokens import Stage1Vocab, decompose_ts_delta
from train.stage1_oracle.training.overfit_32 import (
    REQUIRED_PRETRAINING_GATE_NAMES,
    PretrainingGateValidationError,
    OverfitRunResult,
    build_balanced_epoch_sampling_plan,
    greedy_decode_metrics_for_loader,
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
            result = run_synthetic_smoke(
                output_dir=Path(tmpdir),
                max_steps=2,
                seed=1337,
            )

            self.assertTrue(result.report_path.is_file())
            self.assertTrue(result.checkpoint_path.is_file())
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
            self.assertIn("decode_eos_failure_rate", report["final"])
            self.assertIn("decode_empty_output_rate", report["final"])
            self.assertIn("decode_eos_forced_after_pending_ts_rate", report["final"])
            self.assertEqual(report["final"]["decode_evaluated_window_count"], 2)

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

    def test_overfit_32_requires_eight_retained_maps_per_bin(self) -> None:
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
                ):
                    with self.assertRaisesRegex(ValueError, "requires 8 retained maps per bin"):
                        run_overfit_32(
                            dataset_root=Path("mania-dataset"),
                            index_path=None,
                            gate_manifest_path=Path("gates.json"),
                            output_dir=Path("out"),
                        )

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
