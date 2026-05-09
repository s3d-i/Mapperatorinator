import tempfile
import unittest
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from train.stage_2.data.control_windows import ControlWindowRecord, normalize_difficulty
from train.stage_2.data.mapper_v1_windows import (
    MapperV1WindowDataset,
    collate_mapper_v1_windows,
    control_teacher_cache_path,
    load_control_teacher_cache_entry,
)
from train.stage_2.features.control_v3_targets import MODEL_FEATURE_NAMES
from train.stage_2.model_control_demo_global import ControlDemoGlobalEncoderConfig
from train.stage_2.model_mapper_v1 import MapperV1Config, MapperV1Model
from train.stage_2.training import mapper_v1 as mapper_v1_training
from train.stage_2.training.mapper_v1 import (
    MapperV1PhaseBLossConfig,
    _MapperV1TokenLengthBucketBatchSampler,
    _collate_synthetic_mapper_samples,
    initialize_mapper_v1_from_mapper_checkpoint,
    _loss_for_raw_batch,
    _synthetic_mapper_samples,
    load_run_config,
    precompute_phase_b_control_teacher_cache,
    precompute_phase_b_control_teacher_cache_from_control_dataset,
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
        self.assertTrue(config["length_bucketed_batches"])
        self.assertEqual(config["length_bucket_size_multiplier"], 32)
        self.assertIn("stage2_control_demo_global", config["init_from_control_checkpoint"])
        cached_config = load_run_config("train/stage_2/training/configs/stage2_mapper_v1_phase_b_cached_demo_mps.yaml")
        self.assertTrue(cached_config["length_bucketed_batches"])
        self.assertEqual(cached_config["length_bucket_size_multiplier"], 32)
        density_eos_config = load_run_config(
            "train/stage_2/training/configs/stage2_mapper_v1_phase_b_cached_demo_density_eos_mps.yaml",
        )
        self.assertIn("checkpoint_step_001250.pt", density_eos_config["init_from_mapper_checkpoint"])
        self.assertIn("plus_end.parquet", density_eos_config["index_path"])
        self.assertIn("window_records", density_eos_config["mapper_record_cache_path"])
        self.assertGreater(density_eos_config["loss"]["lambda_density"], 0.0)
        self.assertFalse(density_eos_config["precompute_control_teacher_cache"])

    def test_phase_b_loss_config_allows_density_enablement(self) -> None:
        config = MapperV1PhaseBLossConfig(lambda_density=0.01)

        self.assertEqual(config.lambda_density, 0.01)
        with self.assertRaisesRegex(ValueError, "density teacher loss is not implemented"):
            MapperV1PhaseBLossConfig(lambda_density_teacher=0.01)

    def test_synthetic_loss_path_reports_finite_density_auxiliary(self) -> None:
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
        self.assertGreaterEqual(loss_output.metrics["loss/density"], 0.0)
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
        batch["target_fragment_mask"] = torch.ones_like(batch["target_fragment_tokens"], dtype=torch.bool)
        base = _loss_for_raw_batch(
            model,
            batch,
            device=torch.device("cpu"),
            loss_config=MapperV1PhaseBLossConfig(),
        )
        padded = dict(batch)
        padded["decoder_input_tokens"] = torch.cat(
            [batch["decoder_input_tokens"], torch.zeros((2, 1), dtype=torch.long)],
            dim=1,
        )
        padded["target_fragment_tokens"] = torch.cat(
            [batch["target_fragment_tokens"], torch.zeros((2, 1), dtype=torch.long)],
            dim=1,
        )
        padded["target_fragment_mask"] = torch.cat(
            [batch["target_fragment_mask"], torch.zeros((2, 1), dtype=torch.bool)],
            dim=1,
        )
        padded["target_fragment_states"] = {
            "current_ms": torch.cat(
                [batch["target_fragment_states"]["current_ms"], torch.full((2, 1), 99_999, dtype=torch.long)],
                dim=1,
            ),
            "open_mask": torch.cat(
                [batch["target_fragment_states"]["open_mask"], torch.ones((2, 1, 4), dtype=torch.bool)],
                dim=1,
            ),
            "open_start_ms": torch.cat(
                [batch["target_fragment_states"]["open_start_ms"], torch.full((2, 1, 4), 99_999, dtype=torch.long)],
                dim=1,
            ),
            "open_age_ms": torch.cat(
                [batch["target_fragment_states"]["open_age_ms"], torch.full((2, 1, 4), 99_999, dtype=torch.long)],
                dim=1,
            ),
        }
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

    def test_length_bucket_sampler_groups_by_target_tokens_without_materializing_samples(self) -> None:
        dataset = _TargetLengthDataset([80, 5, 78, 6, 7, 82, 8, 79])

        sampler = _MapperV1TokenLengthBucketBatchSampler(
            dataset,
            batch_size=2,
            bucket_size_multiplier=4,
            shuffle=False,
            seed=1337,
        )

        self.assertEqual(list(iter(sampler)), [[1, 3], [4, 6], [2, 7], [0, 5]])
        self.assertEqual(len(sampler), 4)
        self.assertEqual(dataset.getitem_calls, 0)

    def test_length_bucket_sampler_uses_subset_local_indices(self) -> None:
        dataset = _TargetLengthDataset([80, 5, 78, 6, 7, 82, 8, 79])
        subset = Subset(dataset, [5, 1, 0, 4])

        sampler = _MapperV1TokenLengthBucketBatchSampler(
            subset,
            batch_size=2,
            bucket_size_multiplier=2,
            shuffle=False,
            seed=1337,
        )

        self.assertEqual(list(iter(sampler)), [[1, 3], [2, 0]])
        self.assertEqual(dataset.getitem_calls, 0)

    def test_length_bucket_sampler_uses_record_target_seq_len_without_retokenizing(self) -> None:
        dataset = _RecordLengthDataset([40, 10, 38, 12])

        sampler = _MapperV1TokenLengthBucketBatchSampler(
            dataset,
            batch_size=2,
            bucket_size_multiplier=2,
            shuffle=False,
            seed=1337,
        )

        self.assertEqual(list(iter(sampler)), [[1, 3], [2, 0]])
        self.assertEqual(dataset.tokenize_calls, 0)

    def test_initialize_mapper_from_checkpoint_loads_model_state_only(self) -> None:
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
        source = MapperV1Model(model_config)
        with torch.no_grad():
            for index, parameter in enumerate(source.parameters()):
                parameter.fill_(0.01 * (index + 1))
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "mapper.pt"
            torch.save(
                {
                    "checkpoint_schema_version": mapper_v1_training.CHECKPOINT_SCHEMA_VERSION,
                    "model_state_dict": source.state_dict(),
                    "optimizer_state_dict": {"state": {"would_be_ignored": torch.ones(1)}},
                    "model_config": model_config.__dict__,
                    "control_model_config": None,
                    "training_state": {"step": 1000},
                },
                checkpoint_path,
            )
            target = MapperV1Model(model_config)

            report = initialize_mapper_v1_from_mapper_checkpoint(
                target,
                checkpoint_path,
                expected_model_config=model_config,
                expected_control_model_config=None,
            )

        self.assertEqual(report["kind"], "mapper_v1_model_state")
        self.assertEqual(report["checkpoint_step"], 1000)
        self.assertFalse(report["optimizer_state_loaded"])
        self.assertEqual(report["loaded_keys"], len(source.state_dict()))
        for key, value in source.state_dict().items():
            self.assertTrue(torch.equal(target.state_dict()[key], value), key)

    def test_cache_only_cli_runs_precompute_without_training(self) -> None:
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
                mapper_v1_training,
                "precompute_mapper_v1_phase_b_control_teacher_cache",
                return_value=precompute_result,
                autospec=True,
            ) as precompute:
                with patch.object(mapper_v1_training, "run_mapper_v1_phase_b_training") as train:
                    mapper_v1_training.main(
                        [
                            "--precompute-control-teacher-cache-only",
                            "--control-teacher-cache-dir",
                            str(Path(temp_dir) / "cache"),
                        ]
                    )

        precompute.assert_called_once()
        self.assertNotIn("length_bucketed_batches", precompute.call_args.kwargs)
        self.assertNotIn("length_bucket_size_multiplier", precompute.call_args.kwargs)
        train.assert_not_called()

    def test_main_forwards_length_bucket_options_to_training(self) -> None:
        train_result = SimpleNamespace(
            report_path=Path("report.json"),
            checkpoint_path=Path("checkpoint.pt"),
            final_loss=0.0,
            completed_steps=0,
        )
        with patch.object(
            mapper_v1_training,
            "run_mapper_v1_phase_b_training",
            return_value=train_result,
            autospec=True,
        ) as train:
            mapper_v1_training.main(
                [
                    "--length-bucketed-batches",
                    "--length-bucket-size-multiplier",
                    "7",
                    "--max-steps",
                    "1",
                ]
            )

        train.assert_called_once()
        self.assertTrue(train.call_args.kwargs["length_bucketed_batches"])
        self.assertEqual(train.call_args.kwargs["length_bucket_size_multiplier"], 7)

    def test_main_forwards_mapper_record_cache_path_to_training(self) -> None:
        train_result = SimpleNamespace(
            report_path=Path("report.json"),
            checkpoint_path=Path("checkpoint.pt"),
            final_loss=0.0,
            completed_steps=0,
        )
        with patch.object(
            mapper_v1_training,
            "run_mapper_v1_phase_b_training",
            return_value=train_result,
            autospec=True,
        ) as train:
            mapper_v1_training.main(
                [
                    "--mapper-record-cache-path",
                    "train/artifacts/cache/stage2_mapper_v1/window_records/test.parquet",
                    "--max-steps",
                    "1",
                ]
            )

        train.assert_called_once()
        self.assertEqual(
            train.call_args.kwargs["mapper_record_cache_path"],
            Path("train/artifacts/cache/stage2_mapper_v1/window_records/test.parquet"),
        )

    def test_main_forwards_mapper_checkpoint_init_to_training(self) -> None:
        train_result = SimpleNamespace(
            report_path=Path("report.json"),
            checkpoint_path=Path("checkpoint.pt"),
            final_loss=0.0,
            completed_steps=0,
        )
        with patch.object(
            mapper_v1_training,
            "run_mapper_v1_phase_b_training",
            return_value=train_result,
            autospec=True,
        ) as train:
            mapper_v1_training.main(
                [
                    "--init-from-mapper-checkpoint",
                    "checkpoint_step_001000.pt",
                    "--max-steps",
                    "1",
                ]
            )

        train.assert_called_once()
        self.assertEqual(train.call_args.kwargs["init_from_mapper_checkpoint"], Path("checkpoint_step_001000.pt"))

    def test_mapper_checkpoint_init_skips_control_checkpoint_init(self) -> None:
        model_config = MapperV1Config(
            control_dim=4,
            d_model=4,
            heads=2,
            layers=1,
            ffn_dim=8,
            dropout=0.0,
            max_seq_len=16,
            state_hidden_dim=8,
            ln_close_hidden_dim=8,
        )
        control_model_config = ControlDemoGlobalEncoderConfig(
            d_model=4,
            heads=1,
            layers=1,
            ffn_dim=8,
            conv_blocks=1,
            use_global_memory=False,
            global_fusion_start_layer=0,
        )
        loader = DataLoader(
            _synthetic_mapper_samples(model_config=model_config)[:1],
            batch_size=1,
            collate_fn=_collate_synthetic_mapper_samples,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(
                mapper_v1_training,
                "initialize_mapper_v1_from_mapper_checkpoint",
                return_value={"kind": "mapper_v1_model_state"},
                autospec=True,
            ) as mapper_init:
                with patch.object(
                    mapper_v1_training,
                    "initialize_global_control_demo_from_control_checkpoint",
                    autospec=True,
                ) as control_init:
                    mapper_v1_training._run_training(
                        loader=loader,
                        train_eval_loader=loader,
                        eval_loader=loader,
                        output_dir=Path(temp_dir),
                        model_config=model_config,
                        control_model_config=control_model_config,
                        loss_config=MapperV1PhaseBLossConfig(),
                        max_steps=1,
                        eval_every=1,
                        save_every=1,
                        log_every=None,
                        batch_size=1,
                        learning_rate=1e-4,
                        weight_decay=0.0,
                        seed=11,
                        device_name="cpu",
                        run_name="mapper_init_test",
                        dataset_report={"status": "test"},
                        init_from_control_checkpoint=Path("control.pt"),
                        init_from_mapper_checkpoint=Path("mapper.pt"),
                    )

        mapper_init.assert_called_once()
        control_init.assert_not_called()

    def test_precomputed_control_teacher_cache_feeds_phase_b_loss_without_full_inputs(self) -> None:
        record = ControlWindowRecord(
            beatmap_path=Path("cached.osu"),
            audio_path=Path("cached.mp3"),
            difficulty=4.0,
            frame_count=400,
            target_start_frame=0,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "cache"
            dataset = _TinyMapperDataset([record], cache_dir=cache_dir)
            result = precompute_phase_b_control_teacher_cache(
                dataset,
                cache_dir=cache_dir,
                control_encoder=_TinyControlTeacherEncoder(control_dim=4),
                batch_size=1,
                device=torch.device("cpu"),
            )

            self.assertEqual(result.computed_entries, 1)
            sample = dataset[0]
            self.assertIn("control_memory_8s", sample)
            self.assertIn("density_teacher_8s", sample)
            self.assertNotIn("full_mel", sample)
            batch = collate_mapper_v1_windows([sample])
            model = MapperV1Model(
                MapperV1Config(
                    control_dim=4,
                    d_model=4,
                    heads=2,
                    layers=1,
                    ffn_dim=8,
                    dropout=0.0,
                    max_seq_len=16,
                    state_hidden_dim=8,
                    ln_close_hidden_dim=8,
                )
            )
            loss_output = _loss_for_raw_batch(
                model,
                batch,
                device=torch.device("cpu"),
                loss_config=MapperV1PhaseBLossConfig(),
            )
            self.assertTrue(torch.isfinite(loss_output.total_loss))

    def test_precompute_slice_order_matches_8s_cache_layout(self) -> None:
        records = [
            ControlWindowRecord(
                beatmap_path=Path("first.osu"),
                audio_path=Path("first.mp3"),
                difficulty=4.0,
                frame_count=500,
                target_start_frame=0,
            ),
            ControlWindowRecord(
                beatmap_path=Path("second.osu"),
                audio_path=Path("second.mp3"),
                difficulty=4.0,
                frame_count=900,
                target_start_frame=400,
            ),
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "cache"
            dataset = _TinyMapperDataset(records, cache_dir=cache_dir)
            result = precompute_phase_b_control_teacher_cache(
                dataset,
                cache_dir=cache_dir,
                control_encoder=_TinyControlTeacherEncoder(control_dim=2),
                batch_size=2,
                device=torch.device("cpu"),
            )

            self.assertEqual(result.computed_entries, 2)
            first = dataset[0]["control_memory_8s"][:, 0]
            second = dataset[1]["control_memory_8s"][:, 0]
            self.assertTrue(torch.equal(first, torch.arange(0, 400, 100, dtype=torch.float32).repeat_interleave(100)))
            self.assertTrue(torch.equal(second, torch.arange(400, 800, 100, dtype=torch.float32).repeat_interleave(100)))
            self.assertEqual(dataset[0]["density_teacher_8s"].shape, (400, 1))

    def test_precompute_does_not_stack_four_slices_into_device_batch(self) -> None:
        records = [
            ControlWindowRecord(
                beatmap_path=Path("first.osu"),
                audio_path=Path("first.mp3"),
                difficulty=4.0,
                frame_count=500,
                target_start_frame=0,
            ),
            ControlWindowRecord(
                beatmap_path=Path("second.osu"),
                audio_path=Path("second.mp3"),
                difficulty=4.0,
                frame_count=900,
                target_start_frame=400,
            ),
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "cache"
            dataset = _TinyMapperDataset(records, cache_dir=cache_dir)
            encoder = _TinyControlTeacherEncoder(control_dim=2)

            precompute_phase_b_control_teacher_cache(
                dataset,
                cache_dir=cache_dir,
                control_encoder=encoder,
                batch_size=2,
                device=torch.device("cpu"),
            )

            self.assertEqual(encoder.batch_sizes, [2, 2, 2, 2])

    def test_raw_control_precompute_skips_mapper_tokenization_filter(self) -> None:
        records = [
            ControlWindowRecord(
                beatmap_path=Path("raw.osu"),
                audio_path=Path("raw.mp3"),
                difficulty=4.0,
                frame_count=500,
                target_start_frame=0,
            ),
            ControlWindowRecord(
                beatmap_path=Path("terminal.osu"),
                audio_path=Path("terminal.mp3"),
                difficulty=4.0,
                frame_count=500,
                target_start_frame=100,
            ),
            ControlWindowRecord(
                beatmap_path=Path("stride_skip.osu"),
                audio_path=Path("stride_skip.mp3"),
                difficulty=4.0,
                frame_count=700,
                target_start_frame=100,
            ),
            ControlWindowRecord(
                beatmap_path=Path("short_skip.osu"),
                audio_path=Path("short_skip.mp3"),
                difficulty=4.0,
                frame_count=500,
                target_start_frame=400,
            ),
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "cache"
            encoder = _TinyControlTeacherEncoder(control_dim=2)
            result = precompute_phase_b_control_teacher_cache_from_control_dataset(
                _TinyControlDataset(records),
                cache_dir=cache_dir,
                control_encoder=encoder,
                batch_size=2,
                device=torch.device("cpu"),
            )

            self.assertEqual(result.total_entries, 3)
            self.assertEqual(result.computed_entries, 3)
            entry = load_control_teacher_cache_entry(
                control_teacher_cache_path(cache_dir, records[0]),
                record=records[0],
            )
            self.assertEqual(tuple(entry["control_memory_8s"].shape), (400, 2))
            terminal_entry = load_control_teacher_cache_entry(
                control_teacher_cache_path(cache_dir, records[1]),
                record=records[1],
            )
            self.assertEqual(tuple(terminal_entry["control_memory_8s"].shape), (400, 2))
            self.assertFalse(control_teacher_cache_path(cache_dir, records[2]).exists())
            short_entry = load_control_teacher_cache_entry(
                control_teacher_cache_path(cache_dir, records[3]),
                record=records[3],
            )
            self.assertEqual(tuple(short_entry["control_memory_8s"].shape), (400, 2))
            self.assertIn(([800], [300]), encoder.padding_mask_observations)


class _TinyControlDataset:
    def __init__(self, records: list[ControlWindowRecord]) -> None:
        self.records = records

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        return {
            "full_mel": torch.zeros(record.frame_count, 160, dtype=torch.float32),
            "full_dense_timing_v2": torch.zeros(record.frame_count, 4, dtype=torch.float32),
            "frame_count": torch.tensor(record.frame_count, dtype=torch.long),
            "target_start_frame": torch.tensor(record.target_start_frame, dtype=torch.long),
            "difficulty": torch.tensor(record.difficulty, dtype=torch.float32),
            "normalized_difficulty": torch.tensor(normalize_difficulty(record.difficulty), dtype=torch.float32),
        }

    def target_loader(self, record: ControlWindowRecord) -> torch.Tensor:
        target = torch.zeros(100, len(MODEL_FEATURE_NAMES), dtype=torch.float32)
        target[:, MODEL_FEATURE_NAMES.index("density_confidence")] = 1.0
        return target


class _TargetLengthDataset(Dataset):
    def __init__(self, target_token_lengths: list[int]) -> None:
        self.target_token_lengths = target_token_lengths
        self.getitem_calls = 0

    def __len__(self) -> int:
        return len(self.target_token_lengths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        self.getitem_calls += 1
        raise AssertionError("length bucketing must not materialize samples")


class _RecordLengthDataset(Dataset):
    def __init__(self, target_seq_lengths: list[int]) -> None:
        self.records = [SimpleNamespace(target_seq_len=length) for length in target_seq_lengths]
        self.tokenize_calls = 0

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        raise AssertionError("length bucketing must not materialize samples")

    def _tokenize_record(self, record: object) -> object:
        self.tokenize_calls += 1
        raise AssertionError("target_seq_len should avoid retokenization")


class _TinyMapperDataset(MapperV1WindowDataset):
    def __init__(self, records: list[ControlWindowRecord], *, cache_dir: Path) -> None:
        super().__init__(control_dataset=_TinyControlDataset(records), control_teacher_cache_dir=cache_dir)

    def _load_timepoints(self, beatmap_path: Path) -> tuple:
        return ()


class _TinyControlTeacherEncoder(nn.Module):
    def __init__(self, *, control_dim: int) -> None:
        super().__init__()
        self.control_dim = int(control_dim)
        self.batch_sizes: list[int] = []
        self.padding_true_counts: list[list[int]] = []
        self.padding_mask_observations: list[tuple[list[int], list[int]]] = []

    def forward(self, *, context_mel: torch.Tensor, target_start_frame: torch.Tensor | None = None, **kwargs):
        batch_size, frames = context_mel.shape[:2]
        self.batch_sizes.append(batch_size)
        padding_mask = kwargs.get("padding_mask")
        if isinstance(padding_mask, torch.Tensor):
            padding_true_counts = padding_mask.to(dtype=torch.long).sum(dim=1).tolist()
            self.padding_true_counts.append(padding_true_counts)
            frame_count = kwargs.get("frame_count")
            if isinstance(frame_count, torch.Tensor):
                self.padding_mask_observations.append((frame_count.to(dtype=torch.long).tolist(), padding_true_counts))
        control_memory = torch.zeros(batch_size, frames, self.control_dim, dtype=context_mel.dtype, device=context_mel.device)
        if target_start_frame is not None:
            control_memory[:, :, 0] = target_start_frame.to(device=context_mel.device, dtype=context_mel.dtype).reshape(-1, 1)
        value_pred = torch.zeros(batch_size, 100, 1, dtype=context_mel.dtype, device=context_mel.device)
        return SimpleNamespace(control_memory=control_memory, value_pred=value_pred)


if __name__ == "__main__":
    unittest.main()
