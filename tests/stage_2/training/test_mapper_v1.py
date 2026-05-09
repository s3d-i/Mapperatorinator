import tempfile
import unittest
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from train.stage_2.data.control_windows import ControlWindowRecord, normalize_difficulty
from train.stage_2.data.mapper_v1_windows import (
    MapperV1WindowDataset,
    collate_mapper_v1_windows,
    control_teacher_cache_path,
    load_control_teacher_cache_entry,
)
from train.stage_2.features.control_v3_targets import MODEL_FEATURE_NAMES
from train.stage_2.model_mapper_v1 import MapperV1Config, MapperV1Model
from train.stage_2.training import mapper_v1 as mapper_v1_training
from train.stage_2.training.mapper_v1 import (
    MapperV1PhaseBLossConfig,
    _collate_synthetic_mapper_samples,
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
        self.assertIn("stage2_control_demo_global", config["init_from_control_checkpoint"])

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
        train.assert_not_called()

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
                beatmap_path=Path("stride_skip.osu"),
                audio_path=Path("stride_skip.mp3"),
                difficulty=4.0,
                frame_count=500,
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
            result = precompute_phase_b_control_teacher_cache_from_control_dataset(
                _TinyControlDataset(records),
                cache_dir=cache_dir,
                control_encoder=_TinyControlTeacherEncoder(control_dim=2),
                batch_size=2,
                device=torch.device("cpu"),
            )

            self.assertEqual(result.total_entries, 1)
            self.assertEqual(result.computed_entries, 1)
            entry = load_control_teacher_cache_entry(
                control_teacher_cache_path(cache_dir, records[0]),
                record=records[0],
            )
            self.assertEqual(tuple(entry["control_memory_8s"].shape), (400, 2))
            self.assertFalse(control_teacher_cache_path(cache_dir, records[1]).exists())
            self.assertFalse(control_teacher_cache_path(cache_dir, records[2]).exists())


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

    def forward(self, *, context_mel: torch.Tensor, target_start_frame: torch.Tensor | None = None, **kwargs):
        batch_size, frames = context_mel.shape[:2]
        self.batch_sizes.append(batch_size)
        control_memory = torch.zeros(batch_size, frames, self.control_dim, dtype=context_mel.dtype, device=context_mel.device)
        if target_start_frame is not None:
            control_memory[:, :, 0] = target_start_frame.to(device=context_mel.device, dtype=context_mel.dtype).reshape(-1, 1)
        value_pred = torch.zeros(batch_size, 100, 1, dtype=context_mel.dtype, device=context_mel.device)
        return SimpleNamespace(control_memory=control_memory, value_pred=value_pred)


if __name__ == "__main__":
    unittest.main()
