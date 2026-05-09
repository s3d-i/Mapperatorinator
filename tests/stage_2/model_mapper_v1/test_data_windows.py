import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from train.stage1_oracle.osu.hitobjects import ManiaHitObject, ManiaHitObjectKind
from train.stage_2.data.control_windows import ControlWindowRecord
from train.stage_2.data.mapper_v1_windows import (
    MapperV1WindowDataset,
    collate_mapper_v1_windows,
    concatenate_density_teacher_8s,
    control_teacher_cache_path,
    control_teacher_slice_batch,
    extract_mapper_density_8s,
    load_control_teacher_cache_entry,
    save_control_teacher_cache_entry,
)
from train.stage_2.features.control_v3_targets import MODEL_FEATURE_NAMES, VALUE_FEATURE_NAMES
from train.stage_2.model_mapper_v1.replay import ln_carry_state_tensors
from train.stage_2.model_mapper_v1.tokenizer import encode_mapper_window, hitobjects_to_mapper_timepoints
from train.stage_2.model_mapper_v1.vocab import MapperV1Vocab


class MapperV1DataWindowTests(unittest.TestCase):
    def test_extract_mapper_density_8s_uses_named_control_v3_channels(self) -> None:
        target = torch.zeros(400, len(MODEL_FEATURE_NAMES), dtype=torch.float32)
        target[:, MODEL_FEATURE_NAMES.index("density_level")] = torch.linspace(0.0, 1.0, 400)
        target[:, MODEL_FEATURE_NAMES.index("density_confidence")] = 0.75

        density_target, density_confidence = extract_mapper_density_8s(target)

        self.assertEqual(density_target.shape, (400, 1))
        self.assertEqual(density_confidence.shape, (400, 1))
        self.assertTrue(torch.equal(density_target[:, 0], torch.linspace(0.0, 1.0, 400)))
        self.assertTrue(torch.equal(density_confidence, torch.full((400, 1), 0.75)))

    def test_extract_mapper_density_8s_rejects_nonfinite_values(self) -> None:
        target = torch.zeros(400, len(MODEL_FEATURE_NAMES), dtype=torch.float32)
        target[:, MODEL_FEATURE_NAMES.index("density_confidence")] = 1.0
        target[7, MODEL_FEATURE_NAMES.index("density_confidence")] = float("nan")

        with self.assertRaisesRegex(ValueError, "finite"):
            extract_mapper_density_8s(target)

    def test_collate_mapper_v1_windows_pads_tokens_and_states(self) -> None:
        vocab = MapperV1Vocab()
        first = _sample(encode_mapper_window([], vocab=vocab, write_start_ms=0, write_end_ms=8000))
        second = _sample(encode_mapper_window([], vocab=vocab, write_start_ms=8000, write_end_ms=16000))
        second["decoder_input_tokens"] = second["decoder_input_tokens"][:-1]
        second["target_fragment_tokens"] = second["target_fragment_tokens"][:-1]
        second["target_fragment_states"]["current_ms"] = second["target_fragment_states"]["current_ms"][:-1]
        second["target_fragment_states"]["open_mask"] = second["target_fragment_states"]["open_mask"][:-1]
        second["target_fragment_states"]["open_start_ms"] = second["target_fragment_states"]["open_start_ms"][:-1]
        second["target_fragment_states"]["open_age_ms"] = second["target_fragment_states"]["open_age_ms"][:-1]
        second["close_labels"] = second["close_labels"][:-1]
        second["close_label_mask"] = second["close_label_mask"][:-1]

        batch = collate_mapper_v1_windows([first, second], pad_id=vocab.pad_id)

        self.assertEqual(batch["target_fragment_tokens"].shape, (2, 2))
        self.assertEqual(batch["target_fragment_states"]["open_mask"].shape, (2, 2, 4))
        self.assertTrue(batch["target_fragment_mask"][0].all().item())
        self.assertFalse(batch["target_fragment_mask"][1, -1].item())
        self.assertEqual(int(batch["target_fragment_tokens"][1, -1].item()), vocab.pad_id)
        self.assertEqual(batch["ln_carry_in"]["open_mask"].shape, (2, 4))
        self.assertEqual(batch["density_target_8s"].shape, (2, 400, 1))

    def test_control_teacher_slice_batch_prepares_four_aligned_control_contexts(self) -> None:
        vocab = MapperV1Vocab()
        sample = _sample(encode_mapper_window([], vocab=vocab, write_start_ms=0, write_end_ms=8000))
        sample["full_mel"] = torch.zeros(500, 160, dtype=torch.float32)
        sample["full_dense_timing_v2"] = torch.zeros(500, 4, dtype=torch.float32)
        sample["frame_count"] = torch.tensor(500, dtype=torch.long)
        sample["control_slice_start_frames"] = torch.tensor([0, 100, 200, 300], dtype=torch.long)
        batch = collate_mapper_v1_windows([sample], pad_id=vocab.pad_id)

        control_batch = control_teacher_slice_batch(batch, 3)

        self.assertEqual(control_batch["target_start_frame"].tolist(), [300])
        self.assertEqual(control_batch["context_mel"].shape, (1, 600, 160))
        self.assertEqual(control_batch["context_dense_timing_v2"].shape, (1, 600, 4))

    def test_concatenate_density_teacher_8s_requires_four_two_second_outputs(self) -> None:
        density_index = VALUE_FEATURE_NAMES.index("density_level")
        outputs = []
        for index in range(4):
            value_pred = torch.zeros(2, 100, len(VALUE_FEATURE_NAMES), dtype=torch.float32)
            value_pred[:, :, density_index] = float(index)
            outputs.append(SimpleNamespace(value_pred=value_pred))

        density_teacher = concatenate_density_teacher_8s(outputs)

        self.assertEqual(density_teacher.shape, (2, 400, 1))
        self.assertTrue(torch.equal(density_teacher[:, :100], torch.zeros(2, 100, 1)))
        self.assertTrue(torch.equal(density_teacher[:, 300:], torch.full((2, 100, 1), 3.0)))

    def test_mapper_dataset_filters_unsupported_same_lane_compound_windows(self) -> None:
        records = [
            _record("compound.osu", difficulty=3.0),
            _record("valid.osu", difficulty=4.0),
        ]

        dataset = _MapperDatasetWithUnsupportedActions(records, unsupported_paths={"compound.osu"})

        self.assertEqual(len(dataset.records), 1)
        self.assertEqual(dataset.records[0].control_record.beatmap_path, Path("valid.osu"))
        self.assertEqual(dataset.filter_report.num_total_windows, 2)
        self.assertEqual(dataset.filter_report.num_mapper_eligible_windows, 1)
        self.assertEqual(dataset.filter_report.num_dropped_cross_window_ln_windows, 0)
        self.assertEqual(dataset.filter_report.num_dropped_unsupported_action_windows, 1)
        self.assertEqual(dataset.filter_report.drop_rate, 0.5)
        self.assertEqual(dataset.filter_report.unsupported_action_drop_rate, 0.5)
        self.assertEqual(dataset.filter_report.drop_rate_by_difficulty["3.00"], 1.0)
        self.assertEqual(dataset.filter_report.drop_rate_by_difficulty["4.00"], 0.0)

    def test_control_teacher_cache_hit_skips_full_control_inputs_and_collates_teacher(self) -> None:
        record = _record("cached.osu", difficulty=4.0)
        control_memory = torch.arange(400 * 3, dtype=torch.float32).reshape(400, 3)
        density_teacher = torch.linspace(0.0, 1.0, 400, dtype=torch.float32).reshape(400, 1)

        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = control_teacher_cache_path(temp_dir, record)
            save_control_teacher_cache_entry(
                cache_path,
                record=record,
                control_memory_8s=control_memory,
                density_teacher_8s=density_teacher,
            )
            loaded = load_control_teacher_cache_entry(cache_path, record=record)
            self.assertTrue(torch.equal(loaded["control_memory_8s"], control_memory))
            self.assertTrue(torch.equal(loaded["density_teacher_8s"], density_teacher))

            dataset = _MapperDatasetWithControlTeacherCache([record], cache_dir=Path(temp_dir))
            sample = dataset[0]

            self.assertTrue(torch.equal(sample["control_memory_8s"], control_memory))
            self.assertTrue(torch.equal(sample["density_teacher_8s"], density_teacher))
            self.assertNotIn("full_mel", sample)
            batch = collate_mapper_v1_windows([sample], pad_id=MapperV1Vocab().pad_id)
            self.assertEqual(batch["control_memory_8s"].shape, (1, 400, 3))
            self.assertEqual(batch["density_teacher_8s"].shape, (1, 400, 1))
            self.assertNotIn("full_mel", batch)


def _sample(tokenized) -> dict[str, Any]:
    return {
        "mel_context": torch.zeros(400, 160, dtype=torch.float32),
        "timing_context": torch.zeros(400, 4, dtype=torch.float32),
        "context_padding_mask": torch.zeros(400, dtype=torch.bool),
        "difficulty": torch.zeros(1, dtype=torch.float32),
        "decoder_input_tokens": tokenized.decoder_input_tensor(),
        "target_fragment_tokens": tokenized.target_fragment_tensor(),
        "target_fragment_states": {
            "current_ms": tokenized.target_fragment_current_ms,
            "open_mask": tokenized.target_fragment_open_mask,
            "open_start_ms": tokenized.target_fragment_open_start_ms,
            "open_age_ms": tokenized.target_fragment_open_age_ms,
        },
        "ln_carry_in": ln_carry_state_tensors(tokenized.ln_carry_in),
        "ln_carry_out": ln_carry_state_tensors(tokenized.ln_carry_out),
        "close_labels": tokenized.close_labels,
        "close_label_mask": tokenized.close_label_mask,
        "density_target_8s": torch.zeros(400, 1, dtype=torch.float32),
        "density_confidence_8s": torch.ones(400, 1, dtype=torch.float32),
        "write_start_ms": torch.tensor(tokenized.write_start_ms, dtype=torch.long),
        "write_end_ms": torch.tensor(tokenized.write_end_ms, dtype=torch.long),
        "is_full_chart_start": torch.tensor(tokenized.is_full_chart_start, dtype=torch.bool),
        "is_full_chart_end": torch.tensor(tokenized.is_full_chart_end, dtype=torch.bool),
    }


def _record(beatmap_path: str, *, difficulty: float) -> ControlWindowRecord:
    return ControlWindowRecord(
        beatmap_path=Path(beatmap_path),
        audio_path=Path(f"{beatmap_path}.mp3"),
        difficulty=difficulty,
        frame_count=400,
        target_start_frame=0,
    )


class _MapperDatasetWithUnsupportedActions(MapperV1WindowDataset):
    def __init__(self, records: list[ControlWindowRecord], *, unsupported_paths: set[str]) -> None:
        self.unsupported_paths = unsupported_paths
        super().__init__(control_dataset=SimpleNamespace(records=records))

    def _load_timepoints(self, beatmap_path: Path) -> tuple:
        if beatmap_path.as_posix() in self.unsupported_paths:
            return tuple(
                hitobjects_to_mapper_timepoints(
                    [
                        ManiaHitObject(1000.0, 1000.0, 0, ManiaHitObjectKind.TAP),
                        ManiaHitObject(1000.0, 1200.0, 0, ManiaHitObjectKind.HOLD),
                    ],
                ),
            )
        return ()


class _RaisingControlDataset:
    def __init__(self, records: list[ControlWindowRecord]) -> None:
        self.records = records

    def __getitem__(self, index: int):
        raise AssertionError("control dataset should not be read on cache hit")


class _MapperDatasetWithControlTeacherCache(MapperV1WindowDataset):
    def __init__(self, records: list[ControlWindowRecord], *, cache_dir: Path) -> None:
        super().__init__(control_dataset=_RaisingControlDataset(records), control_teacher_cache_dir=cache_dir)

    def _load_timepoints(self, beatmap_path: Path) -> tuple:
        return ()


if __name__ == "__main__":
    unittest.main()
