from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from train.stage_2.data.control_windows import (
    CONTEXT_LENGTH_FRAMES,
    ControlWindowDataset,
    DEFAULT_MAX_CACHED_MAPS,
    TARGET_OFFSET_IN_CONTEXT,
    build_control_window_index,
    collate_control_context_windows,
    collate_control_windows,
    normalize_difficulty,
    target_valid_mask,
)
from train.stage_2.features.control_v3_targets import (
    CONFIDENCE_FEATURE_NAMES,
    LN_CHANGE_N_EFF_FEATURE_NAME,
    MODEL_FEATURE_NAMES,
)


def _index_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "shard": "0",
                "audio_path": "100/audio.mp3",
                "beatmap_path": "100/easy.osu",
                "difficulty": 1.99,
                "beatmap_id": 1,
            },
            {
                "shard": "0",
                "audio_path": "100/audio.mp3",
                "beatmap_path": "100/normal.osu",
                "difficulty": 2.5,
                "beatmap_id": 2,
            },
            {
                "shard": "0",
                "audio_path": "200/audio.mp3",
                "beatmap_path": "200/hard.osu",
                "difficulty": 6.0,
                "beatmap_id": 3,
            },
            {
                "shard": "0",
                "audio_path": "300/audio.mp3",
                "beatmap_path": "300/extra.osu",
                "difficulty": 6.01,
                "beatmap_id": 4,
            },
        ]
    )


def _mel_loader(audio_path: Path) -> np.ndarray:
    frame_count = 250 if "100/audio.mp3" in audio_path.as_posix() else 75
    return np.ones((frame_count, 160), dtype=np.float32)


def _timing_loader(_beatmap_path: Path, frame_count: int) -> np.ndarray:
    return np.full((frame_count, 4), 0.5, dtype=np.float32)


def _target_loader(record) -> np.ndarray:
    target = np.zeros((100, 20), dtype=np.float32)
    target[:, 0] = float(record.target_start_frame)
    return target


class TrainStage2ControlWindowTests(unittest.TestCase):
    def test_dataset_builds_two_second_records_and_sample_tensors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "index.parquet"
            _index_frame().to_parquet(index_path, index=False)

            dataset = ControlWindowDataset(
                index_path=index_path,
                dataset_root=Path(tmpdir) / "mania-dataset",
                mel_loader=_mel_loader,
                timing_loader=_timing_loader,
                target_loader=_target_loader,
                allow_missing_ln_change_n_eff_target=True,
            )

            self.assertEqual(dataset.source_map_count, 4)
            self.assertEqual(dataset.filtered_map_count, 2)
            self.assertEqual(len(dataset), 4)
            self.assertEqual([record.target_start_frame for record in dataset.records], [0, 100, 200, 0])

            sample = dataset[2]
            self.assertEqual(sample["full_mel"].shape, (250, 160))
            self.assertEqual(sample["full_dense_timing_v2"].shape, (250, 4))
            self.assertEqual(sample["control_v3_target"].shape, (100, 20))
            self.assertEqual(sample["ln_change_n_eff_target"].shape, (100,))
            self.assertTrue(torch.equal(sample["ln_change_n_eff_target"], torch.full((100,), 3.0)))
            self.assertEqual(sample["target_valid_mask"].shape, (100,))
            self.assertEqual(sample["target_valid_mask"].dtype, torch.bool)
            self.assertTrue(sample["target_valid_mask"][:50].all())
            self.assertFalse(sample["target_valid_mask"][50:].any())
            self.assertEqual(sample["difficulty"].dtype, torch.float32)
            self.assertEqual(sample["normalized_difficulty"].dtype, torch.float32)
            self.assertAlmostEqual(float(sample["difficulty"].item()), 2.5)
            self.assertAlmostEqual(float(sample["normalized_difficulty"].item()), -0.75)
            self.assertEqual(int(sample["target_start_frame"].item()), 200)
            self.assertEqual(int(sample["target_start_ms"].item()), 4000)
            self.assertEqual(int(sample["beatmap_id"].item()), 2)
            self.assertTrue(sample["audio_path"].endswith("mania-dataset/0/100/audio.mp3"))
            self.assertTrue(sample["beatmap_path"].endswith("mania-dataset/0/100/normal.osu"))

    def test_collate_pads_full_song_features_and_returns_padding_mask(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "index.parquet"
            _index_frame().to_parquet(index_path, index=False)
            dataset = ControlWindowDataset(
                index_path=index_path,
                dataset_root=Path(tmpdir) / "mania-dataset",
                mel_loader=_mel_loader,
                timing_loader=_timing_loader,
                target_loader=_target_loader,
                allow_missing_ln_change_n_eff_target=True,
            )

            batch = collate_control_windows([dataset[0], dataset[3]])

            self.assertEqual(batch["full_mel"].shape, (2, 250, 160))
            self.assertEqual(batch["full_dense_timing_v2"].shape, (2, 250, 4))
            self.assertEqual(batch["control_v3_target"].shape, (2, 100, 20))
            self.assertEqual(batch["ln_change_n_eff_target"].shape, (2, 100))
            self.assertTrue(torch.equal(batch["ln_change_n_eff_target"], torch.full((2, 100), 3.0)))
            self.assertEqual(batch["target_valid_mask"].shape, (2, 100))
            self.assertEqual(batch["target_valid_mask"].dtype, torch.bool)
            self.assertTrue(batch["target_valid_mask"][0].all())
            self.assertTrue(batch["target_valid_mask"][1, :75].all())
            self.assertFalse(batch["target_valid_mask"][1, 75:].any())
            self.assertFalse(batch["padding_mask"][0].any())
            self.assertFalse(batch["padding_mask"][1, :75].any())
            self.assertTrue(batch["padding_mask"][1, 75:].all())
            self.assertEqual(batch["frame_count"].tolist(), [250, 75])
            self.assertEqual(batch["target_start_ms"].tolist(), [0, 0])

            loader = DataLoader(dataset, batch_size=2, collate_fn=collate_control_windows)
            loader_batch = next(iter(loader))
            self.assertEqual(loader_batch["full_mel"].shape, (2, 250, 160))

    def test_context_collate_slices_fixed_context_without_full_song_padding(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "index.parquet"
            _index_frame().to_parquet(index_path, index=False)
            dataset = ControlWindowDataset(
                index_path=index_path,
                dataset_root=Path(tmpdir) / "mania-dataset",
                mel_loader=_mel_loader,
                timing_loader=_timing_loader,
                target_loader=_target_loader,
                allow_missing_ln_change_n_eff_target=True,
            )

            batch = collate_control_context_windows([dataset[0], dataset[2]])

            self.assertNotIn("full_mel", batch)
            self.assertNotIn("full_dense_timing_v2", batch)
            self.assertEqual(batch["context_mel"].shape, (2, CONTEXT_LENGTH_FRAMES, 160))
            self.assertEqual(batch["context_dense_timing_v2"].shape, (2, CONTEXT_LENGTH_FRAMES, 4))
            self.assertEqual(batch["context_start_frame"].tolist(), [-TARGET_OFFSET_IN_CONTEXT, -50])
            self.assertTrue(batch["context_padding_mask"][0, :TARGET_OFFSET_IN_CONTEXT].all())
            self.assertFalse(batch["context_padding_mask"][0, TARGET_OFFSET_IN_CONTEXT:500].any())
            self.assertTrue(batch["context_padding_mask"][0, 500:].all())
            self.assertTrue(batch["context_padding_mask"][1, :50].all())
            self.assertFalse(batch["context_padding_mask"][1, 50:300].any())
            self.assertTrue(batch["context_padding_mask"][1, 300:].all())

    def test_context_collate_recomputes_target_valid_mask(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "index.parquet"
            _index_frame().to_parquet(index_path, index=False)
            dataset = ControlWindowDataset(
                index_path=index_path,
                dataset_root=Path(tmpdir) / "mania-dataset",
                mel_loader=_mel_loader,
                timing_loader=_timing_loader,
                target_loader=_target_loader,
                allow_missing_ln_change_n_eff_target=True,
            )
            stale_tail_sample = dataset[2]
            stale_tail_sample["target_valid_mask"] = torch.ones(100, dtype=torch.bool)

            batch = collate_control_context_windows([stale_tail_sample])

            self.assertTrue(batch["target_valid_mask"][0, :50].all())
            self.assertFalse(batch["target_valid_mask"][0, 50:].any())

    def test_context_collate_rejects_invalid_frame_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "index.parquet"
            _index_frame().to_parquet(index_path, index=False)
            dataset = ControlWindowDataset(
                index_path=index_path,
                dataset_root=Path(tmpdir) / "mania-dataset",
                mel_loader=_mel_loader,
                timing_loader=_timing_loader,
                target_loader=_target_loader,
                allow_missing_ln_change_n_eff_target=True,
            )
            base_sample = dataset[0]
            cases = [
                ("fractional_target_start_frame", {"target_start_frame": torch.tensor(0.9)}, "target_start_frame"),
                ("negative_target_start_frame", {"target_start_frame": torch.tensor(-1)}, "target_start_frame"),
                ("target_start_frame_past_end", {"target_start_frame": torch.tensor(250)}, "target_start_frame"),
                ("zero_frame_count", {"frame_count": torch.tensor(0)}, "frame_count"),
                ("target_start_ms_mismatch", {"target_start_ms": torch.tensor(20)}, "target_start_ms"),
            ]

            for name, updates, message in cases:
                with self.subTest(name=name):
                    sample = dict(base_sample)
                    sample.update(updates)
                    with self.assertRaisesRegex(ValueError, message):
                        collate_control_context_windows([sample])

    def test_validates_target_and_timing_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "index.parquet"
            _index_frame().iloc[[1]].to_parquet(index_path, index=False)

            def bad_timing_loader(_beatmap_path: Path, _frame_count: int) -> np.ndarray:
                return np.zeros((10, 4), dtype=np.float32)

            dataset = ControlWindowDataset(
                index_path=index_path,
                dataset_root=Path(tmpdir) / "mania-dataset",
                mel_loader=_mel_loader,
                timing_loader=bad_timing_loader,
                target_loader=_target_loader,
                allow_missing_ln_change_n_eff_target=True,
            )
            with self.assertRaisesRegex(ValueError, "full_dense_timing_v2"):
                dataset[0]

            def bad_target_loader(_record) -> np.ndarray:
                return np.zeros((99, 20), dtype=np.float32)

            dataset = ControlWindowDataset(
                index_path=index_path,
                dataset_root=Path(tmpdir) / "mania-dataset",
                mel_loader=_mel_loader,
                timing_loader=_timing_loader,
                target_loader=bad_target_loader,
                allow_missing_ln_change_n_eff_target=True,
            )
            with self.assertRaisesRegex(ValueError, "control_v3_target"):
                dataset[0]

    def test_custom_target_loader_requires_explicit_ln_change_sidecar_or_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "index.parquet"
            _index_frame().iloc[[1]].to_parquet(index_path, index=False)

            with self.assertRaisesRegex(ValueError, "ln_change_n_eff_target_loader"):
                ControlWindowDataset(
                    index_path=index_path,
                    dataset_root=Path(tmpdir) / "mania-dataset",
                    mel_loader=_mel_loader,
                    timing_loader=_timing_loader,
                    target_loader=_target_loader,
                )

    def test_default_target_loader_reads_control_v3_artifact_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "index.parquet"
            _index_frame().iloc[[1]].to_parquet(index_path, index=False)
            timeseries_path = Path(tmpdir) / "control_v3.parquet"
            _timeseries_frame(beatmap_id=2, times=np.arange(0.0, 5.1, 0.1).tolist()).to_parquet(
                timeseries_path,
                index=False,
            )

            dataset = ControlWindowDataset(
                index_path=index_path,
                dataset_root=Path(tmpdir) / "mania-dataset",
                mel_loader=_mel_loader,
                timing_loader=_timing_loader,
                control_v3_timeseries_path=timeseries_path,
            )

            sample = dataset[1]
            self.assertEqual(sample["control_v3_target"].shape, (100, 20))
            self.assertEqual(sample["ln_change_n_eff_target"].shape, (100,))
            self.assertAlmostEqual(float(sample["ln_change_n_eff_target"][0].item()), 5.01, places=5)
            self.assertAlmostEqual(float(sample["control_v3_target"][0, 0].item()), 4.01, places=5)
            self.assertEqual(int(sample["beatmap_id"].item()), 2)

    def test_default_control_v3_rows_cache_uses_dataset_cache_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "index.parquet"
            _index_frame().iloc[[1]].to_parquet(index_path, index=False)
            fake_rows = pd.DataFrame({"time": [0.0]})

            dataset = ControlWindowDataset(
                index_path=index_path,
                dataset_root=Path(tmpdir) / "mania-dataset",
                mel_loader=_mel_loader,
                timing_loader=_timing_loader,
                target_loader=_target_loader,
                allow_missing_ln_change_n_eff_target=True,
                max_cached_maps=0,
            )
            with patch("train.stage_2.data.control_windows._default_control_v3_rows", return_value=fake_rows) as rows_loader:
                dataset._load_control_v3_rows(("filtered_index", 1))
                dataset._load_control_v3_rows(("filtered_index", 1))

            self.assertEqual(rows_loader.call_count, 2)
            self.assertEqual(len(dataset._control_v3_rows_cache), 0)

            cached_dataset = ControlWindowDataset(
                index_path=index_path,
                dataset_root=Path(tmpdir) / "mania-dataset",
                mel_loader=_mel_loader,
                timing_loader=_timing_loader,
                target_loader=_target_loader,
                allow_missing_ln_change_n_eff_target=True,
            )
            with patch("train.stage_2.data.control_windows._default_control_v3_rows", return_value=fake_rows) as rows_loader:
                cached_dataset._load_control_v3_rows(("filtered_index", 1))
                cached_dataset._load_control_v3_rows(("filtered_index", 1))

            self.assertEqual(rows_loader.call_count, 1)
            self.assertEqual(len(cached_dataset._control_v3_rows_cache), 1)
            self.assertEqual(cached_dataset.max_cached_maps, DEFAULT_MAX_CACHED_MAPS)

    def test_default_control_v3_rows_cache_evicts_lru_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "index.parquet"
            _index_frame().iloc[[1]].to_parquet(index_path, index=False)
            dataset = ControlWindowDataset(
                index_path=index_path,
                dataset_root=Path(tmpdir) / "mania-dataset",
                mel_loader=_mel_loader,
                timing_loader=_timing_loader,
                target_loader=_target_loader,
                allow_missing_ln_change_n_eff_target=True,
                max_cached_maps=2,
            )
            with patch(
                "train.stage_2.data.control_windows._default_control_v3_rows",
                side_effect=lambda _path, selector: pd.DataFrame({"selector": [selector[1]]}),
            ) as rows_loader:
                dataset._load_control_v3_rows(("filtered_index", 1))
                dataset._load_control_v3_rows(("filtered_index", 2))
                dataset._load_control_v3_rows(("filtered_index", 3))
                dataset._load_control_v3_rows(("filtered_index", 1))

            self.assertEqual(rows_loader.call_count, 4)
            self.assertEqual(list(dataset._control_v3_rows_cache), [("filtered_index", 3), ("filtered_index", 1)])

    def test_default_target_loader_prefers_filtered_index_over_duplicated_beatmap_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "index.parquet"
            pd.DataFrame(
                [
                    {
                        "shard": "0",
                        "audio_path": "100/audio.mp3",
                        "beatmap_path": "100/normal.osu",
                        "difficulty": 2.5,
                        "beatmap_id": 9,
                        "filtered_index": 111,
                        "source_index": 211,
                    }
                ]
            ).to_parquet(index_path, index=False)
            timeseries_path = Path(tmpdir) / "control_v3.parquet"
            pd.concat(
                [
                    _timeseries_frame(
                        beatmap_id=9,
                        times=np.arange(0.0, 2.1, 0.1).tolist(),
                        filtered_index=111,
                        value_offset=10.0,
                    ),
                    _timeseries_frame(
                        beatmap_id=9,
                        times=np.arange(0.0, 2.1, 0.1).tolist(),
                        filtered_index=222,
                        value_offset=20.0,
                    ),
                ],
                ignore_index=True,
            ).to_parquet(timeseries_path, index=False)

            dataset = ControlWindowDataset(
                index_path=index_path,
                dataset_root=Path(tmpdir) / "mania-dataset",
                mel_loader=_mel_loader,
                timing_loader=_timing_loader,
                control_v3_timeseries_path=timeseries_path,
            )

            sample = dataset[0]
            self.assertAlmostEqual(float(sample["control_v3_target"][0, 0].item()), 10.01, places=5)
            self.assertEqual(int(sample["filtered_index"].item()), 111)

    def test_build_control_window_index_writes_preexpanded_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_index_path = root / "source.parquet"
            summary_path = root / "summary.parquet"
            output_path = root / "stage2_windows.parquet"
            report_path = root / "stage2_windows.json"
            _index_frame().to_parquet(source_index_path, index=False)
            _summary_frame().to_parquet(summary_path, index=False)
            init_mel_calls: list[Path] = []

            def frame_count_loader(audio_path: Path) -> int:
                return 250 if "100/audio.mp3" in audio_path.as_posix() else 75

            report = build_control_window_index(
                source_index_path=source_index_path,
                dataset_root=root / "mania-dataset",
                control_v3_summary_path=summary_path,
                output_path=output_path,
                report_path=report_path,
                frame_count_loader=frame_count_loader,
            )
            window_df = pd.read_parquet(output_path)

            self.assertEqual(report.source_map_count, 4)
            self.assertEqual(report.retained_map_count, 2)
            self.assertEqual(report.unique_audio_count, 2)
            self.assertEqual(report.window_count, 4)
            self.assertTrue(report_path.exists())
            self.assertEqual(window_df["target_start_frame"].tolist(), [0, 100, 200, 0])
            self.assertEqual(window_df["target_start_ms"].tolist(), [0, 2000, 4000, 0])
            self.assertEqual(window_df["filtered_index"].tolist(), [11, 11, 11, 12])
            self.assertEqual(window_df["source_index"].tolist(), [21, 21, 21, 22])

            def init_safe_mel_loader(audio_path: Path) -> np.ndarray:
                init_mel_calls.append(audio_path)
                return _mel_loader(audio_path)

            dataset = ControlWindowDataset(
                index_path=output_path,
                dataset_root=root / "mania-dataset",
                mel_loader=init_safe_mel_loader,
                timing_loader=_timing_loader,
                target_loader=_target_loader,
                allow_missing_ln_change_n_eff_target=True,
            )
            self.assertEqual(init_mel_calls, [])
            self.assertEqual(len(dataset), 4)
            sample = dataset[2]
            self.assertEqual(init_mel_calls, [root / "mania-dataset" / "0" / "100" / "audio.mp3"])
            self.assertEqual(int(sample["target_start_frame"].item()), 200)
            self.assertEqual(int(sample["filtered_index"].item()), 11)

    def test_normalize_difficulty_requires_supported_range(self) -> None:
        self.assertEqual(normalize_difficulty(2.0), -1.0)
        self.assertEqual(normalize_difficulty(4.0), 0.0)
        self.assertEqual(normalize_difficulty(6.0), 1.0)
        with self.assertRaisesRegex(ValueError, "difficulty outside"):
            normalize_difficulty(6.01)

    def test_target_valid_mask_marks_only_in_song_target_frames(self) -> None:
        full = target_valid_mask(target_start_frame=0, frame_count=250)
        self.assertTrue(full.all())

        tail = target_valid_mask(target_start_frame=200, frame_count=250)
        self.assertEqual(int(tail.sum().item()), 50)
        self.assertTrue(tail[:50].all())
        self.assertFalse(tail[50:].any())

        short = target_valid_mask(target_start_frame=0, frame_count=75)
        self.assertEqual(int(short.sum().item()), 75)
        self.assertTrue(short[:75].all())
        self.assertFalse(short[75:].any())

    def test_rejects_index_paths_that_escape_shard_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "index.parquet"
            pd.DataFrame(
                [
                    {
                        "shard": "0",
                        "audio_path": "../escape/audio.mp3",
                        "beatmap_path": "100/map.osu",
                        "difficulty": 4.0,
                    }
                ]
            ).to_parquet(index_path, index=False)

            with self.assertRaisesRegex(ValueError, "parent traversal"):
                ControlWindowDataset(
                    index_path=index_path,
                    dataset_root=Path(tmpdir) / "mania-dataset",
                    mel_loader=_mel_loader,
                    timing_loader=_timing_loader,
                    target_loader=_target_loader,
                    allow_missing_ln_change_n_eff_target=True,
                )

            pd.DataFrame(
                [
                    {
                        "shard": "0",
                        "audio_path": "/tmp/audio.mp3",
                        "beatmap_path": "100/map.osu",
                        "difficulty": 4.0,
                    }
                ]
            ).to_parquet(index_path, index=False)

            with self.assertRaisesRegex(ValueError, "relative"):
                ControlWindowDataset(
                    index_path=index_path,
                    dataset_root=Path(tmpdir) / "mania-dataset",
                    mel_loader=_mel_loader,
                    timing_loader=_timing_loader,
                    target_loader=_target_loader,
                    allow_missing_ln_change_n_eff_target=True,
                )

            pd.DataFrame(
                [
                    {
                        "shard": "../outside",
                        "audio_path": "audio.mp3",
                        "beatmap_path": "map.osu",
                        "difficulty": 4.0,
                    }
                ]
            ).to_parquet(index_path, index=False)

            with self.assertRaisesRegex(ValueError, "shard"):
                ControlWindowDataset(
                    index_path=index_path,
                    dataset_root=Path(tmpdir) / "mania-dataset",
                    mel_loader=_mel_loader,
                    timing_loader=_timing_loader,
                    target_loader=_target_loader,
                    allow_missing_ln_change_n_eff_target=True,
                )

    def test_rejects_nonfinite_feature_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "index.parquet"
            _index_frame().iloc[[1]].to_parquet(index_path, index=False)

            def bad_mel_loader(_audio_path: Path) -> np.ndarray:
                mel = np.ones((100, 160), dtype=np.float32)
                mel[0, 0] = np.nan
                return mel

            with self.assertRaisesRegex(ValueError, "finite"):
                ControlWindowDataset(
                    index_path=index_path,
                    dataset_root=Path(tmpdir) / "mania-dataset",
                    mel_loader=bad_mel_loader,
                    timing_loader=_timing_loader,
                    target_loader=_target_loader,
                    allow_missing_ln_change_n_eff_target=True,
                )

    def test_rejects_out_of_range_control_target_confidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "index.parquet"
            _index_frame().iloc[[1]].to_parquet(index_path, index=False)

            def bad_target_loader(_record) -> np.ndarray:
                target = np.zeros((100, 20), dtype=np.float32)
                target[:, MODEL_FEATURE_NAMES.index(CONFIDENCE_FEATURE_NAMES[0])] = 2.0
                return target

            dataset = ControlWindowDataset(
                index_path=index_path,
                dataset_root=Path(tmpdir) / "mania-dataset",
                mel_loader=_mel_loader,
                timing_loader=_timing_loader,
                target_loader=bad_target_loader,
                allow_missing_ln_change_n_eff_target=True,
            )
            with self.assertRaisesRegex(ValueError, "confidence"):
                dataset[0]


def _summary_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "filtered_index": 11,
                "source_index": 21,
                "beatmap_id": 2,
                "beatmap_set_id": 100,
                "difficulty": 2.5,
                "shard": "0",
                "beatmap_path": "100/normal.osu",
                "finite": True,
                "ranges_ok": True,
                "error_type": "",
            },
            {
                "filtered_index": 12,
                "source_index": 22,
                "beatmap_id": 3,
                "beatmap_set_id": 200,
                "difficulty": 6.0,
                "shard": "0",
                "beatmap_path": "200/hard.osu",
                "finite": True,
                "ranges_ok": True,
                "error_type": "",
            },
        ]
    )


def _timeseries_frame(
    *,
    beatmap_id: int,
    times: list[float],
    filtered_index: int | None = None,
    value_offset: float = 2.0,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "filtered_index": np.full(
                len(times),
                beatmap_id + 10 if filtered_index is None else filtered_index,
                dtype=np.int32,
            ),
            "source_index": np.full(len(times), beatmap_id + 20, dtype=np.int32),
            "beatmap_id": np.full(len(times), beatmap_id, dtype=np.int64),
            "beatmap_set_id": np.full(len(times), beatmap_id + 30, dtype=np.int64),
            "difficulty": np.full(len(times), 4.0, dtype=np.float32),
            "time_s": np.asarray(times, dtype=np.float32),
        }
    )
    for column_index, name in enumerate(MODEL_FEATURE_NAMES):
        if name in CONFIDENCE_FEATURE_NAMES:
            frame[name] = np.full(len(times), 0.5, dtype=np.float32)
        else:
            frame[name] = np.asarray(times, dtype=np.float32) + np.float32(value_offset + column_index)
    frame[LN_CHANGE_N_EFF_FEATURE_NAME] = np.asarray(times, dtype=np.float32) + np.float32(3.0)
    return frame


if __name__ == "__main__":
    unittest.main()
