import math
import struct
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from train.stage1_oracle.data.dataset import ManiaBeatmapDataset
from train.stage1_oracle.data.dataset import build_4k_index
from train.stage1_oracle.data.dataset import build_4k_no_timing_anomaly_index


def _write_wav(path: Path, *, sample_rate: int = 22050, duration_seconds: float = 0.1, frequency: float = 440.0) -> None:
    num_samples = int(sample_rate * duration_seconds)
    amplitude = 16000
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        for index in range(num_samples):
            sample = int(amplitude * math.sin(2 * math.pi * frequency * index / sample_rate))
            wav_file.writeframes(struct.pack("<h", sample))


def _write_osu(
    path: Path,
    *,
    audio_filename: str,
    beatmap_set_id: int,
    beatmap_id: int,
    version: str,
    title: str = "Test Song",
    artist: str = "Test Artist",
    creator: str = "Test Creator",
    mode: int = 3,
    audio_lead_in: int = 0,
    preview_time: int = -1,
    circle_size: float = 4.0,
    overall_difficulty: float = 8.0,
    hp_drain_rate: float = 6.5,
    timing_lines: list[str] | None = None,
) -> None:
    lines = [
        "osu file format v14",
        "",
        "[General]",
        f"AudioFilename: {audio_filename}",
        f"AudioLeadIn: {audio_lead_in}",
        f"PreviewTime: {preview_time}",
        f"Mode: {mode}",
        "",
        "[Metadata]",
        f"Title:{title}",
        f"Artist:{artist}",
        f"Creator:{creator}",
        f"Version:{version}",
        f"BeatmapID:{beatmap_id}",
        f"BeatmapSetID:{beatmap_set_id}",
        "",
        "[Difficulty]",
        f"HPDrainRate:{hp_drain_rate}",
        f"CircleSize:{circle_size}",
        f"OverallDifficulty:{overall_difficulty}",
        "",
    ]
    if timing_lines is not None:
        lines.extend(["[TimingPoints]", *timing_lines, ""])
    lines.append("[HitObjects]")
    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


class TrainDatasetTests(unittest.TestCase):
    def test_build_4k_index_in_train_keeps_only_4k_maps(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            dataset_root = repo_root / "mania-dataset"
            shard_path = dataset_root / "0"
            train_path = repo_root / "train"
            song_path = shard_path / "1234"
            song_path.mkdir(parents=True)
            train_path.mkdir(parents=True)

            _write_wav(song_path / "main_audio.wav")
            _write_osu(
                song_path / "map_4k.osu",
                audio_filename="main_audio.wav",
                beatmap_set_id=1234,
                beatmap_id=1,
                version="4K",
                circle_size=4.0,
            )
            _write_osu(
                song_path / "map_7k.osu",
                audio_filename="main_audio.wav",
                beatmap_set_id=1234,
                beatmap_id=2,
                version="7K",
                circle_size=7.0,
            )

            four_k_index_path = build_4k_index(shard_path, train_path / "beatmap_index_4k.parquet")

            self.assertEqual(four_k_index_path, train_path / "beatmap_index_4k.parquet")
            self.assertTrue(four_k_index_path.exists())

            index_df = pd.read_parquet(four_k_index_path)
            self.assertEqual(len(index_df), 1)
            self.assertEqual(set(index_df["key_count"]), {4})
            self.assertEqual(set(index_df["beatmap_filename"]), {"map_4k.osu"})

            dataset = ManiaBeatmapDataset(
                shard_path=shard_path,
                sample_rate=16000,
                index_path=four_k_index_path,
                build_index_if_missing=False,
            )
            self.assertEqual(len(dataset), 1)
            sample = dataset[0]
            self.assertEqual(sample["beatmap_filename"], "map_4k.osu")
            self.assertEqual(sample["key_count"], 4)

    def test_build_4k_index_writes_expected_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            dataset_root = repo_root / "mania-dataset"
            shard_path = dataset_root / "0"
            train_path = repo_root / "train"
            song_path = shard_path / "1234"
            song_path.mkdir(parents=True)
            train_path.mkdir(parents=True)

            _write_wav(song_path / "main_audio.wav")
            _write_wav(song_path / "hitsound.wav", frequency=880.0)
            _write_osu(
                song_path / "map_a.osu",
                audio_filename="main_audio.wav",
                beatmap_set_id=1234,
                beatmap_id=1,
                version="Easy",
                title="Indexed Song",
                artist="Indexed Artist",
                creator="Indexed Creator",
                preview_time=12345,
                circle_size=4.0,
                overall_difficulty=7.5,
            )
            _write_osu(
                song_path / "map_b.osu",
                audio_filename="main_audio.wav",
                beatmap_set_id=1234,
                beatmap_id=2,
                version="Hard",
                title="Indexed Song",
                artist="Indexed Artist",
                creator="Indexed Creator",
                preview_time=12345,
                circle_size=7.0,
                overall_difficulty=8.2,
            )

            index_path = build_4k_index(shard_path, train_path / "beatmap_index_4k.parquet")

            self.assertEqual(index_path, train_path / "beatmap_index_4k.parquet")
            self.assertTrue(index_path.exists())

            index_df = pd.read_parquet(index_path)
            self.assertEqual(len(index_df), 1)
            self.assertEqual(set(index_df["audio_path"]), {"1234/main_audio.wav"})
            self.assertEqual(set(index_df["beatmap_path"]), {"1234/map_a.osu"})
            self.assertTrue((index_df["shard"] == "0").all())
            self.assertTrue((index_df["beatmap_set_id"] == 1234).all())
            self.assertEqual(
                {
                    "mode",
                    "audio_lead_in",
                    "preview_time",
                    "title",
                    "artist",
                    "creator",
                    "version",
                    "beatmap_id",
                    "hp_drain_rate",
                    "circle_size",
                    "overall_difficulty",
                    "key_count",
                    "difficulty",
                    "sr_difficulties",
                },
                {
                    "mode",
                    "audio_lead_in",
                    "preview_time",
                    "title",
                    "artist",
                    "creator",
                    "version",
                    "beatmap_id",
                    "hp_drain_rate",
                    "circle_size",
                    "overall_difficulty",
                    "key_count",
                    "difficulty",
                    "sr_difficulties",
                }.intersection(index_df.columns),
            )
            easy_row = index_df[index_df["beatmap_filename"] == "map_a.osu"].iloc[0]
            self.assertEqual(easy_row["mode"], 3)
            self.assertEqual(easy_row["audio_lead_in"], 0)
            self.assertEqual(easy_row["preview_time"], 12345)
            self.assertEqual(easy_row["title"], "Indexed Song")
            self.assertEqual(easy_row["artist"], "Indexed Artist")
            self.assertEqual(easy_row["creator"], "Indexed Creator")
            self.assertEqual(easy_row["version"], "Easy")
            self.assertEqual(easy_row["beatmap_id"], 1)
            self.assertEqual(easy_row["circle_size"], 4.0)
            self.assertEqual(easy_row["overall_difficulty"], 7.5)
            self.assertEqual(easy_row["key_count"], 4)
            self.assertEqual(easy_row["difficulty"], 0.0)
            self.assertEqual(list(easy_row["sr_difficulties"]), [0.0, 0.0, 0.0, 0.0, 0.0])

    def test_dataset_uses_explicit_index_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            dataset_root = repo_root / "mania-dataset"
            shard_path = dataset_root / "0"
            train_path = repo_root / "train"
            song_path = shard_path / "5678"
            song_path.mkdir(parents=True)
            train_path.mkdir(parents=True)

            _write_wav(song_path / "song_audio.wav", sample_rate=44100, duration_seconds=0.2)
            _write_wav(song_path / "soft-hitnormal.wav", sample_rate=44100, duration_seconds=0.05, frequency=1760.0)
            _write_osu(
                song_path / "chart.osu",
                audio_filename="song_audio.wav",
                beatmap_set_id=5678,
                beatmap_id=42,
                version="Insane",
                title="Loaded Song",
                artist="Loaded Artist",
                creator="Loaded Creator",
                preview_time=6789,
                circle_size=4.0,
                overall_difficulty=9.1,
            )

            index_path = build_4k_index(shard_path, train_path / "beatmap_index_4k.parquet")
            dataset = ManiaBeatmapDataset(shard_path=shard_path, sample_rate=16000, index_path=index_path)

            sample = dataset[0]

            self.assertEqual(sample["sample_rate"], 16000)
            self.assertEqual(sample["beatmap_set_id"], 5678)
            self.assertEqual(sample["beatmap_filename"], "chart.osu")
            self.assertEqual(Path(sample["audio_path"]), song_path / "song_audio.wav")
            self.assertEqual(Path(sample["beatmap_path"]), song_path / "chart.osu")
            self.assertEqual(sample["mode"], 3)
            self.assertEqual(sample["preview_time"], 6789)
            self.assertEqual(sample["title"], "Loaded Song")
            self.assertEqual(sample["artist"], "Loaded Artist")
            self.assertEqual(sample["creator"], "Loaded Creator")
            self.assertEqual(sample["version"], "Insane")
            self.assertEqual(sample["beatmap_id"], 42)
            self.assertEqual(sample["circle_size"], 4.0)
            self.assertEqual(sample["overall_difficulty"], 9.1)
            self.assertEqual(sample["key_count"], 4)
            self.assertEqual(sample["difficulty"], 0.0)
            self.assertEqual(sample["sr_difficulties"], [0.0, 0.0, 0.0, 0.0, 0.0])
            self.assertEqual(sample["audio"].dtype, np.float32)
            self.assertEqual(sample["audio_num_samples"], len(sample["audio"]))
            self.assertGreater(sample["audio_num_samples"], 0)
            self.assertLessEqual(float(np.max(np.abs(sample["audio"]))), 1.0)

    def test_dataset_defaults_to_timing_clean_training_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            dataset_root = repo_root / "mania-dataset"
            shard_path = dataset_root / "0"
            train_path = repo_root / "train"
            song_path = shard_path / "6789"
            song_path.mkdir(parents=True)
            train_path.mkdir(parents=True)

            _write_wav(song_path / "song_audio.wav", sample_rate=44100, duration_seconds=0.2)
            _write_osu(
                song_path / "valid.osu",
                audio_filename="song_audio.wav",
                beatmap_set_id=6789,
                beatmap_id=10,
                version="Valid",
                circle_size=4.0,
                timing_lines=["0,500,4,2,1,60,1,0"],
            )
            _write_osu(
                song_path / "invalid.osu",
                audio_filename="song_audio.wav",
                beatmap_set_id=6789,
                beatmap_id=11,
                version="Invalid",
                circle_size=4.0,
                timing_lines=["0,1000000,4,2,1,60,1,0"],
            )
            raw_index_path = train_path / "beatmap_index_4k.parquet"
            clean_index_path = train_path / "beatmap_index_4k_no_timing_anomalies.parquet"

            with patch("train.stage1_oracle.data.dataset.get_default_4k_index_path", return_value=raw_index_path):
                with patch(
                    "train.stage1_oracle.data.dataset.get_default_4k_no_timing_anomaly_index_path",
                    return_value=clean_index_path,
                ):
                    dataset = ManiaBeatmapDataset(shard_path=shard_path, sample_rate=16000)

            self.assertTrue(raw_index_path.exists())
            self.assertTrue(clean_index_path.exists())
            self.assertEqual(len(dataset), 1)
            self.assertEqual(dataset[0]["beatmap_filename"], "valid.osu")

    def test_build_4k_no_timing_anomaly_index_filters_unusable_red_timing_maps(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            dataset_root = repo_root / "mania-dataset"
            shard_path = dataset_root / "0"
            train_path = repo_root / "train"
            song_path = shard_path / "9999"
            song_path.mkdir(parents=True)
            train_path.mkdir(parents=True)

            _write_wav(song_path / "main_audio.wav")
            _write_osu(
                song_path / "valid.osu",
                audio_filename="main_audio.wav",
                beatmap_set_id=9999,
                beatmap_id=1,
                version="Valid",
                circle_size=4.0,
                timing_lines=["0,500,4,2,1,60,1,0"],
            )
            _write_osu(
                song_path / "invalid.osu",
                audio_filename="main_audio.wav",
                beatmap_set_id=9999,
                beatmap_id=2,
                version="Invalid",
                circle_size=4.0,
                timing_lines=["0,1000000,4,2,1,60,1,0"],
            )
            _write_osu(
                song_path / "missing.osu",
                audio_filename="main_audio.wav",
                beatmap_set_id=9999,
                beatmap_id=3,
                version="Missing",
                circle_size=4.0,
            )
            source_index_path = build_4k_index(shard_path, train_path / "beatmap_index_4k.parquet")

            report = build_4k_no_timing_anomaly_index(
                source_index_path=source_index_path,
                dataset_root=dataset_root,
                output_path=train_path / "beatmap_index_4k_no_timing_anomalies.parquet",
            )

            clean_df = pd.read_parquet(report.output_path)
            self.assertEqual(report.source_map_count, 3)
            self.assertEqual(report.clean_map_count, 1)
            self.assertEqual(report.invalid_red_timing_map_count, 1)
            self.assertEqual(report.missing_red_timing_map_count, 1)
            self.assertEqual(set(clean_df["beatmap_filename"]), {"valid.osu"})
            self.assertEqual(list(clean_df.columns), list(pd.read_parquet(source_index_path).columns))


if __name__ == "__main__":
    unittest.main()
