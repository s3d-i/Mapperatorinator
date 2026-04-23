import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from train.stage1_oracle.data.splits import build_audio_group_split


def _write_index(path: Path) -> None:
    index_df = pd.DataFrame.from_records(
        [
            {"shard": "0", "audio_path": "100/song_a.mp3", "beatmap_path": "100/easy.osu", "difficulty": 2.2},
            {"shard": "0", "audio_path": "100/song_a.mp3", "beatmap_path": "100/normal.osu", "difficulty": 3.2},
            {"shard": "0", "audio_path": "101/song_b.mp3", "beatmap_path": "101/easy.osu", "difficulty": 2.4},
            {"shard": "0", "audio_path": "102/song_c.mp3", "beatmap_path": "102/normal.osu", "difficulty": 3.5},
            {"shard": "0", "audio_path": "103/song_d.mp3", "beatmap_path": "103/hard.osu", "difficulty": 4.2},
            {"shard": "0", "audio_path": "104/song_e.mp3", "beatmap_path": "104/hard.osu", "difficulty": 4.6},
            {"shard": "0", "audio_path": "104/song_e.mp3", "beatmap_path": "104/insane.osu", "difficulty": 5.4},
            {"shard": "0", "audio_path": "105/song_f.mp3", "beatmap_path": "105/insane.osu", "difficulty": 5.7},
        ]
    )
    index_df.to_parquet(path, index=False)


def _load_manifest(path: Path) -> list[dict[str, object]]:
    return json.loads(path.read_text(encoding="utf-8"))


class Stage1SplitTests(unittest.TestCase):
    def test_audio_groups_stay_whole_across_train_eval_and_probe(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            index_path = tmp_path / "index.parquet"
            output_dir = tmp_path / "split"
            _write_index(index_path)

            artifacts = build_audio_group_split(
                index_path=index_path,
                output_dir=output_dir,
                eval_ratio=0.5,
                seed=1337,
                rollout_probe_maps_per_bin=1,
                required_train_maps_per_bin=1,
            )

            train_manifest = _load_manifest(artifacts.train_manifest_path)
            eval_manifest = _load_manifest(artifacts.eval_manifest_path)
            probe_manifest = _load_manifest(artifacts.rollout_probe_manifest_path)

            train_audio_groups = {entry["audio_group"] for entry in train_manifest}
            eval_audio_groups = {entry["audio_group"] for entry in eval_manifest}
            probe_audio_groups = {entry["audio_group"] for entry in probe_manifest}

            self.assertTrue(train_audio_groups.isdisjoint(eval_audio_groups))
            self.assertTrue(probe_audio_groups.issubset(eval_audio_groups))
            self.assertNotIn("0:100/song_a.mp3", train_audio_groups & eval_audio_groups)
            self.assertNotIn("0:104/song_e.mp3", train_audio_groups & eval_audio_groups)

            report = json.loads(artifacts.report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["eligible"]["beatmap_count"], 8)
            self.assertEqual(report["train"]["beatmap_count_by_bin"], {"2-3": 1, "3-4": 1, "4-5": 1, "5-6": 1})
            self.assertGreaterEqual(report["eval"]["beatmap_count_by_bin"]["2-3"], 1)
            self.assertGreaterEqual(report["eval"]["beatmap_count_by_bin"]["3-4"], 1)
            self.assertGreaterEqual(report["eval"]["beatmap_count_by_bin"]["4-5"], 1)
            self.assertGreaterEqual(report["eval"]["beatmap_count_by_bin"]["5-6"], 1)
            self.assertGreaterEqual(report["rollout_probe"]["beatmap_count_by_bin"]["2-3"], 1)
            self.assertGreaterEqual(report["rollout_probe"]["beatmap_count_by_bin"]["5-6"], 1)

    def test_split_output_is_deterministic_for_same_seed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            index_path = tmp_path / "index.parquet"
            _write_index(index_path)

            first = build_audio_group_split(
                index_path=index_path,
                output_dir=tmp_path / "split_a",
                eval_ratio=0.5,
                seed=2026,
                rollout_probe_maps_per_bin=1,
            )
            second = build_audio_group_split(
                index_path=index_path,
                output_dir=tmp_path / "split_b",
                eval_ratio=0.5,
                seed=2026,
                rollout_probe_maps_per_bin=1,
            )

            self.assertEqual(
                _load_manifest(first.train_manifest_path),
                _load_manifest(second.train_manifest_path),
            )
            self.assertEqual(
                _load_manifest(first.eval_manifest_path),
                _load_manifest(second.eval_manifest_path),
            )
            self.assertEqual(
                _load_manifest(first.rollout_probe_manifest_path),
                _load_manifest(second.rollout_probe_manifest_path),
            )
            first_report = json.loads(first.report_path.read_text(encoding="utf-8"))
            second_report = json.loads(second.report_path.read_text(encoding="utf-8"))
            self.assertEqual(first_report["eval_target_beatmap_count_by_bin"], second_report["eval_target_beatmap_count_by_bin"])
            self.assertEqual(first_report["rollout_probe"], second_report["rollout_probe"])

    def test_required_train_maps_per_bin_rejects_insufficient_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            index_path = tmp_path / "index.parquet"
            _write_index(index_path)

            with self.assertRaisesRegex(ValueError, "below required 2"):
                build_audio_group_split(
                    index_path=index_path,
                    output_dir=tmp_path / "split",
                    eval_ratio=0.5,
                    seed=1337,
                    required_train_maps_per_bin=2,
                )


if __name__ == "__main__":
    unittest.main()
