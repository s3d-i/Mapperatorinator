from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from train.stage_2.data.index import build_dense_timing_v2_local_bpm_norm_unique_index


def _write_osu(path: Path, *, timing_lines: list[str]) -> None:
    path.write_text(
        "\n".join(
            [
                "osu file format v14",
                "",
                "[TimingPoints]",
                *timing_lines,
                "",
                "[HitObjects]",
            ]
        ),
        encoding="utf-8",
    )


def _index_row(
    *,
    beatmap_set_id: int,
    beatmap_path: str,
    beatmap_id: int,
    version: str,
) -> dict[str, object]:
    return {
        "shard": "0",
        "beatmap_set_id": beatmap_set_id,
        "beatmap_set_path": str(beatmap_set_id),
        "beatmap_path": beatmap_path,
        "beatmap_filename": Path(beatmap_path).name,
        "audio_path": f"{beatmap_set_id}/audio.mp3",
        "audio_filename": "audio.mp3",
        "audio_lead_in": 0,
        "preview_time": 0,
        "mode": 3,
        "title": f"Set {beatmap_set_id}",
        "artist": "Artist",
        "creator": "Creator",
        "version": version,
        "beatmap_id": beatmap_id,
        "hp_drain_rate": 8.0,
        "circle_size": 4.0,
        "overall_difficulty": 8.0,
        "key_count": 4,
        "difficulty": 4.0,
        "sr_difficulties": [2.0, 3.0, 4.0, 5.0, 6.0],
    }


class TrainStage2DataIndexTests(unittest.TestCase):
    def test_filters_entire_beatmapset_by_dense_timing_v2_local_bpm_norm_unique_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_root = root / "mania-dataset"
            shard = dataset_root / "0"
            keep_set = shard / "100"
            drop_set = shard / "200"
            keep_set.mkdir(parents=True)
            drop_set.mkdir(parents=True)

            _write_osu(
                keep_set / "easy.osu",
                timing_lines=[
                    "0,500,4,2,1,60,1,0",
                    "1000,400,4,2,1,60,1,0",
                ],
            )
            _write_osu(
                keep_set / "hard.osu",
                timing_lines=[
                    "0,500,4,2,1,60,1,0",
                    "1000,333.3333333333,4,2,1,60,1,0",
                ],
            )
            _write_osu(
                drop_set / "easy.osu",
                timing_lines=[
                    "0,500,4,2,1,60,1,0",
                    "1000,400,4,2,1,60,1,0",
                    "2000,333.3333333333,4,2,1,60,1,0",
                    "3000,250,4,2,1,60,1,0",
                ],
            )
            _write_osu(
                drop_set / "hard.osu",
                timing_lines=["0,500,4,2,1,60,1,0"],
            )

            source_index_path = root / "source.parquet"
            output_path = root / "filtered.parquet"
            report_path = root / "report.json"
            pd.DataFrame(
                [
                    _index_row(
                        beatmap_set_id=100,
                        beatmap_path="100/easy.osu",
                        beatmap_id=1,
                        version="Easy",
                    ),
                    _index_row(
                        beatmap_set_id=100,
                        beatmap_path="100/hard.osu",
                        beatmap_id=2,
                        version="Hard",
                    ),
                    _index_row(
                        beatmap_set_id=200,
                        beatmap_path="200/easy.osu",
                        beatmap_id=3,
                        version="Easy",
                    ),
                    _index_row(
                        beatmap_set_id=200,
                        beatmap_path="200/hard.osu",
                        beatmap_id=4,
                        version="Hard",
                    ),
                ]
            ).to_parquet(source_index_path, index=False)

            report = build_dense_timing_v2_local_bpm_norm_unique_index(
                source_index_path=source_index_path,
                dataset_root=dataset_root,
                output_path=output_path,
                report_path=report_path,
                max_local_bpm_norm_unique_per_beatmapset=3,
            )

            filtered_df = pd.read_parquet(output_path)
            self.assertEqual(report.source_map_count, 4)
            self.assertEqual(report.retained_map_count, 2)
            self.assertEqual(report.dropped_map_count, 2)
            self.assertEqual(report.source_beatmapset_count, 2)
            self.assertEqual(report.retained_beatmapset_count, 1)
            self.assertEqual(report.dropped_beatmapset_count, 1)
            self.assertEqual(set(filtered_df["beatmap_set_id"]), {100})
            self.assertTrue(report_path.exists())


if __name__ == "__main__":
    unittest.main()
