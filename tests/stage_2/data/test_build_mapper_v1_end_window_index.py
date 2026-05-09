import tempfile
import unittest
from pathlib import Path

import pandas as pd

from train.stage_2.data.build_mapper_v1_end_window_index import build_mapper_v1_end_window_index


class BuildMapperV1EndWindowIndexTests(unittest.TestCase):
    def test_adds_only_missing_terminal_stride_windows(self) -> None:
        source_df = pd.DataFrame.from_records(
            [
                _row("a.osu", frame_count=900, target_start_frame=0),
                _row("a.osu", frame_count=900, target_start_frame=400),
                _row("b.osu", frame_count=800, target_start_frame=0),
                _row("b.osu", frame_count=800, target_start_frame=400),
                _row("short.osu", frame_count=399, target_start_frame=0),
            ],
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_path = temp_path / "source.parquet"
            output_path = temp_path / "output.parquet"
            report_path = temp_path / "report.json"
            source_df.to_parquet(source_path, index=False)

            report = build_mapper_v1_end_window_index(
                source_index_path=source_path,
                output_path=output_path,
                report_path=report_path,
            )
            output_df = pd.read_parquet(output_path)

        self.assertEqual(report.source_rows, 5)
        self.assertEqual(report.added_end_window_rows, 1)
        self.assertEqual(report.existing_end_window_rows, 2)
        self.assertEqual(report.maps_shorter_than_write_window, 1)
        self.assertEqual(len(output_df), 6)
        added = output_df[(output_df["beatmap_path"] == "a.osu") & (output_df["target_start_frame"] == 500)]
        self.assertEqual(len(added), 0)
        added = output_df[(output_df["beatmap_path"] == "a.osu") & (output_df["target_start_frame"] == 800)]
        self.assertEqual(len(added), 1)
        self.assertEqual(int(added.iloc[0]["target_start_ms"]), 16_000)
        self.assertEqual(len(output_df[(output_df["beatmap_path"] == "b.osu") & (output_df["target_start_frame"] == 400)]), 1)


def _row(beatmap_path: str, *, frame_count: int, target_start_frame: int) -> dict[str, object]:
    return {
        "shard": "shard-a",
        "beatmap_path": beatmap_path,
        "audio_path": beatmap_path.replace(".osu", ".mp3"),
        "difficulty": 4.0,
        "frame_count": frame_count,
        "target_start_frame": target_start_frame,
        "target_start_ms": target_start_frame * 20,
        "filtered_index": 1,
        "source_index": 1,
    }


if __name__ == "__main__":
    unittest.main()
