import tempfile
import textwrap
import unittest
from pathlib import Path

import numpy as np

from train.stage_2.timing.providers.oracle import (
    oracle_timing_grid_from_beatmap,
    render_oracle_dense_timing_v2,
)


class Stage2OracleTimingProviderTests(unittest.TestCase):
    def test_renders_oracle_dense_timing_from_osu_red_points(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            osu_path = Path(tmp_dir) / "map.osu"
            _write_osu(
                osu_path,
                [
                    "0,500,4,2,0,80,1,0",
                    "1000,250,4,2,0,80,1,0",
                ],
            )

            grid = oracle_timing_grid_from_beatmap(osu_path)
            track = render_oracle_dense_timing_v2(osu_path, frame_count=4)

        self.assertEqual([segment.offset_ms for segment in grid.segments], [0.0, 1000.0])
        self.assertEqual(track.shape, (4, 4))
        self.assertEqual(track.dtype, np.dtype("float32"))
        np.testing.assert_allclose(track[:, 3], np.full(4, 120.0, dtype=np.float32))

    def test_derives_frame_count_from_audio_duration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            osu_path = Path(tmp_dir) / "map.osu"
            _write_osu(osu_path, ["0,500,4,2,0,80,1,0"])

            track = render_oracle_dense_timing_v2(osu_path, audio_duration_ms=100.0)

        self.assertEqual(track.shape, (5, 4))

    def test_rejects_contradictory_frame_count_and_duration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            osu_path = Path(tmp_dir) / "map.osu"
            _write_osu(osu_path, ["0,500,4,2,0,80,1,0"])

            with self.assertRaisesRegex(ValueError, "frame_count or audio duration"):
                render_oracle_dense_timing_v2(osu_path, frame_count=10, audio_duration_ms=2000.0)


def _write_osu(path: Path, timing_lines: list[str]) -> None:
    path.write_text(
        textwrap.dedent(
            f"""\
            osu file format v14

            [TimingPoints]
            {chr(10).join(timing_lines)}
            """
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
