import tempfile
import unittest
from pathlib import Path

from train.stage1_oracle.osu.timing import (
    InvalidRedTimingError,
    MissingRedTimingError,
    RedTimingPoint,
    parse_red_timing_points,
    red_timing_point_at,
    require_red_timing_points,
)


def _write_osu(path: Path, timing_lines: list[str]) -> None:
    path.write_text(
        "\n".join(
            [
                "osu file format v14",
                "",
                "[General]",
                "AudioFilename: song.mp3",
                "Mode: 3",
                "",
                "[TimingPoints]",
                *timing_lines,
                "",
                "[HitObjects]",
                "64,192,1000,1,0,0:0:0:0:",
            ],
        ),
        encoding="utf-8",
    )


class TrainOsuTimingTests(unittest.TestCase):
    def test_parse_red_timing_points_ignores_green_lines_and_sorts_by_offset(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            osu_path = Path(tmpdir) / "chart.osu"
            _write_osu(
                osu_path,
                [
                    "1000,500,4,2,0,80,1,0",
                    "1250,-100,4,2,0,80,0,0",
                    "-250,600,3,2,0,70,1,0",
                    "2000,333.333,7,2,0,80,1,8",
                ],
            )

            timing_points = parse_red_timing_points(osu_path)

            self.assertEqual(
                timing_points,
                [
                    RedTimingPoint(offset_ms=-250.0, beat_length_ms=600.0, meter=3),
                    RedTimingPoint(offset_ms=1000.0, beat_length_ms=500.0, meter=4),
                    RedTimingPoint(offset_ms=2000.0, beat_length_ms=333.333, meter=7),
                ],
            )

    def test_require_red_timing_points_rejects_maps_without_red_timing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            osu_path = Path(tmpdir) / "chart.osu"
            _write_osu(osu_path, ["0,-100,4,2,0,80,0,0"])

            self.assertEqual(parse_red_timing_points(osu_path), [])
            with self.assertRaisesRegex(MissingRedTimingError, "no red timing point"):
                require_red_timing_points(osu_path)

    def test_red_timing_point_at_extrapolates_with_first_and_last_red_point(self) -> None:
        timing_points = [
            RedTimingPoint(offset_ms=1000.0, beat_length_ms=500.0, meter=4),
            RedTimingPoint(offset_ms=3000.0, beat_length_ms=250.0, meter=3),
        ]

        self.assertEqual(red_timing_point_at(timing_points, -2000.0), timing_points[0])
        self.assertEqual(red_timing_point_at(timing_points, 999.0), timing_points[0])
        self.assertEqual(red_timing_point_at(timing_points, 1000.0), timing_points[0])
        self.assertEqual(red_timing_point_at(timing_points, 2999.0), timing_points[0])
        self.assertEqual(red_timing_point_at(timing_points, 3000.0), timing_points[1])
        self.assertEqual(red_timing_point_at(timing_points, 6000.0), timing_points[1])

    def test_timing_lines_can_omit_optional_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            osu_path = Path(tmpdir) / "chart.osu"
            _write_osu(osu_path, ["0,500"])

            self.assertEqual(
                parse_red_timing_points(osu_path),
                [RedTimingPoint(offset_ms=0.0, beat_length_ms=500.0, meter=4)],
            )

    def test_non_positive_uninherited_timing_does_not_count_as_red_timing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            osu_path = Path(tmpdir) / "chart.osu"
            _write_osu(osu_path, ["154048,-1E-40,4,2,0,25,1,0"])

            with self.assertRaisesRegex(InvalidRedTimingError, "nonpositive"):
                parse_red_timing_points(osu_path)
            with self.assertRaisesRegex(InvalidRedTimingError, "nonpositive"):
                require_red_timing_points(osu_path)

    def test_implausible_positive_red_timing_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            osu_path = Path(tmpdir) / "chart.osu"
            _write_osu(
                osu_path,
                [
                    "0,500,4,2,0,80,1,0",
                    "1000,1e-100,4,1,0,70,1,0",
                    "2000,1E+308,4,1,0,70,1,0",
                ],
            )

            with self.assertRaisesRegex(InvalidRedTimingError, "implausible"):
                parse_red_timing_points(osu_path)


if __name__ == "__main__":
    unittest.main()
