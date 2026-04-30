import tempfile
import textwrap
import unittest
from pathlib import Path

from train.stage_2.osu_core.timing import (
    InvalidRedTimingError,
    MissingRedTimingError,
    RedTimingPoint,
    require_red_timing_points,
)


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


class Stage2OsuTimingTest(unittest.TestCase):
    def test_requires_valid_red_timing_points_sorted_by_offset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            osu_path = Path(tmp_dir) / "map.osu"
            _write_osu(
                osu_path,
                [
                    "1000,500,4,2,0,80,1,0",
                    "500,-100,4,2,0,80,0,0",
                    "-250,600,3,2,0,80,1,0",
                ],
            )

            self.assertEqual(
                require_red_timing_points(osu_path),
                [
                    RedTimingPoint(offset_ms=-250.0, beat_length_ms=600.0, meter=3),
                    RedTimingPoint(offset_ms=1000.0, beat_length_ms=500.0, meter=4),
                ],
            )

    def test_rejects_missing_and_invalid_red_timing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            missing_path = Path(tmp_dir) / "missing.osu"
            _write_osu(missing_path, ["0,-100,4,2,0,80,0,0"])
            with self.assertRaisesRegex(MissingRedTimingError, "no red timing point"):
                require_red_timing_points(missing_path)

            invalid_path = Path(tmp_dir) / "invalid.osu"
            _write_osu(invalid_path, ["0,0,4,2,0,80,1,0"])
            with self.assertRaisesRegex(InvalidRedTimingError, "nonpositive=1"):
                require_red_timing_points(invalid_path)


if __name__ == "__main__":
    unittest.main()
