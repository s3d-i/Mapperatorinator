import tempfile
import unittest
from pathlib import Path

from train.stage1_oracle.audits.window_boundary import (
    OsuWindowBoundaryMapInput,
    WindowBoundaryMap,
    audit_osu_window_boundaries,
    audit_window_boundaries,
)
from train.stage1_oracle.events.canonical import CanonicalTimepoint, LaneAction


def _lane_actions(*actions: LaneAction) -> tuple[LaneAction, ...]:
    padded = list(actions)
    while len(padded) < 4:
        padded.append(LaneAction.NONE)
    return tuple(padded)


def _write_osu(path: Path, hitobject_lines: list[str], *, timing_lines: list[str] | None = None) -> None:
    timing_lines = timing_lines or []
    path.write_text(
        "\n".join(
            [
                "osu file format v14",
                "",
                "[General]",
                "AudioFilename: song.mp3",
                "Mode: 3",
                "",
                "[Difficulty]",
                "CircleSize:4",
                "",
                "[TimingPoints]",
                *timing_lines,
                "",
                "[HitObjects]",
                *hitobject_lines,
            ],
        ),
        encoding="utf-8",
    )


class TrainWindowBoundaryAuditTests(unittest.TestCase):
    def test_audit_window_boundaries_reports_boundary_events_and_hold_crossings(self) -> None:
        report = audit_window_boundaries(
            [
                WindowBoundaryMap(
                    difficulty=5.0,
                    generation_end_ms=16000,
                    timepoints=[
                        CanonicalTimepoint(7000, _lane_actions(LaneAction.HOLD_START)),
                        CanonicalTimepoint(8000, _lane_actions(LaneAction.NONE, LaneAction.TAP)),
                        CanonicalTimepoint(9000, _lane_actions(LaneAction.HOLD_END)),
                    ],
                ),
            ],
        )

        self.assertEqual(report.total_map_count, 1)
        self.assertEqual(report.audited_map_count, 1)
        self.assertEqual(report.boundary_count, 1)
        self.assertEqual(report.boundary_event_count, 1)
        self.assertEqual(report.boundary_lane_action_count, 1)
        self.assertAlmostEqual(report.boundary_event_density, 1.0)
        self.assertEqual(report.hold_crossing_boundary_count, 1)
        self.assertAlmostEqual(report.hold_crossing_boundary_rate, 1.0)
        self.assertEqual(report.hold_crossing_lane_count, 1)
        self.assertEqual(report.stitch_duplicate_timepoint_count, 0)
        self.assertEqual(report.stitch_collision_timepoint_count, 0)
        self.assertEqual(report.stitch_roundtrip_mismatch_count, 0)
        self.assertEqual(report.bins["5-6"].boundary_event_count, 1)

    def test_audit_window_boundaries_does_not_fold_boundary_events_into_open_mask(self) -> None:
        report = audit_window_boundaries(
            [
                WindowBoundaryMap(
                    difficulty=3.0,
                    generation_end_ms=16000,
                    timepoints=[
                        CanonicalTimepoint(8000, _lane_actions(LaneAction.HOLD_START)),
                        CanonicalTimepoint(9000, _lane_actions(LaneAction.HOLD_END)),
                    ],
                ),
            ],
        )

        self.assertEqual(report.boundary_count, 1)
        self.assertEqual(report.boundary_event_count, 1)
        self.assertEqual(report.hold_crossing_boundary_count, 0)
        self.assertEqual(report.hold_crossing_lane_count, 0)

    def test_audit_osu_window_boundaries_filters_illegal_maps_before_reporting(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            valid_path = root / "valid.osu"
            missing_red_path = root / "missing_red.osu"
            negative_path = root / "negative.osu"
            unsupported_compound_path = root / "unsupported_compound.osu"
            four_state_unsupported_path = root / "end_tap.osu"
            out_of_range_path = root / "out_of_range.osu"

            _write_osu(
                valid_path,
                [
                    "64,192,7000,128,0,9000:0:0:0:0:",
                    "192,192,7996,1,0,0:0:0:0:",
                ],
                timing_lines=["0,500,4,2,0,80,1,0"],
            )
            _write_osu(
                missing_red_path,
                ["64,192,1000,1,0,0:0:0:0:"],
                timing_lines=["0,-100,4,2,0,80,0,0"],
            )
            _write_osu(
                negative_path,
                ["64,192,-1,1,0,0:0:0:0:"],
                timing_lines=["0,500,4,2,0,80,1,0"],
            )
            _write_osu(
                unsupported_compound_path,
                [
                    "64,192,1000,1,0,0:0:0:0:",
                    "64,192,1004,1,0,0:0:0:0:",
                ],
                timing_lines=["0,500,4,2,0,80,1,0"],
            )
            _write_osu(
                four_state_unsupported_path,
                [
                    "64,192,1000,128,0,1500:0:0:0:0:",
                    "64,192,1504,1,0,0:0:0:0:",
                ],
                timing_lines=["0,500,4,2,0,80,1,0"],
            )
            _write_osu(
                out_of_range_path,
                ["64,192,1000,1,0,0:0:0:0:"],
                timing_lines=["0,500,4,2,0,80,1,0"],
            )

            report = audit_osu_window_boundaries(
                [
                    OsuWindowBoundaryMapInput(valid_path, difficulty=2.5, generation_end_ms=16000),
                    OsuWindowBoundaryMapInput(missing_red_path, difficulty=2.5, generation_end_ms=16000),
                    OsuWindowBoundaryMapInput(negative_path, difficulty=2.5, generation_end_ms=16000),
                    OsuWindowBoundaryMapInput(
                        unsupported_compound_path,
                        difficulty=2.5,
                        generation_end_ms=16000,
                    ),
                    OsuWindowBoundaryMapInput(
                        four_state_unsupported_path,
                        difficulty=2.5,
                        generation_end_ms=16000,
                    ),
                    OsuWindowBoundaryMapInput(out_of_range_path, difficulty=6.5, generation_end_ms=16000),
                ],
            )

        self.assertEqual(report.total_map_count, 6)
        self.assertEqual(report.audited_map_count, 1)
        self.assertEqual(report.out_of_range_map_count, 1)
        self.assertEqual(report.missing_red_timing_map_count, 1)
        self.assertEqual(report.negative_time_hitobject_map_count, 1)
        self.assertEqual(report.unsupported_compound_map_count, 1)
        self.assertEqual(report.four_state_unsupported_map_count, 1)
        self.assertEqual(report.boundary_event_count, 1)
        self.assertEqual(report.hold_crossing_boundary_count, 1)


if __name__ == "__main__":
    unittest.main()
