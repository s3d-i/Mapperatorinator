import tempfile
import unittest
from pathlib import Path

from train.stage1_oracle.events.canonical import (
    CanonicalTimepoint,
    LaneAction,
    NegativeHitObjectTimeError,
    UnsupportedCompoundLaneActionError,
    build_canonical_quantized_events,
    quantize_10ms_half_up,
)
from train.stage1_oracle.audits.event_space import audit_event_space
from train.stage1_oracle.audits.event_space import audit_osu_event_space
from train.stage1_oracle.osu.hitobjects import ManiaHitObject, ManiaHitObjectKind, parse_mania_hit_objects


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


class TrainEventSpaceAuditTests(unittest.TestCase):
    def test_parse_mania_hit_objects_preserves_hold_kind_for_zero_length_holds(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            osu_path = Path(tmpdir) / "chart.osu"
            _write_osu(
                osu_path,
                [
                    "64,192,1000,1,0,0:0:0:0:",
                    "192,192,2000,128,0,2000:0:0:0:0:",
                ],
            )

            hitobjects = parse_mania_hit_objects(osu_path)

            self.assertEqual(
                hitobjects,
                [
                    ManiaHitObject(
                        start_time_ms=1000.0,
                        end_time_ms=1000.0,
                        lane=0,
                        kind=ManiaHitObjectKind.TAP,
                    ),
                    ManiaHitObject(
                        start_time_ms=2000.0,
                        end_time_ms=2000.0,
                        lane=1,
                        kind=ManiaHitObjectKind.HOLD,
                    ),
                ],
            )

    def test_build_canonical_quantized_events_merges_compound_lane_actions(self) -> None:
        hitobjects = [
            ManiaHitObject(1000.0, 1500.0, 0, ManiaHitObjectKind.HOLD),
            ManiaHitObject(1504.0, 1504.0, 0, ManiaHitObjectKind.TAP),
            ManiaHitObject(2000.0, 2500.0, 1, ManiaHitObjectKind.HOLD),
            ManiaHitObject(2502.0, 3000.0, 1, ManiaHitObjectKind.HOLD),
            ManiaHitObject(3996.0, 3999.0, 2, ManiaHitObjectKind.HOLD),
        ]

        result = build_canonical_quantized_events(hitobjects, key_count=4)

        none = LaneAction.NONE
        self.assertEqual(quantize_10ms_half_up(7996), 8000)
        self.assertEqual(result.zero_length_hold_normalized_count, 1)
        self.assertEqual(
            result.timepoints,
            [
                CanonicalTimepoint(1000, (LaneAction.HOLD_START, none, none, none)),
                CanonicalTimepoint(1500, (LaneAction.END_TAP, none, none, none)),
                CanonicalTimepoint(2000, (none, LaneAction.HOLD_START, none, none)),
                CanonicalTimepoint(2500, (none, LaneAction.END_START, none, none)),
                CanonicalTimepoint(3000, (none, LaneAction.HOLD_END, none, none)),
                CanonicalTimepoint(4000, (none, none, LaneAction.TAP, none)),
            ],
        )

    def test_build_canonical_quantized_events_rejects_negative_times(self) -> None:
        hitobjects = [ManiaHitObject(-1.0, 10.0, 0, ManiaHitObjectKind.HOLD)]

        with self.assertRaisesRegex(NegativeHitObjectTimeError, "negative"):
            build_canonical_quantized_events(hitobjects, key_count=4)

    def test_build_canonical_quantized_events_rejects_unsupported_compounds(self) -> None:
        hitobjects = [
            ManiaHitObject(1000.0, 1000.0, 0, ManiaHitObjectKind.TAP),
            ManiaHitObject(1004.0, 1004.0, 0, ManiaHitObjectKind.TAP),
        ]

        with self.assertRaisesRegex(UnsupportedCompoundLaneActionError, "Unsupported"):
            build_canonical_quantized_events(hitobjects, key_count=4)

    def test_audit_event_space_reports_compound_drop_impact_before_vocab_choice(self) -> None:
        result = build_canonical_quantized_events(
            [
                ManiaHitObject(1000.0, 1500.0, 0, ManiaHitObjectKind.HOLD),
                ManiaHitObject(1504.0, 1504.0, 0, ManiaHitObjectKind.TAP),
                ManiaHitObject(2000.0, 2500.0, 1, ManiaHitObjectKind.HOLD),
                ManiaHitObject(2502.0, 3000.0, 1, ManiaHitObjectKind.HOLD),
            ],
            key_count=4,
        )

        report = audit_event_space([result.timepoints], top_k=2, rare_event_threshold=1)

        self.assertEqual(report.map_count, 1)
        self.assertEqual(report.total_timepoints, 5)
        self.assertEqual(report.total_non_empty_lane_actions, 5)
        self.assertEqual(report.action_counts[LaneAction.END_TAP], 1)
        self.assertEqual(report.action_counts[LaneAction.END_START], 1)
        self.assertEqual(report.same_lane_compound_event_count, 2)
        self.assertAlmostEqual(report.same_lane_compound_event_frequency, 2 / 5)
        self.assertEqual(report.four_state_unsupported_map_count, 1)
        self.assertEqual(report.four_state_unsupported_lane_action_count, 2)
        self.assertAlmostEqual(report.four_state_unsupported_lane_action_rate, 2 / 5)
        self.assertEqual(len(report.top_k_event_counts), 2)
        self.assertAlmostEqual(report.top_k_event_coverage, 2 / 5)
        self.assertEqual(report.rare_event_count, 5)

    def test_audit_osu_event_space_filters_illegal_maps_and_counts_normalizations(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            valid_path = root / "valid.osu"
            missing_red_path = root / "missing_red.osu"
            negative_path = root / "negative.osu"
            _write_osu(
                valid_path,
                ["64,192,1000,128,0,1002:0:0:0:0:"],
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

            report = audit_osu_event_space([valid_path, missing_red_path, negative_path])

            self.assertEqual(report.total_map_count, 3)
            self.assertEqual(report.audited_map_count, 1)
            self.assertEqual(report.missing_red_timing_map_count, 1)
            self.assertEqual(report.negative_time_hitobject_map_count, 1)
            self.assertEqual(report.unsupported_compound_map_count, 0)
            self.assertEqual(report.zero_length_hold_normalized_count, 1)
            self.assertEqual(report.event_space.action_counts[LaneAction.TAP], 1)

    def test_audit_osu_event_space_counts_and_skips_unsupported_compound_maps(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            unsupported_path = root / "unsupported.osu"
            _write_osu(
                unsupported_path,
                [
                    "64,192,1000,1,0,0:0:0:0:",
                    "64,192,1004,1,0,0:0:0:0:",
                ],
                timing_lines=["0,500,4,2,0,80,1,0"],
            )

            report = audit_osu_event_space([unsupported_path])

            self.assertEqual(report.total_map_count, 1)
            self.assertEqual(report.audited_map_count, 0)
            self.assertEqual(report.unsupported_compound_map_count, 1)
            self.assertEqual(report.event_space.total_timepoints, 0)


if __name__ == "__main__":
    unittest.main()
