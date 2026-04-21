import tempfile
import unittest
from pathlib import Path

from train.stage1_oracle.audits.quantization import (
    OsuQuantizationMapInput,
    QuantizationMap,
    audit_osu_quantization,
    audit_quantization,
    build_quantization_gate_decision,
)
from train.stage1_oracle.osu.hitobjects import ManiaHitObject, ManiaHitObjectKind


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


class TrainQuantizationAuditTests(unittest.TestCase):
    def test_audit_quantization_reports_half_up_errors_and_post_quantization_collisions(self) -> None:
        report = audit_quantization(
            [
                QuantizationMap(
                    difficulty=4.0,
                    hitobjects=[
                        ManiaHitObject(1000.0, 1000.0, 0, ManiaHitObjectKind.TAP),
                        ManiaHitObject(1004.0, 1004.0, 0, ManiaHitObjectKind.TAP),
                        ManiaHitObject(1001.0, 1001.0, 1, ManiaHitObjectKind.TAP),
                        ManiaHitObject(1002.0, 1002.0, 1, ManiaHitObjectKind.TAP),
                        ManiaHitObject(2005.0, 2016.0, 1, ManiaHitObjectKind.HOLD),
                    ],
                ),
            ],
        )

        self.assertEqual(report.total_map_count, 1)
        self.assertEqual(report.audited_map_count, 1)
        self.assertEqual(report.key_count, 4)
        self.assertEqual(report.quantizer, "10ms_half_up")
        self.assertEqual(report.quantizer_grid_ms, 10)
        self.assertEqual(report.quantizer_tie_break, "floor((t_ms + 5) / 10)")
        self.assertEqual(report.timestamp_count, 6)
        self.assertAlmostEqual(report.quantization_error_ms.mean, 16 / 6)
        self.assertEqual(report.quantization_error_ms.p95, 5)
        self.assertEqual(report.quantization_error_ms.max, 5)
        self.assertEqual(report.primitive_lane_action_count, 6)
        self.assertEqual(report.post_quantization_collision_timepoint_count, 1)
        self.assertEqual(report.post_quantization_collision_lane_time_cell_count, 2)
        self.assertEqual(report.post_quantization_collision_lane_action_count, 4)
        self.assertEqual(report.post_quantization_collision_affected_map_count, 1)
        self.assertEqual(report.post_quantization_collision_rate_denominator, "primitive_lane_action_count")
        self.assertAlmostEqual(report.post_quantization_collision_rate, 4 / 6)
        self.assertEqual(report.bins["4-5"].timestamp_count, 6)
        self.assertEqual(report.bins["4-5"].post_quantization_collision_timepoint_count, 1)
        self.assertEqual(report.bins["4-5"].post_quantization_collision_lane_time_cell_count, 2)

    def test_audit_quantization_reports_zero_length_hold_normalization(self) -> None:
        report = audit_quantization(
            [
                QuantizationMap(
                    difficulty=3.5,
                    hitobjects=[
                        ManiaHitObject(1005.0, 1005.0, 0, ManiaHitObjectKind.HOLD),
                        ManiaHitObject(2000.0, 2020.0, 1, ManiaHitObjectKind.HOLD),
                    ],
                ),
            ],
        )

        self.assertEqual(report.timestamp_count, 4)
        self.assertEqual(report.primitive_lane_action_count, 3)
        self.assertEqual(report.zero_length_hold_normalized_count, 1)
        self.assertEqual(report.zero_length_hold_normalized_map_count, 1)
        self.assertEqual(report.bins["3-4"].zero_length_hold_normalized_count, 1)
        self.assertEqual(report.bins["3-4"].zero_length_hold_normalized_map_count, 1)

    def test_audit_osu_quantization_filters_legality_before_reporting(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            valid_path = root / "valid.osu"
            missing_red_path = root / "missing_red.osu"
            negative_path = root / "negative.osu"
            out_of_range_path = root / "out_of_range.osu"

            _write_osu(
                valid_path,
                ["64,192,1005,128,0,1016:0:0:0:0:"],
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
                out_of_range_path,
                ["64,192,1000,1,0,0:0:0:0:"],
                timing_lines=["0,500,4,2,0,80,1,0"],
            )

            report = audit_osu_quantization(
                [
                    OsuQuantizationMapInput(valid_path, difficulty=2.5),
                    OsuQuantizationMapInput(missing_red_path, difficulty=2.5),
                    OsuQuantizationMapInput(negative_path, difficulty=2.5),
                    OsuQuantizationMapInput(out_of_range_path, difficulty=6.5),
                ],
            )

        self.assertEqual(report.total_map_count, 4)
        self.assertEqual(report.audited_map_count, 1)
        self.assertEqual(report.missing_red_timing_map_count, 1)
        self.assertEqual(report.negative_time_hitobject_map_count, 1)
        self.assertEqual(report.out_of_range_map_count, 1)
        self.assertEqual(report.timestamp_count, 2)
        self.assertEqual(report.quantization_error_ms.max, 5)

    def test_quantization_gate_requires_full_design_doc_accounting(self) -> None:
        report = audit_quantization(
            [
                QuantizationMap(
                    difficulty=2.5,
                    hitobjects=[ManiaHitObject(1005.0, 1016.0, 0, ManiaHitObjectKind.HOLD)],
                ),
            ],
        )

        decision = build_quantization_gate_decision(
            report,
            eligible_map_count=1,
            difficulty_source="approved-source",
            expected_difficulty_source="approved-source",
            code_dirty=True,
            dirty_patch_sha256="abc123",
        )

        self.assertEqual(decision.status, "PASS")
        self.assertEqual(decision.coverage_status, "PASS")
        self.assertEqual(decision.key_count_status, "PASS")
        self.assertEqual(decision.quantizer_status, "PASS")
        self.assertEqual(decision.reproducibility_status, "PATCH_HASH_RECORDED")
        self.assertEqual(decision.post_quantization_collision_threshold_status, "NOT_APPLICABLE_NO_DESIGN_THRESHOLD")
        self.assertEqual(decision.failure_reasons, [])

    def test_quantization_gate_fails_without_clean_code_or_dirty_patch_hash(self) -> None:
        report = audit_quantization(
            [
                QuantizationMap(
                    difficulty=2.5,
                    hitobjects=[ManiaHitObject(1005.0, 1016.0, 0, ManiaHitObjectKind.HOLD)],
                ),
            ],
        )

        decision = build_quantization_gate_decision(
            report,
            eligible_map_count=1,
            difficulty_source="approved-source",
            expected_difficulty_source="approved-source",
            code_dirty=True,
            dirty_patch_sha256=None,
        )

        self.assertEqual(decision.status, "FAIL")
        self.assertEqual(decision.reproducibility_status, "FAIL")
        self.assertIn("code dirty without recorded dirty patch sha256", decision.failure_reasons)


if __name__ == "__main__":
    unittest.main()
