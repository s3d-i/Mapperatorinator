import math
import tempfile
import unittest
from pathlib import Path

from train.stage1_oracle.audits.dense_timing import (
    DenseTimingAuditReport,
    DenseTimingMapInput,
    audit_dense_timing_tracks,
    build_dense_timing_gate_decision,
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
                "[Difficulty]",
                "CircleSize:4",
                "",
                "[TimingPoints]",
                *timing_lines,
                "",
                "[HitObjects]",
                "64,192,1000,1,0,0:0:0:0:",
                "192,192,1500,128,0,2000:0:0:0:0:",
            ],
        ),
        encoding="utf-8",
    )


class TrainDenseTimingAuditTests(unittest.TestCase):
    def test_audit_reports_required_dense_timing_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            good = Path(tmpdir) / "good.osu"
            missing = Path(tmpdir) / "missing.osu"
            _write_osu(good, ["0,500,4,2,0,80,1,0", "1000,250,4,2,0,80,1,0"])
            _write_osu(missing, ["0,-100,4,2,0,80,0,0"])

            report = audit_dense_timing_tracks(
                [
                    DenseTimingMapInput(good, difficulty=2.5, audio_duration_ms=3000.0),
                    DenseTimingMapInput(missing, difficulty=3.5, audio_duration_ms=3000.0),
                    DenseTimingMapInput(good, difficulty=6.5, audio_duration_ms=3000.0),
                ],
            )

        self.assertEqual(report.total_map_count, 3)
        self.assertEqual(report.audited_map_count, 1)
        self.assertEqual(report.missing_red_timing_map_count, 1)
        self.assertEqual(report.invalid_red_timing_map_count, 0)
        self.assertEqual(report.invalid_red_timing_point_count, 0)
        self.assertEqual(report.nonfinite_red_timing_point_count, 0)
        self.assertEqual(report.nonpositive_red_timing_point_count, 0)
        self.assertEqual(report.implausible_red_timing_point_count, 0)
        self.assertEqual(report.out_of_range_map_count, 1)
        self.assertEqual(report.window_count, 1)
        self.assertEqual(report.frame_count, 600)
        self.assertEqual(report.timing_track_nan_count, 0)
        self.assertEqual(report.timing_track_inf_count, 0)
        self.assertLess(report.phase_unit_norm_error_max, 1e-6)
        self.assertGreater(report.beat_pulse_nonzero_ratio, 0.0)
        self.assertGreater(report.local_bpm_log_norm_std, 0.0)
        self.assertEqual(report.raw_bpm_min, 120.0)
        self.assertEqual(report.raw_bpm_p01, 120.0)
        self.assertEqual(report.raw_bpm_p50, 240.0)
        self.assertEqual(report.raw_bpm_p99, 240.0)
        self.assertEqual(report.raw_bpm_max, 240.0)
        self.assertEqual(report.raw_beat_length_min, 250.0)
        self.assertEqual(report.raw_beat_length_max, 500.0)
        self.assertEqual(report.bpm_norm_clipped_low_count, 0)
        self.assertEqual(report.bpm_norm_clipped_high_count, 0)
        self.assertEqual(report.bpm_norm_clipped_ratio, 0.0)
        expected_bpm_log_mean = 0.25 * math.log(120.0) + 0.75 * math.log(240.0)
        self.assertAlmostEqual(report.bpm_log_mean, expected_bpm_log_mean, places=6)
        self.assertGreater(report.bpm_log_std, 0.0)

        gate = build_dense_timing_gate_decision(report)

        self.assertEqual(gate.status, "FAIL")
        self.assertEqual(gate.timing_track_version, "timing_track_20ms_v1")
        self.assertEqual(gate.timing_frame_count_per_window, 600)
        self.assertEqual(gate.bpm_norm_clipped_high_count, 0)

    def test_gate_passes_when_filtered_timing_anomalies_stay_under_cap(self) -> None:
        report = _dense_timing_report(
            total_map_count=11047,
            audited_map_count=10977,
            invalid_red_timing_map_count=70,
            invalid_red_timing_point_count=60396,
            nonpositive_red_timing_point_count=20,
            implausible_red_timing_point_count=60376,
            raw_bpm_min=20.0,
            raw_bpm_p01=88.7,
            raw_bpm_p50=179.0,
            raw_bpm_p99=300.0,
            raw_bpm_max=896.0,
            raw_beat_length_min=66.9642857142857,
            raw_beat_length_max=3000.0,
            bpm_norm_clipped_low_count=237829,
            bpm_norm_clipped_high_count=134948,
            bpm_norm_clipped_ratio=0.0027750630904258885,
        )

        gate = build_dense_timing_gate_decision(report)

        self.assertEqual(gate.status, "PASS")
        self.assertEqual(gate.renderer_numerics_status, "PASS")
        self.assertEqual(gate.timing_anomaly_status, "PASS")
        self.assertEqual(gate.coverage_status, "PASS")
        self.assertAlmostEqual(gate.timing_anomaly_map_ratio, 70 / 11047)
        self.assertEqual(gate.failure_reasons, [])

    def test_audit_counts_bpm_norm_clipping(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            osu_path = Path(tmpdir) / "outlier.osu"
            _write_osu(
                osu_path,
                [
                    "0,500,4,2,0,80,1,0",
                    "9980,60,4,2,0,80,1,0",
                ],
            )

            report = audit_dense_timing_tracks(
                [
                    DenseTimingMapInput(osu_path, difficulty=2.5, audio_duration_ms=3000.0),
                ],
            )

        self.assertEqual(report.raw_bpm_min, 120.0)
        self.assertEqual(report.raw_bpm_max, 1000.0)
        self.assertEqual(report.raw_beat_length_min, 60.0)
        self.assertEqual(report.raw_beat_length_max, 500.0)
        self.assertEqual(report.bpm_norm_clipped_low_count, 0)
        self.assertEqual(report.bpm_norm_clipped_high_count, 1)
        self.assertAlmostEqual(report.bpm_norm_clipped_ratio, 1 / 600)

    def test_gate_fails_when_positive_red_timing_is_implausible(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            good = Path(tmpdir) / "good.osu"
            implausible = Path(tmpdir) / "implausible.osu"
            _write_osu(good, ["0,500,4,2,0,80,1,0"])
            _write_osu(
                implausible,
                [
                    "0,500,4,2,0,80,1,0",
                    "1000,1e-100,4,1,0,70,1,0",
                    "2000,1E+308,4,1,0,70,1,0",
                ],
            )

            report = audit_dense_timing_tracks(
                [
                    DenseTimingMapInput(good, difficulty=2.5, audio_duration_ms=3000.0),
                    DenseTimingMapInput(implausible, difficulty=3.5, audio_duration_ms=3000.0),
                ],
            )

        self.assertEqual(report.audited_map_count, 1)
        self.assertEqual(report.invalid_red_timing_map_count, 1)
        self.assertEqual(report.invalid_red_timing_point_count, 2)
        self.assertEqual(report.implausible_red_timing_point_count, 2)
        self.assertEqual(build_dense_timing_gate_decision(report).status, "FAIL")


def _dense_timing_report(**overrides: object) -> DenseTimingAuditReport:
    defaults = dict(
        total_map_count=1,
        audited_map_count=1,
        out_of_range_map_count=0,
        missing_red_timing_map_count=0,
        invalid_red_timing_map_count=0,
        invalid_red_timing_point_count=0,
        nonfinite_red_timing_point_count=0,
        nonpositive_red_timing_point_count=0,
        implausible_red_timing_point_count=0,
        negative_time_hitobject_map_count=0,
        audio_duration_failure_count=0,
        window_count=1,
        frame_count=600,
        timing_track_nan_count=0,
        timing_track_inf_count=0,
        phase_unit_norm_error_mean=0.0,
        phase_unit_norm_error_max=0.0,
        beat_pulse_nonzero_ratio=0.2,
        local_bpm_log_norm_mean=0.0,
        local_bpm_log_norm_std=1.0,
        local_bpm_log_norm_min=-4.0,
        local_bpm_log_norm_max=4.0,
        raw_bpm_min=120.0,
        raw_bpm_p01=120.0,
        raw_bpm_p50=180.0,
        raw_bpm_p99=240.0,
        raw_bpm_max=240.0,
        raw_beat_length_min=250.0,
        raw_beat_length_max=500.0,
        bpm_norm_clipped_low_count=0,
        bpm_norm_clipped_high_count=0,
        bpm_norm_clipped_ratio=0.0,
        bpm_log_mean=5.0,
        bpm_log_std=0.2,
        bins={},
        debug_plot_paths=[],
    )
    defaults.update(overrides)
    return DenseTimingAuditReport(**defaults)


if __name__ == "__main__":
    unittest.main()
