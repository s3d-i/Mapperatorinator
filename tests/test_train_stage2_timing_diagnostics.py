import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from train.stage_2.timing.diagnostics.compare_to_oracle import (
    _enforce_average_fit_seconds,
    _comparison_row as _oracle_comparison_row,
    _summarize_rows,
    compare_timing_grids,
    TimingGridComparison,
)
from train.stage_2.timing.diagnostics.scatter import (
    DEFAULT_SCATTER_SPECS,
    _scatter_points,
    write_scatter_artifacts,
)
from train.stage_2.timing.schema import FittedTimingGrid, TimingSegment


class Stage2TimingDiagnosticsTest(unittest.TestCase):
    def test_compare_identical_grids_has_zero_error(self) -> None:
        grid = FittedTimingGrid(segments=(TimingSegment(offset_ms=0.0, beat_length_ms=500.0),))

        comparison = compare_timing_grids(grid, grid, frame_count=50)

        self.assertEqual(comparison.beat_pulse_mae, 0.0)
        self.assertEqual(comparison.local_bpm_mae, 0.0)
        self.assertEqual(comparison.mean_phase_error_beats, 0.0)
        self.assertEqual(comparison.mean_phase_error_ms, 0.0)

    def test_compare_reports_wrapped_phase_error_in_oracle_milliseconds(self) -> None:
        oracle = FittedTimingGrid(segments=(TimingSegment(offset_ms=0.0, beat_length_ms=500.0),))
        predicted = FittedTimingGrid(segments=(TimingSegment(offset_ms=20.0, beat_length_ms=500.0),))

        comparison = compare_timing_grids(predicted, oracle, frame_count=50)

        self.assertAlmostEqual(comparison.mean_phase_error_beats, 0.04, places=6)
        self.assertAlmostEqual(comparison.mean_phase_error_ms, 20.0, delta=1e-5)
        self.assertEqual(comparison.local_bpm_mae, 0.0)

    def test_compare_reports_alias_aware_local_bpm_error(self) -> None:
        oracle = FittedTimingGrid(segments=(TimingSegment(offset_ms=0.0, beat_length_ms=250.0),))
        predicted = FittedTimingGrid(segments=(TimingSegment(offset_ms=0.0, beat_length_ms=500.0),))

        comparison = compare_timing_grids(predicted, oracle, frame_count=50)

        self.assertEqual(comparison.local_bpm_mae, 120.0)
        self.assertEqual(comparison.local_bpm_alias_mae, 0.0)

    def test_summary_reports_fit_seconds_and_enforces_average_limit(self) -> None:
        rows = [
            _comparison_row(fit_seconds=0.25),
            _comparison_row(fit_seconds=1.25),
        ]

        summary = _summarize_rows(rows)

        self.assertEqual(summary["fit_seconds_mean"], 0.75)
        self.assertEqual(summary["fit_seconds_max"], 1.25)
        _enforce_average_fit_seconds(summary, max_average_fit_seconds=0.75)
        with self.assertRaisesRegex(RuntimeError, "average fitter time"):
            _enforce_average_fit_seconds(summary, max_average_fit_seconds=0.5)

    def test_default_scatter_specs_cover_first_step_diagnostics(self) -> None:
        spec_pairs = {(spec.x_metric, spec.y_metric) for spec in DEFAULT_SCATTER_SPECS}

        self.assertEqual(
            spec_pairs,
            {
                ("candidate_count", "fit_seconds"),
                ("frame_count", "fit_seconds"),
                ("segment_count_delta", "mean_phase_error_ms"),
                ("tempo_multiplier", "local_bpm_mae"),
                ("tempo_multiplier", "local_bpm_alias_mae"),
            },
        )

    def test_scatter_points_skip_non_finite_values_and_require_metrics(self) -> None:
        spec = DEFAULT_SCATTER_SPECS[0]
        rows = [
            {"candidate_count": 1000, "fit_seconds": 0.5, "sample_index": 1},
            {"candidate_count": None, "fit_seconds": 0.7, "sample_index": 2},
            {"candidate_count": 2000, "fit_seconds": "nan", "sample_index": 3},
        ]

        points = _scatter_points(rows, spec)

        self.assertEqual(points.x.tolist(), [1000.0])
        self.assertEqual(points.y.tolist(), [0.5])
        self.assertEqual(points.sample_indexes, [1])
        with self.assertRaisesRegex(ValueError, "missing required scatter metric"):
            _scatter_points([{"fit_seconds": 0.5}], spec)

    def test_write_scatter_artifacts_creates_pngs_and_manifest(self) -> None:
        rows = [
            {
                "sample_index": 0,
                "candidate_count": 1000,
                "frame_count": 5000,
                "fit_seconds": 0.5,
                "segment_count_delta": 0,
                "mean_phase_error_ms": 20.0,
                "tempo_multiplier": 1.0,
                "local_bpm_mae": 2.0,
                "local_bpm_alias_mae": 2.0,
            },
            {
                "sample_index": 1,
                "candidate_count": 4000,
                "frame_count": 12000,
                "fit_seconds": 1.5,
                "segment_count_delta": -4,
                "mean_phase_error_ms": 55.0,
                "tempo_multiplier": 2.0,
                "local_bpm_mae": 80.0,
                "local_bpm_alias_mae": 0.0,
            },
        ]

        with TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            manifest = write_scatter_artifacts(
                {"rows": rows},
                output_dir=output_dir,
                source_report_path=Path("audit.json"),
            )

            self.assertEqual(manifest["source_report_path"], "audit.json")
            self.assertEqual(len(manifest["plots"]), 5)
            self.assertTrue((output_dir / "diagnostics_scatter_manifest.json").exists())
            for plot in manifest["plots"]:
                self.assertEqual(plot["point_count"], 2)
                self.assertTrue((output_dir / plot["path"]).exists())

    def test_write_scatter_artifacts_skips_alias_plot_for_legacy_rows(self) -> None:
        rows = [
            {
                "sample_index": 0,
                "candidate_count": 1000,
                "frame_count": 5000,
                "fit_seconds": 0.5,
                "segment_count_delta": 0,
                "mean_phase_error_ms": 20.0,
                "tempo_multiplier": 1.0,
                "local_bpm_mae": 2.0,
            },
            {
                "sample_index": 1,
                "candidate_count": 4000,
                "frame_count": 12000,
                "fit_seconds": 1.5,
                "segment_count_delta": -4,
                "mean_phase_error_ms": 55.0,
                "tempo_multiplier": 2.0,
                "local_bpm_mae": 80.0,
            },
        ]

        with TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            manifest = write_scatter_artifacts({"rows": rows}, output_dir=output_dir)

            self.assertEqual(len(manifest["plots"]), 4)
            self.assertEqual(
                manifest["skipped_plots"],
                [
                    {
                        "name": "local_bpm_alias_mae_by_tempo_multiplier",
                        "reason": "missing metric",
                        "missing_metrics": ["local_bpm_alias_mae"],
                    }
                ],
            )
            self.assertFalse((output_dir / "local_bpm_alias_mae_by_tempo_multiplier.png").exists())

    def test_oracle_comparison_rows_include_scatter_workload_metrics(self) -> None:
        predicted_segments = (
            TimingSegment(offset_ms=0.0, beat_length_ms=500.0),
            TimingSegment(offset_ms=16000.0, beat_length_ms=250.0),
        )
        oracle_segments = (TimingSegment(offset_ms=0.0, beat_length_ms=500.0),)

        row = _oracle_comparison_row(
            beatmap_path=Path("map.osu"),
            audio_path=Path("song.mp3"),
            frame_count=1000,
            frame_rate_hz=50.0,
            prediction_seconds=0.4,
            candidate_count=321,
            alias_candidate_count=45,
            fit_score=1.0,
            fit_seconds=0.6,
            total_seconds=1.1,
            predicted_segments=predicted_segments,
            oracle_segments=oracle_segments,
            raw_selected_bpm=120.0,
            raw_score=0.9,
            half_tempo_score=0.1,
            double_tempo_score=0.8,
            tempo_multiplier=2.0,
            tempo_multiplier_distribution={"2": 2},
            segment_alias_switch_count=1,
            comparison=TimingGridComparison(
                frame_count=1000,
                beat_pulse_mae=0.1,
                local_bpm_mae=20.0,
                local_bpm_alias_mae=0.0,
                mean_phase_error_beats=0.02,
                max_phase_error_beats=0.1,
                mean_phase_error_ms=10.0,
                max_phase_error_ms=50.0,
            ),
        )

        self.assertEqual(row["candidate_count"], 321)
        self.assertEqual(row["alias_candidate_count"], 45)
        self.assertEqual(row["audio_duration_seconds"], 20.0)
        self.assertEqual(row["prediction_seconds"], 0.4)
        self.assertEqual(row["total_seconds"], 1.1)
        self.assertEqual(row["predicted_segment_count"], 2)
        self.assertEqual(row["oracle_segment_count"], 1)
        self.assertEqual(row["segment_count_delta"], 1)
        self.assertEqual(row["first_bpm_abs_error"], 0.0)
        self.assertEqual(row["first_bpm_alias_error"], 0.0)
        self.assertEqual(row["segment_alias_switch_count"], 1)
        self.assertEqual(row["tempo_multiplier_distribution"], {"2": 2})


def _comparison_row(*, fit_seconds: float) -> dict[str, float]:
    return {
        "fit_score": 1.0,
        "fit_seconds": fit_seconds,
        "beat_pulse_mae": 0.0,
        "local_bpm_mae": 0.0,
        "local_bpm_alias_mae": 0.0,
        "first_bpm_abs_error": 0.0,
        "first_bpm_alias_error": 0.0,
        "mean_phase_error_beats": 0.0,
        "max_phase_error_beats": 0.0,
        "mean_phase_error_ms": 0.0,
        "max_phase_error_ms": 0.0,
    }


if __name__ == "__main__":
    unittest.main()
