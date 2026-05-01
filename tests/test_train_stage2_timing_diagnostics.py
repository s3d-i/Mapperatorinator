import unittest

from train.stage_2.timing.diagnostics.compare_to_oracle import (
    _enforce_average_fit_seconds,
    _summarize_rows,
    compare_timing_grids,
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


def _comparison_row(*, fit_seconds: float) -> dict[str, float]:
    return {
        "fit_score": 1.0,
        "fit_seconds": fit_seconds,
        "beat_pulse_mae": 0.0,
        "local_bpm_mae": 0.0,
        "mean_phase_error_beats": 0.0,
        "max_phase_error_beats": 0.0,
        "mean_phase_error_ms": 0.0,
        "max_phase_error_ms": 0.0,
    }


if __name__ == "__main__":
    unittest.main()
