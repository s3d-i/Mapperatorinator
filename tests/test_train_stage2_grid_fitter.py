import unittest
from unittest.mock import patch

import numpy as np

import train.stage_2.timing.grid_fitting.scoring as scoring_module
from train.stage_2.timing.grid_fitting import GridFitter, GridFitterConfig
from train.stage_2.timing.grid_fitting.config import _effective_config_for_prediction
from train.stage_2.timing.grid_fitting.segments import _timing_segments_from_fits
from train.stage_2.timing.grid_fitting.splitting import _merge_adjacent_segment_fits
from train.stage_2.timing.grid_fitting.types import _SegmentFit
from train.stage_2.timing.diagnostics.compare_to_oracle import compare_timing_grids
from train.stage_2.timing.rendering.dense_timing_v2 import render_dense_timing_v2
from train.stage_2.timing.schema import FittedTimingGrid, FrameTimingPrediction, TimingSegment


def _synthetic_prediction(
    *,
    frame_count: int = 1000,
    frame_rate_hz: float = 50.0,
    offset_ms: float,
    beat_length_ms: float,
    pulse_width_ms: float = 40.0,
    baseline: float = 0.05,
) -> FrameTimingPrediction:
    frame_times_ms = np.arange(frame_count, dtype=np.float64) / frame_rate_hz * 1000.0
    beat_pos = (frame_times_ms - offset_ms) / beat_length_ms
    phase = beat_pos - np.floor(beat_pos)
    distance_ms = np.minimum(phase, 1.0 - phase) * beat_length_ms
    beat_prob = np.maximum(0.0, 1.0 - distance_ms / pulse_width_ms)
    beat_prob = np.maximum(beat_prob, baseline).astype(np.float32)
    return FrameTimingPrediction(
        provider="synthetic",
        beat_prob=beat_prob,
        downbeat_prob=np.zeros_like(beat_prob),
        frame_rate_hz=frame_rate_hz,
    )


def _synthetic_pulse(
    *,
    frame_count: int,
    frame_rate_hz: float,
    offset_ms: float,
    beat_length_ms: float,
    pulse_width_ms: float = 40.0,
    amplitude: float = 1.0,
) -> np.ndarray:
    frame_times_ms = np.arange(frame_count, dtype=np.float64) / frame_rate_hz * 1000.0
    beat_pos = (frame_times_ms - offset_ms) / beat_length_ms
    phase = beat_pos - np.floor(beat_pos)
    distance_ms = np.minimum(phase, 1.0 - phase) * beat_length_ms
    return amplitude * np.maximum(0.0, 1.0 - distance_ms / pulse_width_ms)


def _jittered_single_tempo_prediction(
    *,
    frame_count: int = 6000,
    frame_rate_hz: float = 50.0,
    offset_ms: float = 0.0,
    beat_length_ms: float = 500.0,
    jitter_ms: float = 25.0,
    seed: int = 4,
) -> FrameTimingPrediction:
    rng = np.random.default_rng(seed)
    frame_times_ms = np.arange(frame_count, dtype=np.float64) / frame_rate_hz * 1000.0
    beat_times_ms = np.arange(offset_ms, frame_times_ms[-1] + beat_length_ms, beat_length_ms, dtype=np.float64)
    beat_times_ms = beat_times_ms + rng.normal(0.0, jitter_ms, size=beat_times_ms.shape)
    beat_prob = np.zeros_like(frame_times_ms)
    for beat_time_ms in beat_times_ms:
        distance_ms = np.abs(frame_times_ms - beat_time_ms)
        beat_prob = np.maximum(beat_prob, np.maximum(0.0, 1.0 - distance_ms / 40.0))
    return FrameTimingPrediction(
        provider="synthetic",
        beat_prob=beat_prob.astype(np.float32),
        downbeat_prob=np.zeros(frame_count, dtype=np.float32),
        frame_rate_hz=frame_rate_hz,
    )


class Stage2GridFitterTest(unittest.TestCase):
    def test_scores_grid_without_materializing_dense_pulse_template(self) -> None:
        frame_times_ms = np.arange(7000, dtype=np.float64) / 50.0 * 1000.0
        target_template = scoring_module._pulse_template(
            frame_times_ms,
            beat_length_ms=500.0,
            offset_ms=120.0,
            pulse_width_ms=40.0,
        )
        distractor_template = scoring_module._pulse_template(
            frame_times_ms,
            beat_length_ms=760.0,
            offset_ms=200.0,
            pulse_width_ms=40.0,
        )
        signal = (
            target_template
            + 0.25 * distractor_template
            + np.linspace(0.0, 0.1, frame_times_ms.shape[0])
        )
        centered_signal = signal - float(np.mean(signal))
        signal_norm = float(np.linalg.norm(centered_signal))
        centered_template = target_template - float(np.mean(target_template))
        expected_score = float(
            np.dot(centered_signal, centered_template) / (signal_norm * np.linalg.norm(centered_template))
        )

        with patch.object(
            scoring_module,
            "_pulse_template",
            side_effect=AssertionError("dense pulse template should not be materialized"),
        ):
            score = scoring_module._score_grid(
                centered_signal,
                signal_norm=signal_norm,
                frame_times_ms=frame_times_ms,
                beat_length_ms=500.0,
                offset_ms=120.0,
                pulse_width_ms=40.0,
            )

        self.assertAlmostEqual(score, expected_score, delta=1e-12)

    def test_fits_single_segment_offset_and_bpm_from_beat_probabilities(self) -> None:
        prediction = _synthetic_prediction(offset_ms=120.0, beat_length_ms=500.0)

        result = GridFitter().fit(prediction)

        self.assertGreater(result.score, 0.95)
        self.assertEqual(len(result.grid.segments), 1)
        segment = result.grid.segments[0]
        self.assertAlmostEqual(segment.offset_ms, 120.0, delta=1e-6)
        self.assertAlmostEqual(segment.beat_length_ms, 500.0, delta=1e-6)
        self.assertAlmostEqual(segment.local_bpm, 120.0, delta=1e-6)
        self.assertEqual(result.diagnostics.selected_period_frames, 25)
        self.assertEqual(result.diagnostics.selected_offset_frames, 6)

    def test_prefers_true_tempo_over_half_and_double_tempo_aliases(self) -> None:
        prediction = _synthetic_prediction(offset_ms=80.0, beat_length_ms=500.0)
        fitter = GridFitter(GridFitterConfig(min_bpm=60.0, max_bpm=300.0))

        result = fitter.fit(prediction)

        self.assertAlmostEqual(result.grid.segments[0].beat_length_ms, 500.0, delta=1e-6)
        self.assertGreater(result.score, result.diagnostics.half_tempo_score)
        self.assertGreater(result.score, result.diagnostics.double_tempo_score)
        self.assertEqual(result.diagnostics.tempo_multiplier, 1.0)

    def test_can_promote_close_double_tempo_candidate_for_osu_subdivision(self) -> None:
        prediction = _synthetic_prediction(offset_ms=120.0, beat_length_ms=500.0)
        fitter = GridFitter(GridFitterConfig(double_tempo_score_ratio_threshold=0.0))

        result = fitter.fit(prediction)

        self.assertAlmostEqual(result.grid.segments[0].beat_length_ms, 250.0, delta=1e-6)
        self.assertAlmostEqual(result.grid.segments[0].local_bpm, 240.0, delta=1e-6)
        self.assertEqual(result.diagnostics.tempo_multiplier, 2.0)

    def test_rejects_predictions_too_short_to_fit_tempo_range(self) -> None:
        prediction = _synthetic_prediction(frame_count=4, offset_ms=0.0, beat_length_ms=500.0)

        with self.assertRaisesRegex(ValueError, "too short"):
            GridFitter().fit(prediction)

    def test_fits_tempo_change_from_oracle_rendered_pulses(self) -> None:
        oracle_grid = FittedTimingGrid(
            segments=(
                TimingSegment(offset_ms=0.0, beat_length_ms=500.0),
                TimingSegment(offset_ms=10000.0, beat_length_ms=250.0),
            )
        )
        track = render_dense_timing_v2(oracle_grid, input_start_ms=0.0, frame_count=1000)
        prediction = FrameTimingPrediction(
            provider="oracle-rendered",
            beat_prob=track[:, 0],
            downbeat_prob=np.zeros(track.shape[0], dtype=np.float32),
            frame_rate_hz=50.0,
        )

        result = GridFitter().fit(prediction)

        self.assertEqual(len(result.grid.segments), 2)
        self.assertAlmostEqual(result.grid.segments[0].local_bpm, 120.0, delta=1e-6)
        self.assertAlmostEqual(result.grid.segments[1].local_bpm, 240.0, delta=1e-6)
        self.assertAlmostEqual(result.grid.segments[1].offset_ms, 10000.0, delta=20.0)

    def test_fits_same_tempo_phase_change_from_oracle_rendered_pulses(self) -> None:
        oracle_grid = FittedTimingGrid(
            segments=(
                TimingSegment(offset_ms=0.0, beat_length_ms=500.0),
                TimingSegment(offset_ms=10040.0, beat_length_ms=500.0),
            )
        )
        track = render_dense_timing_v2(oracle_grid, input_start_ms=0.0, frame_count=1000)
        prediction = FrameTimingPrediction(
            provider="oracle-rendered",
            beat_prob=track[:, 0],
            downbeat_prob=np.zeros(track.shape[0], dtype=np.float32),
            frame_rate_hz=50.0,
        )

        result = GridFitter().fit(prediction)

        self.assertEqual(len(result.grid.segments), 2)
        self.assertAlmostEqual(result.grid.segments[0].local_bpm, 120.0, delta=1e-6)
        self.assertAlmostEqual(result.grid.segments[1].local_bpm, 120.0, delta=1e-6)
        self.assertAlmostEqual(result.grid.segments[1].offset_ms, 10040.0, delta=20.0)

    def test_caps_over_budget_tempo_changes_instead_of_discarding_all_splits(self) -> None:
        oracle_grid = FittedTimingGrid(
            segments=tuple(
                TimingSegment(
                    offset_ms=index * 10000.0,
                    beat_length_ms=500.0 if index % 2 == 0 else 250.0,
                )
                for index in range(17)
            )
        )
        frame_count = 8600
        track = render_dense_timing_v2(oracle_grid, input_start_ms=0.0, frame_count=frame_count)
        prediction = FrameTimingPrediction(
            provider="oracle-rendered",
            beat_prob=track[:, 0],
            downbeat_prob=np.zeros(track.shape[0], dtype=np.float32),
            frame_rate_hz=50.0,
        )

        result = GridFitter().fit(prediction)
        comparison = compare_timing_grids(result.grid, oracle_grid, frame_count=frame_count)

        self.assertLessEqual(len(result.grid.segments), 16)
        self.assertGreaterEqual(len(result.grid.segments), 15)
        self.assertLess(comparison.local_bpm_mae, 10.0)
        self.assertLess(comparison.mean_phase_error_ms, 25.0)

    def test_default_fitter_handles_oracle_tempos_outside_60_to_300_bpm(self) -> None:
        for beat_length_ms, expected_bpm in ((1500.0, 40.0), (125.0, 480.0)):
            with self.subTest(expected_bpm=expected_bpm):
                prediction = _synthetic_prediction(
                    frame_count=3000,
                    offset_ms=100.0,
                    beat_length_ms=beat_length_ms,
                )

                result = GridFitter().fit(prediction)

                self.assertAlmostEqual(result.grid.segments[0].local_bpm, expected_bpm, delta=1.0)
                self.assertLessEqual(result.diagnostics.candidate_count, 3000)

    def test_recursively_recovers_gradual_tempo_changes_from_oracle_rendered_pulses(self) -> None:
        segment_duration_ms = 12000.0
        oracle_grid = FittedTimingGrid(
            segments=tuple(
                TimingSegment(offset_ms=index * segment_duration_ms, beat_length_ms=beat_length_ms)
                for index, beat_length_ms in enumerate((480.0, 500.0, 520.0, 500.0, 480.0))
            )
        )
        frame_count = 3200
        track = render_dense_timing_v2(oracle_grid, input_start_ms=0.0, frame_count=frame_count)
        prediction = FrameTimingPrediction(
            provider="oracle-rendered",
            beat_prob=track[:, 0],
            downbeat_prob=np.zeros(track.shape[0], dtype=np.float32),
            frame_rate_hz=50.0,
        )

        result = GridFitter().fit(prediction)
        comparison = compare_timing_grids(result.grid, oracle_grid, frame_count=frame_count)

        self.assertGreaterEqual(len(result.grid.segments), 4)
        self.assertLess(comparison.local_bpm_mae, 2.0)
        self.assertLess(comparison.mean_phase_error_ms, 35.0)

    def test_uses_downbeat_tie_breaker_to_reject_close_double_tempo_alias(self) -> None:
        frame_count = 3000
        beat_prob = np.maximum(
            _synthetic_pulse(
                frame_count=frame_count,
                frame_rate_hz=50.0,
                offset_ms=0.0,
                beat_length_ms=500.0,
                amplitude=1.0,
            ),
            _synthetic_pulse(
                frame_count=frame_count,
                frame_rate_hz=50.0,
                offset_ms=0.0,
                beat_length_ms=250.0,
                amplitude=0.4,
            ),
        )
        downbeat_prob = _synthetic_pulse(
            frame_count=frame_count,
            frame_rate_hz=50.0,
            offset_ms=0.0,
            beat_length_ms=2000.0,
            amplitude=1.0,
        )
        prediction = FrameTimingPrediction(
            provider="synthetic",
            beat_prob=beat_prob.astype(np.float32),
            downbeat_prob=downbeat_prob.astype(np.float32),
            frame_rate_hz=50.0,
        )

        result = GridFitter().fit(prediction)

        self.assertAlmostEqual(result.grid.segments[0].local_bpm, 120.0, delta=1.0)

    def test_uses_downbeat_to_choose_bar_phase_among_beat_equivalent_offsets(self) -> None:
        frame_count = 3000
        beat_prob = _synthetic_pulse(
            frame_count=frame_count,
            frame_rate_hz=50.0,
            offset_ms=0.0,
            beat_length_ms=500.0,
        )
        downbeat_prob = _synthetic_pulse(
            frame_count=frame_count,
            frame_rate_hz=50.0,
            offset_ms=500.0,
            beat_length_ms=2000.0,
        )
        prediction = FrameTimingPrediction(
            provider="synthetic",
            beat_prob=beat_prob.astype(np.float32),
            downbeat_prob=downbeat_prob.astype(np.float32),
            frame_rate_hz=50.0,
        )

        result = GridFitter().fit(prediction)

        self.assertAlmostEqual(result.grid.segments[0].local_bpm, 120.0, delta=1.0)
        self.assertAlmostEqual(result.grid.segments[0].offset_ms % 2000.0, 500.0, delta=20.0)

    def test_merges_adjacent_same_tempo_phase_compatible_segments(self) -> None:
        frame_times_ms = np.arange(1200, dtype=np.float64) / 50.0 * 1000.0
        segment_fits = (
            _SegmentFit(
                start_frame=0,
                end_frame=500,
                score=0.8,
                beat_length_ms=500.0,
                offset_ms=20.0,
                half_tempo_score=0.1,
                double_tempo_score=0.2,
                raw_bpm=120.0,
                raw_score=0.8,
                tempo_multiplier=1.0,
                candidate_count=10,
            ),
            _SegmentFit(
                start_frame=500,
                end_frame=900,
                score=0.8,
                beat_length_ms=500.0,
                offset_ms=20.0,
                half_tempo_score=0.1,
                double_tempo_score=0.2,
                raw_bpm=120.0,
                raw_score=0.8,
                tempo_multiplier=1.0,
                candidate_count=10,
            ),
        )

        segments = _timing_segments_from_fits(segment_fits, frame_times_ms, config=GridFitterConfig())

        self.assertEqual(segments, (TimingSegment(offset_ms=20.0, beat_length_ms=500.0),))

    def test_collapses_repeated_alias_tempo_flips_after_many_splits(self) -> None:
        frame_times_ms = np.arange(2400, dtype=np.float64) / 50.0 * 1000.0
        signal = _synthetic_pulse(
            frame_count=2400,
            frame_rate_hz=50.0,
            offset_ms=0.0,
            beat_length_ms=500.0,
        ).astype(np.float64)
        segment_fits = (
            _SegmentFit(0, 500, 0.8, 500.0, 0.0, 0.1, 0.2, 120.0, 0.8, 1.0, 10),
            _SegmentFit(500, 1000, 0.8, 1000.0, 0.0, 0.1, 0.2, 60.0, 0.8, 1.0, 10),
            _SegmentFit(1000, 1500, 0.8, 500.0, 0.0, 0.1, 0.2, 120.0, 0.8, 1.0, 10),
            _SegmentFit(1500, 2000, 0.8, 1000.0, 0.0, 0.1, 0.2, 60.0, 0.8, 1.0, 10),
        )

        merged_fits = _merge_adjacent_segment_fits(
            segment_fits,
            signal,
            frame_times_ms=frame_times_ms,
            downbeat_signal=signal,
            config=GridFitterConfig(),
            fit_cache={},
        )
        merged_segments = _timing_segments_from_fits(merged_fits, frame_times_ms, config=GridFitterConfig())

        self.assertEqual(len(merged_segments), 1)
        self.assertAlmostEqual(merged_segments[0].beat_length_ms, 500.0, delta=1e-6)
        self.assertAlmostEqual(merged_segments[0].offset_ms % 500.0, 0.0, delta=1e-6)

    def test_collapses_many_same_tempo_phase_resets(self) -> None:
        frame_times_ms = np.arange(2400, dtype=np.float64) / 50.0 * 1000.0
        signal = _synthetic_pulse(
            frame_count=2400,
            frame_rate_hz=50.0,
            offset_ms=0.0,
            beat_length_ms=500.0,
        ).astype(np.float64)
        segment_fits = (
            _SegmentFit(0, 500, 0.8, 500.0, 0.0, 0.1, 0.2, 120.0, 0.8, 1.0, 10),
            _SegmentFit(500, 1000, 0.8, 500.0, 80.0, 0.1, 0.2, 120.0, 0.8, 1.0, 10),
            _SegmentFit(1000, 1500, 0.8, 500.0, 160.0, 0.1, 0.2, 120.0, 0.8, 1.0, 10),
            _SegmentFit(1500, 2000, 0.8, 500.0, 240.0, 0.1, 0.2, 120.0, 0.8, 1.0, 10),
        )

        merged_fits = _merge_adjacent_segment_fits(
            segment_fits,
            signal,
            frame_times_ms=frame_times_ms,
            downbeat_signal=np.zeros_like(signal),
            config=GridFitterConfig(),
            fit_cache={},
        )
        segments = _timing_segments_from_fits(merged_fits, frame_times_ms, config=GridFitterConfig())

        self.assertEqual(len(segments), 1)
        self.assertAlmostEqual(segments[0].beat_length_ms, 500.0, delta=1e-6)
        self.assertAlmostEqual(segments[0].offset_ms % 500.0, 0.0, delta=1e-6)

    def test_limits_downbeat_refinement_to_top_beat_candidates(self) -> None:
        frame_count = 3000
        beat_prob = _synthetic_pulse(
            frame_count=frame_count,
            frame_rate_hz=50.0,
            offset_ms=0.0,
            beat_length_ms=500.0,
        )
        downbeat_prob = _synthetic_pulse(
            frame_count=frame_count,
            frame_rate_hz=50.0,
            offset_ms=0.0,
            beat_length_ms=2000.0,
        )
        prediction = FrameTimingPrediction(
            provider="synthetic",
            beat_prob=beat_prob.astype(np.float32),
            downbeat_prob=downbeat_prob.astype(np.float32),
            frame_rate_hz=50.0,
        )
        config = GridFitterConfig(
            min_bpm=80.0,
            max_bpm=180.0,
            max_grid_candidates_per_segment=240,
            downbeat_refine_candidate_count=8,
        )
        call_count = 0
        original_downbeat_fit = scoring_module._best_downbeat_grid_fit

        def counted_downbeat_fit(*args: object, **kwargs: object) -> tuple[float, float]:
            nonlocal call_count
            call_count += 1
            return original_downbeat_fit(*args, **kwargs)

        with patch.object(scoring_module, "_best_downbeat_grid_fit", side_effect=counted_downbeat_fit):
            result = GridFitter(config).fit(prediction)

        self.assertAlmostEqual(result.grid.segments[0].local_bpm, 120.0, delta=1.0)
        self.assertLessEqual(call_count, 30)

    def test_does_not_over_split_noisy_single_tempo_signal(self) -> None:
        prediction = _jittered_single_tempo_prediction()

        result = GridFitter().fit(prediction)

        self.assertEqual(len(result.grid.segments), 1)
        self.assertAlmostEqual(result.grid.segments[0].local_bpm, 120.0, delta=1.5)

    def test_long_predictions_use_larger_search_budget(self) -> None:
        config = GridFitterConfig()

        short_config = _effective_config_for_prediction(3000, frame_rate_hz=50.0, config=config)
        long_config = _effective_config_for_prediction(36000, frame_rate_hz=50.0, config=config)

        self.assertEqual(short_config.max_segments, config.max_segments)
        self.assertEqual(short_config.max_grid_candidates_per_segment, config.max_grid_candidates_per_segment)
        self.assertGreater(long_config.max_segments, config.max_segments)
        self.assertEqual(long_config.max_grid_candidates_per_segment, config.max_grid_candidates_per_segment)
        self.assertGreater(long_config.max_split_candidates_per_segment, config.max_split_candidates_per_segment)


if __name__ == "__main__":
    unittest.main()
