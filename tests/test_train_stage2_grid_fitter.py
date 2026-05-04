import math
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import numpy as np

import train.stage_2.timing.grid_fitting.scoring as scoring_module
from train.stage_2.timing.grid_fitting import GridFitter, GridFitterConfig
from train.stage_2.timing.grid_fitting.alias import _canonicalize_tempo_aliases, _segment_alias_switch_count
from train.stage_2.timing.grid_fitting.config import _effective_config_for_prediction
from train.stage_2.timing.grid_fitting.segments import _timing_segments_from_fits
from train.stage_2.timing.grid_fitting.splitting import _merge_adjacent_segment_fits
from train.stage_2.timing.grid_fitting.types import _SegmentFit
from train.stage_2.osu_core.timing import require_red_timing_points
from train.stage_2.timing.diagnostics.compare_to_oracle import compare_timing_grids
from train.stage_2.timing.rendering.dense_timing_v2 import render_dense_timing_v2
from train.stage_2.timing.schema import FittedTimingGrid, FrameTimingPrediction, TimingSegment


_REPO_ROOT = Path(__file__).resolve().parents[1]
_DATASET_ROOT = _REPO_ROOT / "mania-dataset"
_FRACTIONAL_FIXTURE_INDEX_PATH = (
    _REPO_ROOT / "train/artifacts/indexes/beatmap_index_4k_no_timing_anomalies_2to6.parquet"
)


@dataclass(frozen=True)
class _IndexedFractionalBpmFixture:
    name: str
    family_part: float
    shard: str
    audio_path: str
    beatmap_path: str
    frame_count: int = 3000

    @property
    def dataset_audio_path(self) -> Path:
        return _DATASET_ROOT / self.shard / self.audio_path

    @property
    def dataset_beatmap_path(self) -> Path:
        return _DATASET_ROOT / self.shard / self.beatmap_path

    @property
    def index_key(self) -> tuple[str, str, str]:
        return self.shard, self.audio_path, self.beatmap_path


_INDEXED_FRACTIONAL_BPM_FIXTURES = (
    _IndexedFractionalBpmFixture(
        name="ninth_111_dear_you",
        family_part=1.0 / 9.0,
        shard="0",
        audio_path="286309/Dear You.mp3",
        beatmap_path="286309/DJ Genericname - Dear You (Satoshi Kazuki) [HD+].osu",
        frame_count=10000,
    ),
    _IndexedFractionalBpmFixture(
        name="ninth_222_freedom_dive",
        family_part=2.0 / 9.0,
        shard="0",
        audio_path="173612/Freedom Dive.mp3",
        beatmap_path="173612/xi - FREEDOM DiVE (razlteh) [4K Hyper].osu",
    ),
    _IndexedFractionalBpmFixture(
        name="eighth_125_ultra_beatdown",
        family_part=1.0 / 8.0,
        shard="0",
        audio_path="728851/ULTRA BEATDOWN SUPREME.mp3",
        beatmap_path="728851/DragonForce - ULTRA BEATDOWN SUPREME (IcyWorld) [Marathon].osu",
    ),
    _IndexedFractionalBpmFixture(
        name="eighth_375_swagg_anthem",
        family_part=3.0 / 8.0,
        shard="0",
        audio_path="2021203/audio.ogg",
        beatmap_path="2021203/natimernero! - #SWAGG ANTHEM! (Relae) [CHAT, OPZIONI, BLOCCO!].osu",
    ),
    _IndexedFractionalBpmFixture(
        name="eighth_875_impulse",
        family_part=7.0 / 8.0,
        shard="0",
        audio_path="1349658/Impulse.mp3",
        beatmap_path="1349658/Culprate & Au5 - Impulse (Pope Gadget) [The Reaction].osu",
    ),
    _IndexedFractionalBpmFixture(
        name="third_333_wizdomiot",
        family_part=1.0 / 3.0,
        shard="0",
        audio_path="1360248/audio.mp3",
        beatmap_path="1360248/LeaF - Wizdomiot (extended ver.) (FAMoss) [Green-eyed Jealousy].osu",
    ),
    _IndexedFractionalBpmFixture(
        name="third_667_nest",
        family_part=2.0 / 3.0,
        shard="0",
        audio_path="576883/Nest 1.2.mp3",
        beatmap_path="576883/Cardboard Box - Nest (Guilhermeziat) [Yolk 1.2].osu",
    ),
)


def _indexed_fractional_fixture_keys() -> set[tuple[str, str, str]]:
    import pandas as pd

    index_frame = pd.read_parquet(_FRACTIONAL_FIXTURE_INDEX_PATH)
    return {
        (str(row.shard), str(row.audio_path), str(row.beatmap_path))
        for row in index_frame.itertuples(index=False)
    }


def _red_timing_point_nearest_fractional_part(
    beatmap_path: Path,
    *,
    fractional_part: float,
) -> tuple[float, float]:
    best_distance = math.inf
    best_beat_length_ms = math.nan
    best_bpm = math.nan
    for timing_point in require_red_timing_points(beatmap_path):
        bpm = 60000.0 / timing_point.beat_length_ms
        if bpm < 80.0 or bpm > 300.0:
            continue
        fractional_distance = abs((bpm - math.floor(bpm)) - fractional_part)
        if fractional_distance < best_distance:
            best_distance = fractional_distance
            best_beat_length_ms = timing_point.beat_length_ms
            best_bpm = bpm
    if not math.isfinite(best_distance) or best_distance > 0.0035:
        raise AssertionError(
            f"{beatmap_path} has no indexed red BPM near fractional part {fractional_part:.6f}"
        )
    return best_beat_length_ms, best_bpm


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


def _alternating_pulse(
    *,
    frame_count: int,
    frame_rate_hz: float,
    offset_ms: float,
    beat_length_ms: float,
    odd_amplitude: float,
    even_amplitude: float = 1.0,
    pulse_width_ms: float = 40.0,
) -> np.ndarray:
    frame_times_ms = np.arange(frame_count, dtype=np.float64) / frame_rate_hz * 1000.0
    beat_times_ms = np.arange(offset_ms, frame_times_ms[-1] + beat_length_ms, beat_length_ms, dtype=np.float64)
    signal = np.zeros_like(frame_times_ms)
    for beat_index, beat_time_ms in enumerate(beat_times_ms):
        amplitude = even_amplitude if beat_index % 2 == 0 else odd_amplitude
        distance_ms = np.abs(frame_times_ms - beat_time_ms)
        signal = np.maximum(signal, amplitude * np.maximum(0.0, 1.0 - distance_ms / pulse_width_ms))
    return signal


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
    def test_expands_half_bpm_candidates_with_hardcoded_fractional_parts(self) -> None:
        config = GridFitterConfig(min_bpm=220.0, max_bpm=224.0)

        candidates = scoring_module._with_fractional_bpm_candidates(
            np.asarray([222.0, 222.5], dtype=np.float64),
            config=config,
        )

        self.assertTrue(np.any(np.isclose(candidates, 222.0 + 2.0 / 9.0)))
        self.assertTrue(np.any(np.isclose(candidates, 222.0 + 1.0 / 8.0)))
        self.assertTrue(np.any(np.isclose(candidates, 222.0 + 1.0 / 3.0)))
        self.assertTrue(np.any(np.isclose(candidates, 222.0 + 2.0 / 3.0)))

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

    def test_fits_fractional_ninth_family_bpm_from_beat_probabilities(self) -> None:
        target_bpm = 222.0 + 2.0 / 9.0
        prediction = _synthetic_prediction(
            frame_count=3000,
            offset_ms=0.0,
            beat_length_ms=60000.0 / target_bpm,
        )
        fitter = GridFitter(
            GridFitterConfig(
                min_bpm=200.0,
                max_bpm=240.0,
                max_segments=1,
                max_grid_candidates_per_segment=2000,
            )
        )

        result = fitter.fit(prediction)

        self.assertAlmostEqual(result.grid.segments[0].local_bpm, target_bpm, delta=1e-6)

    def test_fits_indexed_fractional_red_tempos_from_rendered_oracles(self) -> None:
        if not _FRACTIONAL_FIXTURE_INDEX_PATH.exists():
            self.skipTest("fractional BPM fixture index is not available")
        indexed_keys = _indexed_fractional_fixture_keys()
        fitter = GridFitter(
            GridFitterConfig(
                min_bpm=80.0,
                max_bpm=300.0,
                max_segments=1,
                max_grid_candidates_per_segment=3000,
            )
        )

        for fixture in _INDEXED_FRACTIONAL_BPM_FIXTURES:
            with self.subTest(fixture=fixture.name):
                self.assertIn(fixture.index_key, indexed_keys)
                if not fixture.dataset_audio_path.exists() or not fixture.dataset_beatmap_path.exists():
                    self.skipTest(f"{fixture.name} indexed fixture files are not available")
                beat_length_ms, expected_bpm = _red_timing_point_nearest_fractional_part(
                    fixture.dataset_beatmap_path,
                    fractional_part=fixture.family_part,
                )
                oracle_grid = FittedTimingGrid(
                    segments=(TimingSegment(offset_ms=0.0, beat_length_ms=beat_length_ms),)
                )
                track = render_dense_timing_v2(
                    oracle_grid,
                    input_start_ms=0.0,
                    frame_count=fixture.frame_count,
                )
                prediction = FrameTimingPrediction(
                    provider="indexed-oracle-rendered",
                    beat_prob=track[:, 0],
                    downbeat_prob=np.zeros(track.shape[0], dtype=np.float32),
                    frame_rate_hz=50.0,
                    source_path=fixture.dataset_audio_path.as_posix(),
                )

                result = fitter.fit(prediction)
                fitted_bpm = result.grid.segments[0].local_bpm
                nearest_half_bpm = round(expected_bpm * 2.0) / 2.0

                self.assertLess(abs(fitted_bpm - expected_bpm), abs(nearest_half_bpm - expected_bpm))
                self.assertAlmostEqual(fitted_bpm, expected_bpm, delta=0.005)

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

    def test_post_pass_canonicalizes_adjacent_alias_tempo_switches(self) -> None:
        frame_times_ms = np.arange(1200, dtype=np.float64) / 50.0 * 1000.0
        signal = _synthetic_pulse(
            frame_count=1200,
            frame_rate_hz=50.0,
            offset_ms=0.0,
            beat_length_ms=500.0,
        ).astype(np.float64)
        segment_fits = (
            _SegmentFit(0, 600, 0.8, 500.0, 0.0, 0.1, 0.2, 120.0, 0.8, 1.0, 10),
            _SegmentFit(600, 1200, 0.8, 1000.0, 0.0, 0.1, 0.2, 60.0, 0.8, 1.0, 10),
        )

        canonical_result = _canonicalize_tempo_aliases(
            segment_fits,
            signal,
            frame_times_ms=frame_times_ms,
            downbeat_signal=np.zeros_like(signal),
            config=GridFitterConfig(),
        )
        canonical_fits = canonical_result.segment_fits
        segments = _timing_segments_from_fits(canonical_fits, frame_times_ms, config=GridFitterConfig())

        self.assertAlmostEqual(canonical_fits[1].bpm, 120.0, delta=1e-6)
        self.assertEqual(_segment_alias_switch_count(segments, config=GridFitterConfig()), 0)
        self.assertLessEqual(canonical_result.alias_candidate_count, 16)

    def test_post_pass_can_lower_promoted_alias_tempo_switches(self) -> None:
        frame_times_ms = np.arange(1200, dtype=np.float64) / 50.0 * 1000.0
        signal = _synthetic_pulse(
            frame_count=1200,
            frame_rate_hz=50.0,
            offset_ms=0.0,
            beat_length_ms=500.0,
        ).astype(np.float64)
        segment_fits = (
            _SegmentFit(0, 600, 0.95, 500.0, 0.0, 0.1, 0.2, 120.0, 0.95, 1.0, 10),
            _SegmentFit(600, 1200, 0.85, 250.0, 0.0, 0.1, 0.85, 120.0, 0.84, 2.0, 10),
        )

        canonical_result = _canonicalize_tempo_aliases(
            segment_fits,
            signal,
            frame_times_ms=frame_times_ms,
            downbeat_signal=np.zeros_like(signal),
            config=GridFitterConfig(),
        )
        canonical_fits = canonical_result.segment_fits
        segments = _timing_segments_from_fits(canonical_fits, frame_times_ms, config=GridFitterConfig())

        self.assertAlmostEqual(canonical_fits[1].bpm, 120.0, delta=1e-6)
        self.assertEqual(canonical_fits[1].tempo_multiplier, 1.0)
        self.assertEqual(_segment_alias_switch_count(segments, config=GridFitterConfig()), 0)

    def test_post_pass_rejects_supported_first_segment_demotion(self) -> None:
        frame_times_ms = np.arange(2400, dtype=np.float64) / 50.0 * 1000.0
        signal = _alternating_pulse(
            frame_count=2400,
            frame_rate_hz=50.0,
            offset_ms=0.0,
            beat_length_ms=320.0,
            odd_amplitude=0.7,
        ).astype(np.float64)
        segment_fits = (
            _SegmentFit(0, 2400, 0.5, 320.0, 0.0, 0.1, 0.2, 187.5, 0.5, 1.0, 10),
        )
        config = GridFitterConfig(
            alias_current_tempo_bonus=0.0,
            alias_preferred_band_bonus=0.0,
        )

        canonical_result = _canonicalize_tempo_aliases(
            segment_fits,
            signal,
            frame_times_ms=frame_times_ms,
            downbeat_signal=np.zeros_like(signal),
            config=config,
        )

        self.assertAlmostEqual(canonical_result.segment_fits[0].bpm, 187.5, delta=1e-6)
        self.assertGreater(canonical_result.alias_candidate_count, 0)

    def test_alias_post_pass_preserves_grid_candidate_count_diagnostics(self) -> None:
        prediction = _synthetic_prediction(
            frame_count=1800,
            offset_ms=0.0,
            beat_length_ms=500.0,
        )
        base_config = GridFitterConfig(canonicalize_tempo_aliases=False)
        alias_config = GridFitterConfig(canonicalize_tempo_aliases=True)

        base_result = GridFitter(base_config).fit(prediction)
        alias_result = GridFitter(alias_config).fit(prediction)

        self.assertEqual(alias_result.diagnostics.candidate_count, base_result.diagnostics.candidate_count)
        self.assertGreater(alias_result.diagnostics.alias_candidate_count, 0)

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
