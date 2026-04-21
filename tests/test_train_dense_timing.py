import math
import unittest

import numpy as np

from train.stage1_oracle.features.timing import (
    TIMING_TRACK_CHANNELS,
    TimingTrackConfig,
    render_timing_track_20ms_v1,
)
from train.stage1_oracle.osu.timing import RedTimingPoint


class TrainDenseTimingFeatureTests(unittest.TestCase):
    def test_renderer_outputs_fixed_600_by_5_window_and_phase_unit_vectors(self) -> None:
        track = render_timing_track_20ms_v1(
            [RedTimingPoint(offset_ms=0.0, beat_length_ms=500.0)],
            input_start_ms=-2000.0,
            bpm_log_mean=math.log(120.0),
            bpm_log_std=1.0,
        )

        self.assertEqual(track.shape, (600, 5))
        self.assertEqual(
            TIMING_TRACK_CHANNELS,
            (
                "beat_pulse",
                "beat_phase_sin",
                "beat_phase_cos",
                "local_bpm_log_norm",
                "timing_confidence",
            ),
        )
        phase_norm = np.sqrt(track[:, 1] ** 2 + track[:, 2] ** 2)
        self.assertLess(float(np.max(np.abs(phase_norm - 1.0))), 1e-6)
        self.assertTrue(np.all(track[:, 4] == 1.0))

    def test_renderer_uses_frame_centers_triangular_pulses_and_red_timing_sections(self) -> None:
        track = render_timing_track_20ms_v1(
            [
                RedTimingPoint(offset_ms=0.0, beat_length_ms=500.0),
                RedTimingPoint(offset_ms=1000.0, beat_length_ms=250.0),
            ],
            input_start_ms=960.0,
            frame_count=4,
            bpm_log_mean=math.log(120.0),
            bpm_log_std=1.0,
        )

        # Frame centers are 970, 990, 1010, and 1030ms.
        self.assertAlmostEqual(float(track[0, 0]), 0.25)
        self.assertAlmostEqual(float(track[1, 0]), 0.75)
        self.assertAlmostEqual(float(track[2, 0]), 0.75)
        self.assertAlmostEqual(float(track[3, 0]), 0.25)
        self.assertAlmostEqual(float(track[1, 3]), 0.0, places=6)
        self.assertAlmostEqual(float(track[2, 3]), math.log(2.0), places=6)

    def test_renderer_clips_local_bpm_log_norm_and_rejects_bad_stats(self) -> None:
        config = TimingTrackConfig(pulse_width_ms=40.0)

        fast_track = render_timing_track_20ms_v1(
            [RedTimingPoint(offset_ms=0.0, beat_length_ms=100.0)],
            input_start_ms=0.0,
            frame_count=1,
            bpm_log_mean=math.log(1.0),
            bpm_log_std=0.1,
            config=config,
        )
        self.assertEqual(float(fast_track[0, 3]), 4.0)

        with self.assertRaisesRegex(ValueError, "bpm_log_std must be positive"):
            render_timing_track_20ms_v1(
                [RedTimingPoint(offset_ms=0.0, beat_length_ms=500.0)],
                input_start_ms=0.0,
                frame_count=1,
                bpm_log_mean=math.log(120.0),
                bpm_log_std=0.0,
            )

    def test_renderer_rejects_implausible_positive_red_timing(self) -> None:
        with self.assertRaisesRegex(ValueError, "implausible"):
            render_timing_track_20ms_v1(
                [RedTimingPoint(offset_ms=0.0, beat_length_ms=1e-100)],
                input_start_ms=0.0,
                frame_count=1,
                bpm_log_mean=math.log(120.0),
                bpm_log_std=1.0,
            )


if __name__ == "__main__":
    unittest.main()
