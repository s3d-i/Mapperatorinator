import math
import unittest

import numpy as np

from train.stage_2.timing.rendering.dense_timing_v2 import (
    DENSE_TIMING_V2_CHANNELS,
    DenseTimingV2Config,
    render_dense_timing_v2,
)
from train.stage_2.timing.schema import FittedTimingGrid, TimingSegment


class Stage2DenseTimingV2RendererTest(unittest.TestCase):
    def test_outputs_four_channel_window_without_timing_confidence(self) -> None:
        track = render_dense_timing_v2(
            FittedTimingGrid(segments=(TimingSegment(offset_ms=0.0, beat_length_ms=500.0),)),
            input_start_ms=-2000.0,
        )

        self.assertEqual(track.shape, (600, 4))
        self.assertEqual(
            DENSE_TIMING_V2_CHANNELS,
            ("beat_pulse", "phase_sin", "phase_cos", "local_bpm"),
        )
        phase_norm = np.sqrt(track[:, 1] ** 2 + track[:, 2] ** 2)
        self.assertLess(float(np.max(np.abs(phase_norm - 1.0))), 1e-6)
        self.assertTrue(np.all(track[:, 3] == np.float32(120.0)))

    def test_uses_frame_centers_pulses_phase_and_segment_local_bpm(self) -> None:
        track = render_dense_timing_v2(
            FittedTimingGrid(
                segments=(
                    TimingSegment(offset_ms=0.0, beat_length_ms=500.0),
                    TimingSegment(offset_ms=1000.0, beat_length_ms=250.0),
                )
            ),
            input_start_ms=960.0,
            frame_count=4,
        )

        np.testing.assert_allclose(track[:, 0], np.asarray([0.25, 0.75, 0.75, 0.25], dtype=np.float32))
        np.testing.assert_allclose(track[:, 3], np.asarray([120.0, 120.0, 240.0, 240.0], dtype=np.float32))

        expected_phase = np.asarray([0.94, 0.98, 0.04, 0.12], dtype=np.float64)
        np.testing.assert_allclose(track[:, 1], np.sin(2.0 * math.pi * expected_phase), rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(track[:, 2], np.cos(2.0 * math.pi * expected_phase), rtol=1e-6, atol=1e-6)

    def test_rejects_invalid_grid_and_config_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "beat_length_ms must be positive"):
            FittedTimingGrid(segments=(TimingSegment(offset_ms=0.0, beat_length_ms=0.0),))

        with self.assertRaisesRegex(ValueError, "segments must be non-empty"):
            FittedTimingGrid(segments=())

        with self.assertRaisesRegex(ValueError, "pulse_width_ms must be positive"):
            render_dense_timing_v2(
                FittedTimingGrid(segments=(TimingSegment(offset_ms=0.0, beat_length_ms=500.0),)),
                input_start_ms=0.0,
                frame_count=1,
                config=DenseTimingV2Config(pulse_width_ms=0.0),
            )


if __name__ == "__main__":
    unittest.main()
