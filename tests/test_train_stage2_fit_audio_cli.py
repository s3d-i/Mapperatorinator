import contextlib
import io
import json
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from train.stage_2.timing.schema import FrameTimingPrediction


def _synthetic_prediction() -> FrameTimingPrediction:
    frame_count = 1000
    frame_rate_hz = 50.0
    frame_times_ms = np.arange(frame_count, dtype=np.float64) / frame_rate_hz * 1000.0
    beat_length_ms = 500.0
    offset_ms = 120.0
    phase = ((frame_times_ms - offset_ms) / beat_length_ms) % 1.0
    distance_ms = np.minimum(phase, 1.0 - phase) * beat_length_ms
    beat_prob = np.maximum(0.0, 1.0 - distance_ms / 40.0).astype(np.float32)
    return FrameTimingPrediction(
        provider="fake-beat-this",
        checkpoint_path="fake-checkpoint",
        source_path="song.mp3",
        beat_prob=beat_prob,
        downbeat_prob=np.zeros(frame_count, dtype=np.float32),
        frame_rate_hz=frame_rate_hz,
    )


class _FakeBeatThisTimingProvider:
    def __init__(self, *, checkpoint_path: str, device: str, float16: bool) -> None:
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.float16 = float16

    def predict_file(self, audio_path: Path) -> FrameTimingPrediction:
        self.audio_path = audio_path
        return _synthetic_prediction()


class Stage2FitAudioCliTest(unittest.TestCase):
    def test_main_outputs_timing_segments_as_text(self) -> None:
        from train.stage_2.timing import fit_audio

        stdout = io.StringIO()
        with mock.patch.object(fit_audio, "BeatThisTimingProvider", _FakeBeatThisTimingProvider):
            with contextlib.redirect_stdout(stdout):
                exit_code = fit_audio.main(["song.mp3"])

        self.assertEqual(exit_code, 0)
        output = stdout.getvalue()
        self.assertIn("source: song.mp3", output)
        self.assertIn("segments:", output)
        self.assertIn("offset_ms=120.000", output)
        self.assertIn("beat_length_ms=500.000", output)
        self.assertIn("bpm=120.000", output)

    def test_main_can_emit_json(self) -> None:
        from train.stage_2.timing import fit_audio

        stdout = io.StringIO()
        with mock.patch.object(fit_audio, "BeatThisTimingProvider", _FakeBeatThisTimingProvider):
            with contextlib.redirect_stdout(stdout):
                exit_code = fit_audio.main(["song.mp3", "--json"])

        self.assertEqual(exit_code, 0)
        report = json.loads(stdout.getvalue())
        self.assertEqual(report["source_path"], "song.mp3")
        self.assertEqual(report["provider"], "fake-beat-this")
        self.assertEqual(report["segments"][0]["offset_ms"], 120.0)
        self.assertEqual(report["segments"][0]["beat_length_ms"], 500.0)
        self.assertEqual(report["segments"][0]["bpm"], 120.0)


if __name__ == "__main__":
    unittest.main()
