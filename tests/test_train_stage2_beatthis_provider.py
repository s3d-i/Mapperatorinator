import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from train.stage_2.timing.providers import beatthis
from train.stage_2.timing.providers.beatthis import (
    BEATTHIS_FRAME_RATE_HZ,
    DEFAULT_BEATTHIS_CHECKPOINT,
    DEFAULT_BEATTHIS_DEVICE,
    BeatThisTimingProvider,
)


class _FakeAudio2Frames:
    instances: list["_FakeAudio2Frames"] = []

    def __init__(self, checkpoint_path: str, device: str, float16: bool) -> None:
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.float16 = float16
        self.calls: list[tuple[np.ndarray, int]] = []
        self.__class__.instances.append(self)

    def __call__(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
        self.calls.append((audio, sample_rate))
        return (
            np.asarray([-2.0, 0.0, 2.0], dtype=np.float32),
            np.asarray([2.0, 0.0, -2.0], dtype=np.float32),
        )


def _fake_load_audio(path: str | Path) -> tuple[np.ndarray, int]:
    return np.asarray([0.25, -0.25, 0.0], dtype=np.float32), 44100


class BeatThisTimingProviderTest(unittest.TestCase):
    def setUp(self) -> None:
        _FakeAudio2Frames.instances.clear()

    def test_predict_audio_uses_default_top_level_device_const(self) -> None:
        audio = np.asarray([0.0, 0.5, -0.5], dtype=np.float32)
        provider = BeatThisTimingProvider()

        with mock.patch.object(
            beatthis,
            "_load_beat_this_api",
            return_value=beatthis.BeatThisAPI(_FakeAudio2Frames, _fake_load_audio),
        ):
            prediction = provider.predict_audio(audio, sample_rate=22050)

        self.assertEqual(len(_FakeAudio2Frames.instances), 1)
        frame_model = _FakeAudio2Frames.instances[0]
        self.assertEqual(frame_model.checkpoint_path, DEFAULT_BEATTHIS_CHECKPOINT)
        self.assertEqual(frame_model.device, DEFAULT_BEATTHIS_DEVICE)
        self.assertFalse(frame_model.float16)
        self.assertEqual(len(frame_model.calls), 1)
        np.testing.assert_array_equal(frame_model.calls[0][0], audio)
        self.assertEqual(frame_model.calls[0][1], 22050)

        self.assertEqual(prediction.provider, "beat-this")
        self.assertEqual(prediction.checkpoint_path, DEFAULT_BEATTHIS_CHECKPOINT)
        self.assertEqual(prediction.frame_rate_hz, BEATTHIS_FRAME_RATE_HZ)
        np.testing.assert_allclose(
            prediction.beat_prob,
            np.asarray([0.11920292, 0.5, 0.88079708], dtype=np.float32),
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            prediction.downbeat_prob,
            np.asarray([0.88079708, 0.5, 0.11920292], dtype=np.float32),
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            prediction.frame_times_seconds,
            np.asarray([0.0, 0.02, 0.04], dtype=np.float32),
            rtol=1e-6,
        )

    def test_predict_file_loads_audio_with_beat_this_loader(self) -> None:
        provider = BeatThisTimingProvider(device="cpu")

        with mock.patch.object(
            beatthis,
            "_load_beat_this_api",
            return_value=beatthis.BeatThisAPI(_FakeAudio2Frames, _fake_load_audio),
        ):
            prediction = provider.predict_file(Path("song.mp3"))

        self.assertEqual(len(_FakeAudio2Frames.instances), 1)
        frame_model = _FakeAudio2Frames.instances[0]
        self.assertEqual(len(frame_model.calls), 1)
        np.testing.assert_array_equal(
            frame_model.calls[0][0],
            np.asarray([0.25, -0.25, 0.0], dtype=np.float32),
        )
        self.assertEqual(frame_model.calls[0][1], 44100)
        self.assertEqual(prediction.source_path, "song.mp3")

    def test_rejects_mismatched_frame_output_lengths(self) -> None:
        class BadAudio2Frames(_FakeAudio2Frames):
            def __call__(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
                return np.asarray([0.0, 1.0], dtype=np.float32), np.asarray([0.0], dtype=np.float32)

        provider = BeatThisTimingProvider(device="cpu")

        with mock.patch.object(
            beatthis,
            "_load_beat_this_api",
            return_value=beatthis.BeatThisAPI(BadAudio2Frames, _fake_load_audio),
        ):
            with self.assertRaisesRegex(ValueError, "same length"):
                provider.predict_audio(np.asarray([0.0], dtype=np.float32), sample_rate=22050)


if __name__ == "__main__":
    unittest.main()
