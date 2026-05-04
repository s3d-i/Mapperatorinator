import unittest

import numpy as np

from train.stage1_oracle.features.mel import LOG_MEL_SILENCE_VALUE
from train.stage_2.features.mel import (
    full_song_packed_mel_20ms_from_waveform,
    pack_full_song_mel_20ms,
)


class Stage2MelFeatureTests(unittest.TestCase):
    def test_pack_full_song_mel_pairs_10ms_frames_and_pads_odd_tail(self) -> None:
        mel = np.vstack(
            [
                np.full((1, 80), 1.0, dtype=np.float32),
                np.full((1, 80), 2.0, dtype=np.float32),
                np.full((1, 80), 3.0, dtype=np.float32),
            ]
        )

        packed = pack_full_song_mel_20ms(mel)

        self.assertEqual(packed.shape, (2, 160))
        np.testing.assert_array_equal(packed[0, :80], np.full(80, 1.0, dtype=np.float32))
        np.testing.assert_array_equal(packed[0, 80:], np.full(80, 2.0, dtype=np.float32))
        np.testing.assert_array_equal(packed[1, :80], np.full(80, 3.0, dtype=np.float32))
        np.testing.assert_array_equal(packed[1, 80:], np.full(80, LOG_MEL_SILENCE_VALUE, dtype=np.float32))

    def test_pack_full_song_mel_accepts_empty_track(self) -> None:
        packed = pack_full_song_mel_20ms(np.empty((0, 80), dtype=np.float32))

        self.assertEqual(packed.shape, (0, 160))

    def test_waveform_entrypoint_requires_config_sample_rate(self) -> None:
        with self.assertRaisesRegex(ValueError, "expected 16000Hz"):
            full_song_packed_mel_20ms_from_waveform(
                np.zeros(400, dtype=np.float32),
                sample_rate=44100,
                audio_cache_key="sample-rate-mismatch",
            )


if __name__ == "__main__":
    unittest.main()
