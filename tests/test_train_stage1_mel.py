import tempfile
import unittest
from pathlib import Path

import numpy as np

from train.stage1_oracle.features.mel import (
    MelCacheConfig,
    compute_log_mel_10ms,
    load_or_create_log_mel_cache,
    pack_mel_20ms_window,
)


class Stage1MelFeatureTests(unittest.TestCase):
    def test_compute_log_mel_outputs_80_bin_10ms_frames(self) -> None:
        waveform = np.zeros(16000, dtype=np.float32)
        mel = compute_log_mel_10ms(waveform, sample_rate=16000)

        self.assertEqual(mel.dtype, np.float32)
        self.assertEqual(mel.shape[1], 80)
        self.assertEqual(mel.shape[0], 100)

    def test_compute_log_mel_pads_to_full_10ms_time_axis(self) -> None:
        sample_rate = 16000
        sample_times = np.arange(sample_rate, dtype=np.float32) / sample_rate
        waveform = np.sin(2.0 * np.pi * 440.0 * sample_times).astype(np.float32)

        mel = compute_log_mel_10ms(waveform, sample_rate=sample_rate)
        packed = pack_mel_20ms_window(mel, input_start_ms=0, frame_count=50)

        self.assertEqual(mel.shape, (100, 80))
        np.testing.assert_array_equal(packed[-1, :80], mel[98])
        np.testing.assert_array_equal(packed[-1, 80:], mel[99])

    def test_pair_pack_uses_silence_padding_outside_audio(self) -> None:
        mel = np.arange(10 * 80, dtype=np.float32).reshape(10, 80)
        packed = pack_mel_20ms_window(mel, input_start_ms=-20, frame_count=3)

        self.assertEqual(packed.shape, (3, 160))
        np.testing.assert_array_equal(packed[0], np.full(160, np.log(1e-5), dtype=np.float32))
        np.testing.assert_array_equal(packed[1, :80], mel[0])
        np.testing.assert_array_equal(packed[1, 80:], mel[1])
        np.testing.assert_array_equal(packed[2, :80], mel[2])
        np.testing.assert_array_equal(packed[2, 80:], mel[3])

    def test_cache_roundtrip_uses_stable_config_directory(self) -> None:
        waveform = np.zeros(3200, dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MelCacheConfig(cache_root=Path(tmpdir))
            first = load_or_create_log_mel_cache(
                waveform,
                sample_rate=16000,
                audio_cache_key="song",
                config=config,
            )
            second = load_or_create_log_mel_cache(
                waveform + 1.0,
                sample_rate=16000,
                audio_cache_key="song",
                config=config,
            )

        np.testing.assert_array_equal(first, second)

    def test_cache_directory_includes_mel_config_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            first = MelCacheConfig(cache_root=Path(tmpdir), fmax=8000.0)
            second = MelCacheConfig(cache_root=Path(tmpdir), fmax=7000.0)

            self.assertNotEqual(first.mel_config_hash, second.mel_config_hash)
            self.assertNotEqual(first.cache_dir, second.cache_dir)


if __name__ == "__main__":
    unittest.main()
