import io
import math
import struct
import tempfile
import unittest
import wave
from contextlib import redirect_stdout
from pathlib import Path

from train.difficulty import calculate_mania_difficulty, main


def _write_wav(path: Path, *, sample_rate: int = 22050, duration_seconds: float = 0.2, frequency: float = 440.0) -> None:
    num_samples = int(sample_rate * duration_seconds)
    amplitude = 16000
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        for index in range(num_samples):
            sample = int(amplitude * math.sin(2 * math.pi * frequency * index / sample_rate))
            wav_file.writeframes(struct.pack("<h", sample))


def _write_mania_osu(path: Path, *, audio_filename: str) -> None:
    path.write_text(
        "\n".join(
            [
                "osu file format v14",
                "",
                "[General]",
                f"AudioFilename: {audio_filename}",
                "Mode: 3",
                "",
                "[Metadata]",
                "Version:Test SR",
                "",
                "[Difficulty]",
                "CircleSize:4",
                "OverallDifficulty:8",
                "",
                "[HitObjects]",
                "64,192,0,1,0,0:0:0:0:",
                "192,192,100,1,0,0:0:0:0:",
                "320,192,200,1,0,0:0:0:0:",
                "448,192,300,1,0,0:0:0:0:",
                "64,192,400,128,0,700:0:0:0:0:",
                "192,192,500,1,0,0:0:0:0:",
                "320,192,650,128,0,900:0:0:0:0:",
                "448,192,800,1,0,0:0:0:0:",
            ],
        ),
        encoding="utf-8",
    )


class TrainDifficultyTests(unittest.TestCase):
    def test_calculate_mania_difficulty_matches_reference_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audio_path = root / "song.wav"
            osu_path = root / "chart.osu"
            _write_wav(audio_path)
            _write_mania_osu(osu_path, audio_filename="song.wav")

            difficulty_1x = calculate_mania_difficulty(osu_path, audio_path, speed=1.0)
            difficulty_15x = calculate_mania_difficulty(osu_path, audio_path, speed=1.5)

            self.assertAlmostEqual(difficulty_1x, 0.26729119012271224)
            self.assertAlmostEqual(difficulty_15x, 0.33449854624310965)

    def test_calculate_mania_difficulty_rejects_audio_filename_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audio_path = root / "different.wav"
            osu_path = root / "chart.osu"
            _write_wav(audio_path)
            _write_mania_osu(osu_path, audio_filename="song.wav")

            with self.assertRaisesRegex(ValueError, "AudioFilename"):
                calculate_mania_difficulty(osu_path, audio_path, speed=1.0)

    def test_cli_prints_two_decimal_difficulty(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audio_path = root / "song.wav"
            osu_path = root / "chart.osu"
            _write_wav(audio_path)
            _write_mania_osu(osu_path, audio_filename="song.wav")

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = main([str(osu_path), str(audio_path), "--speed", "1.5"])

            self.assertEqual(exit_code, 0)
            self.assertEqual(stdout.getvalue(), "0.33\n")


if __name__ == "__main__":
    unittest.main()
