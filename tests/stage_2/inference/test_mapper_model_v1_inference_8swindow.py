import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from train.stage_2.inference.mapper_model_v1_inference_8swindow import (
    DEFAULT_CHECKPOINT_PATH,
    InferenceRunConfig,
    TimingConfig,
    _audio_filename_for_output,
    _ensure_completed_generation,
    _make_generation_generator,
    build_timing_grid,
    load_run_config,
    prepare_feature_batch,
)
from train.stage_2.inference.osu_stream import OsuStreamMetadata, format_osu_stream
from train.stage1_oracle.events.canonical import CanonicalTimepoint
from train.stage1_oracle.events.canonical import LaneAction as CanonicalLaneAction
from train.stage_2.model_mapper_v1.generation import grammar_constrained_window_generation
from train.stage_2.model_mapper_v1.replay import empty_ln_carry_state
from train.stage_2.model_mapper_v1.tokenizer import MAPPER_WRITE_MS
from train.stage_2.model_mapper_v1.vocab import MapperV1Vocab


class MapperModelV1Inference8sWindowTests(unittest.TestCase):
    def test_load_run_config_reads_milestone_defaults(self) -> None:
        config = load_run_config(
            "train/stage_2/inference/configs/mapper_model_v1_inference_8swindow.yaml",
        )

        self.assertIsInstance(config, InferenceRunConfig)
        self.assertEqual(config.checkpoint_path, DEFAULT_CHECKPOINT_PATH)
        self.assertEqual(config.difficulty, 4.49)
        self.assertEqual(config.timing.mode, "constant")
        self.assertEqual(config.max_tokens, 512)

    def test_constant_timing_grid_is_used_for_osu_timing_points(self) -> None:
        grid = build_timing_grid(
            audio_path=Path("audio.mp3"),
            timing_config=TimingConfig(mode="constant", bpm=150.0, offset_ms=-20.0, meter=4),
        )
        timepoints = [
            CanonicalTimepoint(
                time_ms=100,
                lane_actions=(
                    CanonicalLaneAction.TAP,
                    CanonicalLaneAction.NONE,
                    CanonicalLaneAction.NONE,
                    CanonicalLaneAction.NONE,
                ),
            ),
        ]

        text = format_osu_stream(
            timepoints,
            metadata=OsuStreamMetadata(audio_filename="audio.mp3"),
            timing_grid=grid,
        )

        self.assertIn("[TimingPoints]\n-20,400,4,2,0,100,1,0", text)
        self.assertIn("[HitObjects]\n64,192,100,1,0,0:0:0:0:", text)

    def test_load_run_config_accepts_stdout_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "run.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        f"checkpoint_path: {DEFAULT_CHECKPOINT_PATH.as_posix()}",
                        "audio_path: song.mp3",
                        "difficulty: 4.0",
                        "output_path: stdout",
                    ],
                ),
                encoding="utf-8",
            )

            config = load_run_config(config_path)

        self.assertIsNone(config.output_path)

    def test_incomplete_generation_is_rejected_before_export(self) -> None:
        vocab = MapperV1Vocab()
        generated = grammar_constrained_window_generation(
            vocab=vocab,
            write_start_ms=0,
            write_end_ms=MAPPER_WRITE_MS,
            ln_carry_in=empty_ln_carry_state(0),
            ln_carry_out=empty_ln_carry_state(MAPPER_WRITE_MS),
            is_full_chart_start=True,
            is_full_chart_end=True,
            max_tokens=1,
        )

        with self.assertRaisesRegex(RuntimeError, "did not complete the exact 8s window"):
            _ensure_completed_generation(generated)

    def test_prepare_feature_batch_rejects_short_audio(self) -> None:
        short_mel = np.zeros((399, 160), dtype=np.float32)

        with patch(
            "train.stage_2.inference.mapper_model_v1_inference_8swindow.load_full_song_packed_mel_20ms",
            return_value=short_mel,
        ):
            with self.assertRaisesRegex(ValueError, "requires at least 400 packed 20ms frames"):
                prepare_feature_batch(
                    audio_path=Path("short.mp3"),
                    difficulty=4.0,
                    timing_config=TimingConfig(mode="constant"),
                    device=torch.device("cpu"),
                )

    def test_seeded_generator_uses_inference_device(self) -> None:
        generator = _make_generation_generator(123, device=torch.device("cpu"))

        assert generator is not None
        self.assertEqual(generator.initial_seed(), 123)
        self.assertEqual(str(generator.device), "cpu")

    def test_audio_filename_is_relative_to_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_path = root / "out" / "map.osu"
            audio_path = root / "audio" / "song.mp3"

            filename = _audio_filename_for_output(audio_path, output_path)

        self.assertEqual(filename, "../audio/song.mp3")


if __name__ == "__main__":
    unittest.main()
