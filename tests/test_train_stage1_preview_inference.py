import math
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from train.stage1_oracle.events.canonical import LaneAction
from train.stage1_oracle.events.tokens import Stage1Vocab
from train.stage1_oracle.events.windowing import WindowSpec
from train.stage1_oracle.features.timing import TIMING_TRACK_VERSION, render_timing_track_20ms_v1
from train.stage1_oracle.inference.assembler import StreamingAssembler
from train.stage1_oracle.inference.canvas_preview import CANVAS_PREVIEW_HTML
from train.stage1_oracle.inference.checkpoint_runner import Stage1CheckpointRunner
from train.stage1_oracle.inference.feature_source import FullSongFeatureSource
from train.stage1_oracle.inference.preview_server import (
    PreviewServer,
    PreviewEvent,
    build_preview_event_stream,
    create_preview_server,
)
from train.stage1_oracle.inference.scheduler import BufferPolicy, DecodedTokenWindow
from train.stage1_oracle.inference.stream_probe import (
    find_latest_checkpoint,
    format_probe_event,
    resolve_probe_inputs,
)
from train.stage1_oracle.models.mapper import Stage1OracleMapper, Stage1OracleMapperConfig
from train.stage1_oracle.osu.timing import RedTimingPoint


def _lane_actions(*actions: LaneAction) -> tuple[LaneAction, ...]:
    padded = list(actions)
    while len(padded) < 4:
        padded.append(LaneAction.NONE)
    return tuple(padded)


def _target_tokens(vocab: Stage1Vocab, events: list[tuple[int, tuple[LaneAction, ...]]]) -> list[int]:
    tokens: list[int] = []
    previous: int | None = None
    for time_ms, actions in events:
        delta = time_ms if previous is None else time_ms - previous
        while delta > 1000:
            tokens.append(vocab.ts_token_id(1000))
            delta -= 1000
        tokens.append(vocab.ts_token_id(delta))
        tokens.append(vocab.encode_timepoint_event(actions))
        previous = time_ms
    tokens.append(vocab.eos_id)
    return tokens


class Stage1PreviewFeatureSourceTests(unittest.TestCase):
    def test_full_song_features_render_oracle_timing_for_fixed_12s_window_context(self) -> None:
        bpm_log_mean = math.log(120.0)
        bpm_log_std = 1.0

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audio_path = root / "audio.wav"
            beatmap_path = root / "map.osu"
            beatmap_path.write_text(
                "\n".join(
                    [
                        "osu file format v14",
                        "",
                        "[TimingPoints]",
                        "1000,500,4,2,0,100,1,0",
                    ],
                ),
                encoding="utf-8",
            )
            timing_points = [RedTimingPoint(offset_ms=1000.0, beat_length_ms=500.0)]
            window = WindowSpec(write_start_ms=0, write_end_ms=3000, input_start_ms=-2000, input_end_ms=10000)

            with patch(
                "train.stage1_oracle.inference.feature_source.load_audio_file",
                return_value=np.zeros(48000, dtype=np.float32),
            ):
                with patch(
                    "train.stage1_oracle.inference.feature_source.load_or_create_log_mel_cache",
                    return_value=np.zeros((300, 80), dtype=np.float32),
                ):
                    source = FullSongFeatureSource(
                        audio_path=audio_path,
                        beatmap_path=beatmap_path,
                        bpm_log_mean=bpm_log_mean,
                        bpm_log_std=bpm_log_std,
                    )
                    features = source.features_for_window(window)

        expected_timing = render_timing_track_20ms_v1(
            timing_points,
            input_start_ms=-2000,
            bpm_log_mean=bpm_log_mean,
            bpm_log_std=bpm_log_std,
        )
        np.testing.assert_allclose(features.timing_track.numpy(), expected_timing)
        self.assertEqual(features.packed_audio.shape, (600, 160))


class Stage1PreviewCheckpointRunnerTests(unittest.TestCase):
    def test_checkpoint_runner_loads_config_and_state_dict(self) -> None:
        vocab = Stage1Vocab()
        config = Stage1OracleMapperConfig(
            vocab_size=vocab.size,
            d_model=32,
            heads=4,
            encoder_layers=1,
            decoder_layers=1,
            ffn_dim=64,
            dropout=0.0,
            max_decode_len=16,
        )
        model = Stage1OracleMapper(config)
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config.__dict__,
                    "run_name": "unit",
                    "timing_track": {
                        "timing_track_version": TIMING_TRACK_VERSION,
                        "bpm_log_mean": 5.0,
                        "bpm_log_std": 0.25,
                    },
                },
                checkpoint_path,
            )

            runner = Stage1CheckpointRunner.load(checkpoint_path, device_name="cpu")

        self.assertEqual(runner.config.max_decode_len, 16)
        self.assertEqual(runner.checkpoint_name, "checkpoint.pt")
        self.assertEqual(runner.bpm_log_mean, 5.0)
        self.assertEqual(runner.bpm_log_std, 0.25)
        self.assertFalse(runner.model.training)

    def test_checkpoint_runner_rejects_checkpoint_without_timing_training_stats(self) -> None:
        vocab = Stage1Vocab()
        config = Stage1OracleMapperConfig(
            vocab_size=vocab.size,
            d_model=32,
            heads=4,
            encoder_layers=1,
            decoder_layers=1,
            ffn_dim=64,
            dropout=0.0,
            max_decode_len=16,
        )
        model = Stage1OracleMapper(config)
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config.__dict__,
                    "run_name": "unit",
                },
                checkpoint_path,
            )

            with self.assertRaisesRegex(ValueError, "checkpoint missing timing_track"):
                Stage1CheckpointRunner.inspect_checkpoint(checkpoint_path)

    def test_existing_stage1_artifact_exposes_checkpoint_config(self) -> None:
        checkpoint_path = Path("train/artifacts/runs/stage1_oracle/overfit_4_dropout0_3000_mps/checkpoint.pt")
        if not checkpoint_path.is_file():
            self.skipTest(f"local Stage 1 checkpoint artifact not available: {checkpoint_path}")

        checkpoint = Stage1CheckpointRunner.inspect_checkpoint(checkpoint_path)

        self.assertEqual(checkpoint["config"]["vocab_size"], Stage1Vocab().size)
        self.assertIn("model_state_dict", checkpoint)


class Stage1PreviewAssemblerTests(unittest.TestCase):
    def test_decoded_tokens_emit_tap_and_hold_notes_in_one_second_batches(self) -> None:
        vocab = Stage1Vocab()
        assembler = StreamingAssembler(vocab=vocab)

        batches = assembler.process_window(
            DecodedTokenWindow(
                write_start_ms=8000,
                write_end_ms=12000,
                token_ids=_target_tokens(
                    vocab,
                    [
                        (230, _lane_actions(LaneAction.TAP)),
                        (510, _lane_actions(LaneAction.NONE, LaneAction.NONE, LaneAction.HOLD_START)),
                        (920, _lane_actions(LaneAction.NONE, LaneAction.NONE, LaneAction.HOLD_END)),
                    ],
                ),
            ),
        )

        batch = next(item for item in batches if item.start_ms == 8000)
        self.assertEqual(
            batch.to_json()["notes"],
            [
                {"time_ms": 8230, "lane": 0, "kind": "tap"},
                {"time_ms": 8510, "lane": 2, "kind": "hold", "end_time_ms": 8920},
            ],
        )
        self.assertEqual(batch.generated_through_ms, 12000)

    def test_hold_continuation_across_adjacent_windows_delays_start_batch_until_closed(self) -> None:
        vocab = Stage1Vocab()
        assembler = StreamingAssembler(vocab=vocab)

        first_batches = assembler.process_window(
            DecodedTokenWindow(
                write_start_ms=0,
                write_end_ms=8000,
                token_ids=_target_tokens(vocab, [(7900, _lane_actions(LaneAction.HOLD_START))]),
            ),
        )
        second_batches = assembler.process_window(
            DecodedTokenWindow(
                write_start_ms=8000,
                write_end_ms=16000,
                token_ids=_target_tokens(vocab, [(100, _lane_actions(LaneAction.HOLD_END))]),
            ),
        )

        self.assertEqual([batch.start_ms for batch in first_batches], list(range(0, 7000, 1000)))
        delayed = next(batch for batch in second_batches if batch.start_ms == 7000)
        self.assertEqual(
            delayed.to_json()["notes"],
            [{"time_ms": 7900, "lane": 0, "kind": "hold", "end_time_ms": 8100}],
        )

    def test_window_batches_commit_empty_seconds_through_generated_time(self) -> None:
        vocab = Stage1Vocab()
        assembler = StreamingAssembler(vocab=vocab)

        batches = assembler.process_window(
            DecodedTokenWindow(
                write_start_ms=0,
                write_end_ms=8000,
                token_ids=_target_tokens(
                    vocab,
                    [
                        (0, _lane_actions(LaneAction.TAP)),
                        (1000, _lane_actions(LaneAction.NONE, LaneAction.TAP)),
                        (7990, _lane_actions(LaneAction.NONE, LaneAction.NONE, LaneAction.TAP)),
                    ],
                ),
            ),
        )

        self.assertEqual([batch.start_ms for batch in batches], list(range(0, 8000, 1000)))
        self.assertEqual([batch.end_ms for batch in batches], list(range(1000, 9000, 1000)))
        self.assertEqual(assembler.committed_through_ms, 8000)

    def test_invalid_hold_transition_fails_fast(self) -> None:
        vocab = Stage1Vocab()
        assembler = StreamingAssembler(vocab=vocab)

        with self.assertRaisesRegex(ValueError, "illegal target token"):
            assembler.process_window(
                DecodedTokenWindow(
                    write_start_ms=0,
                    write_end_ms=8000,
                    token_ids=_target_tokens(vocab, [(100, _lane_actions(LaneAction.HOLD_END))]),
                ),
            )

    def test_incomplete_decode_window_fails_before_committing_batches(self) -> None:
        vocab = Stage1Vocab()
        assembler = StreamingAssembler(vocab=vocab)

        with self.assertRaisesRegex(ValueError, "incomplete decoded window"):
            assembler.process_window(
                DecodedTokenWindow(
                    write_start_ms=0,
                    write_end_ms=8000,
                    token_ids=[vocab.ts_token_id(1000)],
                    max_decode_len_reached=True,
                    eos_emitted_by_model=False,
                    eos_forced_after_pending_ts=True,
                ),
            )
        self.assertEqual(assembler.committed_through_ms, 0)

    def test_max_decode_len_forced_eos_still_commits_completed_events(self) -> None:
        vocab = Stage1Vocab()
        assembler = StreamingAssembler(vocab=vocab)

        batches = assembler.process_window(
            DecodedTokenWindow(
                write_start_ms=0,
                write_end_ms=8000,
                token_ids=_target_tokens(vocab, [(100, _lane_actions(LaneAction.TAP))]),
                max_decode_len_reached=True,
                eos_emitted_by_model=False,
                eos_forced_after_pending_ts=True,
            ),
        )

        first_batch = batches[0]
        self.assertEqual(first_batch.start_ms, 0)
        self.assertEqual(first_batch.to_json()["notes"], [{"time_ms": 100, "lane": 0, "kind": "tap"}])

    def test_illegal_target_token_grammar_fails_before_committing_batches(self) -> None:
        vocab = Stage1Vocab()
        tap_event = vocab.encode_timepoint_event(_lane_actions(LaneAction.TAP))

        for token_ids in (
            [vocab.ts_token_id(500), vocab.eos_id],
            [vocab.ts_token_id(100), tap_event, tap_event, vocab.eos_id],
        ):
            assembler = StreamingAssembler(vocab=vocab)
            with self.subTest(token_ids=token_ids):
                with self.assertRaisesRegex(ValueError, "illegal target token"):
                    assembler.process_window(
                        DecodedTokenWindow(
                            write_start_ms=0,
                            write_end_ms=8000,
                            token_ids=token_ids,
                        ),
                    )
                self.assertEqual(assembler.committed_through_ms, 0)


class Stage1PreviewSchedulerServerTests(unittest.TestCase):
    def test_create_preview_server_rejects_out_of_range_difficulty_before_loading_files(self) -> None:
        with self.assertRaisesRegex(ValueError, r"difficulty outside supported 2\.0\*\.\.6\.0\* range"):
            create_preview_server(
                checkpoint_path="missing-checkpoint.pt",
                audio_path="missing-audio.mp3",
                beatmap_path="missing-map.osu",
                difficulty=6.25,
                device_name="cpu",
            )

    def test_buffer_policy_waits_for_startup_target_and_rebuffers_below_floor(self) -> None:
        policy = BufferPolicy(target_buffer_ms=2000, rebuffer_floor_ms=750)

        self.assertEqual(policy.update(playhead_ms=0, committed_through_ms=1990).state, "startup")
        self.assertEqual(policy.update(playhead_ms=0, committed_through_ms=2000).state, "ready")
        self.assertEqual(policy.update(playhead_ms=1300, committed_through_ms=2000).state, "buffering")
        self.assertEqual(policy.update(playhead_ms=1300, committed_through_ms=3400).state, "ready")

    def test_sse_stream_uses_validated_batches_without_raw_tokens(self) -> None:
        vocab = Stage1Vocab()

        class FakeSource:
            audio_duration_ms = 2500

            def iter_windows(self):
                return [
                    WindowSpec(write_start_ms=0, write_end_ms=2500, input_start_ms=-2000, input_end_ms=10000),
                ]

            def features_for_window(self, window):
                return object()

        class FakeRunner:
            checkpoint_name = "fake.pt"

            def decode_window(self, *, features, window, difficulty, open_hold_mask):
                return DecodedTokenWindow(
                    write_start_ms=window.write_start_ms,
                    write_end_ms=window.write_end_ms,
                    token_ids=_target_tokens(vocab, [(0, _lane_actions(LaneAction.TAP))]),
                )

        events = list(
            build_preview_event_stream(
                source=FakeSource(),
                runner=FakeRunner(),
                assembler=StreamingAssembler(vocab=vocab),
                difficulty=4.5,
                target_buffer_ms=2000,
            ),
        )

        event_names = [event.name for event in events]
        payload = "\n".join(event.to_sse() for event in events)
        self.assertIn("metadata", event_names)
        self.assertIn("ready", event_names)
        self.assertIn("batch", event_names)
        self.assertIn("done", event_names)
        self.assertNotIn("token", payload)

    def test_sse_stream_sanitizes_error_payloads(self) -> None:
        class FakeSource:
            audio_duration_ms = 2500

            def iter_windows(self):
                return [
                    WindowSpec(write_start_ms=0, write_end_ms=2500, input_start_ms=-2000, input_end_ms=10000),
                ]

            def features_for_window(self, window):
                return object()

        class FakeRunner:
            checkpoint_name = "fake.pt"

            def decode_window(self, *, features, window, difficulty, open_hold_mask):
                raise ValueError("DecodedTokenWindow token_ids=[1, 2, 3] failed")

        with self.assertLogs("train.stage1_oracle.inference.preview_server", level="ERROR"):
            events = list(
                build_preview_event_stream(
                    source=FakeSource(),
                    runner=FakeRunner(),
                    assembler=StreamingAssembler(),
                    difficulty=4.5,
                    target_buffer_ms=2000,
                ),
            )

        error = events[-1]
        payload = error.to_sse()
        self.assertEqual(error.name, "error")
        self.assertEqual(error.data["message"], "Stage 1 preview generation failed. Check server logs for details.")
        self.assertEqual(error.data["code"], "preview_stream_failed")
        self.assertNotIn("token", payload.lower())
        self.assertNotIn("[1,2,3]", payload.replace(" ", ""))

    def test_canvas_preview_blocks_audio_playback_until_ready(self) -> None:
        self.assertIn('<audio id="audio" src="/audio" preload="auto"></audio>', CANVAS_PREVIEW_HTML)
        self.assertIn('audio.addEventListener("play"', CANVAS_PREVIEW_HTML)
        self.assertIn("if (!ready || buffering)", CANVAS_PREVIEW_HTML)
        self.assertIn("if (!audio.paused) audio.pause();", CANVAS_PREVIEW_HTML)

    def test_canvas_preview_closes_stream_and_disables_controls_on_custom_error(self) -> None:
        self.assertIn('stream.addEventListener("error"', CANVAS_PREVIEW_HTML)
        self.assertIn("stream.close();", CANVAS_PREVIEW_HTML)
        self.assertIn("play.disabled = true;", CANVAS_PREVIEW_HTML)
        self.assertIn("audio.controls = false;", CANVAS_PREVIEW_HTML)

    def test_preview_server_serves_audio_and_sse(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = Path(tmpdir) / "audio.mp3"
            audio_path.write_bytes(b"fake-audio")
            server = PreviewServer(
                audio_path=audio_path,
                event_factory=lambda: iter(
                    [
                        ("metadata", {"audio_duration_ms": 1000, "difficulty": 4.5, "checkpoint_name": "fake.pt"}),
                        ("done", {"generated_through_ms": 1000, "committed_through_ms": 1000}),
                    ],
                ),
            )
            client = server.create_app().test_client()

            audio_response = client.get("/audio", buffered=True)
            sse_response = client.get("/events", buffered=True)
            audio_response.close()
            sse_response.close()

        self.assertEqual(audio_response.status_code, 200)
        self.assertEqual(sse_response.status_code, 200)
        self.assertIn(b"event: metadata", sse_response.data)
        self.assertNotIn(b"token", sse_response.data)

    def test_preview_server_serves_relative_audio_path_from_working_directory(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmpdir:
            audio_path = Path(tmpdir) / "audio.mp3"
            audio_path.write_bytes(b"fake-audio")
            relative_audio_path = audio_path.relative_to(Path.cwd())
            server = PreviewServer(
                audio_path=relative_audio_path,
                event_factory=lambda: iter([]),
            )
            client = server.create_app().test_client()

            audio_response = client.get("/audio", buffered=True)
            audio_response.close()

        self.assertEqual(audio_response.status_code, 200)
        self.assertEqual(audio_response.data, b"fake-audio")


class Stage1StreamProbeTests(unittest.TestCase):
    def test_find_latest_checkpoint_uses_newest_checkpoint_under_runs_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            older = root / "old" / "checkpoint.pt"
            newer = root / "new" / "nested" / "checkpoint.pt"
            older.parent.mkdir(parents=True)
            newer.parent.mkdir(parents=True)
            older.write_bytes(b"old")
            newer.write_bytes(b"new")
            os.utime(older, (1000, 1000))
            os.utime(newer, (2000, 2000))

            self.assertEqual(find_latest_checkpoint(root), newer)

    def test_resolve_probe_inputs_defaults_to_index_row_and_matching_audio(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_root = root / "mania-dataset"
            map_dir = dataset_root / "0" / "123"
            map_dir.mkdir(parents=True)
            audio_path = map_dir / "audio.mp3"
            beatmap_path = map_dir / "Artist - Title (Creator) [Hard].osu"
            audio_path.write_bytes(b"audio")
            beatmap_path.write_text(
                "\n".join(
                    [
                        "osu file format v14",
                        "",
                        "[General]",
                        "AudioFilename: audio.mp3",
                        "Mode: 3",
                        "",
                        "[Difficulty]",
                        "CircleSize: 4",
                        "",
                        "[TimingPoints]",
                        "0,500,4,2,0,100,1,0",
                    ],
                ),
                encoding="utf-8",
            )
            index_path = root / "index.parquet"
            pd.DataFrame(
                [
                    {
                        "shard": "0",
                        "beatmap_path": "123/Artist - Title (Creator) [Hard].osu",
                        "audio_path": "123/audio.mp3",
                        "difficulty": 4.4,
                        "key_count": 4,
                    },
                ],
            ).to_parquet(index_path)
            checkpoint_path = root / "runs" / "checkpoint.pt"
            checkpoint_path.parent.mkdir()
            checkpoint_path.write_bytes(b"checkpoint")

            inputs = resolve_probe_inputs(
                checkpoint_path=checkpoint_path,
                audio_path=None,
                beatmap_path=None,
                difficulty=None,
                dataset_root=dataset_root,
                index_path=index_path,
            )

        self.assertEqual(inputs.checkpoint_path, checkpoint_path)
        self.assertEqual(inputs.audio_path, audio_path)
        self.assertEqual(inputs.beatmap_path, beatmap_path)
        self.assertEqual(inputs.difficulty, 4.4)

    def test_format_probe_event_outputs_jsonl_with_event_name_and_data(self) -> None:
        line = format_probe_event(PreviewEvent("batch", {"start_ms": 0, "notes": []}), output_format="jsonl")

        self.assertEqual(json.loads(line), {"event": "batch", "data": {"start_ms": 0, "notes": []}})


if __name__ == "__main__":
    unittest.main()
