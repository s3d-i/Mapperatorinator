import asyncio
import json
import math
import tempfile
import unittest
import wave
from collections.abc import AsyncIterator
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import torch

from train.stage_2.inference.mapper_v2_ws_endpoint import (
    DecoderWindow,
    HitObjectToken,
    InferenceEndpoint,
    MapperV2InferenceBackend,
    MapperV2WsConfig,
    ProtocolError,
    ReferenceClock,
    _apply_time_shift_length_penalty,
    _mapper_v2_logits_fn,
    _time_shift_length_penalty_scalar,
    _time_shift_length_penalty_tensors,
    audio_end_reset_local_machine_ms,
    audio_path_from_message,
    choose_decoder_window,
    clamp_decoder_window_to_audio,
    difficulty_from_message,
    infer_message_type,
    local_machine_ms_reached,
    local_computer_time_ms_since_midnight,
    parse_json_message,
    reference_clock_from_message,
    ws_status_log_payload,
)
from train.stage_2.data.control_windows import normalize_difficulty
from train.stage_2.model_mapper_v1.generation import MapperGenerationStep
from train.stage_2.model_mapper_v1.replay import empty_ln_carry_state
from train.stage_2.model_mapper_v1.vocab import MapperV1Vocab


MANIFEST_PATH = Path("train/stage_2/inference/mapper_v2_hitobject_token_manifest.json")


class FakePeer:
    def __init__(self) -> None:
        self.messages: list[dict] = []

    async def send_json(self, payload: dict) -> None:
        self.messages.append(dict(payload))


class FakeIncrementalMapperModel:
    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = int(vocab_size)
        self.calls: list[tuple[int, int]] = []
        self.control_attention_kv_cache_args: list[object] = []

    def create_empty_decode_state(self, *, batch_size: int, device: torch.device):
        del batch_size, device
        return SimpleNamespace(steps=0)

    def incremental_decode_next_token(self, *, decode_state, decoder_input_token, position: int, **kwargs):
        token_id = int(decoder_input_token.reshape(-1)[0].item())
        self.calls.append((int(position), token_id))
        self.control_attention_kv_cache_args.append(kwargs.get("control_attention_kv_cache"))
        logits = torch.zeros((1, self.vocab_size), dtype=torch.float32)
        logits[0, token_id] = 1.0
        return SimpleNamespace(
            decode_state=SimpleNamespace(steps=int(getattr(decode_state, "steps", 0)) + 1),
            logits_final=logits,
        )


class MapperV2WsProtocolTests(unittest.TestCase):
    def test_parse_json_message_requires_object(self) -> None:
        message = parse_json_message(json.dumps({"type": "ready", "control": "ready"}))

        self.assertEqual(message["type"], "ready")
        with self.assertRaisesRegex(ProtocolError, "JSON object"):
            parse_json_message("[1, 2, 3]")

    def test_infer_message_type_accepts_control_fallbacks(self) -> None:
        self.assertEqual(infer_message_type({"type": "audio_path"}), "audio_path")
        self.assertEqual(infer_message_type({"type": "audio"}), "audio")
        self.assertEqual(infer_message_type({"control": "ready"}), "ready")
        self.assertEqual(infer_message_type({"control": "end_session"}), "stop")
        self.assertEqual(infer_message_type({"session_id": "s1", "audio_path": "/tmp/song.wav"}), "audio_path")
        self.assertEqual(infer_message_type({"session_id": "s1", "audio": {"path": "/tmp/song.wav"}}), "audio_path")
        self.assertEqual(
            infer_message_type(
                {
                    "session_id": "s1",
                    "reference_audio_ms": 100,
                    "send_local_machine_ms": 200,
                },
            ),
            "reference_time",
        )

    def test_audio_path_from_message_accepts_audio_alias(self) -> None:
        self.assertEqual(audio_path_from_message({"audio_path": "/tmp/a.wav"}), "/tmp/a.wav")
        self.assertEqual(audio_path_from_message({"audio": "/tmp/b.wav"}), "/tmp/b.wav")
        self.assertEqual(audio_path_from_message({"audio": {"path": "/tmp/c.wav"}}), "/tmp/c.wav")

    def test_difficulty_from_message_validates_supported_mapper_range(self) -> None:
        self.assertEqual(difficulty_from_message({}, default=4.0), 4.0)
        self.assertEqual(difficulty_from_message({"difficulty": 5.0}, default=4.0), 5.0)
        with self.assertRaisesRegex(ProtocolError, "difficulty"):
            difficulty_from_message({"difficulty": 7.0}, default=4.0)

    def test_local_time_ms_uses_local_time_of_day(self) -> None:
        value = local_computer_time_ms_since_midnight(datetime(2026, 5, 10, 1, 2, 3, 456_000))

        self.assertEqual(value, 3_723_456)

    def test_reference_clock_accepts_app_alias_names(self) -> None:
        clock = reference_clock_from_message(
            {
                "session_id": "s1",
                "reference_audio_ms": 1_000,
                "send_local_machine_ms": 50_000,
            },
        )

        self.assertEqual(clock.ref_time_ms, 1_000)
        self.assertEqual(clock.local_computer_time_send_ms, 50_000)

    def test_audio_end_reset_deadline_uses_sent_clock_and_reference_audio_ms(self) -> None:
        deadline = audio_end_reset_local_machine_ms(
            reference_clock=ReferenceClock(
                ref_time_ms=1_000,
                local_computer_time_send_ms=50_000,
                received_local_computer_time_ms=50_250,
            ),
            audio_length_ms=10_000,
            reset_after_audio_end_ms=2_000,
        )

        self.assertEqual(deadline, 61_000)
        self.assertFalse(local_machine_ms_reached(deadline, now_ms=60_999))
        self.assertTrue(local_machine_ms_reached(deadline, now_ms=61_000))

    def test_ws_status_log_payload_includes_status_transition(self) -> None:
        payload = ws_status_log_payload(
            session_id="s1",
            from_status="audio_ready",
            to_status="streaming",
            reason="reference_time",
            reference_audio_ms=1_234,
            reset_local_machine_ms=90_000,
        )

        self.assertEqual(payload["event"], "ws_status")
        self.assertEqual(payload["session_id"], "s1")
        self.assertEqual(payload["from"], "audio_ready")
        self.assertEqual(payload["to"], "streaming")
        self.assertEqual(payload["reason"], "reference_time")
        self.assertEqual(payload["reference_audio_ms"], 1_234)
        self.assertEqual(payload["reset_local_machine_ms"], 90_000)

    def test_choose_decoder_window_adds_elapsed_time_and_lead(self) -> None:
        clock = ReferenceClock(
            ref_time_ms=1_234,
            local_computer_time_send_ms=10_000,
            received_local_computer_time_ms=10_500,
        )

        window = choose_decoder_window(
            clock,
            MapperV2WsConfig(decoder_window_ms=8_000, decoder_lead_ms=2_000),
        )

        self.assertEqual(window, DecoderWindow(start_ms=0, end_ms=8_000))

    def test_clamp_decoder_window_keeps_reference_window_inside_audio(self) -> None:
        config = MapperV2WsConfig(decoder_window_ms=8_000)

        window = clamp_decoder_window_to_audio(
            DecoderWindow(start_ms=24_000, end_ms=32_000),
            audio_length_ms=18_500,
            config=config,
        )

        self.assertEqual(window, DecoderWindow(start_ms=16_000, end_ms=24_000))

    def test_time_shift_length_penalty_ramps_from_ts_50_to_ts_200(self) -> None:
        vocab = MapperV1Vocab()
        logits = torch.zeros(vocab.size, dtype=torch.float32)
        ts_40 = vocab.time_shift_token_id(40)
        ts_50 = vocab.time_shift_token_id(50)
        ts_1000 = vocab.time_shift_token_id(1000)
        ts_200 = vocab.time_shift_token_id(200)
        event_id = vocab.event_token_ids[0]
        logits[event_id] = 4.0

        penalty = _time_shift_length_penalty_tensors(
            vocab,
            alpha=0.5,
            device=torch.device("cpu"),
        )
        adjusted = _apply_time_shift_length_penalty(logits, time_shift_penalty=penalty)

        self.assertAlmostEqual(_time_shift_length_penalty_scalar(50, max_scalar=0.5), 0.1)
        self.assertAlmostEqual(_time_shift_length_penalty_scalar(200, max_scalar=0.5), 0.5)
        self.assertAlmostEqual(float(adjusted[ts_40].item()), 0.0)
        self.assertAlmostEqual(float(adjusted[event_id].item()), 4.0)
        self.assertAlmostEqual(float(adjusted[ts_50].item()), -0.1 * math.log(5.0))
        self.assertAlmostEqual(float(adjusted[ts_200].item()), -0.5 * math.log(20.0))
        self.assertAlmostEqual(float(adjusted[ts_1000].item()), -0.5 * math.log(100.0))

    def test_mapper_v2_logits_fn_incremental_decode_appends_only_new_prefix_token(self) -> None:
        vocab = MapperV1Vocab()
        model = FakeIncrementalMapperModel(vocab.size)
        ln_carry_in = empty_ln_carry_state(0)
        ln_carry_out = empty_ln_carry_state(8000)
        control_batch = {
            "density_teacher_8s": torch.zeros((1, 400, 1), dtype=torch.float32),
            "projected_control_memory_8s": torch.zeros((1, 400, 16), dtype=torch.float32),
            "control_attention_kv_cache": ((torch.zeros((1, 1, 400, 16)), torch.zeros((1, 1, 400, 16))),),
        }
        logits_fn = _mapper_v2_logits_fn(
            model=model,
            vocab=vocab,
            device=torch.device("cpu"),
            normalized_difficulty=0.0,
            audio_batch={},
            control_batch=control_batch,
            ln_carry_in=ln_carry_in,
            ln_carry_out=ln_carry_out,
            is_full_chart_start=True,
            is_full_chart_end=False,
            use_incremental_decode=True,
            time_shift_length_penalty_alpha=0.0,
        )

        logits_fn(
            MapperGenerationStep(
                decoder_input_tokens=torch.tensor([vocab.bos_id], dtype=torch.long),
                generated_tokens=(),
                state=ln_carry_in,
                valid_token_mask=torch.ones(vocab.size, dtype=torch.bool),
                token_index=0,
                write_start_ms=0,
                write_end_ms=8000,
                ln_carry_in=ln_carry_in,
                ln_carry_out=ln_carry_out,
            ),
        )
        logits = logits_fn(
            MapperGenerationStep(
                decoder_input_tokens=torch.tensor(
                    [vocab.bos_id, vocab.time_shift_token_id(10)],
                    dtype=torch.long,
                ),
                generated_tokens=(vocab.time_shift_token_id(10),),
                state=empty_ln_carry_state(10),
                valid_token_mask=torch.ones(vocab.size, dtype=torch.bool),
                token_index=1,
                write_start_ms=0,
                write_end_ms=8000,
                ln_carry_in=ln_carry_in,
                ln_carry_out=ln_carry_out,
            ),
        )

        self.assertEqual(model.calls, [(0, vocab.bos_id), (1, vocab.time_shift_token_id(10))])
        self.assertTrue(
            all(cache is control_batch["control_attention_kv_cache"] for cache in model.control_attention_kv_cache_args)
        )
        self.assertEqual(int(torch.argmax(logits).item()), vocab.time_shift_token_id(10))


class MapperV2WsEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def test_audio_path_requires_ready(self) -> None:
        endpoint = InferenceEndpoint(
            config=MapperV2WsConfig(token_send_interval_s=0.0),
            backend=FakeInferenceBackend(),
        )

        with self.assertRaisesRegex(ProtocolError, "send ready"):
            await endpoint.handle_message(
                {"type": "audio_path", "session_id": "s1", "audio_path": "/tmp/song.wav"},
                FakePeer(),
            )

    async def test_reference_time_starts_hitobject_token_stream(self) -> None:
        config = MapperV2WsConfig(token_send_interval_s=0.0)
        backend = FakeInferenceBackend(
            tokens=(
                HitObjectToken(10, "EVENT_A", 1_240, ("tap", "none", "none", "none")),
                HitObjectToken(11, "EVENT_B", 1_500, ("none", "tap", "none", "none")),
            ),
        )
        endpoint = InferenceEndpoint(config=config, backend=backend)
        peer = FakePeer()

        await endpoint.handle_message({"type": "ready", "control": "ready"}, peer)
        await endpoint.handle_message(
            {
                "type": "audio_path",
                "session_id": "s1",
                "audio_path": "/Users/ken/audio/song1.wav",
                "audio_length_ms": 180_000,
                "difficulty": 5.0,
            },
            peer,
        )
        await endpoint.handle_message(
            {
                "type": "reference_time",
                "session_id": "s1",
                "ref_time_ms": 1_234,
                "local_computer_time_send_ms": local_computer_time_ms_since_midnight(),
            },
            peer,
        )
        task = endpoint.sessions["s1"].stream_task
        assert task is not None
        await task

        self.assertEqual([message["type"] for message in peer.messages], ["hitobject_tokens"] * 2)
        self.assertTrue(all(message["session_id"] == "s1" for message in peer.messages))
        self.assertTrue(all(set(message) == {"type", "session_id", "token"} for message in peer.messages))
        self.assertEqual([message["token"] for message in peer.messages], [[10, 1_240], [11, 1_500]])
        self.assertEqual(
            backend.prepared_audio,
            [
                {
                    "session_id": "s1",
                    "audio_path": Path("/Users/ken/audio/song1.wav"),
                    "audio_length_ms": 180_000,
                    "difficulty": 5.0,
                },
            ],
        )
        self.assertEqual(len(backend.iter_calls), 1)
        self.assertEqual(backend.iter_calls[0]["audio_path"], Path("/Users/ken/audio/song1.wav"))
        self.assertEqual(backend.iter_calls[0]["audio_length_ms"], 180_000)
        self.assertIsInstance(backend.iter_calls[0]["window"], DecoderWindow)
        await endpoint.stop_session("s1")

    async def test_audio_path_requires_length_or_readable_audio_file(self) -> None:
        endpoint = InferenceEndpoint(
            config=MapperV2WsConfig(token_send_interval_s=0.0),
            backend=FakeInferenceBackend(),
        )
        peer = FakePeer()

        await endpoint.handle_message({"type": "ready", "control": "ready"}, peer)
        with self.assertRaisesRegex(ProtocolError, "audio_length_ms"):
            await endpoint.handle_message(
                {"type": "audio_path", "session_id": "s1", "audio_path": "/tmp/nonexistent-song.wav"},
                peer,
            )

    async def test_audio_path_file_duration_allows_reference_time_without_message_length(self) -> None:
        backend = FakeInferenceBackend()
        endpoint = InferenceEndpoint(config=MapperV2WsConfig(token_send_interval_s=0.0), backend=backend)
        peer = FakePeer()

        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = Path(tmpdir) / "song.wav"
            _write_silent_wav(audio_path, duration_ms=1_000)

            await endpoint.handle_message({"type": "ready", "control": "ready"}, peer)
            await endpoint.handle_message(
                {"type": "audio_path", "session_id": "s1", "audio_path": str(audio_path)},
                peer,
            )
            await endpoint.handle_message(
                {
                    "type": "reference_time",
                    "session_id": "s1",
                    "ref_time_ms": 0,
                    "local_computer_time_send_ms": local_computer_time_ms_since_midnight(),
                },
                peer,
            )
            task = endpoint.sessions["s1"].stream_task
            assert task is not None
            await task

            self.assertEqual(endpoint.sessions["s1"].audio_length_ms, 1_000)
            self.assertEqual(backend.prepared_audio[0]["audio_length_ms"], 1_000)
            self.assertEqual(backend.prepared_audio[0]["difficulty"], 4.0)
            self.assertEqual(backend.iter_calls[0]["audio_length_ms"], 1_000)
            await endpoint.stop_session("s1")

    async def test_audio_message_alias_prepares_ws_audio(self) -> None:
        backend = FakeInferenceBackend()
        endpoint = InferenceEndpoint(config=MapperV2WsConfig(token_send_interval_s=0.0), backend=backend)
        peer = FakePeer()

        await endpoint.handle_message({"control": "ready"}, peer)
        await endpoint.handle_message(
            {
                "type": "audio",
                "session_id": "s1",
                "audio": {"path": "/tmp/song.wav"},
                "audio_length_ms": 2_000,
            },
            peer,
        )

        self.assertEqual(backend.prepared_audio[0]["audio_path"], Path("/tmp/song.wav"))
        self.assertEqual(backend.prepared_audio[0]["audio_length_ms"], 2_000)

    async def test_reference_time_difficulty_does_not_override_audio_path_difficulty(self) -> None:
        backend = FakeInferenceBackend()
        endpoint = InferenceEndpoint(config=MapperV2WsConfig(token_send_interval_s=0.0), backend=backend)
        peer = FakePeer()

        await endpoint.handle_message({"control": "ready"}, peer)
        await endpoint.handle_message(
            {
                "type": "audio_path",
                "session_id": "s1",
                "audio_path": "/tmp/song.wav",
                "audio_length_ms": 2_000,
                "difficulty": 5.0,
            },
            peer,
        )
        await endpoint.handle_message(
            {
                "type": "reference_time",
                "session_id": "s1",
                "ref_time_ms": 0,
                "local_computer_time_send_ms": local_computer_time_ms_since_midnight(),
                "difficulty": 3.0,
            },
            peer,
        )
        task = endpoint.sessions["s1"].stream_task
        assert task is not None
        await task

        self.assertEqual(endpoint.sessions["s1"].difficulty, 5.0)
        self.assertEqual(backend.prepared_audio[0]["difficulty"], 5.0)
        await endpoint.stop_session("s1")

    async def test_mapper_v2_backend_prepares_session_runtime_from_ws_audio(self) -> None:
        loader_configs = []
        created = []
        fake_session = FakeSessionRuntime()

        def runtime_loader(config):
            loader_configs.append(config)
            return SimpleNamespace(device="cpu", vocab=MapperV1Vocab(), mapper_model=None)

        def session_factory(session_id, model_runtime, config):
            created.append((session_id, model_runtime, config))
            return fake_session

        config = MapperV2WsConfig(
            mapper_checkpoint_path="mapper.pt",
            control_checkpoint_path="control.pt",
            device="cpu",
            token_send_interval_s=0.0,
        )
        backend = MapperV2InferenceBackend(
            config,
            runtime_loader=runtime_loader,
            session_runtime_factory=session_factory,
        )

        await backend.startup()
        await backend.prepare_audio(
            session_id="s1",
            audio_path=Path("/tmp/song.wav"),
            audio_length_ms=1_234,
            difficulty=5.0,
        )

        self.assertTrue(backend.models_ready)
        self.assertEqual(loader_configs[0].mapper_checkpoint_path, Path.cwd() / "mapper.pt")
        self.assertEqual(loader_configs[0].control_checkpoint_path, Path.cwd() / "control.pt")
        self.assertEqual(created[0][0], "s1")
        self.assertAlmostEqual(created[0][2].default_normalized_difficulty, normalize_difficulty(5.0))
        self.assertEqual(fake_session.prepare_audio_calls, [(Path("/tmp/song.wav"), 1_234, 0)])

    def test_hitobject_token_manifest_matches_full_mapper_event_vocab(self) -> None:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        mapping = manifest["event_token_id_to_lane_action"]
        vocab = MapperV1Vocab()

        expected = {
            str(token_id): [action.value for action in vocab.decode_event(token_id)]
            for token_id in vocab.event_token_ids
        }

        self.assertEqual(mapping, expected)
        self.assertEqual(manifest["event_token_count"], len(vocab.event_token_ids))
        self.assertEqual(manifest["event_token_id_range"], [min(vocab.event_token_ids), max(vocab.event_token_ids)])

    async def test_stop_cancels_stream_and_resets_session(self) -> None:
        config = MapperV2WsConfig(token_send_interval_s=1.0)
        backend = FakeInferenceBackend(block_after_first=True)
        endpoint = InferenceEndpoint(config=config, backend=backend)
        peer = FakePeer()

        await endpoint.handle_message({"type": "ready", "control": "ready"}, peer)
        await endpoint.handle_message(
            {
                "type": "audio_path",
                "session_id": "s1",
                "audio_path": str(Path("/tmp/song.wav")),
                "audio_length_ms": 180_000,
            },
            peer,
        )
        await endpoint.handle_message(
            {
                "type": "reference_time",
                "session_id": "s1",
                "ref_time_ms": 0,
                "local_computer_time_send_ms": local_computer_time_ms_since_midnight(),
            },
            peer,
        )
        await asyncio.sleep(0)
        await endpoint.handle_message({"type": "stop", "session_id": "s1", "control": "end_session"}, peer)

        self.assertNotIn("s1", endpoint.sessions)
        self.assertEqual(backend.reset_sessions, ["s1"])

    async def test_wall_clock_resets_session_after_audio_end_grace(self) -> None:
        config = MapperV2WsConfig(
            token_send_interval_s=0.0,
            reset_after_audio_end_ms=20,
            wall_clock_check_interval_s=0.01,
        )
        endpoint = InferenceEndpoint(config=config, backend=FakeInferenceBackend())
        peer = FakePeer()
        now_ms = local_computer_time_ms_since_midnight()

        await endpoint.handle_message({"control": "ready"}, peer)
        await endpoint.handle_message(
            {
                "session_id": "s1",
                "audio_path": "/tmp/song.wav",
                "audio_length_ms": 1,
            },
            peer,
        )
        await endpoint.handle_message(
            {
                "session_id": "s1",
                "reference_audio_ms": 1,
                "send_local_machine_ms": now_ms,
            },
            peer,
        )

        await asyncio.sleep(0.2)

        self.assertNotIn("s1", endpoint.sessions)


class FakeInferenceBackend:
    def __init__(
        self,
        *,
        tokens: tuple[HitObjectToken, ...] | None = None,
        block_after_first: bool = False,
    ) -> None:
        self.models_ready = False
        self.tokens = (
            HitObjectToken(10, "EVENT_A", 0, ("tap", "none", "none", "none")),
        ) if tokens is None else tokens
        self.block_after_first = bool(block_after_first)
        self.prepared_audio: list[dict[str, object]] = []
        self.iter_calls: list[dict[str, object]] = []
        self.reset_sessions: list[str] = []

    async def startup(self) -> None:
        self.models_ready = True

    async def prepare_audio(
        self,
        *,
        session_id: str,
        audio_path: Path,
        audio_length_ms: int,
        difficulty: float | None,
    ) -> None:
        self.prepared_audio.append(
            {
                "session_id": session_id,
                "audio_path": audio_path,
                "audio_length_ms": audio_length_ms,
                "difficulty": difficulty,
            },
        )
        await asyncio.sleep(0)

    async def iter_hitobject_tokens(
        self,
        *,
        session_id: str,
        audio_path: Path,
        audio_length_ms: int,
        window: DecoderWindow,
    ) -> AsyncIterator[HitObjectToken]:
        self.iter_calls.append(
            {
                "session_id": session_id,
                "audio_path": audio_path,
                "audio_length_ms": audio_length_ms,
                "window": window,
            },
        )
        for index, token in enumerate(self.tokens):
            yield token
            if index == 0 and self.block_after_first:
                await asyncio.sleep(10)

    async def reset_session(self, session_id: str) -> None:
        self.reset_sessions.append(session_id)


class FakeSessionRuntime:
    def __init__(self) -> None:
        self.prepare_audio_calls: list[tuple[Path, int, int]] = []

    def prepare_audio(
        self,
        audio_path: str | Path,
        *,
        audio_length_ms: int,
        start_ms: int = 0,
    ) -> SimpleNamespace:
        self.prepare_audio_calls.append((Path(audio_path), audio_length_ms, start_ms))
        return SimpleNamespace(audio_length_ms=audio_length_ms)

    def reset_audio_cache(self) -> None:
        pass


def _write_silent_wav(path: Path, *, duration_ms: int, sample_rate: int = 8_000) -> None:
    frame_count = int(round(sample_rate * duration_ms / 1000.0))
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        audio.writeframes(b"\x00\x00" * frame_count)


if __name__ == "__main__":
    unittest.main()
