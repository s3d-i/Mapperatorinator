import asyncio
import json
import unittest
from datetime import datetime
from pathlib import Path

from train.stage_2.inference.mapper_v2_ws_endpoint import (
    DatasetHitObjectBackend,
    DecoderWindow,
    InferenceEndpoint,
    MapperV2WsConfig,
    ProtocolError,
    ReferenceClock,
    audio_end_reset_local_machine_ms,
    choose_decoder_window,
    infer_message_type,
    local_machine_ms_reached,
    local_computer_time_ms_since_midnight,
    parse_json_message,
    reference_clock_from_message,
    ws_status_log_payload,
)
from train.stage_2.model_mapper_v1.vocab import MapperV1Vocab


MANIFEST_PATH = Path("train/stage_2/inference/mapper_v2_hitobject_token_manifest.json")


class FakePeer:
    def __init__(self) -> None:
        self.messages: list[dict] = []

    async def send_json(self, payload: dict) -> None:
        self.messages.append(dict(payload))


class MapperV2WsProtocolTests(unittest.TestCase):
    def test_parse_json_message_requires_object(self) -> None:
        message = parse_json_message(json.dumps({"type": "ready", "control": "ready"}))

        self.assertEqual(message["type"], "ready")
        with self.assertRaisesRegex(ProtocolError, "JSON object"):
            parse_json_message("[1, 2, 3]")

    def test_infer_message_type_accepts_control_fallbacks(self) -> None:
        self.assertEqual(infer_message_type({"type": "audio_path"}), "audio_path")
        self.assertEqual(infer_message_type({"control": "ready"}), "ready")
        self.assertEqual(infer_message_type({"control": "end_session"}), "stop")
        self.assertEqual(infer_message_type({"session_id": "s1", "audio_path": "/tmp/song.wav"}), "audio_path")
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


class MapperV2WsEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def test_audio_path_requires_ready(self) -> None:
        endpoint = InferenceEndpoint(config=MapperV2WsConfig(token_send_interval_s=0.0))

        with self.assertRaisesRegex(ProtocolError, "send ready"):
            await endpoint.handle_message(
                {"type": "audio_path", "session_id": "s1", "audio_path": "/tmp/song.wav"},
                FakePeer(),
            )

    async def test_reference_time_starts_hitobject_token_stream(self) -> None:
        config = MapperV2WsConfig(token_send_interval_s=0.0)
        endpoint = InferenceEndpoint(config=config, backend=DatasetHitObjectBackend(config))
        peer = FakePeer()

        await endpoint.handle_message({"type": "ready", "control": "ready"}, peer)
        await endpoint.handle_message(
            {
                "type": "audio_path",
                "session_id": "s1",
                "audio_path": "/Users/ken/audio/song1.wav",
                "audio_length_ms": 180_000,
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

        self.assertEqual([message["type"] for message in peer.messages], ["hitobject_tokens"] * 3_188)
        self.assertTrue(all(message["session_id"] == "s1" for message in peer.messages))
        self.assertTrue(all(set(message) == {"type", "session_id", "token"} for message in peer.messages))
        self.assertTrue(all(isinstance(message["token"][0], int) for message in peer.messages))
        self.assertTrue(all(isinstance(message["token"][1], int) for message in peer.messages))
        self.assertTrue(any(message["token"][1] < 60_000 for message in peer.messages))
        self.assertTrue(any(60_000 <= message["token"][1] < 120_000 for message in peer.messages))
        self.assertTrue(any(message["token"][1] >= 120_000 for message in peer.messages))
        await endpoint.stop_session("s1")

    async def test_reference_time_requires_audio_length_or_readable_audio_file(self) -> None:
        endpoint = InferenceEndpoint(config=MapperV2WsConfig(token_send_interval_s=0.0))
        peer = FakePeer()

        await endpoint.handle_message({"type": "ready", "control": "ready"}, peer)
        await endpoint.handle_message(
            {"type": "audio_path", "session_id": "s1", "audio_path": "/tmp/nonexistent-song.wav"},
            peer,
        )

        with self.assertRaisesRegex(ProtocolError, "audio_length_ms"):
            await endpoint.handle_message(
                {
                    "type": "reference_time",
                    "session_id": "s1",
                    "ref_time_ms": 0,
                    "local_computer_time_send_ms": local_computer_time_ms_since_midnight(),
                },
                peer,
            )

    async def test_real_hitobject_stream_splits_selected_dataset_map_into_three_batches(self) -> None:
        config = MapperV2WsConfig(token_send_interval_s=0.0)
        backend = DatasetHitObjectBackend(config)

        stream = backend.real_hitobject_batches()

        self.assertEqual(len(stream), 3)
        self.assertTrue(all(stream))
        self.assertTrue(all(token.ms_in_ref_audio < 60_000 for token in stream[0]))
        self.assertTrue(all(60_000 <= token.ms_in_ref_audio < 120_000 for token in stream[1]))
        self.assertTrue(all(token.ms_in_ref_audio >= 120_000 for token in stream[2]))
        self.assertEqual(sum(len(batch) for batch in stream), 3_188)

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
        endpoint = InferenceEndpoint(config=config, backend=DatasetHitObjectBackend(config))
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

    async def test_wall_clock_resets_session_after_audio_end_grace(self) -> None:
        config = MapperV2WsConfig(
            token_send_interval_s=0.0,
            reset_after_audio_end_ms=20,
            wall_clock_check_interval_s=0.01,
        )
        endpoint = InferenceEndpoint(config=config, backend=DatasetHitObjectBackend(config))
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


if __name__ == "__main__":
    unittest.main()
