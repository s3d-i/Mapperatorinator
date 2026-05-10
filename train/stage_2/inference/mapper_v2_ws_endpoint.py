from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import math
import struct
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from train.stage1_oracle.osu.hitobjects import parse_mania_hit_objects
from train.stage_2.model_mapper_v1.tokenizer import MAPPER_WRITE_MS, hitobjects_to_mapper_timepoints
from train.stage_2.model_mapper_v1.vocab import MapperV1Vocab


PULSEFIELD_WS_URL = "ws://localhost:8765"
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 8765
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_HITOBJECT_BEATMAP_PATH = Path(
    "mania-dataset/0/1047817/Camellia - Arche (-mint-) [drago vs. mint's apeiron].osu",
)
HITOBJECT_CHUNK_BOUNDARIES_MS = (60_000, 120_000)


class ProtocolError(ValueError):
    pass


class JsonPeer(Protocol):
    async def send_json(self, payload: Mapping[str, Any]) -> None:
        ...


@dataclass(frozen=True)
class MapperV2WsConfig:
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    decoder_window_ms: int = MAPPER_WRITE_MS
    decoder_lead_ms: int = 2_000
    token_send_interval_s: float = 5.0
    hitobject_beatmap_path: str | Path = DEFAULT_HITOBJECT_BEATMAP_PATH
    reset_after_audio_end_ms: int = 2_000
    wall_clock_check_interval_s: float = 0.05


@dataclass(frozen=True)
class ReferenceClock:
    ref_time_ms: int
    local_computer_time_send_ms: int
    received_local_computer_time_ms: int


@dataclass(frozen=True)
class DecoderWindow:
    start_ms: int
    end_ms: int


@dataclass(frozen=True)
class HitObjectToken:
    token_id: int
    token_name: str
    ms_in_ref_audio: int
    actions: tuple[str, ...]

    def message_token(self) -> list[int]:
        return [self.token_id, self.ms_in_ref_audio]


@dataclass
class SessionState:
    session_id: str
    audio_path: Path | None = None
    audio_length_ms: int | None = None
    audio_prepared: bool = False
    reference_clock: ReferenceClock | None = None
    stream_task: asyncio.Task[None] | None = None
    wall_clock_reset_task: asyncio.Task[None] | None = None
    decoder_window: DecoderWindow | None = None


def session_status(session: SessionState | None) -> str:
    if session is None:
        return "no_session"
    if session.reference_clock is not None and session.decoder_window is not None:
        return "streaming"
    if session.audio_prepared:
        return "audio_ready"
    return "audio_preparing"


class DatasetHitObjectBackend:
    def __init__(self, config: MapperV2WsConfig) -> None:
        self.config = config
        self.vocab = MapperV1Vocab()
        self.models_ready = False
        self._hitobject_tokens: tuple[HitObjectToken, ...] | None = None

    async def startup(self) -> None:
        self._load_hitobject_tokens()
        self.models_ready = True

    async def prepare_audio(self, *, session_id: str, audio_path: Path) -> None:
        del session_id, audio_path
        self._load_hitobject_tokens()
        await asyncio.sleep(0)

    async def iter_hitobject_tokens(
        self,
        *,
        session_id: str,
        audio_path: Path,
    ) -> AsyncIterator[HitObjectToken]:
        del session_id, audio_path
        batches = self.real_hitobject_batches()
        for index, batch in enumerate(batches):
            for token in batch:
                yield token
            if index + 1 < len(batches):
                await asyncio.sleep(max(0.0, float(self.config.token_send_interval_s)))

    def real_hitobject_batches(self) -> list[tuple[HitObjectToken, ...]]:
        tokens = self._load_hitobject_tokens()
        first_boundary_ms, second_boundary_ms = HITOBJECT_CHUNK_BOUNDARIES_MS
        return [
            tuple(token for token in tokens if token.ms_in_ref_audio < first_boundary_ms),
            tuple(token for token in tokens if first_boundary_ms <= token.ms_in_ref_audio < second_boundary_ms),
            tuple(token for token in tokens if token.ms_in_ref_audio >= second_boundary_ms),
        ]

    def _load_hitobject_tokens(self) -> tuple[HitObjectToken, ...]:
        if self._hitobject_tokens is not None:
            return self._hitobject_tokens

        beatmap_path = _resolve_hitobject_beatmap_path(self.config.hitobject_beatmap_path)
        hitobjects = parse_mania_hit_objects(beatmap_path, expected_key_count=4)
        timepoints = hitobjects_to_mapper_timepoints(hitobjects)
        tokens: list[HitObjectToken] = []
        for timepoint in timepoints:
            token_id = int(self.vocab.encode_event(timepoint.lane_actions))
            tokens.append(
                HitObjectToken(
                    token_id=token_id,
                    token_name=self.vocab.token_name(token_id),
                    ms_in_ref_audio=int(timepoint.time_ms),
                    actions=tuple(action.value for action in timepoint.lane_actions),
                ),
            )
        self._hitobject_tokens = tuple(tokens)
        return self._hitobject_tokens


@dataclass
class InferenceEndpoint:
    config: MapperV2WsConfig = field(default_factory=MapperV2WsConfig)
    backend: DatasetHitObjectBackend | None = None

    def __post_init__(self) -> None:
        if self.backend is None:
            self.backend = DatasetHitObjectBackend(self.config)
        self.sessions: dict[str, SessionState] = {}
        self._startup_lock = asyncio.Lock()

    async def handle_message(self, raw_message: str | bytes | Mapping[str, Any], peer: JsonPeer) -> None:
        message = parse_json_message(raw_message)
        message_type = infer_message_type(message)

        if message_type == "ready":
            await self.startup()
            return
        if message_type == "audio_path":
            await self._handle_audio_path(message)
            return
        if message_type == "reference_time":
            await self._handle_reference_time(message, peer)
            return
        if message_type == "stop":
            await self.stop_session(require_session_id(message))
            return
        raise ProtocolError(f"unsupported message type: {message_type!r}")

    async def startup(self) -> None:
        assert self.backend is not None
        async with self._startup_lock:
            if self.backend.models_ready:
                return
            await self.backend.startup()
            log_ws_status(
                session_id=None,
                from_status="cold",
                to_status="ready",
                reason="ready",
            )

    async def stop_session(self, session_id: str, *, reason: str = "client_stop") -> None:
        session = self.sessions.pop(session_id, None)
        if session is None:
            return
        from_status = session_status(session)
        await _cancel_task(session.stream_task)
        await _cancel_task(session.wall_clock_reset_task)
        log_ws_status(
            session_id=session_id,
            from_status=from_status,
            to_status="stopped/reset",
            reason=reason,
        )

    async def _handle_audio_path(self, message: Mapping[str, Any]) -> None:
        assert self.backend is not None
        if not self.backend.models_ready:
            raise ProtocolError("send ready before audio_path")
        session_id = require_session_id(message)
        raw_audio_path = message.get("audio_path")
        if not isinstance(raw_audio_path, str) or not raw_audio_path.strip():
            raise ProtocolError("audio_path must be a non-empty string")
        audio_path = Path(raw_audio_path).expanduser()
        audio_length_ms = audio_length_ms_from_message(message)
        if audio_length_ms is None:
            audio_length_ms = audio_length_ms_from_file(audio_path)

        existing = self.sessions.get(session_id)
        if existing is not None:
            await self.stop_session(session_id, reason="replace_audio_path")

        session = SessionState(
            session_id=session_id,
            audio_path=audio_path,
            audio_length_ms=audio_length_ms,
        )
        self.sessions[session_id] = session
        log_ws_status(
            session_id=session_id,
            from_status="no_session",
            to_status="audio_preparing",
            reason="audio_path",
            audio_path=str(audio_path),
            audio_length_ms=audio_length_ms,
        )
        await self.backend.prepare_audio(session_id=session_id, audio_path=audio_path)
        session.audio_prepared = True
        log_ws_status(
            session_id=session_id,
            from_status="audio_preparing",
            to_status="audio_ready",
            reason="audio_prepared",
            audio_path=str(audio_path),
            audio_length_ms=audio_length_ms,
        )

    async def _handle_reference_time(self, message: Mapping[str, Any], peer: JsonPeer) -> None:
        session_id = require_session_id(message)
        session = self.sessions.get(session_id)
        if session is None or session.audio_path is None or not session.audio_prepared:
            raise ProtocolError("send audio_path before reference_time")

        clock = reference_clock_from_message(message)
        message_audio_length_ms = audio_length_ms_from_message(message)
        if message_audio_length_ms is not None:
            session.audio_length_ms = message_audio_length_ms
        if session.audio_length_ms is None:
            raise ProtocolError("audio_length_ms is required or must be readable from audio_path")
        from_status = session_status(session)
        window = choose_decoder_window(clock, self.config)
        session.reference_clock = clock
        session.decoder_window = window
        await _cancel_task(session.stream_task)
        await _cancel_task(session.wall_clock_reset_task)
        session.stream_task = asyncio.create_task(self._stream_tokens(session, window, peer))
        session.wall_clock_reset_task = asyncio.create_task(self._reset_after_audio_end(session))
        reset_local_machine_ms = audio_end_reset_local_machine_ms(
            reference_clock=clock,
            audio_length_ms=session.audio_length_ms,
            reset_after_audio_end_ms=self.config.reset_after_audio_end_ms,
        )
        log_ws_status(
            session_id=session_id,
            from_status=from_status,
            to_status="streaming",
            reason="reference_time",
            reference_audio_ms=clock.ref_time_ms,
            send_local_machine_ms=clock.local_computer_time_send_ms,
            received_local_machine_ms=clock.received_local_computer_time_ms,
            audio_length_ms=session.audio_length_ms,
            reset_local_machine_ms=reset_local_machine_ms,
        )

    async def _reset_after_audio_end(self, session: SessionState) -> None:
        if session.reference_clock is None or session.audio_length_ms is None:
            return
        reset_local_machine_ms = audio_end_reset_local_machine_ms(
            reference_clock=session.reference_clock,
            audio_length_ms=session.audio_length_ms,
            reset_after_audio_end_ms=self.config.reset_after_audio_end_ms,
        )
        check_interval_s = max(0.01, float(self.config.wall_clock_check_interval_s))
        while self.sessions.get(session.session_id) is session:
            if local_machine_ms_reached(reset_local_machine_ms):
                await self.stop_session(session.session_id, reason="wall_clock_audio_end")
                return
            await asyncio.sleep(check_interval_s)

    async def _stream_tokens(self, session: SessionState, window: DecoderWindow, peer: JsonPeer) -> None:
        assert self.backend is not None
        assert session.audio_path is not None
        del window
        async for hitobject in self.backend.iter_hitobject_tokens(
            session_id=session.session_id,
            audio_path=session.audio_path,
        ):
            if self.sessions.get(session.session_id) is not session:
                return
            await peer.send_json(
                {
                    "type": "hitobject_tokens",
                    "session_id": session.session_id,
                    "token": hitobject.message_token(),
                },
            )


def _resolve_hitobject_beatmap_path(path: str | Path) -> Path:
    beatmap_path = Path(path).expanduser()
    if beatmap_path.is_absolute():
        return beatmap_path
    cwd_path = Path.cwd() / beatmap_path
    if cwd_path.is_file():
        return cwd_path
    return REPOSITORY_ROOT / beatmap_path


def parse_json_message(raw_message: str | bytes | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(raw_message, Mapping):
        return raw_message
    if isinstance(raw_message, bytes):
        raw_message = raw_message.decode("utf-8")
    try:
        message = json.loads(raw_message)
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"invalid json message: {exc.msg}") from exc
    if not isinstance(message, Mapping):
        raise ProtocolError("websocket message must be a JSON object")
    return message


def infer_message_type(message: Mapping[str, Any]) -> str:
    raw_type = message.get("type")
    if isinstance(raw_type, str) and raw_type:
        return raw_type
    control = message.get("control")
    if control == "ready":
        return "ready"
    if control == "end_session":
        return "stop"
    if "audio_path" in message:
        return "audio_path"
    has_reference_audio_ms = "ref_time_ms" in message or "reference_audio_ms" in message
    has_send_local_machine_ms = "local_computer_time_send_ms" in message or "send_local_machine_ms" in message
    if has_reference_audio_ms and has_send_local_machine_ms:
        return "reference_time"
    raise ProtocolError("message must include type or a recognized control field")


def require_session_id(message: Mapping[str, Any]) -> str:
    session_id = message.get("session_id")
    if not isinstance(session_id, str) or not session_id.strip():
        raise ProtocolError("session_id must be a non-empty string")
    return session_id


def reference_clock_from_message(message: Mapping[str, Any]) -> ReferenceClock:
    return ReferenceClock(
        ref_time_ms=_required_int_alias(message, "ref_time_ms", "reference_audio_ms"),
        local_computer_time_send_ms=_required_int_alias(
            message,
            "local_computer_time_send_ms",
            "send_local_machine_ms",
        ),
        received_local_computer_time_ms=local_machine_ms_since_midnight(),
    )


def choose_decoder_window(clock: ReferenceClock, config: MapperV2WsConfig) -> DecoderWindow:
    elapsed_ms = max(0, clock.received_local_computer_time_ms - clock.local_computer_time_send_ms)
    estimated_ref_ms = max(0, clock.ref_time_ms + elapsed_ms)
    target_ms = estimated_ref_ms + max(0, int(config.decoder_lead_ms))
    window_ms = int(config.decoder_window_ms)
    if window_ms <= 0:
        raise ValueError("decoder_window_ms must be positive")
    start_ms = (target_ms // window_ms) * window_ms
    return DecoderWindow(start_ms=start_ms, end_ms=start_ms + window_ms)


def local_computer_time_ms_since_midnight(now: datetime | None = None) -> int:
    return local_machine_ms_since_midnight(now)


def local_machine_ms_since_midnight(now: datetime | None = None) -> int:
    now = datetime.now().astimezone() if now is None else now
    # Protocol simplification: this is local time-of-day milliseconds and does
    # not account for crossing midnight between App send and Server receive.
    return (
        now.hour * 60 * 60 * 1000
        + now.minute * 60 * 1000
        + now.second * 1000
        + now.microsecond // 1000
    )


def audio_end_reset_local_machine_ms(
    *,
    reference_clock: ReferenceClock,
    audio_length_ms: int,
    reset_after_audio_end_ms: int,
) -> int:
    remaining_audio_ms = max(0, int(audio_length_ms) - int(reference_clock.ref_time_ms))
    return int(reference_clock.local_computer_time_send_ms) + remaining_audio_ms + max(
        0,
        int(reset_after_audio_end_ms),
    )


def local_machine_ms_reached(deadline_ms: int, now_ms: int | None = None) -> bool:
    now_ms = local_machine_ms_since_midnight() if now_ms is None else int(now_ms)
    return now_ms >= int(deadline_ms)


def ws_status_log_payload(
    *,
    session_id: str | None,
    from_status: str,
    to_status: str,
    reason: str,
    **fields: Any,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "event": "ws_status",
        "local_machine_ms": local_machine_ms_since_midnight(),
        "session_id": session_id,
        "from": from_status,
        "to": to_status,
        "reason": reason,
    }
    for key, value in fields.items():
        if value is None:
            continue
        payload[key] = value
    return payload


def log_ws_status(
    *,
    session_id: str | None,
    from_status: str,
    to_status: str,
    reason: str,
    **fields: Any,
) -> None:
    payload = ws_status_log_payload(
        session_id=session_id,
        from_status=from_status,
        to_status=to_status,
        reason=reason,
        **fields,
    )
    print(f"ws_status {json.dumps(payload, separators=(',', ':'), ensure_ascii=False)}", flush=True)


def audio_length_ms_from_message(message: Mapping[str, Any]) -> int | None:
    value = _optional_int_alias(message, "audio_length_ms", "audio_length")
    if value is not None:
        if value <= 0:
            raise ProtocolError("audio_length_ms must be positive")
        return value
    seconds = message.get("audio_length_s")
    if seconds is None:
        return None
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
        raise ProtocolError("audio_length_s must be numeric")
    seconds = float(seconds)
    if not math.isfinite(seconds) or seconds <= 0:
        raise ProtocolError("audio_length_s must be positive and finite")
    return int(round(seconds * 1000.0))


def audio_length_ms_from_file(audio_path: Path) -> int | None:
    if not audio_path.exists():
        return None
    try:
        from mutagen import File as MutagenFile
    except ImportError:
        return None
    try:
        audio = MutagenFile(audio_path)
    except Exception:
        return None
    if audio is None or getattr(audio, "info", None) is None:
        return None
    length = getattr(audio.info, "length", None)
    if not isinstance(length, (int, float)) or not math.isfinite(float(length)) or float(length) <= 0:
        return None
    return int(round(float(length) * 1000.0))


def _required_int(message: Mapping[str, Any], key: str) -> int:
    value = message.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProtocolError(f"{key} must be an integer")
    return int(value)


def _required_int_alias(message: Mapping[str, Any], *keys: str) -> int:
    value = _optional_int_alias(message, *keys)
    if value is None:
        raise ProtocolError(f"{'/'.join(keys)} must be an integer")
    return value


def _optional_int_alias(message: Mapping[str, Any], *keys: str) -> int | None:
    for key in keys:
        if key not in message:
            continue
        value = message[key]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ProtocolError(f"{key} must be an integer")
        return int(value)
    return None


async def _cancel_task(task: asyncio.Task[None] | None) -> None:
    if task is None or task.done() or task is asyncio.current_task():
        return
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


class _WebSocketPeer:
    def __init__(self, writer: asyncio.StreamWriter) -> None:
        self._writer = writer
        self._send_lock = asyncio.Lock()

    async def send_json(self, payload: Mapping[str, Any]) -> None:
        data = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        async with self._send_lock:
            self._writer.write(_encode_server_text_frame(data))
            await self._writer.drain()


async def serve_forever(endpoint: InferenceEndpoint | None = None) -> None:
    config = MapperV2WsConfig()
    endpoint = InferenceEndpoint(config=config) if endpoint is None else endpoint
    server = await asyncio.start_server(
        lambda reader, writer: _handle_websocket_client(endpoint, reader, writer),
        host=endpoint.config.host,
        port=endpoint.config.port,
    )
    sockets = ", ".join(str(sock.getsockname()) for sock in server.sockets or ())
    print(f"mapper_v2_ws_endpoint listening on {PULSEFIELD_WS_URL} ({sockets})", flush=True)
    async with server:
        await server.serve_forever()


async def _handle_websocket_client(
    endpoint: InferenceEndpoint,
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
) -> None:
    peer = _WebSocketPeer(writer)
    try:
        await _accept_websocket_handshake(reader, writer)
        while True:
            message = await _read_client_text_frame(reader, writer)
            if message is None:
                break
            try:
                await endpoint.handle_message(message, peer)
            except ProtocolError as exc:
                await peer.send_json({"type": "error", "error": str(exc)})
    finally:
        writer.close()
        await writer.wait_closed()


async def _accept_websocket_handshake(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
) -> None:
    request = await reader.readuntil(b"\r\n\r\n")
    header_text = request.decode("latin1")
    lines = header_text.split("\r\n")
    if not lines or not lines[0].startswith("GET "):
        raise ProtocolError("websocket handshake must use GET")
    headers: dict[str, str] = {}
    for line in lines[1:]:
        if not line or ":" not in line:
            continue
        name, value = line.split(":", 1)
        headers[name.strip().lower()] = value.strip()
    key = headers.get("sec-websocket-key")
    if not key:
        raise ProtocolError("websocket handshake missing Sec-WebSocket-Key")
    accept = base64.b64encode(
        hashlib.sha1((key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode("ascii")).digest(),
    ).decode("ascii")
    writer.write(
        (
            "HTTP/1.1 101 Switching Protocols\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Accept: {accept}\r\n"
            "\r\n"
        ).encode("ascii"),
    )
    await writer.drain()


async def _read_client_text_frame(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
) -> str | None:
    while True:
        header = await reader.readexactly(2)
        first, second = header
        opcode = first & 0x0F
        masked = bool(second & 0x80)
        length = second & 0x7F
        if length == 126:
            length = struct.unpack("!H", await reader.readexactly(2))[0]
        elif length == 127:
            length = struct.unpack("!Q", await reader.readexactly(8))[0]
        mask = await reader.readexactly(4) if masked else b""
        payload = await reader.readexactly(length)
        if masked:
            payload = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))

        if opcode == 0x8:
            writer.write(b"\x88\x00")
            await writer.drain()
            return None
        if opcode == 0x9:
            writer.write(_encode_server_frame(payload, opcode=0xA))
            await writer.drain()
            continue
        if opcode != 0x1:
            raise ProtocolError(f"unsupported websocket opcode: {opcode}")
        return payload.decode("utf-8")


def _encode_server_text_frame(payload: bytes) -> bytes:
    return _encode_server_frame(payload, opcode=0x1)


def _encode_server_frame(payload: bytes, *, opcode: int) -> bytes:
    length = len(payload)
    if length < 126:
        prefix = bytes([0x80 | opcode, length])
    elif length <= 0xFFFF:
        prefix = bytes([0x80 | opcode, 126]) + struct.pack("!H", length)
    else:
        prefix = bytes([0x80 | opcode, 127]) + struct.pack("!Q", length)
    return prefix + payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=f"Run Mapper V2 local WS endpoint at {PULSEFIELD_WS_URL}.")
    parser.parse_args(argv)
    asyncio.run(serve_forever())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
