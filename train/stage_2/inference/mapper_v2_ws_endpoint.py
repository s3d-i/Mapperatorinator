from __future__ import annotations

import argparse
import asyncio
import base64
import errno
import hashlib
import json
import math
import struct
import time
import wave
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import torch

from train.stage_2.data.control_windows import normalize_difficulty
from train.stage_2.inference.model_runtime import ModelRuntime, ModelRuntimeConfig, load_model_runtime
from train.stage_2.inference.session_runtime import (
    DEFAULT_MAX_CONTROL_BATCH_SIZE,
    SessionRuntime,
    SessionRuntimeConfig,
)
from train.stage_2.model_mapper_v1.generation import (
    MapperGeneratedWindow,
    MapperGenerationStep,
    grammar_constrained_window_generation,
    transition_carry_state,
)
from train.stage_2.model_mapper_v1.replay import LNCarryState, empty_ln_carry_state, ln_carry_state_tensors
from train.stage_2.model_mapper_v1.tokenizer import MAPPER_WRITE_MS
from train.stage_2.model_mapper_v1.vocab import MapperV1Vocab


PULSEFIELD_WS_URL = "ws://localhost:8765"
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 8765
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MAPPER_CHECKPOINT_PATH = Path(
    "train/artifacts/runs/stage2_mapper_v2/"
    "stage2_mapper_v2_phase_b_global_d768_l8_b1/checkpoint.pt",
)
DEFAULT_CONTROL_CHECKPOINT_PATH = Path(
    "train/artifacts/runs/stage2_control_demo/"
    "stage2_control_demo_global_d384_l3_stride16_b6/checkpoints/checkpoint_step_002000.pt",
)
DEFAULT_TIME_SHIFT_LENGTH_PENALTY = 5.2


class ProtocolError(ValueError):
    pass


class PeerDisconnected(ConnectionError):
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
    token_send_interval_s: float = 0.02
    mapper_checkpoint_path: str | Path = DEFAULT_MAPPER_CHECKPOINT_PATH
    control_checkpoint_path: str | Path = DEFAULT_CONTROL_CHECKPOINT_PATH
    device: str = "mps"
    beatthis_device: str | None = None
    beatthis_float16: bool = False
    eager_load_beatthis: bool = True
    default_difficulty: float = 4.0
    max_control_batch_size: int = DEFAULT_MAX_CONTROL_BATCH_SIZE
    max_tokens: int = 512
    temperature: float = 0.0
    top_p: float | None = None
    use_incremental_mapper_decode: bool = True
    time_shift_length_penalty_alpha: float = DEFAULT_TIME_SHIFT_LENGTH_PENALTY
    seed: int | None = None
    reset_after_audio_end_ms: int = 2_000
    wall_clock_check_interval_s: float = 0.05


@dataclass(frozen=True)
class ReferenceClock:
    ref_time_ms: int
    local_host_time_send_ms: float
    received_local_host_time_ms: float


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
    difficulty: float | None = None
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


class MapperV2Backend(Protocol):
    models_ready: bool

    async def startup(self) -> None:
        ...

    async def prepare_audio(
        self,
        *,
        session_id: str,
        audio_path: Path,
        audio_length_ms: int,
        difficulty: float | None,
    ) -> None:
        ...

    async def iter_hitobject_tokens(
        self,
        *,
        session_id: str,
        audio_path: Path,
        audio_length_ms: int,
        window: DecoderWindow,
    ) -> AsyncIterator[HitObjectToken]:
        ...

    async def reset_session(self, session_id: str) -> None:
        ...


RuntimeLoader = Callable[[ModelRuntimeConfig], ModelRuntime]
SessionRuntimeFactory = Callable[[str, ModelRuntime, SessionRuntimeConfig], SessionRuntime]


class MapperV2InferenceBackend:
    """Real Mapper V2 inference backend for the local WebSocket endpoint."""

    def __init__(
        self,
        config: MapperV2WsConfig,
        *,
        runtime_loader: RuntimeLoader = load_model_runtime,
        session_runtime_factory: SessionRuntimeFactory | None = None,
    ) -> None:
        self.config = config
        self.models_ready = False
        self.model_runtime: ModelRuntime | None = None
        self._runtime_loader = runtime_loader
        self._session_runtime_factory = (
            _default_session_runtime_factory if session_runtime_factory is None else session_runtime_factory
        )
        self._session_runtimes: dict[str, SessionRuntime] = {}
        self._last_context_token_by_session: dict[str, int] = {}
        self._last_carry_state_by_session: dict[str, LNCarryState] = {}

    async def startup(self) -> None:
        if self.models_ready:
            return
        self.model_runtime = await asyncio.to_thread(self._load_model_runtime)
        self.models_ready = True

    async def prepare_audio(
        self,
        *,
        session_id: str,
        audio_path: Path,
        audio_length_ms: int,
        difficulty: float | None,
    ) -> None:
        model_runtime = self._require_model_runtime()
        normalized_difficulty = normalize_difficulty(
            self.config.default_difficulty if difficulty is None else float(difficulty),
        )
        session_runtime = self._session_runtime_factory(
            session_id,
            model_runtime,
            SessionRuntimeConfig(
                device=self.config.device,
                default_normalized_difficulty=normalized_difficulty,
                max_control_batch_size=int(self.config.max_control_batch_size),
            ),
        )
        await asyncio.to_thread(
            session_runtime.prepare_audio,
            audio_path,
            audio_length_ms=audio_length_ms,
            start_ms=0,
        )
        self._session_runtimes[session_id] = session_runtime
        self._last_context_token_by_session.pop(session_id, None)
        self._last_carry_state_by_session.pop(session_id, None)

    async def iter_hitobject_tokens(
        self,
        *,
        session_id: str,
        audio_path: Path,
        audio_length_ms: int,
        window: DecoderWindow,
    ) -> AsyncIterator[HitObjectToken]:
        del audio_path
        session_runtime = self._session_runtimes.get(session_id)
        if session_runtime is None:
            raise RuntimeError(f"session audio has not been prepared: {session_id}")
        starting_window = clamp_decoder_window_to_audio(window, audio_length_ms=audio_length_ms, config=self.config)
        for decode_window in decoder_windows_until_audio_end(
            starting_window,
            audio_length_ms=audio_length_ms,
            config=self.config,
        ):
            generated = await asyncio.to_thread(
                self._generate_window,
                session_id,
                session_runtime,
                decode_window,
                audio_length_ms,
            )
            for token in _hitobject_tokens_from_generated(generated, self._vocab()):
                yield token
                interval = max(0.0, float(self.config.token_send_interval_s))
                if interval:
                    await asyncio.sleep(interval)

    async def reset_session(self, session_id: str) -> None:
        session_runtime = self._session_runtimes.pop(session_id, None)
        self._last_context_token_by_session.pop(session_id, None)
        self._last_carry_state_by_session.pop(session_id, None)
        if session_runtime is not None:
            await asyncio.to_thread(session_runtime.reset_audio_cache)

    def _load_model_runtime(self) -> ModelRuntime:
        return self._runtime_loader(
            ModelRuntimeConfig(
                mapper_checkpoint_path=_resolve_repo_path(self.config.mapper_checkpoint_path),
                control_checkpoint_path=_resolve_repo_path(self.config.control_checkpoint_path),
                device=self.config.device,
                beatthis_device=self.config.beatthis_device,
                beatthis_float16=bool(self.config.beatthis_float16),
                eager_load_beatthis=bool(self.config.eager_load_beatthis),
            ),
        )

    def _generate_window(
        self,
        session_id: str,
        session_runtime: SessionRuntime,
        window: DecoderWindow,
        audio_length_ms: int,
    ) -> MapperGeneratedWindow:
        if session_runtime.audio_cache is None:
            raise RuntimeError("prepare_audio must finish before mapper generation")
        write_start_ms = int(window.start_ms)
        write_end_ms = int(window.end_ms)
        if write_end_ms - write_start_ms != int(self.config.decoder_window_ms):
            raise ValueError("decoder window span does not match config.decoder_window_ms")
        mapper_window_cache = session_runtime.prepare_mapper_window(
            start_ms=write_start_ms,
            end_ms=write_end_ms,
            include_control_attention_kv_cache=bool(self.config.use_incremental_mapper_decode),
        )

        vocab = self._vocab()
        carry_in = self._carry_in_for_window(session_id, write_start_ms)
        carry_out = empty_ln_carry_state(write_end_ms)
        is_full_chart_start = write_start_ms == 0 and not any(carry_in.open_mask)
        is_full_chart_end = write_end_ms >= int(audio_length_ms)
        left_context_tokens: tuple[int, ...] = ()
        if not is_full_chart_start:
            left_context_tokens = (self._left_context_token(session_id, vocab),)

        generator = _make_torch_generator(self.config.seed, device=session_runtime.device)
        logits_fn = _mapper_v2_logits_fn(
            model=session_runtime.model_runtime.mapper_model,
            vocab=vocab,
            device=session_runtime.device,
            normalized_difficulty=mapper_window_cache.normalized_difficulty,
            audio_batch={},
            control_batch=mapper_window_cache.as_model_batch(),
            ln_carry_in=carry_in,
            ln_carry_out=carry_out,
            is_full_chart_start=is_full_chart_start,
            is_full_chart_end=is_full_chart_end,
            use_incremental_decode=bool(self.config.use_incremental_mapper_decode),
            time_shift_length_penalty_alpha=float(self.config.time_shift_length_penalty_alpha),
        )
        generated = grammar_constrained_window_generation(
            vocab=vocab,
            write_start_ms=write_start_ms,
            write_end_ms=write_end_ms,
            ln_carry_in=carry_in,
            ln_carry_out=carry_out,
            logits_fn=logits_fn,
            left_context_tokens=left_context_tokens,
            is_full_chart_start=is_full_chart_start,
            is_full_chart_end=is_full_chart_end,
            max_tokens=int(self.config.max_tokens),
            temperature=float(self.config.temperature),
            top_p=self.config.top_p,
            generator=generator,
        )
        if generated.tokens:
            self._last_context_token_by_session[session_id] = int(generated.tokens[-1])
        if generated.completed:
            self._last_carry_state_by_session[session_id] = generated.terminal_state
        return generated

    def _carry_in_for_window(self, session_id: str, write_start_ms: int) -> LNCarryState:
        previous = self._last_carry_state_by_session.get(session_id)
        if previous is not None and int(previous.current_ms) == int(write_start_ms):
            return previous
        return empty_ln_carry_state(write_start_ms)

    def _left_context_token(self, session_id: str, vocab: MapperV1Vocab) -> int:
        token = self._last_context_token_by_session.get(session_id)
        if token is not None and token != vocab.bos_id:
            return int(token)
        return int(vocab.time_shift_token_id(10))

    def _vocab(self) -> MapperV1Vocab:
        runtime = self._require_model_runtime()
        return runtime.vocab

    def _require_model_runtime(self) -> ModelRuntime:
        if self.model_runtime is None:
            raise RuntimeError("models are not loaded; call startup first")
        return self.model_runtime


@dataclass
class InferenceEndpoint:
    config: MapperV2WsConfig = field(default_factory=MapperV2WsConfig)
    backend: MapperV2Backend | None = None

    def __post_init__(self) -> None:
        if self.backend is None:
            self.backend = MapperV2InferenceBackend(self.config)
        self.sessions: dict[str, SessionState] = {}
        self._startup_lock = asyncio.Lock()

    async def handle_message(self, raw_message: str | bytes | Mapping[str, Any], peer: JsonPeer) -> None:
        message = parse_json_message(raw_message)
        message_type = infer_message_type(message)

        if message_type == "ready":
            await self.startup()
            return
        if message_type in {"audio_path", "audio"}:
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
        assert self.backend is not None
        await self.backend.reset_session(session_id)
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
        raw_audio_path = audio_path_from_message(message)
        if not isinstance(raw_audio_path, str) or not raw_audio_path.strip():
            raise ProtocolError("audio_path must be a non-empty string")
        audio_path = Path(raw_audio_path).expanduser()
        audio_length_ms = audio_length_ms_from_message(message)
        if audio_length_ms is None:
            audio_length_ms = audio_length_ms_from_file(audio_path)
        if audio_length_ms is None:
            raise ProtocolError("audio_length_ms was omitted and audio duration could not be read from audio_path")
        difficulty = difficulty_from_message(message, default=self.config.default_difficulty)

        existing = self.sessions.get(session_id)
        if existing is not None:
            await self.stop_session(session_id, reason="replace_audio_path")

        session = SessionState(
            session_id=session_id,
            audio_path=audio_path,
            audio_length_ms=audio_length_ms,
            difficulty=difficulty,
        )
        self.sessions[session_id] = session
        log_ws_status(
            session_id=session_id,
            from_status="no_session",
            to_status="audio_preparing",
            reason="audio_path",
            audio_path=str(audio_path),
            audio_length_ms=audio_length_ms,
            difficulty=difficulty,
        )
        await self.backend.prepare_audio(
            session_id=session_id,
            audio_path=audio_path,
            audio_length_ms=audio_length_ms,
            difficulty=difficulty,
        )
        session.audio_prepared = True
        log_ws_status(
            session_id=session_id,
            from_status="audio_preparing",
            to_status="audio_ready",
            reason="audio_prepared",
            audio_path=str(audio_path),
            audio_length_ms=session.audio_length_ms,
            difficulty=difficulty,
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
        audio_length_ms = session.audio_length_ms
        from_status = session_status(session)
        window = clamp_decoder_window_to_audio(
            choose_decoder_window(clock, self.config),
            audio_length_ms=audio_length_ms,
            config=self.config,
        )
        session.reference_clock = clock
        session.decoder_window = window
        await _cancel_task(session.stream_task)
        await _cancel_task(session.wall_clock_reset_task)
        session.stream_task = asyncio.create_task(self._stream_tokens(session, window, peer))
        session.wall_clock_reset_task = asyncio.create_task(self._reset_after_audio_end(session))
        reset_local_host_time_ms = audio_end_reset_host_time_ms(
            reference_clock=clock,
            audio_length_ms=audio_length_ms,
            reset_after_audio_end_ms=self.config.reset_after_audio_end_ms,
        )
        log_ws_status(
            session_id=session_id,
            from_status=from_status,
            to_status="streaming",
            reason="reference_time",
            ref_time_ms=clock.ref_time_ms,
            send_local_host_time_ms=clock.local_host_time_send_ms,
            received_local_host_time_ms=clock.received_local_host_time_ms,
            audio_length_ms=audio_length_ms,
            difficulty=session.difficulty,
            reset_local_host_time_ms=reset_local_host_time_ms,
        )

    async def _reset_after_audio_end(self, session: SessionState) -> None:
        if session.reference_clock is None or session.audio_length_ms is None:
            return
        reset_local_host_time_ms = audio_end_reset_host_time_ms(
            reference_clock=session.reference_clock,
            audio_length_ms=session.audio_length_ms,
            reset_after_audio_end_ms=self.config.reset_after_audio_end_ms,
        )
        check_interval_s = max(0.01, float(self.config.wall_clock_check_interval_s))
        while self.sessions.get(session.session_id) is session:
            if host_time_ms_reached(reset_local_host_time_ms):
                await self.stop_session(session.session_id, reason="wall_clock_audio_end")
                return
            await asyncio.sleep(check_interval_s)

    async def _stream_tokens(self, session: SessionState, window: DecoderWindow, peer: JsonPeer) -> None:
        assert self.backend is not None
        assert session.audio_path is not None
        if session.audio_length_ms is None:
            raise RuntimeError("audio duration must be resolved before streaming")
        try:
            async for hitobject in self.backend.iter_hitobject_tokens(
                session_id=session.session_id,
                audio_path=session.audio_path,
                audio_length_ms=session.audio_length_ms,
                window=window,
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
        except PeerDisconnected:
            return
        except Exception as exc:
            if _is_expected_socket_disconnect(exc):
                return
            if self.sessions.get(session.session_id) is session:
                try:
                    await peer.send_json({"type": "error", "session_id": session.session_id, "error": str(exc)})
                except PeerDisconnected:
                    return
                except Exception as send_exc:
                    if _is_expected_socket_disconnect(send_exc):
                        return
                    raise


def _default_session_runtime_factory(
    session_id: str,
    model_runtime: ModelRuntime,
    config: SessionRuntimeConfig,
) -> SessionRuntime:
    return SessionRuntime(session_id=session_id, model_runtime=model_runtime, config=config)


def _resolve_repo_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    if resolved.is_absolute():
        return resolved
    cwd_path = Path.cwd() / resolved
    if cwd_path.exists():
        return cwd_path
    return REPOSITORY_ROOT / resolved


def _make_torch_generator(seed: int | None, *, device: torch.device) -> torch.Generator | None:
    if seed is None:
        return None
    generator_device = "cpu" if device.type == "mps" else device.type
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(int(seed))
    return generator


def _hitobject_tokens_from_generated(
    generated: MapperGeneratedWindow,
    vocab: MapperV1Vocab,
) -> tuple[HitObjectToken, ...]:
    tokens: list[HitObjectToken] = []
    for token_id, state_before in zip(generated.tokens, generated.states_before, strict=True):
        token_id = int(token_id)
        if not vocab.is_event_token(token_id):
            continue
        actions = vocab.decode_event(token_id)
        tokens.append(
            HitObjectToken(
                token_id=token_id,
                token_name=vocab.token_name(token_id),
                ms_in_ref_audio=int(state_before.current_ms),
                actions=tuple(action.value for action in actions),
            ),
        )
    return tuple(tokens)


def _mapper_v2_logits_fn(
    *,
    model: torch.nn.Module,
    vocab: MapperV1Vocab,
    device: torch.device,
    normalized_difficulty: float,
    audio_batch: Mapping[str, torch.Tensor],
    control_batch: Mapping[str, torch.Tensor],
    ln_carry_in: LNCarryState,
    ln_carry_out: LNCarryState,
    is_full_chart_start: bool,
    is_full_chart_end: bool,
    use_incremental_decode: bool,
    time_shift_length_penalty_alpha: float,
):
    time_shift_penalty = _time_shift_length_penalty_tensors(
        vocab,
        alpha=time_shift_length_penalty_alpha,
        device=device,
    )
    incremental_decode = (
        bool(use_incremental_decode)
        and hasattr(model, "create_empty_decode_state")
        and hasattr(model, "incremental_decode_next_token")
    )
    decode_state: Any | None = None
    decoded_prefix_tokens: tuple[int, ...] = ()
    last_incremental_logits: torch.Tensor | None = None
    write_start_ms_tensor = torch.tensor([ln_carry_in.current_ms], dtype=torch.long, device=device)
    write_end_ms_tensor: torch.Tensor | None = None
    full_start_tensor = torch.tensor([bool(is_full_chart_start)], dtype=torch.bool, device=device)
    full_end_tensor = torch.tensor([bool(is_full_chart_end)], dtype=torch.bool, device=device)
    difficulty_tensor = torch.tensor([float(normalized_difficulty)], dtype=torch.float32, device=device)
    carry_in_batch = _carry_state_batch(ln_carry_in, device=device)
    carry_out_batch = _carry_state_batch(ln_carry_out, device=device)
    control_attention_kv_cache = control_batch.get("control_attention_kv_cache")

    def logits_fn(step: MapperGenerationStep) -> torch.Tensor:
        nonlocal decode_state, decoded_prefix_tokens, last_incremental_logits, write_end_ms_tensor

        decoder_input_tokens = step.decoder_input_tokens.to(device=device, dtype=torch.long).unsqueeze(0)
        states = _target_fragment_state_batch(
            generated_tokens=step.generated_tokens,
            vocab=vocab,
            write_start_ms=step.write_start_ms,
            write_end_ms=step.write_end_ms,
            ln_carry_in=ln_carry_in,
            ln_carry_out=ln_carry_out,
            device=device,
        )
        write_end_ms_tensor = torch.tensor([step.write_end_ms], dtype=torch.long, device=device)
        if incremental_decode:
            prefix_tokens = tuple(int(token) for token in step.decoder_input_tokens.reshape(-1).tolist())
            if not prefix_tokens:
                raise RuntimeError("mapper decoder prefix cannot be empty")
            if decode_state is None or prefix_tokens[: len(decoded_prefix_tokens)] != decoded_prefix_tokens:
                decode_state = model.create_empty_decode_state(batch_size=1, device=device)
                decoded_prefix_tokens = ()
                last_incremental_logits = None
            if len(prefix_tokens) < len(decoded_prefix_tokens):
                decode_state = model.create_empty_decode_state(batch_size=1, device=device)
                decoded_prefix_tokens = ()
                last_incremental_logits = None
            for position in range(len(decoded_prefix_tokens), len(prefix_tokens)):
                with torch.inference_mode():
                    output = model.incremental_decode_next_token(
                        decode_state=decode_state,
                        decoder_input_token=decoder_input_tokens[:, position],
                        current_ms=states["current_ms"][:, position],
                        open_mask=states["open_mask"][:, position],
                        open_start_ms=states["open_start_ms"][:, position],
                        open_age_ms=states["open_age_ms"][:, position],
                        write_start_ms=write_start_ms_tensor,
                        write_end_ms=write_end_ms_tensor,
                        is_full_chart_start=full_start_tensor,
                        is_full_chart_end=full_end_tensor,
                        ln_carry_in=carry_in_batch,
                        ln_carry_out=carry_out_batch,
                        density_teacher_8s=control_batch["density_teacher_8s"],
                        control_memory_8s=control_batch.get("control_memory_8s"),
                        projected_control_memory_8s=control_batch.get("projected_control_memory_8s"),
                        control_attention_kv_cache=control_attention_kv_cache,
                        normalized_difficulty=difficulty_tensor,
                        global_memory=control_batch.get("global_memory"),
                        global_memory_padding_mask=control_batch.get("global_memory_padding_mask"),
                        global_position_features=control_batch.get("global_position_features"),
                        global_attention_kv_cache=control_batch.get("global_attention_kv_cache"),
                        position=position,
                    )
                decode_state = output.decode_state
                last_incremental_logits = output.logits_final[0].detach()
            decoded_prefix_tokens = prefix_tokens
            if last_incremental_logits is None:
                raise RuntimeError("incremental mapper decode did not produce logits")
            return _apply_time_shift_length_penalty(
                last_incremental_logits,
                time_shift_penalty=time_shift_penalty,
            )

        current_ms = states["current_ms"]
        target_tokens = torch.full_like(decoder_input_tokens, vocab.pad_id)
        at_write_end = current_ms == int(step.write_end_ms)
        target_tokens = torch.where(at_write_end, torch.full_like(target_tokens, vocab.eos_id), target_tokens)
        batch: dict[str, torch.Tensor | Mapping[str, torch.Tensor]] = {
            **audio_batch,
            **control_batch,
            "decoder_input_tokens": decoder_input_tokens,
            "target_fragment_tokens": target_tokens,
            "target_fragment_mask": torch.ones_like(decoder_input_tokens, dtype=torch.bool),
            "target_fragment_states": states,
            "ln_carry_in": carry_in_batch,
            "ln_carry_out": carry_out_batch,
            "write_start_ms": write_start_ms_tensor,
            "write_end_ms": write_end_ms_tensor,
            "is_full_chart_start": full_start_tensor,
            "is_full_chart_end": full_end_tensor,
            "normalized_difficulty": difficulty_tensor,
        }
        with torch.inference_mode():
            output = model(batch)
        logits = output.logits_final[0, -1].detach()
        return _apply_time_shift_length_penalty(logits, time_shift_penalty=time_shift_penalty)

    return logits_fn


def _time_shift_length_penalty_tensors(
    vocab: MapperV1Vocab,
    *,
    alpha: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    alpha = float(alpha)
    if alpha < 0.0:
        raise ValueError(f"time_shift_length_penalty_alpha must be non-negative, got {alpha}")
    if alpha == 0.0:
        return None

    token_ids = [int(token_id) for token_id in vocab.time_shift_token_ids]
    if not token_ids:
        return None
    return (
        torch.tensor(token_ids, dtype=torch.long, device=device),
        torch.full((len(token_ids),), alpha, dtype=torch.float32, device=device),
    )


def _apply_time_shift_length_penalty(
    logits: torch.Tensor,
    *,
    time_shift_penalty: tuple[torch.Tensor, torch.Tensor] | None,
) -> torch.Tensor:
    if time_shift_penalty is None:
        return logits
    token_ids, penalties = time_shift_penalty
    adjusted = logits.clone()
    adjusted[token_ids] -= penalties.to(dtype=adjusted.dtype)
    return adjusted


def _target_fragment_state_batch(
    *,
    generated_tokens: Sequence[int],
    vocab: MapperV1Vocab,
    write_start_ms: int,
    write_end_ms: int,
    ln_carry_in: LNCarryState,
    ln_carry_out: LNCarryState,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    states = [ln_carry_in]
    state = ln_carry_in
    for token_id in generated_tokens:
        state = transition_carry_state(
            state,
            int(token_id),
            vocab=vocab,
            write_start_ms=write_start_ms,
            write_end_ms=write_end_ms,
            allow_bos=False,
            allow_eos=False,
        )
        states.append(state)
    if states[-1] != ln_carry_out and int(states[-1].current_ms) == int(write_end_ms):
        raise ValueError("generated prefix reached write_end_ms without matching ln_carry_out")
    tensors = [_carry_state_tensors_1d(state, device=device) for state in states]
    return {
        "current_ms": torch.stack([item["current_ms"] for item in tensors], dim=0).unsqueeze(0),
        "open_mask": torch.stack([item["open_mask"] for item in tensors], dim=0).unsqueeze(0),
        "open_start_ms": torch.stack([item["open_start_ms"] for item in tensors], dim=0).unsqueeze(0),
        "open_age_ms": torch.stack([item["open_age_ms"] for item in tensors], dim=0).unsqueeze(0),
    }


def _carry_state_batch(carry: LNCarryState, *, device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.unsqueeze(0) for key, value in _carry_state_tensors_1d(carry, device=device).items()}


def _carry_state_tensors_1d(carry: LNCarryState, *, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device=device)
        for key, value in ln_carry_state_tensors(carry).items()
    }


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
    if "audio_path" in message or "audio" in message:
        return "audio_path"
    has_ref_time_ms = "ref_time_ms" in message
    has_send_local_host_time_ms = "local_host_time_send_ms" in message
    if has_ref_time_ms and has_send_local_host_time_ms:
        return "reference_time"
    raise ProtocolError("message must include type or a recognized control field")


def require_session_id(message: Mapping[str, Any]) -> str:
    session_id = message.get("session_id")
    if not isinstance(session_id, str) or not session_id.strip():
        raise ProtocolError("session_id must be a non-empty string")
    return session_id


def audio_path_from_message(message: Mapping[str, Any]) -> str | None:
    value = message.get("audio_path")
    if isinstance(value, str):
        return value
    audio = message.get("audio")
    if isinstance(audio, str):
        return audio
    if isinstance(audio, Mapping):
        nested = audio.get("audio_path", audio.get("path"))
        if isinstance(nested, str):
            return nested
    return None


def reference_clock_from_message(message: Mapping[str, Any]) -> ReferenceClock:
    return ReferenceClock(
        ref_time_ms=_required_int(message, "ref_time_ms"),
        local_host_time_send_ms=_required_float(message, "local_host_time_send_ms"),
        received_local_host_time_ms=current_host_time_ms(),
    )


def choose_decoder_window(clock: ReferenceClock, config: MapperV2WsConfig) -> DecoderWindow:
    elapsed_ms = max(0.0, clock.received_local_host_time_ms - clock.local_host_time_send_ms)
    estimated_ref_ms = max(0, clock.ref_time_ms + elapsed_ms)
    target_ms = estimated_ref_ms + max(0, int(config.decoder_lead_ms))
    window_ms = int(config.decoder_window_ms)
    if window_ms <= 0:
        raise ValueError("decoder_window_ms must be positive")
    start_ms = int((target_ms + window_ms - 1) // window_ms) * window_ms
    return DecoderWindow(start_ms=start_ms, end_ms=start_ms + window_ms)


def clamp_decoder_window_to_audio(
    window: DecoderWindow,
    *,
    audio_length_ms: int,
    config: MapperV2WsConfig,
) -> DecoderWindow:
    window_ms = int(config.decoder_window_ms)
    if window_ms <= 0:
        raise ValueError("decoder_window_ms must be positive")
    latest_start_ms = (max(1, int(audio_length_ms)) - 1) // window_ms * window_ms
    if int(window.start_ms) <= latest_start_ms:
        return window
    return DecoderWindow(start_ms=latest_start_ms, end_ms=latest_start_ms + window_ms)


def decoder_windows_until_audio_end(
    window: DecoderWindow,
    *,
    audio_length_ms: int,
    config: MapperV2WsConfig,
) -> tuple[DecoderWindow, ...]:
    window_ms = int(config.decoder_window_ms)
    if window_ms <= 0:
        raise ValueError("decoder_window_ms must be positive")
    start_ms = int(window.start_ms)
    end_ms = int(window.end_ms)
    if end_ms - start_ms != window_ms:
        raise ValueError("decoder window span does not match config.decoder_window_ms")
    latest_start_ms = (max(1, int(audio_length_ms)) - 1) // window_ms * window_ms
    if start_ms > latest_start_ms:
        start_ms = latest_start_ms
    return tuple(
        DecoderWindow(start_ms=current_start_ms, end_ms=current_start_ms + window_ms)
        for current_start_ms in range(start_ms, latest_start_ms + 1, window_ms)
    )


def current_host_time_ms() -> float:
    return time.monotonic() * 1000.0


def audio_end_reset_host_time_ms(
    *,
    reference_clock: ReferenceClock,
    audio_length_ms: int,
    reset_after_audio_end_ms: int,
) -> float:
    remaining_audio_ms = max(0, int(audio_length_ms) - int(reference_clock.ref_time_ms))
    return float(reference_clock.local_host_time_send_ms) + remaining_audio_ms + max(
        0.0,
        float(reset_after_audio_end_ms),
    )


def host_time_ms_reached(deadline_ms: float, now_ms: float | None = None) -> bool:
    now_ms = current_host_time_ms() if now_ms is None else float(now_ms)
    return now_ms >= float(deadline_ms)


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
        "local_host_time_ms": current_host_time_ms(),
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


def difficulty_from_message(message: Mapping[str, Any], *, default: float) -> float:
    value = message.get("difficulty", default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolError("difficulty must be numeric")
    value = float(value)
    if not math.isfinite(value):
        raise ProtocolError("difficulty must be finite")
    try:
        normalize_difficulty(value)
    except ValueError as exc:
        raise ProtocolError(str(exc)) from exc
    return value


def audio_length_ms_from_file(audio_path: Path) -> int | None:
    if not audio_path.exists():
        return None
    try:
        from mutagen import File as MutagenFile
    except ImportError:
        pass
    else:
        try:
            audio = MutagenFile(audio_path)
        except Exception:
            audio = None
        if audio is not None and getattr(audio, "info", None) is not None:
            length = getattr(audio.info, "length", None)
            if isinstance(length, (int, float)) and math.isfinite(float(length)) and float(length) > 0:
                return int(round(float(length) * 1000.0))
    try:
        with wave.open(str(audio_path), "rb") as audio:
            frame_count = int(audio.getnframes())
            frame_rate = int(audio.getframerate())
    except (EOFError, OSError, wave.Error):
        return None
    if frame_count <= 0 or frame_rate <= 0:
        return None
    return int(round(frame_count / frame_rate * 1000.0))


def _required_int(message: Mapping[str, Any], key: str) -> int:
    value = message.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProtocolError(f"{key} must be an integer")
    return int(value)


def _required_float(message: Mapping[str, Any], key: str) -> float:
    value = message.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolError(f"{key} must be numeric")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ProtocolError(f"{key} must be finite and non-negative")
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


_EXPECTED_SOCKET_DISCONNECT_ERRNOS = {
    errno.ECONNABORTED,
    errno.ECONNRESET,
    errno.ENOTCONN,
    errno.EPIPE,
    errno.ETIMEDOUT,
}


def _is_expected_socket_disconnect(exc: BaseException) -> bool:
    if isinstance(exc, (asyncio.IncompleteReadError, ConnectionError)):
        return True
    if isinstance(exc, OSError) and exc.errno in _EXPECTED_SOCKET_DISCONNECT_ERRNOS:
        return True
    return False


async def _drain_writer(writer: asyncio.StreamWriter) -> None:
    if writer.is_closing():
        raise PeerDisconnected("websocket peer disconnected")
    try:
        await writer.drain()
    except Exception as exc:
        if _is_expected_socket_disconnect(exc):
            raise PeerDisconnected("websocket peer disconnected") from exc
        raise


async def _close_writer(writer: asyncio.StreamWriter) -> None:
    writer.close()
    try:
        await writer.wait_closed()
    except Exception as exc:
        if not _is_expected_socket_disconnect(exc):
            raise


class _WebSocketPeer:
    def __init__(self, writer: asyncio.StreamWriter) -> None:
        self._writer = writer
        self._send_lock = asyncio.Lock()

    async def send_json(self, payload: Mapping[str, Any]) -> None:
        data = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        async with self._send_lock:
            try:
                self._writer.write(_encode_server_text_frame(data))
                await _drain_writer(self._writer)
            except Exception as exc:
                if _is_expected_socket_disconnect(exc):
                    raise PeerDisconnected("websocket peer disconnected") from exc
                raise


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
        try:
            await _accept_websocket_handshake(reader, writer)
        except ProtocolError as exc:
            await _send_http_error(writer, status=400, reason="Bad Request", body=str(exc))
            return
        while True:
            message = await _read_client_text_frame(reader, writer)
            if message is None:
                break
            try:
                await endpoint.handle_message(message, peer)
            except ProtocolError as exc:
                await peer.send_json({"type": "error", "error": str(exc)})
            except PeerDisconnected:
                return
    except Exception as exc:
        if not _is_expected_socket_disconnect(exc):
            raise
    finally:
        await _close_writer(writer)


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
    await _drain_writer(writer)


async def _send_http_error(
    writer: asyncio.StreamWriter,
    *,
    status: int,
    reason: str,
    body: str,
) -> None:
    body_bytes = body.encode("utf-8", errors="replace")
    writer.write(
        (
            f"HTTP/1.1 {int(status)} {reason}\r\n"
            "Connection: close\r\n"
            "Content-Type: text/plain; charset=utf-8\r\n"
            f"Content-Length: {len(body_bytes)}\r\n"
            "\r\n"
        ).encode("ascii")
        + body_bytes,
    )
    await _drain_writer(writer)


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
            await _drain_writer(writer)
            return None
        if opcode == 0x9:
            writer.write(_encode_server_frame(payload, opcode=0xA))
            await _drain_writer(writer)
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
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--beatthis-device", default=None)
    parser.add_argument("--difficulty", type=float, default=4.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--mapper-checkpoint-path", type=Path, default=DEFAULT_MAPPER_CHECKPOINT_PATH)
    parser.add_argument("--control-checkpoint-path", type=Path, default=DEFAULT_CONTROL_CHECKPOINT_PATH)
    args = parser.parse_args(argv)

    config = MapperV2WsConfig(
        host=args.host,
        port=args.port,
        mapper_checkpoint_path=args.mapper_checkpoint_path,
        control_checkpoint_path=args.control_checkpoint_path,
        device=args.device,
        beatthis_device=args.beatthis_device,
        default_difficulty=float(args.difficulty),
        max_tokens=int(args.max_tokens),
    )
    asyncio.run(serve_forever(InferenceEndpoint(config=config)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
