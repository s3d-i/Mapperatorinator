from __future__ import annotations

import gc
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch

from train.stage_2.data.control_windows import TARGET_WINDOW_LENGTH_FRAMES
from train.stage_2.data.mapper_v1_windows import control_teacher_stacked_slices_batch
from train.stage_2.features.mel import load_full_song_packed_mel_20ms
from train.stage_2.inference.model_runtime import ModelRuntime, release_torch_cache
from train.stage_2.model_control.context import TARGET_OFFSET_IN_CONTEXT
from train.stage_2.model_mapper_v1.tokenizer import MAPPER_DENSITY_FRAME_MS, MAPPER_DENSITY_FRAMES
from train.stage_2.timing.grid_fitting import GridFitter, GridFitterConfig, TimingFitResult
from train.stage_2.timing.rendering.dense_timing_v2 import render_dense_timing_v2
from train.stage_2.timing.schema import FittedTimingGrid, FrameTimingPrediction


PACKED_MEL_CHANNELS = 160
DENSE_TIMING_V2_CHANNELS = 4
DEFAULT_MAX_CONTROL_BATCH_SIZE = 12

MelLoader = Callable[[str | Path], Any]


class TimingProvider(Protocol):
    def predict_file(self, audio_path: str | Path) -> FrameTimingPrediction:
        ...


class TimingGridFitter(Protocol):
    def fit(self, prediction: FrameTimingPrediction) -> TimingFitResult:
        ...


@dataclass(frozen=True)
class SessionRuntimeConfig:
    device: str | torch.device | None = None
    minimum_frame_count: int = MAPPER_DENSITY_FRAMES
    default_normalized_difficulty: float = 0.0
    max_control_batch_size: int = DEFAULT_MAX_CONTROL_BATCH_SIZE
    grid_fitter_config: GridFitterConfig = field(default_factory=GridFitterConfig)

    def __post_init__(self) -> None:
        if isinstance(self.minimum_frame_count, bool) or not isinstance(self.minimum_frame_count, int):
            raise TypeError("minimum_frame_count must be an integer")
        if self.minimum_frame_count <= 0:
            raise ValueError("minimum_frame_count must be positive")
        if not isinstance(self.default_normalized_difficulty, (int, float)) or isinstance(
            self.default_normalized_difficulty,
            bool,
        ):
            raise TypeError("default_normalized_difficulty must be numeric")
        if not np.isfinite(float(self.default_normalized_difficulty)):
            raise ValueError("default_normalized_difficulty must be finite")
        if isinstance(self.max_control_batch_size, bool) or not isinstance(self.max_control_batch_size, int):
            raise TypeError("max_control_batch_size must be an integer")
        if self.max_control_batch_size <= 0:
            raise ValueError("max_control_batch_size must be positive")


@dataclass(frozen=True)
class SessionAudioCache:
    session_id: str
    audio_path: Path
    audio_length_ms: int
    audio_length_source: str
    full_mel: torch.Tensor
    full_dense_timing_v2: torch.Tensor
    padding_mask: torch.Tensor
    frame_count_tensor: torch.Tensor
    source_frame_count_tensor: torch.Tensor
    source_frame_count: int
    padded_frame_count: int
    beatthis_prediction: FrameTimingPrediction
    timing_fit_result: TimingFitResult
    timing_grid: FittedTimingGrid

    def as_model_batch(self) -> dict[str, torch.Tensor]:
        return {
            "full_mel": self.full_mel,
            "full_dense_timing_v2": self.full_dense_timing_v2,
            "padding_mask": self.padding_mask,
            "frame_count": self.frame_count_tensor,
            "source_frame_count": self.source_frame_count_tensor,
        }


@dataclass(frozen=True)
class SessionControlCache:
    session_id: str
    start_ms: int
    target_start_frame: int
    normalized_difficulty: float
    control_slice_start_frames: torch.Tensor
    target_start_frame_tensor: torch.Tensor
    normalized_difficulty_tensor: torch.Tensor
    control_memory_8s: torch.Tensor

    def as_model_batch(self) -> dict[str, torch.Tensor]:
        return {
            "control_memory_8s": self.control_memory_8s,
            "control_slice_start_frames": self.control_slice_start_frames,
            "target_start_frame": self.target_start_frame_tensor,
            "normalized_difficulty": self.normalized_difficulty_tensor,
        }


@dataclass(frozen=True)
class SessionControlBatchCache:
    session_id: str
    start_ms_values: tuple[int, ...]
    target_start_frames: tuple[int, ...]
    normalized_difficulty: float
    control_slice_start_frames: torch.Tensor
    target_start_frame_tensor: torch.Tensor
    normalized_difficulty_tensor: torch.Tensor
    control_memory_8s: torch.Tensor

    def as_model_batch(self) -> dict[str, torch.Tensor]:
        return {
            "control_memory_8s": self.control_memory_8s,
            "control_slice_start_frames": self.control_slice_start_frames,
            "target_start_frame": self.target_start_frame_tensor,
            "normalized_difficulty": self.normalized_difficulty_tensor,
        }


@dataclass(frozen=True)
class SessionFullControlCache:
    session_id: str
    start_ms_values: tuple[int, ...]
    target_start_frames: tuple[int, ...]
    normalized_difficulty: float
    max_batch_size: int
    control_memory_8s: torch.Tensor

    def as_model_batch(self) -> dict[str, torch.Tensor]:
        return {
            "control_memory_8s": self.control_memory_8s,
            "target_start_frame": torch.tensor(
                self.target_start_frames,
                dtype=torch.long,
                device=self.control_memory_8s.device,
            ),
        }


@dataclass
class SessionRuntime:
    session_id: str
    model_runtime: ModelRuntime
    config: SessionRuntimeConfig = field(default_factory=SessionRuntimeConfig)
    mel_loader: MelLoader = load_full_song_packed_mel_20ms
    grid_fitter: TimingGridFitter | None = None
    audio_cache: SessionAudioCache | None = field(default=None, init=False)
    control_cache: SessionControlCache | None = field(default=None, init=False)
    control_batch_cache: SessionControlBatchCache | None = field(default=None, init=False)
    full_control_cache: SessionFullControlCache | None = field(default=None, init=False)
    device: torch.device = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.session_id, str) or not self.session_id.strip():
            raise ValueError("session_id must be a non-empty string")
        self.device = _resolve_session_device(self.model_runtime, self.config.device)
        if self.grid_fitter is None:
            self.grid_fitter = GridFitter(self.config.grid_fitter_config)

    def prepare_audio(
        self,
        audio_path: str | Path,
        *,
        audio_length_ms: int | None = None,
        start_ms: int = 0,
    ) -> SessionAudioCache:
        self.reset_audio_cache()

        path = Path(audio_path).expanduser()
        packed_mel_cpu = _as_2d_float32_cpu_tensor(
            self.mel_loader(path),
            channels=PACKED_MEL_CHANNELS,
            name="packed_mel",
            source=path,
        )
        source_frame_count = int(packed_mel_cpu.shape[0])
        if source_frame_count <= 0:
            raise ValueError(f"packed_mel for {path} must contain at least one frame")
        padded_frame_count = max(source_frame_count, int(self.config.minimum_frame_count))
        resolved_audio_length_ms, audio_length_source = _resolve_audio_length_ms(
            audio_length_ms=audio_length_ms,
            source_frame_count=source_frame_count,
        )

        provider = _timing_provider(self.model_runtime)
        assert self.grid_fitter is not None
        with torch.inference_mode():
            prediction = provider.predict_file(path)
        fit_result = self.grid_fitter.fit(prediction)
        dense_timing_cpu = _as_2d_float32_cpu_tensor(
            render_dense_timing_v2(
                fit_result.grid,
                input_start_ms=0.0,
                frame_count=padded_frame_count,
            ),
            channels=DENSE_TIMING_V2_CHANNELS,
            name="full_dense_timing_v2",
            source=path,
        )

        full_mel_cpu = torch.zeros((padded_frame_count, PACKED_MEL_CHANNELS), dtype=torch.float32)
        full_mel_cpu[:source_frame_count].copy_(packed_mel_cpu)
        padding_mask = torch.ones((1, padded_frame_count), dtype=torch.bool, device=self.device)
        padding_mask[:, :source_frame_count] = False

        cache = SessionAudioCache(
            session_id=self.session_id,
            audio_path=path,
            audio_length_ms=resolved_audio_length_ms,
            audio_length_source=audio_length_source,
            full_mel=full_mel_cpu.unsqueeze(0).to(device=self.device, dtype=torch.float32),
            full_dense_timing_v2=dense_timing_cpu.unsqueeze(0).to(device=self.device, dtype=torch.float32),
            padding_mask=padding_mask,
            frame_count_tensor=torch.tensor([padded_frame_count], dtype=torch.long, device=self.device),
            source_frame_count_tensor=torch.tensor([source_frame_count], dtype=torch.long, device=self.device),
            source_frame_count=source_frame_count,
            padded_frame_count=padded_frame_count,
            beatthis_prediction=prediction,
            timing_fit_result=fit_result,
            timing_grid=fit_result.grid,
        )
        self.audio_cache = cache
        self.prepare_control(start_ms=start_ms)
        assert self.audio_cache is not None
        return self.audio_cache

    def prepare_control(self, *, start_ms: int = 0) -> SessionControlCache:
        batch_cache = self.prepare_control_batch(start_ms_values=(start_ms,))
        start_ms = batch_cache.start_ms_values[0]
        target_start_frame = batch_cache.target_start_frames[0]
        control_slice_start_frames = batch_cache.control_slice_start_frames[:1].contiguous()
        target_start_frame_tensor = batch_cache.target_start_frame_tensor[:1].contiguous()
        normalized_difficulty_tensor = batch_cache.normalized_difficulty_tensor[:1].contiguous()
        control_memory_8s = batch_cache.control_memory_8s[:1].contiguous()

        cache = SessionControlCache(
            session_id=self.session_id,
            start_ms=start_ms,
            target_start_frame=target_start_frame,
            normalized_difficulty=batch_cache.normalized_difficulty,
            control_slice_start_frames=control_slice_start_frames,
            target_start_frame_tensor=target_start_frame_tensor,
            normalized_difficulty_tensor=normalized_difficulty_tensor,
            control_memory_8s=control_memory_8s,
        )
        self.control_cache = cache
        return cache

    def prepare_control_batch(
        self,
        *,
        start_ms_values: Sequence[int],
        max_batch_size: int | None = None,
    ) -> SessionControlBatchCache:
        self.control_cache = None
        self.control_batch_cache = None
        self.full_control_cache = None
        start_ms_tuple = _validate_start_ms_values(start_ms_values)
        effective_max_batch_size = _validate_max_batch_size(
            self.config.max_control_batch_size if max_batch_size is None else max_batch_size,
        )
        if len(start_ms_tuple) > effective_max_batch_size:
            raise ValueError(
                f"control batch size must be <= {effective_max_batch_size}, got {len(start_ms_tuple)}"
            )
        target_start_frames = tuple(start_ms // MAPPER_DENSITY_FRAME_MS for start_ms in start_ms_tuple)
        required_frame_count = max(target_start_frames) + MAPPER_DENSITY_FRAMES
        audio_cache = self._ensure_audio_cache_frame_count(required_frame_count)

        if MAPPER_DENSITY_FRAMES % TARGET_WINDOW_LENGTH_FRAMES != 0:
            raise ValueError("MAPPER_DENSITY_FRAMES must be divisible by TARGET_WINDOW_LENGTH_FRAMES")
        slice_count = MAPPER_DENSITY_FRAMES // TARGET_WINDOW_LENGTH_FRAMES
        if slice_count != 4:
            raise ValueError(f"control teacher requires four aligned slices, got {slice_count}")

        batch_size = len(start_ms_tuple)
        control_slice_start_frames = torch.tensor(
            [
                [
                    target_start_frame + offset
                    for offset in range(0, MAPPER_DENSITY_FRAMES, TARGET_WINDOW_LENGTH_FRAMES)
                ]
                for target_start_frame in target_start_frames
            ],
            dtype=torch.long,
            device=self.device,
        )
        target_start_frame_tensor = torch.tensor(target_start_frames, dtype=torch.long, device=self.device)
        normalized_difficulty = float(self.config.default_normalized_difficulty)
        normalized_difficulty_tensor = torch.full(
            (batch_size,),
            normalized_difficulty,
            dtype=torch.float32,
            device=self.device,
        )
        control_batch = {
            "full_mel": audio_cache.full_mel.expand(batch_size, -1, -1),
            "full_dense_timing_v2": audio_cache.full_dense_timing_v2.expand(batch_size, -1, -1),
            "padding_mask": audio_cache.padding_mask.expand(batch_size, -1),
            "frame_count": audio_cache.frame_count_tensor.expand(batch_size),
            "source_frame_count": audio_cache.source_frame_count_tensor.expand(batch_size),
            "target_start_frame": target_start_frame_tensor,
            "control_slice_start_frames": control_slice_start_frames,
            "normalized_difficulty": normalized_difficulty_tensor,
        }

        control_model = _control_model(self.model_runtime)
        with torch.inference_mode():
            stacked_control_batch = control_teacher_stacked_slices_batch(control_batch)
            control_output = control_model(
                context_mel=stacked_control_batch["context_mel"],
                context_dense_timing_v2=stacked_control_batch["context_dense_timing_v2"],
                normalized_difficulty=stacked_control_batch["normalized_difficulty"].reshape(batch_size * slice_count),
                context_padding_mask=stacked_control_batch["context_padding_mask"],
                full_mel=stacked_control_batch.get("full_mel"),
                full_dense_timing_v2=stacked_control_batch.get("full_dense_timing_v2"),
                padding_mask=stacked_control_batch.get("padding_mask"),
                frame_count=stacked_control_batch.get("frame_count"),
                target_start_frame=stacked_control_batch.get("target_start_frame"),
            )
            control_memory_8s = _stacked_control_memory_8s(
                control_output,
                batch_size=batch_size,
                slice_count=slice_count,
            )

        control_memory_8s = _as_batched_float32_device_tensor(
            control_memory_8s,
            batch_size=batch_size,
            frames=MAPPER_DENSITY_FRAMES,
            channels=None,
            name="control_memory_8s",
            device=self.device,
        )

        cache = SessionControlBatchCache(
            session_id=self.session_id,
            start_ms_values=start_ms_tuple,
            target_start_frames=target_start_frames,
            normalized_difficulty=normalized_difficulty,
            control_slice_start_frames=control_slice_start_frames,
            target_start_frame_tensor=target_start_frame_tensor,
            normalized_difficulty_tensor=normalized_difficulty_tensor,
            control_memory_8s=control_memory_8s,
        )
        self.control_batch_cache = cache
        return cache

    def prepare_full_control(self, *, max_batch_size: int | None = None) -> SessionFullControlCache:
        if self.audio_cache is None:
            raise RuntimeError("prepare_audio must be called before prepare_full_control")
        effective_max_batch_size = _validate_max_batch_size(
            self.config.max_control_batch_size if max_batch_size is None else max_batch_size,
        )
        start_ms_values = tuple(
            frame * MAPPER_DENSITY_FRAME_MS
            for frame in range(0, self.audio_cache.source_frame_count, MAPPER_DENSITY_FRAMES)
        )
        if not start_ms_values:
            raise RuntimeError("audio cache must contain at least one source frame")

        batches: list[torch.Tensor] = []
        for start in range(0, len(start_ms_values), effective_max_batch_size):
            chunk = start_ms_values[start : start + effective_max_batch_size]
            batch_cache = self.prepare_control_batch(
                start_ms_values=chunk,
                max_batch_size=effective_max_batch_size,
            )
            batches.append(batch_cache.control_memory_8s)

        control_memory_8s = torch.cat(batches, dim=0).contiguous()
        target_start_frames = tuple(start_ms // MAPPER_DENSITY_FRAME_MS for start_ms in start_ms_values)
        cache = SessionFullControlCache(
            session_id=self.session_id,
            start_ms_values=start_ms_values,
            target_start_frames=target_start_frames,
            normalized_difficulty=float(self.config.default_normalized_difficulty),
            max_batch_size=effective_max_batch_size,
            control_memory_8s=control_memory_8s,
        )
        self.full_control_cache = cache
        return cache

    def _ensure_audio_cache_frame_count(self, required_frame_count: int) -> SessionAudioCache:
        if self.audio_cache is None:
            raise RuntimeError("prepare_audio must be called before prepare_control")
        if isinstance(required_frame_count, bool) or not isinstance(required_frame_count, int):
            raise TypeError("required_frame_count must be an integer")
        if required_frame_count <= 0:
            raise ValueError("required_frame_count must be positive")
        cache = self.audio_cache
        if required_frame_count <= cache.padded_frame_count:
            return cache

        padded_frame_count = int(required_frame_count)
        dense_timing_cpu = _as_2d_float32_cpu_tensor(
            render_dense_timing_v2(
                cache.timing_grid,
                input_start_ms=0.0,
                frame_count=padded_frame_count,
            ),
            channels=DENSE_TIMING_V2_CHANNELS,
            name="full_dense_timing_v2",
            source=cache.audio_path,
        )
        full_mel = cache.full_mel.new_zeros((1, padded_frame_count, PACKED_MEL_CHANNELS))
        full_mel[:, : cache.padded_frame_count].copy_(cache.full_mel)
        padding_mask = torch.ones((1, padded_frame_count), dtype=torch.bool, device=self.device)
        padding_mask[:, : cache.source_frame_count] = False

        expanded = SessionAudioCache(
            session_id=cache.session_id,
            audio_path=cache.audio_path,
            audio_length_ms=cache.audio_length_ms,
            audio_length_source=cache.audio_length_source,
            full_mel=full_mel.contiguous(),
            full_dense_timing_v2=dense_timing_cpu.unsqueeze(0).to(device=self.device, dtype=torch.float32),
            padding_mask=padding_mask,
            frame_count_tensor=torch.tensor([padded_frame_count], dtype=torch.long, device=self.device),
            source_frame_count_tensor=cache.source_frame_count_tensor,
            source_frame_count=cache.source_frame_count,
            padded_frame_count=padded_frame_count,
            beatthis_prediction=cache.beatthis_prediction,
            timing_fit_result=cache.timing_fit_result,
            timing_grid=cache.timing_grid,
        )
        self.audio_cache = expanded
        return expanded

    def reset_audio_cache(self) -> None:
        self.control_cache = None
        self.control_batch_cache = None
        self.full_control_cache = None
        self.audio_cache = None
        gc.collect()
        release_torch_cache(self.device)

    def reset_control_cache(self) -> None:
        self.control_cache = None
        self.control_batch_cache = None
        self.full_control_cache = None
        gc.collect()
        release_torch_cache(self.device)


def _resolve_session_device(model_runtime: ModelRuntime, requested: str | torch.device | None) -> torch.device:
    if requested is not None:
        return torch.device(requested)
    runtime_device = getattr(model_runtime, "device", None)
    if runtime_device is None:
        return torch.device("mps")
    return torch.device(runtime_device)


def _timing_provider(model_runtime: ModelRuntime) -> TimingProvider:
    provider = getattr(model_runtime, "beatthis_provider", None)
    if provider is None or not callable(getattr(provider, "predict_file", None)):
        raise TypeError("model_runtime must expose beatthis_provider.predict_file")
    return provider


def _control_model(model_runtime: ModelRuntime) -> torch.nn.Module:
    model = getattr(model_runtime, "control_model", None)
    if model is None or not callable(getattr(model, "forward", None)):
        raise TypeError("model_runtime must expose control_model")
    return model


def _as_2d_float32_cpu_tensor(
    value: Any,
    *,
    channels: int,
    name: str,
    source: object,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().to(device="cpu", dtype=torch.float32)
    else:
        array = np.asarray(value, dtype=np.float32)
        tensor = torch.from_numpy(np.ascontiguousarray(array))
    if tensor.ndim != 2 or int(tensor.shape[1]) != int(channels):
        raise ValueError(f"{name} for {source} must have shape [frames,{channels}], got {tuple(tensor.shape)}")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} for {source} must contain only finite values")
    return tensor.contiguous()


def _as_batched_float32_device_tensor(
    value: torch.Tensor,
    *,
    batch_size: int,
    frames: int,
    channels: int | None,
    name: str,
    device: torch.device,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name} must be a torch.Tensor")
    if value.ndim != 3:
        raise ValueError(f"{name} must have shape [B,F,C], got {tuple(value.shape)}")
    if int(value.shape[0]) != int(batch_size) or int(value.shape[1]) != int(frames):
        raise ValueError(f"{name} must have shape [{batch_size},{frames},C], got {tuple(value.shape)}")
    if channels is not None and int(value.shape[2]) != int(channels):
        raise ValueError(f"{name} must have {channels} channels, got {tuple(value.shape)}")
    tensor = value.detach().to(device=device, dtype=torch.float32).contiguous()
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must contain only finite values")
    return tensor


def _stacked_control_memory_8s(output: Any, *, batch_size: int, slice_count: int) -> torch.Tensor:
    memory = getattr(output, "control_memory", None)
    if not isinstance(memory, torch.Tensor) or memory.ndim != 3:
        raise ValueError("stacked control output control_memory must have shape [B*4,T,D]")
    expected_batch = int(batch_size) * int(slice_count)
    if int(memory.shape[0]) != expected_batch:
        raise ValueError(f"stacked control output batch must be {expected_batch}, got {memory.shape[0]}")
    start = TARGET_OFFSET_IN_CONTEXT
    end = start + TARGET_WINDOW_LENGTH_FRAMES
    if int(memory.shape[1]) < end:
        raise ValueError(f"stacked control output memory is too short for target slice: {memory.shape[1]} < {end}")
    return memory[:, start:end].reshape(
        int(batch_size),
        int(slice_count),
        TARGET_WINDOW_LENGTH_FRAMES,
        memory.shape[-1],
    ).reshape(
        int(batch_size),
        MAPPER_DENSITY_FRAMES,
        memory.shape[-1],
    ).contiguous()


def _resolve_audio_length_ms(
    *,
    audio_length_ms: int | None,
    source_frame_count: int,
) -> tuple[int, str]:
    if audio_length_ms is not None:
        if isinstance(audio_length_ms, bool) or not isinstance(audio_length_ms, int):
            raise TypeError("audio_length_ms must be an integer")
        if audio_length_ms <= 0:
            raise ValueError("audio_length_ms must be positive")
        return int(audio_length_ms), "provided"
    return int(source_frame_count * MAPPER_DENSITY_FRAME_MS), "mel_frame_estimate"


def _validate_start_ms(start_ms: int) -> int:
    if isinstance(start_ms, bool) or not isinstance(start_ms, int):
        raise TypeError("start_ms must be an integer")
    if start_ms < 0:
        raise ValueError("start_ms must be non-negative")
    if start_ms % MAPPER_DENSITY_FRAME_MS != 0:
        raise ValueError(f"start_ms must be divisible by {MAPPER_DENSITY_FRAME_MS}")
    return int(start_ms)


def _validate_start_ms_values(start_ms_values: Sequence[int]) -> tuple[int, ...]:
    if isinstance(start_ms_values, (str, bytes)):
        raise TypeError("start_ms_values must be a sequence of integers")
    values = tuple(_validate_start_ms(value) for value in start_ms_values)
    if not values:
        raise ValueError("start_ms_values must contain at least one value")
    return values


def _validate_max_batch_size(max_batch_size: int) -> int:
    if isinstance(max_batch_size, bool) or not isinstance(max_batch_size, int):
        raise TypeError("max_batch_size must be an integer")
    if max_batch_size <= 0:
        raise ValueError("max_batch_size must be positive")
    return int(max_batch_size)


__all__ = [
    "DEFAULT_MAX_CONTROL_BATCH_SIZE",
    "DENSE_TIMING_V2_CHANNELS",
    "PACKED_MEL_CHANNELS",
    "SessionAudioCache",
    "SessionControlBatchCache",
    "SessionControlCache",
    "SessionFullControlCache",
    "SessionRuntime",
    "SessionRuntimeConfig",
]
