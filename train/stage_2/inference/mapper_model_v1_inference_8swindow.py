from __future__ import annotations

import argparse
import math
import os
import pickle
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from train.stage_2.data.control_windows import normalize_difficulty
from train.stage_2.features.mel import load_full_song_packed_mel_20ms
from train.stage_2.model_control_demo_global import ControlDemoGlobalEncoder, ControlDemoGlobalEncoderConfig
from train.stage_2.model_mapper_v1.generation import (
    MapperGeneratedWindow,
    MapperGenerationStep,
    grammar_constrained_window_generation,
    transition_carry_state,
)
from train.stage_2.model_mapper_v1.model import MapperV1Config, MapperV1Model, compute_control_teacher_8s
from train.stage_2.model_mapper_v1.replay import LNCarryState, empty_ln_carry_state, ln_carry_state_tensors
from train.stage_2.model_mapper_v1.tokenizer import MAPPER_DENSITY_FRAMES, MAPPER_WRITE_MS
from train.stage_2.model_mapper_v1.vocab import MapperV1Vocab
from train.stage_2.timing.grid_fitting import GridFitter, GridFitterConfig
from train.stage_2.timing.providers.beatthis import (
    DEFAULT_BEATTHIS_CHECKPOINT,
    DEFAULT_BEATTHIS_DEVICE,
    BeatThisTimingProvider,
)
from train.stage_2.timing.rendering.dense_timing_v2 import render_dense_timing_v2
from train.stage_2.timing.schema import FittedTimingGrid, TimingSegment

from .osu_stream import OsuStreamMetadata, decode_mapper_tokens_to_timepoints, format_osu_stream


DEFAULT_CHECKPOINT_PATH = Path(
    "train/artifacts/runs/stage2_mapper_v1/"
    "stage2_mapper_v1_phase_b_cached_demo_d384_l4_b4_init1000_memfix/"
    "checkpoints/checkpoint_step_000750.pt"
)
DEFAULT_MAX_TOKENS = 512


@dataclass(frozen=True)
class TimingConfig:
    mode: str = "constant"
    bpm: float = 180.0
    offset_ms: float = 0.0
    meter: int = 4
    beatthis_checkpoint: str = DEFAULT_BEATTHIS_CHECKPOINT
    beatthis_device: str = DEFAULT_BEATTHIS_DEVICE
    beatthis_float16: bool = False
    fitter_min_bpm: float | None = None
    fitter_max_bpm: float | None = None
    fitter_max_segments: int | None = None


@dataclass(frozen=True)
class RunMetadata:
    title: str | None = None
    artist: str = "Unknown"
    creator: str = "Mapperatorinator"
    version: str | None = None


@dataclass(frozen=True)
class InferenceRunConfig:
    checkpoint_path: Path
    audio_path: Path
    difficulty: float
    output_path: Path | None = None
    device: str = "auto"
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = 0.0
    top_p: float | None = None
    seed: int | None = None
    timing: TimingConfig = TimingConfig()
    metadata: RunMetadata = RunMetadata()


@dataclass(frozen=True)
class LoadedMapperModel:
    model: MapperV1Model
    vocab: MapperV1Vocab
    model_config: MapperV1Config
    control_model_config: ControlDemoGlobalEncoderConfig | None
    checkpoint_step: int | None
    checkpoint_path: Path


@dataclass(frozen=True)
class PreparedFeatureBatch:
    full_mel: torch.Tensor
    full_dense_timing_v2: torch.Tensor
    padding_mask: torch.Tensor
    frame_count: torch.Tensor
    control_slice_start_frames: torch.Tensor
    timing_grid: FittedTimingGrid
    source_frame_count: int
    inference_frame_count: int


@dataclass(frozen=True)
class InferenceRunResult:
    osu_text: str
    generated_window: MapperGeneratedWindow
    timing_grid: FittedTimingGrid
    source_frame_count: int
    inference_frame_count: int
    checkpoint_step: int | None


def load_run_config(config_path: str | Path) -> InferenceRunConfig:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, Mapping):
        raise ValueError(f"inference config must contain a mapping: {path}")

    timing = _timing_config(raw.get("timing", {}))
    metadata = _metadata_config(raw.get("metadata", {}))
    checkpoint_path = Path(_required(raw, "checkpoint_path"))
    audio_path = Path(_required(raw, "audio_path"))
    output_value = raw.get("output_path")
    output_path = None if output_value in (None, "", "stdout") else Path(str(output_value))

    return InferenceRunConfig(
        checkpoint_path=checkpoint_path,
        audio_path=audio_path,
        difficulty=_finite_float(_required(raw, "difficulty"), name="difficulty"),
        output_path=output_path,
        device=str(raw.get("device", "auto")),
        max_tokens=_positive_int(raw.get("max_tokens", DEFAULT_MAX_TOKENS), name="max_tokens"),
        temperature=_nonnegative_float(raw.get("temperature", 0.0), name="temperature"),
        top_p=_optional_probability(raw.get("top_p")),
        seed=None if raw.get("seed") is None else int(raw["seed"]),
        timing=timing,
        metadata=metadata,
    )


def run_inference(config: InferenceRunConfig) -> InferenceRunResult:
    device = select_inference_device(config.device)
    loaded = load_mapper_model(config.checkpoint_path, device=device)
    features = prepare_feature_batch(
        audio_path=config.audio_path,
        difficulty=config.difficulty,
        timing_config=config.timing,
        device=device,
    )
    vocab = loaded.vocab
    ln_carry_in = empty_ln_carry_state(0)
    ln_carry_out = empty_ln_carry_state(MAPPER_WRITE_MS)
    generator = _make_generation_generator(config.seed, device=device)

    with torch.inference_mode():
        if loaded.model.control_encoder is None:
            raise ValueError("mapper inference requires a checkpoint with an embedded control encoder")
        control_batch = _control_teacher_batch(features, difficulty=config.difficulty, vocab=vocab, device=device)
        control_memory_8s, density_teacher_8s = compute_control_teacher_8s(
            loaded.model.control_encoder,
            control_batch,
            stack_slices=True,
        )

        logits_fn = _mapper_logits_fn(
            model=loaded.model,
            vocab=vocab,
            difficulty=config.difficulty,
            device=device,
            control_memory_8s=control_memory_8s,
            density_teacher_8s=density_teacher_8s,
            ln_carry_in=ln_carry_in,
            ln_carry_out=ln_carry_out,
        )
        generated = grammar_constrained_window_generation(
            vocab=vocab,
            write_start_ms=0,
            write_end_ms=MAPPER_WRITE_MS,
            ln_carry_in=ln_carry_in,
            ln_carry_out=ln_carry_out,
            logits_fn=logits_fn,
            is_full_chart_start=True,
            is_full_chart_end=True,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            generator=generator,
        )
    _ensure_completed_generation(generated)

    timepoints = decode_mapper_tokens_to_timepoints(
        generated.tokens,
        vocab,
        start_ms=0,
        end_ms=MAPPER_WRITE_MS,
    )
    metadata = OsuStreamMetadata(
        audio_filename=_audio_filename_for_output(config.audio_path, config.output_path),
        title=config.metadata.title or config.audio_path.stem,
        artist=config.metadata.artist,
        creator=config.metadata.creator,
        version=config.metadata.version or f"Stage2 mapper v1 8s diff {config.difficulty:.2f}",
        difficulty=config.difficulty,
    )
    osu_text = format_osu_stream(
        timepoints=timepoints,
        timing_grid=features.timing_grid,
        metadata=metadata,
        generated_tokens=generated.tokens,
        vocab=vocab,
    )
    return InferenceRunResult(
        osu_text=osu_text,
        generated_window=generated,
        timing_grid=features.timing_grid,
        source_frame_count=features.source_frame_count,
        inference_frame_count=features.inference_frame_count,
        checkpoint_step=loaded.checkpoint_step,
    )


def load_mapper_model(checkpoint_path: str | Path, *, device: torch.device) -> LoadedMapperModel:
    path = Path(checkpoint_path)
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError as exc:
        raise ValueError(
            "mapper checkpoint could not be loaded safely with weights_only=True; "
            "use a checkpoint written by the mapper trainer",
        ) from exc
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"mapper checkpoint must contain a mapping: {path}")
    raw_model_config = checkpoint.get("model_config")
    if not isinstance(raw_model_config, Mapping):
        raise ValueError("mapper checkpoint missing model_config mapping")
    model_config = MapperV1Config(**dict(raw_model_config))

    raw_control_config = checkpoint.get("control_model_config")
    control_config = None
    control_encoder = None
    if raw_control_config is not None:
        if not isinstance(raw_control_config, Mapping):
            raise ValueError("mapper checkpoint control_model_config must be a mapping or null")
        control_config = ControlDemoGlobalEncoderConfig(**dict(raw_control_config))
        control_encoder = ControlDemoGlobalEncoder(control_config)

    model = MapperV1Model(model_config, control_encoder=control_encoder)
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("mapper checkpoint missing model_state_dict")
    non_tensor_keys = [str(key) for key, value in state.items() if not isinstance(value, torch.Tensor)]
    if non_tensor_keys:
        raise ValueError(f"mapper checkpoint model_state_dict contains non-tensor values: {non_tensor_keys}")
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    training_state = checkpoint.get("training_state")
    checkpoint_step = None
    if isinstance(training_state, Mapping) and isinstance(training_state.get("step"), int):
        checkpoint_step = int(training_state["step"])
    return LoadedMapperModel(
        model=model,
        vocab=model.vocab,
        model_config=model_config,
        control_model_config=control_config,
        checkpoint_step=checkpoint_step,
        checkpoint_path=path,
    )


def prepare_feature_batch(
    *,
    audio_path: Path,
    difficulty: float,
    timing_config: TimingConfig,
    device: torch.device,
) -> PreparedFeatureBatch:
    normalize_difficulty(difficulty)
    packed_mel = load_full_song_packed_mel_20ms(audio_path)
    packed_mel = _as_2d_float32(packed_mel, channels=160, name="packed_mel")
    source_frame_count = int(packed_mel.shape[0])
    if source_frame_count <= 0:
        raise ValueError(
            "mapper v1 8s inference requires at least one packed 20ms frame, "
            f"got {source_frame_count}",
        )
    inference_frame_count = max(source_frame_count, MAPPER_DENSITY_FRAMES)
    full_mel = np.zeros((inference_frame_count, packed_mel.shape[1]), dtype=np.float32)
    full_mel[:source_frame_count] = packed_mel

    timing_grid = build_timing_grid(audio_path=audio_path, timing_config=timing_config)
    dense_timing = render_dense_timing_v2(
        timing_grid,
        input_start_ms=0.0,
        frame_count=inference_frame_count,
    )
    dense_timing = _as_2d_float32(dense_timing, channels=4, name="dense_timing_v2")

    padding_mask = torch.zeros((1, inference_frame_count), dtype=torch.bool, device=device)
    padding_mask[:, source_frame_count:] = True

    return PreparedFeatureBatch(
        full_mel=torch.as_tensor(full_mel, dtype=torch.float32, device=device).unsqueeze(0),
        full_dense_timing_v2=torch.as_tensor(dense_timing, dtype=torch.float32, device=device).unsqueeze(0),
        padding_mask=padding_mask,
        frame_count=torch.tensor([inference_frame_count], dtype=torch.long, device=device),
        control_slice_start_frames=torch.tensor([[0, 100, 200, 300]], dtype=torch.long, device=device),
        timing_grid=timing_grid,
        source_frame_count=source_frame_count,
        inference_frame_count=inference_frame_count,
    )


def build_timing_grid(*, audio_path: Path, timing_config: TimingConfig) -> FittedTimingGrid:
    mode = timing_config.mode.strip().lower()
    if mode == "constant":
        return _constant_timing_grid(timing_config)
    if mode == "beatthis":
        provider = BeatThisTimingProvider(
            checkpoint_path=timing_config.beatthis_checkpoint,
            device=timing_config.beatthis_device,
            float16=timing_config.beatthis_float16,
        )
        prediction = provider.predict_file(audio_path)
        fit_config = GridFitterConfig(
            min_bpm=GridFitterConfig().min_bpm if timing_config.fitter_min_bpm is None else timing_config.fitter_min_bpm,
            max_bpm=GridFitterConfig().max_bpm if timing_config.fitter_max_bpm is None else timing_config.fitter_max_bpm,
            max_segments=(
                GridFitterConfig().max_segments
                if timing_config.fitter_max_segments is None
                else timing_config.fitter_max_segments
            ),
        )
        return GridFitter(fit_config).fit(prediction).grid
    raise ValueError(f"unsupported timing.mode {timing_config.mode!r}; expected constant or beatthis")


def select_inference_device(value: str | torch.device) -> torch.device:
    requested = str(value)
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def write_or_stream_output(text: str, output_path: Path | None) -> None:
    if output_path is None:
        for line in text.splitlines():
            print(line, flush=True)
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


def _ensure_completed_generation(generated: MapperGeneratedWindow) -> None:
    if generated.completed and not generated.dead_end and not generated.max_tokens_exceeded:
        return
    raise RuntimeError(
        "mapper generation did not complete the exact 8s window: "
        f"completed={int(generated.completed)} "
        f"dead_end={int(generated.dead_end)} "
        f"max_tokens_exceeded={int(generated.max_tokens_exceeded)} "
        f"tokens={len(generated.tokens)} "
        f"terminal_ms={generated.terminal_state.current_ms}",
    )


def _make_generation_generator(seed: int | None, *, device: torch.device) -> torch.Generator | None:
    if seed is None:
        return None
    try:
        generator = torch.Generator(device=device)
    except RuntimeError as exc:
        raise RuntimeError(f"could not create torch generator on inference device {device}") from exc
    generator.manual_seed(int(seed))
    return generator


def _audio_filename_for_output(audio_path: Path, output_path: Path | None) -> str:
    if output_path is None:
        return audio_path.name
    try:
        relative = os.path.relpath(
            audio_path.resolve(strict=False),
            start=output_path.parent.resolve(strict=False),
        )
    except ValueError:
        return audio_path.name
    return Path(relative).as_posix()


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    config = load_run_config(args.config)
    config = _apply_cli_overrides(config, args)
    result = run_inference(config)
    write_or_stream_output(result.osu_text, config.output_path)
    _print_run_report(config, result)
    return 0


def _mapper_logits_fn(
    *,
    model: MapperV1Model,
    vocab: MapperV1Vocab,
    difficulty: float,
    device: torch.device,
    control_memory_8s: torch.Tensor,
    density_teacher_8s: torch.Tensor,
    ln_carry_in: LNCarryState,
    ln_carry_out: LNCarryState,
):
    normalized = normalize_difficulty(difficulty)

    def logits_fn(step: MapperGenerationStep) -> torch.Tensor:
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
        current_ms = states["current_ms"]
        target_tokens = torch.full_like(decoder_input_tokens, vocab.pad_id)
        at_write_end = current_ms == int(step.write_end_ms)
        target_tokens = torch.where(at_write_end, torch.full_like(target_tokens, vocab.eos_id), target_tokens)
        batch = {
            "decoder_input_tokens": decoder_input_tokens,
            "target_fragment_tokens": target_tokens,
            "target_fragment_mask": torch.ones_like(decoder_input_tokens, dtype=torch.bool),
            "target_fragment_states": states,
            "ln_carry_in": _carry_state_batch(ln_carry_in, device=device),
            "ln_carry_out": _carry_state_batch(ln_carry_out, device=device),
            "write_start_ms": torch.tensor([step.write_start_ms], dtype=torch.long, device=device),
            "write_end_ms": torch.tensor([step.write_end_ms], dtype=torch.long, device=device),
            "is_full_chart_start": torch.tensor([True], dtype=torch.bool, device=device),
            "is_full_chart_end": torch.tensor([True], dtype=torch.bool, device=device),
            "difficulty": torch.tensor([[difficulty]], dtype=torch.float32, device=device),
            "normalized_difficulty": torch.tensor([normalized], dtype=torch.float32, device=device),
        }
        output = model(
            batch,
            control_memory_8s=control_memory_8s,
            density_teacher_8s=density_teacher_8s,
        )
        return output.logits_final[0, -1].detach()

    return logits_fn


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


def _control_teacher_batch(
    features: PreparedFeatureBatch,
    *,
    difficulty: float,
    vocab: MapperV1Vocab,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    normalized = normalize_difficulty(difficulty)
    return {
        "decoder_input_tokens": torch.tensor([[vocab.bos_id]], dtype=torch.long, device=device),
        "full_mel": features.full_mel,
        "full_dense_timing_v2": features.full_dense_timing_v2,
        "padding_mask": features.padding_mask,
        "frame_count": features.frame_count,
        "control_slice_start_frames": features.control_slice_start_frames,
        "difficulty": torch.tensor([[difficulty]], dtype=torch.float32, device=device),
        "normalized_difficulty": torch.tensor([normalized], dtype=torch.float32, device=device),
    }


def _carry_state_batch(carry: LNCarryState, *, device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.unsqueeze(0) for key, value in _carry_state_tensors_1d(carry, device=device).items()}


def _carry_state_tensors_1d(carry: LNCarryState, *, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device=device)
        for key, value in ln_carry_state_tensors(carry).items()
    }


def _constant_timing_grid(config: TimingConfig) -> FittedTimingGrid:
    bpm = _positive_float(config.bpm, name="timing.bpm")
    offset_ms = _finite_float(config.offset_ms, name="timing.offset_ms")
    meter = _positive_int(config.meter, name="timing.meter")
    return FittedTimingGrid((TimingSegment(offset_ms=offset_ms, beat_length_ms=60000.0 / bpm, meter=meter),))


def _timing_config(raw: object) -> TimingConfig:
    if raw is None:
        return TimingConfig()
    if not isinstance(raw, Mapping):
        raise ValueError("timing config must be a mapping")
    return TimingConfig(
        mode=str(raw.get("mode", "constant")),
        bpm=_positive_float(raw.get("bpm", 180.0), name="timing.bpm"),
        offset_ms=_finite_float(raw.get("offset_ms", 0.0), name="timing.offset_ms"),
        meter=_positive_int(raw.get("meter", 4), name="timing.meter"),
        beatthis_checkpoint=str(raw.get("beatthis_checkpoint", DEFAULT_BEATTHIS_CHECKPOINT)),
        beatthis_device=str(raw.get("beatthis_device", DEFAULT_BEATTHIS_DEVICE)),
        beatthis_float16=bool(raw.get("beatthis_float16", False)),
        fitter_min_bpm=None if raw.get("fitter_min_bpm") is None else _positive_float(raw["fitter_min_bpm"], name="timing.fitter_min_bpm"),
        fitter_max_bpm=None if raw.get("fitter_max_bpm") is None else _positive_float(raw["fitter_max_bpm"], name="timing.fitter_max_bpm"),
        fitter_max_segments=None if raw.get("fitter_max_segments") is None else _positive_int(raw["fitter_max_segments"], name="timing.fitter_max_segments"),
    )


def _metadata_config(raw: object) -> RunMetadata:
    if raw is None:
        return RunMetadata()
    if not isinstance(raw, Mapping):
        raise ValueError("metadata config must be a mapping")
    return RunMetadata(
        title=None if raw.get("title") is None else str(raw["title"]),
        artist=str(raw.get("artist", "Unknown")),
        creator=str(raw.get("creator", "Mapperatorinator")),
        version=None if raw.get("version") is None else str(raw["version"]),
    )


def _apply_cli_overrides(config: InferenceRunConfig, args: argparse.Namespace) -> InferenceRunConfig:
    output_path = config.output_path
    if args.output is not None:
        output_path = None if args.output == "stdout" else Path(args.output)
    return InferenceRunConfig(
        checkpoint_path=Path(args.checkpoint) if args.checkpoint is not None else config.checkpoint_path,
        audio_path=Path(args.audio) if args.audio is not None else config.audio_path,
        difficulty=float(args.difficulty) if args.difficulty is not None else config.difficulty,
        output_path=output_path,
        device=args.device if args.device is not None else config.device,
        max_tokens=int(args.max_tokens) if args.max_tokens is not None else config.max_tokens,
        temperature=float(args.temperature) if args.temperature is not None else config.temperature,
        top_p=float(args.top_p) if args.top_p is not None else config.top_p,
        seed=int(args.seed) if args.seed is not None else config.seed,
        timing=config.timing,
        metadata=config.metadata,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run mapper v1 inference for the first exact 8s window.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--difficulty", type=float)
    parser.add_argument("--output", help="Output .osu path, or 'stdout'.")
    parser.add_argument("--device")
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--seed", type=int)
    return parser


def _print_run_report(config: InferenceRunConfig, result: InferenceRunResult) -> None:
    generated = result.generated_window
    print(
        "mapper_model_v1_inference_8swindow "
        f"checkpoint_step={result.checkpoint_step} "
        f"audio={config.audio_path.as_posix()} "
        f"difficulty={config.difficulty:.3f} "
        f"source_frames={result.source_frame_count} "
        f"inference_frames={result.inference_frame_count} "
        f"tokens={len(generated.tokens)} "
        f"completed={int(generated.completed)} "
        f"dead_end={int(generated.dead_end)} "
        f"max_tokens_exceeded={int(generated.max_tokens_exceeded)}",
        file=sys.stderr,
        flush=True,
    )


def _as_2d_float32(value: object, *, channels: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2 or int(array.shape[1]) != int(channels):
        raise ValueError(f"{name} must have shape [frames,{channels}], got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(array)


def _required(raw: Mapping[str, Any], key: str) -> Any:
    value = raw.get(key)
    if value is None:
        raise ValueError(f"inference config missing required field {key!r}")
    return value


def _optional_probability(value: object) -> float | None:
    if value is None:
        return None
    probability = _positive_float(value, name="top_p")
    if probability > 1.0:
        raise ValueError(f"top_p must be <= 1.0, got {probability}")
    return probability


def _finite_float(value: object, *, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return number


def _nonnegative_float(value: object, *, name: str) -> float:
    number = _finite_float(value, name=name)
    if number < 0.0:
        raise ValueError(f"{name} must be non-negative, got {number}")
    return number


def _positive_float(value: object, *, name: str) -> float:
    number = _finite_float(value, name=name)
    if number <= 0.0:
        raise ValueError(f"{name} must be positive, got {number}")
    return number


def _positive_int(value: object, *, name: str) -> int:
    number = int(value)
    if number <= 0:
        raise ValueError(f"{name} must be positive, got {number}")
    return number


if __name__ == "__main__":
    raise SystemExit(main())
