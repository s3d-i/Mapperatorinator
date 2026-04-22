from __future__ import annotations

import math
from dataclasses import fields
from pathlib import Path
from typing import Any, Mapping

import torch

from ..events.grammar import constrained_greedy_decode
from ..events.tokens import Stage1Vocab
from ..events.windowing import WindowSpec
from ..features.timing import TIMING_TRACK_VERSION
from ..models.mapper import Stage1OracleMapper, Stage1OracleMapperConfig
from ..training.overfit_32 import select_torch_device
from .feature_source import WindowFeatures
from .scheduler import DecodedTokenWindow


class Stage1CheckpointRunner:
    def __init__(
        self,
        *,
        model: Stage1OracleMapper,
        config: Stage1OracleMapperConfig,
        checkpoint_path: Path,
        device: torch.device,
        bpm_log_mean: float,
        bpm_log_std: float,
        vocab: Stage1Vocab | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.checkpoint_name = checkpoint_path.name
        self.device = device
        self.bpm_log_mean = bpm_log_mean
        self.bpm_log_std = bpm_log_std
        self.vocab = Stage1Vocab() if vocab is None else vocab

    @classmethod
    def load(
        cls,
        checkpoint_path: str | Path,
        *,
        device_name: str = "auto",
        vocab: Stage1Vocab | None = None,
    ) -> "Stage1CheckpointRunner":
        path = Path(checkpoint_path)
        checkpoint = cls.inspect_checkpoint(path)
        config = _checkpoint_config(checkpoint)
        timing_track = _checkpoint_timing_track_metadata(checkpoint)
        resolved_vocab = Stage1Vocab() if vocab is None else vocab
        if config.vocab_size != resolved_vocab.size:
            raise ValueError(f"checkpoint vocab_size {config.vocab_size} does not match Stage1Vocab size {resolved_vocab.size}")

        device = select_torch_device(device_name)
        model = Stage1OracleMapper(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        return cls(
            model=model,
            config=config,
            checkpoint_path=path,
            device=device,
            bpm_log_mean=float(timing_track["bpm_log_mean"]),
            bpm_log_std=float(timing_track["bpm_log_std"]),
            vocab=resolved_vocab,
        )

    @staticmethod
    def inspect_checkpoint(checkpoint_path: str | Path) -> dict[str, Any]:
        checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict):
            raise ValueError(f"checkpoint must contain a mapping: {checkpoint_path}")
        if "config" not in checkpoint:
            raise ValueError(f"checkpoint missing config: {checkpoint_path}")
        if "model_state_dict" not in checkpoint:
            raise ValueError(f"checkpoint missing model_state_dict: {checkpoint_path}")
        if not isinstance(checkpoint["model_state_dict"], Mapping):
            raise ValueError("checkpoint model_state_dict must be a mapping")
        _checkpoint_config(checkpoint)
        _checkpoint_timing_track_metadata(checkpoint)
        return checkpoint

    @torch.no_grad()
    def decode_window(
        self,
        *,
        features: WindowFeatures,
        window: WindowSpec,
        difficulty: float,
        open_hold_mask: int,
    ) -> DecodedTokenWindow:
        self.model.eval()
        condition_ids = [
            self.vocab.bos_id,
            self.vocab.diff_token_id(self.vocab.difficulty_bucket_id(difficulty)),
            self.vocab.open_token_id(open_hold_mask),
        ]
        packed_audio = features.packed_audio.to(self.device, dtype=torch.float32).unsqueeze(0)
        timing_track = features.timing_track.to(self.device, dtype=torch.float32).unsqueeze(0)
        difficulty_bucket = torch.tensor(
            [self.vocab.difficulty_bucket_id(difficulty)],
            dtype=torch.long,
            device=self.device,
        )
        result = constrained_greedy_decode(
            self.model,
            packed_audio=packed_audio,
            timing_track=timing_track,
            difficulty_bucket=difficulty_bucket,
            condition_ids=condition_ids,
            open_hold_mask=open_hold_mask,
            write_duration_ms=window.write_duration_ms,
            vocab=self.vocab,
            max_decode_len=self.config.max_decode_len,
        )
        return DecodedTokenWindow(
            write_start_ms=window.write_start_ms,
            write_end_ms=window.write_end_ms,
            token_ids=result.token_ids[len(condition_ids) :],
            max_decode_len_reached=result.max_decode_len_reached,
            eos_emitted_by_model=result.eos_emitted_by_model,
            eos_forced_after_pending_ts=result.eos_forced_after_pending_ts,
        )


def _checkpoint_config(checkpoint: Mapping[str, Any]) -> Stage1OracleMapperConfig:
    raw_config = checkpoint.get("config")
    if not isinstance(raw_config, Mapping):
        raise ValueError("checkpoint config must be a mapping")
    allowed_keys = {field.name for field in fields(Stage1OracleMapperConfig)}
    unknown_keys = sorted(set(raw_config) - allowed_keys)
    if unknown_keys:
        raise ValueError(f"checkpoint config contains unknown keys: {unknown_keys}")
    return Stage1OracleMapperConfig(**dict(raw_config))


def _checkpoint_timing_track_metadata(checkpoint: Mapping[str, Any]) -> Mapping[str, Any]:
    timing_track = checkpoint.get("timing_track")
    if not isinstance(timing_track, Mapping):
        raise ValueError("checkpoint missing timing_track metadata")
    version = timing_track.get("timing_track_version")
    if version != TIMING_TRACK_VERSION:
        raise ValueError(f"checkpoint timing_track_version must be {TIMING_TRACK_VERSION}: {version}")
    _finite_timing_float(timing_track, "bpm_log_mean")
    bpm_log_std = _finite_timing_float(timing_track, "bpm_log_std")
    if bpm_log_std <= 0:
        raise ValueError("checkpoint timing_track bpm_log_std must be positive")
    return timing_track


def _finite_timing_float(timing_track: Mapping[str, Any], key: str) -> float:
    value = timing_track.get(key)
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"checkpoint timing_track {key} must be numeric") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"checkpoint timing_track {key} must be finite")
    return parsed
