from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch

from ..events.windowing import WindowSpec, compute_generation_end_ms, iter_window_specs
from ..features.audio import load_audio_file
from ..features.mel import DEFAULT_MEL_CACHE_CONFIG, MelCacheConfig
from ..features.mel import load_or_create_log_mel_cache, pack_mel_20ms_window
from ..features.timing import DEFAULT_TIMING_TRACK_CONFIG, render_timing_track_20ms_v1
from ..osu.timing import require_red_timing_points


@dataclass(frozen=True)
class WindowFeatures:
    packed_audio: torch.Tensor
    timing_track: torch.Tensor


class FullSongFeatureSource:
    def __init__(
        self,
        *,
        audio_path: str | Path,
        beatmap_path: str | Path,
        bpm_log_mean: float,
        bpm_log_std: float,
        sample_rate: int = 16000,
        mel_config: MelCacheConfig = DEFAULT_MEL_CACHE_CONFIG,
    ) -> None:
        if not math.isfinite(bpm_log_mean):
            raise ValueError(f"bpm_log_mean must be finite: {bpm_log_mean}")
        if not math.isfinite(bpm_log_std) or bpm_log_std <= 0:
            raise ValueError(f"bpm_log_std must be positive: {bpm_log_std}")
        self.audio_path = Path(audio_path)
        self.beatmap_path = Path(beatmap_path)
        self.bpm_log_mean = float(bpm_log_mean)
        self.bpm_log_std = float(bpm_log_std)
        self.sample_rate = sample_rate
        self.mel_config = mel_config
        self.timing_points = require_red_timing_points(self.beatmap_path)
        self.audio = load_audio_file(self.audio_path, sample_rate=sample_rate)
        self.audio_duration_ms = _audio_duration_ms(self.audio.shape[0], sample_rate)
        self.generation_end_ms = compute_generation_end_ms(self.audio_duration_ms, [])
        self.mel = load_or_create_log_mel_cache(
            self.audio,
            sample_rate=sample_rate,
            audio_cache_key=self.audio_path.as_posix(),
            config=mel_config,
        )

    def iter_windows(self) -> list[WindowSpec]:
        return iter_window_specs(self.generation_end_ms)

    def features_for_window(self, window: WindowSpec) -> WindowFeatures:
        packed_audio = pack_mel_20ms_window(self.mel, input_start_ms=window.input_start_ms)
        timing_track = render_timing_track_20ms_v1(
            self.timing_points,
            input_start_ms=window.input_start_ms,
            bpm_log_mean=self.bpm_log_mean,
            bpm_log_std=self.bpm_log_std,
            frame_count=DEFAULT_TIMING_TRACK_CONFIG.frame_count,
        )
        return WindowFeatures(
            packed_audio=torch.from_numpy(packed_audio),
            timing_track=torch.from_numpy(timing_track),
        )


def _audio_duration_ms(sample_count: int, sample_rate: int) -> float:
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive: {sample_rate}")
    if sample_count < 0:
        raise ValueError(f"sample_count must be non-negative: {sample_count}")
    return sample_count * 1000.0 / sample_rate if sample_count else 0.0
