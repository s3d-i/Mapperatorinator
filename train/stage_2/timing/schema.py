from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class FrameTimingPrediction:
    provider: str
    beat_prob: NDArray[np.float32]
    downbeat_prob: NDArray[np.float32]
    frame_rate_hz: float
    checkpoint_path: str | None = None
    source_path: str | None = None

    def __post_init__(self) -> None:
        beat_prob = _as_probability_vector(self.beat_prob, "beat_prob")
        downbeat_prob = _as_probability_vector(self.downbeat_prob, "downbeat_prob")

        if beat_prob.shape != downbeat_prob.shape:
            raise ValueError("beat_prob and downbeat_prob must have the same length")
        if not np.isfinite(self.frame_rate_hz) or self.frame_rate_hz <= 0.0:
            raise ValueError(f"frame_rate_hz must be a positive finite value, got {self.frame_rate_hz!r}")
        if not self.provider:
            raise ValueError("provider must be non-empty")

        object.__setattr__(self, "beat_prob", beat_prob)
        object.__setattr__(self, "downbeat_prob", downbeat_prob)
        if self.source_path is not None:
            object.__setattr__(self, "source_path", Path(self.source_path).as_posix())

    @property
    def frame_count(self) -> int:
        return int(self.beat_prob.shape[0])

    @property
    def frame_times_seconds(self) -> NDArray[np.float32]:
        frame_indexes = np.arange(self.frame_count, dtype=np.float32)
        return frame_indexes / np.float32(self.frame_rate_hz)


def _as_probability_vector(value: object, name: str) -> NDArray[np.float32]:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1-D vector")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any(array < 0.0) or np.any(array > 1.0):
        raise ValueError(f"{name} must contain probabilities in [0, 1]")
    return array
