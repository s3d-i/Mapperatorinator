from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from ..core.difficulty import calculate_mania_difficulty
from ..events.tokens import Stage1Vocab
from ..osu.metadata import parse_osu_metadata
from .assembler import StreamingAssembler
from .checkpoint_runner import Stage1CheckpointRunner
from .feature_source import FullSongFeatureSource
from .preview_server import PreviewEvent, build_preview_event_stream


DEFAULT_RUNS_ROOT = Path("train/artifacts/runs/stage1_oracle")
DEFAULT_INDEX_PATH = Path("train/artifacts/indexes/beatmap_index_4k_no_timing_anomalies.parquet")
DEFAULT_DATASET_ROOT = Path("mania-dataset")
DEFAULT_TARGET_DIFFICULTY = 4.5
DEFAULT_MAX_WINDOWS = 2


@dataclass(frozen=True)
class StreamProbeInputs:
    checkpoint_path: Path
    audio_path: Path
    beatmap_path: Path
    difficulty: float


@dataclass(frozen=True)
class _DefaultBeatmapInput:
    audio_path: Path
    beatmap_path: Path
    difficulty: float


class _WindowLimitedSource:
    def __init__(self, source: FullSongFeatureSource, max_windows: int) -> None:
        if max_windows <= 0:
            raise ValueError(f"max_windows must be positive: {max_windows}")
        self._source = source
        self.max_windows = int(max_windows)

    @property
    def audio_duration_ms(self) -> float:
        return self._source.audio_duration_ms

    def iter_windows(self):
        return self._source.iter_windows()[: self.max_windows]

    def features_for_window(self, window):
        return self._source.features_for_window(window)


def find_latest_checkpoint(runs_root: str | Path = DEFAULT_RUNS_ROOT) -> Path:
    root = Path(runs_root)
    checkpoints = [path for path in root.glob("**/checkpoint.pt") if path.is_file()]
    if not checkpoints:
        raise FileNotFoundError(f"no Stage 1 checkpoint.pt found under {root}")
    return max(checkpoints, key=lambda path: (path.stat().st_mtime_ns, path.as_posix()))


def resolve_probe_inputs(
    *,
    checkpoint_path: str | Path | None,
    audio_path: str | Path | None,
    beatmap_path: str | Path | None,
    difficulty: float | None,
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    index_path: str | Path = DEFAULT_INDEX_PATH,
    runs_root: str | Path = DEFAULT_RUNS_ROOT,
    target_difficulty: float = DEFAULT_TARGET_DIFFICULTY,
) -> StreamProbeInputs:
    resolved_checkpoint = Path(checkpoint_path) if checkpoint_path is not None else find_latest_checkpoint(runs_root)
    if not resolved_checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint file not found: {resolved_checkpoint}")

    resolved_audio = Path(audio_path) if audio_path is not None else None
    resolved_beatmap = Path(beatmap_path) if beatmap_path is not None else None
    resolved_difficulty = float(difficulty) if difficulty is not None else None

    if resolved_beatmap is None:
        if resolved_audio is not None:
            raise ValueError("--beatmap-path is required when --audio-path is supplied")
        default = _select_default_index_row(
            dataset_root=Path(dataset_root),
            index_path=Path(index_path),
            target_difficulty=target_difficulty,
        )
        resolved_audio = default.audio_path
        resolved_beatmap = default.beatmap_path
        if resolved_difficulty is None:
            resolved_difficulty = default.difficulty
    elif resolved_audio is None:
        resolved_audio = _audio_path_from_beatmap(resolved_beatmap)

    if resolved_audio is None:
        raise ValueError("audio path could not be resolved")
    if not resolved_audio.is_file():
        raise FileNotFoundError(f"audio file not found: {resolved_audio}")
    if not resolved_beatmap.is_file():
        raise FileNotFoundError(f"beatmap file not found: {resolved_beatmap}")

    if resolved_difficulty is None:
        resolved_difficulty = calculate_mania_difficulty(resolved_beatmap, resolved_audio, speed=1.0)
    Stage1Vocab().difficulty_bucket_id(resolved_difficulty)

    return StreamProbeInputs(
        checkpoint_path=resolved_checkpoint,
        audio_path=resolved_audio,
        beatmap_path=resolved_beatmap,
        difficulty=resolved_difficulty,
    )


def iter_probe_events(
    inputs: StreamProbeInputs,
    *,
    device_name: str = "auto",
    target_buffer_ms: int = 2000,
    rebuffer_floor_ms: int = 750,
    max_windows: int = DEFAULT_MAX_WINDOWS,
) -> Iterator[PreviewEvent]:
    if max_windows < 0:
        raise ValueError(f"max_windows must be non-negative: {max_windows}")

    runner = Stage1CheckpointRunner.load(inputs.checkpoint_path, device_name=device_name)
    source = FullSongFeatureSource(
        audio_path=inputs.audio_path,
        beatmap_path=inputs.beatmap_path,
        bpm_log_mean=runner.bpm_log_mean,
        bpm_log_std=runner.bpm_log_std,
    )
    event_source = source if max_windows == 0 else _WindowLimitedSource(source, max_windows=max_windows)
    yield PreviewEvent(
        "probe",
        {
            "checkpoint_path": inputs.checkpoint_path.as_posix(),
            "audio_path": inputs.audio_path.as_posix(),
            "beatmap_path": inputs.beatmap_path.as_posix(),
            "difficulty": float(inputs.difficulty),
            "max_windows": max_windows,
        },
    )
    yield from build_preview_event_stream(
        source=event_source,
        runner=runner,
        assembler=StreamingAssembler(vocab=runner.vocab),
        difficulty=inputs.difficulty,
        target_buffer_ms=target_buffer_ms,
        rebuffer_floor_ms=rebuffer_floor_ms,
    )


def format_probe_event(event: PreviewEvent, *, output_format: str = "jsonl") -> str:
    if output_format == "jsonl":
        return json.dumps({"event": event.name, "data": event.data}, sort_keys=True, separators=(",", ":"))
    if output_format == "sse":
        return event.to_sse().rstrip("\n")
    raise ValueError(f"unsupported output format: {output_format}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Print Stage 1 oracle mapper streaming output for an audio/.osu pair.")
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--audio-path", type=Path, default=None)
    parser.add_argument("--beatmap-path", type=Path, default=None)
    parser.add_argument("--difficulty", type=float, default=None)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--index-path", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--target-difficulty", type=float, default=DEFAULT_TARGET_DIFFICULTY)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--target-buffer-ms", default=2000, type=int)
    parser.add_argument("--rebuffer-floor-ms", default=750, type=int)
    parser.add_argument(
        "--max-windows",
        default=DEFAULT_MAX_WINDOWS,
        type=int,
        help="Number of 8s write windows to decode; use 0 for the full song.",
    )
    parser.add_argument("--format", choices=("jsonl", "sse"), default="jsonl")
    args = parser.parse_args(argv)

    try:
        inputs = resolve_probe_inputs(
            checkpoint_path=args.checkpoint_path,
            audio_path=args.audio_path,
            beatmap_path=args.beatmap_path,
            difficulty=args.difficulty,
            dataset_root=args.dataset_root,
            index_path=args.index_path,
            runs_root=args.runs_root,
            target_difficulty=args.target_difficulty,
        )
        for event in iter_probe_events(
            inputs,
            device_name=args.device,
            target_buffer_ms=args.target_buffer_ms,
            rebuffer_floor_ms=args.rebuffer_floor_ms,
            max_windows=args.max_windows,
        ):
            print(format_probe_event(event, output_format=args.format), flush=True)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


def _audio_path_from_beatmap(beatmap_path: Path) -> Path:
    metadata = parse_osu_metadata(beatmap_path)
    if not metadata.audio_filename:
        raise ValueError(f"{beatmap_path} is missing AudioFilename")
    return beatmap_path.parent / metadata.audio_filename


def _select_default_index_row(
    *,
    dataset_root: Path,
    index_path: Path,
    target_difficulty: float,
) -> _DefaultBeatmapInput:
    if not math.isfinite(target_difficulty):
        raise ValueError(f"target_difficulty must be finite: {target_difficulty}")
    if not index_path.is_file():
        raise FileNotFoundError(f"default beatmap index not found: {index_path}")

    import pandas as pd

    frame = pd.read_parquet(index_path)
    required_columns = {"shard", "beatmap_path", "audio_path", "difficulty", "key_count"}
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        raise ValueError(f"default beatmap index missing columns: {missing_columns}")

    eligible = frame[
        (frame["key_count"] == 4)
        & (frame["difficulty"].astype(float) >= 2.0)
        & (frame["difficulty"].astype(float) <= 6.0)
    ].copy()
    if eligible.empty:
        raise ValueError(f"default beatmap index has no 4K maps in the supported 2.0..6.0 range: {index_path}")

    eligible["_target_distance"] = (eligible["difficulty"].astype(float) - target_difficulty).abs()
    eligible = eligible.sort_values(["_target_distance", "difficulty", "beatmap_path"], kind="mergesort")

    for row in eligible.itertuples(index=False):
        shard = str(getattr(row, "shard"))
        beatmap = dataset_root / shard / str(getattr(row, "beatmap_path"))
        audio = dataset_root / shard / str(getattr(row, "audio_path"))
        if beatmap.is_file() and audio.is_file():
            return _DefaultBeatmapInput(
                audio_path=audio,
                beatmap_path=beatmap,
                difficulty=float(getattr(row, "difficulty")),
            )

    raise FileNotFoundError(
        f"no eligible default beatmap/audio files from {index_path} exist under {dataset_root}",
    )


if __name__ == "__main__":
    raise SystemExit(main())
