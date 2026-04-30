from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from train.stage_2.timing.grid_fitting import GridFitter, GridFitterConfig
from train.stage_2.timing.rendering.dense_timing_v2 import (
    DEFAULT_DENSE_TIMING_V2_CONFIG,
    DenseTimingV2Config,
    active_timing_arrays,
    dense_timing_v2_frame_times,
    render_dense_timing_v2,
)
from train.stage_2.timing.schema import FittedTimingGrid, TimingSegment


@dataclass(frozen=True)
class TimingGridComparison:
    frame_count: int
    beat_pulse_mae: float
    local_bpm_mae: float
    mean_phase_error_beats: float
    max_phase_error_beats: float
    mean_phase_error_ms: float
    max_phase_error_ms: float


def compare_timing_grids(
    predicted_grid: FittedTimingGrid,
    oracle_grid: FittedTimingGrid,
    *,
    frame_count: int,
    input_start_ms: float = 0.0,
    config: DenseTimingV2Config = DEFAULT_DENSE_TIMING_V2_CONFIG,
) -> TimingGridComparison:
    if frame_count <= 0:
        raise ValueError(f"frame_count must be positive, got {frame_count!r}")

    predicted_track = render_dense_timing_v2(
        predicted_grid,
        input_start_ms=input_start_ms,
        frame_count=frame_count,
        config=config,
    )
    oracle_track = render_dense_timing_v2(
        oracle_grid,
        input_start_ms=input_start_ms,
        frame_count=frame_count,
        config=config,
    )

    angle_delta = np.arctan2(
        predicted_track[:, 1] * oracle_track[:, 2] - predicted_track[:, 2] * oracle_track[:, 1],
        predicted_track[:, 2] * oracle_track[:, 2] + predicted_track[:, 1] * oracle_track[:, 1],
    )
    phase_error_beats = np.abs(angle_delta) / (2.0 * np.pi)

    frame_times_ms = dense_timing_v2_frame_times(
        input_start_ms,
        frame_count=frame_count,
        config=config,
    )
    _, oracle_beat_lengths_ms = active_timing_arrays(oracle_grid, frame_times_ms)
    phase_error_ms = phase_error_beats * oracle_beat_lengths_ms

    return TimingGridComparison(
        frame_count=frame_count,
        beat_pulse_mae=float(np.mean(np.abs(predicted_track[:, 0] - oracle_track[:, 0]))),
        local_bpm_mae=float(np.mean(np.abs(predicted_track[:, 3] - oracle_track[:, 3]))),
        mean_phase_error_beats=float(np.mean(phase_error_beats)),
        max_phase_error_beats=float(np.max(phase_error_beats)),
        mean_phase_error_ms=float(np.mean(phase_error_ms)),
        max_phase_error_ms=float(np.max(phase_error_ms)),
    )


def oracle_grid_from_red_timing_points(red_timing_points: Sequence[object]) -> FittedTimingGrid:
    return FittedTimingGrid(
        segments=tuple(
            TimingSegment(
                offset_ms=float(point.offset_ms),
                beat_length_ms=float(point.beat_length_ms),
                meter=int(getattr(point, "meter", 4)),
            )
            for point in red_timing_points
        )
    )


def run_beatthis_oracle_comparison(
    *,
    index_path: Path,
    dataset_root: Path,
    sample_size: int,
    seed: int,
    device: str,
    double_tempo_score_ratio_threshold: float | None = None,
    max_average_fit_seconds: float | None = 1.0,
) -> dict[str, object]:
    import pandas as pd

    from train.stage_2.osu_core.timing import require_red_timing_points
    from train.stage_2.timing.providers.beatthis import BeatThisTimingProvider

    if sample_size <= 0:
        raise ValueError(f"sample_size must be positive, got {sample_size!r}")

    index_df = pd.read_parquet(index_path)
    sample_df = index_df.sample(n=min(sample_size, len(index_df)), random_state=seed)
    provider = BeatThisTimingProvider(device=device)
    fitter_config = (
        GridFitterConfig()
        if double_tempo_score_ratio_threshold is None
        else GridFitterConfig(double_tempo_score_ratio_threshold=double_tempo_score_ratio_threshold)
    )
    fitter = GridFitter(fitter_config)

    rows: list[dict[str, object]] = []
    for _, row in sample_df.iterrows():
        beatmap_path = dataset_root / str(row["shard"]) / str(row["beatmap_path"])
        audio_path = dataset_root / str(row["shard"]) / str(row["audio_path"])
        oracle_grid = oracle_grid_from_red_timing_points(require_red_timing_points(beatmap_path))
        prediction = provider.predict_file(audio_path)
        fit_start_seconds = time.perf_counter()
        fit_result = fitter.fit(prediction)
        fit_seconds = time.perf_counter() - fit_start_seconds
        comparison = compare_timing_grids(
            fit_result.grid,
            oracle_grid,
            frame_count=prediction.frame_count,
        )
        predicted_segment = fit_result.grid.segments[0]
        oracle_segment = oracle_grid.segments[0]
        rows.append(
            {
                "beatmap_path": beatmap_path.as_posix(),
                "audio_path": audio_path.as_posix(),
                "frame_count": prediction.frame_count,
                "fit_score": fit_result.score,
                "fit_seconds": fit_seconds,
                "predicted_bpm": predicted_segment.local_bpm,
                "predicted_offset_ms": predicted_segment.offset_ms,
                "oracle_first_bpm": oracle_segment.local_bpm,
                "oracle_first_offset_ms": oracle_segment.offset_ms,
                "oracle_segment_count": len(oracle_grid.segments),
                "raw_selected_bpm": fit_result.diagnostics.raw_selected_bpm,
                "raw_score": fit_result.diagnostics.raw_score,
                "half_tempo_score": _finite_float_or_none(fit_result.diagnostics.half_tempo_score),
                "double_tempo_score": _finite_float_or_none(fit_result.diagnostics.double_tempo_score),
                "tempo_multiplier": fit_result.diagnostics.tempo_multiplier,
                **asdict(comparison),
            }
        )

    summary = _summarize_rows(rows)
    _enforce_average_fit_seconds(summary, max_average_fit_seconds=max_average_fit_seconds)
    return {
        "sample_size": len(rows),
        "max_average_fit_seconds": max_average_fit_seconds,
        "rows": rows,
        "summary": summary,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare BeatThis-fitted dense timing v2 against osu red timing.")
    parser.add_argument("--index-path", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--double-tempo-score-ratio-threshold", type=float, default=None)
    parser.add_argument(
        "--max-average-fit-seconds",
        type=float,
        default=1.0,
        help="Fail if average GridFitter time exceeds this value; use 0 to disable.",
    )
    args = parser.parse_args(argv)
    max_average_fit_seconds = None if args.max_average_fit_seconds <= 0.0 else args.max_average_fit_seconds

    report = run_beatthis_oracle_comparison(
        index_path=args.index_path,
        dataset_root=args.dataset_root,
        sample_size=args.sample_size,
        seed=args.seed,
        device=args.device,
        double_tempo_score_ratio_threshold=args.double_tempo_score_ratio_threshold,
        max_average_fit_seconds=max_average_fit_seconds,
    )
    print(json.dumps(report, allow_nan=False, indent=2, sort_keys=True))
    return 0


def _summarize_rows(rows: Sequence[dict[str, object]]) -> dict[str, float]:
    if not rows:
        return {}

    metric_names = (
        "fit_score",
        "fit_seconds",
        "beat_pulse_mae",
        "local_bpm_mae",
        "mean_phase_error_beats",
        "max_phase_error_beats",
        "mean_phase_error_ms",
        "max_phase_error_ms",
    )
    summary: dict[str, float] = {}
    for name in metric_names:
        values = np.asarray([float(row[name]) for row in rows], dtype=np.float64)
        summary[f"{name}_mean"] = float(np.mean(values))
        summary[f"{name}_max"] = float(np.max(values))
    return summary


def _enforce_average_fit_seconds(
    summary: dict[str, float],
    *,
    max_average_fit_seconds: float | None,
) -> None:
    if max_average_fit_seconds is None or not summary:
        return
    if max_average_fit_seconds <= 0.0:
        raise ValueError(f"max_average_fit_seconds must be positive or None, got {max_average_fit_seconds!r}")
    average_fit_seconds = summary.get("fit_seconds_mean")
    if average_fit_seconds is not None and average_fit_seconds > max_average_fit_seconds:
        raise RuntimeError(
            "average fitter time exceeded hard limit: "
            f"{average_fit_seconds:.3f}s > {max_average_fit_seconds:.3f}s"
        )


def _finite_float_or_none(value: float) -> float | None:
    if not np.isfinite(value):
        return None
    return float(value)


if __name__ == "__main__":
    raise SystemExit(main())
