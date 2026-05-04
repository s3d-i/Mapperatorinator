from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from train.stage_2.osu_core.timing import require_red_timing_points
from train.stage_2.timing.rendering.dense_timing_v2 import DENSE_TIMING_V2_CHANNELS
from train.stage_2.timing.rendering.dense_timing_v2 import DENSE_TIMING_V2_VERSION
from train.stage_2.timing.schema import FittedTimingGrid
from train.stage_2.timing.schema import TimingSegment


DEFAULT_SOURCE_INDEX_PATH = Path("train/artifacts/indexes/beatmap_index_4k_no_timing_anomalies_2to6.parquet")
DEFAULT_DATASET_ROOT = Path("mania-dataset")
DEFAULT_OUTPUT_PATH = Path(
    "train/artifacts/indexes/"
    "beatmap_index_4k_no_timing_anomalies_2to6_dense_local_bpm_norm_unique_le3.parquet"
)
DEFAULT_REPORT_PATH = Path(
    "train/artifacts/reports/indexes/"
    "beatmap_index_4k_no_timing_anomalies_2to6_dense_local_bpm_norm_unique_le3.json"
)
DEFAULT_MAX_LOCAL_BPM_NORM_UNIQUE_PER_BEATMAPSET = 3
DEFAULT_BPM_ROUND_DECIMALS = 6
_LOCAL_BPM_CHANNEL = DENSE_TIMING_V2_CHANNELS.index("local_bpm")
_REQUIRED_INDEX_COLUMNS = frozenset(("shard", "beatmap_set_id", "beatmap_path"))


@dataclass(frozen=True)
class DenseTimingV2LocalBpmNormUniqueIndexReport:
    source_index_path: Path
    output_path: Path
    dataset_root: Path
    source_map_count: int
    retained_map_count: int
    dropped_map_count: int
    source_beatmapset_count: int
    retained_beatmapset_count: int
    dropped_beatmapset_count: int
    max_local_bpm_norm_unique_per_beatmapset: int
    max_observed_local_bpm_norm_unique_per_beatmapset: int
    bpm_round_decimals: int
    global_min_local_bpm: float | None
    global_max_local_bpm: float | None
    dropped_examples: list[dict[str, Any]]


def build_dense_timing_v2_local_bpm_norm_unique_index(
    *,
    source_index_path: str | Path = DEFAULT_SOURCE_INDEX_PATH,
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    report_path: str | Path | None = DEFAULT_REPORT_PATH,
    max_local_bpm_norm_unique_per_beatmapset: int = DEFAULT_MAX_LOCAL_BPM_NORM_UNIQUE_PER_BEATMAPSET,
    bpm_round_decimals: int = DEFAULT_BPM_ROUND_DECIMALS,
    dropped_example_limit: int = 20,
    progress_every: int = 0,
    command: str | None = None,
) -> DenseTimingV2LocalBpmNormUniqueIndexReport:
    if max_local_bpm_norm_unique_per_beatmapset < 1:
        raise ValueError(
            "max_local_bpm_norm_unique_per_beatmapset must be positive, "
            f"got {max_local_bpm_norm_unique_per_beatmapset!r}",
        )
    if bpm_round_decimals < 0:
        raise ValueError(f"bpm_round_decimals must be non-negative, got {bpm_round_decimals!r}")
    if dropped_example_limit < 0:
        raise ValueError(f"dropped_example_limit must be non-negative, got {dropped_example_limit!r}")

    started_at = time.perf_counter()
    source_index_path = Path(source_index_path)
    dataset_root = Path(dataset_root)
    output_path = Path(output_path)
    report_path = None if report_path is None else Path(report_path)

    source_df = pd.read_parquet(source_index_path)
    _require_index_columns(source_df, source_index_path)

    group_rows: dict[tuple[str, str], list[int]] = {}
    group_values: dict[tuple[str, str], set[float]] = {}
    group_metadata: dict[tuple[str, str], dict[str, Any]] = {}
    local_bpm_min: float | None = None
    local_bpm_max: float | None = None

    for row_index, row in enumerate(source_df.itertuples(index=False), start=1):
        group_key = _beatmapset_group_key(row)
        local_bpms = dense_timing_v2_local_bpms_for_beatmap(dataset_root, row)
        if local_bpms.size:
            row_min = float(np.min(local_bpms))
            row_max = float(np.max(local_bpms))
            local_bpm_min = row_min if local_bpm_min is None else min(local_bpm_min, row_min)
            local_bpm_max = row_max if local_bpm_max is None else max(local_bpm_max, row_max)

        group_rows.setdefault(group_key, []).append(row_index - 1)
        group_values.setdefault(group_key, set()).update(_local_bpm_norm_unique_values(local_bpms, bpm_round_decimals))
        group_metadata.setdefault(group_key, _group_metadata(row))

        if progress_every > 0 and row_index % progress_every == 0:
            print(f"processed {row_index}/{len(source_df)} maps", file=sys.stderr)

    dropped_groups = {
        key
        for key, values in group_values.items()
        if len(values) > max_local_bpm_norm_unique_per_beatmapset
    }
    keep_mask = [
        _beatmapset_group_key(row) not in dropped_groups
        for row in source_df.itertuples(index=False)
    ]
    output_df = source_df.loc[keep_mask].reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_output_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if tmp_output_path.exists():
        tmp_output_path.unlink()
    output_df.to_parquet(tmp_output_path, index=False)
    tmp_output_path.replace(output_path)

    report = DenseTimingV2LocalBpmNormUniqueIndexReport(
        source_index_path=source_index_path,
        output_path=output_path,
        dataset_root=dataset_root,
        source_map_count=len(source_df),
        retained_map_count=len(output_df),
        dropped_map_count=len(source_df) - len(output_df),
        source_beatmapset_count=len(group_rows),
        retained_beatmapset_count=len(group_rows) - len(dropped_groups),
        dropped_beatmapset_count=len(dropped_groups),
        max_local_bpm_norm_unique_per_beatmapset=max_local_bpm_norm_unique_per_beatmapset,
        max_observed_local_bpm_norm_unique_per_beatmapset=max(
            (len(values) for values in group_values.values()),
            default=0,
        ),
        bpm_round_decimals=bpm_round_decimals,
        global_min_local_bpm=local_bpm_min,
        global_max_local_bpm=local_bpm_max,
        dropped_examples=_dropped_examples(
            dropped_groups,
            group_rows=group_rows,
            group_values=group_values,
            group_metadata=group_metadata,
            limit=dropped_example_limit,
        ),
    )

    if report_path is not None:
        _write_report(
            report_path,
            report,
            source_index_sha256=_sha256(source_index_path),
            output_sha256=_sha256(output_path),
            elapsed_s=time.perf_counter() - started_at,
            command=command,
        )
    return report


def dense_timing_v2_local_bpms_for_beatmap(dataset_root: str | Path, row: Mapping[str, object] | object) -> np.ndarray:
    beatmap_path = Path(dataset_root) / str(_row_value(row, "shard")) / str(_row_value(row, "beatmap_path"))
    red_timing_points = require_red_timing_points(beatmap_path)
    grid = _timing_grid_from_red_timing_points(red_timing_points)
    return np.asarray([segment.local_bpm for segment in grid.segments], dtype=np.float64)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build dense timing v2 filtered beatmap indexes.")
    parser.add_argument("--source-index-path", type=Path, default=DEFAULT_SOURCE_INDEX_PATH)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument(
        "--max-local-bpm-norm-unique-per-beatmapset",
        "--max-localbpmnorm-unique-per-beatmapset",
        dest="max_local_bpm_norm_unique_per_beatmapset",
        type=int,
        default=DEFAULT_MAX_LOCAL_BPM_NORM_UNIQUE_PER_BEATMAPSET,
    )
    parser.add_argument("--bpm-round-decimals", type=int, default=DEFAULT_BPM_ROUND_DECIMALS)
    parser.add_argument("--dropped-example-limit", type=int, default=20)
    parser.add_argument("--progress-every", type=int, default=0)
    args = parser.parse_args(argv)

    report = build_dense_timing_v2_local_bpm_norm_unique_index(
        source_index_path=args.source_index_path,
        dataset_root=args.dataset_root,
        output_path=args.output_path,
        report_path=args.report_path,
        max_local_bpm_norm_unique_per_beatmapset=args.max_local_bpm_norm_unique_per_beatmapset,
        bpm_round_decimals=args.bpm_round_decimals,
        dropped_example_limit=args.dropped_example_limit,
        progress_every=args.progress_every,
        command=_format_command(argv),
    )
    print(
        "retained "
        f"{report.retained_map_count}/{report.source_map_count} maps; "
        f"dropped {report.dropped_beatmapset_count} beatmapsets",
    )
    return 0


def _require_index_columns(index_df: pd.DataFrame, index_path: Path) -> None:
    missing_columns = sorted(_REQUIRED_INDEX_COLUMNS.difference(index_df.columns))
    if missing_columns:
        raise ValueError(f"{index_path} is missing required column(s): {missing_columns}")


def _timing_grid_from_red_timing_points(red_timing_points: Sequence[object]) -> FittedTimingGrid:
    segments_by_offset: dict[float, TimingSegment] = {}
    for point in red_timing_points:
        offset_ms = float(point.offset_ms)
        segments_by_offset[offset_ms] = TimingSegment(
            offset_ms=offset_ms,
            beat_length_ms=float(point.beat_length_ms),
            meter=int(getattr(point, "meter", 4)),
        )
    return FittedTimingGrid(tuple(segments_by_offset[offset] for offset in sorted(segments_by_offset)))


def _beatmapset_group_key(row: Mapping[str, object] | object) -> tuple[str, str]:
    return str(_row_value(row, "shard")), str(_row_value(row, "beatmap_set_id"))


def _row_value(row: Mapping[str, object] | object, name: str) -> object:
    if isinstance(row, Mapping):
        return row[name]
    return getattr(row, name)


def _group_metadata(row: Mapping[str, object] | object) -> dict[str, Any]:
    return {
        "shard": str(_row_value(row, "shard")),
        "beatmap_set_id": _json_scalar(_row_value(row, "beatmap_set_id")),
        "beatmap_set_path": str(_row_value(row, "beatmap_set_path")) if hasattr(row, "beatmap_set_path") else None,
        "title": str(_row_value(row, "title")) if hasattr(row, "title") else None,
        "artist": str(_row_value(row, "artist")) if hasattr(row, "artist") else None,
    }


def _local_bpm_norm_unique_values(local_bpms: np.ndarray, decimals: int) -> set[float]:
    return {round(float(value), decimals) for value in local_bpms}


def _dropped_examples(
    dropped_groups: set[tuple[str, str]],
    *,
    group_rows: Mapping[tuple[str, str], Sequence[int]],
    group_values: Mapping[tuple[str, str], set[float]],
    group_metadata: Mapping[tuple[str, str], Mapping[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    ranked_groups = sorted(
        dropped_groups,
        key=lambda key: (len(group_values[key]), len(group_rows[key]), key[0], key[1]),
        reverse=True,
    )
    for key in ranked_groups[:limit]:
        values = sorted(group_values[key])
        examples.append(
            {
                **dict(group_metadata[key]),
                "map_count": len(group_rows[key]),
                "local_bpm_norm_unique_count": len(values),
                "local_bpm_norm_values_head": values[:12],
            }
        )
    return examples


def _write_report(
    report_path: Path,
    report: DenseTimingV2LocalBpmNormUniqueIndexReport,
    *,
    source_index_sha256: str,
    output_sha256: str,
    elapsed_s: float,
    command: str | None,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "source_index_path": report.source_index_path.as_posix(),
        "source_index_sha256": source_index_sha256,
        "output_path": report.output_path.as_posix(),
        "output_sha256": output_sha256,
        "dataset_root": report.dataset_root.as_posix(),
        "dense_timing_version": DENSE_TIMING_V2_VERSION,
        "dense_timing_channels": list(DENSE_TIMING_V2_CHANNELS),
        "local_bpm_channel_index": _LOCAL_BPM_CHANNEL,
        "local_bpm_norm_definition": (
            "dense_timing_v2 segment local_bpm rounded to bpm_round_decimals for stable unique counting"
        ),
        "drop_policy": (
            "drop_entire_shard_beatmap_set_id_group_when_dense_timing_v2_local_bpm_norm_unique_count_exceeds_threshold"
        ),
        "max_local_bpm_norm_unique_per_beatmapset": report.max_local_bpm_norm_unique_per_beatmapset,
        "max_observed_local_bpm_norm_unique_per_beatmapset": (
            report.max_observed_local_bpm_norm_unique_per_beatmapset
        ),
        "bpm_round_decimals": report.bpm_round_decimals,
        "source_map_count": report.source_map_count,
        "retained_map_count": report.retained_map_count,
        "dropped_map_count": report.dropped_map_count,
        "source_beatmapset_count": report.source_beatmapset_count,
        "retained_beatmapset_count": report.retained_beatmapset_count,
        "dropped_beatmapset_count": report.dropped_beatmapset_count,
        "global_min_local_bpm": report.global_min_local_bpm,
        "global_max_local_bpm": report.global_max_local_bpm,
        "dropped_examples": report.dropped_examples,
        "elapsed_s": elapsed_s,
        "code_commit": _git_stdout("rev-parse", "HEAD"),
        "code_dirty": bool(_git_stdout("status", "--porcelain")),
        "command": command,
    }
    report_path.write_text(json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_stdout(*args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def _format_command(argv: Sequence[str] | None) -> str:
    args = list(sys.argv[1:] if argv is None else argv)
    return " ".join(
        shlex.quote(part)
        for part in ["uv", "run", "python", "-m", "train.stage_2.data.index", *args]
    )


def _json_scalar(value: object) -> object:
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        return value.item()
    return value


if __name__ == "__main__":
    raise SystemExit(main())
