from __future__ import annotations

import argparse
import json
import shlex
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from train.stage_2.data.control_windows import FRAME_HOP_MS
from train.stage_2.data.mapper_v1_windows import MAPPER_WRITE_FRAMES


DEFAULT_SOURCE_INDEX_PATH = Path("train/artifacts/indexes/stage2_mapper_v1_phase_b_mps_cached_windows.parquet")
DEFAULT_OUTPUT_PATH = Path("train/artifacts/indexes/stage2_mapper_v1_phase_b_mps_cached_windows_plus_end.parquet")
DEFAULT_REPORT_PATH = Path("train/artifacts/reports/indexes/stage2_mapper_v1_phase_b_mps_cached_windows_plus_end.json")
_REQUIRED_COLUMNS = frozenset(("beatmap_path", "frame_count", "target_start_frame", "target_start_ms"))
_MAP_IDENTITY_COLUMNS = ("shard", "beatmap_path", "audio_path", "difficulty", "frame_count")


@dataclass(frozen=True)
class MapperV1EndWindowIndexReport:
    source_index_path: Path
    output_path: Path
    source_rows: int
    output_rows: int
    source_unique_maps: int
    output_unique_maps: int
    maps_shorter_than_write_window: int
    existing_end_window_rows: int
    added_end_window_rows: int
    write_window_frames: int
    frame_hop_ms: int
    elapsed_s: float


def build_mapper_v1_end_window_index(
    *,
    source_index_path: str | Path = DEFAULT_SOURCE_INDEX_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    report_path: str | Path | None = DEFAULT_REPORT_PATH,
    write_window_frames: int = MAPPER_WRITE_FRAMES,
    command: str | None = None,
) -> MapperV1EndWindowIndexReport:
    if write_window_frames <= 0:
        raise ValueError(f"write_window_frames must be positive, got {write_window_frames!r}")

    started_at = time.perf_counter()
    source_index_path = Path(source_index_path)
    output_path = Path(output_path)
    report_path = None if report_path is None else Path(report_path)

    source_df = pd.read_parquet(source_index_path)
    _require_columns(source_df, source_index_path)

    map_columns = _available_map_identity_columns(source_df)
    map_df = source_df.drop_duplicates(subset=map_columns, keep="first").copy()
    map_df["frame_count"] = pd.to_numeric(map_df["frame_count"], errors="raise").astype("int64")
    short_mask = map_df["frame_count"] < int(write_window_frames)
    eligible_maps = map_df.copy()
    eligible_maps["target_start_frame"] = (
        (eligible_maps["frame_count"] - 1) // int(write_window_frames)
    ) * int(write_window_frames)
    eligible_maps["target_start_ms"] = eligible_maps["target_start_frame"] * int(FRAME_HOP_MS)

    key_columns = [*map_columns, "target_start_frame"]
    source_key_df = source_df.loc[:, key_columns].copy()
    source_key_df["target_start_frame"] = pd.to_numeric(
        source_key_df["target_start_frame"],
        errors="raise",
    ).astype("int64")
    source_keys = set(source_key_df.itertuples(index=False, name=None))
    missing_end_mask = [
        row not in source_keys
        for row in eligible_maps.loc[:, key_columns].itertuples(index=False, name=None)
    ]
    end_rows = eligible_maps.loc[missing_end_mask, source_df.columns].copy()

    output_df = pd.concat([source_df, end_rows], ignore_index=True)
    output_df = output_df.astype(source_df.dtypes.to_dict(), copy=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_output_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if tmp_output_path.exists():
        tmp_output_path.unlink()
    output_df.to_parquet(tmp_output_path, index=False, compression="zstd")
    tmp_output_path.replace(output_path)

    report = MapperV1EndWindowIndexReport(
        source_index_path=source_index_path,
        output_path=output_path,
        source_rows=len(source_df),
        output_rows=len(output_df),
        source_unique_maps=len(map_df),
        output_unique_maps=len(output_df.drop_duplicates(subset=map_columns)),
        maps_shorter_than_write_window=int(short_mask.sum()),
        existing_end_window_rows=len(eligible_maps) - len(end_rows),
        added_end_window_rows=len(end_rows),
        write_window_frames=int(write_window_frames),
        frame_hop_ms=int(FRAME_HOP_MS),
        elapsed_s=time.perf_counter() - started_at,
    )
    if report_path is not None:
        _write_report(report_path, report, command=command)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Add full-length terminal EOS windows to a mapper v1 window index.")
    parser.add_argument("--source-index-path", type=Path, default=DEFAULT_SOURCE_INDEX_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--write-window-frames", type=int, default=MAPPER_WRITE_FRAMES)
    args = parser.parse_args(argv)

    report = build_mapper_v1_end_window_index(
        source_index_path=args.source_index_path,
        output_path=args.output_path,
        report_path=args.report_path,
        write_window_frames=args.write_window_frames,
        command=_format_command(argv),
    )
    print(
        "mapper_v1_end_window_index "
        f"source_rows={report.source_rows} output_rows={report.output_rows} "
        f"added_end_window_rows={report.added_end_window_rows} "
        f"existing_end_window_rows={report.existing_end_window_rows}",
        flush=True,
    )
    return 0


def _require_columns(index_df: pd.DataFrame, index_path: Path) -> None:
    missing = sorted(_REQUIRED_COLUMNS.difference(index_df.columns))
    if missing:
        raise ValueError(f"{index_path} is missing required column(s): {missing}")


def _available_map_identity_columns(index_df: pd.DataFrame) -> list[str]:
    columns = [column for column in _MAP_IDENTITY_COLUMNS if column in index_df.columns]
    if "beatmap_path" not in columns:
        raise ValueError("beatmap_path is required in the map identity columns")
    return columns


def _write_report(path: Path, report: MapperV1EndWindowIndexReport, *, command: str | None) -> None:
    payload: dict[str, Any] = {
        key: value.as_posix() if isinstance(value, Path) else value
        for key, value in asdict(report).items()
    }
    if command is not None:
        payload["command"] = command
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def _format_command(argv: Sequence[str] | None) -> str:
    args = list(argv) if argv is not None else []
    return "uv run python -m train.stage_2.data.build_mapper_v1_end_window_index " + " ".join(
        shlex.quote(str(arg)) for arg in args
    )


if __name__ == "__main__":
    raise SystemExit(main())
