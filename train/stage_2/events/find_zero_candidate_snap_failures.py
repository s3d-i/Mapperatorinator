from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from train.stage_2.events.audit_high_precision_grid_adapter import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_INDEX_PATH,
    D_UNIVERSE_2TO6,
    AdapterConfig,
    FittedTimingSection,
    PrimitiveEvent,
    generate_tick_candidates,
    parse_primitive_events,
    sections_from_fitted_grid,
)
from train.stage_2.osu_core.timing import InvalidRedTimingError, MissingRedTimingError, require_red_timing_points
from train.stage_2.timing.providers.oracle import fitted_timing_grid_from_red_points


DEFAULT_OUTPUT_DIR = Path("train/artifacts/reports/events")
DEFAULT_OUTPUT_JSON_PATH = DEFAULT_OUTPUT_DIR / "zero_candidate_snap_failures.json"
DEFAULT_OUTPUT_MD_PATH = DEFAULT_OUTPUT_DIR / "zero_candidate_snap_failures.md"
DEFAULT_OUTPUT_CSV_PATH = DEFAULT_OUTPUT_DIR / "zero_candidate_snap_failures.csv"
DEFAULT_CONTEXT_RADIUS = 6
DEFAULT_RECORD_TIME_LIMIT = 32
WIDER_TOLERANCES_MS = (0.75, 1.0, 1.5, 2.0, 5.0, 10.0, 25.0)


def find_zero_candidate_snap_failures(
    *,
    index_path: str | Path = DEFAULT_INDEX_PATH,
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    output_json_path: str | Path = DEFAULT_OUTPUT_JSON_PATH,
    output_md_path: str | Path = DEFAULT_OUTPUT_MD_PATH,
    output_csv_path: str | Path = DEFAULT_OUTPUT_CSV_PATH,
    divisors: Sequence[int] = D_UNIVERSE_2TO6,
    offset_min_ms: int = -2,
    offset_max_ms: int = 2,
    residual_tolerance_ms: float = 0.5,
    max_maps: int | None = None,
    record_time_limit: int = DEFAULT_RECORD_TIME_LIMIT,
    progress_every: int = 250,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    index_path = Path(index_path)
    dataset_root = Path(dataset_root)
    output_json_path = Path(output_json_path)
    output_md_path = Path(output_md_path)
    output_csv_path = Path(output_csv_path)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_md_path.parent.mkdir(parents=True, exist_ok=True)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    config = AdapterConfig(
        name="zero_candidate_scan",
        divisors=tuple(int(divisor) for divisor in divisors),
        offset_min_ms=int(offset_min_ms),
        offset_max_ms=int(offset_max_ms),
        residual_tolerance_ms=float(residual_tolerance_ms),
    )
    index_df = _load_filtered_unique_index(index_path)
    if max_maps is not None:
        index_df = index_df.head(max_maps).copy()

    map_rows: list[dict[str, Any]] = []
    map_reports: list[dict[str, Any]] = []
    parse_failures: Counter[str] = Counter()
    counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    zero_candidate_time_total = 0

    for row_number, row in enumerate(index_df.itertuples(index=False), start=1):
        beatmap_key = _beatmap_key(row)
        beatmap_path = dataset_root / str(row.shard) / str(row.beatmap_path)
        try:
            sections = _oracle_sections_for_beatmap(beatmap_key=beatmap_key, beatmap_path=beatmap_path)
            events = parse_primitive_events(beatmap_path)
        except MissingRedTimingError:
            parse_failures["missing_red_timing"] += 1
            continue
        except InvalidRedTimingError:
            parse_failures["invalid_red_timing"] += 1
            continue
        except (OSError, ValueError) as exc:
            parse_failures[_parse_failure_type(str(exc))] += 1
            continue

        counts["beatmaps_scanned"] += 1
        counts["primitive_events_scanned"] += len(events)
        for event in events:
            action_counts[event.action] += 1
        unique_times = sorted({event.original_time_ms for event in events})
        counts["unique_raw_times_scanned"] += len(unique_times)
        events_by_time = _events_by_time(events)

        zero_times: list[dict[str, Any]] = []
        zero_time_count = 0
        first_best: dict[str, Any] | None = None
        for index, raw_time_ms in enumerate(unique_times):
            candidates = generate_tick_candidates(
                raw_time_ms,
                sections,
                config.divisors,
                offsets=config.offsets,
                residual_tolerance_ms=config.residual_tolerance_ms,
            )
            if candidates:
                continue
            zero_time_count += 1
            zero_candidate_time_total += 1
            if len(zero_times) < record_time_limit:
                detail = _zero_time_detail(
                    raw_time_ms=raw_time_ms,
                    time_index=index,
                    unique_times=unique_times,
                    events_at_time=events_by_time[raw_time_ms],
                    sections=sections,
                    config=config,
                )
                if first_best is None:
                    first_best = detail.get("nearest_wide_candidate")
                zero_times.append(detail)

        if zero_time_count > 0:
            first_time = zero_times[0] if zero_times else None
            counts["beatmaps_with_zero_candidate"] += 1
            map_report = {
                "beatmap_key": beatmap_key,
                "beatmap_path": beatmap_path.as_posix(),
                "title": getattr(row, "title", None),
                "artist": getattr(row, "artist", None),
                "creator": getattr(row, "creator", None),
                "version": getattr(row, "version", None),
                "difficulty": _float_or_none(getattr(row, "difficulty", None)),
                "primitive_event_count": len(events),
                "unique_raw_time_count": len(unique_times),
                "timing_section_count": len(sections),
                "zero_candidate_time_count": int(zero_time_count),
                "recorded_zero_candidate_times": zero_times,
            }
            map_reports.append(map_report)
            map_rows.append(
                {
                    "beatmap_key": beatmap_key,
                    "beatmap_path": beatmap_path.as_posix(),
                    "difficulty": map_report["difficulty"],
                    "primitive_event_count": len(events),
                    "unique_raw_time_count": len(unique_times),
                    "timing_section_count": len(sections),
                    "zero_candidate_time_count": map_report["zero_candidate_time_count"],
                    "first_zero_candidate_raw_time_ms": first_time["raw_time_ms"] if first_time else None,
                    "first_zero_candidate_time_index": first_time["time_index"] if first_time else None,
                    "first_nearest_abs_residual_ms": _nearest_field(first_best, "abs_residual_ms"),
                    "first_nearest_residual_ms": _nearest_field(first_best, "residual_ms"),
                    "first_nearest_offset_ms": _nearest_field(first_best, "offset_ms"),
                    "first_nearest_divisor": _nearest_field(first_best, "divisor"),
                    "first_nearest_section_id": _nearest_field(first_best, "section_id"),
                    "title": map_report["title"],
                    "artist": map_report["artist"],
                    "creator": map_report["creator"],
                    "version": map_report["version"],
                }
            )

        if progress_every > 0 and (row_number == 1 or row_number % progress_every == 0):
            print(
                f"zero_candidate_scan progress maps={row_number}/{len(index_df)} "
                f"scanned={counts['beatmaps_scanned']} "
                f"zero_maps={counts['beatmaps_with_zero_candidate']} "
                f"zero_times={zero_candidate_time_total}",
                file=sys.stderr,
                flush=True,
            )

    map_reports.sort(key=lambda item: (-int(item["zero_candidate_time_count"]), item["beatmap_path"]))
    map_rows.sort(key=lambda item: (-int(item["zero_candidate_time_count"]), item["beatmap_path"]))

    payload = {
        "audit_name": "stage2_zero_candidate_snap_failure_scan",
        "pinned_commit": _git_rev_parse("HEAD"),
        "index_path": index_path.as_posix(),
        "dataset_root": dataset_root.as_posix(),
        "config": {
            "divisors": list(config.divisors),
            "offset_range_ms": [config.offset_min_ms, config.offset_max_ms],
            "residual_tolerance_ms": config.residual_tolerance_ms,
            "grid_source": "oracle_red_timing",
            "candidate_definition": (
                "A raw time has a candidate when any integer offset in the configured range lands within "
                "the residual tolerance of an oracle high-precision grid tick for the configured divisor set."
            ),
        },
        "summary": {
            "beatmaps_in_index": int(len(index_df)),
            "beatmaps_scanned": int(counts["beatmaps_scanned"]),
            "beatmaps_with_zero_candidate": int(counts["beatmaps_with_zero_candidate"]),
            "zero_candidate_beatmap_rate": _rate(
                counts["beatmaps_with_zero_candidate"],
                counts["beatmaps_scanned"],
            ),
            "primitive_events_scanned": int(counts["primitive_events_scanned"]),
            "unique_raw_times_scanned": int(counts["unique_raw_times_scanned"]),
            "zero_candidate_time_count": int(zero_candidate_time_total),
            "zero_candidate_time_rate": _rate(
                zero_candidate_time_total,
                counts["unique_raw_times_scanned"],
            ),
            "action_counts": dict(sorted(action_counts.items())),
            "parse_failures": dict(sorted(parse_failures.items())),
            "elapsed_seconds": time.perf_counter() - started_at,
        },
        "worst_beatmaps": map_reports[:100],
        "all_zero_candidate_beatmaps_csv": output_csv_path.as_posix(),
    }

    output_json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    output_md_path.write_text(_render_markdown(payload, map_rows[:50]), encoding="utf-8")
    _write_csv(output_csv_path, map_rows)
    return payload


def _load_filtered_unique_index(index_path: Path) -> pd.DataFrame:
    index_df = pd.read_parquet(index_path)
    required_columns = {"shard", "beatmap_path", "difficulty", "mode", "key_count"}
    missing = sorted(required_columns.difference(index_df.columns))
    if missing:
        raise ValueError(f"{index_path} is missing required column(s): {missing}")
    filtered = index_df[
        (index_df["mode"] == 3)
        & (index_df["key_count"] == 4)
        & (index_df["difficulty"] >= 2.0)
        & (index_df["difficulty"] <= 6.0)
    ].copy()
    return filtered.drop_duplicates(["shard", "beatmap_path"]).reset_index(drop=True)


def _oracle_sections_for_beatmap(*, beatmap_key: str, beatmap_path: Path) -> tuple[FittedTimingSection, ...]:
    grid = fitted_timing_grid_from_red_points(require_red_timing_points(beatmap_path))
    return sections_from_fitted_grid(
        beatmap_key=beatmap_key,
        beatmap_path=beatmap_path,
        grid=grid,
        source="oracle_red_timing",
    )


def _beatmap_key(row: object) -> str:
    beatmap_id = getattr(row, "beatmap_id", None)
    if beatmap_id is not None and not pd.isna(beatmap_id):
        return str(int(beatmap_id))
    return f"{getattr(row, 'shard')}:{getattr(row, 'beatmap_path')}"


def _events_by_time(events: Sequence[PrimitiveEvent]) -> dict[int, list[PrimitiveEvent]]:
    result: dict[int, list[PrimitiveEvent]] = {}
    for event in events:
        result.setdefault(event.original_time_ms, []).append(event)
    return result


def _zero_time_detail(
    *,
    raw_time_ms: int,
    time_index: int,
    unique_times: Sequence[int],
    events_at_time: Sequence[PrimitiveEvent],
    sections: Sequence[FittedTimingSection],
    config: AdapterConfig,
) -> dict[str, Any]:
    nearest = _nearest_wide_candidate(raw_time_ms, sections, config)
    section = _section_for_raw_time(raw_time_ms, sections)
    left = max(0, time_index - DEFAULT_CONTEXT_RADIUS)
    right = min(len(unique_times), time_index + DEFAULT_CONTEXT_RADIUS + 1)
    return {
        "raw_time_ms": int(raw_time_ms),
        "time_index": int(time_index),
        "section": _section_payload(section),
        "events_at_time": [_event_payload(event) for event in events_at_time],
        "context_raw_times_ms": [int(value) for value in unique_times[left:right]],
        "nearest_wide_candidate": nearest,
        "passes_wider_tolerance_ms": _passes_wider_tolerance(nearest),
    }


def _nearest_wide_candidate(
    raw_time_ms: int,
    sections: Sequence[FittedTimingSection],
    config: AdapterConfig,
) -> dict[str, Any] | None:
    candidates = generate_tick_candidates(
        raw_time_ms,
        sections,
        config.divisors,
        offsets=config.offsets,
        residual_tolerance_ms=max(WIDER_TOLERANCES_MS),
    )
    if not candidates:
        return None
    candidate = min(
        candidates,
        key=lambda item: (
            abs(item.residual_ms),
            item.divisor,
            abs(item.offset_ms),
            item.section_id,
            item.tick_index,
        ),
    )
    return {
        "offset_ms": int(candidate.offset_ms),
        "adapted_time_ms": float(candidate.adapted_time_ms),
        "section_id": int(candidate.section_id),
        "divisor": int(candidate.divisor),
        "tick_index": int(candidate.tick_index),
        "tick_time_ms_hp": float(candidate.tick_time_ms_hp),
        "residual_ms": float(candidate.residual_ms),
        "abs_residual_ms": abs(float(candidate.residual_ms)),
        "cross_section": bool(candidate.cross_section),
    }


def _passes_wider_tolerance(nearest: dict[str, Any] | None) -> dict[str, bool]:
    if nearest is None:
        return {str(tolerance): False for tolerance in WIDER_TOLERANCES_MS}
    abs_residual = float(nearest["abs_residual_ms"])
    return {str(tolerance): abs_residual <= tolerance for tolerance in WIDER_TOLERANCES_MS}


def _section_for_raw_time(raw_time_ms: int, sections: Sequence[FittedTimingSection]) -> FittedTimingSection:
    selected = sections[0]
    for section in sections:
        if section.start_ms_hp <= raw_time_ms:
            selected = section
        else:
            break
    return selected


def _section_payload(section: FittedTimingSection) -> dict[str, Any]:
    return {
        "section_id": int(section.section_id),
        "start_ms_hp": float(section.start_ms_hp),
        "end_ms_hp": None if section.end_ms_hp is None else float(section.end_ms_hp),
        "beat_length_ms_hp": float(section.beat_length_ms_hp),
        "bpm_hp": float(section.bpm_hp),
        "meter": int(section.meter),
    }


def _event_payload(event: PrimitiveEvent) -> dict[str, Any]:
    return {
        "object_id": int(event.object_id),
        "lane": int(event.lane),
        "action": str(event.action),
    }


def _nearest_field(nearest: dict[str, Any] | None, field: str) -> Any:
    if nearest is None:
        return None
    return nearest.get(field)


def _float_or_none(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _parse_failure_type(message: str) -> str:
    lowered = message.lower()
    if "not a 4k" in lowered or "not an osu!mania" in lowered:
        return "unsupported_key_count_or_mode"
    if "hold" in lowered:
        return "malformed_hold"
    if "non-integer" in lowered:
        return "non_integer_hitobject_time"
    return "unknown_parser_issue"


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fieldnames = [
        "beatmap_key",
        "beatmap_path",
        "difficulty",
        "primitive_event_count",
        "unique_raw_time_count",
        "timing_section_count",
        "zero_candidate_time_count",
        "first_zero_candidate_raw_time_ms",
        "first_zero_candidate_time_index",
        "first_nearest_abs_residual_ms",
        "first_nearest_residual_ms",
        "first_nearest_offset_ms",
        "first_nearest_divisor",
        "first_nearest_section_id",
        "title",
        "artist",
        "creator",
        "version",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _render_markdown(payload: dict[str, Any], rows: Sequence[dict[str, Any]]) -> str:
    summary = payload["summary"]
    lines = [
        "---",
        f"pinned_commit: {payload['pinned_commit']}",
        "audit: stage2_zero_candidate_snap_failure_scan",
        "---",
        "",
        "# Stage 2 Zero-Candidate Snap Failure Scan",
        "",
        "This report lists beatmaps that contain at least one unique raw integer-ms event time with no candidate "
        "oracle timing-grid tick under the configured snap candidate rule.",
        "",
        "## Config",
        "",
        f"- Index: `{payload['index_path']}`",
        f"- Grid source: `{payload['config']['grid_source']}`",
        f"- Divisors: `{payload['config']['divisors']}`",
        f"- Offset range: `{payload['config']['offset_range_ms']}` ms",
        f"- Residual tolerance: `{payload['config']['residual_tolerance_ms']}` ms",
        "",
        "## Summary",
        "",
        f"- Beatmaps scanned: {summary['beatmaps_scanned']:,}",
        f"- Beatmaps with zero-candidate times: {summary['beatmaps_with_zero_candidate']:,} "
        f"({_pct(summary['zero_candidate_beatmap_rate'])})",
        f"- Unique raw times scanned: {summary['unique_raw_times_scanned']:,}",
        f"- Zero-candidate raw times: {summary['zero_candidate_time_count']:,} "
        f"({_pct(summary['zero_candidate_time_rate'])})",
        f"- Parse failures: `{summary['parse_failures']}`",
        f"- CSV: `{payload['all_zero_candidate_beatmaps_csv']}`",
        "",
        "## Worst Beatmaps",
        "",
        "| zero times | first time | nearest abs residual | difficulty | beatmap |",
        "|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        beatmap = str(row["beatmap_path"]).replace("|", "\\|")
        residual = row["first_nearest_abs_residual_ms"]
        lines.append(
            f"| {int(row['zero_candidate_time_count']):,} | "
            f"{_num(row['first_zero_candidate_raw_time_ms'])} | "
            f"{_num(residual)} | "
            f"{_num(row['difficulty'])} | "
            f"`{beatmap}` |"
        )
    lines.append("")
    return "\n".join(lines)


def _rate(numerator: int | float, denominator: int | float) -> float:
    if not denominator:
        return 0.0
    return float(numerator) / float(denominator)


def _pct(value: float) -> str:
    return f"{100.0 * float(value):.4f}%"


def _num(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _git_rev_parse(revision: str) -> str:
    try:
        result = subprocess.run(["git", "rev-parse", revision], check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Find Stage 2 beatmaps with raw times that have zero snap candidates.")
    parser.add_argument("--index-path", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON_PATH)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD_PATH)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV_PATH)
    parser.add_argument("--divisors", type=int, nargs="+", default=list(D_UNIVERSE_2TO6))
    parser.add_argument("--offset-min-ms", type=int, default=-2)
    parser.add_argument("--offset-max-ms", type=int, default=2)
    parser.add_argument("--residual-tolerance-ms", type=float, default=0.5)
    parser.add_argument("--max-maps", type=int, default=None)
    parser.add_argument("--record-time-limit", type=int, default=DEFAULT_RECORD_TIME_LIMIT)
    parser.add_argument("--progress-every", type=int, default=250)
    args = parser.parse_args(argv)

    payload = find_zero_candidate_snap_failures(
        index_path=args.index_path,
        dataset_root=args.dataset_root,
        output_json_path=args.output_json,
        output_md_path=args.output_md,
        output_csv_path=args.output_csv,
        divisors=tuple(args.divisors),
        offset_min_ms=args.offset_min_ms,
        offset_max_ms=args.offset_max_ms,
        residual_tolerance_ms=args.residual_tolerance_ms,
        max_maps=args.max_maps,
        record_time_limit=args.record_time_limit,
        progress_every=args.progress_every,
    )
    summary = payload["summary"]
    print(
        "zero_candidate_snap_failure_scan "
        f"beatmaps={summary['beatmaps_scanned']} "
        f"zero_maps={summary['beatmaps_with_zero_candidate']} "
        f"zero_times={summary['zero_candidate_time_count']} "
        f"json={args.output_json} md={args.output_md} csv={args.output_csv}",
        flush=True,
    )


if __name__ == "__main__":
    main()
