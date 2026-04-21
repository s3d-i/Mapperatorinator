from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class RedTimingPoint:
    offset_ms: float
    beat_length_ms: float
    meter: int = 4


class MissingRedTimingError(ValueError):
    pass


def parse_red_timing_points(beatmap_path: str | Path) -> list[RedTimingPoint]:
    beatmap_path = Path(beatmap_path)
    section: str | None = None
    timing_points: list[RedTimingPoint] = []

    with beatmap_path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("//"):
                continue

            if line.startswith("[") and line.endswith("]"):
                section = line[1:-1]
                continue

            if section != "TimingPoints":
                continue

            line = line.split("//", 1)[0].strip()
            if not line:
                continue

            try:
                timing_points.append(_parse_timing_line(beatmap_path, line))
            except _NonRedTimingPoint:
                continue

    return sorted(timing_points, key=lambda point: point.offset_ms)


def require_red_timing_points(beatmap_path: str | Path) -> list[RedTimingPoint]:
    timing_points = parse_red_timing_points(beatmap_path)
    if not timing_points:
        raise MissingRedTimingError(f"{beatmap_path} has no red timing point")
    return timing_points


def red_timing_point_at(timing_points: Sequence[RedTimingPoint], time_ms: float) -> RedTimingPoint:
    if not timing_points:
        raise MissingRedTimingError("cannot look up timing with no red timing points")

    sorted_points = sorted(timing_points, key=lambda point: point.offset_ms)
    offsets = [point.offset_ms for point in sorted_points]
    index = bisect_right(offsets, time_ms) - 1
    if index < 0:
        return sorted_points[0]
    return sorted_points[index]


def _parse_timing_line(beatmap_path: Path, line: str) -> RedTimingPoint:
    parts = [part.strip() for part in line.split(",")]
    if len(parts) < 2:
        raise ValueError(f"Malformed timing point in {beatmap_path}: {line}")

    try:
        offset_ms = float(parts[0])
        beat_length_ms = float(parts[1])
        meter = _parse_optional_int(parts, 2, default=4)
        uninherited = _parse_optional_int(parts, 6, default=1)
    except ValueError as exc:
        raise ValueError(f"Malformed timing point in {beatmap_path}: {line}") from exc

    if uninherited == 0:
        raise _NonRedTimingPoint
    if beat_length_ms <= 0:
        raise _NonRedTimingPoint
    if meter <= 0:
        raise ValueError(f"Malformed red timing point in {beatmap_path}: meter must be positive: {line}")

    return RedTimingPoint(offset_ms=offset_ms, beat_length_ms=beat_length_ms, meter=meter)


def _parse_optional_int(parts: Sequence[str], index: int, *, default: int) -> int:
    if index >= len(parts) or parts[index] == "":
        return default
    return int(float(parts[index]))


class _NonRedTimingPoint(Exception):
    pass
