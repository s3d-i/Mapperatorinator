from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .token_statistics_artifact import DIFFICULTY_SOURCE
from .window_boundary import OsuWindowBoundaryMapInput
from .window_boundary import WindowBoundaryAuditReport
from .window_boundary import audit_osu_window_boundaries
from .window_boundary import build_window_boundary_gate_decision


@dataclass(frozen=True)
class WindowBoundaryAuditProvenance:
    index_path: str
    index_sha256: str
    dataset_root: str
    eligible_map_count: int
    unique_audio_count: int
    difficulty_source: str
    difficulty_column: str
    code_commit: str
    code_dirty: bool
    audit_command: str
    audio_duration_source: str
    audio_duration_failure_count: int


def build_window_boundary_artifact_payload(
    report: WindowBoundaryAuditReport,
    *,
    provenance: WindowBoundaryAuditProvenance,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "audit_name": "window_boundary_4k_2to6",
        "report": asdict(report),
        "gate_decision": asdict(build_window_boundary_gate_decision(report)),
        "provenance": asdict(provenance),
    }


def run_window_boundary_artifact(
    *,
    index_path: str | Path,
    dataset_root: str | Path,
    output_json_path: str | Path,
    audit_command: str | None = None,
) -> dict[str, Any]:
    index_path = Path(index_path)
    dataset_root = Path(dataset_root)
    output_json_path = Path(output_json_path)

    index_df = pd.read_parquet(index_path)
    eligible_df = index_df[(index_df["difficulty"] >= 2.0) & (index_df["difficulty"] <= 6.0)].copy()
    unique_audio_count = eligible_df[["shard", "audio_path"]].drop_duplicates().shape[0]

    duration_cache: dict[Path, float] = {}
    duration_failure_count = 0
    map_inputs: list[OsuWindowBoundaryMapInput] = []
    for row in eligible_df.itertuples(index=False):
        shard = str(row.shard)
        beatmap_path = dataset_root / shard / row.beatmap_path
        audio_path = dataset_root / shard / row.audio_path
        try:
            audio_duration_ms = duration_cache[audio_path]
        except KeyError:
            try:
                audio_duration_ms = _ffprobe_duration_ms(audio_path)
            except (OSError, subprocess.CalledProcessError, ValueError):
                duration_failure_count += 1
                continue
            duration_cache[audio_path] = audio_duration_ms

        map_inputs.append(
            OsuWindowBoundaryMapInput(
                beatmap_path=beatmap_path,
                difficulty=float(row.difficulty),
                audio_duration_ms=audio_duration_ms,
            ),
        )

    if duration_failure_count:
        raise RuntimeError(f"failed to read audio duration for {duration_failure_count} eligible maps")

    report = audit_osu_window_boundaries(map_inputs)
    provenance = WindowBoundaryAuditProvenance(
        index_path=index_path.as_posix(),
        index_sha256=_sha256_file(index_path),
        dataset_root=dataset_root.as_posix(),
        eligible_map_count=len(eligible_df),
        unique_audio_count=unique_audio_count,
        difficulty_source=DIFFICULTY_SOURCE,
        difficulty_column="difficulty",
        code_commit=_git_rev_parse("HEAD"),
        code_dirty=_git_dirty(),
        audit_command=audit_command or " ".join(sys.argv),
        audio_duration_source="ffprobe",
        audio_duration_failure_count=duration_failure_count,
    )
    payload = build_window_boundary_artifact_payload(report, provenance=provenance)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Stage 1 window boundary audit artifact.")
    parser.add_argument("--index-path", default="train/artifacts/indexes/beatmap_index_4k.parquet")
    parser.add_argument("--dataset-root", default="mania-dataset")
    parser.add_argument(
        "--output-json",
        default="train/artifacts/reports/audits/window_boundary_4k_2to6_2026-04-21.json",
    )
    args = parser.parse_args(argv)
    audit_command = (
        "uv run python -m train.stage1_oracle.audits.window_boundary_artifact "
        + " ".join(sys.argv[1:] if argv is None else argv)
    )

    payload = run_window_boundary_artifact(
        index_path=args.index_path,
        dataset_root=args.dataset_root,
        output_json_path=args.output_json,
        audit_command=audit_command,
    )
    report = payload["report"]
    gate_decision = payload["gate_decision"]
    print(f"total_map_count {report['total_map_count']}")
    print(f"audited_map_count {report['audited_map_count']}")
    print(f"gate_status {gate_decision['status']}")
    print(f"boundary_count {report['boundary_count']}")
    print(f"boundary_event_density {report['boundary_event_density']}")
    print(f"hold_crossing_boundary_rate {report['hold_crossing_boundary_rate']}")
    print(f"stitch_duplicate_timepoint_count {report['stitch_duplicate_timepoint_count']}")
    print(f"stitch_collision_timepoint_count {report['stitch_collision_timepoint_count']}")
    print(f"stitch_roundtrip_mismatch_count {report['stitch_roundtrip_mismatch_count']}")
    print(f"output_json {args.output_json}")
    return 0


def _ffprobe_duration_ms(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip()) * 1000.0


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_rev_parse(revision: str) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", revision],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


def _git_dirty() -> bool:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return True
    return bool(result.stdout.strip())


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


if __name__ == "__main__":
    raise SystemExit(main())
