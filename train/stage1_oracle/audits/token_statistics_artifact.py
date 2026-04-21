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

from .token_statistics import OsuTokenStatisticsMapInput
from .token_statistics import TokenStatisticsAuditReport
from .token_statistics import audit_osu_token_statistics
from .token_statistics import build_token_statistics_gate_decision


DIFFICULTY_SOURCE = (
    "train.stage1_oracle.data.dataset.build_4k_index:"
    "calculate_mania_difficulties(speed=1.0)->calculate_mania_difficulty->"
    "compute_mania_star_rating_20241007"
)


@dataclass(frozen=True)
class AuditProvenance:
    index_path: str
    index_sha256: str
    dataset_root: str
    eligible_map_count: int
    unique_audio_count: int
    difficulty_source: str
    difficulty_column: str
    difficulty_code_commit: str
    code_commit: str
    code_dirty: bool
    audit_command: str
    audio_duration_source: str
    audio_duration_failure_count: int


def build_token_statistics_artifact_payload(
    report: TokenStatisticsAuditReport,
    *,
    provenance: AuditProvenance,
    configured_max_decode_len: int,
    empty_window_cap_ratio: float,
) -> dict[str, Any]:
    gate_decision = build_token_statistics_gate_decision(
        report,
        configured_max_decode_len=configured_max_decode_len,
        empty_window_cap_ratio=empty_window_cap_ratio,
    )
    return {
        "schema_version": 1,
        "audit_name": "token_statistics_4k_2to6",
        "report": asdict(report),
        "gate_decision": asdict(gate_decision),
        "provenance": asdict(provenance),
    }


def run_token_statistics_artifact(
    *,
    index_path: str | Path,
    dataset_root: str | Path,
    output_json_path: str | Path,
    configured_max_decode_len: int,
    empty_window_cap_ratio: float,
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
    map_inputs: list[OsuTokenStatisticsMapInput] = []
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
            OsuTokenStatisticsMapInput(
                beatmap_path=beatmap_path,
                difficulty=float(row.difficulty),
                audio_duration_ms=audio_duration_ms,
            ),
        )

    if duration_failure_count:
        raise RuntimeError(f"failed to read audio duration for {duration_failure_count} eligible maps")

    report = audit_osu_token_statistics(map_inputs)
    provenance = AuditProvenance(
        index_path=index_path.as_posix(),
        index_sha256=_sha256_file(index_path),
        dataset_root=dataset_root.as_posix(),
        eligible_map_count=len(eligible_df),
        unique_audio_count=unique_audio_count,
        difficulty_source=DIFFICULTY_SOURCE,
        difficulty_column="difficulty",
        difficulty_code_commit=_git_rev_parse("HEAD"),
        code_commit=_git_rev_parse("HEAD"),
        code_dirty=_git_dirty(),
        audit_command=audit_command or " ".join(sys.argv),
        audio_duration_source="ffprobe",
        audio_duration_failure_count=duration_failure_count,
    )
    payload = build_token_statistics_artifact_payload(
        report,
        provenance=provenance,
        configured_max_decode_len=configured_max_decode_len,
        empty_window_cap_ratio=empty_window_cap_ratio,
    )

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Stage 1 token statistics audit artifact.")
    parser.add_argument("--index-path", default="train/artifacts/indexes/beatmap_index_4k.parquet")
    parser.add_argument("--dataset-root", default="mania-dataset")
    parser.add_argument(
        "--output-json",
        default="train/artifacts/reports/audits/token_statistics_4k_2to6_2026-04-21.json",
    )
    parser.add_argument("--max-decode-len", type=int, default=512)
    parser.add_argument("--empty-window-cap-ratio", type=float, default=0.05)
    args = parser.parse_args(argv)
    audit_command = (
        "uv run python -m train.stage1_oracle.audits.token_statistics_artifact "
        + " ".join(sys.argv[1:] if argv is None else argv)
    )

    payload = run_token_statistics_artifact(
        index_path=args.index_path,
        dataset_root=args.dataset_root,
        output_json_path=args.output_json,
        configured_max_decode_len=args.max_decode_len,
        empty_window_cap_ratio=args.empty_window_cap_ratio,
        audit_command=audit_command,
    )
    report = payload["report"]
    gate_decision = payload["gate_decision"]
    print(f"total_map_count {report['total_map_count']}")
    print(f"audited_map_count {report['audited_map_count']}")
    print(f"gate_status {gate_decision['status']}")
    print(f"observed_max_target_tokens {gate_decision['observed_max_target_tokens']}")
    print(f"configured_max_decode_len {gate_decision['configured_max_decode_len']}")
    print(f"empty_window_cap_ratio {gate_decision['empty_window_cap_ratio']}")
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
