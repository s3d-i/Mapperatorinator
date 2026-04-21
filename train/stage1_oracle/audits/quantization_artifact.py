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

from .quantization import OsuQuantizationMapInput
from .quantization import QuantizationAuditReport
from .quantization import audit_osu_quantization
from .quantization import build_quantization_gate_decision
from .token_statistics_artifact import DIFFICULTY_SOURCE


EXPECTED_DIFFICULTY_SOURCE = DIFFICULTY_SOURCE
DIRTY_CODE_DIFF_PATHS = ("train/stage1_oracle",)


@dataclass(frozen=True)
class QuantizationAuditProvenance:
    index_path: str
    index_sha256: str
    dataset_root: str
    eligible_map_count: int
    difficulty_source: str
    difficulty_column: str
    code_commit: str
    code_dirty: bool
    dirty_patch_sha256: str | None
    dirty_patch_file_count: int
    audit_command: str


def build_quantization_artifact_payload(
    report: QuantizationAuditReport,
    *,
    provenance: QuantizationAuditProvenance,
    expected_difficulty_source: str = EXPECTED_DIFFICULTY_SOURCE,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "audit_name": "quantization_4k_2to6",
        "report": asdict(report),
        "gate_decision": asdict(
            build_quantization_gate_decision(
                report,
                eligible_map_count=provenance.eligible_map_count,
                difficulty_source=provenance.difficulty_source,
                expected_difficulty_source=expected_difficulty_source,
                code_dirty=provenance.code_dirty,
                dirty_patch_sha256=provenance.dirty_patch_sha256,
            )
        ),
        "provenance": asdict(provenance),
    }


def run_quantization_artifact(
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
    map_inputs = [
        OsuQuantizationMapInput(
            beatmap_path=dataset_root / str(row.shard) / row.beatmap_path,
            difficulty=float(row.difficulty),
        )
        for row in eligible_df.itertuples(index=False)
    ]

    report = audit_osu_quantization(map_inputs)
    code_dirty = _git_dirty()
    provenance = QuantizationAuditProvenance(
        index_path=index_path.as_posix(),
        index_sha256=_sha256_file(index_path),
        dataset_root=dataset_root.as_posix(),
        eligible_map_count=len(eligible_df),
        difficulty_source=DIFFICULTY_SOURCE,
        difficulty_column="difficulty",
        code_commit=_git_rev_parse("HEAD"),
        code_dirty=code_dirty,
        dirty_patch_sha256=_git_dirty_patch_sha256() if code_dirty else None,
        dirty_patch_file_count=_git_dirty_patch_file_count() if code_dirty else 0,
        audit_command=audit_command or " ".join(sys.argv),
    )
    payload = build_quantization_artifact_payload(report, provenance=provenance)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Stage 1 quantization audit artifact.")
    parser.add_argument("--index-path", default="train/artifacts/indexes/beatmap_index_4k.parquet")
    parser.add_argument("--dataset-root", default="mania-dataset")
    parser.add_argument(
        "--output-json",
        default="train/artifacts/reports/audits/quantization_4k_2to6_2026-04-21.json",
    )
    args = parser.parse_args(argv)
    audit_command = (
        "uv run python -m train.stage1_oracle.audits.quantization_artifact "
        + " ".join(sys.argv[1:] if argv is None else argv)
    )

    payload = run_quantization_artifact(
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
    print(f"mean_quantization_error_ms {report['quantization_error_ms']['mean']}")
    print(f"p95_quantization_error_ms {report['quantization_error_ms']['p95']}")
    print(f"max_quantization_error_ms {report['quantization_error_ms']['max']}")
    print(f"zero_length_hold_normalized_count {report['zero_length_hold_normalized_count']}")
    print(
        "post_quantization_collision_lane_time_cell_count "
        f"{report['post_quantization_collision_lane_time_cell_count']}"
    )
    print(f"post_quantization_collision_timepoint_count {report['post_quantization_collision_timepoint_count']}")
    print(
        "post_quantization_collision_affected_map_count "
        f"{report['post_quantization_collision_affected_map_count']}"
    )
    print(f"post_quantization_collision_rate {report['post_quantization_collision_rate']}")
    print(f"reproducibility_status {gate_decision['reproducibility_status']}")
    print(f"dirty_patch_sha256 {payload['provenance']['dirty_patch_sha256']}")
    print(f"output_json {args.output_json}")
    return 0


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
            ["git", "diff", "--quiet", "HEAD", "--", *DIRTY_CODE_DIFF_PATHS],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return True
    return result.returncode != 0


def _git_dirty_patch_sha256() -> str | None:
    try:
        result = subprocess.run(
            ["git", "diff", "--binary", "HEAD", "--", *DIRTY_CODE_DIFF_PATHS],
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    if not result.stdout:
        return None
    return hashlib.sha256(result.stdout).hexdigest()


def _git_dirty_patch_file_count() -> int:
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", "HEAD", "--", *DIRTY_CODE_DIFF_PATHS],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return 0
    return len([line for line in result.stdout.splitlines() if line.strip()])


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


if __name__ == "__main__":
    raise SystemExit(main())
