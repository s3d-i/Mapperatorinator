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

from .dense_timing import DenseTimingAuditReport
from .dense_timing import DenseTimingMapInput
from .dense_timing import audit_dense_timing_tracks
from .dense_timing import build_dense_timing_gate_decision
from .token_statistics_artifact import DIFFICULTY_SOURCE


@dataclass(frozen=True)
class DenseTimingAuditProvenance:
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


def build_dense_timing_artifact_payload(
    report: DenseTimingAuditReport,
    *,
    provenance: DenseTimingAuditProvenance,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "audit_name": "dense_timing_track_4k_2to6",
        "report": asdict(report),
        "gate_decision": asdict(build_dense_timing_gate_decision(report)),
        "provenance": asdict(provenance),
    }


def run_dense_timing_artifact(
    *,
    index_path: str | Path,
    dataset_root: str | Path,
    output_json_path: str | Path,
    debug_plot_dir: str | Path | None,
    debug_plot_count: int,
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
    map_inputs: list[DenseTimingMapInput] = []
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
            DenseTimingMapInput(
                beatmap_path=beatmap_path,
                difficulty=float(row.difficulty),
                audio_duration_ms=audio_duration_ms,
                audio_path=audio_path,
            ),
        )

    if duration_failure_count:
        raise RuntimeError(f"failed to read audio duration for {duration_failure_count} eligible maps")

    report = audit_dense_timing_tracks(
        map_inputs,
        debug_plot_dir=debug_plot_dir,
        debug_plot_count=debug_plot_count,
    )
    provenance = DenseTimingAuditProvenance(
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
    payload = build_dense_timing_artifact_payload(report, provenance=provenance)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Stage 1 dense timing track audit artifact.")
    parser.add_argument("--index-path", default="train/artifacts/indexes/beatmap_index_4k.parquet")
    parser.add_argument("--dataset-root", default="mania-dataset")
    parser.add_argument(
        "--output-json",
        default="train/artifacts/reports/audits/dense_timing_track_4k_2to6_2026-04-21.json",
    )
    parser.add_argument(
        "--debug-plot-dir",
        default="train/artifacts/reports/audits/dense_timing_debug_2026-04-21",
    )
    parser.add_argument("--debug-plot-count", type=int, default=8)
    args = parser.parse_args(argv)
    audit_command = (
        "uv run python -m train.stage1_oracle.audits.dense_timing_artifact "
        + " ".join(sys.argv[1:] if argv is None else argv)
    )

    payload = run_dense_timing_artifact(
        index_path=args.index_path,
        dataset_root=args.dataset_root,
        output_json_path=args.output_json,
        debug_plot_dir=args.debug_plot_dir,
        debug_plot_count=args.debug_plot_count,
        audit_command=audit_command,
    )
    report = payload["report"]
    gate_decision = payload["gate_decision"]
    print(f"total_map_count {report['total_map_count']}")
    print(f"audited_map_count {report['audited_map_count']}")
    print(f"gate_status {gate_decision['status']}")
    print(f"renderer_numerics_status {gate_decision['renderer_numerics_status']}")
    print(f"valid_timing_subset_status {gate_decision['valid_timing_subset_status']}")
    print(f"timing_anomaly_status {gate_decision['timing_anomaly_status']}")
    print(f"coverage_status {gate_decision['coverage_status']}")
    print(f"timing_anomaly_map_ratio {gate_decision['timing_anomaly_map_ratio']}")
    print(f"max_timing_anomaly_map_ratio {gate_decision['max_timing_anomaly_map_ratio']}")
    print(f"window_count {report['window_count']}")
    print(f"frame_count {report['frame_count']}")
    print(f"timing_track_nan_count {report['timing_track_nan_count']}")
    print(f"timing_track_inf_count {report['timing_track_inf_count']}")
    print(f"phase_unit_norm_error_mean {report['phase_unit_norm_error_mean']}")
    print(f"phase_unit_norm_error_max {report['phase_unit_norm_error_max']}")
    print(f"beat_pulse_nonzero_ratio {report['beat_pulse_nonzero_ratio']}")
    print(f"local_bpm_log_norm_mean {report['local_bpm_log_norm_mean']}")
    print(f"local_bpm_log_norm_std {report['local_bpm_log_norm_std']}")
    print(f"local_bpm_log_norm_min {report['local_bpm_log_norm_min']}")
    print(f"local_bpm_log_norm_max {report['local_bpm_log_norm_max']}")
    print(f"raw_bpm_min {report['raw_bpm_min']}")
    print(f"raw_bpm_p01 {report['raw_bpm_p01']}")
    print(f"raw_bpm_p50 {report['raw_bpm_p50']}")
    print(f"raw_bpm_p99 {report['raw_bpm_p99']}")
    print(f"raw_bpm_max {report['raw_bpm_max']}")
    print(f"raw_beat_length_min {report['raw_beat_length_min']}")
    print(f"raw_beat_length_max {report['raw_beat_length_max']}")
    print(f"bpm_norm_clipped_low_count {report['bpm_norm_clipped_low_count']}")
    print(f"bpm_norm_clipped_high_count {report['bpm_norm_clipped_high_count']}")
    print(f"bpm_norm_clipped_ratio {report['bpm_norm_clipped_ratio']}")
    print(f"missing_red_timing_map_count {report['missing_red_timing_map_count']}")
    print(f"invalid_red_timing_map_count {report['invalid_red_timing_map_count']}")
    print(f"invalid_red_timing_point_count {report['invalid_red_timing_point_count']}")
    print(f"nonfinite_red_timing_point_count {report['nonfinite_red_timing_point_count']}")
    print(f"nonpositive_red_timing_point_count {report['nonpositive_red_timing_point_count']}")
    print(f"implausible_red_timing_point_count {report['implausible_red_timing_point_count']}")
    print(f"bpm_log_mean {report['bpm_log_mean']}")
    print(f"bpm_log_std {report['bpm_log_std']}")
    print(f"failure_reasons {gate_decision['failure_reasons']}")
    print(f"debug_plot_count {len(report['debug_plot_paths'])}")
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
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    return value


if __name__ == "__main__":
    raise SystemExit(main())
