from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from dataclasses import asdict, dataclass
from itertools import cycle
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from ..data.windows import MapsPerBinCap, OracleWindowDataset, collate_oracle_windows
from ..events.canonical import LaneAction
from ..events.grammar import constrained_greedy_decode
from ..events.stitch import DecodedWindowEvents, stitch_decoded_windows
from ..events.tokens import Stage1Vocab
from ..events.tokens import decode_target_tokens
from ..features.timing import DEFAULT_TIMING_TRACK_CONFIG, TIMING_TRACK_CHANNELS, TIMING_TRACK_VERSION
from ..models.mapper import Stage1OracleMapper, Stage1OracleMapperConfig


COARSE_BIN_LABELS = ("2-3", "3-4", "4-5", "5-6")
SYNTHETIC_SMOKE_BPM_LOG_MEAN = 5.160359804030509
SYNTHETIC_SMOKE_BPM_LOG_STD = 0.23196550602850757
REQUIRED_PRETRAINING_GATE_NAMES = (
    "round_trip",
    "dense_timing_debug_plots",
    "stitch_dry_run",
    "event_space",
    "token_statistics",
    "quantization",
    "window_boundary",
    "dense_timing_track",
)
PRETRAINING_GATE_MANIFEST_SCHEMA_VERSION = 2
CHECKPOINT_SCHEMA_VERSION = 1
RUN_CONFIG_KEYS = {
    "dataset_root",
    "index_path",
    "gate_manifest",
    "output_dir",
    "maps_per_bin",
    "max_steps",
    "eval_every",
    "batch_size",
    "learning_rate",
    "dropout",
    "seed",
    "device",
    "run_name",
    "save_every",
    "resume_from",
    "synthetic_smoke",
    "train_manifest",
    "eval_manifest",
    "rollout_probe_manifest",
    "rollout_eval_every",
    "model",
}
MODEL_CONFIG_OVERRIDE_KEYS = ("d_model", "heads", "encoder_layers", "decoder_layers", "ffn_dim")


def select_torch_device(device_name: str = "auto") -> torch.device:
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("requested cuda device is not available")
        return torch.device("cuda")
    if device_name == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError("requested mps device is not available")
        return torch.device("mps")
    raise ValueError(f"unknown device: {device_name}")


def load_run_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML run config: {path}") from exc

    if loaded is None:
        return {"model": {}}
    if not isinstance(loaded, dict):
        raise ValueError(f"run config must be a mapping: {path}")

    config = _normalize_config_mapping(loaded, source_name="run config")
    unknown_keys = sorted(set(config) - RUN_CONFIG_KEYS)
    if unknown_keys:
        raise ValueError(f"unknown run config keys: {unknown_keys}")

    model_config = config.get("model", {})
    if model_config is None:
        model_config = {}
    if not isinstance(model_config, dict):
        raise ValueError("run config model section must be a mapping")
    normalized_model_config = _normalize_config_mapping(model_config, source_name="model config")
    unknown_model_keys = sorted(set(normalized_model_config) - set(MODEL_CONFIG_OVERRIDE_KEYS))
    if unknown_model_keys:
        raise ValueError(f"unknown model config keys: {unknown_model_keys}")

    config["model"] = normalized_model_config
    return config


def _normalize_config_mapping(source: dict[Any, Any], *, source_name: str) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for raw_key, value in source.items():
        if not isinstance(raw_key, str):
            raise ValueError(f"{source_name} keys must be strings")
        key = raw_key.replace("-", "_")
        if key in normalized:
            raise ValueError(f"{source_name} contains duplicate key after normalization: {key}")
        normalized[key] = value
    return normalized


@dataclass(frozen=True)
class OverfitRunResult:
    report_path: Path
    checkpoint_path: Path
    final_loss: float
    final_token_accuracy: float


@dataclass(frozen=True)
class BalancedEpochSamplingPlan:
    indices: list[int]
    empty_window_cap_ratio: float
    empty_window_cap_by_bin: dict[str, float]
    source_window_count_by_bin: dict[str, int]
    source_empty_window_count_by_bin: dict[str, int]
    sample_count_by_bin: dict[str, int]
    empty_sample_count_by_bin: dict[str, int]


@dataclass(frozen=True)
class _DecodeWindowInput:
    write_start_ms: int
    packed_audio: torch.Tensor
    timing_track: torch.Tensor
    difficulty_bucket: torch.Tensor
    condition_ids: list[int]
    oracle_open_hold_mask: int
    write_duration_ms: int
    difficulty_bin_label: str


@dataclass
class _DecodeMetricCounts:
    eos_failures: int = 0
    eos_forced_after_pending_ts: int = 0
    max_decode_len_reached: int = 0
    empty_outputs: int = 0
    time_monotonicity_errors: int = 0
    oracle_invalid_hold_end_count: int = 0
    oracle_same_lane_collision_count: int = 0
    stitched_invalid_hold_end_count: int = 0
    stitched_same_lane_collision_count: int = 0
    stitched_unclosed_hold_count: int = 0
    stitched_generated_event_count: int = 0
    generated_event_count: int = 0
    generated_note_count: int = 0
    reference_note_count: int = 0
    evaluated_window_count: int = 0
    stitched_boundary_errors: int = 0
    stitched_boundary_count: int = 0
    active_stitched_boundary_matches: int = 0
    active_stitched_boundary_count: int = 0
    generated_lengths: list[int] | None = None

    def __post_init__(self) -> None:
        if self.generated_lengths is None:
            self.generated_lengths = []

    @property
    def oracle_invalid_hold_transition_count(self) -> int:
        return self.oracle_invalid_hold_end_count + self.oracle_same_lane_collision_count

    @property
    def stitched_invalid_hold_transition_count(self) -> int:
        return self.stitched_invalid_hold_end_count + self.stitched_same_lane_collision_count


def _rollout_progress_totals(loader: Any) -> tuple[int | None, int | None]:
    dataset = getattr(loader, "dataset", None)
    if dataset is not None:
        return len(dataset), _rollout_total_map_count_from_dataset(dataset)
    if isinstance(loader, Sequence):
        return _rollout_progress_totals_from_batches(loader)
    return None, None


def _rollout_total_map_count_from_dataset(dataset: Any) -> int | None:
    filter_report = getattr(dataset, "filter_report", None)
    retained_map_count = getattr(filter_report, "retained_map_count", None)
    if isinstance(retained_map_count, int):
        return retained_map_count

    records = getattr(dataset, "records", None)
    if isinstance(records, Sequence):
        return _count_rollout_map_keys(records)
    if isinstance(dataset, (list, tuple)):
        return _count_rollout_map_keys(dataset)
    return None


def _rollout_progress_totals_from_batches(batches: Sequence[Any]) -> tuple[int | None, int | None]:
    total_window_count = 0
    saw_window_count = False
    map_keys: set[tuple[str, str]] = set()
    for batch in batches:
        batch_window_count = _rollout_batch_window_count(batch)
        if batch_window_count is not None:
            total_window_count += batch_window_count
            saw_window_count = True
        map_keys.update(_rollout_batch_map_keys(batch))
    return (
        total_window_count if saw_window_count else None,
        len(map_keys) if map_keys else None,
    )


def _rollout_batch_window_count(batch: Any) -> int | None:
    if not isinstance(batch, Mapping):
        return None
    decoder_input_ids = batch.get("decoder_input_ids")
    if isinstance(decoder_input_ids, torch.Tensor):
        return int(decoder_input_ids.shape[0])
    metadata = batch.get("metadata")
    if isinstance(metadata, Sequence):
        return len(metadata)
    return None


def _rollout_batch_map_keys(batch: Any) -> set[tuple[str, str]]:
    if not isinstance(batch, Mapping):
        return set()
    metadata = batch.get("metadata")
    if not isinstance(metadata, Sequence):
        return set()
    return {
        map_key
        for item in metadata
        if (map_key := _rollout_map_key(item)) is not None
    }


def _count_rollout_map_keys(entries: Sequence[Any]) -> int | None:
    map_keys = {
        map_key
        for entry in entries
        if (map_key := _rollout_map_key(entry)) is not None
    }
    if not map_keys:
        return None
    return len(map_keys)


def _rollout_map_key(entry: Any) -> tuple[str, str] | None:
    try:
        beatmap_path = _record_value(entry, "beatmap_path")
    except (AttributeError, KeyError, TypeError):
        return None
    try:
        audio_path = _record_value(entry, "audio_path")
    except (AttributeError, KeyError, TypeError):
        audio_path = ""
    return str(beatmap_path), str(audio_path)


def _format_rollout_progress_count(index: int, total: int | None) -> str:
    if total is None:
        return f"{index}/?"
    return f"{index}/{total}"


def _print_rollout_progress(
    *,
    pass_name: str,
    window_index: int,
    total_window_count: int | None,
    map_index: int,
    total_map_count: int | None,
    avg_generated_len: float,
    status: str | None = None,
) -> None:
    status_fragment = f" status={status}" if status is not None else ""
    print(
        f"rollout_progress pass={pass_name}{status_fragment} "
        f"window={_format_rollout_progress_count(window_index, total_window_count)} "
        f"map={_format_rollout_progress_count(map_index, total_map_count)} "
        f"avg_generated_len={avg_generated_len:.2f}",
        flush=True,
    )


class PretrainingGateValidationError(ValueError):
    pass


def build_timing_track_metadata(*, bpm_log_mean: float, bpm_log_std: float) -> dict[str, Any]:
    return {
        "timing_track_version": TIMING_TRACK_VERSION,
        "timing_frame_hop_ms": DEFAULT_TIMING_TRACK_CONFIG.frame_hop_ms,
        "timing_frame_center_offset_ms": DEFAULT_TIMING_TRACK_CONFIG.frame_center_offset_ms,
        "timing_channels": list(TIMING_TRACK_CHANNELS),
        "pulse_shape": "triangular",
        "pulse_width_ms": DEFAULT_TIMING_TRACK_CONFIG.pulse_width_ms,
        "bpm_log_mean": float(bpm_log_mean),
        "bpm_log_std": float(bpm_log_std),
        "red_timing_source": "reference_osu_red_points_for_oracle_phase",
    }


def validate_pretraining_gate_manifest(
    manifest_path: str | Path,
    *,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    root = Path(repo_root) if repo_root is not None else _repo_root()
    resolved_manifest_path = _resolve_manifest_path(manifest_path, root)
    if not resolved_manifest_path.is_file():
        raise PretrainingGateValidationError(f"pretraining gate manifest not found: {resolved_manifest_path}")

    try:
        manifest = json.loads(resolved_manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PretrainingGateValidationError(f"invalid pretraining gate manifest JSON: {resolved_manifest_path}") from exc

    if manifest.get("schema_version") != PRETRAINING_GATE_MANIFEST_SCHEMA_VERSION:
        raise PretrainingGateValidationError(
            f"pretraining gate manifest schema_version must be {PRETRAINING_GATE_MANIFEST_SCHEMA_VERSION}",
        )
    training = _extract_training_config(manifest.get("training"))
    gates = manifest.get("gates")
    if not isinstance(gates, dict):
        raise PretrainingGateValidationError("pretraining gate manifest must contain a gates object")

    resolved_gates: dict[str, Any] = {}
    for gate_name in REQUIRED_PRETRAINING_GATE_NAMES:
        gate = gates.get(gate_name)
        if not isinstance(gate, dict):
            raise PretrainingGateValidationError(f"pretraining gate missing or invalid: {gate_name}")
        resolved_gates[gate_name] = _validate_pretraining_gate(
            gate_name,
            gate,
            repo_root=root,
        )
    training = _training_config_checked_against_token_statistics(
        training,
        resolved_gates["token_statistics"],
    )

    return {
        "status": "PASS",
        "schema_version": PRETRAINING_GATE_MANIFEST_SCHEMA_VERSION,
        "manifest_path": resolved_manifest_path.as_posix(),
        "training": training,
        "gates": resolved_gates,
    }


def _validate_pretraining_gate(
    gate_name: str,
    gate: dict[str, Any],
    *,
    repo_root: Path,
) -> dict[str, Any]:
    artifact_path_value = gate.get("artifact_path")
    if not isinstance(artifact_path_value, str) or not artifact_path_value:
        raise PretrainingGateValidationError(f"{gate_name} gate requires artifact_path")

    artifact_path = _resolve_manifest_path(Path(artifact_path_value), repo_root)
    if not artifact_path.is_file():
        raise PretrainingGateValidationError(f"pretraining gate artifact not found for {gate_name}: {artifact_path}")
    try:
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PretrainingGateValidationError(
            f"invalid pretraining gate artifact JSON for {gate_name}: {artifact_path}",
        ) from exc
    gate_decision = artifact.get("gate_decision")
    if not isinstance(gate_decision, dict) or gate_decision.get("status") != "PASS":
        raise PretrainingGateValidationError(f"pretraining gate artifact did not PASS: {gate_name}")

    resolved: dict[str, Any] = {
        "status": "PASS",
        "source": "artifact",
        "artifact_path": artifact_path.as_posix(),
    }
    if gate_name == "dense_timing_track":
        resolved.update(_extract_dense_timing_training_stats(artifact))
    if gate_name == "token_statistics":
        resolved.update(_extract_token_statistics_training_config(artifact))

    if gate_name == "dense_timing_debug_plots":
        debug_plot_dir = gate.get("debug_plot_dir")
        if not isinstance(debug_plot_dir, str) or not debug_plot_dir:
            raise PretrainingGateValidationError("dense_timing_debug_plots gate requires debug_plot_dir")
        resolved_debug_plot_dir = _resolve_manifest_path(Path(debug_plot_dir), repo_root)
        debug_plot_paths = sorted(resolved_debug_plot_dir.glob("*.png")) if resolved_debug_plot_dir.is_dir() else []
        if not debug_plot_paths:
            raise PretrainingGateValidationError(
                f"dense_timing_debug_plots gate has no debug .png files: {resolved_debug_plot_dir}",
            )
        resolved["debug_plot_dir"] = resolved_debug_plot_dir.as_posix()
        resolved["debug_plot_count"] = len(debug_plot_paths)

    return resolved


def _extract_token_statistics_training_config(artifact: dict[str, Any]) -> dict[str, Any]:
    gate_decision = artifact.get("gate_decision")
    if not isinstance(gate_decision, dict):
        raise PretrainingGateValidationError("token_statistics artifact requires gate_decision")

    extracted: dict[str, Any] = {}
    if "configured_max_decode_len" in gate_decision:
        extracted["configured_max_decode_len"] = _positive_int(
            gate_decision["configured_max_decode_len"],
            "token_statistics.configured_max_decode_len",
        )
    if "empty_window_cap_ratio" in gate_decision:
        extracted["empty_window_cap_ratio"] = _ratio_float(
            gate_decision["empty_window_cap_ratio"],
            "token_statistics.empty_window_cap_ratio",
        )
    if "empty_window_cap_by_bin" in gate_decision:
        extracted["empty_window_cap_by_bin"] = _empty_window_cap_by_bin(
            gate_decision["empty_window_cap_by_bin"],
            default_ratio=extracted.get("empty_window_cap_ratio"),
            source_name="token_statistics.empty_window_cap_by_bin",
        )
    return extracted


def _extract_training_config(source: object) -> dict[str, Any]:
    if not isinstance(source, dict):
        raise PretrainingGateValidationError("pretraining gate manifest requires a training object")
    if "max_decode_len" not in source:
        raise PretrainingGateValidationError("training config requires max_decode_len")
    if "empty_window_cap_ratio" not in source:
        raise PretrainingGateValidationError("training config requires empty_window_cap_ratio")

    max_decode_len = _positive_int(source["max_decode_len"], "max_decode_len")
    empty_window_cap_ratio = _ratio_float(source["empty_window_cap_ratio"], "empty_window_cap_ratio")
    empty_window_cap_by_bin = _empty_window_cap_by_bin(
        source.get("empty_window_cap_by_bin"),
        default_ratio=empty_window_cap_ratio,
        source_name="empty_window_cap_by_bin",
    )

    return {
        "max_decode_len": max_decode_len,
        "empty_window_cap_ratio": empty_window_cap_ratio,
        "empty_window_cap_by_bin": empty_window_cap_by_bin,
    }


def _extract_dense_timing_training_stats(artifact: dict[str, Any]) -> dict[str, float]:
    for source_key in ("gate_decision", "report"):
        source = artifact.get(source_key)
        if not isinstance(source, dict):
            continue
        if "bpm_log_mean" not in source or "bpm_log_std" not in source:
            continue
        bpm_log_mean = _finite_float(source["bpm_log_mean"], "bpm_log_mean")
        bpm_log_std = _finite_float(source["bpm_log_std"], "bpm_log_std")
        if bpm_log_std <= 0:
            raise PretrainingGateValidationError("dense_timing_track bpm_log_std must be positive")
        return {
            "bpm_log_mean": bpm_log_mean,
            "bpm_log_std": bpm_log_std,
        }
    raise PretrainingGateValidationError(
        "dense_timing_track artifact must contain bpm_log_mean and bpm_log_std",
    )


def timing_training_stats_from_pretraining_gates(pretraining_gates: dict[str, Any]) -> dict[str, float]:
    gates = pretraining_gates.get("gates")
    if not isinstance(gates, dict):
        raise PretrainingGateValidationError("pretraining gate result must contain gates")
    dense_timing_gate = gates.get("dense_timing_track")
    if not isinstance(dense_timing_gate, dict):
        raise PretrainingGateValidationError("pretraining gate result missing dense_timing_track")
    if "bpm_log_mean" not in dense_timing_gate or "bpm_log_std" not in dense_timing_gate:
        raise PretrainingGateValidationError(
            "dense_timing_track gate must provide bpm_log_mean and bpm_log_std from its audit artifact",
        )
    bpm_log_mean = _finite_float(dense_timing_gate["bpm_log_mean"], "bpm_log_mean")
    bpm_log_std = _finite_float(dense_timing_gate["bpm_log_std"], "bpm_log_std")
    if bpm_log_std <= 0:
        raise PretrainingGateValidationError("dense_timing_track bpm_log_std must be positive")
    return {
        "bpm_log_mean": bpm_log_mean,
        "bpm_log_std": bpm_log_std,
    }


def training_config_from_pretraining_gates(pretraining_gates: dict[str, Any]) -> dict[str, Any]:
    training = _extract_training_config(pretraining_gates.get("training"))
    gates = pretraining_gates.get("gates")
    if isinstance(gates, dict) and isinstance(gates.get("token_statistics"), dict):
        return _training_config_checked_against_token_statistics(training, gates["token_statistics"])
    return training


def _training_config_checked_against_token_statistics(
    training: dict[str, Any],
    token_statistics_gate: dict[str, Any],
) -> dict[str, Any]:
    checked = dict(training)
    token_max_decode_len = token_statistics_gate.get("configured_max_decode_len")
    if token_max_decode_len is not None and checked["max_decode_len"] < int(token_max_decode_len):
        raise PretrainingGateValidationError(
            "token_statistics max_decode_len exceeds training max_decode_len: "
            f"{token_max_decode_len} > {checked['max_decode_len']}",
        )

    token_empty_ratio = token_statistics_gate.get("empty_window_cap_ratio")
    if token_empty_ratio is not None and not math.isclose(
        checked["empty_window_cap_ratio"],
        float(token_empty_ratio),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise PretrainingGateValidationError(
            "token_statistics empty_window_cap_ratio differs from training config: "
            f"{token_empty_ratio} != {checked['empty_window_cap_ratio']}",
        )

    token_empty_by_bin = token_statistics_gate.get("empty_window_cap_by_bin")
    if isinstance(token_empty_by_bin, dict):
        checked["empty_window_cap_by_bin"] = dict(token_empty_by_bin)
    return checked


def _finite_float(value: object, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise PretrainingGateValidationError(f"{name} must be numeric") from exc
    if not math.isfinite(parsed):
        raise PretrainingGateValidationError(f"{name} must be finite")
    return parsed


def _positive_int(value: object, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise PretrainingGateValidationError(f"{name} must be an integer") from exc
    if parsed <= 0:
        raise PretrainingGateValidationError(f"{name} must be positive")
    return parsed


def _ratio_float(value: object, name: str) -> float:
    parsed = _finite_float(value, name)
    if not 0.0 <= parsed <= 1.0:
        raise PretrainingGateValidationError(f"{name} must be within [0, 1]")
    return parsed


def _empty_window_cap_by_bin(
    source: object,
    *,
    default_ratio: float | None,
    source_name: str,
) -> dict[str, float]:
    if not isinstance(source, dict):
        if default_ratio is None:
            raise PretrainingGateValidationError(f"{source_name} must be an object")
        return {label: default_ratio for label in COARSE_BIN_LABELS}
    if default_ratio is None:
        missing = [label for label in COARSE_BIN_LABELS if label not in source]
        if missing:
            raise PretrainingGateValidationError(f"{source_name} missing bins: {missing}")
    return {
        label: _ratio_float(source.get(label, default_ratio), f"{source_name}[{label}]")
        for label in COARSE_BIN_LABELS
    }


def _resolve_manifest_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def summarize_overfit_coverage(records: Sequence[Any]) -> dict[str, Any]:
    map_paths_by_bin = {label: set() for label in COARSE_BIN_LABELS}
    window_count_by_bin = {label: 0 for label in COARSE_BIN_LABELS}

    for record in records:
        difficulty = float(_record_value(record, "difficulty"))
        label = _difficulty_bin_label(difficulty)
        if label is None:
            continue
        map_paths_by_bin[label].add(str(_record_value(record, "beatmap_path")))
        window_count_by_bin[label] += 1

    map_count_by_bin = {label: len(map_paths_by_bin[label]) for label in COARSE_BIN_LABELS}
    all_map_paths = set().union(*map_paths_by_bin.values()) if map_paths_by_bin else set()
    return {
        "unique_map_count": len(all_map_paths),
        "map_count_by_bin": map_count_by_bin,
        "window_count_by_bin": window_count_by_bin,
        "missing_bins": [label for label in COARSE_BIN_LABELS if map_count_by_bin[label] == 0],
    }


def build_balanced_epoch_sampling_plan(
    records: Sequence[Any],
    *,
    empty_window_cap_ratio: float,
    empty_window_cap_by_bin: dict[str, float] | None = None,
    seed: int,
) -> BalancedEpochSamplingPlan:
    if not 0.0 <= empty_window_cap_ratio <= 1.0:
        raise ValueError(f"empty_window_cap_ratio must be within [0, 1]: {empty_window_cap_ratio}")
    resolved_empty_window_cap_by_bin = _empty_window_cap_by_bin_for_sampling(
        empty_window_cap_by_bin,
        default_ratio=empty_window_cap_ratio,
    )

    rng = random.Random(seed)
    non_empty_by_bin = {label: [] for label in COARSE_BIN_LABELS}
    empty_by_bin = {label: [] for label in COARSE_BIN_LABELS}

    for index, record in enumerate(records):
        label = _difficulty_bin_label(float(_record_value(record, "difficulty")))
        if label is None:
            continue
        if _record_is_empty_window(record):
            empty_by_bin[label].append(index)
        else:
            non_empty_by_bin[label].append(index)

    source_window_count_by_bin = {
        label: len(non_empty_by_bin[label]) + len(empty_by_bin[label])
        for label in COARSE_BIN_LABELS
    }
    source_empty_window_count_by_bin = {label: len(empty_by_bin[label]) for label in COARSE_BIN_LABELS}
    capped_size_by_bin = {
        label: len(non_empty_by_bin[label])
        + min(
            len(empty_by_bin[label]),
            _max_empty_windows_for_non_empty_count(
                len(non_empty_by_bin[label]),
                resolved_empty_window_cap_by_bin[label],
            ),
        )
        for label in COARSE_BIN_LABELS
        if source_window_count_by_bin[label] > 0
    }
    target_per_bin = max(capped_size_by_bin.values(), default=0)

    sampled_indices: list[int] = []
    sample_count_by_bin = {label: 0 for label in COARSE_BIN_LABELS}
    empty_sample_count_by_bin = {label: 0 for label in COARSE_BIN_LABELS}
    for label in COARSE_BIN_LABELS:
        if source_window_count_by_bin[label] == 0 or target_per_bin == 0:
            continue

        non_empty = list(non_empty_by_bin[label])
        empty = list(empty_by_bin[label])
        rng.shuffle(non_empty)
        rng.shuffle(empty)

        if non_empty:
            max_empty_for_target = min(
                len(empty),
                math.floor(target_per_bin * resolved_empty_window_cap_by_bin[label]),
            )
            selected_empty = empty[:max_empty_for_target]
            selected_non_empty = _sample_with_replacement(
                non_empty,
                target_per_bin - len(selected_empty),
                rng,
            )
            selected = selected_non_empty + selected_empty
        else:
            selected = _sample_with_replacement(empty, target_per_bin, rng)
            selected_empty = selected

        rng.shuffle(selected)
        sampled_indices.extend(selected)
        sample_count_by_bin[label] = len(selected)
        empty_sample_count_by_bin[label] = sum(1 for index in selected if _record_is_empty_window(records[index]))

    rng.shuffle(sampled_indices)
    return BalancedEpochSamplingPlan(
        indices=sampled_indices,
        empty_window_cap_ratio=empty_window_cap_ratio,
        empty_window_cap_by_bin=resolved_empty_window_cap_by_bin,
        source_window_count_by_bin=source_window_count_by_bin,
        source_empty_window_count_by_bin=source_empty_window_count_by_bin,
        sample_count_by_bin=sample_count_by_bin,
        empty_sample_count_by_bin=empty_sample_count_by_bin,
    )


def run_synthetic_smoke(
    *,
    output_dir: Path,
    max_steps: int = 2,
    save_every: int | None = None,
    seed: int = 1337,
    device_name: str = "auto",
    resume_from: Path | None = None,
) -> OverfitRunResult:
    torch.manual_seed(seed)
    vocab = Stage1Vocab()
    samples = _synthetic_samples(vocab)
    loader = DataLoader(
        samples,
        batch_size=2,
        shuffle=False,
        collate_fn=lambda batch: collate_oracle_windows(batch, pad_id=vocab.pad_id),
    )
    config = Stage1OracleMapperConfig(
        vocab_size=vocab.size,
        d_model=32,
        heads=4,
        encoder_layers=1,
        decoder_layers=1,
        ffn_dim=64,
        dropout=0.0,
        max_decode_len=16,
    )
    return _run_training(
        loader=loader,
        output_dir=output_dir,
        config=config,
        max_steps=max_steps,
        eval_every=max(1, max_steps),
        save_every=save_every,
        learning_rate=1e-2,
        seed=seed,
        device_name=device_name,
        run_name="synthetic_smoke",
        vocab=vocab,
        train_eval_loader=loader,
        eval_loader=loader,
        rollout_probe_loader=loader,
        rollout_eval_every=max(1, max_steps),
        timing_track=build_timing_track_metadata(
            bpm_log_mean=SYNTHETIC_SMOKE_BPM_LOG_MEAN,
            bpm_log_std=SYNTHETIC_SMOKE_BPM_LOG_STD,
        ),
        overfit_coverage=None,
        pretraining_gates={
            "status": "SKIPPED_SYNTHETIC_SMOKE",
            "reason": "synthetic smoke does not train on the Stage 1 corpus",
        },
        dataset_filter_report=None,
        training_sampling=None,
        resume_from=resume_from,
    )


def run_overfit_32(
    *,
    dataset_root: Path,
    index_path: Path | None,
    gate_manifest_path: Path,
    output_dir: Path,
    maps_per_bin: MapsPerBinCap = 8,
    max_steps: int = 5000,
    eval_every: int = 100,
    batch_size: int = 8,
    learning_rate: float = 3e-4,
    dropout: float = 0.1,
    seed: int = 1337,
    device_name: str = "auto",
    run_name: str = "overfit_32",
    save_every: int | None = None,
    resume_from: Path | None = None,
    model_config_overrides: dict[str, int] | None = None,
    train_manifest_path: Path | None = None,
    eval_manifest_path: Path | None = None,
    rollout_probe_manifest_path: Path | None = None,
    rollout_eval_every: int | None = None,
) -> OverfitRunResult:
    maps_per_bin = _validate_maps_per_bin_cap(maps_per_bin)
    if dropout < 0.0:
        raise ValueError(f"dropout must be non-negative, got {dropout}")
    if rollout_eval_every is not None and rollout_eval_every <= 0:
        raise ValueError(f"rollout_eval_every must be positive when set, got {rollout_eval_every}")
    torch.manual_seed(seed)
    vocab = Stage1Vocab()
    print("gate_progress status=validating", flush=True)
    pretraining_gates = validate_pretraining_gate_manifest(gate_manifest_path)
    print("gate_progress status=pass", flush=True)
    timing_stats = timing_training_stats_from_pretraining_gates(pretraining_gates)
    training_config = training_config_from_pretraining_gates(pretraining_gates)
    train_dataset = _build_oracle_window_dataset(
        source_name="train",
        dataset_root=dataset_root,
        index_path=index_path,
        manifest_path=train_manifest_path,
        vocab=vocab,
        timing_stats=timing_stats,
        max_maps_per_bin=maps_per_bin,
    )
    if len(train_dataset) == 0:
        raise ValueError("train OracleWindowDataset produced no windows")

    if eval_manifest_path is None:
        eval_dataset = train_dataset
    else:
        eval_dataset = _build_oracle_window_dataset(
            source_name="eval",
            dataset_root=dataset_root,
            index_path=index_path,
            manifest_path=eval_manifest_path,
            vocab=vocab,
            timing_stats=timing_stats,
            max_maps_per_bin=None,
        )
        if len(eval_dataset) == 0:
            raise ValueError("eval OracleWindowDataset produced no windows")

    rollout_probe_dataset = None
    if rollout_probe_manifest_path is not None:
        rollout_probe_dataset = _build_oracle_window_dataset(
            source_name="rollout_probe",
            dataset_root=dataset_root,
            index_path=index_path,
            manifest_path=rollout_probe_manifest_path,
            vocab=vocab,
            timing_stats=timing_stats,
            max_maps_per_bin=None,
        )
        if len(rollout_probe_dataset) == 0:
            raise ValueError("rollout probe OracleWindowDataset produced no windows")

    overfit_coverage = summarize_overfit_coverage(train_dataset.records)
    sampling_plan = build_balanced_epoch_sampling_plan(
        train_dataset.records,
        empty_window_cap_ratio=training_config["empty_window_cap_ratio"],
        empty_window_cap_by_bin=training_config["empty_window_cap_by_bin"],
        seed=seed,
    )
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampling_plan.indices,
        collate_fn=lambda batch: collate_oracle_windows(batch, pad_id=vocab.pad_id),
    )
    train_eval_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_oracle_windows(batch, pad_id=vocab.pad_id),
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_oracle_windows(batch, pad_id=vocab.pad_id),
    )
    rollout_probe_loader = (
        DataLoader(
            rollout_probe_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda batch: collate_oracle_windows(batch, pad_id=vocab.pad_id),
        )
        if rollout_probe_dataset is not None
        else None
    )
    model_config_overrides = _validate_model_config_overrides(model_config_overrides)
    return _run_training(
        loader=loader,
        output_dir=output_dir,
        config=Stage1OracleMapperConfig(
            vocab_size=vocab.size,
            max_decode_len=training_config["max_decode_len"],
            dropout=dropout,
            **model_config_overrides,
        ),
        max_steps=max_steps,
        eval_every=eval_every,
        save_every=save_every,
        learning_rate=learning_rate,
        seed=seed,
        device_name=device_name,
        run_name=run_name,
        vocab=vocab,
        train_eval_loader=train_eval_loader,
        eval_loader=eval_loader,
        rollout_probe_loader=rollout_probe_loader,
        rollout_eval_every=rollout_eval_every,
        timing_track=build_timing_track_metadata(
            bpm_log_mean=train_dataset.bpm_log_mean,
            bpm_log_std=train_dataset.bpm_log_std,
        ),
        overfit_coverage=overfit_coverage,
        pretraining_gates=pretraining_gates,
        dataset_filter_report=asdict(train_dataset.filter_report),
        training_sampling=_sampling_plan_report(sampling_plan),
        resume_from=resume_from,
    )


def _build_oracle_window_dataset(
    *,
    source_name: str,
    dataset_root: Path,
    index_path: Path | None,
    manifest_path: Path | None,
    vocab: Stage1Vocab,
    timing_stats: dict[str, float],
    max_maps_per_bin: MapsPerBinCap | None,
) -> OracleWindowDataset:
    print(f"dataset_progress phase=build_windows status=start source={source_name}", flush=True)
    dataset = OracleWindowDataset(
        dataset_root=dataset_root,
        index_path=index_path,
        manifest_path=manifest_path,
        vocab=vocab,
        bpm_log_mean=timing_stats["bpm_log_mean"],
        bpm_log_std=timing_stats["bpm_log_std"],
        max_maps_per_bin=max_maps_per_bin,
        progress=True,
    )
    print(
        f"dataset_progress phase=build_windows status=done source={source_name} "
        f"retained_maps={dataset.filter_report.retained_map_count} windows={len(dataset)}",
        flush=True,
    )
    return dataset


def _validate_maps_per_bin_cap(maps_per_bin: MapsPerBinCap) -> MapsPerBinCap:
    if isinstance(maps_per_bin, int):
        if maps_per_bin <= 0:
            raise ValueError(f"maps_per_bin must be positive, got {maps_per_bin}")
        return maps_per_bin
    if not isinstance(maps_per_bin, dict):
        raise ValueError(f"maps_per_bin must be an integer or per-bin mapping, got {maps_per_bin}")

    missing = [label for label in COARSE_BIN_LABELS if label not in maps_per_bin]
    unknown = sorted(set(maps_per_bin) - set(COARSE_BIN_LABELS))
    if missing:
        raise ValueError(f"maps_per_bin missing bins: {missing}")
    if unknown:
        raise ValueError(f"maps_per_bin unknown bins: {unknown}")

    normalized: dict[str, int] = {}
    for label in COARSE_BIN_LABELS:
        value = int(maps_per_bin[label])
        if value <= 0:
            raise ValueError(f"maps_per_bin[{label}] must be positive, got {maps_per_bin[label]}")
        normalized[label] = value
    return normalized


def _validate_model_config_overrides(overrides: dict[str, int] | None) -> dict[str, int]:
    if overrides is None:
        return {}
    unknown_keys = sorted(set(overrides) - set(MODEL_CONFIG_OVERRIDE_KEYS))
    if unknown_keys:
        raise ValueError(f"unknown model config override keys: {unknown_keys}")

    validated: dict[str, int] = {}
    for key in MODEL_CONFIG_OVERRIDE_KEYS:
        if key not in overrides or overrides[key] is None:
            continue
        value = int(overrides[key])
        if value <= 0:
            raise ValueError(f"model config override {key} must be positive, got {overrides[key]}")
        validated[key] = value

    default_config = Stage1OracleMapperConfig()
    d_model = validated.get("d_model", default_config.d_model)
    heads = validated.get("heads", default_config.heads)
    if d_model % heads != 0:
        raise ValueError(f"d_model must be divisible by heads, got d_model={d_model} heads={heads}")
    return validated


def _resolve_save_every(save_every: int | None, eval_every: int) -> int:
    if eval_every <= 0:
        raise ValueError(f"eval_every must be positive, got {eval_every}")
    if save_every is None:
        return eval_every
    resolved = int(save_every)
    if resolved <= 0:
        raise ValueError(f"save_every must be positive, got {save_every}")
    return resolved


def _should_write_checkpoint(*, step: int, max_steps: int, save_every: int) -> bool:
    return step == 1 or step % save_every == 0 or step == max_steps


def _capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python_random": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if (
        hasattr(torch, "mps")
        and torch.backends.mps.is_available()
        and hasattr(torch.mps, "get_rng_state")
    ):
        try:
            state["mps"] = torch.mps.get_rng_state()
        except RuntimeError:
            pass
    return state


def _restore_rng_state(raw_state: object) -> None:
    if not isinstance(raw_state, Mapping):
        raise ValueError("resume checkpoint training_state.rng_state must be a mapping")

    python_state = raw_state.get("python_random")
    if python_state is not None:
        random.setstate(python_state)

    torch_state = raw_state.get("torch")
    if torch_state is not None:
        if not isinstance(torch_state, torch.Tensor):
            raise ValueError("resume checkpoint torch RNG state must be a tensor")
        torch.set_rng_state(torch_state.cpu())

    cuda_state = raw_state.get("cuda")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)

    mps_state = raw_state.get("mps")
    if (
        mps_state is not None
        and hasattr(torch, "mps")
        and torch.backends.mps.is_available()
        and hasattr(torch.mps, "set_rng_state")
    ):
        if not isinstance(mps_state, torch.Tensor):
            raise ValueError("resume checkpoint mps RNG state must be a tensor")
        torch.mps.set_rng_state(mps_state.cpu())


def _load_resume_checkpoint(
    resume_from: Path,
    *,
    expected_config: Stage1OracleMapperConfig,
) -> dict[str, Any]:
    checkpoint = torch.load(resume_from, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"resume checkpoint must contain a mapping: {resume_from}")
    raw_config = checkpoint.get("config")
    if not isinstance(raw_config, Mapping):
        raise ValueError(f"resume checkpoint missing config: {resume_from}")
    loaded_config = Stage1OracleMapperConfig(**dict(raw_config))
    if loaded_config != expected_config:
        raise ValueError("resume checkpoint config does not match the requested run config")
    if "model_state_dict" not in checkpoint:
        raise ValueError(f"resume checkpoint missing model_state_dict: {resume_from}")
    if "optimizer_state_dict" not in checkpoint:
        raise ValueError("resume checkpoint missing optimizer_state_dict; old inference-only checkpoints cannot resume")
    training_state = checkpoint.get("training_state")
    if not isinstance(training_state, Mapping):
        raise ValueError("resume checkpoint missing training_state")
    step = training_state.get("step")
    if not isinstance(step, int) or step < 0:
        raise ValueError("resume checkpoint training_state.step must be a non-negative integer")
    if "rng_state" not in training_state:
        raise ValueError("resume checkpoint missing training_state.rng_state")
    history = checkpoint.get("history")
    if not isinstance(history, list):
        raise ValueError("resume checkpoint history must be a list")
    return checkpoint


def _move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _advance_training_iterator(iterator: Any, loader: DataLoader, completed_step: int) -> Any:
    batches_per_epoch = len(loader)
    if batches_per_epoch <= 0:
        return iterator
    for _ in range(completed_step % batches_per_epoch):
        next(iterator)
    return iterator


def _checkpoint_metric_group(training_state: Mapping[str, Any], key: str) -> dict[str, Any]:
    raw_metrics = training_state.get(key)
    if isinstance(raw_metrics, Mapping):
        return dict(raw_metrics)
    return {}


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    torch.save(dict(payload), tmp_path)
    tmp_path.replace(path)


def _copy_file_atomically(source_path: Path, destination_path: Path) -> None:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination_path.with_name(f".{destination_path.name}.tmp")
    shutil.copy2(source_path, tmp_path)
    tmp_path.replace(destination_path)


def _write_report(report_path: Path, payload: Mapping[str, Any]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = report_path.with_name(f".{report_path.name}.tmp")
    tmp_path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    tmp_path.replace(report_path)


def _training_report_payload(
    *,
    run_name: str,
    seed: int,
    max_steps: int,
    completed_steps: int,
    eval_every: int,
    save_every: int,
    learning_rate: float,
    config: Stage1OracleMapperConfig,
    device: torch.device,
    parameter_count: int,
    timing_track: dict[str, Any],
    pretraining_gates: dict[str, Any],
    dataset_filter_report: dict[str, Any] | None,
    training_sampling: dict[str, Any] | None,
    overfit_coverage: dict[str, Any] | None,
    history: list[dict[str, Any]],
    final_train_teacher_forced: dict[str, Any],
    final_val_teacher_forced: dict[str, Any],
    final_rollout_probe: dict[str, Any],
    resume_from: Path | None,
) -> dict[str, Any]:
    return {
        "run_name": run_name,
        "seed": seed,
        "max_steps": max_steps,
        "completed_steps": completed_steps,
        "is_complete": completed_steps >= max_steps,
        "eval_every": eval_every,
        "save_every": save_every,
        "learning_rate": learning_rate,
        "config": asdict(config),
        "device": str(device),
        "parameter_count": parameter_count,
        "timing_track": timing_track,
        "pretraining_gates": pretraining_gates,
        "dataset_filter_report": dataset_filter_report,
        "training_sampling": training_sampling,
        "overfit_coverage": overfit_coverage,
        "history": history,
        "final_train_teacher_forced": final_train_teacher_forced,
        "final_val_teacher_forced": final_val_teacher_forced,
        "final_rollout_probe": final_rollout_probe,
        "resume_from": resume_from.as_posix() if resume_from is not None else None,
    }


def _training_checkpoint_payload(
    *,
    model: Stage1OracleMapper,
    optimizer: torch.optim.Optimizer,
    config: Stage1OracleMapperConfig,
    seed: int,
    run_name: str,
    history: list[dict[str, Any]],
    timing_track: dict[str, Any],
    pretraining_gates: dict[str, Any],
    dataset_filter_report: dict[str, Any] | None,
    training_sampling: dict[str, Any] | None,
    step: int,
    max_steps: int,
    eval_every: int,
    save_every: int,
    learning_rate: float,
    device: torch.device,
    final_train_teacher_forced: dict[str, Any],
    final_val_teacher_forced: dict[str, Any],
    final_rollout_probe: dict[str, Any],
    last_train_loss: float,
    resume_from: Path | None,
) -> dict[str, Any]:
    return {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": asdict(config),
        "seed": seed,
        "run_name": run_name,
        "history": history,
        "timing_track": timing_track,
        "pretraining_gates": pretraining_gates,
        "dataset_filter_report": dataset_filter_report,
        "training_sampling": training_sampling,
        "training_state": {
            "step": step,
            "max_steps": max_steps,
            "is_complete": step >= max_steps,
            "eval_every": eval_every,
            "save_every": save_every,
            "learning_rate": learning_rate,
            "device": str(device),
            "last_train_loss": last_train_loss,
            "final_train_teacher_forced": final_train_teacher_forced,
            "final_val_teacher_forced": final_val_teacher_forced,
            "final_rollout_probe": final_rollout_probe,
            "resume_from": resume_from.as_posix() if resume_from is not None else None,
            "rng_state": _capture_rng_state(),
        },
    }


def _write_training_checkpoint(
    *,
    output_dir: Path,
    latest_checkpoint_path: Path,
    payload: Mapping[str, Any],
    step: int,
) -> Path:
    archive_path = output_dir / "checkpoints" / f"checkpoint_step_{step:06d}.pt"
    _atomic_torch_save(payload, archive_path)
    _copy_file_atomically(archive_path, latest_checkpoint_path)
    return archive_path


def _run_training(
    *,
    loader: DataLoader,
    output_dir: Path,
    config: Stage1OracleMapperConfig,
    max_steps: int,
    eval_every: int,
    save_every: int | None = None,
    learning_rate: float,
    seed: int,
    device_name: str,
    run_name: str,
    vocab: Stage1Vocab,
    train_eval_loader: DataLoader,
    eval_loader: DataLoader,
    rollout_probe_loader: DataLoader | None,
    rollout_eval_every: int | None,
    timing_track: dict[str, Any],
    overfit_coverage: dict[str, Any] | None,
    pretraining_gates: dict[str, Any],
    dataset_filter_report: dict[str, Any] | None,
    training_sampling: dict[str, Any] | None,
    resume_from: Path | None = None,
) -> OverfitRunResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    device = select_torch_device(device_name)
    save_every = _resolve_save_every(save_every, eval_every)
    checkpoint_path = output_dir / "checkpoint.pt"
    report_path = output_dir / "report.json"

    model = Stage1OracleMapper(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    history: list[dict[str, Any]] = []
    iterator = cycle(loader)
    completed_step = 0
    last_train_loss = float("nan")
    final_train_teacher_forced: dict[str, Any] = {}
    final_val_teacher_forced: dict[str, Any] = {"loss": float("nan"), "token_accuracy": 0.0}
    final_rollout_probe: dict[str, Any] = {}

    if resume_from is not None:
        resume_checkpoint = _load_resume_checkpoint(resume_from, expected_config=config)
        model.load_state_dict(resume_checkpoint["model_state_dict"])
        optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        _move_optimizer_state_to_device(optimizer, device)

        training_state = resume_checkpoint["training_state"]
        completed_step = int(training_state["step"])
        history = [dict(entry) for entry in resume_checkpoint["history"]]
        final_train_teacher_forced = _checkpoint_metric_group(training_state, "final_train_teacher_forced")
        final_val_teacher_forced = _checkpoint_metric_group(training_state, "final_val_teacher_forced")
        final_rollout_probe = _checkpoint_metric_group(training_state, "final_rollout_probe")
        last_train_loss = float(
            training_state.get(
                "last_train_loss",
                final_val_teacher_forced.get("loss", final_train_teacher_forced.get("loss", float("nan"))),
            ),
        )
        _restore_rng_state(training_state["rng_state"])
        iterator = _advance_training_iterator(iterator, loader, completed_step)
        print(
            f"resume_progress checkpoint={resume_from} step={completed_step}/{max_steps}",
            flush=True,
        )

    parameter_count = model.parameter_count()
    for step in range(completed_step + 1, max_steps + 1):
        model.train()
        batch = _move_batch(next(iterator), device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(
            packed_audio=batch["packed_audio"],
            timing_track=batch["timing_track"],
            difficulty_bucket=batch["difficulty_bucket"],
            decoder_input_ids=batch["decoder_input_ids"],
            decoder_padding_mask=batch["decoder_padding_mask"],
        )
        loss = F.cross_entropy(
            logits.reshape(-1, config.vocab_size),
            batch["labels"].reshape(-1),
            ignore_index=-100,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        last_train_loss = float(loss.item())

        should_eval = step == 1 or step % eval_every == 0 or step == max_steps
        should_save = _should_write_checkpoint(step=step, max_steps=max_steps, save_every=save_every)
        if should_eval or should_save:
            print(
                f"train_progress step={step}/{max_steps} train_loss={loss.item():.6f}",
                flush=True,
            )
        if should_eval:
            final_val_teacher_forced = teacher_forced_metrics_for_loader(model, eval_loader, device=device)
            history_entry: dict[str, Any] = {
                "step": step,
                "train_loss": last_train_loss,
                "val_teacher_forced": final_val_teacher_forced,
            }
            if step == max_steps:
                final_train_teacher_forced = teacher_forced_metrics_for_loader(model, train_eval_loader, device=device)
                history_entry["train_teacher_forced"] = final_train_teacher_forced

            should_rollout_probe = (
                rollout_probe_loader is not None
                and (
                    step == max_steps
                    or (
                        rollout_eval_every is not None
                        and step % rollout_eval_every == 0
                    )
                )
            )
            if should_rollout_probe and rollout_probe_loader is not None:
                final_rollout_probe = greedy_decode_metrics_for_loader(
                    model,
                    rollout_probe_loader,
                    vocab=vocab,
                    device=device,
                )
                history_entry["rollout_probe"] = final_rollout_probe
            if step == max_steps:
                if not final_train_teacher_forced:
                    final_train_teacher_forced = teacher_forced_metrics_for_loader(
                        model,
                        train_eval_loader,
                        device=device,
                    )
                if rollout_probe_loader is None:
                    final_rollout_probe = greedy_decode_metrics_for_loader(
                        model,
                        eval_loader,
                        vocab=vocab,
                        device=device,
                    )
                    history_entry["rollout_probe"] = final_rollout_probe
            print(
                f"eval_progress step={step}/{max_steps} "
                f"val_loss={final_val_teacher_forced['loss']:.6f} "
                f"val_token_accuracy={final_val_teacher_forced['token_accuracy']:.6f}",
                flush=True,
            )
            history.append(history_entry)

        if should_save:
            checkpoint_payload = _training_checkpoint_payload(
                model=model,
                optimizer=optimizer,
                config=config,
                seed=seed,
                run_name=run_name,
                history=history,
                timing_track=timing_track,
                pretraining_gates=pretraining_gates,
                dataset_filter_report=dataset_filter_report,
                training_sampling=training_sampling,
                step=step,
                max_steps=max_steps,
                eval_every=eval_every,
                save_every=save_every,
                learning_rate=learning_rate,
                device=device,
                final_train_teacher_forced=final_train_teacher_forced,
                final_val_teacher_forced=final_val_teacher_forced,
                final_rollout_probe=final_rollout_probe,
                last_train_loss=last_train_loss,
                resume_from=resume_from,
            )
            archived_checkpoint_path = _write_training_checkpoint(
                output_dir=output_dir,
                latest_checkpoint_path=checkpoint_path,
                payload=checkpoint_payload,
                step=step,
            )
            _write_report(
                report_path,
                _training_report_payload(
                    run_name=run_name,
                    seed=seed,
                    max_steps=max_steps,
                    completed_steps=step,
                    eval_every=eval_every,
                    save_every=save_every,
                    learning_rate=learning_rate,
                    config=config,
                    device=device,
                    parameter_count=parameter_count,
                    timing_track=timing_track,
                    pretraining_gates=pretraining_gates,
                    dataset_filter_report=dataset_filter_report,
                    training_sampling=training_sampling,
                    overfit_coverage=overfit_coverage,
                    history=history,
                    final_train_teacher_forced=final_train_teacher_forced,
                    final_val_teacher_forced=final_val_teacher_forced,
                    final_rollout_probe=final_rollout_probe,
                    resume_from=resume_from,
                ),
            )
            print(
                f"checkpoint_progress step={step}/{max_steps} "
                f"path={archived_checkpoint_path} latest_path={checkpoint_path}",
                flush=True,
            )

    if not checkpoint_path.is_file():
        checkpoint_payload = _training_checkpoint_payload(
            model=model,
            optimizer=optimizer,
            config=config,
            seed=seed,
            run_name=run_name,
            history=history,
            timing_track=timing_track,
            pretraining_gates=pretraining_gates,
            dataset_filter_report=dataset_filter_report,
            training_sampling=training_sampling,
            step=completed_step,
            max_steps=max_steps,
            eval_every=eval_every,
            save_every=save_every,
            learning_rate=learning_rate,
            device=device,
            final_train_teacher_forced=final_train_teacher_forced,
            final_val_teacher_forced=final_val_teacher_forced,
            final_rollout_probe=final_rollout_probe,
            last_train_loss=last_train_loss,
            resume_from=resume_from,
        )
        archived_checkpoint_path = _write_training_checkpoint(
            output_dir=output_dir,
            latest_checkpoint_path=checkpoint_path,
            payload=checkpoint_payload,
            step=completed_step,
        )
        _write_report(
            report_path,
            _training_report_payload(
                run_name=run_name,
                seed=seed,
                max_steps=max_steps,
                completed_steps=completed_step,
                eval_every=eval_every,
                save_every=save_every,
                learning_rate=learning_rate,
                config=config,
                device=device,
                parameter_count=parameter_count,
                timing_track=timing_track,
                pretraining_gates=pretraining_gates,
                dataset_filter_report=dataset_filter_report,
                training_sampling=training_sampling,
                overfit_coverage=overfit_coverage,
                history=history,
                final_train_teacher_forced=final_train_teacher_forced,
                final_val_teacher_forced=final_val_teacher_forced,
                final_rollout_probe=final_rollout_probe,
                resume_from=resume_from,
            ),
        )
        print(
            f"checkpoint_progress step={completed_step}/{max_steps} "
            f"path={archived_checkpoint_path} latest_path={checkpoint_path}",
            flush=True,
        )

    result_metrics = final_val_teacher_forced or final_train_teacher_forced
    return OverfitRunResult(
        report_path=report_path,
        checkpoint_path=checkpoint_path,
        final_loss=float(result_metrics.get("loss", float("nan"))),
        final_token_accuracy=float(result_metrics.get("token_accuracy", 0.0)),
    )


@torch.no_grad()
def teacher_forced_metrics_for_loader(
    model: Stage1OracleMapper,
    loader: DataLoader,
    *,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_label_count = 0
    total_correct = 0
    for raw_batch in loader:
        batch = _move_batch(raw_batch, device)
        logits = model(
            packed_audio=batch["packed_audio"],
            timing_track=batch["timing_track"],
            difficulty_bucket=batch["difficulty_bucket"],
            decoder_input_ids=batch["decoder_input_ids"],
            decoder_padding_mask=batch["decoder_padding_mask"],
        )
        labels = batch["labels"]
        mask = labels != -100
        label_count = int(mask.sum().item())
        if label_count == 0:
            continue
        loss = F.cross_entropy(
            logits.reshape(-1, model.config.vocab_size),
            labels.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )
        total_loss += float(loss.item())
        total_label_count += label_count
        total_correct += int((logits.argmax(dim=-1)[mask] == labels[mask]).sum().item())

    denominator = max(total_label_count, 1)
    return {
        "loss": total_loss / denominator,
        "token_accuracy": total_correct / denominator,
    }


@torch.no_grad()
def greedy_decode_metrics_for_loader(
    model: Stage1OracleMapper,
    loader: DataLoader,
    *,
    vocab: Stage1Vocab,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    counts = _DecodeMetricCounts()
    counts_by_bin = {label: _DecodeMetricCounts() for label in COARSE_BIN_LABELS}
    decode_inputs_by_map: dict[tuple[str, str], list[_DecodeWindowInput]] = {}
    total_window_count, total_map_count = _rollout_progress_totals(loader)
    oracle_window_index = 0
    oracle_generated_length_sum = 0
    _print_rollout_progress(
        pass_name="oracle",
        status="start",
        window_index=0,
        total_window_count=total_window_count,
        map_index=0,
        total_map_count=total_map_count,
        avg_generated_len=0.0,
    )

    for raw_batch in loader:
        batch_inputs = _move_rollout_batch_inputs(raw_batch, device)
        for index in range(int(raw_batch["decoder_input_ids"].shape[0])):
            metadata = raw_batch["metadata"][index]
            difficulty_bin_label = _difficulty_bin_label(float(metadata["difficulty"]))
            if difficulty_bin_label is None:
                raise ValueError(f"difficulty outside supported coarse bins: {metadata['difficulty']}")
            bin_counts = counts_by_bin[difficulty_bin_label]
            counts.evaluated_window_count += 1
            bin_counts.evaluated_window_count += 1
            condition_ids = [int(token_id) for token_id in raw_batch["decoder_input_ids"][index, :3].tolist()]
            write_duration_ms = int(raw_batch["write_duration_ms"][index].item())
            map_key = (str(metadata["beatmap_path"]), str(metadata["audio_path"]))
            write_start_ms = int(metadata["write_start_ms"])
            oracle_open_hold_mask = int(raw_batch["open_hold_mask"][index].item())
            decode_result = constrained_greedy_decode(
                model,
                packed_audio=batch_inputs["packed_audio"][index : index + 1],
                timing_track=batch_inputs["timing_track"][index : index + 1],
                difficulty_bucket=batch_inputs["difficulty_bucket"][index : index + 1],
                condition_ids=condition_ids,
                open_hold_mask=oracle_open_hold_mask,
                write_duration_ms=write_duration_ms,
                vocab=vocab,
                max_decode_len=model.config.max_decode_len,
            )
            target_tokens = decode_result.token_ids[3:]
            counts.generated_lengths.append(len(target_tokens))
            bin_counts.generated_lengths.append(len(target_tokens))
            oracle_window_index += 1
            oracle_generated_length_sum += len(target_tokens)
            if decode_result.max_decode_len_reached:
                counts.max_decode_len_reached += 1
                bin_counts.max_decode_len_reached += 1
            if decode_result.eos_forced_after_pending_ts:
                counts.eos_forced_after_pending_ts += 1
                bin_counts.eos_forced_after_pending_ts += 1
            if not decode_result.eos_emitted_by_model:
                counts.eos_failures += 1
                bin_counts.eos_failures += 1
            if target_tokens and target_tokens[0] == vocab.eos_id:
                counts.empty_outputs += 1
                bin_counts.empty_outputs += 1

            label_tokens = [int(token_id) for token_id in raw_batch["labels"][index].tolist() if int(token_id) != -100]
            reference_timepoints = decode_target_tokens(
                label_tokens,
                vocab=vocab,
                write_duration_ms=write_duration_ms,
            )
            reference_note_count = _note_start_count_for_timepoints(reference_timepoints)
            counts.reference_note_count += reference_note_count
            bin_counts.reference_note_count += reference_note_count

            try:
                generated_timepoints = decode_target_tokens(
                    target_tokens,
                    vocab=vocab,
                    write_duration_ms=write_duration_ms,
                )
            except ValueError:
                counts.time_monotonicity_errors += 1
                bin_counts.time_monotonicity_errors += 1
                generated_timepoints = []
            generated_note_count = _note_start_count_for_timepoints(generated_timepoints)
            counts.generated_event_count += len(generated_timepoints)
            counts.generated_note_count += generated_note_count
            bin_counts.generated_event_count += len(generated_timepoints)
            bin_counts.generated_note_count += generated_note_count
            oracle_window = stitch_decoded_windows(
                [
                    DecodedWindowEvents(
                        write_start_ms=write_start_ms,
                        timepoints=generated_timepoints,
                    ),
                ],
                initial_open_hold_mask=oracle_open_hold_mask,
            )
            counts.oracle_invalid_hold_end_count += oracle_window.invalid_hold_end_count
            counts.oracle_same_lane_collision_count += oracle_window.same_lane_collision_count
            bin_counts.oracle_invalid_hold_end_count += oracle_window.invalid_hold_end_count
            bin_counts.oracle_same_lane_collision_count += oracle_window.same_lane_collision_count
            decode_inputs_by_map.setdefault(map_key, []).append(
                _DecodeWindowInput(
                    write_start_ms=write_start_ms,
                    packed_audio=_clone_tensor_to_cpu(raw_batch["packed_audio"][index : index + 1]),
                    timing_track=_clone_tensor_to_cpu(raw_batch["timing_track"][index : index + 1]),
                    difficulty_bucket=_clone_tensor_to_cpu(raw_batch["difficulty_bucket"][index : index + 1]),
                    condition_ids=condition_ids,
                    oracle_open_hold_mask=oracle_open_hold_mask,
                    write_duration_ms=write_duration_ms,
                    difficulty_bin_label=difficulty_bin_label,
                ),
            )
            _print_rollout_progress(
                pass_name="oracle",
                window_index=oracle_window_index,
                total_window_count=total_window_count,
                map_index=len(decode_inputs_by_map),
                total_map_count=total_map_count,
                avg_generated_len=oracle_generated_length_sum / max(oracle_window_index, 1),
            )

    total_window_count = counts.evaluated_window_count if total_window_count is None else total_window_count
    total_map_count = len(decode_inputs_by_map) if total_map_count is None else total_map_count
    stitched_window_index = 0
    stitched_generated_length_sum = 0
    _print_rollout_progress(
        pass_name="stitched",
        status="start",
        window_index=0,
        total_window_count=total_window_count,
        map_index=0,
        total_map_count=total_map_count,
        avg_generated_len=0.0,
    )
    for map_index, decode_inputs in enumerate(decode_inputs_by_map.values(), start=1):
        carried_open_hold_mask = 0
        sorted_decode_inputs = sorted(decode_inputs, key=lambda item: item.write_start_ms)
        last_bin_counts: _DecodeMetricCounts | None = None
        for decode_input in sorted_decode_inputs:
            bin_counts = counts_by_bin[decode_input.difficulty_bin_label]
            last_bin_counts = bin_counts
            if decode_input.write_start_ms != 0:
                counts.stitched_boundary_count += 1
                bin_counts.stitched_boundary_count += 1
                if decode_input.oracle_open_hold_mask != 0:
                    counts.active_stitched_boundary_count += 1
                    bin_counts.active_stitched_boundary_count += 1
                if carried_open_hold_mask != decode_input.oracle_open_hold_mask:
                    counts.stitched_boundary_errors += 1
                    bin_counts.stitched_boundary_errors += 1
                elif decode_input.oracle_open_hold_mask != 0:
                    counts.active_stitched_boundary_matches += 1
                    bin_counts.active_stitched_boundary_matches += 1

            stitched_condition_ids = [
                decode_input.condition_ids[0],
                decode_input.condition_ids[1],
                vocab.open_token_id(carried_open_hold_mask),
            ]
            packed_audio, timing_track, difficulty_bucket = _decode_window_input_to_device(
                decode_input,
                device=device,
            )
            decode_result = constrained_greedy_decode(
                model,
                packed_audio=packed_audio,
                timing_track=timing_track,
                difficulty_bucket=difficulty_bucket,
                condition_ids=stitched_condition_ids,
                open_hold_mask=carried_open_hold_mask,
                write_duration_ms=decode_input.write_duration_ms,
                vocab=vocab,
                max_decode_len=model.config.max_decode_len,
            )
            stitched_target_tokens = decode_result.token_ids[3:]
            stitched_window_index += 1
            stitched_generated_length_sum += len(stitched_target_tokens)
            try:
                stitched_timepoints = decode_target_tokens(
                    stitched_target_tokens,
                    vocab=vocab,
                    write_duration_ms=decode_input.write_duration_ms,
                )
            except ValueError:
                stitched_timepoints = []
            stitched_window = stitch_decoded_windows(
                [
                    DecodedWindowEvents(
                        write_start_ms=decode_input.write_start_ms,
                        timepoints=stitched_timepoints,
                    ),
                ],
                initial_open_hold_mask=carried_open_hold_mask,
            )
            counts.stitched_invalid_hold_end_count += stitched_window.invalid_hold_end_count
            counts.stitched_same_lane_collision_count += stitched_window.same_lane_collision_count
            counts.stitched_generated_event_count += len(stitched_timepoints)
            bin_counts.stitched_invalid_hold_end_count += stitched_window.invalid_hold_end_count
            bin_counts.stitched_same_lane_collision_count += stitched_window.same_lane_collision_count
            bin_counts.stitched_generated_event_count += len(stitched_timepoints)
            carried_open_hold_mask = stitched_window.final_open_hold_mask
            _print_rollout_progress(
                pass_name="stitched",
                window_index=stitched_window_index,
                total_window_count=total_window_count,
                map_index=map_index,
                total_map_count=total_map_count,
                avg_generated_len=stitched_generated_length_sum / max(stitched_window_index, 1),
            )
        if last_bin_counts is not None:
            unclosed_hold_count = carried_open_hold_mask.bit_count()
            counts.stitched_unclosed_hold_count += unclosed_hold_count
            last_bin_counts.stitched_unclosed_hold_count += unclosed_hold_count

    metrics = _decode_metric_report(counts)
    metrics["decode_per_difficulty_bin"] = {
        label: _decode_metric_report(bin_counts)
        for label, bin_counts in counts_by_bin.items()
    }
    metrics["boundary_error_by_bin"] = {
        label: metrics["decode_per_difficulty_bin"][label]["stitched_boundary_open_mask_error_rate"]
        for label in COARSE_BIN_LABELS
    }
    return metrics


def _decode_metric_report(counts: _DecodeMetricCounts) -> dict[str, float | int]:
    density_error = 0.0
    if counts.reference_note_count > 0:
        density_error = abs(counts.generated_note_count - counts.reference_note_count) / counts.reference_note_count

    generated_lengths = counts.generated_lengths or []
    window_denominator = max(counts.evaluated_window_count, 1)
    oracle_event_denominator = max(counts.generated_event_count, 1)
    stitched_event_denominator = max(counts.stitched_generated_event_count, 1)
    boundary_denominator = max(counts.stitched_boundary_count, 1)
    active_boundary_denominator = max(counts.active_stitched_boundary_count, 1)
    return {
        "decode_density_error": float(density_error),
        "decode_time_monotonicity_error": float(counts.time_monotonicity_errors),
        "decode_eos_failure_rate": counts.eos_failures / window_denominator,
        "decode_eos_forced_after_pending_ts_rate": counts.eos_forced_after_pending_ts / window_denominator,
        "decode_max_decode_len_reached_rate": counts.max_decode_len_reached / window_denominator,
        "decode_empty_output_rate": counts.empty_outputs / window_denominator,
        "oracle_boundary_invalid_hold_end_rate": counts.oracle_invalid_hold_end_count / oracle_event_denominator,
        "oracle_boundary_same_lane_collision_rate": counts.oracle_same_lane_collision_count / oracle_event_denominator,
        "oracle_boundary_invalid_hold_transition_rate": (
            counts.oracle_invalid_hold_transition_count / oracle_event_denominator
        ),
        "stitched_boundary_invalid_hold_end_rate": counts.stitched_invalid_hold_end_count / stitched_event_denominator,
        "stitched_boundary_same_lane_collision_rate": (
            counts.stitched_same_lane_collision_count / stitched_event_denominator
        ),
        "stitched_boundary_invalid_hold_transition_rate": (
            counts.stitched_invalid_hold_transition_count / stitched_event_denominator
        ),
        "stitched_boundary_unclosed_hold_rate": counts.stitched_unclosed_hold_count / stitched_event_denominator,
        "decode_average_generated_length": sum(generated_lengths) / max(len(generated_lengths), 1),
        "decode_evaluated_window_count": counts.evaluated_window_count,
        "stitched_boundary_open_mask_error_rate": counts.stitched_boundary_errors / boundary_denominator,
        "stitched_boundary_evaluated_boundary_count": counts.stitched_boundary_count,
        "active_boundary_exact_match": counts.active_stitched_boundary_matches / active_boundary_denominator,
        "active_boundary_evaluated_boundary_count": counts.active_stitched_boundary_count,
    }


def _note_start_count_for_timepoints(timepoints: Sequence[Any]) -> int:
    return sum(
        1
        for timepoint in timepoints
        for action in timepoint.lane_actions
        if action in {LaneAction.TAP, LaneAction.HOLD_START}
    )


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = dict(batch)
    for key in (
        "packed_audio",
        "timing_track",
        "difficulty_bucket",
        "open_hold_mask",
        "write_duration_ms",
        "decoder_input_ids",
        "decoder_padding_mask",
        "labels",
    ):
        moved[key] = batch[key].to(device)
    return moved


def _move_rollout_batch_inputs(batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "packed_audio": batch["packed_audio"].to(device),
        "timing_track": batch["timing_track"].to(device),
        "difficulty_bucket": batch["difficulty_bucket"].to(device),
    }


def _clone_tensor_to_cpu(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().cpu().clone()


def _decode_window_input_to_device(
    decode_input: _DecodeWindowInput,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        decode_input.packed_audio.to(device),
        decode_input.timing_track.to(device),
        decode_input.difficulty_bucket.to(device),
    )


def _record_value(record: Any, key: str) -> Any:
    if isinstance(record, dict):
        return record[key]
    return getattr(record, key)


def _difficulty_bin_label(stars: float) -> str | None:
    if 2.0 <= stars < 3.0:
        return "2-3"
    if 3.0 <= stars < 4.0:
        return "3-4"
    if 4.0 <= stars < 5.0:
        return "4-5"
    if 5.0 <= stars <= 6.0:
        return "5-6"
    return None


def _record_is_empty_window(record: Any) -> bool:
    window = _record_value(record, "window")
    write_start_ms = int(_record_value(window, "write_start_ms"))
    write_end_ms = int(_record_value(window, "write_end_ms"))
    return not any(
        write_start_ms <= int(_record_value(timepoint, "time_ms")) < write_end_ms
        for timepoint in _record_value(record, "timepoints")
    )


def _max_empty_windows_for_non_empty_count(non_empty_count: int, empty_window_cap_ratio: float) -> int:
    if non_empty_count <= 0 or empty_window_cap_ratio <= 0.0:
        return 0
    if empty_window_cap_ratio >= 1.0:
        return 2**31 - 1
    return math.floor((non_empty_count * empty_window_cap_ratio) / (1.0 - empty_window_cap_ratio))


def _empty_window_cap_by_bin_for_sampling(
    source: dict[str, float] | None,
    *,
    default_ratio: float,
) -> dict[str, float]:
    if source is None:
        return {label: default_ratio for label in COARSE_BIN_LABELS}
    missing = [label for label in COARSE_BIN_LABELS if label not in source]
    if missing:
        raise ValueError(f"empty_window_cap_by_bin missing bins: {missing}")
    resolved: dict[str, float] = {}
    for label in COARSE_BIN_LABELS:
        ratio = float(source[label])
        if not 0.0 <= ratio <= 1.0:
            raise ValueError(f"empty_window_cap_by_bin[{label}] must be within [0, 1]: {ratio}")
        resolved[label] = ratio
    return resolved


def _sample_with_replacement(pool: Sequence[int], count: int, rng: random.Random) -> list[int]:
    if count <= 0 or not pool:
        return []
    sampled = list(pool)
    rng.shuffle(sampled)
    while len(sampled) < count:
        sampled.append(rng.choice(pool))
    return sampled[:count]


def _sampling_plan_report(plan: BalancedEpochSamplingPlan) -> dict[str, Any]:
    return {
        "empty_window_cap_ratio": plan.empty_window_cap_ratio,
        "empty_window_cap_by_bin": plan.empty_window_cap_by_bin,
        "epoch_window_count": len(plan.indices),
        "source_window_count_by_bin": plan.source_window_count_by_bin,
        "source_empty_window_count_by_bin": plan.source_empty_window_count_by_bin,
        "sample_count_by_bin": plan.sample_count_by_bin,
        "empty_sample_count_by_bin": plan.empty_sample_count_by_bin,
    }


def _synthetic_samples(vocab: Stage1Vocab) -> list[dict[str, Any]]:
    condition = [
        vocab.bos_id,
        vocab.diff_token_id(vocab.difficulty_bucket_id(2.5)),
        vocab.open_token_id(0),
    ]
    target = [
        vocab.ts_token_id(0),
        vocab.encode_timepoint_event((LaneAction.TAP, LaneAction.NONE, LaneAction.NONE, LaneAction.NONE)),
        vocab.eos_id,
    ]
    decoder_input = torch.tensor(condition + target[:-1], dtype=torch.long)
    labels = torch.tensor([-100, -100] + target, dtype=torch.long)
    sample = {
        "packed_audio": torch.zeros(600, 160),
        "timing_track": torch.zeros(600, 5),
        "difficulty_bucket": torch.tensor(vocab.difficulty_bucket_id(2.5), dtype=torch.long),
        "open_hold_mask": torch.tensor(0, dtype=torch.long),
        "write_duration_ms": torch.tensor(8000, dtype=torch.long),
        "decoder_input_ids": decoder_input,
        "labels": labels,
        "beatmap_path": "synthetic.osu",
        "audio_path": "synthetic.mp3",
        "difficulty": 2.5,
        "write_start_ms": 0,
    }
    return [sample, dict(sample)]


def main(argv: Sequence[str] | None = None) -> None:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=None, help="YAML run config; CLI flags override config values")
    config_args, _ = config_parser.parse_known_args(argv)
    config_defaults = load_run_config(config_args.config) if config_args.config is not None else {"model": {}}
    model_defaults = config_defaults["model"]

    parser = argparse.ArgumentParser(
        description="Run Stage 1 oracle mapper map-subset overfit.",
        parents=[config_parser],
    )
    parser.add_argument("--dataset-root", default=config_defaults.get("dataset_root", "mania-dataset"))
    parser.add_argument("--index-path", default=config_defaults.get("index_path"))
    parser.add_argument("--gate-manifest", default=config_defaults.get("gate_manifest"))
    parser.add_argument(
        "--output-dir",
        default=config_defaults.get("output_dir", "train/artifacts/runs/stage1_oracle/overfit_32"),
    )
    parser.add_argument("--maps-per-bin", type=int, default=config_defaults.get("maps_per_bin", 8))
    parser.add_argument("--max-steps", type=int, default=config_defaults.get("max_steps", 5000))
    parser.add_argument("--eval-every", type=int, default=config_defaults.get("eval_every", 100))
    parser.add_argument("--batch-size", type=int, default=config_defaults.get("batch_size", 8))
    parser.add_argument("--learning-rate", type=float, default=config_defaults.get("learning_rate", 3e-4))
    parser.add_argument("--dropout", type=float, default=config_defaults.get("dropout", 0.1))
    parser.add_argument("--seed", type=int, default=config_defaults.get("seed", 1337))
    parser.add_argument("--device", default=config_defaults.get("device", "auto"), choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--run-name", default=config_defaults.get("run_name", "overfit_32"))
    parser.add_argument("--save-every", type=int, default=config_defaults.get("save_every"))
    parser.add_argument("--resume-from", default=config_defaults.get("resume_from"))
    parser.add_argument("--train-manifest", default=config_defaults.get("train_manifest"))
    parser.add_argument("--eval-manifest", default=config_defaults.get("eval_manifest"))
    parser.add_argument("--rollout-probe-manifest", default=config_defaults.get("rollout_probe_manifest"))
    parser.add_argument("--rollout-eval-every", type=int, default=config_defaults.get("rollout_eval_every"))
    parser.add_argument("--synthetic-smoke", action="store_true", default=bool(config_defaults.get("synthetic_smoke", False)))
    parser.add_argument("--d-model", type=int, default=model_defaults.get("d_model"))
    parser.add_argument("--heads", type=int, default=model_defaults.get("heads"))
    parser.add_argument("--encoder-layers", type=int, default=model_defaults.get("encoder_layers"))
    parser.add_argument("--decoder-layers", type=int, default=model_defaults.get("decoder_layers"))
    parser.add_argument("--ffn-dim", type=int, default=model_defaults.get("ffn_dim"))
    args = parser.parse_args(argv)
    model_config_overrides = {
        key: getattr(args, key)
        for key in MODEL_CONFIG_OVERRIDE_KEYS
        if getattr(args, key) is not None
    }

    if args.synthetic_smoke:
        result = run_synthetic_smoke(
            output_dir=Path(args.output_dir),
            max_steps=args.max_steps,
            save_every=args.save_every,
            seed=args.seed,
            device_name=args.device,
            resume_from=Path(args.resume_from) if args.resume_from is not None else None,
        )
    else:
        if args.gate_manifest is None:
            parser.error("--gate-manifest is required unless --synthetic-smoke is set")
        result = run_overfit_32(
            dataset_root=Path(args.dataset_root),
            index_path=Path(args.index_path) if args.index_path is not None else None,
            gate_manifest_path=Path(args.gate_manifest),
            output_dir=Path(args.output_dir),
            maps_per_bin=args.maps_per_bin,
            max_steps=args.max_steps,
            eval_every=args.eval_every,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            dropout=args.dropout,
            seed=args.seed,
            device_name=args.device,
            run_name=args.run_name,
            save_every=args.save_every,
            resume_from=Path(args.resume_from) if args.resume_from is not None else None,
            model_config_overrides=model_config_overrides,
            train_manifest_path=Path(args.train_manifest) if args.train_manifest is not None else None,
            eval_manifest_path=Path(args.eval_manifest) if args.eval_manifest is not None else None,
            rollout_probe_manifest_path=(
                Path(args.rollout_probe_manifest)
                if args.rollout_probe_manifest is not None
                else None
            ),
            rollout_eval_every=args.rollout_eval_every,
        )

    print(f"report_path {result.report_path}")
    print(f"checkpoint_path {result.checkpoint_path}")
    print(f"final_loss {result.final_loss:.6f}")
    print(f"final_token_accuracy {result.final_token_accuracy:.6f}")


if __name__ == "__main__":
    main()
