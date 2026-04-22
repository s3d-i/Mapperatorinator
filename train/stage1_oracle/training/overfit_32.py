from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from itertools import cycle
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..data.windows import OracleWindowDataset, collate_oracle_windows
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
    seed: int = 1337,
    device_name: str = "auto",
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
        learning_rate=1e-2,
        seed=seed,
        device_name=device_name,
        run_name="synthetic_smoke",
        vocab=vocab,
        eval_loader=loader,
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
    )


def run_overfit_32(
    *,
    dataset_root: Path,
    index_path: Path | None,
    gate_manifest_path: Path,
    output_dir: Path,
    maps_per_bin: int = 8,
    max_steps: int = 5000,
    eval_every: int = 100,
    batch_size: int = 8,
    learning_rate: float = 3e-4,
    dropout: float = 0.1,
    seed: int = 1337,
    device_name: str = "auto",
    run_name: str = "overfit_32",
) -> OverfitRunResult:
    if maps_per_bin <= 0:
        raise ValueError(f"maps_per_bin must be positive, got {maps_per_bin}")
    if dropout < 0.0:
        raise ValueError(f"dropout must be non-negative, got {dropout}")
    torch.manual_seed(seed)
    vocab = Stage1Vocab()
    print("gate_progress status=validating", flush=True)
    pretraining_gates = validate_pretraining_gate_manifest(gate_manifest_path)
    print("gate_progress status=pass", flush=True)
    timing_stats = timing_training_stats_from_pretraining_gates(pretraining_gates)
    training_config = training_config_from_pretraining_gates(pretraining_gates)
    print("dataset_progress phase=build_windows status=start", flush=True)
    dataset = OracleWindowDataset(
        dataset_root=dataset_root,
        index_path=index_path,
        vocab=vocab,
        bpm_log_mean=timing_stats["bpm_log_mean"],
        bpm_log_std=timing_stats["bpm_log_std"],
        max_maps_per_bin=maps_per_bin,
        progress=True,
    )
    print(
        f"dataset_progress phase=build_windows status=done "
        f"retained_maps={dataset.filter_report.retained_map_count} windows={len(dataset)}",
        flush=True,
    )
    if len(dataset) == 0:
        raise ValueError("OracleWindowDataset produced no windows for overfit_32")
    overfit_coverage = summarize_overfit_coverage(dataset.records)
    underfilled_bins = {
        label: count
        for label, count in overfit_coverage["map_count_by_bin"].items()
        if count != maps_per_bin
    }
    if underfilled_bins:
        raise ValueError(f"overfit_32 requires {maps_per_bin} retained maps per bin, got {underfilled_bins}")
    sampling_plan = build_balanced_epoch_sampling_plan(
        dataset.records,
        empty_window_cap_ratio=training_config["empty_window_cap_ratio"],
        empty_window_cap_by_bin=training_config["empty_window_cap_by_bin"],
        seed=seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampling_plan.indices,
        collate_fn=lambda batch: collate_oracle_windows(batch, pad_id=vocab.pad_id),
    )
    eval_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_oracle_windows(batch, pad_id=vocab.pad_id),
    )
    return _run_training(
        loader=loader,
        output_dir=output_dir,
        config=Stage1OracleMapperConfig(
            vocab_size=vocab.size,
            max_decode_len=training_config["max_decode_len"],
            dropout=dropout,
        ),
        max_steps=max_steps,
        eval_every=eval_every,
        learning_rate=learning_rate,
        seed=seed,
        device_name=device_name,
        run_name=run_name,
        vocab=vocab,
        eval_loader=eval_loader,
        timing_track=build_timing_track_metadata(
            bpm_log_mean=dataset.bpm_log_mean,
            bpm_log_std=dataset.bpm_log_std,
        ),
        overfit_coverage=overfit_coverage,
        pretraining_gates=pretraining_gates,
        dataset_filter_report=asdict(dataset.filter_report),
        training_sampling=_sampling_plan_report(sampling_plan),
    )


def _run_training(
    *,
    loader: DataLoader,
    output_dir: Path,
    config: Stage1OracleMapperConfig,
    max_steps: int,
    eval_every: int,
    learning_rate: float,
    seed: int,
    device_name: str,
    run_name: str,
    vocab: Stage1Vocab,
    eval_loader: DataLoader,
    timing_track: dict[str, Any],
    overfit_coverage: dict[str, Any] | None,
    pretraining_gates: dict[str, Any],
    dataset_filter_report: dict[str, Any] | None,
    training_sampling: dict[str, Any] | None,
) -> OverfitRunResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    device = select_torch_device(device_name)
    model = Stage1OracleMapper(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    history: list[dict[str, Any]] = []
    iterator = cycle(loader)

    final_metrics = {"loss": float("nan"), "token_accuracy": 0.0}
    for step in range(1, max_steps + 1):
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

        if step == 1 or step % eval_every == 0 or step == max_steps:
            print(
                f"train_progress step={step}/{max_steps} train_loss={loss.item():.6f}",
                flush=True,
            )
            final_metrics = teacher_forced_metrics_for_loader(model, eval_loader, device=device)
            if step == max_steps:
                final_metrics.update(greedy_decode_metrics_for_loader(model, eval_loader, vocab=vocab, device=device))
            print(
                f"eval_progress step={step}/{max_steps} "
                f"loss={final_metrics['loss']:.6f} "
                f"token_accuracy={final_metrics['token_accuracy']:.6f}",
                flush=True,
            )
            history.append({"step": step, **final_metrics})

    checkpoint_path = output_dir / "checkpoint.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(config),
            "seed": seed,
            "run_name": run_name,
            "history": history,
            "timing_track": timing_track,
            "pretraining_gates": pretraining_gates,
            "dataset_filter_report": dataset_filter_report,
            "training_sampling": training_sampling,
        },
        checkpoint_path,
    )
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(
            {
                "run_name": run_name,
                "seed": seed,
                "max_steps": max_steps,
                "eval_every": eval_every,
                "learning_rate": learning_rate,
                "config": asdict(config),
                "device": str(device),
                "parameter_count": model.parameter_count(),
                "timing_track": timing_track,
                "pretraining_gates": pretraining_gates,
                "dataset_filter_report": dataset_filter_report,
                "training_sampling": training_sampling,
                "overfit_coverage": overfit_coverage,
                "history": history,
                "final": final_metrics,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return OverfitRunResult(
        report_path=report_path,
        checkpoint_path=checkpoint_path,
        final_loss=float(final_metrics["loss"]),
        final_token_accuracy=float(final_metrics["token_accuracy"]),
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

    for raw_batch in loader:
        batch = _move_batch(raw_batch, device)
        for index in range(int(batch["decoder_input_ids"].shape[0])):
            metadata = batch["metadata"][index]
            difficulty_bin_label = _difficulty_bin_label(float(metadata["difficulty"]))
            if difficulty_bin_label is None:
                raise ValueError(f"difficulty outside supported coarse bins: {metadata['difficulty']}")
            bin_counts = counts_by_bin[difficulty_bin_label]
            counts.evaluated_window_count += 1
            bin_counts.evaluated_window_count += 1
            condition_ids = [int(token_id) for token_id in batch["decoder_input_ids"][index, :3].tolist()]
            write_duration_ms = int(batch["write_duration_ms"][index].item())
            map_key = (str(metadata["beatmap_path"]), str(metadata["audio_path"]))
            write_start_ms = int(metadata["write_start_ms"])
            oracle_open_hold_mask = int(batch["open_hold_mask"][index].item())
            decode_result = constrained_greedy_decode(
                model,
                packed_audio=batch["packed_audio"][index : index + 1],
                timing_track=batch["timing_track"][index : index + 1],
                difficulty_bucket=batch["difficulty_bucket"][index : index + 1],
                condition_ids=condition_ids,
                open_hold_mask=oracle_open_hold_mask,
                write_duration_ms=write_duration_ms,
                vocab=vocab,
                max_decode_len=model.config.max_decode_len,
            )
            target_tokens = decode_result.token_ids[3:]
            counts.generated_lengths.append(len(target_tokens))
            bin_counts.generated_lengths.append(len(target_tokens))
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

            label_tokens = [int(token_id) for token_id in batch["labels"][index].tolist() if int(token_id) != -100]
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
                    packed_audio=batch["packed_audio"][index : index + 1],
                    timing_track=batch["timing_track"][index : index + 1],
                    difficulty_bucket=batch["difficulty_bucket"][index : index + 1],
                    condition_ids=condition_ids,
                    oracle_open_hold_mask=oracle_open_hold_mask,
                    write_duration_ms=write_duration_ms,
                    difficulty_bin_label=difficulty_bin_label,
                ),
            )

    for decode_inputs in decode_inputs_by_map.values():
        carried_open_hold_mask = 0
        sorted_decode_inputs = sorted(decode_inputs, key=lambda item: item.write_start_ms)
        last_bin_counts: _DecodeMetricCounts | None = None
        for decode_input in sorted_decode_inputs:
            bin_counts = counts_by_bin[decode_input.difficulty_bin_label]
            last_bin_counts = bin_counts
            if decode_input.write_start_ms != 0:
                counts.stitched_boundary_count += 1
                bin_counts.stitched_boundary_count += 1
                if carried_open_hold_mask != decode_input.oracle_open_hold_mask:
                    counts.stitched_boundary_errors += 1
                    bin_counts.stitched_boundary_errors += 1

            stitched_condition_ids = [
                decode_input.condition_ids[0],
                decode_input.condition_ids[1],
                vocab.open_token_id(carried_open_hold_mask),
            ]
            decode_result = constrained_greedy_decode(
                model,
                packed_audio=decode_input.packed_audio,
                timing_track=decode_input.timing_track,
                difficulty_bucket=decode_input.difficulty_bucket,
                condition_ids=stitched_condition_ids,
                open_hold_mask=carried_open_hold_mask,
                write_duration_ms=decode_input.write_duration_ms,
                vocab=vocab,
                max_decode_len=model.config.max_decode_len,
            )
            try:
                stitched_timepoints = decode_target_tokens(
                    decode_result.token_ids[3:],
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
        if last_bin_counts is not None:
            unclosed_hold_count = carried_open_hold_mask.bit_count()
            counts.stitched_unclosed_hold_count += unclosed_hold_count
            last_bin_counts.stitched_unclosed_hold_count += unclosed_hold_count

    metrics = _decode_metric_report(counts)
    metrics["decode_per_difficulty_bin"] = {
        label: _decode_metric_report(bin_counts)
        for label, bin_counts in counts_by_bin.items()
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
    parser = argparse.ArgumentParser(description="Run Stage 1 oracle mapper map-subset overfit.")
    parser.add_argument("--dataset-root", default="mania-dataset")
    parser.add_argument("--index-path", default=None)
    parser.add_argument("--gate-manifest", default=None)
    parser.add_argument("--output-dir", default="train/artifacts/runs/stage1_oracle/overfit_32")
    parser.add_argument("--maps-per-bin", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--run-name", default="overfit_32")
    parser.add_argument("--synthetic-smoke", action="store_true")
    args = parser.parse_args(argv)

    if args.synthetic_smoke:
        result = run_synthetic_smoke(
            output_dir=Path(args.output_dir),
            max_steps=args.max_steps,
            seed=args.seed,
            device_name=args.device,
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
        )

    print(f"report_path {result.report_path}")
    print(f"checkpoint_path {result.checkpoint_path}")
    print(f"final_loss {result.final_loss:.6f}")
    print(f"final_token_accuracy {result.final_token_accuracy:.6f}")


if __name__ == "__main__":
    main()
