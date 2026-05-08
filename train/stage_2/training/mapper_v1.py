from __future__ import annotations

import argparse
import math
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset

from train.stage_2.data.control_windows import (
    ControlWindowDataset,
    ControlWindowRecord,
    DEFAULT_MAX_CACHED_MAPS,
    DENSE_TIMING_V2_CHANNELS,
    PACKED_MEL_CHANNELS,
    TARGET_WINDOW_LENGTH_FRAMES,
    normalize_difficulty,
)
from train.stage_2.data.mapper_v1_windows import (
    MAPPER_WRITE_FRAMES,
    MapperV1WindowDataset,
    collate_mapper_v1_windows,
    control_teacher_cache_path,
    save_control_teacher_cache_entry,
)
from train.stage_2.model_control_demo_global import ControlDemoGlobalEncoder, ControlDemoGlobalEncoderConfig
from train.stage_2.model_mapper_v1 import MapperV1Config, MapperV1Model, MapperV1ModelOutput, MapperV1Vocab
from train.stage_2.model_mapper_v1.loss import adapter_bias_regularization, token_cross_entropy
from train.stage_2.model_mapper_v1.model import compute_control_teacher_8s
from train.stage_2.training.control import (
    CHECKPOINT_SCHEMA_VERSION,
    DEFAULT_FINAL_TRAIN_EVAL_SIZE,
    ControlTrainingResult,
    _atomic_torch_save,
    _capture_rng_state,
    _copy_file_atomically,
    _infinite_loader,
    _json_metrics,
    _json_safe,
    _metric_is_count,
    _move_batch_tensors,
    _safe_float_div,
    _set_deterministic_seed,
    _validate_training_args,
    _write_report,
    limit_final_train_eval_dataset,
    select_torch_device,
    split_train_eval_dataset,
)
from train.stage_2.training.control_demo_global import initialize_global_control_demo_from_control_checkpoint


DEFAULT_RUNS_ROOT = Path("train/artifacts/runs/stage2_mapper_v1")
DEFAULT_OUTPUT_DIR = DEFAULT_RUNS_ROOT / "phase_b_teacher_forced"
RUN_CONFIG_KEYS = {
    "dataset_root",
    "index_path",
    "eval_index_path",
    "control_v3_timeseries_path",
    "output_dir",
    "max_steps",
    "eval_every",
    "save_every",
    "log_every",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "seed",
    "device",
    "run_name",
    "init_from_control_checkpoint",
    "eval_fraction",
    "eval_size",
    "final_train_eval_size",
    "num_workers",
    "max_cached_maps",
    "dataset_progress",
    "control_teacher_cache_dir",
    "precompute_control_teacher_cache",
    "precompute_control_teacher_cache_only",
    "control_teacher_precompute_batch_size",
    "require_control_teacher_cache",
    "control_teacher_cache_overwrite",
    "synthetic_smoke",
    "model",
    "control_model",
    "loss",
}
MODEL_CONFIG_KEYS = {field.name for field in fields(MapperV1Config)}
CONTROL_MODEL_CONFIG_KEYS = {field.name for field in fields(ControlDemoGlobalEncoderConfig)}
LOSS_CONFIG_KEYS = {
    "lambda_ln_close",
    "lambda_adapter_reg",
    "lambda_density",
    "lambda_density_teacher",
    "close_pos_weight_max",
}
MAPPER_BATCH_TENSOR_KEYS = frozenset(
    (
        "target_tokens",
        "target_token_mask",
        "teacher_current_ms",
        "teacher_open_mask",
        "teacher_open_age_ms",
        "close_labels",
        "close_label_mask",
        "density_target_8s",
        "density_confidence_8s",
        "density_teacher_8s",
        "control_memory_8s",
        "control_memory_padding_mask_8s",
        "write_start_ms",
        "write_end_ms",
        "difficulty",
        "normalized_difficulty",
        "full_mel",
        "full_dense_timing_v2",
        "padding_mask",
        "frame_count",
        "control_slice_start_frames",
    )
)


@dataclass(frozen=True)
class MapperV1PhaseBLossConfig:
    lambda_ln_close: float = 0.05
    lambda_adapter_reg: float = 1e-5
    lambda_density: float = 0.0
    lambda_density_teacher: float = 0.0
    close_pos_weight_max: float = 20.0

    def __post_init__(self) -> None:
        if self.lambda_density != 0.0 or self.lambda_density_teacher != 0.0:
            raise ValueError("density loss is disabled for Phase B")
        if self.lambda_ln_close < 0.0:
            raise ValueError("lambda_ln_close must be non-negative")
        if self.lambda_adapter_reg < 0.0:
            raise ValueError("lambda_adapter_reg must be non-negative")
        if self.close_pos_weight_max < 1.0:
            raise ValueError("close_pos_weight_max must be at least 1")


@dataclass(frozen=True)
class MapperV1LossOutput:
    total_loss: torch.Tensor
    metrics: dict[str, float]
    metric_numerators: dict[str, float]
    metric_denominators: dict[str, float]
    model_output: MapperV1ModelOutput


@dataclass(frozen=True)
class MapperV1ControlTeacherCachePrecomputeResult:
    cache_dir: Path
    total_entries: int
    computed_entries: int
    skipped_entries: int
    elapsed_s: float

    def to_report(self) -> dict[str, Any]:
        return {
            "cache_dir": self.cache_dir.as_posix(),
            "total_entries": self.total_entries,
            "computed_entries": self.computed_entries,
            "skipped_entries": self.skipped_entries,
            "elapsed_s": self.elapsed_s,
        }


@dataclass(frozen=True)
class MapperV1ControlTeacherCachePrecomputeRunResult:
    reports: list[dict[str, Any]]
    source_control_dataset: ControlWindowDataset
    eval_control_dataset: ControlWindowDataset | None


def load_run_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML run config: {path}") from exc
    if loaded is None:
        return {"model": {}, "control_model": {}, "loss": {}}
    if not isinstance(loaded, dict):
        raise ValueError(f"run config must be a mapping: {path}")

    config = _normalize_config_mapping(loaded, source_name="run config")
    unknown = sorted(set(config) - RUN_CONFIG_KEYS)
    if unknown:
        raise ValueError(f"unknown run config keys: {unknown}")
    config["model"] = _normalized_section(config.get("model", {}), allowed=MODEL_CONFIG_KEYS, name="model config")
    config["control_model"] = _normalized_section(
        config.get("control_model", {}),
        allowed=CONTROL_MODEL_CONFIG_KEYS,
        name="control model config",
    )
    config["loss"] = _normalized_section(config.get("loss", {}), allowed=LOSS_CONFIG_KEYS, name="loss config")
    return config


def run_synthetic_smoke(
    *,
    output_dir: Path = DEFAULT_RUNS_ROOT / "synthetic_phase_b_smoke",
    max_steps: int = 2,
    eval_every: int | None = None,
    save_every: int | None = None,
    log_every: int | None = None,
    batch_size: int = 2,
    learning_rate: float = 1e-2,
    seed: int = 1337,
    device_name: str = "auto",
    final_train_eval_size: int | None = DEFAULT_FINAL_TRAIN_EVAL_SIZE,
    model_config_overrides: Mapping[str, Any] | None = None,
    loss_config_overrides: Mapping[str, Any] | None = None,
) -> ControlTrainingResult:
    model_defaults: dict[str, Any] = {
        "control_dim": 32,
        "d_model": 32,
        "heads": 4,
        "layers": 1,
        "ffn_dim": 64,
        "dropout": 0.0,
        "max_seq_len": 32,
        "state_hidden_dim": 32,
        "lane_embedding_dim": 8,
        "skip_scale": 0.0,
    }
    model_defaults.update(dict(model_config_overrides or {}))
    model_config = MapperV1Config(**model_defaults)
    loss_config = MapperV1PhaseBLossConfig(**dict(loss_config_overrides or {}))
    samples = _synthetic_mapper_samples(model_config=model_config)
    train_eval_dataset = limit_final_train_eval_dataset(
        samples,
        final_train_eval_size=final_train_eval_size,
        seed=seed,
    )
    loader = DataLoader(samples, batch_size=batch_size, shuffle=False, collate_fn=_collate_synthetic_mapper_samples)
    train_eval_loader = DataLoader(
        train_eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate_synthetic_mapper_samples,
    )
    return _run_training(
        loader=loader,
        train_eval_loader=train_eval_loader,
        eval_loader=loader,
        output_dir=output_dir,
        model_config=model_config,
        control_model_config=None,
        loss_config=loss_config,
        max_steps=max_steps,
        eval_every=max(1, max_steps) if eval_every is None else eval_every,
        save_every=save_every,
        log_every=log_every,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.0,
        seed=seed,
        device_name=device_name,
        run_name="stage2_mapper_v1_phase_b_synthetic_smoke",
        dataset_report={
            "status": "synthetic_smoke",
            "sample_count": len(samples),
            "final_train_eval_size": final_train_eval_size,
            "final_train_eval_window_count": len(train_eval_dataset),
        },
        init_from_control_checkpoint=None,
    )


def run_mapper_v1_phase_b_training(
    *,
    dataset_root: Path = Path("mania-dataset"),
    index_path: Path | None = None,
    eval_index_path: Path | None = None,
    control_v3_timeseries_path: Path | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    max_steps: int = 5000,
    eval_every: int = 100,
    save_every: int | None = None,
    log_every: int | None = None,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    weight_decay: float = 0.01,
    seed: int = 1337,
    device_name: str = "auto",
    run_name: str = "mapper_v1_phase_b_teacher_forced",
    init_from_control_checkpoint: Path | None = None,
    eval_fraction: float = 0.1,
    eval_size: int | None = None,
    final_train_eval_size: int | None = DEFAULT_FINAL_TRAIN_EVAL_SIZE,
    num_workers: int = 0,
    max_cached_maps: int | None = None,
    dataset_progress: bool | None = None,
    control_teacher_cache_dir: Path | None = None,
    precompute_control_teacher_cache: bool = False,
    control_teacher_precompute_batch_size: int | None = None,
    require_control_teacher_cache: bool = False,
    control_teacher_cache_overwrite: bool = False,
    model_config_overrides: Mapping[str, Any] | None = None,
    control_model_config_overrides: Mapping[str, Any] | None = None,
    loss_config_overrides: Mapping[str, Any] | None = None,
) -> ControlTrainingResult:
    _set_deterministic_seed(seed)
    control_model_config = ControlDemoGlobalEncoderConfig(**dict(control_model_config_overrides or {}))
    model_values = dict(model_config_overrides or {})
    model_values.setdefault("control_dim", control_model_config.d_model)
    model_config = MapperV1Config(**model_values)
    if model_config.control_dim != control_model_config.d_model:
        raise ValueError("mapper control_dim must match frozen control_model d_model")
    loss_config = MapperV1PhaseBLossConfig(**dict(loss_config_overrides or {}))

    dataset_kwargs: dict[str, Any] = {"dataset_root": dataset_root}
    if index_path is not None:
        dataset_kwargs["index_path"] = index_path
    if control_v3_timeseries_path is not None:
        dataset_kwargs["control_v3_timeseries_path"] = control_v3_timeseries_path
    effective_max_cached_maps = DEFAULT_MAX_CACHED_MAPS if max_cached_maps is None else max_cached_maps
    dataset_kwargs["max_cached_maps"] = effective_max_cached_maps
    effective_dataset_progress = bool(precompute_control_teacher_cache) if dataset_progress is None else bool(dataset_progress)
    dataset_kwargs["progress"] = effective_dataset_progress

    cache_precompute_reports: list[dict[str, Any]] = []
    source_control_dataset: ControlWindowDataset | None = None
    eval_control_dataset: ControlWindowDataset | None = None
    if precompute_control_teacher_cache:
        precompute_run = precompute_mapper_v1_phase_b_control_teacher_cache(
            dataset_root=dataset_root,
            index_path=index_path,
            eval_index_path=eval_index_path,
            control_v3_timeseries_path=control_v3_timeseries_path,
            batch_size=batch_size,
            seed=seed,
            device_name=device_name,
            init_from_control_checkpoint=init_from_control_checkpoint,
            num_workers=num_workers,
            max_cached_maps=max_cached_maps,
            dataset_progress=effective_dataset_progress,
            control_teacher_cache_dir=control_teacher_cache_dir,
            control_teacher_precompute_batch_size=control_teacher_precompute_batch_size,
            control_teacher_cache_overwrite=control_teacher_cache_overwrite,
            control_model_config=control_model_config,
        )
        cache_precompute_reports = precompute_run.reports
        source_control_dataset = precompute_run.source_control_dataset
        eval_control_dataset = precompute_run.eval_control_dataset

    mapper_dataset_kwargs: dict[str, Any]
    if source_control_dataset is None:
        mapper_dataset_kwargs = dict(dataset_kwargs)
    else:
        mapper_dataset_kwargs = {"control_dataset": source_control_dataset, "progress": effective_dataset_progress}
    if control_teacher_cache_dir is not None:
        mapper_dataset_kwargs["control_teacher_cache_dir"] = control_teacher_cache_dir
        mapper_dataset_kwargs["require_control_teacher_cache"] = bool(require_control_teacher_cache)
    train_source = MapperV1WindowDataset(**mapper_dataset_kwargs)
    if len(train_source) == 0:
        raise ValueError("MapperV1WindowDataset produced no training windows")

    if eval_index_path is not None:
        if eval_control_dataset is None:
            eval_kwargs = dict(dataset_kwargs)
            eval_kwargs["index_path"] = eval_index_path
        else:
            eval_kwargs = {"control_dataset": eval_control_dataset, "progress": effective_dataset_progress}
        if control_teacher_cache_dir is not None:
            eval_kwargs["control_teacher_cache_dir"] = control_teacher_cache_dir
            eval_kwargs["require_control_teacher_cache"] = bool(require_control_teacher_cache)
        eval_dataset: Dataset[Any] = MapperV1WindowDataset(**eval_kwargs)
        train_dataset: Dataset[Any] = train_source
    else:
        train_dataset, eval_dataset = split_train_eval_dataset(
            train_source,
            eval_fraction=eval_fraction,
            eval_size=eval_size,
            seed=seed,
        )
    if len(train_dataset) == 0:
        raise ValueError("training split is empty")
    if len(eval_dataset) == 0:
        eval_dataset = train_dataset

    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=num_workers,
        collate_fn=collate_mapper_v1_windows,
    )
    train_eval_dataset = limit_final_train_eval_dataset(
        train_dataset,
        final_train_eval_size=final_train_eval_size,
        seed=seed,
    )
    train_eval_loader = DataLoader(
        train_eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_mapper_v1_windows,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_mapper_v1_windows,
    )
    return _run_training(
        loader=loader,
        train_eval_loader=train_eval_loader,
        eval_loader=eval_loader,
        output_dir=output_dir,
        model_config=model_config,
        control_model_config=control_model_config,
        loss_config=loss_config,
        max_steps=max_steps,
        eval_every=eval_every,
        save_every=save_every,
        log_every=log_every,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        seed=seed,
        device_name=device_name,
        run_name=run_name,
        dataset_report={
            "train_window_count": len(train_dataset),
            "eval_window_count": len(eval_dataset),
            "source_window_count": len(train_source),
            "eval_index_path": eval_index_path.as_posix() if eval_index_path is not None else None,
            "eval_fraction": eval_fraction,
            "eval_size": eval_size,
            "final_train_eval_size": final_train_eval_size,
            "final_train_eval_window_count": len(train_eval_dataset),
            "filter_report": asdict(train_source.filter_report),
            "max_cached_maps": int(getattr(train_source.control_dataset, "max_cached_maps", effective_max_cached_maps)),
            "dataset_progress": bool(effective_dataset_progress),
            "num_workers": num_workers,
            "control_teacher_cache_dir": (
                control_teacher_cache_dir.as_posix() if control_teacher_cache_dir is not None else None
            ),
            "precompute_control_teacher_cache": bool(precompute_control_teacher_cache),
            "control_teacher_precompute_batch_size": control_teacher_precompute_batch_size,
            "require_control_teacher_cache": bool(require_control_teacher_cache),
            "control_teacher_cache_overwrite": bool(control_teacher_cache_overwrite),
            "control_teacher_cache_precompute": cache_precompute_reports,
        },
        init_from_control_checkpoint=init_from_control_checkpoint,
    )


class _ControlTeacherPrecomputeDataset(Dataset[Any]):
    def __init__(
        self,
        control_dataset: Dataset[Any],
        indexed_records: Sequence[tuple[int, ControlWindowRecord]],
    ) -> None:
        self.control_dataset = control_dataset
        self.indexed_records = [
            (int(control_record_index), record)
            for control_record_index, record in indexed_records
        ]

    def __len__(self) -> int:
        return len(self.indexed_records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        control_record_index, record = self.indexed_records[index]
        control_dataset = self.control_dataset
        full_mel_loader = getattr(control_dataset, "_load_full_mel", None)
        dense_timing_loader = getattr(control_dataset, "_load_dense_timing_v2", None)
        if callable(full_mel_loader) and callable(dense_timing_loader):
            full_mel = full_mel_loader(record.audio_path, expected_frame_count=record.frame_count)
            full_dense_timing_v2 = dense_timing_loader(record.beatmap_path, frame_count=record.frame_count)
        else:
            base_sample = control_dataset[control_record_index]
            full_mel = base_sample["full_mel"]
            full_dense_timing_v2 = base_sample["full_dense_timing_v2"]
        return {
            "full_mel": torch.as_tensor(full_mel, dtype=torch.float32),
            "full_dense_timing_v2": torch.as_tensor(full_dense_timing_v2, dtype=torch.float32),
            "frame_count": torch.tensor(record.frame_count, dtype=torch.long),
            "control_slice_start_frames": torch.tensor(
                [
                    record.target_start_frame + offset
                    for offset in range(0, MAPPER_WRITE_FRAMES, TARGET_WINDOW_LENGTH_FRAMES)
                ],
                dtype=torch.long,
            ),
            "difficulty": torch.tensor(record.difficulty, dtype=torch.float32),
            "normalized_difficulty": torch.tensor(normalize_difficulty(record.difficulty), dtype=torch.float32),
            "metadata": {
                "beatmap_path": record.beatmap_path.as_posix(),
                "audio_path": record.audio_path.as_posix(),
                "difficulty": record.difficulty,
                "target_start_frame": record.target_start_frame,
                "target_start_ms": record.target_start_ms,
                "control_record_index": control_record_index,
            },
        }


def _collate_mapper_v1_control_teacher_precompute(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("_collate_mapper_v1_control_teacher_precompute requires at least one sample")
    batch_size = len(samples)
    frame_counts = [int(sample["frame_count"].item()) for sample in samples]
    max_frame_count = max(frame_counts)
    full_mel = torch.zeros((batch_size, max_frame_count, PACKED_MEL_CHANNELS), dtype=torch.float32)
    full_dense_timing_v2 = torch.zeros(
        (batch_size, max_frame_count, DENSE_TIMING_V2_CHANNELS),
        dtype=torch.float32,
    )
    padding_mask = torch.ones((batch_size, max_frame_count), dtype=torch.bool)
    for batch_index, sample in enumerate(samples):
        frame_count = frame_counts[batch_index]
        sample_full_mel = sample["full_mel"].to(dtype=torch.float32)
        sample_full_dense_timing_v2 = sample["full_dense_timing_v2"].to(dtype=torch.float32)
        if tuple(sample_full_mel.shape) != (frame_count, PACKED_MEL_CHANNELS):
            raise ValueError(
                f"full_mel sample {batch_index} must have shape {(frame_count, PACKED_MEL_CHANNELS)}"
            )
        if tuple(sample_full_dense_timing_v2.shape) != (frame_count, DENSE_TIMING_V2_CHANNELS):
            raise ValueError(
                "full_dense_timing_v2 sample "
                f"{batch_index} must have shape {(frame_count, DENSE_TIMING_V2_CHANNELS)}"
            )
        full_mel[batch_index, :frame_count] = sample_full_mel
        full_dense_timing_v2[batch_index, :frame_count] = sample_full_dense_timing_v2
        padding_mask[batch_index, :frame_count] = False
    return {
        "full_mel": full_mel,
        "full_dense_timing_v2": full_dense_timing_v2,
        "padding_mask": padding_mask,
        "frame_count": torch.tensor(frame_counts, dtype=torch.long),
        "control_slice_start_frames": torch.stack(
            [sample["control_slice_start_frames"].to(dtype=torch.long) for sample in samples],
        ),
        "difficulty": torch.stack([sample["difficulty"].to(dtype=torch.float32) for sample in samples]).reshape(
            batch_size,
        ),
        "normalized_difficulty": torch.stack(
            [sample["normalized_difficulty"].to(dtype=torch.float32) for sample in samples],
        ).reshape(batch_size),
        "metadata": [sample.get("metadata", {}) for sample in samples],
    }


def _release_torch_device_cache(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        return
    if device.type != "mps" or not hasattr(torch, "mps"):
        return
    synchronize = getattr(torch.mps, "synchronize", None)
    if synchronize is not None:
        synchronize()
    empty_cache = getattr(torch.mps, "empty_cache", None)
    if empty_cache is not None:
        empty_cache()


def precompute_mapper_v1_phase_b_control_teacher_cache(
    *,
    dataset_root: Path = Path("mania-dataset"),
    index_path: Path | None = None,
    eval_index_path: Path | None = None,
    control_v3_timeseries_path: Path | None = None,
    batch_size: int = 4,
    seed: int = 1337,
    device_name: str = "auto",
    init_from_control_checkpoint: Path | None = None,
    num_workers: int = 0,
    max_cached_maps: int | None = None,
    dataset_progress: bool | None = None,
    control_teacher_cache_dir: Path | None = None,
    control_teacher_precompute_batch_size: int | None = None,
    control_teacher_cache_overwrite: bool = False,
    control_model_config: ControlDemoGlobalEncoderConfig | None = None,
    control_model_config_overrides: Mapping[str, Any] | None = None,
) -> MapperV1ControlTeacherCachePrecomputeRunResult:
    _set_deterministic_seed(seed)
    if control_teacher_cache_dir is None:
        raise ValueError("control teacher cache precompute requires control_teacher_cache_dir")
    effective_precompute_batch_size = (
        batch_size if control_teacher_precompute_batch_size is None else int(control_teacher_precompute_batch_size)
    )
    if effective_precompute_batch_size <= 0:
        raise ValueError("control_teacher_precompute_batch_size must be positive")

    dataset_kwargs: dict[str, Any] = {"dataset_root": dataset_root}
    if index_path is not None:
        dataset_kwargs["index_path"] = index_path
    if control_v3_timeseries_path is not None:
        dataset_kwargs["control_v3_timeseries_path"] = control_v3_timeseries_path
    effective_max_cached_maps = DEFAULT_MAX_CACHED_MAPS if max_cached_maps is None else max_cached_maps
    dataset_kwargs["max_cached_maps"] = effective_max_cached_maps
    dataset_kwargs["progress"] = True if dataset_progress is None else bool(dataset_progress)

    resolved_control_model_config = (
        ControlDemoGlobalEncoderConfig(**dict(control_model_config_overrides or {}))
        if control_model_config is None
        else control_model_config
    )
    precompute_device = select_torch_device(device_name)
    precompute_encoder = ControlDemoGlobalEncoder(resolved_control_model_config).to(precompute_device)
    try:
        if init_from_control_checkpoint is not None:
            initialize_global_control_demo_from_control_checkpoint(
                precompute_encoder,
                init_from_control_checkpoint,
            )
        source_control_dataset = ControlWindowDataset(**dataset_kwargs)
        source_result = precompute_phase_b_control_teacher_cache_from_control_dataset(
            source_control_dataset,
            cache_dir=control_teacher_cache_dir,
            control_encoder=precompute_encoder,
            batch_size=effective_precompute_batch_size,
            device=precompute_device,
            num_workers=num_workers,
            overwrite=control_teacher_cache_overwrite,
        )
        reports = [{"split": "source", **source_result.to_report()}]
        eval_control_dataset: ControlWindowDataset | None = None
        if eval_index_path is not None:
            eval_dataset_kwargs = dict(dataset_kwargs)
            eval_dataset_kwargs["index_path"] = eval_index_path
            eval_control_dataset = ControlWindowDataset(**eval_dataset_kwargs)
            eval_result = precompute_phase_b_control_teacher_cache_from_control_dataset(
                eval_control_dataset,
                cache_dir=control_teacher_cache_dir,
                control_encoder=precompute_encoder,
                batch_size=effective_precompute_batch_size,
                device=precompute_device,
                num_workers=num_workers,
                overwrite=control_teacher_cache_overwrite,
            )
            reports.append({"split": "eval", **eval_result.to_report()})
    finally:
        del precompute_encoder
        _release_torch_device_cache(precompute_device)

    return MapperV1ControlTeacherCachePrecomputeRunResult(
        reports=reports,
        source_control_dataset=source_control_dataset,
        eval_control_dataset=eval_control_dataset,
    )


def _mapper_v1_raw_control_indexed_records(
    control_dataset: Dataset[Any],
    *,
    mapper_stride_frames: int = MAPPER_WRITE_FRAMES,
) -> list[tuple[int, ControlWindowRecord]]:
    records = getattr(control_dataset, "records", None)
    if not isinstance(records, Sequence):
        raise TypeError("control teacher cache precompute requires a dataset with records")
    indexed_records: list[tuple[int, ControlWindowRecord]] = []
    skipped_stride = 0
    skipped_short = 0
    for index, record in enumerate(records):
        if not isinstance(record, ControlWindowRecord):
            raise TypeError(f"control dataset record {index} must be a ControlWindowRecord")
        if record.target_start_frame % mapper_stride_frames != 0:
            skipped_stride += 1
            continue
        if record.target_start_frame + MAPPER_WRITE_FRAMES > record.frame_count:
            skipped_short += 1
            continue
        indexed_records.append((index, record))
    print(
        "mapper_v1_control_teacher_cache_precompute raw_control_select "
        f"source_windows={len(records)} selected_windows={len(indexed_records)} "
        f"skipped_stride={skipped_stride} skipped_short={skipped_short}",
        flush=True,
    )
    return indexed_records


def _precompute_phase_b_control_teacher_cache_for_indexed_records(
    *,
    control_dataset: Dataset[Any],
    indexed_records: Sequence[tuple[int, ControlWindowRecord]],
    cache_dir: Path,
    control_encoder: ControlDemoGlobalEncoder,
    batch_size: int,
    device: torch.device,
    num_workers: int,
    overwrite: bool,
    source_label: str,
) -> MapperV1ControlTeacherCachePrecomputeResult:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        control_teacher_cache_path(cache_dir, record)
        for _, record in indexed_records
    ]
    missing_indices = [
        index
        for index, path in enumerate(paths)
        if overwrite or not path.exists()
    ]
    skipped_entries = len(paths) - len(missing_indices)
    start_time = time.monotonic()
    print(
        f"mapper_v1_control_teacher_cache_precompute start source={source_label} "
        f"total={len(paths)} missing={len(missing_indices)} cache_dir={cache_dir.as_posix()}",
        flush=True,
    )
    if not missing_indices:
        elapsed_s = time.monotonic() - start_time
        print(
            f"mapper_v1_control_teacher_cache_precompute done source={source_label} "
            f"computed=0 skipped={skipped_entries} elapsed_s={elapsed_s:.1f}",
            flush=True,
        )
        return MapperV1ControlTeacherCachePrecomputeResult(
            cache_dir=cache_dir,
            total_entries=len(paths),
            computed_entries=0,
            skipped_entries=skipped_entries,
            elapsed_s=elapsed_s,
        )

    computed_entries = 0
    loader = DataLoader(
        _ControlTeacherPrecomputeDataset(
            control_dataset,
            [indexed_records[index] for index in missing_indices],
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate_mapper_v1_control_teacher_precompute,
    )
    control_encoder.to(device)
    control_encoder.eval()
    offset = 0
    for raw_batch in loader:
        current_batch_size = int(raw_batch["control_slice_start_frames"].shape[0])
        batch_indices = missing_indices[offset : offset + current_batch_size]
        offset += current_batch_size
        batch = _move_batch_tensors(raw_batch, device, keys=MAPPER_BATCH_TENSOR_KEYS)
        with torch.no_grad():
            control_memory_8s, density_teacher_8s = compute_control_teacher_8s(
                control_encoder,
                batch,
                stack_slices=True,
            )
        control_memory_8s = control_memory_8s.detach().cpu()
        density_teacher_8s = density_teacher_8s.detach().cpu()
        for batch_index, record_index in enumerate(batch_indices):
            record = indexed_records[record_index][1]
            save_control_teacher_cache_entry(
                paths[record_index],
                record=record,
                control_memory_8s=control_memory_8s[batch_index],
                density_teacher_8s=density_teacher_8s[batch_index],
            )
            computed_entries += 1
        if computed_entries == len(batch_indices) or computed_entries % max(batch_size * 25, 1) == 0:
            print(
                f"mapper_v1_control_teacher_cache_precompute progress source={source_label} "
                f"computed={computed_entries}/{len(missing_indices)}",
                flush=True,
            )

    elapsed_s = time.monotonic() - start_time
    print(
        f"mapper_v1_control_teacher_cache_precompute done source={source_label} "
        f"computed={computed_entries} skipped={skipped_entries} elapsed_s={elapsed_s:.1f}",
        flush=True,
    )
    return MapperV1ControlTeacherCachePrecomputeResult(
        cache_dir=cache_dir,
        total_entries=len(paths),
        computed_entries=computed_entries,
        skipped_entries=skipped_entries,
        elapsed_s=elapsed_s,
    )


def precompute_phase_b_control_teacher_cache_from_control_dataset(
    control_dataset: Dataset[Any],
    *,
    cache_dir: Path,
    control_encoder: ControlDemoGlobalEncoder,
    batch_size: int = 4,
    device: torch.device,
    num_workers: int = 0,
    overwrite: bool = False,
) -> MapperV1ControlTeacherCachePrecomputeResult:
    indexed_records = _mapper_v1_raw_control_indexed_records(control_dataset)
    return _precompute_phase_b_control_teacher_cache_for_indexed_records(
        control_dataset=control_dataset,
        indexed_records=indexed_records,
        cache_dir=cache_dir,
        control_encoder=control_encoder,
        batch_size=batch_size,
        device=device,
        num_workers=num_workers,
        overwrite=overwrite,
        source_label="raw_control",
    )


def precompute_phase_b_control_teacher_cache(
    dataset: MapperV1WindowDataset,
    *,
    cache_dir: Path,
    control_encoder: ControlDemoGlobalEncoder,
    batch_size: int = 4,
    device: torch.device,
    num_workers: int = 0,
    overwrite: bool = False,
) -> MapperV1ControlTeacherCachePrecomputeResult:
    if not isinstance(dataset, MapperV1WindowDataset):
        raise TypeError("precompute_phase_b_control_teacher_cache requires a MapperV1WindowDataset")
    result = _precompute_phase_b_control_teacher_cache_for_indexed_records(
        control_dataset=dataset.control_dataset,
        indexed_records=[
            (mapper_record.control_record_index, mapper_record.control_record)
            for mapper_record in dataset.records
        ],
        cache_dir=cache_dir,
        control_encoder=control_encoder,
        batch_size=batch_size,
        device=device,
        num_workers=num_workers,
        overwrite=overwrite,
        source_label="mapper_filtered",
    )
    dataset.control_teacher_cache_dir = Path(cache_dir)
    return result


def compute_phase_b_loss(
    model_output: MapperV1ModelOutput,
    *,
    target_tokens: torch.Tensor,
    target_token_mask: torch.Tensor | None = None,
    close_labels: torch.Tensor,
    close_label_mask: torch.Tensor,
    vocab: MapperV1Vocab,
    loss_config: MapperV1PhaseBLossConfig = MapperV1PhaseBLossConfig(),
) -> MapperV1LossOutput:
    loss_target = target_tokens[:, 1:].to(dtype=torch.long, device=model_output.logits_final.device)
    if tuple(loss_target.shape) != tuple(model_output.logits_final.shape[:2]):
        raise ValueError("loss target must align with teacher-forced decoder output")
    if target_token_mask is None:
        target_mask = loss_target != vocab.pad_id
        input_mask = target_tokens[:, :-1].to(device=model_output.logits_final.device, dtype=torch.long) != vocab.pad_id
    else:
        if tuple(target_token_mask.shape) != tuple(target_tokens.shape):
            raise ValueError("target_token_mask must match target_tokens")
        target_mask = target_token_mask[:, 1:].to(device=model_output.logits_final.device, dtype=torch.bool)
        input_mask = (
            target_token_mask[:, :-1].to(device=model_output.logits_final.device, dtype=torch.bool)
            & target_mask
        )
    token_loss = token_cross_entropy(
        model_output.logits_final,
        loss_target,
        pad_id=vocab.pad_id,
        target_mask=target_mask,
    )
    close_loss, close_metrics = _ln_close_loss(
        close_logits=model_output.close_logits,
        labels=close_labels[:, :-1].to(device=model_output.close_logits.device),
        mask=close_label_mask[:, :-1].to(device=model_output.close_logits.device),
        max_pos_weight=loss_config.close_pos_weight_max,
    )
    adapter_reg = adapter_bias_regularization(
        model_output.state_prior_bias,
        model_output.ln_close_bias,
        model_output.time_shift_bias,
        mask=input_mask,
    )
    density_loss = token_loss.new_zeros(())
    total_loss = (
        token_loss
        + float(loss_config.lambda_ln_close) * close_loss
        + float(loss_config.lambda_adapter_reg) * adapter_reg
        + density_loss
    )
    metrics = {
        "loss/total": float(total_loss.detach().cpu()),
        "loss/token": float(token_loss.detach().cpu()),
        "loss/ln_close": float(close_loss.detach().cpu()),
        "loss/adapter_reg": float(adapter_reg.detach().cpu()),
        "loss/density": 0.0,
        "target/token_count": float(target_mask.sum().detach().cpu()),
        **close_metrics,
    }
    return MapperV1LossOutput(
        total_loss=total_loss,
        metrics=metrics,
        metric_numerators={
            "loss/token": float(token_loss.detach().cpu()) * max(metrics["target/token_count"], 1.0),
            "loss/ln_close": float(close_loss.detach().cpu()) * max(metrics["ln_close/open_lane_count"], 1.0),
        },
        metric_denominators={
            "loss/token": max(metrics["target/token_count"], 1.0),
            "loss/ln_close": max(metrics["ln_close/open_lane_count"], 1.0),
        },
        model_output=model_output,
    )


def _loss_for_raw_batch(
    model: MapperV1Model,
    raw_batch: Mapping[str, Any],
    *,
    device: torch.device,
    loss_config: MapperV1PhaseBLossConfig | None = None,
) -> MapperV1LossOutput:
    batch = _move_batch_tensors(raw_batch, device, keys=MAPPER_BATCH_TENSOR_KEYS)
    if isinstance(batch.get("control_memory_padding_mask_8s"), torch.Tensor):
        raise ValueError("control_memory_padding_mask_8s is not supported in Phase B")
    model_output = model(
        target_tokens=batch["target_tokens"],
        target_token_mask=batch.get("target_token_mask"),
        teacher_current_ms=batch["teacher_current_ms"],
        teacher_open_mask=batch["teacher_open_mask"],
        teacher_open_age_ms=batch["teacher_open_age_ms"],
        write_start_ms=batch["write_start_ms"],
        write_end_ms=batch["write_end_ms"],
        difficulty=batch.get("difficulty"),
        normalized_difficulty=batch.get("normalized_difficulty"),
        control_memory_8s=batch.get("control_memory_8s"),
        density_teacher_8s=batch.get("density_teacher_8s"),
        full_mel=batch.get("full_mel"),
        full_dense_timing_v2=batch.get("full_dense_timing_v2"),
        padding_mask=batch.get("padding_mask"),
        frame_count=batch.get("frame_count"),
        control_slice_start_frames=batch.get("control_slice_start_frames"),
    )
    return compute_phase_b_loss(
        model_output,
        target_tokens=batch["target_tokens"],
        target_token_mask=batch.get("target_token_mask"),
        close_labels=batch["close_labels"],
        close_label_mask=batch["close_label_mask"],
        vocab=model.vocab,
        loss_config=MapperV1PhaseBLossConfig() if loss_config is None else loss_config,
    )


@torch.no_grad()
def metrics_for_loader(
    model: MapperV1Model,
    loader: DataLoader,
    *,
    device: torch.device,
    loss_config: MapperV1PhaseBLossConfig,
) -> dict[str, float]:
    model.eval()
    count_totals: dict[str, float] = {}
    mean_numerators: dict[str, float] = {}
    mean_denominators: dict[str, float] = {}
    fallback_totals: dict[str, float] = {}
    fallback_weights: dict[str, float] = {}
    for raw_batch in loader:
        loss_output = _loss_for_raw_batch(model, raw_batch, device=device, loss_config=loss_config)
        for key, value in loss_output.metrics.items():
            if _metric_is_count(key):
                count_totals[key] = count_totals.get(key, 0.0) + float(value)
        for key, numerator in loss_output.metric_numerators.items():
            mean_numerators[key] = mean_numerators.get(key, 0.0) + float(numerator)
        for key, denominator in loss_output.metric_denominators.items():
            mean_denominators[key] = mean_denominators.get(key, 0.0) + float(denominator)
        unresolved = set(loss_output.metrics) - set(loss_output.metric_numerators) - set(count_totals)
        weight = max(float(loss_output.metrics.get("target/token_count", 0.0)), 1.0)
        for key in unresolved:
            fallback_totals[key] = fallback_totals.get(key, 0.0) + float(loss_output.metrics[key]) * weight
            fallback_weights[key] = fallback_weights.get(key, 0.0) + weight
    if not (count_totals or mean_numerators or fallback_totals):
        return {"loss/total": math.nan, "loss/token": math.nan, "loss/density": 0.0}
    metrics = dict(count_totals)
    for key, numerator in mean_numerators.items():
        metrics[key] = _safe_float_div(numerator, mean_denominators.get(key, 0.0))
    for key, total in fallback_totals.items():
        metrics[key] = _safe_float_div(total, fallback_weights[key])
    metrics["loss/total"] = (
        metrics.get("loss/token", 0.0)
        + loss_config.lambda_ln_close * metrics.get("loss/ln_close", 0.0)
        + loss_config.lambda_adapter_reg * metrics.get("loss/adapter_reg", 0.0)
    )
    metrics["loss/density"] = 0.0
    return metrics


def _run_training(
    *,
    loader: DataLoader,
    train_eval_loader: DataLoader,
    eval_loader: DataLoader,
    output_dir: Path,
    model_config: MapperV1Config,
    control_model_config: ControlDemoGlobalEncoderConfig | None,
    loss_config: MapperV1PhaseBLossConfig,
    max_steps: int,
    eval_every: int,
    save_every: int | None,
    log_every: int | None,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    device_name: str,
    run_name: str,
    dataset_report: Mapping[str, Any],
    init_from_control_checkpoint: Path | None,
) -> ControlTrainingResult:
    _validate_training_args(
        max_steps=max_steps,
        eval_every=eval_every,
        save_every=save_every,
        log_every=log_every,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    _set_deterministic_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = select_torch_device(device_name)
    save_every = eval_every if save_every is None else save_every
    checkpoint_path = output_dir / "checkpoint.pt"
    report_path = output_dir / "report.json"

    control_encoder: ControlDemoGlobalEncoder | None = None
    initialization_report: dict[str, Any] | None = None
    if control_model_config is not None:
        control_encoder = ControlDemoGlobalEncoder(control_model_config)
        if init_from_control_checkpoint is not None:
            initialization_report = initialize_global_control_demo_from_control_checkpoint(
                control_encoder,
                init_from_control_checkpoint,
            )
    model = MapperV1Model(model_config, control_encoder=control_encoder).to(device)
    optimizer = _build_mapper_v1_optimizer(model, learning_rate=learning_rate, weight_decay=weight_decay)
    iterator = _infinite_loader(loader)
    history: list[dict[str, Any]] = []
    completed_step = 0
    last_train_metrics: dict[str, float] = {}
    final_train_metrics: dict[str, float] = {}
    final_eval_metrics: dict[str, float] = {}

    log_start_time = time.monotonic()
    log_start_step = completed_step
    for step in range(1, max_steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_output = _loss_for_raw_batch(model, next(iterator), device=device, loss_config=loss_config)
        loss_output.total_loss.backward()
        torch.nn.utils.clip_grad_norm_((parameter for parameter in model.parameters() if parameter.requires_grad), 1.0)
        optimizer.step()
        last_train_metrics = dict(loss_output.metrics)
        completed_step = step

        should_eval = step == 1 or step % eval_every == 0 or step == max_steps
        should_save = step == 1 or step % save_every == 0 or step == max_steps
        should_log = log_every is not None and (step == 1 or step % log_every == 0 or step == max_steps)
        if should_log or should_eval or should_save:
            elapsed_s = time.monotonic() - log_start_time
            completed_since_start = max(step - log_start_step, 1)
            steps_per_s = completed_since_start / max(elapsed_s, 1e-9)
            print(
                f"mapper_v1_phase_b_progress step={step}/{max_steps} "
                f"loss={last_train_metrics['loss/total']:.6f} "
                f"elapsed_s={elapsed_s:.1f} steps_per_s={steps_per_s:.3f}",
                flush=True,
            )
        if should_eval:
            final_eval_metrics = metrics_for_loader(model, eval_loader, device=device, loss_config=loss_config)
            history_entry: dict[str, Any] = {
                "step": step,
                "train": _json_metrics(last_train_metrics),
                "eval": _json_metrics(final_eval_metrics),
            }
            if step == max_steps:
                final_train_metrics = metrics_for_loader(model, train_eval_loader, device=device, loss_config=loss_config)
                history_entry["train_eval"] = _json_metrics(final_train_metrics)
            history.append(history_entry)
            print(
                f"mapper_v1_phase_b_eval step={step}/{max_steps} "
                f"loss={final_eval_metrics.get('loss/total', float('nan')):.6f}",
                flush=True,
            )
        if should_save:
            _write_checkpoint_and_report(
                output_dir=output_dir,
                checkpoint_path=checkpoint_path,
                report_path=report_path,
                model=model,
                optimizer=optimizer,
                model_config=model_config,
                control_model_config=control_model_config,
                loss_config=loss_config,
                seed=seed,
                run_name=run_name,
                max_steps=max_steps,
                completed_steps=completed_step,
                eval_every=eval_every,
                save_every=save_every,
                log_every=log_every,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                device=device,
                dataset_report=dataset_report,
                history=history,
                last_train_metrics=last_train_metrics,
                final_train_metrics=final_train_metrics,
                final_eval_metrics=final_eval_metrics,
                initialization_report=initialization_report,
            )

    result_metrics = final_eval_metrics or last_train_metrics
    return ControlTrainingResult(
        report_path=report_path,
        checkpoint_path=checkpoint_path,
        final_loss=float(result_metrics.get("loss/total", float("nan"))),
        final_value_loss=float(result_metrics.get("loss/token", float("nan"))),
        final_confidence_loss=0.0,
        completed_steps=completed_step,
    )


def _build_mapper_v1_optimizer(
    model: MapperV1Model,
    *,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.AdamW:
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not params:
        raise ValueError("mapper model has no trainable parameters")
    return torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)


def _ln_close_loss(
    *,
    close_logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    max_pos_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    if tuple(labels.shape) != tuple(close_logits.shape) or tuple(mask.shape) != tuple(close_logits.shape):
        raise ValueError("close labels and mask must align with close_logits")
    mask = mask.to(dtype=torch.bool)
    open_count = int(mask.sum().detach().cpu().item())
    if open_count == 0:
        zero = close_logits.sum() * 0.0
        return zero, {
            "ln_close/open_lane_count": 0.0,
            "ln_close/positive_count": 0.0,
            "ln_close/pos_weight": 1.0,
        }
    labels_f = labels.to(dtype=close_logits.dtype)
    positive_count = float((labels_f * mask.to(dtype=labels_f.dtype)).sum().detach().cpu().item())
    negative_count = max(float(open_count) - positive_count, 0.0)
    pos_weight = 1.0 if positive_count <= 0.0 else min(max(negative_count / positive_count, 1.0), max_pos_weight)
    loss = F.binary_cross_entropy_with_logits(
        close_logits,
        labels_f,
        pos_weight=close_logits.new_tensor(pos_weight),
        reduction="none",
    )
    loss = (loss * mask.to(dtype=loss.dtype)).sum() / max(float(open_count), 1.0)
    return loss, {
        "ln_close/open_lane_count": float(open_count),
        "ln_close/positive_count": positive_count,
        "ln_close/pos_weight": float(pos_weight),
    }


def _write_checkpoint_and_report(
    *,
    output_dir: Path,
    checkpoint_path: Path,
    report_path: Path,
    model: MapperV1Model,
    optimizer: torch.optim.Optimizer,
    model_config: MapperV1Config,
    control_model_config: ControlDemoGlobalEncoderConfig | None,
    loss_config: MapperV1PhaseBLossConfig,
    seed: int,
    run_name: str,
    max_steps: int,
    completed_steps: int,
    eval_every: int,
    save_every: int,
    log_every: int | None,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
    dataset_report: Mapping[str, Any],
    history: list[dict[str, Any]],
    last_train_metrics: Mapping[str, float],
    final_train_metrics: Mapping[str, float],
    final_eval_metrics: Mapping[str, float],
    initialization_report: Mapping[str, Any] | None,
) -> None:
    training_config = {
        "phase": "B",
        "seed": seed,
        "run_name": run_name,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "eval_every": eval_every,
        "save_every": save_every,
        "density_enabled": False,
        "dataset": _json_safe(dataset_report),
    }
    checkpoint_payload = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": asdict(model_config),
        "control_model_config": None if control_model_config is None else asdict(control_model_config),
        "loss_config": asdict(loss_config),
        "training_config": training_config,
        "seed": seed,
        "run_name": run_name,
        "history": history,
        "initialization": None if initialization_report is None else dict(initialization_report),
        "training_state": {
            "step": completed_steps,
            "max_steps": max_steps,
            "is_complete": completed_steps >= max_steps,
            "eval_every": eval_every,
            "save_every": save_every,
            "log_every": log_every,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "device": str(device),
            "last_train_metrics": _json_metrics(last_train_metrics),
            "final_train_metrics": _json_metrics(final_train_metrics),
            "final_eval_metrics": _json_metrics(final_eval_metrics),
            "rng_state": _capture_rng_state(),
        },
    }
    archive_path = output_dir / "checkpoints" / f"checkpoint_step_{completed_steps:06d}.pt"
    _atomic_torch_save(checkpoint_payload, archive_path)
    _copy_file_atomically(archive_path, checkpoint_path)
    report_payload = {
        "run_name": run_name,
        "phase": "B",
        "seed": seed,
        "max_steps": max_steps,
        "completed_steps": completed_steps,
        "is_complete": completed_steps >= max_steps,
        "eval_every": eval_every,
        "save_every": save_every,
        "log_every": log_every,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "model_config": asdict(model_config),
        "control_model_config": None if control_model_config is None else asdict(control_model_config),
        "loss_config": asdict(loss_config),
        "training_config": training_config,
        "device": str(device),
        "parameter_count": model.parameter_count(),
        "dataset": dict(dataset_report),
        "initialization": None if initialization_report is None else dict(initialization_report),
        "history": history,
        "last_train_metrics": _json_metrics(last_train_metrics),
        "final_train_metrics": _json_metrics(final_train_metrics),
        "final_eval_metrics": _json_metrics(final_eval_metrics),
    }
    _write_report(report_path, report_payload)


def _synthetic_mapper_samples(*, model_config: MapperV1Config) -> list[dict[str, Any]]:
    from train.stage_2.model_mapper_v1.replay import close_labels_from_tokens, replay_state_tensors

    vocab = MapperV1Vocab()
    target_ids = [
        vocab.bos_id,
        vocab.time_shift_token_id(4000),
        vocab.time_shift_token_id(4000),
        vocab.eos_id,
    ]
    states = replay_state_tensors(target_ids, vocab=vocab, write_start_ms=0, write_end_ms=8000)
    close_labels, close_label_mask = close_labels_from_tokens(
        target_ids,
        vocab=vocab,
        write_start_ms=0,
        write_end_ms=8000,
    )
    samples: list[dict[str, Any]] = []
    generator = torch.Generator().manual_seed(20260509)
    for index in range(4):
        samples.append(
            {
                "target_tokens": torch.tensor(target_ids, dtype=torch.long),
                "teacher_current_ms": states["current_ms"],
                "teacher_open_mask": states["open_mask"],
                "teacher_open_age_ms": states["open_age_ms"],
                "close_labels": close_labels,
                "close_label_mask": close_label_mask,
                "control_memory_8s": torch.randn(
                    model_config.density_frames,
                    model_config.control_dim,
                    generator=generator,
                    dtype=torch.float32,
                )
                * 0.05,
                "density_teacher_8s": torch.zeros(model_config.density_frames, 1, dtype=torch.float32),
                "density_target_8s": torch.zeros(model_config.density_frames, 1, dtype=torch.float32),
                "density_confidence_8s": torch.zeros(model_config.density_frames, 1, dtype=torch.float32),
                "write_start_ms": torch.tensor(0, dtype=torch.long),
                "write_end_ms": torch.tensor(8000, dtype=torch.long),
                "difficulty": torch.tensor([2.0 + index], dtype=torch.float32),
                "normalized_difficulty": torch.tensor([-0.5 + 0.25 * index], dtype=torch.float32),
            }
        )
    return samples


def _collate_synthetic_mapper_samples(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("synthetic mapper collate requires at least one sample")
    keys = [
        "target_tokens",
        "teacher_current_ms",
        "teacher_open_mask",
        "teacher_open_age_ms",
        "close_labels",
        "close_label_mask",
        "control_memory_8s",
        "density_teacher_8s",
        "density_target_8s",
        "density_confidence_8s",
        "write_start_ms",
        "write_end_ms",
        "difficulty",
        "normalized_difficulty",
    ]
    return {key: torch.stack([sample[key] for sample in samples]) for key in keys}


def _normalized_section(source: object, *, allowed: set[str], name: str) -> dict[str, Any]:
    if source is None:
        return {}
    if not isinstance(source, dict):
        raise ValueError(f"{name} must be a mapping")
    normalized = _normalize_config_mapping(source, source_name=name)
    unknown = sorted(set(normalized) - allowed)
    if unknown:
        raise ValueError(f"unknown {name} keys: {unknown}")
    return normalized


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


def main(argv: Sequence[str] | None = None) -> None:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=None, help="YAML run config; CLI flags override config values")
    config_args, _ = config_parser.parse_known_args(argv)
    config_defaults = load_run_config(config_args.config) if config_args.config is not None else {
        "model": {},
        "control_model": {},
        "loss": {},
    }
    model_defaults = config_defaults["model"]
    control_model_defaults = config_defaults["control_model"]
    loss_defaults = config_defaults["loss"]

    parser = argparse.ArgumentParser(description="Train the Stage 2 mapper v1 Phase B teacher-forced skeleton.")
    parser.add_argument("--config", default=config_args.config)
    parser.add_argument("--dataset-root", default=config_defaults.get("dataset_root", "mania-dataset"))
    parser.add_argument("--index-path", default=config_defaults.get("index_path"))
    parser.add_argument("--eval-index-path", default=config_defaults.get("eval_index_path"))
    parser.add_argument("--control-v3-timeseries-path", default=config_defaults.get("control_v3_timeseries_path"))
    parser.add_argument("--output-dir", default=config_defaults.get("output_dir", DEFAULT_OUTPUT_DIR.as_posix()))
    parser.add_argument("--max-steps", type=int, default=config_defaults.get("max_steps", 5000))
    parser.add_argument("--eval-every", type=int, default=config_defaults.get("eval_every", 100))
    parser.add_argument("--save-every", type=int, default=config_defaults.get("save_every"))
    parser.add_argument("--log-every", type=int, default=config_defaults.get("log_every"))
    parser.add_argument("--batch-size", type=int, default=config_defaults.get("batch_size", 4))
    parser.add_argument("--learning-rate", type=float, default=config_defaults.get("learning_rate", 2e-4))
    parser.add_argument("--weight-decay", type=float, default=config_defaults.get("weight_decay", 0.01))
    parser.add_argument("--seed", type=int, default=config_defaults.get("seed", 1337))
    parser.add_argument("--device", default=config_defaults.get("device", "auto"), choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--run-name", default=config_defaults.get("run_name", "mapper_v1_phase_b_teacher_forced"))
    parser.add_argument("--init-from-control-checkpoint", default=config_defaults.get("init_from_control_checkpoint"))
    parser.add_argument("--eval-fraction", type=float, default=config_defaults.get("eval_fraction", 0.1))
    parser.add_argument("--eval-size", type=int, default=config_defaults.get("eval_size"))
    parser.add_argument(
        "--final-train-eval-size",
        type=int,
        default=config_defaults.get("final_train_eval_size", DEFAULT_FINAL_TRAIN_EVAL_SIZE),
    )
    parser.add_argument("--num-workers", type=int, default=config_defaults.get("num_workers", 0))
    parser.add_argument("--max-cached-maps", type=int, default=config_defaults.get("max_cached_maps"))
    parser.add_argument(
        "--dataset-progress",
        action=argparse.BooleanOptionalAction,
        default=config_defaults.get("dataset_progress"),
    )
    parser.add_argument("--control-teacher-cache-dir", default=config_defaults.get("control_teacher_cache_dir"))
    parser.add_argument(
        "--precompute-control-teacher-cache",
        action="store_true",
        default=bool(config_defaults.get("precompute_control_teacher_cache", False)),
    )
    parser.add_argument(
        "--precompute-control-teacher-cache-only",
        action="store_true",
        default=bool(config_defaults.get("precompute_control_teacher_cache_only", False)),
    )
    parser.add_argument(
        "--control-teacher-precompute-batch-size",
        type=int,
        default=config_defaults.get("control_teacher_precompute_batch_size"),
    )
    parser.add_argument(
        "--require-control-teacher-cache",
        action="store_true",
        default=bool(config_defaults.get("require_control_teacher_cache", False)),
    )
    parser.add_argument(
        "--control-teacher-cache-overwrite",
        action="store_true",
        default=bool(config_defaults.get("control_teacher_cache_overwrite", False)),
    )
    parser.add_argument("--synthetic-smoke", action="store_true", default=bool(config_defaults.get("synthetic_smoke", False)))
    args = parser.parse_args(argv)

    init_from = Path(args.init_from_control_checkpoint) if args.init_from_control_checkpoint is not None else None
    if args.precompute_control_teacher_cache_only:
        if args.synthetic_smoke:
            raise ValueError("precompute_control_teacher_cache_only is not supported with synthetic_smoke")
        result = precompute_mapper_v1_phase_b_control_teacher_cache(
            dataset_root=Path(args.dataset_root),
            index_path=Path(args.index_path) if args.index_path is not None else None,
            eval_index_path=Path(args.eval_index_path) if args.eval_index_path is not None else None,
            control_v3_timeseries_path=(
                Path(args.control_v3_timeseries_path)
                if args.control_v3_timeseries_path is not None
                else None
            ),
            batch_size=args.batch_size,
            seed=args.seed,
            device_name=args.device,
            init_from_control_checkpoint=init_from,
            num_workers=args.num_workers,
            max_cached_maps=args.max_cached_maps,
            dataset_progress=args.dataset_progress,
            control_teacher_cache_dir=(
                Path(args.control_teacher_cache_dir)
                if args.control_teacher_cache_dir is not None
                else None
            ),
            control_teacher_precompute_batch_size=args.control_teacher_precompute_batch_size,
            control_teacher_cache_overwrite=args.control_teacher_cache_overwrite,
            control_model_config_overrides=control_model_defaults,
        )
        for report in result.reports:
            print(
                "control_teacher_cache_report "
                f"split={report['split']} total={report['total_entries']} "
                f"computed={report['computed_entries']} skipped={report['skipped_entries']} "
                f"elapsed_s={float(report['elapsed_s']):.1f}",
            )
        return

    if args.synthetic_smoke:
        result = run_synthetic_smoke(
            output_dir=Path(args.output_dir),
            max_steps=args.max_steps,
            eval_every=args.eval_every,
            save_every=args.save_every,
            log_every=args.log_every,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
            device_name=args.device,
            final_train_eval_size=args.final_train_eval_size,
            model_config_overrides=model_defaults,
            loss_config_overrides=loss_defaults,
        )
    else:
        result = run_mapper_v1_phase_b_training(
            dataset_root=Path(args.dataset_root),
            index_path=Path(args.index_path) if args.index_path is not None else None,
            eval_index_path=Path(args.eval_index_path) if args.eval_index_path is not None else None,
            control_v3_timeseries_path=(
                Path(args.control_v3_timeseries_path)
                if args.control_v3_timeseries_path is not None
                else None
            ),
            output_dir=Path(args.output_dir),
            max_steps=args.max_steps,
            eval_every=args.eval_every,
            save_every=args.save_every,
            log_every=args.log_every,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            seed=args.seed,
            device_name=args.device,
            run_name=args.run_name,
            init_from_control_checkpoint=init_from,
            eval_fraction=args.eval_fraction,
            eval_size=args.eval_size,
            final_train_eval_size=args.final_train_eval_size,
            num_workers=args.num_workers,
            max_cached_maps=args.max_cached_maps,
            dataset_progress=args.dataset_progress,
            control_teacher_cache_dir=(
                Path(args.control_teacher_cache_dir)
                if args.control_teacher_cache_dir is not None
                else None
            ),
            precompute_control_teacher_cache=args.precompute_control_teacher_cache,
            control_teacher_precompute_batch_size=args.control_teacher_precompute_batch_size,
            require_control_teacher_cache=args.require_control_teacher_cache,
            control_teacher_cache_overwrite=args.control_teacher_cache_overwrite,
            model_config_overrides=model_defaults,
            control_model_config_overrides=control_model_defaults,
            loss_config_overrides=loss_defaults,
        )
    print(f"report_path {result.report_path}")
    print(f"checkpoint_path {result.checkpoint_path}")
    print(f"final_loss {result.final_loss:.6f}")
    print(f"completed_steps {result.completed_steps}")


if __name__ == "__main__":
    main()
