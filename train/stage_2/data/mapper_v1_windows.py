from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import Dataset

from train.stage1_oracle.osu.hitobjects import parse_mania_hit_objects
from train.stage_2.model_control.context import prepare_control_context_batch
from train.stage_2.data.control_windows import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_INDEX_PATH,
    FRAME_HOP_MS,
    TARGET_WINDOW_LENGTH_FRAMES,
    ControlWindowDataset,
    ControlWindowRecord,
    normalize_difficulty,
)
from train.stage_2.features.control_v3_targets import MODEL_FEATURE_NAMES, VALUE_FEATURE_NAMES
from train.stage_2.model_mapper_v1.tokenizer import (
    MAPPER_DENSITY_FRAMES,
    MAPPER_WRITE_MS,
    CrossWindowLongNoteError,
    TokenizedMapperWindow,
    UnsupportedMapperActionError,
    cross_window_ln_state_reason,
    encode_mapper_window,
    hitobjects_to_mapper_timepoints,
    window_timepoints,
)
from train.stage_2.model_mapper_v1.vocab import MapperV1Vocab


MAPPER_WRITE_FRAMES = MAPPER_WRITE_MS // FRAME_HOP_MS
MAPPER_CONTEXT_FRAMES = MAPPER_WRITE_FRAMES
DENSITY_LEVEL_TARGET_INDEX = MODEL_FEATURE_NAMES.index("density_level")
DENSITY_CONFIDENCE_TARGET_INDEX = MODEL_FEATURE_NAMES.index("density_confidence")
CONTROL_TEACHER_CACHE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class MapperV1WindowRecord:
    control_record_index: int
    control_record: ControlWindowRecord

    @property
    def write_start_frame(self) -> int:
        return self.control_record.target_start_frame

    @property
    def write_start_ms(self) -> int:
        return self.control_record.target_start_ms

    @property
    def write_end_ms(self) -> int:
        return self.write_start_ms + MAPPER_WRITE_MS


@dataclass(frozen=True)
class MapperV1WindowFilterReport:
    num_total_windows: int
    num_mapper_eligible_windows: int
    num_dropped_short_windows: int
    num_dropped_cross_window_ln_windows: int
    num_dropped_unsupported_action_windows: int
    drop_rate: float
    short_drop_rate: float
    cross_window_ln_drop_rate: float
    unsupported_action_drop_rate: float
    drop_rate_by_difficulty: dict[str, float]
    drop_rate_by_song: dict[str, float]


class MapperV1WindowDataset(Dataset):
    """Mapper v1 8s window dataset derived from the existing Stage 2 control dataset."""

    def __init__(
        self,
        index_path: str | Path = DEFAULT_INDEX_PATH,
        *,
        dataset_root: str | Path = DEFAULT_DATASET_ROOT,
        control_dataset: ControlWindowDataset | None = None,
        vocab: MapperV1Vocab | None = None,
        mapper_stride_frames: int = MAPPER_WRITE_FRAMES,
        control_teacher_cache_dir: str | Path | None = None,
        require_control_teacher_cache: bool = False,
        progress: bool = False,
        **control_dataset_kwargs: Any,
    ) -> None:
        if mapper_stride_frames <= 0:
            raise ValueError(f"mapper_stride_frames must be positive: {mapper_stride_frames}")
        self.control_dataset = control_dataset or ControlWindowDataset(
            index_path=index_path,
            dataset_root=dataset_root,
            progress=progress,
            **control_dataset_kwargs,
        )
        self.vocab = MapperV1Vocab() if vocab is None else vocab
        self.mapper_stride_frames = int(mapper_stride_frames)
        self.control_teacher_cache_dir = None if control_teacher_cache_dir is None else Path(control_teacher_cache_dir)
        self.require_control_teacher_cache = bool(require_control_teacher_cache)
        self._timepoints_by_beatmap: dict[str, tuple] = {}
        self.records, self.filter_report = self._build_records(
            progress=progress,
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        mapper_record = self.records[index]
        record = mapper_record.control_record
        tokenized = self._tokenize_record(record)
        cache_path = self.control_teacher_cache_path(record)
        cache_entry = None
        if cache_path is not None and cache_path.exists():
            cache_entry = load_control_teacher_cache_entry(cache_path, record=record)
        elif self.require_control_teacher_cache and cache_path is not None:
            raise FileNotFoundError(f"missing mapper v1 control teacher cache entry: {cache_path}")
        elif self.require_control_teacher_cache:
            raise ValueError("require_control_teacher_cache=True requires control_teacher_cache_dir")

        metadata = {
            "beatmap_path": record.beatmap_path.as_posix(),
            "audio_path": record.audio_path.as_posix(),
            "difficulty": record.difficulty,
            "target_start_frame": record.target_start_frame,
            "target_start_ms": record.target_start_ms,
            "control_record_index": mapper_record.control_record_index,
        }
        if cache_path is not None:
            metadata["control_teacher_cache_key"] = control_teacher_cache_key(record)
            metadata["control_teacher_cache_path"] = cache_path.as_posix()
            metadata["control_teacher_cache_hit"] = cache_entry is not None

        sample: dict[str, Any] = {
            "difficulty": torch.tensor([record.difficulty], dtype=torch.float32),
            "normalized_difficulty": torch.tensor([normalize_difficulty(record.difficulty)], dtype=torch.float32),
            "target_tokens": tokenized.target_tensor(),
            "teacher_current_ms": tokenized.teacher_current_ms,
            "teacher_open_mask": tokenized.teacher_open_mask,
            "teacher_open_age_ms": tokenized.teacher_open_age_ms,
            "close_labels": tokenized.close_labels,
            "close_label_mask": tokenized.close_label_mask,
            "write_start_ms": torch.tensor(tokenized.write_start_ms, dtype=torch.long),
            "write_end_ms": torch.tensor(tokenized.write_end_ms, dtype=torch.long),
            "metadata": metadata,
        }
        if cache_entry is not None:
            sample["control_memory_8s"] = cache_entry["control_memory_8s"]
            sample["density_teacher_8s"] = cache_entry["density_teacher_8s"]
            return sample

        base_sample = self.control_dataset[mapper_record.control_record_index]
        density_target_8s, density_confidence_8s = extract_mapper_density_8s(
            self._load_control_v3_target_8s(record),
        )
        frame_count = int(base_sample["frame_count"].item())
        write_start_frame = int(base_sample["target_start_frame"].item())
        write_end_frame = write_start_frame + MAPPER_WRITE_FRAMES
        if write_end_frame > frame_count:
            raise ValueError(f"mapper write span exceeds frame_count: {write_end_frame} > {frame_count}")
        full_mel = base_sample["full_mel"]
        full_dense_timing_v2 = base_sample["full_dense_timing_v2"]
        sample.update(
            {
                "full_mel": base_sample["full_mel"],
                "full_dense_timing_v2": base_sample["full_dense_timing_v2"],
                "frame_count": base_sample["frame_count"],
                "target_start_frame": base_sample["target_start_frame"],
                "control_slice_start_frames": torch.tensor(
                    [
                        record.target_start_frame + offset
                        for offset in range(0, MAPPER_WRITE_FRAMES, TARGET_WINDOW_LENGTH_FRAMES)
                    ],
                    dtype=torch.long,
                ),
                "mel_context": full_mel[write_start_frame:write_end_frame].contiguous(),
                "timing_context": full_dense_timing_v2[write_start_frame:write_end_frame].contiguous(),
                "context_padding_mask": torch.zeros(MAPPER_CONTEXT_FRAMES, dtype=torch.bool),
                "difficulty": base_sample["difficulty"].reshape(1),
                "normalized_difficulty": base_sample["normalized_difficulty"].reshape(1),
                "density_target_8s": density_target_8s,
                "density_confidence_8s": density_confidence_8s,
            }
        )
        return sample

    def control_teacher_cache_path(self, record: ControlWindowRecord) -> Path | None:
        if self.control_teacher_cache_dir is None:
            return None
        return control_teacher_cache_path(self.control_teacher_cache_dir, record)

    def _build_records(
        self,
        *,
        progress: bool,
    ) -> tuple[list[MapperV1WindowRecord], MapperV1WindowFilterReport]:
        records: list[MapperV1WindowRecord] = []
        total_windows = 0
        dropped_short = 0
        dropped_cross_window = 0
        dropped_unsupported_action = 0
        valid_length_windows = 0
        dropped_by_difficulty: dict[str, int] = {}
        valid_by_difficulty: dict[str, int] = {}
        valid_by_song: dict[str, int] = {}
        dropped_by_song: dict[str, int] = {}
        for index, record in enumerate(self.control_dataset.records):
            if record.target_start_frame % self.mapper_stride_frames != 0:
                continue
            total_windows += 1
            difficulty_key = _difficulty_report_key(record.difficulty)
            song_key = record.beatmap_path.as_posix()
            if record.target_start_frame + MAPPER_WRITE_FRAMES > record.frame_count:
                dropped_short += 1
                continue
            valid_length_windows += 1
            valid_by_difficulty[difficulty_key] = valid_by_difficulty.get(difficulty_key, 0) + 1
            valid_by_song[song_key] = valid_by_song.get(song_key, 0) + 1
            try:
                self._tokenize_record(record)
            except CrossWindowLongNoteError:
                dropped_cross_window += 1
                _increment_drop(dropped_by_difficulty, difficulty_key)
                _increment_drop(dropped_by_song, song_key)
                continue
            except UnsupportedMapperActionError:
                dropped_unsupported_action += 1
                _increment_drop(dropped_by_difficulty, difficulty_key)
                _increment_drop(dropped_by_song, song_key)
                continue
            records.append(MapperV1WindowRecord(control_record_index=index, control_record=record))
            if progress and len(records) % 1000 == 0:
                print(f"mapper_v1_window_dataset_progress eligible_windows={len(records)}", flush=True)

        dropped = dropped_short + dropped_cross_window + dropped_unsupported_action
        report = MapperV1WindowFilterReport(
            num_total_windows=total_windows,
            num_mapper_eligible_windows=len(records),
            num_dropped_short_windows=dropped_short,
            num_dropped_cross_window_ln_windows=dropped_cross_window,
            num_dropped_unsupported_action_windows=dropped_unsupported_action,
            drop_rate=float(dropped / total_windows) if total_windows else 0.0,
            short_drop_rate=float(dropped_short / total_windows) if total_windows else 0.0,
            cross_window_ln_drop_rate=float(dropped_cross_window / valid_length_windows) if valid_length_windows else 0.0,
            unsupported_action_drop_rate=float(dropped_unsupported_action / valid_length_windows) if valid_length_windows else 0.0,
            drop_rate_by_difficulty=_drop_rates(valid_by_difficulty, dropped_by_difficulty),
            drop_rate_by_song=_drop_rates(valid_by_song, dropped_by_song),
        )
        return records, report

    def _tokenize_record(self, record: ControlWindowRecord) -> TokenizedMapperWindow:
        write_start_ms = record.target_start_ms
        write_end_ms = write_start_ms + MAPPER_WRITE_MS
        timepoints = self._load_timepoints(record.beatmap_path)
        reason = cross_window_ln_state_reason(
            timepoints,
            write_start_ms=write_start_ms,
            write_end_ms=write_end_ms,
        )
        if reason is not None:
            raise CrossWindowLongNoteError(f"window requires {reason} LN state")
        return encode_mapper_window(
            window_timepoints(
                timepoints,
                write_start_ms=write_start_ms,
                write_end_ms=write_end_ms,
            ),
            vocab=self.vocab,
            write_start_ms=write_start_ms,
            write_end_ms=write_end_ms,
        )

    def _load_timepoints(self, beatmap_path: Path) -> tuple:
        key = beatmap_path.as_posix()
        cached = self._timepoints_by_beatmap.get(key)
        if cached is None:
            cached = tuple(hitobjects_to_mapper_timepoints(parse_mania_hit_objects(beatmap_path, expected_key_count=4)))
            self._timepoints_by_beatmap[key] = cached
        return cached

    def _load_control_v3_target_8s(self, record: ControlWindowRecord) -> torch.Tensor:
        slices = []
        for offset_frames in range(0, MAPPER_WRITE_FRAMES, TARGET_WINDOW_LENGTH_FRAMES):
            slice_record = replace(record, target_start_frame=record.target_start_frame + offset_frames)
            target = self.control_dataset.target_loader(slice_record)
            slice_tensor = torch.as_tensor(target, dtype=torch.float32)
            expected_shape = (TARGET_WINDOW_LENGTH_FRAMES, len(MODEL_FEATURE_NAMES))
            if tuple(slice_tensor.shape) != expected_shape:
                raise ValueError(f"control_v3 target slice must have shape {expected_shape}, got {tuple(slice_tensor.shape)}")
            if not torch.isfinite(slice_tensor).all():
                raise ValueError("control_v3 target slice must contain only finite values")
            slices.append(slice_tensor)
        return torch.cat(slices, dim=0).contiguous()


def extract_mapper_density_8s(control_v3_target_8s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    target = torch.as_tensor(control_v3_target_8s, dtype=torch.float32)
    expected_shape = (MAPPER_DENSITY_FRAMES, len(MODEL_FEATURE_NAMES))
    if tuple(target.shape) != expected_shape:
        raise ValueError(f"control_v3_target_8s must have shape {expected_shape}, got {tuple(target.shape)}")
    if not torch.isfinite(target).all():
        raise ValueError("control_v3_target_8s must contain only finite values")
    density_target = target[:, DENSITY_LEVEL_TARGET_INDEX : DENSITY_LEVEL_TARGET_INDEX + 1].contiguous()
    density_confidence = target[:, DENSITY_CONFIDENCE_TARGET_INDEX : DENSITY_CONFIDENCE_TARGET_INDEX + 1].contiguous()
    if not torch.isfinite(density_target).all() or not torch.isfinite(density_confidence).all():
        raise ValueError("density_target_8s and density_confidence_8s must contain only finite values")
    if torch.any((density_confidence < 0.0) | (density_confidence > 1.0)):
        raise ValueError("density_confidence_8s must be in [0, 1]")
    return density_target, density_confidence


def control_teacher_cache_key(record: ControlWindowRecord) -> str:
    identity = "\n".join(
        (
            record.beatmap_path.as_posix(),
            record.audio_path.as_posix(),
            f"difficulty={float(record.difficulty):.8f}",
            f"frame_count={int(record.frame_count)}",
            f"target_start_frame={int(record.target_start_frame)}",
        )
    )
    return hashlib.sha1(identity.encode("utf-8")).hexdigest()


def control_teacher_cache_path(cache_dir: str | Path, record: ControlWindowRecord) -> Path:
    key = control_teacher_cache_key(record)
    return Path(cache_dir) / key[:2] / f"{key}.pt"


def load_control_teacher_cache_entry(path: str | Path, *, record: ControlWindowRecord | None = None) -> dict[str, torch.Tensor]:
    cache_path = Path(path)
    payload = torch.load(cache_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError(f"control teacher cache entry must contain a mapping: {cache_path}")
    schema_version = payload.get("schema_version")
    if int(schema_version) != CONTROL_TEACHER_CACHE_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported control teacher cache schema {schema_version}; "
            f"expected {CONTROL_TEACHER_CACHE_SCHEMA_VERSION}"
        )
    if record is not None:
        expected_key = control_teacher_cache_key(record)
        if payload.get("cache_key") != expected_key:
            raise ValueError(f"control teacher cache key mismatch for {cache_path}")
        if int(payload.get("write_start_ms", -1)) != int(record.target_start_ms):
            raise ValueError(f"control teacher cache write_start_ms mismatch for {cache_path}")
        if int(payload.get("write_end_ms", -1)) != int(record.target_start_ms + MAPPER_WRITE_MS):
            raise ValueError(f"control teacher cache write_end_ms mismatch for {cache_path}")

    control_memory = payload.get("control_memory_8s")
    density_teacher = payload.get("density_teacher_8s")
    if not isinstance(control_memory, torch.Tensor):
        raise ValueError(f"control teacher cache missing control_memory_8s tensor: {cache_path}")
    if not isinstance(density_teacher, torch.Tensor):
        raise ValueError(f"control teacher cache missing density_teacher_8s tensor: {cache_path}")
    control_memory = control_memory.to(dtype=torch.float32).contiguous()
    density_teacher = density_teacher.to(dtype=torch.float32).contiguous()
    if control_memory.ndim != 2 or int(control_memory.shape[0]) != MAPPER_DENSITY_FRAMES:
        raise ValueError(
            f"control_memory_8s cache tensor must have shape [{MAPPER_DENSITY_FRAMES},D], "
            f"got {tuple(control_memory.shape)}"
        )
    if int(control_memory.shape[1]) <= 0:
        raise ValueError("control_memory_8s cache tensor must have a positive hidden dimension")
    if tuple(density_teacher.shape) != (MAPPER_DENSITY_FRAMES, 1):
        raise ValueError(
            f"density_teacher_8s cache tensor must have shape [{MAPPER_DENSITY_FRAMES},1], "
            f"got {tuple(density_teacher.shape)}"
        )
    if not torch.isfinite(control_memory).all() or not torch.isfinite(density_teacher).all():
        raise ValueError(f"control teacher cache contains non-finite values: {cache_path}")
    return {
        "control_memory_8s": control_memory,
        "density_teacher_8s": density_teacher,
    }


def save_control_teacher_cache_entry(
    path: str | Path,
    *,
    record: ControlWindowRecord,
    control_memory_8s: torch.Tensor,
    density_teacher_8s: torch.Tensor,
) -> None:
    cache_path = Path(path)
    control_memory = control_memory_8s.detach().to(device="cpu", dtype=torch.float32).contiguous()
    density_teacher = density_teacher_8s.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if control_memory.ndim != 2 or int(control_memory.shape[0]) != MAPPER_DENSITY_FRAMES:
        raise ValueError(
            f"control_memory_8s must have shape [{MAPPER_DENSITY_FRAMES},D], got {tuple(control_memory.shape)}"
        )
    if tuple(density_teacher.shape) != (MAPPER_DENSITY_FRAMES, 1):
        raise ValueError(
            f"density_teacher_8s must have shape [{MAPPER_DENSITY_FRAMES},1], got {tuple(density_teacher.shape)}"
        )
    if not torch.isfinite(control_memory).all() or not torch.isfinite(density_teacher).all():
        raise ValueError("control teacher cache tensors must contain only finite values")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": CONTROL_TEACHER_CACHE_SCHEMA_VERSION,
        "cache_key": control_teacher_cache_key(record),
        "write_start_ms": int(record.target_start_ms),
        "write_end_ms": int(record.target_start_ms + MAPPER_WRITE_MS),
        "control_dim": int(control_memory.shape[1]),
        "control_memory_8s": control_memory,
        "density_teacher_8s": density_teacher,
    }
    tmp_path = cache_path.with_name(f"{cache_path.name}.tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(cache_path)


def control_teacher_slice_batch(mapper_batch: dict[str, Any], slice_index: int) -> dict[str, Any]:
    if not 0 <= int(slice_index) < 4:
        raise ValueError(f"slice_index must be in 0..3, got {slice_index}")
    control_slice_start_frames = mapper_batch.get("control_slice_start_frames")
    if not isinstance(control_slice_start_frames, torch.Tensor) or control_slice_start_frames.ndim != 2:
        raise ValueError("mapper_batch must contain control_slice_start_frames with shape [B,4]")
    if int(control_slice_start_frames.shape[1]) != 4:
        raise ValueError("control_slice_start_frames must have four aligned 2s starts")
    normalized_difficulty = mapper_batch.get("normalized_difficulty")
    if not isinstance(normalized_difficulty, torch.Tensor):
        raise ValueError("mapper_batch must contain normalized_difficulty")
    normalized_difficulty = normalized_difficulty.reshape(normalized_difficulty.shape[0])
    control_batch = {
        "full_mel": mapper_batch["full_mel"],
        "full_dense_timing_v2": mapper_batch["full_dense_timing_v2"],
        "padding_mask": mapper_batch["padding_mask"],
        "frame_count": mapper_batch["frame_count"],
        "target_start_frame": control_slice_start_frames[:, int(slice_index)],
        "normalized_difficulty": normalized_difficulty,
    }
    return prepare_control_context_batch(control_batch)


def concatenate_density_teacher_8s(control_outputs: Sequence[Any]) -> torch.Tensor:
    if len(control_outputs) != 4:
        raise ValueError(f"expected four 2s control outputs, got {len(control_outputs)}")
    values = []
    density_index = VALUE_FEATURE_NAMES.index("density_level")
    for index, output in enumerate(control_outputs):
        value_pred = getattr(output, "value_pred", None)
        if not isinstance(value_pred, torch.Tensor) or value_pred.ndim != 3 or int(value_pred.shape[1]) != 100:
            raise ValueError(f"control output {index} value_pred must have shape [B,100,C]")
        if int(value_pred.shape[2]) == 1:
            values.append(value_pred)
        elif int(value_pred.shape[2]) == len(VALUE_FEATURE_NAMES):
            values.append(value_pred[:, :, density_index : density_index + 1])
        else:
            raise ValueError(
                f"control output {index} value_pred channel count must be 1 or {len(VALUE_FEATURE_NAMES)}, "
                f"got {value_pred.shape[2]}",
            )
    return torch.cat(values, dim=1).contiguous()


def collate_mapper_v1_windows(samples: Sequence[dict[str, Any]], *, pad_id: int = 0) -> dict[str, Any]:
    if not samples:
        raise ValueError("collate_mapper_v1_windows requires at least one sample")
    batch_size = len(samples)
    max_seq_len = max(int(sample["target_tokens"].shape[0]) for sample in samples)
    target_tokens = torch.full((batch_size, max_seq_len), int(pad_id), dtype=torch.long)
    target_token_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.bool)
    teacher_current_ms = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    teacher_open_mask = torch.zeros((batch_size, max_seq_len, 4), dtype=torch.bool)
    teacher_open_age_ms = torch.zeros((batch_size, max_seq_len, 4), dtype=torch.long)
    close_labels = torch.zeros((batch_size, max_seq_len, 4), dtype=torch.bool)
    close_label_mask = torch.zeros((batch_size, max_seq_len, 4), dtype=torch.bool)

    for batch_index, sample in enumerate(samples):
        length = int(sample["target_tokens"].shape[0])
        target_tokens[batch_index, :length] = sample["target_tokens"].to(dtype=torch.long)
        target_token_mask[batch_index, :length] = True
        teacher_current_ms[batch_index, :length] = sample["teacher_current_ms"].to(dtype=torch.long)
        teacher_open_mask[batch_index, :length] = sample["teacher_open_mask"].to(dtype=torch.bool)
        teacher_open_age_ms[batch_index, :length] = sample["teacher_open_age_ms"].to(dtype=torch.long)
        close_labels[batch_index, :length] = sample["close_labels"].to(dtype=torch.bool)
        close_label_mask[batch_index, :length] = sample["close_label_mask"].to(dtype=torch.bool)

    batch = {
        "difficulty": torch.stack([sample["difficulty"].to(dtype=torch.float32) for sample in samples]),
        "normalized_difficulty": torch.stack(
            [sample.get("normalized_difficulty", sample["difficulty"]).to(dtype=torch.float32) for sample in samples],
        ),
        "target_tokens": target_tokens,
        "target_token_mask": target_token_mask,
        "teacher_current_ms": teacher_current_ms,
        "teacher_open_mask": teacher_open_mask,
        "teacher_open_age_ms": teacher_open_age_ms,
        "close_labels": close_labels,
        "close_label_mask": close_label_mask,
        "write_start_ms": torch.stack([sample["write_start_ms"].to(dtype=torch.long) for sample in samples]).reshape(
            batch_size,
        ),
        "write_end_ms": torch.stack([sample["write_end_ms"].to(dtype=torch.long) for sample in samples]).reshape(
            batch_size,
        ),
        "metadata": [sample.get("metadata", {}) for sample in samples],
    }

    has_control_teacher_cache = [
        "control_memory_8s" in sample or "density_teacher_8s" in sample
        for sample in samples
    ]
    if any(has_control_teacher_cache):
        if not all("control_memory_8s" in sample and "density_teacher_8s" in sample for sample in samples):
            raise ValueError(
                "partial mapper v1 control teacher cache batch is not supported; "
                "precompute all entries or disable the cache"
            )
        batch["control_memory_8s"] = torch.stack(
            [sample["control_memory_8s"].to(dtype=torch.float32) for sample in samples]
        )
        batch["density_teacher_8s"] = torch.stack(
            [sample["density_teacher_8s"].to(dtype=torch.float32) for sample in samples]
        )

    if all("density_target_8s" in sample and "density_confidence_8s" in sample for sample in samples):
        batch["density_target_8s"] = torch.stack(
            [sample["density_target_8s"].to(dtype=torch.float32) for sample in samples]
        )
        batch["density_confidence_8s"] = torch.stack(
            [sample["density_confidence_8s"].to(dtype=torch.float32) for sample in samples],
        )

    has_control_inputs = all(
        "full_mel" in sample and "full_dense_timing_v2" in sample and "frame_count" in sample
        for sample in samples
    )
    if has_control_inputs and not all(has_control_teacher_cache):
        batch["mel_context"] = torch.stack([sample["mel_context"].to(dtype=torch.float32) for sample in samples])
        batch["timing_context"] = torch.stack([sample["timing_context"].to(dtype=torch.float32) for sample in samples])
        batch["context_padding_mask"] = torch.stack(
            [sample["context_padding_mask"].to(dtype=torch.bool) for sample in samples]
        )
        frame_counts = [int(sample["frame_count"].item()) for sample in samples]
        max_frame_count = max(frame_counts)
        full_mel = torch.zeros((batch_size, max_frame_count, 160), dtype=torch.float32)
        full_dense_timing_v2 = torch.zeros((batch_size, max_frame_count, 4), dtype=torch.float32)
        padding_mask = torch.ones((batch_size, max_frame_count), dtype=torch.bool)
        for batch_index, sample in enumerate(samples):
            frame_count = frame_counts[batch_index]
            sample_full_mel = sample["full_mel"].to(dtype=torch.float32)
            sample_full_dense_timing_v2 = sample["full_dense_timing_v2"].to(dtype=torch.float32)
            if tuple(sample_full_mel.shape) != (frame_count, 160):
                raise ValueError(f"full_mel sample {batch_index} must have shape {(frame_count, 160)}")
            if tuple(sample_full_dense_timing_v2.shape) != (frame_count, 4):
                raise ValueError(f"full_dense_timing_v2 sample {batch_index} must have shape {(frame_count, 4)}")
            full_mel[batch_index, :frame_count] = sample_full_mel
            full_dense_timing_v2[batch_index, :frame_count] = sample_full_dense_timing_v2
            padding_mask[batch_index, :frame_count] = False
        batch["full_mel"] = full_mel
        batch["full_dense_timing_v2"] = full_dense_timing_v2
        batch["padding_mask"] = padding_mask
        batch["frame_count"] = torch.tensor(frame_counts, dtype=torch.long)
        if all("control_slice_start_frames" in sample for sample in samples):
            batch["control_slice_start_frames"] = torch.stack(
                [sample["control_slice_start_frames"].to(dtype=torch.long) for sample in samples],
            )
    return batch


def _difficulty_report_key(difficulty: float) -> str:
    return f"{float(difficulty):.2f}"


def _increment_drop(counter: dict[str, int], key: str) -> None:
    counter[key] = counter.get(key, 0) + 1


def _drop_rates(total: dict[str, int], dropped: dict[str, int]) -> dict[str, float]:
    return {
        key: float(dropped.get(key, 0) / count) if count else 0.0
        for key, count in sorted(total.items())
    }
