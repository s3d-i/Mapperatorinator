from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import torch
from torch.utils.data import Dataset

from ..events.canonical import LaneAction, NegativeHitObjectTimeError, UnsupportedCompoundLaneActionError
from ..events.canonical import build_canonical_quantized_events, quantize_10ms_half_up
from ..events.tokens import Stage1Vocab, TokenizedWindow, encode_window_tokens
from ..events.windowing import WindowSpec, compute_generation_end_ms, iter_window_specs
from ..events.windowing import open_hold_mask_at_write_start, window_timepoints
from ..features.audio import load_audio_file
from ..features.mel import DEFAULT_MEL_CACHE_CONFIG, MelCacheConfig
from ..features.mel import load_or_create_log_mel_cache, pack_mel_20ms_window
from ..features.timing import render_timing_track_20ms_v1
from ..osu.hitobjects import ManiaHitObjectKind, parse_mania_hit_objects
from ..osu.timing import InvalidRedTimingError, MissingRedTimingError
from ..osu.timing import RedTimingPoint, require_red_timing_points
from .dataset import get_default_4k_training_index_path, load_index


@dataclass(frozen=True)
class OracleWindowRecord:
    beatmap_path: Path
    audio_path: Path
    difficulty: float
    timing_points: tuple[RedTimingPoint, ...]
    timepoints: tuple
    window: WindowSpec
    open_hold_mask: int


@dataclass(frozen=True)
class OracleWindowFilterReport:
    source_map_count: int
    difficulty_filtered_map_count: int
    candidate_map_count: int
    missing_red_timing_map_count: int
    invalid_red_timing_map_count: int
    negative_time_hitobject_map_count: int
    unsupported_compound_map_count: int
    unsupported_compound_event_count: int
    four_state_unsupported_map_count: int
    four_state_unsupported_lane_action_count: int
    zero_length_hold_normalized_count: int
    retained_map_count: int
    generated_window_count: int
    invalid_hold_transition_map_count: int = 0
    invalid_hold_transition_count: int = 0


class OracleWindowDataset(Dataset):
    def __init__(
        self,
        dataset_root: str | Path = "mania-dataset",
        *,
        index_path: str | Path | None = None,
        manifest_path: str | Path | None = None,
        vocab: Stage1Vocab | None = None,
        sample_rate: int = 16000,
        mel_config: MelCacheConfig = DEFAULT_MEL_CACHE_CONFIG,
        bpm_log_mean: float,
        bpm_log_std: float,
        max_maps_per_bin: int | None = None,
        progress: bool = False,
    ) -> None:
        if not math.isfinite(bpm_log_mean):
            raise ValueError(f"bpm_log_mean must be finite: {bpm_log_mean}")
        if not math.isfinite(bpm_log_std) or bpm_log_std <= 0:
            raise ValueError(f"bpm_log_std must be positive: {bpm_log_std}")
        if max_maps_per_bin is not None and max_maps_per_bin <= 0:
            raise ValueError(f"max_maps_per_bin must be positive when set: {max_maps_per_bin}")

        self.dataset_root = Path(dataset_root)
        self.index_path = Path(index_path) if index_path is not None else get_default_4k_training_index_path()
        self.vocab = Stage1Vocab() if vocab is None else vocab
        self.sample_rate = sample_rate
        self.mel_config = mel_config
        self.bpm_log_mean = bpm_log_mean
        self.bpm_log_std = bpm_log_std

        index_df = load_index(self.index_path)
        if manifest_path is not None:
            index_df = _filter_index_by_manifest(index_df, Path(manifest_path))
        source_map_count = len(index_df)
        index_df = filter_supported_difficulty_range(index_df)
        difficulty_filtered_map_count = source_map_count - len(index_df)

        self.records, self.filter_report = self._build_records(
            index_df,
            source_map_count=source_map_count,
            difficulty_filtered_map_count=difficulty_filtered_map_count,
            max_maps_per_bin=max_maps_per_bin,
            progress=progress,
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        audio = load_audio_file(record.audio_path, sample_rate=self.sample_rate)
        mel = load_or_create_log_mel_cache(
            audio,
            sample_rate=self.sample_rate,
            audio_cache_key=record.audio_path.as_posix(),
            config=self.mel_config,
        )
        packed_audio = pack_mel_20ms_window(mel, input_start_ms=record.window.input_start_ms)
        timing_track = render_timing_track_20ms_v1(
            record.timing_points,
            input_start_ms=record.window.input_start_ms,
            bpm_log_mean=self.bpm_log_mean,
            bpm_log_std=self.bpm_log_std,
        )
        tokenized = encode_window_tokens(
            window_timepoints(
                record.timepoints,
                write_start_ms=record.window.write_start_ms,
                write_end_ms=record.window.write_end_ms,
            ),
            vocab=self.vocab,
            write_start_ms=record.window.write_start_ms,
            write_end_ms=record.window.write_end_ms,
            difficulty=record.difficulty,
            open_hold_mask=record.open_hold_mask,
        )
        decoder_input_ids, labels = build_decoder_tensors(tokenized, self.vocab)

        return {
            "packed_audio": torch.from_numpy(packed_audio),
            "timing_track": torch.from_numpy(timing_track),
            "difficulty_bucket": torch.tensor(self.vocab.difficulty_bucket_id(record.difficulty), dtype=torch.long),
            "open_hold_mask": torch.tensor(record.open_hold_mask, dtype=torch.long),
            "write_duration_ms": torch.tensor(record.window.write_duration_ms, dtype=torch.long),
            "decoder_input_ids": decoder_input_ids,
            "labels": labels,
            "beatmap_path": record.beatmap_path.as_posix(),
            "audio_path": record.audio_path.as_posix(),
            "difficulty": record.difficulty,
            "write_start_ms": record.window.write_start_ms,
        }

    def _build_records(
        self,
        index_df: pd.DataFrame,
        *,
        source_map_count: int,
        difficulty_filtered_map_count: int,
        max_maps_per_bin: int | None,
        progress: bool,
    ) -> tuple[list[OracleWindowRecord], OracleWindowFilterReport]:
        records: list[OracleWindowRecord] = []
        retained_map_count_by_bin = {label: 0 for label in ("2-3", "3-4", "4-5", "5-6")}
        missing_red_timing_map_count = 0
        invalid_red_timing_map_count = 0
        negative_time_hitobject_map_count = 0
        unsupported_compound_map_count = 0
        unsupported_compound_event_count = 0
        four_state_unsupported_map_count = 0
        four_state_unsupported_lane_action_count = 0
        invalid_hold_transition_map_count = 0
        invalid_hold_transition_count = 0
        zero_length_hold_normalized_count = 0
        retained_map_count = 0
        for scanned_map_count, row in enumerate(index_df.itertuples(index=False), start=1):
            if progress and (scanned_map_count == 1 or scanned_map_count % 100 == 0):
                print(
                    f"dataset_progress scanned_maps={scanned_map_count} "
                    f"retained_maps={retained_map_count} windows={len(records)}",
                    flush=True,
                )
            difficulty = float(row.difficulty)
            beatmap_path = self.dataset_root / str(row.shard) / str(row.beatmap_path)
            audio_path = self.dataset_root / str(row.shard) / str(row.audio_path)
            try:
                timing_points = tuple(require_red_timing_points(beatmap_path))
            except InvalidRedTimingError:
                invalid_red_timing_map_count += 1
                continue
            except MissingRedTimingError:
                missing_red_timing_map_count += 1
                continue

            try:
                hitobjects = parse_mania_hit_objects(beatmap_path, expected_key_count=4)
                map_unsupported_compound_event_count = _count_unsupported_compound_events(hitobjects, key_count=4)
                if map_unsupported_compound_event_count:
                    unsupported_compound_map_count += 1
                    unsupported_compound_event_count += map_unsupported_compound_event_count
                    continue
                build_result = build_canonical_quantized_events(hitobjects, key_count=4)
            except NegativeHitObjectTimeError:
                negative_time_hitobject_map_count += 1
                continue
            except UnsupportedCompoundLaneActionError:
                unsupported_compound_map_count += 1
                unsupported_compound_event_count += 1
                continue
            except ValueError:
                unsupported_compound_map_count += 1
                continue

            zero_length_hold_normalized_count += build_result.zero_length_hold_normalized_count
            unsupported_four_state_action_count = _count_unsupported_four_state_actions(build_result.timepoints)
            if unsupported_four_state_action_count:
                four_state_unsupported_map_count += 1
                four_state_unsupported_lane_action_count += unsupported_four_state_action_count
                continue
            invalid_hold_transition_action_count = _count_invalid_hold_transitions(build_result.timepoints)
            if invalid_hold_transition_action_count:
                invalid_hold_transition_map_count += 1
                invalid_hold_transition_count += invalid_hold_transition_action_count
                continue

            if max_maps_per_bin is not None:
                label = _difficulty_bin_label(difficulty)
                if label is None:
                    continue
                if retained_map_count_by_bin[label] >= max_maps_per_bin:
                    continue
                retained_map_count_by_bin[label] += 1

            retained_map_count += 1
            audio = load_audio_file(audio_path, sample_rate=self.sample_rate)
            audio_duration_ms = len(audio) * 1000.0 / self.sample_rate
            generation_end_ms = compute_generation_end_ms(audio_duration_ms, build_result.timepoints)
            for window in iter_window_specs(generation_end_ms):
                records.append(
                    OracleWindowRecord(
                        beatmap_path=beatmap_path,
                        audio_path=audio_path,
                        difficulty=difficulty,
                        timing_points=timing_points,
                        timepoints=tuple(build_result.timepoints),
                        window=window,
                        open_hold_mask=open_hold_mask_at_write_start(build_result.timepoints, window.write_start_ms),
                    ),
                )
            if progress:
                print(
                    f"dataset_progress retained_maps={retained_map_count} "
                    f"windows={len(records)} difficulty={difficulty:.2f}",
                    flush=True,
                )
        return records, OracleWindowFilterReport(
            source_map_count=source_map_count,
            difficulty_filtered_map_count=difficulty_filtered_map_count,
            candidate_map_count=len(index_df),
            missing_red_timing_map_count=missing_red_timing_map_count,
            invalid_red_timing_map_count=invalid_red_timing_map_count,
            negative_time_hitobject_map_count=negative_time_hitobject_map_count,
            unsupported_compound_map_count=unsupported_compound_map_count,
            unsupported_compound_event_count=unsupported_compound_event_count,
            four_state_unsupported_map_count=four_state_unsupported_map_count,
            four_state_unsupported_lane_action_count=four_state_unsupported_lane_action_count,
            zero_length_hold_normalized_count=zero_length_hold_normalized_count,
            retained_map_count=retained_map_count,
            generated_window_count=len(records),
            invalid_hold_transition_map_count=invalid_hold_transition_map_count,
            invalid_hold_transition_count=invalid_hold_transition_count,
        )


def build_decoder_tensors(tokenized: TokenizedWindow, vocab: Stage1Vocab) -> tuple[torch.Tensor, torch.Tensor]:
    decoder_input = tokenized.condition_ids + tokenized.target_ids[:-1]
    labels = [-100] * (len(tokenized.condition_ids) - 1) + tokenized.target_ids
    if len(decoder_input) != len(labels):
        raise ValueError(f"decoder input and labels length mismatch: {len(decoder_input)} != {len(labels)}")
    return torch.tensor(decoder_input, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def collate_oracle_windows(samples: Sequence[dict[str, Any]], *, pad_id: int = 0) -> dict[str, Any]:
    max_len = max(int(sample["decoder_input_ids"].shape[0]) for sample in samples)
    decoder_input_ids = torch.full((len(samples), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(samples), max_len), -100, dtype=torch.long)
    decoder_padding_mask = torch.ones((len(samples), max_len), dtype=torch.bool)

    for row, sample in enumerate(samples):
        length = int(sample["decoder_input_ids"].shape[0])
        decoder_input_ids[row, :length] = sample["decoder_input_ids"]
        labels[row, :length] = sample["labels"]
        decoder_padding_mask[row, :length] = False

    return {
        "packed_audio": torch.stack([sample["packed_audio"] for sample in samples]),
        "timing_track": torch.stack([sample["timing_track"] for sample in samples]),
        "difficulty_bucket": torch.stack([sample["difficulty_bucket"] for sample in samples]),
        "open_hold_mask": torch.stack([sample["open_hold_mask"] for sample in samples]),
        "write_duration_ms": torch.stack([sample["write_duration_ms"] for sample in samples]),
        "decoder_input_ids": decoder_input_ids,
        "decoder_padding_mask": decoder_padding_mask,
        "labels": labels,
        "metadata": [
            {
                "beatmap_path": sample["beatmap_path"],
                "audio_path": sample["audio_path"],
                "difficulty": sample["difficulty"],
                "write_start_ms": sample["write_start_ms"],
            }
            for sample in samples
        ],
    }


def filter_supported_difficulty_range(index_df: pd.DataFrame) -> pd.DataFrame:
    difficulty = pd.to_numeric(index_df["difficulty"], errors="coerce")
    return index_df[(difficulty >= 2.0) & (difficulty <= 6.0)].reset_index(drop=True)


def _filter_index_by_manifest(index_df: pd.DataFrame, manifest_path: Path) -> pd.DataFrame:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    beatmap_paths = {str(item["beatmap_path"] if isinstance(item, dict) else item) for item in manifest}
    return index_df[index_df["beatmap_path"].astype(str).isin(beatmap_paths)].reset_index(drop=True)


def _count_unsupported_four_state_actions(timepoints: Sequence) -> int:
    return sum(
        action in {LaneAction.END_TAP, LaneAction.END_START}
        for timepoint in timepoints
        for action in timepoint.lane_actions
    )


def _count_invalid_hold_transitions(timepoints: Sequence) -> int:
    invalid_transition_count = 0
    open_hold_mask = 0
    for timepoint in sorted(timepoints, key=lambda item: item.time_ms):
        for lane, action in enumerate(timepoint.lane_actions):
            lane_bit = 1 << lane
            is_open = (open_hold_mask & lane_bit) != 0
            if action == LaneAction.TAP:
                if is_open:
                    invalid_transition_count += 1
            elif action == LaneAction.HOLD_START:
                if is_open:
                    invalid_transition_count += 1
                else:
                    open_hold_mask |= lane_bit
            elif action == LaneAction.HOLD_END:
                if not is_open:
                    invalid_transition_count += 1
                else:
                    open_hold_mask &= ~lane_bit
    return invalid_transition_count


def _count_unsupported_compound_events(hitobjects: Sequence, *, key_count: int) -> int:
    primitive_actions: dict[int, dict[int, list[LaneAction]]] = defaultdict(lambda: defaultdict(list))
    for hitobject in hitobjects:
        if not 0 <= hitobject.lane < key_count:
            raise ValueError(f"hit object lane {hitobject.lane} outside 0..{key_count - 1}: {hitobject}")
        q_start = quantize_10ms_half_up(hitobject.start_time_ms)

        if hitobject.kind == ManiaHitObjectKind.TAP:
            primitive_actions[q_start][hitobject.lane].append(LaneAction.TAP)
            continue

        q_end = quantize_10ms_half_up(hitobject.end_time_ms)
        if q_end <= q_start:
            primitive_actions[q_start][hitobject.lane].append(LaneAction.TAP)
            continue

        primitive_actions[q_start][hitobject.lane].append(LaneAction.HOLD_START)
        primitive_actions[q_end][hitobject.lane].append(LaneAction.HOLD_END)

    return sum(
        1
        for lane_actions_by_time in primitive_actions.values()
        for actions in lane_actions_by_time.values()
        if not _is_supported_compound_action_list(actions)
    )


def _is_supported_compound_action_list(actions: Sequence[LaneAction]) -> bool:
    if len(actions) <= 1:
        return True
    action_set = frozenset(actions)
    return len(actions) == 2 and action_set in {
        frozenset({LaneAction.HOLD_END, LaneAction.TAP}),
        frozenset({LaneAction.HOLD_END, LaneAction.HOLD_START}),
    }


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
