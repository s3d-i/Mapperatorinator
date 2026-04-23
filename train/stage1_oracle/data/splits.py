from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from .dataset import load_index
from .windows import COARSE_BIN_LABELS, filter_supported_difficulty_range


AUDIO_GROUP_SPLIT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class AudioGroupManifestEntry:
    beatmap_path: str
    audio_group: str
    audio_path: str
    difficulty: float
    difficulty_bin: str
    shard: str


@dataclass(frozen=True)
class AudioGroupSplitSummary:
    audio_group_count: int
    beatmap_count: int
    beatmap_count_by_bin: dict[str, int]


@dataclass(frozen=True)
class AudioGroupSplitReport:
    schema_version: int
    split_policy: str
    grouping_key: str
    manifest_format: str
    index_path: str
    eval_ratio: float
    seed: int
    eval_target_beatmap_count_by_bin: dict[str, int]
    rollout_probe_target_beatmap_count_by_bin: dict[str, int] | None
    required_train_maps_per_bin: dict[str, int] | None
    eligible: AudioGroupSplitSummary
    train: AudioGroupSplitSummary
    eval: AudioGroupSplitSummary
    rollout_probe: AudioGroupSplitSummary | None
    train_manifest_path: str
    eval_manifest_path: str
    rollout_probe_manifest_path: str | None


@dataclass(frozen=True)
class AudioGroupSplitArtifacts:
    output_dir: Path
    train_manifest_path: Path
    eval_manifest_path: Path
    rollout_probe_manifest_path: Path | None
    report_path: Path
    report: AudioGroupSplitReport


@dataclass(frozen=True)
class _AudioGroupCandidate:
    audio_group: str
    entries: tuple[AudioGroupManifestEntry, ...]
    beatmap_count_by_bin: dict[str, int]
    total_maps: int
    random_order: int


def build_audio_group_split(
    *,
    index_path: str | Path,
    output_dir: str | Path,
    eval_ratio: float = 0.1,
    seed: int = 1337,
    rollout_probe_maps_per_bin: int | None = None,
    required_train_maps_per_bin: int | dict[str, int] | None = None,
) -> AudioGroupSplitArtifacts:
    if not 0.0 < eval_ratio < 1.0:
        raise ValueError(f"eval_ratio must be strictly between 0 and 1, got {eval_ratio}")
    if rollout_probe_maps_per_bin is not None and rollout_probe_maps_per_bin <= 0:
        raise ValueError(
            "rollout_probe_maps_per_bin must be positive when set, "
            f"got {rollout_probe_maps_per_bin}"
        )

    index_path = Path(index_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    eligible_groups = _build_audio_group_candidates(index_path=index_path, seed=seed)
    eligible_summary = _summarize_groups(eligible_groups)
    eval_target_beatmap_count_by_bin = {
        label: _rounded_holdout_target(eligible_summary.beatmap_count_by_bin[label], eval_ratio)
        for label in COARSE_BIN_LABELS
    }
    eval_groups = _select_groups_for_target_counts(
        eligible_groups,
        target_beatmap_count_by_bin=eval_target_beatmap_count_by_bin,
    )
    eval_audio_groups = {group.audio_group for group in eval_groups}
    train_groups = [group for group in eligible_groups if group.audio_group not in eval_audio_groups]
    train_summary = _summarize_groups(train_groups)
    eval_summary = _summarize_groups(eval_groups)

    required_train_maps = _normalize_required_maps_per_bin(required_train_maps_per_bin)
    if required_train_maps is not None:
        for label in COARSE_BIN_LABELS:
            available = train_summary.beatmap_count_by_bin[label]
            required = required_train_maps[label]
            if available < required:
                raise ValueError(
                    f"train split retains only {available} maps in bin {label}, "
                    f"below required {required}"
                )

    rollout_probe_groups: list[_AudioGroupCandidate] = []
    rollout_probe_summary: AudioGroupSplitSummary | None = None
    rollout_probe_target_beatmap_count_by_bin: dict[str, int] | None = None
    rollout_probe_manifest_path: Path | None = None
    if rollout_probe_maps_per_bin is not None:
        rollout_probe_target_beatmap_count_by_bin = {
            label: min(eval_summary.beatmap_count_by_bin[label], rollout_probe_maps_per_bin)
            for label in COARSE_BIN_LABELS
        }
        rollout_probe_groups = _select_groups_for_target_counts(
            eval_groups,
            target_beatmap_count_by_bin=rollout_probe_target_beatmap_count_by_bin,
        )
        rollout_probe_summary = _summarize_groups(rollout_probe_groups)
        rollout_probe_manifest_path = output_dir / "rollout_probe_manifest.json"
        _write_manifest(rollout_probe_manifest_path, rollout_probe_groups)

    train_manifest_path = output_dir / "train_manifest.json"
    eval_manifest_path = output_dir / "eval_manifest.json"
    report_path = output_dir / "split_report.json"
    _write_manifest(train_manifest_path, train_groups)
    _write_manifest(eval_manifest_path, eval_groups)

    report = AudioGroupSplitReport(
        schema_version=AUDIO_GROUP_SPLIT_SCHEMA_VERSION,
        split_policy="audio_group_holdout_before_window_expansion",
        grouping_key="shard+audio_path",
        manifest_format="json list of manifest entry objects",
        index_path=index_path.as_posix(),
        eval_ratio=eval_ratio,
        seed=seed,
        eval_target_beatmap_count_by_bin=eval_target_beatmap_count_by_bin,
        rollout_probe_target_beatmap_count_by_bin=rollout_probe_target_beatmap_count_by_bin,
        required_train_maps_per_bin=required_train_maps,
        eligible=eligible_summary,
        train=train_summary,
        eval=eval_summary,
        rollout_probe=rollout_probe_summary,
        train_manifest_path=train_manifest_path.as_posix(),
        eval_manifest_path=eval_manifest_path.as_posix(),
        rollout_probe_manifest_path=rollout_probe_manifest_path.as_posix()
        if rollout_probe_manifest_path is not None
        else None,
    )
    report_path.write_text(
        json.dumps(asdict(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return AudioGroupSplitArtifacts(
        output_dir=output_dir,
        train_manifest_path=train_manifest_path,
        eval_manifest_path=eval_manifest_path,
        rollout_probe_manifest_path=rollout_probe_manifest_path,
        report_path=report_path,
        report=report,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create Stage 1 audio-group train/eval manifests before window expansion."
    )
    parser.add_argument(
        "--index-path",
        default="train/artifacts/indexes/beatmap_index_4k_no_timing_anomalies.parquet",
    )
    parser.add_argument(
        "--output-dir",
        default="train/artifacts/splits/stage1_oracle/audio_group_holdout_seed1337_eval10",
    )
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--rollout-probe-maps-per-bin", type=int, default=None)
    parser.add_argument("--required-train-maps-per-bin", type=int, default=None)
    args = parser.parse_args(argv)

    artifacts = build_audio_group_split(
        index_path=args.index_path,
        output_dir=args.output_dir,
        eval_ratio=args.eval_ratio,
        seed=args.seed,
        rollout_probe_maps_per_bin=args.rollout_probe_maps_per_bin,
        required_train_maps_per_bin=args.required_train_maps_per_bin,
    )
    report = artifacts.report
    command = "uv run python -m train.stage1_oracle.data.splits " + " ".join(
        sys.argv[1:] if argv is None else argv
    )
    print(f"command {command}")
    print(f"eligible_audio_groups {report.eligible.audio_group_count}")
    print(f"eligible_beatmaps {report.eligible.beatmap_count}")
    print(f"train_beatmaps {report.train.beatmap_count}")
    print(f"eval_beatmaps {report.eval.beatmap_count}")
    if report.rollout_probe is not None:
        print(f"rollout_probe_beatmaps {report.rollout_probe.beatmap_count}")
    print(f"train_manifest {artifacts.train_manifest_path.as_posix()}")
    print(f"eval_manifest {artifacts.eval_manifest_path.as_posix()}")
    if artifacts.rollout_probe_manifest_path is not None:
        print(f"rollout_probe_manifest {artifacts.rollout_probe_manifest_path.as_posix()}")
    print(f"report_json {artifacts.report_path.as_posix()}")
    return 0


def _build_audio_group_candidates(*, index_path: Path, seed: int) -> list[_AudioGroupCandidate]:
    index_df = filter_supported_difficulty_range(load_index(index_path))
    required_columns = {"shard", "audio_path", "beatmap_path", "difficulty"}
    missing_columns = sorted(required_columns - set(index_df.columns))
    if missing_columns:
        raise ValueError(f"index missing required columns: {missing_columns}")

    grouped_entries: dict[str, list[AudioGroupManifestEntry]] = {}
    for row in index_df.sort_values(["shard", "audio_path", "beatmap_path"], kind="mergesort").itertuples(index=False):
        difficulty = float(row.difficulty)
        difficulty_bin = _difficulty_bin_label(difficulty)
        if difficulty_bin is None:
            continue
        shard = str(row.shard)
        audio_path = str(row.audio_path)
        audio_group = _audio_group_key(shard, audio_path)
        grouped_entries.setdefault(audio_group, []).append(
            AudioGroupManifestEntry(
                beatmap_path=str(row.beatmap_path),
                audio_group=audio_group,
                audio_path=audio_path,
                difficulty=difficulty,
                difficulty_bin=difficulty_bin,
                shard=shard,
            )
        )

    ordered_audio_groups = sorted(grouped_entries)
    rng = random.Random(seed)
    shuffled_audio_groups = ordered_audio_groups[:]
    rng.shuffle(shuffled_audio_groups)
    random_order_by_group = {audio_group: index for index, audio_group in enumerate(shuffled_audio_groups)}

    groups: list[_AudioGroupCandidate] = []
    for audio_group in ordered_audio_groups:
        entries = tuple(sorted(grouped_entries[audio_group], key=lambda entry: entry.beatmap_path))
        beatmap_count_by_bin = {label: 0 for label in COARSE_BIN_LABELS}
        for entry in entries:
            beatmap_count_by_bin[entry.difficulty_bin] += 1
        groups.append(
            _AudioGroupCandidate(
                audio_group=audio_group,
                entries=entries,
                beatmap_count_by_bin=beatmap_count_by_bin,
                total_maps=len(entries),
                random_order=random_order_by_group[audio_group],
            )
        )
    return groups


def _select_groups_for_target_counts(
    groups: list[_AudioGroupCandidate],
    *,
    target_beatmap_count_by_bin: dict[str, int],
) -> list[_AudioGroupCandidate]:
    remaining_targets = {
        label: max(0, int(target_beatmap_count_by_bin.get(label, 0)))
        for label in COARSE_BIN_LABELS
    }
    selected: list[_AudioGroupCandidate] = []
    candidates = list(groups)

    while any(remaining_targets[label] > 0 for label in COARSE_BIN_LABELS):
        best_index: int | None = None
        best_score: tuple[float, int, int, int, int, int] | None = None
        for index, group in enumerate(candidates):
            covered = sum(
                min(group.beatmap_count_by_bin[label], remaining_targets[label])
                for label in COARSE_BIN_LABELS
            )
            if covered <= 0:
                continue
            overflow = group.total_maps - covered
            active_bin_count = sum(group.beatmap_count_by_bin[label] > 0 for label in COARSE_BIN_LABELS)
            score = (
                covered / group.total_maps,
                covered,
                -overflow,
                -active_bin_count,
                -group.total_maps,
                -group.random_order,
            )
            if best_score is None or score > best_score:
                best_index = index
                best_score = score
        if best_index is None:
            raise ValueError(
                "unable to satisfy target beatmap counts without splitting an audio group; "
                f"remaining targets: {remaining_targets}"
            )

        chosen = candidates.pop(best_index)
        selected.append(chosen)
        for label in COARSE_BIN_LABELS:
            remaining_targets[label] = max(0, remaining_targets[label] - chosen.beatmap_count_by_bin[label])

    return sorted(selected, key=lambda group: group.audio_group)


def _summarize_groups(groups: list[_AudioGroupCandidate]) -> AudioGroupSplitSummary:
    beatmap_count_by_bin = {label: 0 for label in COARSE_BIN_LABELS}
    for group in groups:
        for label in COARSE_BIN_LABELS:
            beatmap_count_by_bin[label] += group.beatmap_count_by_bin[label]
    return AudioGroupSplitSummary(
        audio_group_count=len(groups),
        beatmap_count=sum(beatmap_count_by_bin.values()),
        beatmap_count_by_bin=beatmap_count_by_bin,
    )


def _write_manifest(path: Path, groups: list[_AudioGroupCandidate]) -> None:
    manifest_entries = [
        asdict(entry)
        for group in sorted(groups, key=lambda item: item.audio_group)
        for entry in group.entries
    ]
    path.write_text(
        json.dumps(manifest_entries, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _normalize_required_maps_per_bin(
    required_train_maps_per_bin: int | dict[str, int] | None,
) -> dict[str, int] | None:
    if required_train_maps_per_bin is None:
        return None
    if isinstance(required_train_maps_per_bin, int):
        if required_train_maps_per_bin <= 0:
            raise ValueError(
                "required_train_maps_per_bin must be positive when set, "
                f"got {required_train_maps_per_bin}"
            )
        return {label: required_train_maps_per_bin for label in COARSE_BIN_LABELS}
    if not isinstance(required_train_maps_per_bin, dict):
        raise ValueError(
            "required_train_maps_per_bin must be an integer or per-bin mapping, "
            f"got {required_train_maps_per_bin}"
        )

    missing = [label for label in COARSE_BIN_LABELS if label not in required_train_maps_per_bin]
    unknown = sorted(set(required_train_maps_per_bin) - set(COARSE_BIN_LABELS))
    if missing:
        raise ValueError(f"required_train_maps_per_bin missing bins: {missing}")
    if unknown:
        raise ValueError(f"required_train_maps_per_bin unknown bins: {unknown}")

    normalized: dict[str, int] = {}
    for label in COARSE_BIN_LABELS:
        value = int(required_train_maps_per_bin[label])
        if value <= 0:
            raise ValueError(
                f"required_train_maps_per_bin[{label}] must be positive, "
                f"got {required_train_maps_per_bin[label]}"
            )
        normalized[label] = value
    return normalized


def _rounded_holdout_target(total_count: int, eval_ratio: float) -> int:
    if total_count <= 1:
        return 0
    rounded = int(math.floor((total_count * eval_ratio) + 0.5))
    return min(total_count - 1, max(1, rounded))


def _audio_group_key(shard: str, audio_path: str) -> str:
    return f"{shard}:{audio_path}"


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


if __name__ == "__main__":
    raise SystemExit(main())
