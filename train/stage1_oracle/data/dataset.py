from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypeAlias, TypedDict, cast, runtime_checkable

import pandas as pd
from torch.utils.data import Dataset

from ..core.difficulty import calculate_mania_difficulties
from ..features.audio import AudioWaveform, load_audio_file
from ..osu.metadata import parse_osu_metadata
from ..osu.timing import InvalidRedTimingError
from ..osu.timing import MissingRedTimingError
from ..osu.timing import require_red_timing_points

INDEX_4K_FILENAME = "beatmap_index_4k.parquet"
INDEX_4K_NO_TIMING_ANOMALIES_FILENAME = "beatmap_index_4k_no_timing_anomalies.parquet"
SR_SPEEDS = (0.5, 0.75, 1.0, 1.25, 1.5)
NULLABLE_INT_COLUMNS = (
    "audio_lead_in",
    "preview_time",
    "mode",
    "beatmap_id",
    "key_count",
)
FLOAT_COLUMNS = (
    "hp_drain_rate",
    "circle_size",
    "overall_difficulty",
    "difficulty",
)
NormalizedValue: TypeAlias = None | bool | int | float | str | bytes | list["NormalizedValue"]


@runtime_checkable
class _SupportsToList(Protocol):
    def tolist(self) -> object: ...


@runtime_checkable
class _SupportsItem(Protocol):
    def item(self) -> object: ...


class ManiaBeatmapSample(TypedDict):
    audio: AudioWaveform
    audio_num_samples: int
    sample_rate: int
    audio_path: str
    beatmap_path: str
    beatmap_set_path: str
    shard: str
    beatmap_set_id: int | str
    beatmap_filename: str
    audio_filename: str
    audio_lead_in: int | None
    preview_time: int | None
    mode: int | None
    title: str | None
    artist: str | None
    creator: str | None
    version: str | None
    beatmap_id: int | None
    hp_drain_rate: float | None
    circle_size: float | None
    overall_difficulty: float | None
    key_count: int | None
    difficulty: float
    sr_difficulties: list[float]


@dataclass(frozen=True)
class BeatmapIndexRecord:
    shard: str
    beatmap_set_id: int | str
    beatmap_set_path: str
    beatmap_path: str
    beatmap_filename: str
    audio_path: str
    audio_filename: str
    audio_lead_in: int | None
    preview_time: int | None
    mode: int | None
    title: str | None
    artist: str | None
    creator: str | None
    version: str | None
    beatmap_id: int | None
    hp_drain_rate: float | None
    circle_size: float | None
    overall_difficulty: float | None
    key_count: int | None
    difficulty: float
    sr_difficulties: list[float]


@dataclass(frozen=True)
class TimingCleanIndexReport:
    source_index_path: Path
    output_path: Path
    dataset_root: Path
    source_map_count: int
    clean_map_count: int
    missing_red_timing_map_count: int
    invalid_red_timing_map_count: int
    invalid_red_timing_point_count: int
    nonfinite_red_timing_point_count: int
    nonpositive_red_timing_point_count: int
    implausible_red_timing_point_count: int


def _build_record(
    shard_path: Path,
    beatmap_set_path: Path,
    beatmap_path: Path,
    *,
    metadata=None,
) -> BeatmapIndexRecord | None:
    metadata = parse_osu_metadata(beatmap_path) if metadata is None else metadata
    if not metadata.audio_filename:
        return None
    if metadata.mode != 3:
        return None

    audio_path = beatmap_set_path / metadata.audio_filename
    if not audio_path.is_file():
        return None

    beatmap_set_id: int | str
    if metadata.beatmap_set_id is not None:
        beatmap_set_id = metadata.beatmap_set_id
    else:
        try:
            beatmap_set_id = int(beatmap_set_path.name)
        except ValueError:
            beatmap_set_id = beatmap_set_path.name

    sr_difficulties = [_round_2f(value) for value in calculate_mania_difficulties(beatmap_path, audio_path, SR_SPEEDS)]
    difficulty = sr_difficulties[SR_SPEEDS.index(1.0)]

    return BeatmapIndexRecord(
        shard=shard_path.name,
        beatmap_set_id=beatmap_set_id,
        beatmap_set_path=beatmap_set_path.relative_to(shard_path).as_posix(),
        beatmap_path=beatmap_path.relative_to(shard_path).as_posix(),
        beatmap_filename=beatmap_path.name,
        audio_path=audio_path.relative_to(shard_path).as_posix(),
        audio_filename=metadata.audio_filename,
        audio_lead_in=metadata.audio_lead_in,
        preview_time=metadata.preview_time,
        mode=metadata.mode,
        title=metadata.title,
        artist=metadata.artist,
        creator=metadata.creator,
        version=metadata.version,
        beatmap_id=metadata.beatmap_id,
        hp_drain_rate=metadata.hp_drain_rate,
        circle_size=metadata.circle_size,
        overall_difficulty=metadata.overall_difficulty,
        key_count=metadata.key_count,
        difficulty=difficulty,
        sr_difficulties=sr_difficulties,
    )


def build_4k_index(shard_path: str | Path, output_path: str | Path) -> Path:
    return build_filtered_index(
        shard_path=shard_path,
        output_path=output_path,
        key_count=4,
    )


def build_filtered_index(
    shard_path: str | Path,
    output_path: str | Path,
    *,
    key_count: int | None = None,
) -> Path:
    shard_path = Path(shard_path)
    output_path = Path(output_path)
    rows: list[dict[str, object]] = []

    for beatmap_set_path in sorted(path for path in shard_path.iterdir() if path.is_dir()):
        for beatmap_path in sorted(beatmap_set_path.glob("*.osu")):
            metadata = parse_osu_metadata(beatmap_path)
            if metadata.mode != 3:
                continue
            if key_count is not None and metadata.key_count != key_count:
                continue

            record = _build_record(
                shard_path,
                beatmap_set_path,
                beatmap_path,
                metadata=metadata,
            )
            if record is None:
                continue
            rows.append(asdict(record))

    index_df = pd.DataFrame.from_records(
        rows,
        columns=[
            "shard",
            "beatmap_set_id",
            "beatmap_set_path",
            "beatmap_path",
            "beatmap_filename",
            "audio_path",
            "audio_filename",
            "audio_lead_in",
            "preview_time",
            "mode",
            "title",
            "artist",
            "creator",
            "version",
            "beatmap_id",
            "hp_drain_rate",
            "circle_size",
            "overall_difficulty",
            "key_count",
            "difficulty",
            "sr_difficulties",
        ],
    )
    index_df = _cast_index_dtypes(index_df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    index_df.to_parquet(output_path, index=False)
    return output_path


def build_4k_no_timing_anomaly_index(
    *,
    source_index_path: str | Path,
    dataset_root: str | Path,
    output_path: str | Path,
) -> TimingCleanIndexReport:
    source_index_path = Path(source_index_path)
    dataset_root = Path(dataset_root)
    output_path = Path(output_path)
    source_df = load_index(source_index_path)

    keep_mask: list[bool] = []
    missing_red_timing_map_count = 0
    invalid_red_timing_map_count = 0
    invalid_red_timing_point_count = 0
    nonfinite_red_timing_point_count = 0
    nonpositive_red_timing_point_count = 0
    implausible_red_timing_point_count = 0

    for row in source_df.itertuples(index=False):
        beatmap_path = dataset_root / str(row.shard) / str(row.beatmap_path)
        try:
            # The dense timing audit classifies maps with missing or impossible red timing
            # as unusable for oracle timing. Keep the training index on that same parser
            # gate so timing anomalies cannot enter feature generation through a stale index.
            require_red_timing_points(beatmap_path)
        except InvalidRedTimingError as exc:
            keep_mask.append(False)
            invalid_red_timing_map_count += 1
            invalid_red_timing_point_count += exc.counts.total
            nonfinite_red_timing_point_count += exc.counts.nonfinite
            nonpositive_red_timing_point_count += exc.counts.nonpositive
            implausible_red_timing_point_count += exc.counts.implausible
        except MissingRedTimingError:
            keep_mask.append(False)
            missing_red_timing_map_count += 1
        else:
            keep_mask.append(True)

    clean_df = _cast_index_dtypes(source_df.loc[keep_mask].reset_index(drop=True))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_parquet(output_path, index=False)

    return TimingCleanIndexReport(
        source_index_path=source_index_path,
        output_path=output_path,
        dataset_root=dataset_root,
        source_map_count=len(source_df),
        clean_map_count=len(clean_df),
        missing_red_timing_map_count=missing_red_timing_map_count,
        invalid_red_timing_map_count=invalid_red_timing_map_count,
        invalid_red_timing_point_count=invalid_red_timing_point_count,
        nonfinite_red_timing_point_count=nonfinite_red_timing_point_count,
        nonpositive_red_timing_point_count=nonpositive_red_timing_point_count,
        implausible_red_timing_point_count=implausible_red_timing_point_count,
    )


def load_index(index_path: str | Path) -> pd.DataFrame:
    return _cast_index_dtypes(pd.read_parquet(index_path))


def get_default_4k_index_path() -> Path:
    return Path(__file__).resolve().parents[2] / "artifacts" / "indexes" / INDEX_4K_FILENAME


def get_default_4k_no_timing_anomaly_index_path() -> Path:
    return Path(__file__).resolve().parents[2] / "artifacts" / "indexes" / INDEX_4K_NO_TIMING_ANOMALIES_FILENAME


def get_default_4k_training_index_path() -> Path:
    return get_default_4k_no_timing_anomaly_index_path()


class ManiaBeatmapDataset(Dataset):
    def __init__(
        self,
        shard_path: str | Path,
        *,
        sample_rate: int,
        speed: float = 1.0,
        normalize: bool = True,
        index_path: str | Path | None = None,
        build_index_if_missing: bool = True,
    ) -> None:
        self.shard_path = Path(shard_path)
        self.sample_rate = sample_rate
        self.speed = speed
        self.normalize = normalize
        using_default_index = index_path is None
        index_path = Path(index_path) if index_path is not None else get_default_4k_training_index_path()
        if not index_path.exists():
            if not build_index_if_missing:
                raise FileNotFoundError(f"index parquet not found: {index_path}")
            if using_default_index:
                _build_default_4k_training_index(self.shard_path, index_path)
            else:
                build_4k_index(self.shard_path, index_path)
        self.index = load_index(index_path).reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> ManiaBeatmapSample:
        row = self.index.iloc[index]
        normalized = {column: _normalize_scalar(row[column]) for column in self.index.columns}
        audio_rel_path = cast(str, normalized["audio_path"])
        beatmap_rel_path = cast(str, normalized["beatmap_path"])
        beatmap_set_rel_path = cast(str, normalized["beatmap_set_path"])
        audio_path = self.shard_path / audio_rel_path
        beatmap_path = self.shard_path / beatmap_rel_path
        beatmap_set_path = self.shard_path / beatmap_set_rel_path
        audio = load_audio_file(
            audio_path,
            sample_rate=self.sample_rate,
            speed=self.speed,
            normalize=self.normalize,
        )

        return {
            "audio": audio,
            "audio_num_samples": len(audio),
            "sample_rate": self.sample_rate,
            "audio_path": str(audio_path),
            "beatmap_path": str(beatmap_path),
            "beatmap_set_path": str(beatmap_set_path),
            "shard": cast(str, normalized["shard"]),
            "beatmap_set_id": cast(int | str, normalized["beatmap_set_id"]),
            "beatmap_filename": cast(str, normalized["beatmap_filename"]),
            "audio_filename": cast(str, normalized["audio_filename"]),
            "audio_lead_in": cast(int | None, normalized["audio_lead_in"]),
            "preview_time": cast(int | None, normalized["preview_time"]),
            "mode": cast(int | None, normalized["mode"]),
            "title": cast(str | None, normalized["title"]),
            "artist": cast(str | None, normalized["artist"]),
            "creator": cast(str | None, normalized["creator"]),
            "version": cast(str | None, normalized["version"]),
            "beatmap_id": cast(int | None, normalized["beatmap_id"]),
            "hp_drain_rate": cast(float | None, normalized["hp_drain_rate"]),
            "circle_size": cast(float | None, normalized["circle_size"]),
            "overall_difficulty": cast(float | None, normalized["overall_difficulty"]),
            "key_count": cast(int | None, normalized["key_count"]),
            "difficulty": cast(float, normalized["difficulty"]),
            "sr_difficulties": cast(list[float], normalized["sr_difficulties"]),
        }


def _normalize_scalar(value: object) -> NormalizedValue:
    if isinstance(value, list):
        return [_normalize_scalar(item) for item in value]
    if isinstance(value, tuple):
        return [_normalize_scalar(item) for item in value]
    if isinstance(value, _SupportsToList) and not isinstance(value, (str, bytes)):
        return _normalize_scalar(value.tolist())
    if pd.isna(value):
        return None
    if isinstance(value, _SupportsItem) and not isinstance(value, (str, bytes)):
        return _normalize_scalar(value.item())
    return cast(NormalizedValue, value)


def _build_default_4k_training_index(shard_path: Path, output_path: Path) -> TimingCleanIndexReport:
    source_index_path = get_default_4k_index_path()
    if not source_index_path.exists():
        build_4k_index(shard_path, source_index_path)
    return build_4k_no_timing_anomaly_index(
        source_index_path=source_index_path,
        dataset_root=shard_path.parent,
        output_path=output_path,
    )


def _cast_index_dtypes(index_df: pd.DataFrame) -> pd.DataFrame:
    for column in NULLABLE_INT_COLUMNS:
        if column in index_df.columns:
            index_df[column] = pd.to_numeric(index_df[column], errors="coerce").astype("Int64")

    for column in FLOAT_COLUMNS:
        if column in index_df.columns:
            index_df[column] = pd.to_numeric(index_df[column], errors="coerce").astype("Float64")

    return index_df


def _round_2f(value: float) -> float:
    return float(f"{value:.2f}")
