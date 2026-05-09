from __future__ import annotations

import argparse
import shlex
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from train.stage_2.data.mapper_v1_windows import CONTROL_TEACHER_CACHE_SCHEMA_VERSION, MAPPER_DENSITY_FRAMES


DEFAULT_CACHE_DIR = Path("train/artifacts/cache/stage2_mapper_v1/control_teacher_d384_l3_stride16_step002000")
CONTROL_MEMORY_KEY = "control_memory_8s"
DENSITY_TEACHER_KEY = "density_teacher_8s"


@dataclass(frozen=True)
class CacheFileCompactionResult:
    path: Path
    before_bytes: int
    after_bytes: int
    changed: bool

    @property
    def saved_bytes(self) -> int:
        return max(0, self.before_bytes - self.after_bytes)


@dataclass(frozen=True)
class CacheCompactionRunReport:
    cache_dir: Path
    scanned_files: int
    processed_files: int
    changed_files: int
    before_bytes: int
    after_bytes: int
    saved_bytes: int
    dry_run: bool
    elapsed_s: float


def compact_mapper_v1_control_teacher_cache(
    *,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    limit: int | None = None,
    dry_run: bool = True,
    verify: bool = True,
    progress_every: int = 100,
) -> CacheCompactionRunReport:
    if limit is not None and limit < 0:
        raise ValueError(f"limit must be non-negative, got {limit!r}")
    if progress_every < 0:
        raise ValueError(f"progress_every must be non-negative, got {progress_every!r}")

    started_at = time.perf_counter()
    cache_dir = Path(cache_dir)
    paths = list(_iter_cache_paths(cache_dir))
    selected_paths = paths if limit is None else paths[:limit]

    processed = 0
    changed = 0
    before_bytes = 0
    after_bytes = 0
    for path in selected_paths:
        result = compact_control_teacher_cache_file(path, dry_run=dry_run, verify=verify)
        processed += 1
        changed += int(result.changed)
        before_bytes += result.before_bytes
        after_bytes += result.after_bytes
        if progress_every and (processed == 1 or processed % progress_every == 0):
            print(
                "mapper_v1_control_teacher_cache_compact progress "
                f"processed={processed}/{len(selected_paths)} changed={changed} "
                f"saved_gib={_gib(before_bytes - after_bytes):.3f}",
                flush=True,
            )

    return CacheCompactionRunReport(
        cache_dir=cache_dir,
        scanned_files=len(paths),
        processed_files=processed,
        changed_files=changed,
        before_bytes=before_bytes,
        after_bytes=after_bytes,
        saved_bytes=max(0, before_bytes - after_bytes),
        dry_run=dry_run,
        elapsed_s=time.perf_counter() - started_at,
    )


def compact_control_teacher_cache_file(
    path: str | Path,
    *,
    dry_run: bool = True,
    verify: bool = True,
) -> CacheFileCompactionResult:
    cache_path = Path(path)
    before_bytes = cache_path.stat().st_size
    payload = _load_payload(cache_path)
    compact_payload = dict(payload)
    compact_payload[CONTROL_MEMORY_KEY] = _compact_tensor(_require_tensor(payload, CONTROL_MEMORY_KEY))
    compact_payload[DENSITY_TEACHER_KEY] = _compact_tensor(_require_tensor(payload, DENSITY_TEACHER_KEY))
    _validate_payload(compact_payload, source=cache_path)

    if dry_run:
        return CacheFileCompactionResult(
            path=cache_path,
            before_bytes=before_bytes,
            after_bytes=before_bytes,
            changed=False,
        )

    tmp_path = cache_path.with_name(f"{cache_path.name}.compact.tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    try:
        torch.save(compact_payload, tmp_path)
        if verify:
            _verify_compacted_payload(tmp_path, reference=compact_payload)
        after_bytes = tmp_path.stat().st_size
        tmp_path.replace(cache_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    return CacheFileCompactionResult(
        path=cache_path,
        before_bytes=before_bytes,
        after_bytes=after_bytes,
        changed=after_bytes < before_bytes,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compact mapper v1 control-teacher cache .pt files.")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--write", action="store_true", help="Rewrite files in place. Default is dry-run.")
    parser.add_argument("--no-verify", action="store_true", help="Skip loading the temporary compacted file before replace.")
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args(argv)

    report = compact_mapper_v1_control_teacher_cache(
        cache_dir=args.cache_dir,
        limit=args.limit,
        dry_run=not args.write,
        verify=not args.no_verify,
        progress_every=args.progress_every,
    )
    mode = "write" if args.write else "dry_run"
    print(
        "mapper_v1_control_teacher_cache_compact "
        f"mode={mode} scanned={report.scanned_files} processed={report.processed_files} "
        f"changed={report.changed_files} before_gib={_gib(report.before_bytes):.3f} "
        f"after_gib={_gib(report.after_bytes):.3f} saved_gib={_gib(report.saved_bytes):.3f} "
        f"elapsed_s={report.elapsed_s:.1f} command={_format_command(argv)}",
        flush=True,
    )
    return 0


def _iter_cache_paths(cache_dir: Path) -> Sequence[Path]:
    if not cache_dir.exists():
        raise FileNotFoundError(f"cache dir does not exist: {cache_dir}")
    return sorted(path for path in cache_dir.glob("*/*.pt") if path.is_file())


def _load_payload(path: Path) -> Mapping[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError(f"cache file must contain a mapping: {path}")
    _validate_payload(payload, source=path)
    return payload


def _validate_payload(payload: Mapping[str, Any], *, source: Path) -> None:
    schema_version = payload.get("schema_version")
    if int(schema_version) != CONTROL_TEACHER_CACHE_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported control teacher cache schema {schema_version}; "
            f"expected {CONTROL_TEACHER_CACHE_SCHEMA_VERSION}: {source}",
        )
    control_memory = _require_tensor(payload, CONTROL_MEMORY_KEY)
    density_teacher = _require_tensor(payload, DENSITY_TEACHER_KEY)
    if control_memory.ndim != 2 or int(control_memory.shape[0]) != MAPPER_DENSITY_FRAMES:
        raise ValueError(
            f"{CONTROL_MEMORY_KEY} must have shape [{MAPPER_DENSITY_FRAMES},D], "
            f"got {tuple(control_memory.shape)}: {source}",
        )
    if int(control_memory.shape[1]) <= 0:
        raise ValueError(f"{CONTROL_MEMORY_KEY} must have positive hidden dim: {source}")
    if tuple(density_teacher.shape) != (MAPPER_DENSITY_FRAMES, 1):
        raise ValueError(
            f"{DENSITY_TEACHER_KEY} must have shape [{MAPPER_DENSITY_FRAMES},1], "
            f"got {tuple(density_teacher.shape)}: {source}",
        )
    if control_memory.dtype != torch.float32 or density_teacher.dtype != torch.float32:
        raise ValueError(f"cache tensors must be float32: {source}")
    if not torch.isfinite(control_memory).all() or not torch.isfinite(density_teacher).all():
        raise ValueError(f"cache tensors must be finite: {source}")


def _verify_compacted_payload(path: Path, *, reference: Mapping[str, Any]) -> None:
    loaded = _load_payload(path)
    for key in (CONTROL_MEMORY_KEY, DENSITY_TEACHER_KEY):
        if not torch.equal(_require_tensor(loaded, key), _require_tensor(reference, key)):
            raise ValueError(f"compacted tensor mismatch for {key}: {path}")
    for key, value in reference.items():
        if key in (CONTROL_MEMORY_KEY, DENSITY_TEACHER_KEY):
            continue
        if loaded.get(key) != value:
            raise ValueError(f"compacted metadata mismatch for {key}: {path}")


def _require_tensor(payload: Mapping[str, Any], key: str) -> torch.Tensor:
    value = payload.get(key)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"cache payload missing tensor {key}")
    return value


def _compact_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to(device="cpu", dtype=torch.float32).clone(memory_format=torch.contiguous_format)


def _gib(byte_count: int) -> float:
    return float(byte_count) / 1024.0 / 1024.0 / 1024.0


def _format_command(argv: Sequence[str] | None) -> str:
    args = list(argv) if argv is not None else []
    return "uv run python -m train.stage_2.data.compact_mapper_v1_control_teacher_cache " + " ".join(
        shlex.quote(str(arg)) for arg in args
    )


if __name__ == "__main__":
    raise SystemExit(main())
