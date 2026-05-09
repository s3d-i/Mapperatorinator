from __future__ import annotations

import argparse
import json
import os
import signal
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml


DEFAULT_CONFIG_PATH = Path("train/stage_2/training/configs/stage2_mapper_v2_phase_b_global_d768_l8_mps.yaml")
DEFAULT_UV_COMMAND = "uv run python -m train.stage_2.training.mapper_v2"


@dataclass(frozen=True)
class RunProgress:
    completed_steps: int
    is_complete: bool


def load_config(config_path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"overnight config must be a YAML mapping: {config_path}")
    return dict(loaded)


def read_progress(report_path: Path, *, max_steps: int) -> RunProgress:
    if not report_path.is_file():
        return RunProgress(completed_steps=0, is_complete=False)
    try:
        loaded = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return RunProgress(completed_steps=0, is_complete=False)
    if not isinstance(loaded, Mapping):
        return RunProgress(completed_steps=0, is_complete=False)
    raw_completed = loaded.get("completed_steps", 0)
    completed_steps = int(raw_completed) if isinstance(raw_completed, int) and raw_completed >= 0 else 0
    is_complete = bool(loaded.get("is_complete", False)) or completed_steps >= max_steps
    return RunProgress(completed_steps=completed_steps, is_complete=is_complete)


def archive_checkpoint_path(output_dir: Path, completed_steps: int) -> Path:
    return output_dir / "checkpoints" / f"checkpoint_step_{completed_steps:06d}.pt"


def existing_resume_checkpoint(config: Mapping[str, Any], output_dir: Path) -> Path | None:
    latest_checkpoint = output_dir / "checkpoint.pt"
    if latest_checkpoint.is_file():
        return latest_checkpoint
    configured_resume = config.get("resume_from")
    if isinstance(configured_resume, str) and configured_resume:
        resume_path = Path(configured_resume)
        if resume_path.is_file():
            return resume_path
    return None


def next_saved_step_target(*, completed_steps: int, max_steps: int, save_every: int, steps_per_process: int) -> int:
    if completed_steps >= max_steps:
        return max_steps
    threshold = min(max_steps, completed_steps + steps_per_process)
    if threshold == max_steps:
        return max_steps
    if threshold <= 1 and completed_steps < 1:
        return 1
    next_boundary = ((threshold + save_every - 1) // save_every) * save_every
    return min(max_steps, max(next_boundary, completed_steps + 1))


def write_child_config(
    *,
    base_config: Mapping[str, Any],
    output_dir: Path,
    config_path: Path,
    resume_checkpoint: Path | None,
) -> None:
    child_config = dict(base_config)
    child_config["output_dir"] = output_dir.as_posix()
    if resume_checkpoint is None:
        child_config.pop("resume_from", None)
    else:
        child_config["resume_from"] = resume_checkpoint.as_posix()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(child_config, sort_keys=False), encoding="utf-8")


def checkpoint_saved_after(
    *,
    output_dir: Path,
    checkpoint_path: Path,
    completed_steps: int,
    target_step: int,
    previous_mtime_ns: int | None,
) -> bool:
    if completed_steps < target_step or not checkpoint_path.is_file():
        return False
    archive_path = archive_checkpoint_path(output_dir, completed_steps)
    if archive_path.is_file():
        return True
    if previous_mtime_ns is None:
        return True
    return checkpoint_path.stat().st_mtime_ns > previous_mtime_ns


def terminate_process_group(process: subprocess.Popen[object], *, timeout_s: float) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=timeout_s)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    process.wait()


def sleep_until_stop_or_timeout(seconds: float, stop_requested: Callable[[], bool]) -> bool:
    deadline = time.monotonic() + max(float(seconds), 0.0)
    while time.monotonic() < deadline:
        if stop_requested():
            return False
        time.sleep(min(1.0, max(deadline - time.monotonic(), 0.0)))
    return not stop_requested()


def run_supervisor(args: argparse.Namespace, trainer_args: Sequence[str]) -> int:
    config_path = Path(args.config)
    base_config = load_config(config_path)
    output_dir = Path(args.output_dir or base_config.get("output_dir", "train/artifacts/runs/stage2_mapper_v2/overnight"))
    max_steps = int(args.max_steps or base_config.get("max_steps", 5000))
    save_every = int(args.save_every or base_config.get("save_every") or base_config.get("eval_every", 100))
    if max_steps <= 0:
        raise ValueError(f"max_steps must be positive, got {max_steps}")
    if save_every <= 0:
        raise ValueError(f"save_every must be positive, got {save_every}")
    steps_per_process = int(args.steps_per_process or save_every)
    if steps_per_process <= 0:
        raise ValueError(f"steps_per_process must be positive, got {steps_per_process}")
    base_config["output_dir"] = output_dir.as_posix()
    base_config["max_steps"] = max_steps
    base_config["save_every"] = save_every

    output_dir.mkdir(parents=True, exist_ok=True)
    supervisor_dir = output_dir / "overnight_supervisor"
    log_dir = Path(args.log_dir) if args.log_dir is not None else supervisor_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    child_config_path = supervisor_dir / "mapper_v2_child.yaml"
    checkpoint_path = output_dir / "checkpoint.pt"
    report_path = output_dir / "report.json"
    stop_signal: int | None = None

    def request_stop(signum: int, _frame: object) -> None:
        nonlocal stop_signal
        if stop_signal is None:
            stop_signal = int(signum)
            print(f"overnight_stop_requested signal={signum}", flush=True)

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    def should_stop() -> bool:
        return stop_signal is not None

    run_count = 0
    consecutive_failures = 0
    while True:
        if should_stop():
            return 128 + int(stop_signal or signal.SIGTERM)
        progress = read_progress(report_path, max_steps=max_steps)
        if progress.is_complete and args.stop_when_complete:
            print(f"overnight_complete step={progress.completed_steps}/{max_steps}", flush=True)
            return 0
        if args.max_runs and run_count >= args.max_runs:
            print(f"overnight_max_runs_reached runs={run_count} step={progress.completed_steps}/{max_steps}", flush=True)
            return 0

        resume_checkpoint = existing_resume_checkpoint(base_config, output_dir)
        write_child_config(
            base_config=base_config,
            output_dir=output_dir,
            config_path=child_config_path,
            resume_checkpoint=resume_checkpoint,
        )
        target_step = next_saved_step_target(
            completed_steps=progress.completed_steps,
            max_steps=max_steps,
            save_every=save_every,
            steps_per_process=steps_per_process,
        )
        previous_mtime_ns = checkpoint_path.stat().st_mtime_ns if checkpoint_path.is_file() else None
        run_count += 1
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        log_path = log_dir / f"attempt_{run_count:04d}_{stamp}.log"
        command = [*shlex.split(args.uv_command), "--config", child_config_path.as_posix(), *trainer_args]
        if args.dry_run:
            print("overnight_dry_run " + " ".join(command), flush=True)
            return 0

        print(
            "overnight_start "
            f"run={run_count} step={progress.completed_steps}/{max_steps} "
            f"target_step={target_step} resume_from={resume_checkpoint} log={log_path}",
            flush=True,
        )
        with log_path.open("a", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                command,
                cwd=Path.cwd(),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
            saved = False
            while process.poll() is None:
                if not sleep_until_stop_or_timeout(float(args.poll_seconds), should_stop):
                    terminate_process_group(process, timeout_s=float(args.terminate_timeout_seconds))
                    return 128 + int(stop_signal or signal.SIGTERM)
                current = read_progress(report_path, max_steps=max_steps)
                saved = checkpoint_saved_after(
                    output_dir=output_dir,
                    checkpoint_path=checkpoint_path,
                    completed_steps=current.completed_steps,
                    target_step=target_step,
                    previous_mtime_ns=previous_mtime_ns,
                )
                if saved:
                    print(
                        "overnight_saved "
                        f"run={run_count} step={current.completed_steps}/{max_steps} "
                        f"terminating_child_pid={process.pid}",
                        flush=True,
                    )
                    sleep_until_stop_or_timeout(float(args.post_save_grace_seconds), should_stop)
                    terminate_process_group(process, timeout_s=float(args.terminate_timeout_seconds))
                    break

            return_code = process.poll()

        current = read_progress(report_path, max_steps=max_steps)
        if not saved:
            saved = checkpoint_saved_after(
                output_dir=output_dir,
                checkpoint_path=checkpoint_path,
                completed_steps=current.completed_steps,
                target_step=target_step,
                previous_mtime_ns=previous_mtime_ns,
            )
        if saved:
            consecutive_failures = 0
            print(f"overnight_restart_ready run={run_count} step={current.completed_steps}/{max_steps}", flush=True)
            if current.is_complete and args.stop_when_complete:
                print(f"overnight_complete step={current.completed_steps}/{max_steps}", flush=True)
                return 0
            sleep_until_stop_or_timeout(float(args.restart_delay_seconds), should_stop)
            continue

        consecutive_failures += 1
        print(
            "overnight_child_failed "
            f"run={run_count} returncode={return_code} failures={consecutive_failures} "
            f"log={log_path}",
            flush=True,
        )
        if consecutive_failures >= int(args.max_consecutive_failures):
            return int(return_code or 1)
        sleep_until_stop_or_timeout(float(args.restart_delay_seconds), should_stop)


def parse_args(argv: Sequence[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Run Stage 2 mapper v2 training in disposable child processes, "
            "restarting each child after a durable checkpoint save."
        )
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH.as_posix())
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--steps-per-process", type=int, default=None)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--post-save-grace-seconds", type=float, default=3.0)
    parser.add_argument("--terminate-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--restart-delay-seconds", type=float, default=20.0)
    parser.add_argument("--max-runs", type=int, default=0, help="0 means run until max_steps is complete")
    parser.add_argument("--max-consecutive-failures", type=int, default=3)
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--stop-when-complete",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exit when report.json says the configured max_steps run is complete.",
    )
    parser.add_argument(
        "--uv-command",
        default=DEFAULT_UV_COMMAND,
        help="Command prefix used to launch the mapper v2 trainer.",
    )
    args, trainer_args = parser.parse_known_args(argv)
    return args, trainer_args


def main(argv: Sequence[str] | None = None) -> None:
    args, trainer_args = parse_args(argv)
    raise SystemExit(run_supervisor(args, trainer_args))


if __name__ == "__main__":
    main(sys.argv[1:])
