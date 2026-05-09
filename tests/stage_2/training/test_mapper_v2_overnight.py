import json
import tempfile
import unittest
from pathlib import Path

import yaml

from train.stage_2.training.mapper_v2_overnight import (
    checkpoint_saved_after,
    existing_resume_checkpoint,
    next_saved_step_target,
    read_progress,
    write_child_config,
)


class MapperV2OvernightSupervisorTests(unittest.TestCase):
    def test_next_saved_step_target_uses_next_checkpoint_boundary(self) -> None:
        self.assertEqual(
            next_saved_step_target(completed_steps=0, max_steps=12000, save_every=250, steps_per_process=250),
            250,
        )
        self.assertEqual(
            next_saved_step_target(completed_steps=1, max_steps=12000, save_every=250, steps_per_process=250),
            500,
        )
        self.assertEqual(
            next_saved_step_target(completed_steps=11800, max_steps=12000, save_every=250, steps_per_process=250),
            12000,
        )

    def test_write_child_config_removes_missing_resume_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "child.yaml"

            write_child_config(
                base_config={
                    "output_dir": "old",
                    "resume_from": "missing.pt",
                    "max_steps": 10,
                },
                output_dir=Path(temp_dir) / "run",
                config_path=path,
                resume_checkpoint=None,
            )

            loaded = yaml.safe_load(path.read_text(encoding="utf-8"))

        self.assertNotIn("resume_from", loaded)
        self.assertEqual(loaded["output_dir"], f"{temp_dir}/run")
        self.assertEqual(loaded["max_steps"], 10)

    def test_write_child_config_sets_existing_resume_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "child.yaml"
            checkpoint = Path(temp_dir) / "checkpoint.pt"
            checkpoint.write_bytes(b"checkpoint")

            write_child_config(
                base_config={"output_dir": "old"},
                output_dir=Path(temp_dir) / "run",
                config_path=path,
                resume_checkpoint=checkpoint,
            )

            loaded = yaml.safe_load(path.read_text(encoding="utf-8"))

        self.assertEqual(loaded["resume_from"], checkpoint.as_posix())

    def test_existing_resume_checkpoint_prefers_latest_output_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "run"
            output_dir.mkdir()
            latest = output_dir / "checkpoint.pt"
            latest.write_bytes(b"latest")
            configured = Path(temp_dir) / "configured.pt"
            configured.write_bytes(b"configured")

            result = existing_resume_checkpoint({"resume_from": configured.as_posix()}, output_dir)

        self.assertEqual(result, latest)

    def test_read_progress_treats_completed_max_steps_as_complete(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            report = Path(temp_dir) / "report.json"
            report.write_text(json.dumps({"completed_steps": 10, "is_complete": False}), encoding="utf-8")

            progress = read_progress(report, max_steps=10)

        self.assertEqual(progress.completed_steps, 10)
        self.assertTrue(progress.is_complete)

    def test_checkpoint_saved_after_accepts_archive_for_reported_target_step(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            checkpoint = output_dir / "checkpoint.pt"
            checkpoint.write_bytes(b"latest")
            archive = output_dir / "checkpoints" / "checkpoint_step_000250.pt"
            archive.parent.mkdir()
            archive.write_bytes(b"archive")

            saved = checkpoint_saved_after(
                output_dir=output_dir,
                checkpoint_path=checkpoint,
                completed_steps=250,
                target_step=250,
                previous_mtime_ns=checkpoint.stat().st_mtime_ns,
            )

        self.assertTrue(saved)


if __name__ == "__main__":
    unittest.main()
