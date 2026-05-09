import tempfile
import unittest
from pathlib import Path

import torch

from train.stage_2.data.compact_mapper_v1_control_teacher_cache import compact_control_teacher_cache_file
from train.stage_2.data.mapper_v1_windows import CONTROL_TEACHER_CACHE_SCHEMA_VERSION, MAPPER_DENSITY_FRAMES


class CompactMapperV1ControlTeacherCacheTests(unittest.TestCase):
    def test_compacts_sliced_storage_and_preserves_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "entry.pt"
            original_payload = _oversized_payload()
            torch.save(original_payload, cache_path)
            before_size = cache_path.stat().st_size

            result = compact_control_teacher_cache_file(cache_path, dry_run=False, verify=True)
            compacted = torch.load(cache_path, map_location="cpu", weights_only=True)

        self.assertLess(result.after_bytes, before_size)
        self.assertTrue(result.changed)
        self.assertEqual(compacted["cache_key"], original_payload["cache_key"])
        self.assertTrue(torch.equal(compacted["control_memory_8s"], original_payload["control_memory_8s"]))
        self.assertTrue(torch.equal(compacted["density_teacher_8s"], original_payload["density_teacher_8s"]))
        self.assertEqual(
            compacted["control_memory_8s"].untyped_storage().nbytes(),
            compacted["control_memory_8s"].numel() * compacted["control_memory_8s"].element_size(),
        )

    def test_dry_run_leaves_file_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "entry.pt"
            torch.save(_oversized_payload(), cache_path)
            before_bytes = cache_path.read_bytes()

            result = compact_control_teacher_cache_file(cache_path, dry_run=True, verify=True)

            self.assertEqual(cache_path.read_bytes(), before_bytes)

        self.assertFalse(result.changed)


def _oversized_payload() -> dict[str, object]:
    control_batch = torch.arange(2 * MAPPER_DENSITY_FRAMES * 384, dtype=torch.float32).reshape(
        2,
        MAPPER_DENSITY_FRAMES,
        384,
    )
    density_batch = torch.arange(2 * MAPPER_DENSITY_FRAMES, dtype=torch.float32).reshape(2, MAPPER_DENSITY_FRAMES, 1)
    return {
        "schema_version": CONTROL_TEACHER_CACHE_SCHEMA_VERSION,
        "cache_key": "abc123",
        "write_start_ms": 0,
        "write_end_ms": 8000,
        "control_dim": 384,
        "control_memory_8s": control_batch[1],
        "density_teacher_8s": density_batch[1],
    }


if __name__ == "__main__":
    unittest.main()
