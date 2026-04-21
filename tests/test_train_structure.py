import importlib.util
import unittest
from pathlib import Path


class TrainStructureTests(unittest.TestCase):
    def test_stage1_oracle_package_exposes_reorganized_modules(self) -> None:
        from train.stage1_oracle.audits.event_space import audit_event_space
        from train.stage1_oracle.core.difficulty import calculate_mania_difficulty
        from train.stage1_oracle.data.dataset import ManiaBeatmapDataset
        from train.stage1_oracle.data.dataset import get_default_4k_index_path
        from train.stage1_oracle.events.canonical import quantize_10ms_half_up
        from train.stage1_oracle.features.audio import load_audio_file
        from train.stage1_oracle.features.mel import pack_mel_20ms_window
        from train.stage1_oracle.models.mapper import Stage1OracleMapper
        from train.stage1_oracle.osu.hitobjects import parse_mania_hit_objects
        from train.stage1_oracle.osu.metadata import parse_osu_metadata
        from train.stage1_oracle.osu.timing import parse_red_timing_points
        from train.stage1_oracle.training.overfit_32 import run_overfit_32

        self.assertEqual(quantize_10ms_half_up(7996), 8000)
        self.assertEqual(
            get_default_4k_index_path().relative_to(Path.cwd()),
            Path("train/artifacts/indexes/beatmap_index_4k.parquet"),
        )
        self.assertTrue(callable(audit_event_space))
        self.assertTrue(callable(calculate_mania_difficulty))
        self.assertTrue(callable(load_audio_file))
        self.assertTrue(callable(pack_mel_20ms_window))
        self.assertTrue(callable(parse_mania_hit_objects))
        self.assertTrue(callable(parse_osu_metadata))
        self.assertTrue(callable(parse_red_timing_points))
        self.assertEqual(ManiaBeatmapDataset.__name__, "ManiaBeatmapDataset")
        self.assertEqual(Stage1OracleMapper.__name__, "Stage1OracleMapper")
        self.assertTrue(callable(run_overfit_32))

    def test_compatibility_and_placeholder_modules_are_not_left_behind(self) -> None:
        removed_modules = [
            "train.canonical_events",
            "train.data_utils",
            "train.dataset",
            "train.difficulty",
            "train.event_space_audit",
            "train.osu_hitobjects",
            "train.osu_metadata",
            "train.osu_timing",
            "train.stage1_oracle.data.index",
        ]

        for module_name in removed_modules:
            with self.subTest(module_name=module_name):
                self.assertIsNone(importlib.util.find_spec(module_name))


if __name__ == "__main__":
    unittest.main()
