import unittest

import numpy as np
import pandas as pd

from train.stage1_oracle.features.control import HitObject
from train.stage1_oracle.features.control_v3 import (
    DEBUG_ARRAY_NAMES,
    MODEL_FEATURE_NAMES,
    FeatureConfigV3,
    extract_control_features,
)
from train.stage1_oracle.features.control_v3_artifact import (
    CONTROL_V3_DIAGNOSTIC_COLUMNS,
    CONTROL_V3_METADATA_COLUMNS,
    CONTROL_V3_FEATURE_CONTRACT_VERSION,
    CONTROL_V3_SCHEMA_VERSION,
    build_timeseries_frame,
    summarize_map_output,
)


class Stage1ControlV3ArtifactTests(unittest.TestCase):
    def test_artifact_metadata_versions_match_v3_contract(self) -> None:
        self.assertEqual(CONTROL_V3_SCHEMA_VERSION, 3)
        self.assertEqual(CONTROL_V3_FEATURE_CONTRACT_VERSION, 3)

    def test_timeseries_frame_contains_v3_model_features_and_diagnostics(self) -> None:
        row = pd.Series(
            {
                "beatmap_id": 123,
                "beatmap_set_id": 456,
                "difficulty": 4.25,
            }
        )
        out = extract_control_features(
            [HitObject(col=0, start=0.0, end=1.0)],
            cfg=FeatureConfigV3(grid_step=0.1),
            start_time=1.0,
            end_time=1.0,
            return_debug=True,
        )

        frame = build_timeseries_frame(row=row, filtered_index=7, source_index=11, out=out)

        self.assertEqual(list(frame.columns[: len(CONTROL_V3_METADATA_COLUMNS)]), CONTROL_V3_METADATA_COLUMNS)
        self.assertIn("ln_change_rate_gated", frame.columns)
        self.assertIn("ln_change_rate_raw", frame.columns)
        self.assertIn("ln_change_confidence", frame.columns)
        self.assertNotIn("ln_change_rate", frame.columns)
        self.assertNotIn("repeat_rhythm", frame.columns)
        self.assertNotIn("repeat_rhythm_n_eff", frame.columns)
        self.assertIn("valid_control_mask", frame.columns)
        self.assertEqual(frame["valid_control_mask"].dtype, np.dtype("bool"))
        for column in CONTROL_V3_DIAGNOSTIC_COLUMNS:
            self.assertIn(column, frame.columns)
        for name in set(MODEL_FEATURE_NAMES) & set(DEBUG_ARRAY_NAMES):
            np.testing.assert_allclose(frame[name].to_numpy(dtype=float), out["features"][name])
            np.testing.assert_allclose(frame[f"{name}_debug_raw"].to_numpy(dtype=float), out["debug"][name])

    def test_summary_row_records_v3_ranges_without_repeat_rhythm(self) -> None:
        row = pd.Series(
            {
                "beatmap_id": 123,
                "beatmap_set_id": 456,
                "difficulty": 4.25,
                "shard": "0",
                "beatmap_path": "set/map.osu",
                "title": "title",
                "artist": "artist",
                "version": "version",
            }
        )
        out = extract_control_features(
            [HitObject(col=0, start=0.0), HitObject(col=1, start=0.1)],
            cfg=FeatureConfigV3(grid_step=0.1),
            start_time=0.0,
            end_time=0.2,
            return_debug=True,
        )

        summary = summarize_map_output(
            row=row,
            filtered_index=7,
            source_index=11,
            hitobjects=2,
            timing_points=1,
            duration_s=0.2,
            parse_s=0.01,
            convert_s=0.02,
            feature_s=0.03,
            total_s=0.06,
            out=out,
            error_type="",
            error="",
        )

        self.assertTrue(summary["finite"])
        self.assertTrue(summary["ranges_ok"])
        self.assertIn("ln_change_rate_gated_mean", summary)
        self.assertNotIn("ln_change_rate_mean", summary)
        self.assertNotIn("repeat_rhythm_mean", summary)


if __name__ == "__main__":
    unittest.main()
