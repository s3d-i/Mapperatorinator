import unittest

import numpy as np
import pandas as pd

from train.stage1_oracle.features.control import HitObject
from train.stage1_oracle.features.control_v2 import FeatureConfigV2, extract_control_features
from train.stage1_oracle.features.control_v2_artifact import (
    CONTROL_V2_DIAGNOSTIC_COLUMNS,
    CONTROL_V2_METADATA_COLUMNS,
    build_timeseries_frame,
    summarize_map_output,
)


class Stage1ControlV2ArtifactTests(unittest.TestCase):
    def test_timeseries_frame_contains_metadata_features_and_numeric_diagnostics(self) -> None:
        row = pd.Series(
            {
                "beatmap_id": 123,
                "beatmap_set_id": 456,
                "difficulty": 4.25,
            }
        )
        hits = [
            HitObject(col=0, start=0.0),
            HitObject(col=1, start=0.1),
            HitObject(col=0, start=0.2, end=0.5),
        ]
        out = extract_control_features(
            hits,
            cfg=FeatureConfigV2(grid_step=0.1),
            start_time=0.0,
            end_time=0.5,
            return_debug=True,
        )

        frame = build_timeseries_frame(
            row=row,
            filtered_index=7,
            source_index=11,
            out=out,
        )

        self.assertEqual(list(frame.columns[: len(CONTROL_V2_METADATA_COLUMNS)]), CONTROL_V2_METADATA_COLUMNS)
        self.assertIn("density_level", frame.columns)
        self.assertIn("repeat_confidence", frame.columns)
        self.assertIn("control_confidence", frame.columns)
        self.assertIn("valid_control_mask", frame.columns)
        for column in CONTROL_V2_DIAGNOSTIC_COLUMNS:
            self.assertIn(column, frame.columns)
        self.assertNotIn("repeat_exact_top_token", frame.columns)
        self.assertEqual(frame["filtered_index"].dtype, np.dtype("int32"))
        self.assertEqual(frame["time_s"].dtype, np.dtype("float32"))
        self.assertEqual(frame["valid_control_mask"].dtype, np.dtype("bool"))
        self.assertEqual(len(frame), len(out["time"]))

    def test_summary_row_records_feature_ranges_and_error_state(self) -> None:
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
            cfg=FeatureConfigV2(grid_step=0.1),
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

        self.assertEqual(summary["filtered_index"], 7)
        self.assertEqual(summary["source_index"], 11)
        self.assertEqual(summary["grid_rows"], len(out["time"]))
        self.assertEqual(summary["onsets"], len(out["debug"]["onsets"]))
        self.assertTrue(summary["finite"])
        self.assertTrue(summary["ranges_ok"])
        self.assertIn("density_level_min", summary)
        self.assertIn("repeat_confidence_mean", summary)
        self.assertIn("control_confidence_mean", summary)


if __name__ == "__main__":
    unittest.main()
