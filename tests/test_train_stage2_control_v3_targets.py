import tempfile
import textwrap
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from train.stage1_oracle.features.control_v3 import (
    CONFIDENCE_FEATURE_NAMES as STAGE1_CONFIDENCE_FEATURE_NAMES,
)
from train.stage1_oracle.features.control_v3 import (
    MODEL_FEATURE_NAMES as STAGE1_MODEL_FEATURE_NAMES,
)
from train.stage1_oracle.features.control_v3 import (
    VALUE_FEATURE_NAMES as STAGE1_VALUE_FEATURE_NAMES,
)
from train.stage1_oracle.features.control_v3 import FeatureConfigV3
from train.stage_2.features.control_v3_targets import (
    CONFIDENCE_FEATURE_NAMES,
    LN_CHANGE_N_EFF_FEATURE_NAME,
    MODEL_FEATURE_NAMES,
    TARGET_DIM,
    TARGET_FRAME_COUNT,
    VALUE_FEATURE_NAMES,
    ControlV3TargetWindow,
    compute_control_v3_full_map_features,
    load_control_v3_timeseries_rows,
    slice_ln_change_n_eff_target_window,
    slice_control_v3_target_window,
    validate_control_v3_timeseries,
)


class Stage2ControlV3TargetTests(unittest.TestCase):
    def test_contract_reuses_stage1_control_v3_feature_names(self) -> None:
        self.assertEqual(VALUE_FEATURE_NAMES, STAGE1_VALUE_FEATURE_NAMES)
        self.assertEqual(CONFIDENCE_FEATURE_NAMES, STAGE1_CONFIDENCE_FEATURE_NAMES)
        self.assertEqual(MODEL_FEATURE_NAMES, STAGE1_MODEL_FEATURE_NAMES)
        self.assertEqual(TARGET_DIM, 20)
        self.assertEqual(TARGET_FRAME_COUNT, 100)

    def test_compute_full_map_features_from_osu_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            osu_path = Path(tmp_dir) / "map.osu"
            _write_osu(osu_path)

            frame = compute_control_v3_full_map_features(osu_path, cfg=FeatureConfigV3(grid_step=0.1))

        self.assertEqual(list(frame.columns), ["time_s", *MODEL_FEATURE_NAMES, LN_CHANGE_N_EFF_FEATURE_NAME])
        self.assertEqual(frame.shape[1], TARGET_DIM + 2)
        np.testing.assert_allclose(frame["time_s"].to_numpy(), [0.0, 0.1, 0.2, 0.3])
        self.assertEqual(frame[MODEL_FEATURE_NAMES].to_numpy().dtype, np.dtype("float32"))
        self.assertTrue(np.all(np.isfinite(frame[MODEL_FEATURE_NAMES].to_numpy())))
        self.assertEqual(frame[LN_CHANGE_N_EFF_FEATURE_NAME].to_numpy().dtype, np.dtype("float32"))
        self.assertTrue(np.all(np.isfinite(frame[LN_CHANGE_N_EFF_FEATURE_NAME].to_numpy())))

    def test_load_timeseries_rows_filters_cached_parquet_and_sorts_time(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "control_v3.parquet"
            rows = pd.concat(
                [
                    _timeseries_frame(beatmap_id=1, times=[0.1, 0.0], value_offset=10.0),
                    _timeseries_frame(beatmap_id=2, times=[0.0], value_offset=20.0),
                ],
                ignore_index=True,
            )
            rows.to_parquet(path, index=False)

            loaded = load_control_v3_timeseries_rows(path, beatmap_id=1)

        assert loaded is not None
        self.assertEqual(loaded["beatmap_id"].tolist(), [1, 1])
        np.testing.assert_allclose(loaded["time_s"].to_numpy(), [0.0, 0.1])
        np.testing.assert_allclose(loaded[MODEL_FEATURE_NAMES[0]].to_numpy(), [10.0, 10.1])

    def test_load_timeseries_rows_requires_exact_integer_selector(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "missing.parquet"

            self.assertIsNone(load_control_v3_timeseries_rows(path, beatmap_id=np.int64(1)))
            with self.assertRaisesRegex(ValueError, "integer id"):
                load_control_v3_timeseries_rows(path, beatmap_id=1.9)
            with self.assertRaisesRegex(ValueError, "non-negative"):
                load_control_v3_timeseries_rows(path, beatmap_id=-1)
            with self.assertRaisesRegex(TypeError, "bool"):
                load_control_v3_timeseries_rows(path, beatmap_id=True)

    def test_slice_target_window_resamples_to_20ms_frame_centers(self) -> None:
        rows = _timeseries_frame(beatmap_id=1, times=np.arange(0.0, 2.1, 0.1).tolist(), value_offset=0.0)

        result = slice_control_v3_target_window(
            rows,
            0.0,
            return_confidence=True,
            return_metadata=True,
        )

        self.assertIsInstance(result, ControlV3TargetWindow)
        assert isinstance(result, ControlV3TargetWindow)
        self.assertEqual(result.target.shape, (100, 20))
        self.assertEqual(result.target.dtype, np.dtype("float32"))
        self.assertAlmostEqual(float(result.target[0, 0]), 0.01, places=6)
        self.assertAlmostEqual(float(result.target[-1, 0]), 1.99, places=6)
        assert result.confidence is not None
        confidence_index = MODEL_FEATURE_NAMES.index("control_confidence")
        np.testing.assert_array_equal(result.confidence, result.target[:, confidence_index])
        self.assertTrue(np.all((result.confidence >= 0.0) & (result.confidence <= 1.0)))
        assert result.metadata is not None
        self.assertEqual(result.metadata["window_start_s"], 0.0)
        self.assertEqual(result.metadata["window_end_s"], 2.0)
        self.assertEqual(result.metadata["feature_names"], tuple(MODEL_FEATURE_NAMES))

    def test_slice_ln_change_n_eff_target_window_resamples_diagnostic_sidecar(self) -> None:
        rows = _timeseries_frame(beatmap_id=1, times=np.arange(0.0, 2.1, 0.1).tolist(), value_offset=0.0)

        target = slice_ln_change_n_eff_target_window(rows, 0.0)

        self.assertEqual(target.shape, (100,))
        self.assertEqual(target.dtype, np.dtype("float32"))
        self.assertAlmostEqual(float(target[0]), 1.01, places=6)
        self.assertAlmostEqual(float(target[-1]), 2.99, places=6)

    def test_slice_target_window_returns_array_by_default_and_zero_fills_edges(self) -> None:
        rows = _timeseries_frame(beatmap_id=1, times=[0.0], value_offset=5.0)

        target = slice_control_v3_target_window(rows, -0.02)

        self.assertEqual(target.shape, (100, 20))
        np.testing.assert_array_equal(target[0], np.zeros(TARGET_DIM, dtype=np.float32))
        np.testing.assert_array_equal(target[-1], np.zeros(TARGET_DIM, dtype=np.float32))

    def test_validation_rejects_unsorted_or_nonfinite_timeseries(self) -> None:
        unsorted = _timeseries_frame(beatmap_id=1, times=[0.1, 0.0], value_offset=0.0)
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            validate_control_v3_timeseries(unsorted)

        nonfinite = _timeseries_frame(beatmap_id=1, times=[0.0, 0.1], value_offset=0.0)
        nonfinite.loc[1, MODEL_FEATURE_NAMES[0]] = np.inf
        with self.assertRaisesRegex(ValueError, "non-finite"):
            validate_control_v3_timeseries(nonfinite)

        with self.assertRaisesRegex(ValueError, "20ms frame grid"):
            slice_control_v3_target_window(nonfinite.iloc[[0]], 0.01)

    def test_validation_rejects_non_100ms_source_grid_and_bad_confidence(self) -> None:
        irregular = _timeseries_frame(beatmap_id=1, times=[0.0, 0.05, 0.1], value_offset=0.0)
        with self.assertRaisesRegex(ValueError, "0.1s source grid"):
            validate_control_v3_timeseries(irregular)

        bad_confidence = _timeseries_frame(beatmap_id=1, times=[0.0, 0.1], value_offset=0.0)
        bad_confidence.loc[0, CONFIDENCE_FEATURE_NAMES[0]] = 2.0
        with self.assertRaisesRegex(ValueError, "confidence"):
            validate_control_v3_timeseries(bad_confidence)


def _write_osu(path: Path) -> None:
    path.write_text(
        textwrap.dedent(
            """\
            osu file format v14

            [General]
            Mode: 3

            [Difficulty]
            CircleSize: 4

            [TimingPoints]
            0,500,4,2,0,80,1,0

            [HitObjects]
            64,192,0,1,0,0:0:0:0:
            192,192,100,128,0,300:0:0:0:
            """
        ),
        encoding="utf-8",
    )


def _timeseries_frame(*, beatmap_id: int, times: list[float], value_offset: float) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "filtered_index": np.full(len(times), beatmap_id + 10, dtype=np.int32),
            "source_index": np.full(len(times), beatmap_id + 20, dtype=np.int32),
            "beatmap_id": np.full(len(times), beatmap_id, dtype=np.int64),
            "beatmap_set_id": np.full(len(times), beatmap_id + 30, dtype=np.int64),
            "difficulty": np.full(len(times), 4.0, dtype=np.float32),
            "time_s": np.asarray(times, dtype=np.float32),
        }
    )
    for column_index, name in enumerate(MODEL_FEATURE_NAMES):
        if name in CONFIDENCE_FEATURE_NAMES:
            frame[name] = np.full(len(times), 0.5, dtype=np.float32)
        else:
            frame[name] = np.asarray(times, dtype=np.float32) + np.float32(value_offset + column_index)
    frame[LN_CHANGE_N_EFF_FEATURE_NAME] = np.asarray(times, dtype=np.float32) + np.float32(1.0)
    return frame


if __name__ == "__main__":
    unittest.main()
