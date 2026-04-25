import unittest
from unittest.mock import patch

import numpy as np

from train.stage1_oracle.features.control import HitObject
from train.stage1_oracle.features.control_v3 import (
    CONFIDENCE_FEATURE_NAMES,
    DEBUG_ARRAY_NAMES,
    FEATURE_NAMES,
    FeatureConfigV3,
    MODEL_FEATURE_NAMES,
    VALUE_FEATURE_NAMES,
    extract_control_features,
    smooth_conf_gate,
)


class Stage1ControlV3FeatureTests(unittest.TestCase):
    def test_model_contract_drops_repeat_rhythm_and_uses_gated_ln_change(self) -> None:
        self.assertIn("ln_change_rate_gated", VALUE_FEATURE_NAMES)
        self.assertIn("ln_change_confidence", CONFIDENCE_FEATURE_NAMES)
        self.assertNotIn("ln_change_rate", MODEL_FEATURE_NAMES)
        self.assertNotIn("ln_change_rate_raw", MODEL_FEATURE_NAMES)
        self.assertNotIn("repeat_rhythm", VALUE_FEATURE_NAMES)
        self.assertNotIn("repeat_rhythm", MODEL_FEATURE_NAMES)
        self.assertNotIn("repeat_rhythm_n_eff", DEBUG_ARRAY_NAMES)
        self.assertNotIn("repeat_rhythm_top1_freq", DEBUG_ARRAY_NAMES)
        self.assertNotIn("repeat_rhythm_pattern_variety", DEBUG_ARRAY_NAMES)
        self.assertEqual(MODEL_FEATURE_NAMES, VALUE_FEATURE_NAMES + CONFIDENCE_FEATURE_NAMES)
        self.assertEqual(FEATURE_NAMES, MODEL_FEATURE_NAMES)

    def test_ln_change_rate_gated_keeps_raw_rate_as_diagnostic(self) -> None:
        cfg = FeatureConfigV3(grid_step=0.1, ln_change_L=1.0)
        hits = [HitObject(col=0, start=0.0, end=1.0)]

        out = extract_control_features(hits, cfg=cfg, start_time=1.0, end_time=1.0)

        self.assertIn("ln_change_rate_gated", out["features"])
        self.assertIn("ln_change_confidence", out["features"])
        self.assertNotIn("ln_change_rate_raw", out["features"])
        self.assertIn("ln_change_rate_raw", out["debug"])
        raw = float(out["debug"]["ln_change_rate_raw"][0])
        confidence = float(out["features"]["ln_change_confidence"][0])
        gated = float(out["features"]["ln_change_rate_gated"][0])
        self.assertGreater(raw, 0.0)
        self.assertLess(confidence, cfg.ln_change_gate_lo)
        self.assertEqual(gated, 0.0)

    def test_smooth_conf_gate_linearly_ramps_between_bounds(self) -> None:
        values = smooth_conf_gate(np.array([0.0, 0.25, 0.5, 0.75, 1.0]), lo=0.25, hi=0.75)

        np.testing.assert_allclose(values, np.array([0.0, 0.0, 0.5, 1.0, 1.0]))

    def test_extract_control_features_returns_model_matrix_without_repeat_rhythm(self) -> None:
        cfg = FeatureConfigV3(grid_step=0.1)
        hits = [
            HitObject(col=0, start=0.0),
            HitObject(col=1, start=0.1),
            HitObject(col=0, start=0.2, end=0.5),
        ]

        out = extract_control_features(hits, cfg=cfg, start_time=0.0, end_time=0.5)

        self.assertEqual(out["X"].shape, (6, len(MODEL_FEATURE_NAMES)))
        self.assertEqual(set(out["features"]), set(MODEL_FEATURE_NAMES))
        self.assertNotIn("repeat_rhythm", out["features"])
        self.assertNotIn("repeat_rhythm_n_eff", out["debug"])
        for column_index, name in enumerate(MODEL_FEATURE_NAMES):
            np.testing.assert_array_equal(out["X"][:, column_index], out["features"][name])

    def test_control_confidence_includes_jack_streak_confidence(self) -> None:
        grid = np.array([0.0, 0.1], dtype=float)
        ones = np.ones_like(grid)
        zeros = np.zeros_like(grid)

        with (
            patch(
                "train.stage1_oracle.features.control_v3.valid_control_mask",
                return_value=np.ones_like(grid, dtype=bool),
            ),
            patch(
                "train.stage1_oracle.features.control_v3.group_onsets",
                return_value=[],
            ),
            patch(
                "train.stage1_oracle.features.control_v3.density_features",
                return_value=(
                    {"density_level": zeros, "density_burst": zeros},
                    {"density_confidence": ones},
                ),
            ),
            patch(
                "train.stage1_oracle.features.control_v3.hold_and_ln_change_features",
                return_value=(
                    {"hold_occupancy": zeros, "ln_change_rate_gated": zeros},
                    {"ln_change_confidence": ones},
                ),
            ),
            patch(
                "train.stage1_oracle.features.control_v3.chord_ratio_feature",
                return_value=(zeros, {"chord_confidence": ones}),
            ),
            patch(
                "train.stage1_oracle.features.control_v3.jack_features",
                return_value=(
                    {"jack_excess": zeros, "jack_streak_exposure": zeros},
                    {"jack_confidence": ones, "jack_streak_confidence": zeros},
                ),
            ),
            patch(
                "train.stage1_oracle.features.control_v3.hand_balance_features",
                return_value=(
                    {"hand_balance_signed": zeros, "hand_imbalance_abs": zeros},
                    {"hand_confidence": ones},
                ),
            ),
            patch(
                "train.stage1_oracle.features.control_v3.repeat_features",
                return_value=(
                    {"repeat_exact": zeros, "repeat_shift": zeros, "repeat_motion": zeros},
                    {"repeat_confidence": ones},
                ),
            ),
        ):
            out = extract_control_features([], grid=grid, return_debug=True)

        np.testing.assert_allclose(out["features"]["control_confidence"], np.full_like(grid, 6.0 / 7.0))


if __name__ == "__main__":
    unittest.main()
