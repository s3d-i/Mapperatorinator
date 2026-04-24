import unittest

import numpy as np

from train.stage1_oracle.features.control import HitObject
from train.stage1_oracle.features.control_v2 import (
    CONFIDENCE_FEATURE_NAMES,
    DEBUG_ARRAY_NAMES,
    FEATURE_NAMES,
    FeatureConfigV2,
    MODEL_FEATURE_NAMES,
    VALUE_FEATURE_NAMES,
    extract_control_features,
    extract_raw_for_norm_fit,
)


class Stage1ControlV2FeatureTests(unittest.TestCase):
    def test_extract_control_features_returns_named_model_track_and_audit_debug(self) -> None:
        cfg = FeatureConfigV2(grid_step=0.1)
        hits = [
            HitObject(col=0, start=0.0),
            HitObject(col=1, start=0.001),
            HitObject(col=0, start=0.10),
            HitObject(col=2, start=0.20, end=0.55),
        ]

        out = extract_control_features(hits, cfg=cfg, start_time=0.0, end_time=0.4)

        self.assertEqual(out["feature_names"], FEATURE_NAMES)
        self.assertEqual(out["model_feature_names"], MODEL_FEATURE_NAMES)
        self.assertEqual(
            VALUE_FEATURE_NAMES,
            [
                "density_level",
                "density_burst",
                "hold_occupancy",
                "ln_change_rate",
                "chord_ratio",
                "jack_excess",
                "jack_streak_exposure",
                "hand_balance_signed",
                "hand_imbalance_abs",
                "repeat_exact",
                "repeat_shift",
                "repeat_motion",
                "repeat_rhythm",
            ],
        )
        self.assertEqual(
            CONFIDENCE_FEATURE_NAMES,
            [
                "density_confidence",
                "ln_change_confidence",
                "chord_confidence",
                "jack_confidence",
                "jack_streak_confidence",
                "hand_confidence",
                "repeat_confidence",
                "control_confidence",
            ],
        )
        self.assertEqual(MODEL_FEATURE_NAMES, VALUE_FEATURE_NAMES + CONFIDENCE_FEATURE_NAMES)
        self.assertEqual(FEATURE_NAMES, MODEL_FEATURE_NAMES)
        self.assertIn("control_confidence", DEBUG_ARRAY_NAMES)
        self.assertIn("jack_streak_confidence", DEBUG_ARRAY_NAMES)
        self.assertEqual(out["time"].shape, (5,))
        self.assertEqual(out["X"].shape, (5, len(MODEL_FEATURE_NAMES)))
        self.assertEqual(out["X_model"].shape, out["X"].shape)
        np.testing.assert_array_equal(out["X_model"], out["X"])
        self.assertEqual(set(out["features"]), set(MODEL_FEATURE_NAMES))
        for column_index, name in enumerate(MODEL_FEATURE_NAMES):
            np.testing.assert_array_equal(out["X"][:, column_index], out["features"][name])

        self.assertTrue(np.all(np.isfinite(out["X"])))
        self.assertIn("density_raw_short", out["debug"])
        self.assertIn("chord_num", out["debug"])
        self.assertIn("jack_expected_null", out["debug"])
        self.assertIn("repeat_exact_top1_freq", out["debug"])
        self.assertIn("valid_control_mask", out["debug"])
        self.assertIn("control_confidence", out["debug"])
        self.assertIn("jack_streak_confidence", out["features"])
        self.assertIn("control_confidence", out["features"])

    def test_control_confidence_is_available_without_debug(self) -> None:
        cfg = FeatureConfigV2(grid_step=0.1)
        grid = np.array([-0.2, 0.0, 0.1, 0.2, 0.6], dtype=float)
        hits = [
            HitObject(col=0, start=0.0),
            HitObject(col=0, start=0.1),
            HitObject(col=0, start=0.2),
        ]

        out = extract_control_features(hits, cfg=cfg, grid=grid, return_debug=False)

        self.assertEqual(out["debug"], {})
        self.assertIn("control_confidence", out["features"])
        self.assertIn("jack_streak_confidence", out["features"])
        self.assertEqual(float(out["features"]["control_confidence"][0]), 0.0)
        self.assertEqual(float(out["features"]["control_confidence"][-1]), 0.0)
        control_index = MODEL_FEATURE_NAMES.index("control_confidence")
        np.testing.assert_array_equal(
            out["X_model"][:, control_index],
            out["features"]["control_confidence"],
        )

    def test_hand_balance_uses_raw_ratio_then_single_confidence_gate(self) -> None:
        cfg = FeatureConfigV2(grid_step=0.1, hand_L=1.0, hand_prior=1.0)
        hits = [HitObject(col=0, start=i * 0.1) for i in range(10)]

        out = extract_control_features(hits, cfg=cfg, start_time=0.5, end_time=0.5)

        left = float(out["debug"]["hand_left_load"][0])
        right = float(out["debug"]["hand_right_load"][0])
        total = left + right
        raw_balance = (left - right) / total
        confidence = total / (total + cfg.hand_prior)
        self.assertGreater(total, 0.0)
        self.assertAlmostEqual(float(out["debug"]["hand_balance_raw"][0]), raw_balance)
        self.assertAlmostEqual(float(out["features"]["hand_confidence"][0]), confidence)
        self.assertAlmostEqual(
            float(out["features"]["hand_balance_signed"][0]),
            confidence * raw_balance,
        )

    def test_low_support_chord_and_hand_ratios_are_gated_to_neutral(self) -> None:
        cfg = FeatureConfigV2(grid_step=0.1, chord_L=2.0, hand_L=2.0)
        hits = [
            HitObject(col=0, start=0.0),
            HitObject(col=1, start=0.001),
            HitObject(col=2, start=0.001),
            HitObject(col=3, start=0.001),
        ]

        out = extract_control_features(hits, cfg=cfg, start_time=0.0, end_time=0.0)

        self.assertGreater(float(out["debug"]["chord_den"][0]), 0.0)
        self.assertLess(float(out["debug"]["chord_n_eff"][0]), cfg.ratio_n_eff_min)
        self.assertEqual(float(out["features"]["chord_ratio"][0]), 0.0)
        self.assertEqual(float(out["features"]["hand_balance_signed"][0]), 0.0)
        self.assertEqual(float(out["features"]["hand_imbalance_abs"][0]), 0.0)

    def test_edge_default_neutralizes_values_before_first_and_after_last_object(self) -> None:
        cfg = FeatureConfigV2(grid_step=0.1)
        grid = np.array([0.0, 4.9, 5.0, 5.1, 6.0], dtype=float)
        hits = [HitObject(col=0, start=5.0)]

        out = extract_control_features(hits, cfg=cfg, grid=grid)

        for index in [0, 1, 3, 4]:
            self.assertFalse(bool(out["debug"]["valid_control_mask"][index]))
            for name in FEATURE_NAMES:
                self.assertEqual(float(out["features"][name][index]), 0.0)

    def test_ln_change_rate_weights_simultaneous_transition_magnitude(self) -> None:
        cfg = FeatureConfigV2(grid_step=0.1, ln_change_L=1.0)
        one_release = [
            HitObject(col=0, start=0.0, end=1.0),
        ]
        three_release = [
            HitObject(col=0, start=0.0, end=1.0),
            HitObject(col=1, start=0.0, end=1.0),
            HitObject(col=2, start=0.0, end=1.0),
        ]

        one_out = extract_control_features(one_release, cfg=cfg, start_time=1.0, end_time=1.0)
        three_out = extract_control_features(three_release, cfg=cfg, start_time=1.0, end_time=1.0)

        self.assertGreater(
            float(three_out["features"]["ln_change_rate"][0]),
            float(one_out["features"]["ln_change_rate"][0]) * 1.5,
        )

    def test_jack_excess_rises_for_same_column_repeats_more_than_alternating_stream(self) -> None:
        cfg = FeatureConfigV2(grid_step=0.05, jack_L=0.5)
        same_col = [HitObject(col=0, start=i * 0.1) for i in range(12)]
        alternating = [HitObject(col=i % 4, start=i * 0.1) for i in range(12)]

        same_out = extract_control_features(same_col, cfg=cfg, start_time=0.0, end_time=1.1)
        alt_out = extract_control_features(alternating, cfg=cfg, start_time=0.0, end_time=1.1)

        self.assertGreater(
            float(np.max(same_out["features"]["jack_excess"])),
            float(np.max(alt_out["features"]["jack_excess"])) + 0.2,
        )
        np.testing.assert_allclose(
            same_out["features"]["density_level"],
            alt_out["features"]["density_level"],
            atol=1e-9,
        )
        np.testing.assert_allclose(
            same_out["features"]["chord_ratio"],
            alt_out["features"]["chord_ratio"],
            atol=1e-9,
        )

    def test_shifted_stair_pattern_raises_shift_and_motion_repeat_without_exact_recurrence(self) -> None:
        cfg = FeatureConfigV2(grid_step=0.1, repeat_L=2.0)
        masks = [
            (0, 1),
            (1, 2),
            (2, 3),
            (0, 1),
            (1, 2),
            (2, 3),
            (0, 1),
            (1, 2),
            (2, 3),
        ]
        hits = [
            HitObject(col=col, start=i * 0.5)
            for i, cols in enumerate(masks)
            for col in cols
        ]

        out = extract_control_features(hits, cfg=cfg, start_time=0.0, end_time=4.0)

        repeat_exact = float(np.max(out["features"]["repeat_exact"]))
        repeat_shift = float(np.max(out["features"]["repeat_shift"]))
        repeat_motion = float(np.max(out["features"]["repeat_motion"]))

        self.assertGreater(repeat_shift, 0.25)
        self.assertGreater(repeat_motion, 0.10)
        self.assertGreater(repeat_shift, repeat_exact + 0.15)

    def test_repeat_motion_tokens_keep_previous_and_current_hand_side(self) -> None:
        cfg = FeatureConfigV2(grid_step=0.1, repeat_L=2.0)
        masks = [
            (0, 1),
            (1, 2),
            (2, 3),
            (0, 1),
            (1, 2),
            (2, 3),
        ]
        hits = [
            HitObject(col=col, start=i * 0.5)
            for i, cols in enumerate(masks)
            for col in cols
        ]

        out = extract_control_features(hits, cfg=cfg, start_time=0.0, end_time=2.5)
        top_tokens = [token for token in out["debug"]["repeat_motion_top_token"] if token is not None]

        self.assertTrue(any(len(token) == 6 for token in top_tokens))
        self.assertTrue(any(token[1:3] == (1, 0) for token in top_tokens))
        self.assertTrue(any(token[1:3] == (0, -1) for token in top_tokens))

    def test_normalizer_fit_uses_valid_confident_values_for_sparse_channels(self) -> None:
        cfg = FeatureConfigV2(grid_step=0.1, jack_L=0.5)
        sparse_jack = [
            HitObject(col=0, start=5.0 + i * 0.1)
            for i in range(12)
        ]
        normalizers = extract_raw_for_norm_fit(
            [sparse_jack],
            cfg=cfg,
            confidence_threshold=0.3,
        )

        self.assertGreater(normalizers["jack_excess"].hi, normalizers["jack_excess"].lo)
        self.assertGreater(normalizers["jack_excess"].lo, 0.0)


if __name__ == "__main__":
    unittest.main()
