import math
import unittest

import numpy as np

from train.stage1_oracle.features.control import (
    FEATURE_NAMES,
    FeatureConfig,
    HitObject,
    Robust01,
    concentration_score_on_grid,
    event_kernel_sum,
    extract_control_features,
    extract_raw_for_norm_fit,
    group_onsets,
    make_grid,
    make_piecewise_beat_length_fn,
    mania_hit_objects_to_control_hits,
    red_timing_points_to_beat_length_fn,
    validate_control_hits,
)
from train.stage1_oracle.osu.hitobjects import ManiaHitObject, ManiaHitObjectKind
from train.stage1_oracle.osu.timing import RedTimingPoint


class Stage1ControlFeatureTests(unittest.TestCase):
    def test_extract_control_features_returns_named_six_column_track(self) -> None:
        cfg = FeatureConfig(grid_step=0.05)
        hits = [
            HitObject(col=0, start=0.0),
            HitObject(col=1, start=0.001),
            HitObject(col=0, start=0.10),
            HitObject(col=2, start=0.20, end=0.55),
        ]

        out = extract_control_features(hits, cfg=cfg, start_time=0.0, end_time=0.25)

        self.assertEqual(out["feature_names"], FEATURE_NAMES)
        self.assertEqual(
            FEATURE_NAMES,
            [
                "density_env",
                "hold_occupancy",
                "chord_rate",
                "jack_risk",
                "hand_balance_ema",
                "repeat_risk",
            ],
        )
        self.assertEqual(out["time"].shape, (6,))
        self.assertEqual(out["X"].shape, (6, 6))
        self.assertEqual(set(out["features"]), set(FEATURE_NAMES))
        for column_index, name in enumerate(FEATURE_NAMES):
            np.testing.assert_array_equal(out["X"][:, column_index], out["features"][name])
        self.assertTrue(np.all(np.isfinite(out["X"])))
        self.assertTrue(
            np.all((0.0 <= out["features"]["hold_occupancy"]) & (out["features"]["hold_occupancy"] <= 1.0))
        )
        self.assertTrue(np.all((0.0 <= out["features"]["chord_rate"]) & (out["features"]["chord_rate"] <= 1.0)))
        self.assertTrue(
            np.all((-1.0 <= out["features"]["hand_balance_ema"]) & (out["features"]["hand_balance_ema"] <= 1.0))
        )
        self.assertTrue(np.all((0.0 <= out["features"]["repeat_risk"]) & (out["features"]["repeat_risk"] <= 1.0)))
        self.assertIn("onsets", out["debug"])
        self.assertIn("repeat_exact", out["debug"])
        self.assertNotIn("repeat_exact", out["features"])

    def test_group_onsets_uses_seconds_and_beat_scaled_epsilon(self) -> None:
        cfg = FeatureConfig(onset_eps_min=0.002, onset_eps_beat_div=768.0)
        beat_length_at = make_piecewise_beat_length_fn([(0.0, 1.536)], default_beat_len=0.5)
        hits = [
            HitObject(col=0, start=1.0000),
            HitObject(col=2, start=1.0015),
            HitObject(col=3, start=1.0031),
        ]

        onsets = group_onsets(hits, beat_length_at, cfg)

        self.assertEqual(len(onsets), 2)
        self.assertTrue(math.isclose(onsets[0].t, 1.00075))
        self.assertEqual(onsets[0].mask, 0b0101)
        self.assertEqual(onsets[0].chord_size, 2)
        self.assertEqual(onsets[1].mask, 0b1000)
        self.assertEqual(onsets[1].chord_size, 1)

    def test_hold_occupancy_integrates_long_notes_per_column(self) -> None:
        cfg = FeatureConfig(key_count=4, hold_L=1.0, min_ln_len=0.03)
        grid = np.array([-2.0, 0.5, 5.0, 10.5, 12.0], dtype=float)
        hits = [
            HitObject(col=0, start=0.0, end=10.0),
            HitObject(col=1, start=0.0, end=10.0),
            HitObject(col=2, start=0.0, end=10.0),
            HitObject(col=3, start=0.0, end=10.0),
            HitObject(col=0, start=3.0, end=3.01),
        ]

        out = extract_control_features(hits, cfg=cfg, grid=grid)

        hold = out["features"]["hold_occupancy"]
        self.assertEqual(hold[0], 0.0)
        self.assertGreater(hold[1], 0.0)
        self.assertEqual(hold[2], 1.0)
        self.assertGreater(hold[3], 0.0)
        self.assertEqual(hold[4], 0.0)

    def test_repeat_risk_detects_repeated_transition_tokens_and_keeps_components_debug_only(self) -> None:
        cfg = FeatureConfig(grid_step=0.25, repeat_L=2.0)
        hits = [
            HitObject(col=0, start=0.0),
            HitObject(col=1, start=0.5),
            HitObject(col=0, start=1.0),
            HitObject(col=1, start=1.5),
            HitObject(col=0, start=2.0),
            HitObject(col=1, start=2.5),
            HitObject(col=0, start=3.0),
            HitObject(col=1, start=3.5),
        ]

        out = extract_control_features(hits, cfg=cfg, start_time=0.0, end_time=4.0)

        self.assertGreater(float(np.max(out["features"]["repeat_risk"])), 0.0)
        self.assertGreater(float(np.max(out["debug"]["repeat_exact"])), 0.0)
        self.assertGreater(float(np.max(out["debug"]["repeat_shift"])), 0.0)
        self.assertGreater(float(np.max(out["debug"]["repeat_motion"])), 0.0)
        self.assertEqual(out["X"].shape[1], len(FEATURE_NAMES))

    def test_normalizer_fit_and_transform_only_unbounded_features(self) -> None:
        cfg = FeatureConfig(grid_step=0.1)
        maps = [
            [
                HitObject(col=0, start=0.0),
                HitObject(col=0, start=0.1),
                HitObject(col=1, start=0.2),
            ],
            [
                HitObject(col=0, start=0.0),
                HitObject(col=1, start=0.0),
                HitObject(col=2, start=0.0),
            ],
        ]
        normalizers = extract_raw_for_norm_fit(maps, cfg=cfg)

        self.assertEqual(set(normalizers), {"density_env", "jack_risk"})
        self.assertIsInstance(normalizers["density_env"], Robust01)
        out = extract_control_features(maps[0], cfg=cfg, normalizers=normalizers)

        self.assertTrue(np.all((0.0 <= out["features"]["density_env"]) & (out["features"]["density_env"] <= 1.0)))
        self.assertTrue(np.all((0.0 <= out["features"]["jack_risk"]) & (out["features"]["jack_risk"] <= 1.0)))
        self.assertTrue(
            np.all((-1.0 <= out["features"]["hand_balance_ema"]) & (out["features"]["hand_balance_ema"] <= 1.0))
        )

    def test_mania_hit_object_adapter_converts_milliseconds_to_seconds(self) -> None:
        mania_hits = [
            ManiaHitObject(
                start_time_ms=1500.0,
                end_time_ms=1500.0,
                lane=0,
                kind=ManiaHitObjectKind.TAP,
            ),
            ManiaHitObject(
                start_time_ms=2000.0,
                end_time_ms=2600.0,
                lane=3,
                kind=ManiaHitObjectKind.HOLD,
            ),
        ]

        hits = mania_hit_objects_to_control_hits(mania_hits)

        self.assertEqual(
            hits,
            [
                HitObject(col=0, start=1.5, end=1.5),
                HitObject(col=3, start=2.0, end=2.6),
            ],
        )

    def test_mania_hit_object_adapter_supports_lane_base_and_missing_end_time(self) -> None:
        mania_hits = [
            ManiaHitObject(
                start_time_ms=1500.0,
                end_time_ms=None,
                lane=1,
                kind=ManiaHitObjectKind.TAP,
            ),
            ManiaHitObject(
                start_time_ms=2000.0,
                end_time_ms=2600.0,
                lane=4,
                kind=ManiaHitObjectKind.HOLD,
            ),
        ]

        hits = mania_hit_objects_to_control_hits(mania_hits, lane_base=1)

        self.assertEqual(
            hits,
            [
                HitObject(col=0, start=1.5, end=1.5),
                HitObject(col=3, start=2.0, end=2.6),
            ],
        )

    def test_validate_control_hits_rejects_invalid_columns_and_backwards_times(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid columns.*min=-1, max=4, count=2"):
            validate_control_hits(
                [
                    HitObject(col=-1, start=0.0),
                    HitObject(col=4, start=1.0),
                ],
                key_count=4,
            )
        with self.assertRaisesRegex(ValueError, "end < start.*count=1"):
            validate_control_hits([HitObject(col=0, start=2.0, end=1.0)], key_count=4)

    def test_extract_control_features_validates_hits_before_grouping(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid columns"):
            extract_control_features(
                [HitObject(col=4, start=0.0)],
                cfg=FeatureConfig(key_count=4),
                start_time=0.0,
                end_time=1.0,
            )

    def test_default_grid_starts_at_song_zero_when_hits_start_later(self) -> None:
        grid = make_grid([HitObject(col=0, start=1.0)], step=0.5)

        np.testing.assert_array_equal(grid, np.array([0.0, 0.5, 1.0], dtype=float))

    def test_return_debug_false_returns_empty_debug_payload(self) -> None:
        out = extract_control_features(
            [HitObject(col=0, start=0.0), HitObject(col=1, start=0.5)],
            start_time=0.0,
            end_time=1.0,
            return_debug=False,
        )

        self.assertEqual(out["debug"], {})

    def test_kernel_helpers_reject_mismatched_inputs(self) -> None:
        grid = np.array([0.0], dtype=float)

        with self.assertRaisesRegex(ValueError, "event_times and weights length mismatch: 2 vs 1"):
            event_kernel_sum(grid, [0.0, 1.0], [1.0], L=1.0)
        with self.assertRaisesRegex(ValueError, "token_times and tokens length mismatch: 2 vs 1"):
            concentration_score_on_grid(grid, np.array([0.0, 1.0], dtype=float), [(1,)], L=2.0)

    def test_red_timing_adapter_converts_milliseconds_to_seconds(self) -> None:
        timing_points = [
            RedTimingPoint(offset_ms=1000.0, beat_length_ms=500.0),
            RedTimingPoint(offset_ms=2500.0, beat_length_ms=250.0),
        ]

        beat_length_at = red_timing_points_to_beat_length_fn(timing_points, default_beat_len=0.75)

        self.assertEqual(beat_length_at(0.5), 0.75)
        self.assertEqual(beat_length_at(1.0), 0.5)
        self.assertEqual(beat_length_at(2.49), 0.5)
        self.assertEqual(beat_length_at(2.5), 0.25)

    def test_red_timing_adapter_ignores_non_positive_beat_lengths(self) -> None:
        timing_points = [
            RedTimingPoint(offset_ms=1000.0, beat_length_ms=-500.0),
            RedTimingPoint(offset_ms=2500.0, beat_length_ms=250.0),
        ]

        beat_length_at = red_timing_points_to_beat_length_fn(timing_points, default_beat_len=0.75)

        self.assertEqual(beat_length_at(1.5), 0.75)
        self.assertEqual(beat_length_at(2.5), 0.25)


if __name__ == "__main__":
    unittest.main()
