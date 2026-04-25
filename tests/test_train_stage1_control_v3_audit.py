import unittest

import pandas as pd

from train.stage1_oracle.features.control_v3_audit import (
    FEATURE_CONFIDENCE_MAP,
    FEATURE_CONTRACT,
    LOW_SUPPORT_THRESHOLDS,
    evaluate_control_v3_audit,
    feature_audit_summary,
    high_value_confidence_audit,
    section_summaries_for_frame,
    stratified_review_queue,
)
from train.stage1_oracle.features.control_v3 import CONFIDENCE_FEATURE_NAMES, MODEL_FEATURE_NAMES


def audit_ready_section_frame(rows: int = 20) -> pd.DataFrame:
    section_df = pd.DataFrame(
        {
            "filtered_index": list(range(rows)),
            "difficulty": [4.0 for _ in range(rows)],
            "valid_fraction": [1.0 for _ in range(rows)],
        }
    )
    for name in MODEL_FEATURE_NAMES:
        section_df[f"{name}_mean"] = 0.0
    for confidence in CONFIDENCE_FEATURE_NAMES:
        section_df[f"{confidence}_p20"] = 1.0
        section_df[f"{confidence}_min"] = 1.0
    for feature, confidence in FEATURE_CONFIDENCE_MAP.items():
        section_df[f"{feature}_p95"] = 0.0
        section_df[f"{feature}_confidence_at_peak"] = 1.0
        section_df[f"{feature}_confidence_top_value_mean"] = 1.0
        section_df[f"{feature}_confidence_top_value_min"] = 1.0
        if feature in LOW_SUPPORT_THRESHOLDS:
            section_df[f"{feature}_n_eff_at_peak"] = LOW_SUPPORT_THRESHOLDS[feature] + 1.0
            section_df[f"{feature}_n_eff_top_value_mean"] = LOW_SUPPORT_THRESHOLDS[feature] + 1.0
    return section_df


class Stage1ControlV3AuditTests(unittest.TestCase):
    def test_section_summary_separates_peak_confidence_from_window_confidence(self) -> None:
        frame = pd.DataFrame(
            {
                "time_s": [0.0, 1.0, 2.0, 3.0],
                "valid_control_mask": [True, True, True, True],
                "ln_change_rate_gated": [0.1, 0.9, 0.2, 0.1],
                "ln_change_confidence": [0.9, 0.1, 0.9, 0.9],
                "ln_change_n_eff": [5.0, 1.0, 5.0, 5.0],
                "jack_streak_exposure": [0.1, 0.9, 0.2, 0.1],
                "jack_streak_confidence": [0.1, 0.9, 0.9, 0.9],
                "jack_streak_n_eff": [5.0, 5.0, 5.0, 5.0],
                "repeat_exact": [0.1, 0.9, 0.2, 0.1],
                "repeat_confidence": [0.9, 0.9, 0.9, 0.9],
                "repeat_exact_n_eff": [5.0, 1.0, 5.0, 5.0],
            }
        )

        section_df = section_summaries_for_frame(None, frame, section_s=4.0, stride_s=4.0)

        self.assertEqual(len(section_df), 1)
        row = section_df.iloc[0]
        self.assertEqual(float(row["ln_change_rate_gated_confidence_at_peak"]), 0.1)
        self.assertEqual(float(row["jack_streak_exposure_confidence_at_peak"]), 0.9)
        self.assertEqual(float(row["jack_streak_confidence_min"]), 0.1)
        self.assertEqual(float(row["repeat_exact_n_eff_at_peak"]), 1.0)

    def test_audit_splits_pointwise_window_only_and_low_support_classes(self) -> None:
        frame = pd.DataFrame(
            {
                "time_s": [0.0, 1.0, 2.0, 3.0],
                "valid_control_mask": [True, True, True, True],
                "ln_change_rate_gated": [0.1, 0.9, 0.2, 0.1],
                "ln_change_confidence": [0.9, 0.1, 0.9, 0.9],
                "ln_change_n_eff": [5.0, 1.0, 5.0, 5.0],
                "jack_streak_exposure": [0.1, 0.9, 0.2, 0.1],
                "jack_streak_confidence": [0.1, 0.9, 0.9, 0.9],
                "jack_streak_n_eff": [5.0, 5.0, 5.0, 5.0],
                "repeat_exact": [0.1, 0.9, 0.2, 0.1],
                "repeat_confidence": [0.9, 0.9, 0.9, 0.9],
                "repeat_exact_n_eff": [5.0, 1.0, 5.0, 5.0],
            }
        )
        section_df = section_summaries_for_frame(None, frame, section_s=4.0, stride_s=4.0)

        audit = high_value_confidence_audit(section_df, value_threshold=0.5, confidence_threshold=0.2)

        classes_by_feature = {
            (row.feature, row.audit_class)
            for row in audit.itertuples(index=False)
        }
        self.assertIn(("ln_change_rate_gated", "pointwise_high_value_low_confidence"), classes_by_feature)
        self.assertIn(("jack_streak_exposure", "section_high_value_low_confidence_elsewhere"), classes_by_feature)
        self.assertIn(("repeat_exact", "low_support_high_value"), classes_by_feature)

    def test_feature_summary_and_gates_use_split_audit_classes(self) -> None:
        section_df = pd.DataFrame(
            {
                "filtered_index": [1, 2],
                "difficulty": [4.0, 5.0],
                "ln_change_rate_gated_p95": [0.8, 0.2],
                "ln_change_rate_gated_peak_value": [0.9, 0.2],
                "ln_change_rate_gated_confidence_at_peak": [0.1, 0.9],
                "ln_change_rate_gated_confidence_top_value_mean": [0.1, 0.9],
                "ln_change_rate_gated_confidence_top_value_min": [0.1, 0.9],
                "ln_change_confidence_p20": [0.1, 0.9],
                "ln_change_confidence_min": [0.1, 0.9],
                "ln_change_rate_gated_n_eff_at_peak": [5.0, 5.0],
                "ln_change_rate_gated_n_eff_top_value_mean": [5.0, 5.0],
            }
        )

        summary = feature_audit_summary(section_df)
        gates = evaluate_control_v3_audit(section_df)

        self.assertIn("pointwise_high_value_low_conf_rate", summary.columns)
        self.assertIn("window_only_low_conf_rate", summary.columns)
        self.assertIn("low_support_high_value_rate", summary.columns)
        self.assertIn("pointwise_high_value_low_confidence", set(gates["gate_name"]))
        self.assertIn("low_support_high_value", set(gates["gate_name"]))
        self.assertNotIn("feature_contract_exception", set(gates["gate_name"]))
        self.assertNotIn("low_confidence_high_value", set(gates["gate_name"]))
        self.assertEqual(FEATURE_CONTRACT["ln_change_rate_raw"], "raw_value_with_side_confidence")

    def test_required_peak_diagnostics_missing_is_a_hard_gate(self) -> None:
        section_df = pd.DataFrame(
            {
                "filtered_index": [1],
                "valid_fraction": [1.0],
                "ln_change_rate_gated_p95": [0.8],
                "ln_change_confidence_p20": [1.0],
                "ln_change_confidence_min": [1.0],
            }
        )

        gates = evaluate_control_v3_audit(section_df)

        gate = gates.loc[gates["gate_name"].eq("required_feature_aligned_diagnostics_present")].iloc[0]
        self.assertFalse(bool(gate["pass"]))
        self.assertGreater(float(gate["value"]), 0.0)
        self.assertIn("ln_change_rate_gated_confidence_at_peak", str(gate["reason"]))

    def test_per_feature_gates_catch_single_feature_failures_hidden_by_global_rate(self) -> None:
        section_df = audit_ready_section_frame(rows=100)
        bad_rows = section_df.index[:15]
        section_df.loc[bad_rows, "ln_change_rate_gated_p95"] = 0.8
        section_df.loc[bad_rows, "ln_change_rate_gated_confidence_at_peak"] = 0.1
        section_df.loc[bad_rows, "ln_change_rate_gated_confidence_top_value_mean"] = 0.1
        section_df.loc[bad_rows, "ln_change_rate_gated_confidence_top_value_min"] = 0.1

        gates = evaluate_control_v3_audit(section_df)

        global_gate = gates.loc[gates["gate_name"].eq("pointwise_high_value_low_confidence")].iloc[0]
        self.assertTrue(bool(global_gate["pass"]))
        by_feature = gates.loc[
            gates["gate_name"].eq("pointwise_high_value_low_confidence_by_feature")
            & gates["feature"].eq("ln_change_rate_gated")
            & gates["metric"].eq("section_rate")
        ].iloc[0]
        self.assertFalse(bool(by_feature["pass"]))
        conditional = gates.loc[
            gates["gate_name"].eq("pointwise_high_value_low_confidence_by_feature")
            & gates["feature"].eq("ln_change_rate_gated")
            & gates["metric"].eq("given_high_value_rate")
        ].iloc[0]
        self.assertFalse(bool(conditional["pass"]))

    def test_hand_balance_signed_is_not_treated_as_positive_high_value_channel(self) -> None:
        section_df = audit_ready_section_frame(rows=1)
        section_df["hand_balance_signed_p95"] = 0.95
        section_df["hand_balance_signed_confidence_at_peak"] = 0.1
        section_df["hand_balance_signed_confidence_top_value_mean"] = 0.1
        section_df["hand_balance_signed_confidence_top_value_min"] = 0.1
        section_df["hand_confidence_p20"] = 0.1
        section_df["hand_confidence_min"] = 0.1

        audit = high_value_confidence_audit(section_df)

        self.assertNotIn("hand_balance_signed", set(audit["feature"]) if not audit.empty else set())

    def test_density_level_uses_explicit_high_value_threshold(self) -> None:
        section_df = audit_ready_section_frame(rows=1)
        section_df.loc[0, "density_level_p95"] = 1.0
        section_df.loc[0, "density_level_confidence_at_peak"] = 0.1
        section_df.loc[0, "density_level_confidence_top_value_mean"] = 0.1
        section_df.loc[0, "density_level_confidence_top_value_min"] = 0.1

        audit = high_value_confidence_audit(section_df)

        self.assertNotIn("density_level", set(audit["feature"]) if not audit.empty else set())

    def test_window_coverage_reports_rates_and_per_confidence_breakdown(self) -> None:
        section_df = audit_ready_section_frame(rows=4)
        section_df.loc[[0, 1], "density_confidence_p20"] = 0.1
        section_df.loc[[0, 1], "density_confidence_min"] = 0.1

        gates = evaluate_control_v3_audit(section_df)

        global_gate = gates.loc[gates["gate_name"].eq("window_coverage_low_confidence")].iloc[0]
        self.assertEqual(global_gate["metric"], "coverage_feature_section_rate")
        density_gate = gates.loc[
            gates["gate_name"].eq("window_coverage_low_confidence_by_confidence")
            & gates["confidence"].eq("density_confidence")
        ].iloc[0]
        self.assertEqual(density_gate["metric"], "section_rate")
        self.assertEqual(float(density_gate["value"]), 0.5)

    def test_saturation_gates_cover_every_configured_saturation_feature(self) -> None:
        section_df = audit_ready_section_frame(rows=10)
        section_df["hand_imbalance_abs_p95"] = 1.0

        gates = evaluate_control_v3_audit(section_df)

        gate = gates.loc[gates["gate_name"].eq("hand_imbalance_abs_near_high")].iloc[0]
        self.assertFalse(bool(gate["pass"]))

    def test_stratified_review_queue_samples_each_new_audit_class(self) -> None:
        section_df = pd.DataFrame(
            {
                "filtered_index": [1],
                "beatmap_id": [10],
                "difficulty": [4.0],
                "section_start_s": [0.0],
                "section_end_s": [8.0],
                "ln_change_rate_gated_p95": [0.8],
                "ln_change_rate_gated_peak_value": [0.9],
                "ln_change_rate_gated_confidence_at_peak": [0.1],
                "ln_change_rate_gated_confidence_top_value_mean": [0.1],
                "ln_change_rate_gated_confidence_top_value_min": [0.1],
                "ln_change_confidence_p20": [0.1],
                "ln_change_confidence_min": [0.1],
                "ln_change_rate_gated_n_eff_at_peak": [5.0],
                "ln_change_rate_gated_n_eff_top_value_mean": [5.0],
            }
        )

        queue = stratified_review_queue(section_df, n_per_bucket=2)

        self.assertIn("review_bucket", queue.columns)
        self.assertIn("pointwise_high_value_low_confidence", set(queue["review_bucket"]))


if __name__ == "__main__":
    unittest.main()
