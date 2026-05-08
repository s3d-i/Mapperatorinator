import unittest
from dataclasses import replace

import torch

from train.stage_2.model_mapper_v1.audits import (
    audit_grammar_replay,
    audit_ln_close_imbalance,
    audit_tokenized_windows,
    build_phase_a_gate_decision,
    build_phase_a_report,
)
from train.stage_2.model_mapper_v1.density_calibration import (
    scatter_tokenized_gold_onset_mass,
    smooth_density_mass,
)
from train.stage_2.model_mapper_v1.tokenizer import MapperTimepoint, encode_mapper_window
from train.stage_2.model_mapper_v1.vocab import LaneAction, MapperV1Vocab


def _actions(*actions: LaneAction) -> tuple[LaneAction, ...]:
    padded = list(actions)
    while len(padded) < 4:
        padded.append(LaneAction.NONE)
    return tuple(padded)


def _filter_report(*, eligible_windows: int, dropped_cross_window_ln_windows: int = 0) -> dict[str, object]:
    total_windows = eligible_windows + dropped_cross_window_ln_windows
    return {
        "num_total_windows": total_windows,
        "num_mapper_eligible_windows": eligible_windows,
        "num_dropped_short_windows": 0,
        "num_dropped_cross_window_ln_windows": dropped_cross_window_ln_windows,
        "num_dropped_unsupported_action_windows": 0,
        "drop_rate": float(dropped_cross_window_ln_windows / total_windows) if total_windows else 0.0,
        "short_drop_rate": 0.0,
        "cross_window_ln_drop_rate": float(dropped_cross_window_ln_windows / total_windows) if total_windows else 0.0,
        "unsupported_action_drop_rate": 0.0,
        "drop_rate_by_difficulty": {},
        "drop_rate_by_song": {},
    }


class MapperV1AuditTests(unittest.TestCase):
    def test_phase_a_audits_report_zero_grammar_violations_and_ln_imbalance(self) -> None:
        vocab = MapperV1Vocab()
        windows = [
            encode_mapper_window([], vocab=vocab, write_start_ms=0, write_end_ms=8000),
            encode_mapper_window(
                [
                    MapperTimepoint(1000, _actions(LaneAction.HOLD_START)),
                    MapperTimepoint(1400, _actions(LaneAction.HOLD_END)),
                ],
                vocab=vocab,
                write_start_ms=0,
                write_end_ms=8000,
            ),
        ]

        tokenizer_report = audit_tokenized_windows(
            windows,
            vocab=vocab,
            filter_report=_filter_report(eligible_windows=2),
        )
        grammar_report = audit_grammar_replay(windows, vocab=vocab)
        close_report = audit_ln_close_imbalance(windows)

        self.assertEqual(tokenizer_report.num_windows, 2)
        self.assertEqual(tokenizer_report.open_mask_nonzero_before_eos_count, 0)
        self.assertEqual(tokenizer_report.invalid_time_delta_count, 0)
        self.assertEqual(tokenizer_report.noncanonical_time_shift_count, 0)
        self.assertEqual(tokenizer_report.invalid_event_count, 0)
        self.assertGreaterEqual(tokenizer_report.p99_seq_len, tokenizer_report.p95_seq_len)
        self.assertEqual(grammar_report.violation_count, 0)
        self.assertGreater(close_report.num_open_lane_steps, 0)
        self.assertEqual(close_report.num_close_positive_steps, 1)
        self.assertGreaterEqual(close_report.pos_weight, 1.0)
        self.assertIn("ln_duration_distribution", close_report.to_dict())

    def test_phase_a_report_contains_design_gate_sections(self) -> None:
        vocab = MapperV1Vocab()
        windows = [encode_mapper_window([], vocab=vocab, write_start_ms=0, write_end_ms=8000)]

        report = build_phase_a_report(
            windows=windows,
            vocab=vocab,
            filter_report=_filter_report(eligible_windows=1),
        )

        self.assertEqual(
            set(report),
            {"window_filter", "tokenizer", "grammar", "density", "ln_close", "density_calibration", "gate_decision"},
        )
        self.assertIn("drop_rate_by_difficulty", report["window_filter"])
        self.assertIn("drop_rate_by_song", report["window_filter"])
        self.assertIn("num_dropped_short_windows", report["window_filter"])
        self.assertIn("short_drop_rate", report["window_filter"])
        self.assertIn("cross_window_ln_drop_rate", report["window_filter"])
        self.assertIn("invalid_time_delta_count", report["tokenizer"])
        self.assertIn("noncanonical_time_shift_count", report["tokenizer"])
        self.assertIn("invalid_event_count", report["tokenizer"])
        self.assertIn("gold_mass_to_density_mae", report["density"])
        self.assertEqual(report["gate_decision"]["status"], "PASS")
        self.assertEqual(report["gate_decision"]["tokenizer_status"], "PASS")
        self.assertEqual(report["gate_decision"]["grammar_status"], "PASS")

    def test_phase_a_report_requires_filter_report_to_avoid_fabricated_drop_rates(self) -> None:
        vocab = MapperV1Vocab()
        windows = [encode_mapper_window([], vocab=vocab, write_start_ms=0, write_end_ms=8000)]

        with self.assertRaisesRegex(ValueError, "filter_report is required"):
            build_phase_a_report(windows=windows, vocab=vocab, filter_report=None)

    def test_phase_a_report_carries_cross_window_ln_drop_counts_from_filter_report(self) -> None:
        vocab = MapperV1Vocab()
        windows = [encode_mapper_window([], vocab=vocab, write_start_ms=0, write_end_ms=8000)]

        report = build_phase_a_report(
            windows=windows,
            vocab=vocab,
            filter_report=_filter_report(eligible_windows=1, dropped_cross_window_ln_windows=2),
        )

        self.assertEqual(report["window_filter"]["num_total_windows"], 3)
        self.assertEqual(report["window_filter"]["num_dropped_cross_window_ln_windows"], 2)
        self.assertEqual(report["tokenizer"]["num_windows"], 3)
        self.assertEqual(report["tokenizer"]["num_dropped_cross_window_ln_windows"], 2)

    def test_tokenizer_audit_counts_invalid_and_noncanonical_tokens(self) -> None:
        vocab = MapperV1Vocab()
        base = encode_mapper_window([], vocab=vocab, write_start_ms=0, write_end_ms=8000)
        invalid_hold_end_event = vocab.encode_event(_actions(LaneAction.HOLD_END))
        malformed = replace(
            base,
            target_ids=[
                vocab.bos_id,
                vocab.time_shift_token_id(100),
                vocab.time_shift_token_id(200),
                invalid_hold_end_event,
                vocab.time_shift_token_id(4000),
                vocab.time_shift_token_id(4000),
                vocab.time_shift_token_id(10),
                vocab.eos_id,
            ],
        )

        report = audit_tokenized_windows(
            [malformed],
            vocab=vocab,
            filter_report=_filter_report(eligible_windows=1),
        )

        self.assertEqual(report.noncanonical_time_shift_count, 1)
        self.assertEqual(report.invalid_event_count, 1)
        self.assertEqual(report.invalid_time_delta_count, 1)

    def test_phase_a_gate_fails_design_hard_fail_counters(self) -> None:
        decision = build_phase_a_gate_decision(
            tokenizer={
                "open_mask_nonzero_before_eos_count": 0,
                "invalid_time_delta_count": 1,
                "invalid_event_count": 2,
                "noncanonical_time_shift_count": 3,
            },
            grammar={"violation_count": 4},
        )

        self.assertEqual(decision.status, "FAIL")
        self.assertEqual(decision.tokenizer_status, "FAIL")
        self.assertEqual(decision.grammar_status, "FAIL")
        self.assertEqual(
            decision.failure_reasons,
            [
                "invalid_time_delta_count > 0",
                "invalid_event_count > 0",
                "noncanonical_time_shift_count > 0",
                "grammar violation_count > 0",
            ],
        )

    def test_phase_a_report_fits_density_calibration_from_gold_tokens(self) -> None:
        vocab = MapperV1Vocab()
        windows = [
            encode_mapper_window(
                [
                    MapperTimepoint(1000, _actions(LaneAction.TAP)),
                    MapperTimepoint(1400, _actions(LaneAction.TAP, LaneAction.TAP)),
                ],
                vocab=vocab,
                write_start_ms=0,
                write_end_ms=8000,
            ),
            encode_mapper_window(
                [
                    MapperTimepoint(2000, _actions(LaneAction.HOLD_START)),
                    MapperTimepoint(2400, _actions(LaneAction.HOLD_END)),
                ],
                vocab=vocab,
                write_start_ms=0,
                write_end_ms=8000,
            ),
        ]
        gold_mass = torch.stack([scatter_tokenized_gold_onset_mass(window, vocab=vocab) for window in windows])
        density_target = (0.2 + 0.7 * smooth_density_mass(gold_mass)).unsqueeze(-1)
        density_confidence = torch.ones_like(density_target)

        report = build_phase_a_report(
            windows=windows,
            vocab=vocab,
            filter_report=_filter_report(eligible_windows=2),
            density_target=density_target,
            density_confidence=density_confidence,
        )

        self.assertAlmostEqual(report["density_calibration"]["scale"], 0.7, places=5)
        self.assertAlmostEqual(report["density_calibration"]["bias"], 0.2, places=5)
        self.assertLess(report["density"]["gold_mass_to_density_mae"], 1e-6)
        self.assertLess(report["density"]["density_frame_mae"], 1e-6)


if __name__ == "__main__":
    unittest.main()
