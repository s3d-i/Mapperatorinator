import unittest

from train.stage_2.model_mapper_v1.audits import (
    audit_grammar_replay,
    audit_ln_close_imbalance,
    audit_tokenized_windows,
    build_phase_a_report,
)
from train.stage_2.model_mapper_v1.tokenizer import MapperTimepoint, encode_mapper_window
from train.stage_2.model_mapper_v1.vocab import LaneAction, MapperV1Vocab


def _actions(*actions: LaneAction) -> tuple[LaneAction, ...]:
    padded = list(actions)
    while len(padded) < 4:
        padded.append(LaneAction.NONE)
    return tuple(padded)


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

        tokenizer_report = audit_tokenized_windows(windows, vocab=vocab)
        grammar_report = audit_grammar_replay(windows, vocab=vocab)
        close_report = audit_ln_close_imbalance(windows)

        self.assertEqual(tokenizer_report.num_windows, 2)
        self.assertEqual(tokenizer_report.open_mask_nonzero_before_eos_count, 0)
        self.assertGreaterEqual(tokenizer_report.p99_seq_len, tokenizer_report.p95_seq_len)
        self.assertEqual(grammar_report.violation_count, 0)
        self.assertGreater(close_report.num_open_lane_steps, 0)
        self.assertEqual(close_report.num_close_positive_steps, 1)
        self.assertGreaterEqual(close_report.pos_weight, 1.0)
        self.assertIn("ln_duration_distribution", close_report.to_dict())

    def test_phase_a_report_contains_design_gate_sections(self) -> None:
        vocab = MapperV1Vocab()
        windows = [encode_mapper_window([], vocab=vocab, write_start_ms=0, write_end_ms=8000)]

        report = build_phase_a_report(windows=windows, vocab=vocab)

        self.assertEqual(
            set(report),
            {"window_filter", "tokenizer", "grammar", "density", "ln_close", "density_calibration"},
        )
        self.assertIn("drop_rate_by_difficulty", report["window_filter"])
        self.assertIn("drop_rate_by_song", report["window_filter"])
        self.assertIn("num_dropped_short_windows", report["window_filter"])
        self.assertIn("short_drop_rate", report["window_filter"])
        self.assertIn("cross_window_ln_drop_rate", report["window_filter"])
        self.assertIn("invalid_time_delta_count", report["tokenizer"])
        self.assertIn("noncanonical_time_shift_count", report["tokenizer"])
        self.assertIn("gold_mass_to_density_mae", report["density"])


if __name__ == "__main__":
    unittest.main()
