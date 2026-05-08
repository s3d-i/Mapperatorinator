import math
import unittest

from train.stage_2.events.audit_high_precision_grid_adapter import (
    ACTION_HOLD_END,
    ACTION_HOLD_START,
    ACTION_TAP,
    GridAdapterConfig,
    RawEvent,
    TimingSection,
    adapt_events_to_high_precision_grid,
    canonicalize_divisors,
    decompose_time_shift,
    match_raw_time_to_grid,
    validate_timing_sections,
)


def _round_half_up(value: float) -> int:
    return int(math.floor(value + 0.5))


def _event(time_ms: int, lane: int, action: str) -> RawEvent:
    return RawEvent(raw_time_ms=time_ms, lane=lane, action=action)


def _token_tuples(result) -> list[tuple[int, int, int]]:
    return [(token.section_index, token.divisor, token.k) for token in result.tokens]


class Stage2HighPrecisionGridAdapterAuditTests(unittest.TestCase):
    def test_fractional_bpm_match_uses_unrounded_beat_length(self) -> None:
        bpm = 119.875
        beat_length_ms = 60000.0 / bpm
        raw_time_ms = _round_half_up(3 * beat_length_ms)
        sections = [TimingSection(offset_ms=0.0, beat_length_ms=beat_length_ms)]

        match = match_raw_time_to_grid(raw_time_ms, sections, divisors=[1])

        self.assertTrue(match.integer_ms_exact)
        self.assertEqual(match.section_index, 0)
        self.assertEqual(match.divisor, 1)
        self.assertEqual(match.tick_index, 3)
        self.assertAlmostEqual(match.grid_time_ms, 3 * beat_length_ms, places=9)
        self.assertLess(abs(match.raw_time_ms - match.grid_time_ms), 0.5)
        self.assertGreater(abs(match.grid_time_ms - 1500.0), 1.0)

    def test_silent_gap_keeps_high_precision_cursor_and_records_integer_ms_correction(self) -> None:
        beat_length_ms = 1000.0 / 3.0
        sections = [TimingSection(offset_ms=0.0, beat_length_ms=beat_length_ms)]
        events = [
            _event(0, 0, ACTION_TAP),
            _event(_round_half_up(11 * beat_length_ms), 1, ACTION_TAP),
            _event(_round_half_up(12 * beat_length_ms), 2, ACTION_TAP),
        ]

        result = adapt_events_to_high_precision_grid(
            events,
            sections,
            config=GridAdapterConfig(divisors=(1,), k_max=4),
        )

        self.assertTrue(result.ok)
        self.assertEqual([timepoint.tick_index for timepoint in result.timepoints], [0, 11, 12])
        self.assertEqual(
            [timepoint.raw_time_ms for timepoint in result.timepoints],
            [0, 3667, 4000],
        )
        self.assertAlmostEqual(result.timepoints[1].grid_time_ms, 11 * beat_length_ms, places=9)
        self.assertAlmostEqual(result.timepoints[1].integer_ms_correction_ms, 1.0 / 3.0, places=9)
        self.assertAlmostEqual(result.timepoints[2].integer_ms_correction_ms, 0.0, places=9)

    def test_timing_section_validation_rejects_non_strict_and_dense_staircase_knots(self) -> None:
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            validate_timing_sections(
                [
                    TimingSection(offset_ms=0.0, beat_length_ms=500.0),
                    TimingSection(offset_ms=1000.0, beat_length_ms=500.0),
                    TimingSection(offset_ms=1000.0, beat_length_ms=400.0),
                ]
            )

        with self.assertRaisesRegex(ValueError, "dense staircase"):
            validate_timing_sections(
                [
                    TimingSection(offset_ms=0.0, beat_length_ms=500.0),
                    TimingSection(offset_ms=1000.00, beat_length_ms=500.0),
                    TimingSection(offset_ms=1000.25, beat_length_ms=499.5),
                    TimingSection(offset_ms=1000.50, beat_length_ms=501.0),
                ],
                min_section_span_ms=1.0,
            )

    def test_chord_events_at_equal_raw_time_share_one_grid_timepoint(self) -> None:
        sections = [TimingSection(offset_ms=0.0, beat_length_ms=500.0)]
        result = adapt_events_to_high_precision_grid(
            [
                _event(1000, 0, ACTION_TAP),
                _event(1000, 2, ACTION_TAP),
                _event(1000, 3, ACTION_TAP),
            ],
            sections,
            config=GridAdapterConfig(divisors=(1, 2, 4), k_max=16),
        )

        self.assertTrue(result.ok)
        self.assertEqual(len(result.timepoints), 1)
        self.assertEqual(result.timepoints[0].raw_time_ms, 1000)
        self.assertAlmostEqual(result.timepoints[0].grid_time_ms, 1000.0, places=9)
        self.assertEqual(
            result.timepoints[0].lane_actions,
            (ACTION_TAP, None, ACTION_TAP, ACTION_TAP),
        )

    def test_ln_legality_accepts_closed_holds_and_rejects_invalid_lane_state(self) -> None:
        sections = [TimingSection(offset_ms=0.0, beat_length_ms=500.0)]
        config = GridAdapterConfig(divisors=(1, 2), k_max=16)

        legal = adapt_events_to_high_precision_grid(
            [
                _event(0, 0, ACTION_HOLD_START),
                _event(500, 0, ACTION_HOLD_END),
            ],
            sections,
            config=config,
        )
        self.assertTrue(legal.ok)
        self.assertEqual(
            [timepoint.lane_actions[0] for timepoint in legal.timepoints],
            [ACTION_HOLD_START, ACTION_HOLD_END],
        )

        with self.assertRaisesRegex(ValueError, "HOLD_END without open hold"):
            adapt_events_to_high_precision_grid(
                [_event(500, 0, ACTION_HOLD_END)],
                sections,
                config=config,
            )

        with self.assertRaisesRegex(ValueError, "HOLD_START while hold is open"):
            adapt_events_to_high_precision_grid(
                [
                    _event(0, 0, ACTION_HOLD_START),
                    _event(500, 0, ACTION_HOLD_START),
                ],
                sections,
                config=config,
            )

    def test_bpm_boundary_match_and_cross_section_time_shift_split(self) -> None:
        sections = [
            TimingSection(offset_ms=0.0, beat_length_ms=500.0),
            TimingSection(offset_ms=1000.0, beat_length_ms=250.0),
        ]
        config = GridAdapterConfig(divisors=(1, 2, 4), k_max=16)

        boundary_match = match_raw_time_to_grid(1000, sections, divisors=config.divisors)
        self.assertEqual(boundary_match.section_index, 1)
        self.assertEqual(boundary_match.tick_index, 0)
        self.assertAlmostEqual(boundary_match.grid_time_ms, 1000.0, places=9)

        result = decompose_time_shift(500.0, 1250.0, sections, config=config)

        self.assertTrue(result.ok)
        self.assertEqual(_token_tuples(result), [(0, 1, 1), (1, 1, 1)])

    def test_divisor_canonicalization_sorts_deduplicates_and_rejects_invalid_values(self) -> None:
        self.assertEqual(
            canonicalize_divisors([16, 4, 4, 3, 1, 12, 6, 2]),
            (1, 2, 3, 4, 6, 12, 16),
        )

        with self.assertRaisesRegex(ValueError, "positive integer"):
            canonicalize_divisors([1, 0, 4])

        with self.assertRaisesRegex(ValueError, "positive integer"):
            canonicalize_divisors([1, 2.5, 4])

    def test_long_time_shift_uses_greedy_canonical_decomposition(self) -> None:
        sections = [TimingSection(offset_ms=0.0, beat_length_ms=500.0)]

        result = decompose_time_shift(
            0.0,
            5000.0,
            sections,
            config=GridAdapterConfig(divisors=(1, 2, 4), k_max=4),
        )

        self.assertTrue(result.ok)
        self.assertEqual(_token_tuples(result), [(0, 1, 4), (0, 1, 4), (0, 1, 2)])


if __name__ == "__main__":
    unittest.main()
