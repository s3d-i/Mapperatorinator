import tempfile
import unittest
from pathlib import Path

from train.stage1_oracle.audits.token_statistics import (
    OsuTokenStatisticsMapInput,
    TokenStatisticsMap,
    audit_osu_token_statistics,
    audit_token_statistics,
    build_token_statistics_gate_decision,
)
from train.stage1_oracle.events.canonical import CanonicalTimepoint, LaneAction


def _lane_actions(*actions: LaneAction) -> tuple[LaneAction, ...]:
    padded = list(actions)
    while len(padded) < 4:
        padded.append(LaneAction.NONE)
    return tuple(padded)


def _write_osu(path: Path, hitobject_lines: list[str], *, timing_lines: list[str] | None = None) -> None:
    timing_lines = timing_lines or []
    path.write_text(
        "\n".join(
            [
                "osu file format v14",
                "",
                "[General]",
                "AudioFilename: song.mp3",
                "Mode: 3",
                "",
                "[Difficulty]",
                "CircleSize:4",
                "",
                "[TimingPoints]",
                *timing_lines,
                "",
                "[HitObjects]",
                *hitobject_lines,
            ],
        ),
        encoding="utf-8",
    )


class TrainTokenStatisticsAuditTests(unittest.TestCase):
    def test_audit_token_statistics_reports_per_bin_token_density_and_hold_crossing(self) -> None:
        report = audit_token_statistics(
            [
                TokenStatisticsMap(
                    difficulty=2.5,
                    generation_end_ms=16000,
                    timepoints=[
                        CanonicalTimepoint(0, _lane_actions(LaneAction.TAP)),
                        CanonicalTimepoint(
                            1500,
                            _lane_actions(LaneAction.NONE, LaneAction.TAP, LaneAction.HOLD_START),
                        ),
                        CanonicalTimepoint(7990, _lane_actions(LaneAction.TAP)),
                        CanonicalTimepoint(9000, _lane_actions(LaneAction.NONE, LaneAction.NONE, LaneAction.HOLD_END)),
                    ],
                ),
            ],
        )

        bin_report = report.bins["2-3"]

        self.assertEqual(report.total_map_count, 1)
        self.assertEqual(report.audited_map_count, 1)
        self.assertEqual(bin_report.map_count, 1)
        self.assertEqual(bin_report.window_count, 2)
        self.assertEqual(bin_report.tokens_per_window.mean, 8.5)
        self.assertEqual(bin_report.tokens_per_window.p95, 14)
        self.assertEqual(bin_report.tokens_per_window.p99, 14)
        self.assertEqual(bin_report.tokens_per_window.max, 14)
        self.assertEqual(bin_report.max_decode_len, 14)
        self.assertAlmostEqual(bin_report.event_timepoints_per_second, 4 / 16)
        self.assertAlmostEqual(bin_report.note_events_per_second, 4 / 16)
        self.assertAlmostEqual(bin_report.ln_ratio, 1 / 4)
        self.assertEqual(bin_report.chord_size_counts, {1: 2, 2: 1})
        self.assertEqual(bin_report.ts_counts, {0: 1, 490: 1, 500: 1, 1000: 8})
        self.assertEqual(bin_report.empty_window_count, 0)
        self.assertEqual(bin_report.empty_window_ratio, 0.0)
        self.assertEqual(bin_report.hold_crossing_window_count, 2)
        self.assertEqual(bin_report.hold_crossing_window_ratio, 1.0)
        self.assertEqual(bin_report.max_event_delta_ms, 6490)
        self.assertEqual(bin_report.max_ts_tokens_per_delta, 7)
        self.assertEqual(bin_report.windows_requiring_multi_ts_count, 1)
        self.assertEqual(bin_report.windows_requiring_multi_ts_ratio, 0.5)

    def test_audit_token_statistics_uses_half_open_window_ownership(self) -> None:
        report = audit_token_statistics(
            [
                TokenStatisticsMap(
                    difficulty=6.0,
                    generation_end_ms=16000,
                    timepoints=[
                        CanonicalTimepoint(8000, _lane_actions(LaneAction.TAP)),
                    ],
                ),
            ],
        )

        bin_report = report.bins["5-6"]

        self.assertEqual(bin_report.map_count, 1)
        self.assertEqual(bin_report.window_count, 2)
        self.assertEqual(bin_report.tokens_per_window.mean, 2.0)
        self.assertEqual(bin_report.tokens_per_window.max, 3)
        self.assertEqual(bin_report.ts_counts, {0: 1})
        self.assertEqual(bin_report.empty_window_count, 1)
        self.assertEqual(bin_report.empty_window_ratio, 0.5)

    def test_audit_osu_token_statistics_filters_illegal_maps_before_summarizing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            valid_path = root / "valid.osu"
            missing_red_path = root / "missing_red.osu"
            negative_path = root / "negative.osu"
            unsupported_compound_path = root / "unsupported_compound.osu"
            four_state_unsupported_path = root / "end_tap.osu"
            out_of_range_path = root / "out_of_range.osu"
            _write_osu(
                valid_path,
                ["64,192,1000,128,0,1002:0:0:0:0:"],
                timing_lines=["0,500,4,2,0,80,1,0"],
            )
            _write_osu(
                missing_red_path,
                ["64,192,1000,1,0,0:0:0:0:"],
                timing_lines=["0,-100,4,2,0,80,0,0"],
            )
            _write_osu(
                negative_path,
                ["64,192,-1,1,0,0:0:0:0:"],
                timing_lines=["0,500,4,2,0,80,1,0"],
            )
            _write_osu(
                unsupported_compound_path,
                [
                    "64,192,1000,1,0,0:0:0:0:",
                    "64,192,1004,1,0,0:0:0:0:",
                ],
                timing_lines=["0,500,4,2,0,80,1,0"],
            )
            _write_osu(
                four_state_unsupported_path,
                [
                    "64,192,1000,128,0,1500:0:0:0:0:",
                    "64,192,1504,1,0,0:0:0:0:",
                ],
                timing_lines=["0,500,4,2,0,80,1,0"],
            )
            _write_osu(
                out_of_range_path,
                ["64,192,1000,1,0,0:0:0:0:"],
                timing_lines=["0,500,4,2,0,80,1,0"],
            )

            report = audit_osu_token_statistics(
                [
                    OsuTokenStatisticsMapInput(valid_path, difficulty=2.1, generation_end_ms=8000),
                    OsuTokenStatisticsMapInput(missing_red_path, difficulty=2.2, generation_end_ms=8000),
                    OsuTokenStatisticsMapInput(negative_path, difficulty=2.3, generation_end_ms=8000),
                    OsuTokenStatisticsMapInput(
                        unsupported_compound_path,
                        difficulty=2.35,
                        generation_end_ms=8000,
                    ),
                    OsuTokenStatisticsMapInput(
                        four_state_unsupported_path,
                        difficulty=2.4,
                        generation_end_ms=8000,
                    ),
                    OsuTokenStatisticsMapInput(out_of_range_path, difficulty=6.5, generation_end_ms=8000),
                ],
            )

            self.assertEqual(report.total_map_count, 6)
            self.assertEqual(report.audited_map_count, 1)
            self.assertEqual(report.out_of_range_map_count, 1)
            self.assertEqual(report.missing_red_timing_map_count, 1)
            self.assertEqual(report.negative_time_hitobject_map_count, 1)
            self.assertEqual(report.unsupported_compound_map_count, 1)
            self.assertEqual(report.unsupported_compound_event_count, 1)
            self.assertEqual(report.unsupported_compound_lane_action_count, 2)
            self.assertEqual(report.four_state_unsupported_map_count, 1)
            self.assertEqual(report.four_state_unsupported_event_count, 1)
            self.assertEqual(report.four_state_unsupported_lane_action_count, 1)
            self.assertEqual(report.filtered_event_count, 2)
            self.assertEqual(report.filtered_lane_action_count, 3)
            self.assertEqual(report.zero_length_hold_normalized_count, 1)
            self.assertEqual(report.bins["2-3"].map_count, 1)
            self.assertEqual(report.bins["2-3"].tokens_per_window.max, 3)

    def test_audit_osu_token_statistics_requires_precomputed_generation_end_to_match_audio_rule(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            osu_path = Path(tmpdir) / "chart.osu"
            _write_osu(
                osu_path,
                ["64,192,1000,1,0,0:0:0:0:"],
                timing_lines=["0,500,4,2,0,80,1,0"],
            )

            with self.assertRaisesRegex(ValueError, "generation_end_ms does not match"):
                audit_osu_token_statistics(
                    [
                        OsuTokenStatisticsMapInput(
                            osu_path,
                            difficulty=2.5,
                            generation_end_ms=8010,
                            audio_duration_ms=8000.0,
                        ),
                    ],
                )

    def test_audit_token_statistics_rejects_non_4k_empty_or_unknown_actions(self) -> None:
        cases = [
            CanonicalTimepoint(0, (LaneAction.TAP, LaneAction.NONE, LaneAction.NONE)),
            CanonicalTimepoint(0, _lane_actions()),
            CanonicalTimepoint(0, ("UNKNOWN", LaneAction.NONE, LaneAction.NONE, LaneAction.NONE)),  # type: ignore[arg-type]
        ]

        for timepoint in cases:
            with self.subTest(timepoint=timepoint):
                with self.assertRaises(ValueError):
                    audit_token_statistics(
                        [
                            TokenStatisticsMap(
                                difficulty=2.5,
                                generation_end_ms=8000,
                                timepoints=[timepoint],
                            ),
                        ],
                    )

    def test_audit_token_statistics_rejects_invalid_hold_state_sequences(self) -> None:
        with self.assertRaisesRegex(ValueError, "HOLD_END without open hold"):
            audit_token_statistics(
                [
                    TokenStatisticsMap(
                        difficulty=2.5,
                        generation_end_ms=8000,
                        timepoints=[
                            CanonicalTimepoint(1000, _lane_actions(LaneAction.HOLD_END)),
                        ],
                    ),
                ],
            )

    def test_gate_decision_records_exact_caps_and_target_length_semantics(self) -> None:
        report = audit_token_statistics(
            [
                TokenStatisticsMap(
                    difficulty=5.5,
                    generation_end_ms=8000,
                    timepoints=[
                        CanonicalTimepoint(0, _lane_actions(LaneAction.TAP)),
                    ],
                ),
            ],
        )

        decision = build_token_statistics_gate_decision(
            report,
            configured_max_decode_len=512,
            empty_window_cap_ratio=0.05,
        )

        self.assertEqual(decision.status, "PASS")
        self.assertEqual(decision.configured_max_decode_len, 512)
        self.assertEqual(decision.max_decode_len_applies_to, "target_tokens_excluding_bos_and_condition_prefix")
        self.assertEqual(decision.empty_window_cap_policy, "per_difficulty_bin_per_epoch")
        self.assertEqual(decision.empty_window_cap_ratio, 0.05)
        self.assertFalse(decision.covers_quantization_audit)
        self.assertFalse(decision.covers_window_boundary_audit)

    def test_gate_decision_fails_when_observed_target_length_exceeds_config(self) -> None:
        report = audit_token_statistics(
            [
                TokenStatisticsMap(
                    difficulty=2.5,
                    generation_end_ms=8000,
                    timepoints=[
                        CanonicalTimepoint(0, _lane_actions(LaneAction.TAP)),
                        CanonicalTimepoint(1000, _lane_actions(LaneAction.TAP)),
                    ],
                ),
            ],
        )

        decision = build_token_statistics_gate_decision(
            report,
            configured_max_decode_len=2,
            empty_window_cap_ratio=0.05,
        )

        self.assertEqual(decision.status, "FAIL")
        self.assertEqual(decision.observed_max_target_tokens, 5)


if __name__ == "__main__":
    unittest.main()
