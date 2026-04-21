import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from train.stage1_oracle.data.windows import OracleWindowDataset, OracleWindowFilterReport
from train.stage1_oracle.events.canonical import (
    CanonicalEventBuildResult,
    CanonicalTimepoint,
    LaneAction,
    NegativeHitObjectTimeError,
    UnsupportedCompoundLaneActionError,
)
from train.stage1_oracle.events.windowing import (
    WRITE_WINDOW_MS,
    compute_generation_end_ms,
    iter_window_specs,
    open_hold_mask_at_write_start,
    window_timepoints,
)
from train.stage1_oracle.osu.timing import (
    InvalidRedTimingError,
    MissingRedTimingError,
    RedTimingInvalidCounts,
    RedTimingPoint,
)
from train.stage1_oracle.osu.hitobjects import ManiaHitObjectKind


def _lane_actions(*actions: LaneAction) -> tuple[LaneAction, ...]:
    padded = list(actions)
    while len(padded) < 4:
        padded.append(LaneAction.NONE)
    return tuple(padded)


class Stage1WindowingTests(unittest.TestCase):
    def test_generation_end_uses_audio_ceil_and_last_event_plus_grid(self) -> None:
        self.assertEqual(
            compute_generation_end_ms(
                7991.0,
                [CanonicalTimepoint(7990, _lane_actions(LaneAction.TAP))],
            ),
            8000,
        )
        self.assertEqual(
            compute_generation_end_ms(
                7991.0,
                [CanonicalTimepoint(8000, _lane_actions(LaneAction.TAP))],
            ),
            8010,
        )

    def test_iter_window_specs_uses_fixed_input_and_short_final_write(self) -> None:
        windows = list(iter_window_specs(9000))

        self.assertEqual(len(windows), 2)
        self.assertEqual(windows[0].write_start_ms, 0)
        self.assertEqual(windows[0].write_end_ms, WRITE_WINDOW_MS)
        self.assertEqual(windows[0].input_start_ms, -2000)
        self.assertEqual(windows[0].input_end_ms, 10000)
        self.assertEqual(windows[1].write_start_ms, 8000)
        self.assertEqual(windows[1].write_end_ms, 9000)
        self.assertEqual(windows[1].write_duration_ms, 1000)
        self.assertEqual(windows[1].input_start_ms, 6000)
        self.assertEqual(windows[1].input_end_ms, 18000)

    def test_half_open_window_ownership_places_8000_in_next_window(self) -> None:
        timepoints = [
            CanonicalTimepoint(7990, _lane_actions(LaneAction.TAP)),
            CanonicalTimepoint(8000, _lane_actions(LaneAction.NONE, LaneAction.TAP)),
        ]

        self.assertEqual(window_timepoints(timepoints, write_start_ms=0, write_end_ms=8000), [timepoints[0]])
        self.assertEqual(window_timepoints(timepoints, write_start_ms=8000, write_end_ms=16000), [timepoints[1]])

    def test_open_hold_mask_excludes_events_exactly_at_write_start(self) -> None:
        timepoints = [
            CanonicalTimepoint(7000, _lane_actions(LaneAction.HOLD_START)),
            CanonicalTimepoint(8000, _lane_actions(LaneAction.NONE, LaneAction.HOLD_START)),
            CanonicalTimepoint(9000, _lane_actions(LaneAction.HOLD_END, LaneAction.HOLD_END)),
        ]

        self.assertEqual(open_hold_mask_at_write_start(timepoints, 0), 0)
        self.assertEqual(open_hold_mask_at_write_start(timepoints, 8000), 0b0001)

    def test_oracle_window_dataset_filters_unsupported_difficulty_before_record_build(self) -> None:
        index_df = pd.DataFrame(
            {
                "difficulty": [1.99, 2.0, 3.5, 6.0, 6.01],
                "shard": ["s"] * 5,
                "beatmap_path": [f"map_{index}.osu" for index in range(5)],
                "audio_path": [f"audio_{index}.mp3" for index in range(5)],
            },
        )

        with patch("train.stage1_oracle.data.windows.load_index", return_value=index_df):
            empty_report = OracleWindowFilterReport(
                source_map_count=0,
                difficulty_filtered_map_count=0,
                candidate_map_count=0,
                missing_red_timing_map_count=0,
                invalid_red_timing_map_count=0,
                negative_time_hitobject_map_count=0,
                unsupported_compound_map_count=0,
                unsupported_compound_event_count=0,
                four_state_unsupported_map_count=0,
                four_state_unsupported_lane_action_count=0,
                zero_length_hold_normalized_count=0,
                retained_map_count=0,
                generated_window_count=0,
            )
            with patch.object(OracleWindowDataset, "_build_records", return_value=([], empty_report)) as build_records:
                OracleWindowDataset(index_path="index.parquet", bpm_log_mean=5.0, bpm_log_std=0.25)

        filtered_df = build_records.call_args.args[0]
        self.assertEqual(filtered_df["difficulty"].tolist(), [2.0, 3.5, 6.0])

    def test_oracle_window_dataset_exposes_filter_report_counts(self) -> None:
        index_df = pd.DataFrame(
            {
                "difficulty": [1.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5],
                "shard": ["s"] * 7,
                "beatmap_path": [
                    "out_of_range.osu",
                    "missing.osu",
                    "invalid.osu",
                    "negative.osu",
                    "compound.osu",
                    "four_state.osu",
                    "valid.osu",
                ],
                "audio_path": [f"audio_{index}.mp3" for index in range(7)],
            },
        )

        def require_red_timing_points(beatmap_path: Path) -> list[RedTimingPoint]:
            if beatmap_path.name == "missing.osu":
                raise MissingRedTimingError("missing")
            if beatmap_path.name == "invalid.osu":
                raise InvalidRedTimingError(beatmap_path, RedTimingInvalidCounts(nonpositive=1))
            return [RedTimingPoint(offset_ms=0.0, beat_length_ms=500.0)]

        def build_canonical_quantized_events(hitobjects, *, key_count: int):
            source = hitobjects[0].source
            if source == "negative.osu":
                raise NegativeHitObjectTimeError("negative")
            if source == "four_state.osu":
                return CanonicalEventBuildResult(
                    timepoints=[
                        CanonicalTimepoint(
                            0,
                            _lane_actions(LaneAction.END_TAP, LaneAction.END_START),
                        ),
                    ],
                    zero_length_hold_normalized_count=0,
                )
            return CanonicalEventBuildResult(
                timepoints=[CanonicalTimepoint(0, _lane_actions(LaneAction.TAP))],
                zero_length_hold_normalized_count=3,
            )

        def parse_mania_hit_objects(beatmap_path: Path, expected_key_count: int):
            source = Path(beatmap_path).name
            if source == "compound.osu":
                return [
                    SimpleNamespace(
                        source=source,
                        start_time_ms=100.0,
                        end_time_ms=100.0,
                        lane=0,
                        kind=ManiaHitObjectKind.TAP,
                    ),
                    SimpleNamespace(
                        source=source,
                        start_time_ms=100.0,
                        end_time_ms=100.0,
                        lane=0,
                        kind=ManiaHitObjectKind.TAP,
                    ),
                ]
            return [
                SimpleNamespace(
                    source=source,
                    start_time_ms=0.0,
                    end_time_ms=0.0,
                    lane=0,
                    kind=ManiaHitObjectKind.TAP,
                ),
            ]

        with patch("train.stage1_oracle.data.windows.load_index", return_value=index_df):
            with patch("train.stage1_oracle.data.windows.require_red_timing_points", side_effect=require_red_timing_points):
                with patch(
                    "train.stage1_oracle.data.windows.parse_mania_hit_objects",
                    side_effect=parse_mania_hit_objects,
                ):
                    with patch(
                        "train.stage1_oracle.data.windows.build_canonical_quantized_events",
                        side_effect=build_canonical_quantized_events,
                    ):
                        with patch("train.stage1_oracle.data.windows.load_audio_file", return_value=[0.0] * 16000):
                            dataset = OracleWindowDataset(
                                index_path="index.parquet",
                                bpm_log_mean=5.0,
                                bpm_log_std=0.25,
                            )

        self.assertEqual(len(dataset.records), 1)
        self.assertEqual(dataset.filter_report.source_map_count, 7)
        self.assertEqual(dataset.filter_report.difficulty_filtered_map_count, 1)
        self.assertEqual(dataset.filter_report.missing_red_timing_map_count, 1)
        self.assertEqual(dataset.filter_report.invalid_red_timing_map_count, 1)
        self.assertEqual(dataset.filter_report.negative_time_hitobject_map_count, 1)
        self.assertEqual(dataset.filter_report.unsupported_compound_map_count, 1)
        self.assertEqual(dataset.filter_report.unsupported_compound_event_count, 1)
        self.assertEqual(dataset.filter_report.four_state_unsupported_map_count, 1)
        self.assertEqual(dataset.filter_report.four_state_unsupported_lane_action_count, 2)
        self.assertEqual(dataset.filter_report.zero_length_hold_normalized_count, 3)
        self.assertEqual(dataset.filter_report.retained_map_count, 1)
        self.assertEqual(dataset.filter_report.generated_window_count, 1)

    def test_oracle_window_dataset_selects_overfit_records_after_legality_filters(self) -> None:
        index_df = pd.DataFrame(
            {
                "difficulty": [2.5, 2.5, 2.5],
                "shard": ["s"] * 3,
                "beatmap_path": ["missing.osu", "valid_1.osu", "valid_2.osu"],
                "audio_path": ["missing.mp3", "valid_1.mp3", "valid_2.mp3"],
            },
        )

        def require_red_timing_points(beatmap_path: Path) -> list[RedTimingPoint]:
            if beatmap_path.name == "missing.osu":
                raise MissingRedTimingError("missing")
            return [RedTimingPoint(offset_ms=0.0, beat_length_ms=500.0)]

        with patch("train.stage1_oracle.data.windows.load_index", return_value=index_df):
            with patch("train.stage1_oracle.data.windows.require_red_timing_points", side_effect=require_red_timing_points):
                with patch(
                    "train.stage1_oracle.data.windows.parse_mania_hit_objects",
                    side_effect=lambda beatmap_path, expected_key_count: [
                        SimpleNamespace(
                            source=Path(beatmap_path).name,
                            start_time_ms=0.0,
                            end_time_ms=0.0,
                            lane=0,
                            kind=ManiaHitObjectKind.TAP,
                        ),
                    ],
                ):
                    with patch(
                        "train.stage1_oracle.data.windows.build_canonical_quantized_events",
                        return_value=CanonicalEventBuildResult(
                            timepoints=[CanonicalTimepoint(0, _lane_actions(LaneAction.TAP))],
                            zero_length_hold_normalized_count=0,
                        ),
                    ):
                        with patch("train.stage1_oracle.data.windows.load_audio_file", return_value=[0.0] * 16000):
                            dataset = OracleWindowDataset(
                                index_path="index.parquet",
                                bpm_log_mean=5.0,
                                bpm_log_std=0.25,
                                max_maps_per_bin=2,
                            )

        self.assertEqual(dataset.filter_report.missing_red_timing_map_count, 1)
        self.assertEqual(dataset.filter_report.retained_map_count, 2)
        self.assertEqual(
            [record.beatmap_path.name for record in dataset.records],
            ["valid_1.osu", "valid_2.osu"],
        )

    def test_oracle_window_dataset_filters_invalid_hold_transition_maps(self) -> None:
        index_df = pd.DataFrame(
            {
                "difficulty": [2.5, 2.5],
                "shard": ["s", "s"],
                "beatmap_path": ["overlap.osu", "valid.osu"],
                "audio_path": ["overlap.mp3", "valid.mp3"],
            },
        )

        def parse_mania_hit_objects(beatmap_path: Path, expected_key_count: int):
            source = Path(beatmap_path).name
            if source == "overlap.osu":
                return [
                    SimpleNamespace(
                        source=source,
                        start_time_ms=0.0,
                        end_time_ms=1000.0,
                        lane=0,
                        kind=ManiaHitObjectKind.HOLD,
                    ),
                    SimpleNamespace(
                        source=source,
                        start_time_ms=500.0,
                        end_time_ms=1500.0,
                        lane=0,
                        kind=ManiaHitObjectKind.HOLD,
                    ),
                ]
            return [
                SimpleNamespace(
                    source=source,
                    start_time_ms=0.0,
                    end_time_ms=100.0,
                    lane=0,
                    kind=ManiaHitObjectKind.HOLD,
                ),
            ]

        with patch("train.stage1_oracle.data.windows.load_index", return_value=index_df):
            with patch(
                "train.stage1_oracle.data.windows.require_red_timing_points",
                return_value=[RedTimingPoint(offset_ms=0.0, beat_length_ms=500.0)],
            ):
                with patch(
                    "train.stage1_oracle.data.windows.parse_mania_hit_objects",
                    side_effect=parse_mania_hit_objects,
                ):
                    with patch("train.stage1_oracle.data.windows.load_audio_file", return_value=[0.0] * 16000):
                        dataset = OracleWindowDataset(
                            index_path="index.parquet",
                            bpm_log_mean=5.0,
                            bpm_log_std=0.25,
                        )

        self.assertEqual(len(dataset.records), 1)
        self.assertEqual(dataset.records[0].beatmap_path.name, "valid.osu")
        self.assertEqual(dataset.filter_report.invalid_hold_transition_map_count, 1)
        self.assertEqual(dataset.filter_report.invalid_hold_transition_count, 2)


if __name__ == "__main__":
    unittest.main()
