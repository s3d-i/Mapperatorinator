import unittest

from train.stage1_oracle.events.canonical import CanonicalTimepoint, LaneAction
from train.stage1_oracle.events.stitch import DecodedWindowEvents, stitch_decoded_windows
from train.stage1_oracle.events.tokens import Stage1Vocab, decode_target_tokens
from train.stage1_oracle.osu.export import format_hitobjects


def _lane_actions(*actions: LaneAction) -> tuple[LaneAction, ...]:
    padded = list(actions)
    while len(padded) < 4:
        padded.append(LaneAction.NONE)
    return tuple(padded)


class Stage1StitchExportTests(unittest.TestCase):
    def test_stitch_offsets_relative_window_events_without_deduping(self) -> None:
        result = stitch_decoded_windows(
            [
                DecodedWindowEvents(
                    write_start_ms=0,
                    timepoints=[CanonicalTimepoint(7990, _lane_actions(LaneAction.TAP))],
                ),
                DecodedWindowEvents(
                    write_start_ms=8000,
                    timepoints=[CanonicalTimepoint(0, _lane_actions(LaneAction.NONE, LaneAction.TAP))],
                ),
            ],
        )

        self.assertEqual([timepoint.time_ms for timepoint in result.timepoints], [7990, 8000])

    def test_stitch_carries_boundary_hold_state_across_windows(self) -> None:
        result = stitch_decoded_windows(
            [
                DecodedWindowEvents(
                    write_start_ms=0,
                    timepoints=[CanonicalTimepoint(7000, _lane_actions(LaneAction.HOLD_START))],
                ),
                DecodedWindowEvents(
                    write_start_ms=8000,
                    timepoints=[CanonicalTimepoint(100, _lane_actions(LaneAction.HOLD_END))],
                ),
            ],
        )

        self.assertEqual([timepoint.time_ms for timepoint in result.timepoints], [7000, 8100])
        self.assertEqual(result.boundary_open_masks, {0: 0, 8000: 0b0001})
        self.assertEqual(result.final_open_hold_mask, 0)
        self.assertEqual(result.invalid_hold_end_count, 0)

    def test_stitch_counts_same_lane_collisions_while_hold_is_open(self) -> None:
        result = stitch_decoded_windows(
            [
                DecodedWindowEvents(
                    write_start_ms=0,
                    timepoints=[
                        CanonicalTimepoint(1000, _lane_actions(LaneAction.HOLD_START)),
                        CanonicalTimepoint(1200, _lane_actions(LaneAction.TAP)),
                        CanonicalTimepoint(1400, _lane_actions(LaneAction.HOLD_START)),
                        CanonicalTimepoint(1600, _lane_actions(LaneAction.HOLD_END)),
                    ],
                ),
            ],
        )

        self.assertEqual(result.invalid_hold_end_count, 0)
        self.assertEqual(result.same_lane_collision_count, 2)
        self.assertEqual(result.invalid_hold_transition_count, 2)

    def test_decoded_nonzero_window_stitches_once_to_absolute_times(self) -> None:
        vocab = Stage1Vocab()
        decoded = decode_target_tokens(
            [
                vocab.ts_token_id(0),
                vocab.encode_timepoint_event(_lane_actions(LaneAction.TAP)),
                vocab.ts_token_id(1000),
                vocab.ts_token_id(500),
                vocab.encode_timepoint_event(_lane_actions(LaneAction.NONE, LaneAction.TAP)),
                vocab.eos_id,
            ],
            vocab=vocab,
            write_duration_ms=4000,
        )

        result = stitch_decoded_windows(
            [
                DecodedWindowEvents(
                    write_start_ms=8000,
                    timepoints=decoded,
                ),
            ],
        )

        self.assertEqual([timepoint.time_ms for timepoint in decoded], [0, 1500])
        self.assertEqual([timepoint.time_ms for timepoint in result.timepoints], [8000, 9500])

    def test_export_formats_taps_and_holds_without_repair(self) -> None:
        lines = format_hitobjects(
            [
                CanonicalTimepoint(1000, _lane_actions(LaneAction.TAP)),
                CanonicalTimepoint(1200, _lane_actions(LaneAction.NONE, LaneAction.HOLD_START)),
                CanonicalTimepoint(1800, _lane_actions(LaneAction.NONE, LaneAction.HOLD_END)),
            ],
        )

        self.assertEqual(
            lines,
            [
                "64,192,1000,1,0,0:0:0:0:",
                "192,192,1200,128,0,1800:0:0:0:0:",
            ],
        )

    def test_export_rejects_raw_invalid_hold_end(self) -> None:
        with self.assertRaisesRegex(ValueError, "HOLD_END without open hold"):
            format_hitobjects(
                [
                    CanonicalTimepoint(1000, _lane_actions(LaneAction.HOLD_END)),
                ],
            )


if __name__ == "__main__":
    unittest.main()
