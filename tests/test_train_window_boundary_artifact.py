import unittest

from train.stage1_oracle.audits.window_boundary import WindowBoundaryMap, audit_window_boundaries
from train.stage1_oracle.audits.window_boundary_artifact import (
    WindowBoundaryAuditProvenance,
    build_window_boundary_artifact_payload,
)
from train.stage1_oracle.events.canonical import CanonicalTimepoint, LaneAction


def _lane_actions(*actions: LaneAction) -> tuple[LaneAction, ...]:
    padded = list(actions)
    while len(padded) < 4:
        padded.append(LaneAction.NONE)
    return tuple(padded)


class TrainWindowBoundaryArtifactTests(unittest.TestCase):
    def test_artifact_payload_contains_report_gate_decision_and_provenance(self) -> None:
        report = audit_window_boundaries(
            [
                WindowBoundaryMap(
                    difficulty=2.5,
                    generation_end_ms=16000,
                    timepoints=[
                        CanonicalTimepoint(8000, _lane_actions(LaneAction.TAP)),
                    ],
                ),
            ],
        )
        provenance = WindowBoundaryAuditProvenance(
            index_path="train/artifacts/indexes/beatmap_index_4k.parquet",
            index_sha256="abc123",
            dataset_root="mania-dataset",
            eligible_map_count=1,
            unique_audio_count=1,
            difficulty_source="calculate_mania_difficulty",
            difficulty_column="difficulty",
            code_commit="deadbeef",
            code_dirty=True,
            audit_command="uv run python -m train.stage1_oracle.audits.window_boundary_artifact",
            audio_duration_source="ffprobe",
            audio_duration_failure_count=0,
        )

        payload = build_window_boundary_artifact_payload(report, provenance=provenance)

        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["audit_name"], "window_boundary_4k_2to6")
        self.assertEqual(payload["provenance"]["index_sha256"], "abc123")
        self.assertEqual(payload["gate_decision"]["status"], "PASS")
        self.assertEqual(payload["gate_decision"]["window_ownership"], "half_open_write_intervals")
        self.assertEqual(payload["report"]["boundary_event_count"], 1)
        self.assertEqual(payload["report"]["stitch_duplicate_timepoint_count"], 0)


if __name__ == "__main__":
    unittest.main()
