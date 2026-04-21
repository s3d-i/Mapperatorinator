import unittest

from train.stage1_oracle.audits.token_statistics import TokenStatisticsMap, audit_token_statistics
from train.stage1_oracle.audits.token_statistics_artifact import (
    AuditProvenance,
    build_token_statistics_artifact_payload,
)
from train.stage1_oracle.events.canonical import CanonicalTimepoint, LaneAction


def _lane_actions(*actions: LaneAction) -> tuple[LaneAction, ...]:
    padded = list(actions)
    while len(padded) < 4:
        padded.append(LaneAction.NONE)
    return tuple(padded)


class TrainTokenStatisticsArtifactTests(unittest.TestCase):
    def test_artifact_payload_contains_full_report_gate_config_and_provenance(self) -> None:
        report = audit_token_statistics(
            [
                TokenStatisticsMap(
                    difficulty=2.5,
                    generation_end_ms=8000,
                    timepoints=[
                        CanonicalTimepoint(0, _lane_actions(LaneAction.TAP)),
                        CanonicalTimepoint(1500, _lane_actions(LaneAction.TAP)),
                    ],
                ),
            ],
        )
        provenance = AuditProvenance(
            index_path="train/artifacts/indexes/beatmap_index_4k.parquet",
            index_sha256="abc123",
            dataset_root="mania-dataset",
            eligible_map_count=1,
            unique_audio_count=1,
            difficulty_source="calculate_mania_difficulty",
            difficulty_column="difficulty",
            difficulty_code_commit="deadbeef",
            code_commit="deadbeef",
            code_dirty=True,
            audit_command="uv run python -m train.stage1_oracle.audits.token_statistics_artifact",
            audio_duration_source="ffprobe",
            audio_duration_failure_count=0,
        )

        payload = build_token_statistics_artifact_payload(
            report,
            provenance=provenance,
            configured_max_decode_len=512,
            empty_window_cap_ratio=0.05,
        )

        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["provenance"]["index_sha256"], "abc123")
        self.assertEqual(payload["gate_decision"]["status"], "PASS")
        self.assertEqual(payload["gate_decision"]["empty_window_cap_by_bin"]["2-3"], 0.05)
        self.assertEqual(
            payload["gate_decision"]["max_decode_len_applies_to"],
            "target_tokens_excluding_bos_and_condition_prefix",
        )
        self.assertIn("ts_counts", payload["report"]["bins"]["2-3"])
        self.assertIn("ts_distribution", payload["report"]["bins"]["2-3"])
        self.assertEqual(payload["report"]["bins"]["2-3"]["ts_counts"], {0: 1, 1000: 1, 500: 1})


if __name__ == "__main__":
    unittest.main()
