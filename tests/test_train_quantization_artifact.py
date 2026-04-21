import unittest

from train.stage1_oracle.audits.quantization import QuantizationMap, audit_quantization
from train.stage1_oracle.audits.quantization_artifact import (
    QuantizationAuditProvenance,
    build_quantization_artifact_payload,
)
from train.stage1_oracle.osu.hitobjects import ManiaHitObject, ManiaHitObjectKind


class TrainQuantizationArtifactTests(unittest.TestCase):
    def test_artifact_payload_contains_report_gate_decision_and_provenance(self) -> None:
        report = audit_quantization(
            [
                QuantizationMap(
                    difficulty=2.5,
                    hitobjects=[ManiaHitObject(1005.0, 1016.0, 0, ManiaHitObjectKind.HOLD)],
                ),
            ],
        )
        provenance = QuantizationAuditProvenance(
            index_path="train/artifacts/indexes/beatmap_index_4k.parquet",
            index_sha256="abc123",
            dataset_root="mania-dataset",
            eligible_map_count=1,
            difficulty_source="calculate_mania_difficulty",
            difficulty_column="difficulty",
            code_commit="deadbeef",
            code_dirty=True,
            dirty_patch_sha256="patch123",
            dirty_patch_file_count=2,
            audit_command="uv run python -m train.stage1_oracle.audits.quantization_artifact",
        )

        payload = build_quantization_artifact_payload(
            report,
            provenance=provenance,
            expected_difficulty_source="calculate_mania_difficulty",
        )

        self.assertEqual(payload["schema_version"], 2)
        self.assertEqual(payload["audit_name"], "quantization_4k_2to6")
        self.assertEqual(payload["provenance"]["index_sha256"], "abc123")
        self.assertEqual(payload["provenance"]["dirty_patch_sha256"], "patch123")
        self.assertEqual(payload["gate_decision"]["status"], "PASS")
        self.assertEqual(payload["gate_decision"]["deterministic_quantizer"], "10ms_half_up")
        self.assertEqual(payload["gate_decision"]["reproducibility_status"], "PATCH_HASH_RECORDED")
        self.assertEqual(payload["report"]["quantization_error_ms"]["max"], 5)
        self.assertEqual(payload["report"]["zero_length_hold_normalized_count"], 0)
        self.assertIn("post_quantization_collision_lane_time_cell_count", payload["report"])
        self.assertIn("post_quantization_collision_rate", payload["report"])


if __name__ == "__main__":
    unittest.main()
