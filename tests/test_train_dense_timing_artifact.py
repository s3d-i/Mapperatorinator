import unittest

from train.stage1_oracle.audits.dense_timing import DenseTimingAuditReport
from train.stage1_oracle.audits.dense_timing_artifact import (
    DenseTimingAuditProvenance,
    build_dense_timing_artifact_payload,
)


class TrainDenseTimingArtifactTests(unittest.TestCase):
    def test_artifact_payload_contains_report_gate_decision_and_provenance(self) -> None:
        report = DenseTimingAuditReport(
            total_map_count=1,
            audited_map_count=1,
            out_of_range_map_count=0,
            missing_red_timing_map_count=0,
            invalid_red_timing_map_count=0,
            invalid_red_timing_point_count=0,
            nonfinite_red_timing_point_count=0,
            nonpositive_red_timing_point_count=0,
            implausible_red_timing_point_count=0,
            negative_time_hitobject_map_count=0,
            audio_duration_failure_count=0,
            window_count=1,
            frame_count=600,
            timing_track_nan_count=0,
            timing_track_inf_count=0,
            phase_unit_norm_error_mean=0.0,
            phase_unit_norm_error_max=0.0,
            beat_pulse_nonzero_ratio=0.1,
            local_bpm_log_norm_mean=0.0,
            local_bpm_log_norm_std=1.0,
            local_bpm_log_norm_min=-1.0,
            local_bpm_log_norm_max=1.0,
            raw_bpm_min=120.0,
            raw_bpm_p01=120.0,
            raw_bpm_p50=180.0,
            raw_bpm_p99=240.0,
            raw_bpm_max=240.0,
            raw_beat_length_min=250.0,
            raw_beat_length_max=500.0,
            bpm_norm_clipped_low_count=0,
            bpm_norm_clipped_high_count=2,
            bpm_norm_clipped_ratio=2 / 600,
            bpm_log_mean=4.8,
            bpm_log_std=0.2,
            bins={},
            debug_plot_paths=[],
        )
        provenance = DenseTimingAuditProvenance(
            index_path="train/artifacts/indexes/beatmap_index_4k.parquet",
            index_sha256="abc123",
            dataset_root="mania-dataset",
            eligible_map_count=1,
            unique_audio_count=1,
            difficulty_source="calculate_mania_difficulty",
            difficulty_column="difficulty",
            code_commit="deadbeef",
            code_dirty=True,
            audit_command="uv run python -m train.stage1_oracle.audits.dense_timing_artifact",
            audio_duration_source="ffprobe",
            audio_duration_failure_count=0,
        )

        payload = build_dense_timing_artifact_payload(report, provenance=provenance)

        self.assertEqual(payload["schema_version"], 2)
        self.assertEqual(payload["audit_name"], "dense_timing_track_4k_2to6")
        self.assertEqual(payload["provenance"]["index_sha256"], "abc123")
        self.assertEqual(payload["gate_decision"]["status"], "PASS")
        self.assertEqual(payload["gate_decision"]["renderer_numerics_status"], "PASS")
        self.assertEqual(payload["gate_decision"]["timing_anomaly_status"], "PASS")
        self.assertEqual(payload["gate_decision"]["timing_track_version"], "timing_track_20ms_v1")
        self.assertEqual(payload["gate_decision"]["bpm_norm_clipped_high_count"], 2)
        self.assertEqual(payload["report"]["timing_track_nan_count"], 0)
        self.assertEqual(payload["report"]["raw_bpm_p99"], 240.0)


if __name__ == "__main__":
    unittest.main()
