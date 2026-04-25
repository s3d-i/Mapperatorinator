import json
import unittest
from pathlib import Path


NOTEBOOK_PATH = Path("train/notebooks/control_feature_visualization_v3.ipynb")


def notebook_source() -> str:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )


class ControlFeatureVisualizationV3NotebookTests(unittest.TestCase):
    def test_notebook_imports_v3_extractor_and_split_audit_contract(self) -> None:
        source = notebook_source()

        self.assertIn("train.stage1_oracle.features.control_v3", source)
        self.assertIn("train.stage1_oracle.features.control_v3_audit", source)
        self.assertIn("MODEL_CHANNELS = list(MODEL_FEATURE_NAMES)", source)
        self.assertIn("evaluate_control_v3_audit", source)
        self.assertIn("high_value_confidence_audit", source)
        self.assertIn("feature_contract_report", source)
        self.assertIn("saturation_report", source)
        self.assertIn("stratified_review_queue", source)

    def test_notebook_uses_v3_paths_and_does_not_reference_repeat_rhythm(self) -> None:
        source = notebook_source()

        self.assertIn("control_v3_timeseries_4k_no_timing_anomalies_2to6.parquet", source)
        self.assertIn("control_v3_artifact_metadata_4k_no_timing_anomalies_2to6.json", source)
        self.assertIn("control_v3_section_audit_8s_stride4.parquet", source)
        self.assertIn("ln_change_rate_gated", source)
        self.assertIn("ln_change_rate_raw", source)
        self.assertIn("feature_contract_version", source)
        self.assertIn("error_count", source)
        self.assertNotIn("repeat_rhythm", source)
        self.assertNotIn("low_confidence_high_value", source)


if __name__ == "__main__":
    unittest.main()
