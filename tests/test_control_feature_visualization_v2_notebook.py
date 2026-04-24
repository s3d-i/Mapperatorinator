import json
import unittest
from pathlib import Path


NOTEBOOK_PATH = Path("train/notebooks/control_feature_visualization_v2.ipynb")


def notebook_source() -> str:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )


class ControlFeatureVisualizationV2NotebookTests(unittest.TestCase):
    def test_notebook_imports_extractor_channel_contract(self) -> None:
        source = notebook_source()

        self.assertIn("VALUE_FEATURE_NAMES", source)
        self.assertIn("CONFIDENCE_FEATURE_NAMES", source)
        self.assertIn("MODEL_FEATURE_NAMES", source)
        self.assertIn("DEBUG_ARRAY_NAMES", source)
        self.assertIn("MODEL_CHANNELS = list(MODEL_FEATURE_NAMES)", source)
        self.assertIn("DEBUG_ARRAY_COLUMNS = list(DEBUG_ARRAY_NAMES)", source)

    def test_notebook_keeps_debug_arrays_separate_from_model_columns(self) -> None:
        source = notebook_source()

        self.assertIn('frame[f"{name}_debug_raw"] = array', source)
        self.assertIn("frame[name] = array", source)

    def test_notebook_has_gate_ready_perturbation_and_dataset_checks(self) -> None:
        source = notebook_source()

        self.assertIn("convert_singles_to_doubles", source)
        self.assertIn("single_stream_singles_to_doubles", source)
        self.assertIn("normal_jack_time_stretch_1p1", source)
        self.assertIn("normal_jack_column_shuffle", source)
        self.assertIn("double_stair_event_order_randomize", source)
        self.assertIn("def evaluate_control_v2_audit(", source)

    def test_notebook_can_build_section_audit_from_timeseries_parquet(self) -> None:
        source = notebook_source()

        self.assertIn("TIMESERIES_PATH = REPO_ROOT / \"train/artifacts/features/control_v2_timeseries_4k_no_timing_anomalies_2to6.parquet\"", source)
        self.assertIn("def build_section_audit_from_timeseries_parquet(", source)
        self.assertIn("timeseries_path: Path = TIMESERIES_PATH", source)
        self.assertIn("pd.read_parquet(timeseries_path)", source)
        self.assertIn("section_summaries_for_frame(", source)


if __name__ == "__main__":
    unittest.main()
