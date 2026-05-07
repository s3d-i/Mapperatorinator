import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import torch

from train.stage_2.data.control_demo_windows import (
    CONTROL_DEMO_TARGET_FEATURE_NAMES,
    collate_control_demo_context_windows,
)
from train.stage_2.features.control_v3_targets import CONFIDENCE_FEATURE_NAMES, MODEL_FEATURE_NAMES, VALUE_FEATURE_NAMES
from train.stage_2.model_control.model import ControlEncoder, ControlEncoderConfig
from train.stage_2.model_control_demo import (
    ControlDemoEncoder,
    ControlDemoEncoderConfig,
    ControlDemoModelLoss,
)
from train.stage_2.model_control_demo.model import ControlDemoEncoderOutput
from train.stage_2.training.control_demo import initialize_from_control_checkpoint, run_synthetic_smoke


class Stage2ControlDemoTests(unittest.TestCase):
    def test_demo_collate_slices_density_target_to_two_channels(self) -> None:
        batch = collate_control_demo_context_windows([_sample(target_start_frame=0, frame_count=260)])

        self.assertNotIn("control_v3_target", batch)
        self.assertNotIn("ln_change_n_eff_target", batch)
        self.assertEqual(batch["control_demo_target"].shape, (1, 100, 2))
        self.assertEqual(CONTROL_DEMO_TARGET_FEATURE_NAMES, ("density_level", "density_confidence"))
        self.assertTrue(torch.equal(batch["control_demo_target"][0, :, 0], torch.linspace(0.0, 1.0, 100)))
        self.assertTrue(torch.equal(batch["control_demo_target"][0, :, 1], torch.full((100,), 0.25)))
        self.assertEqual(batch["context_mel"].shape, (1, 600, 160))
        self.assertTrue(batch["context_padding_mask"][0, :250].all())

    def test_demo_encoder_predicts_only_density_level(self) -> None:
        model = ControlDemoEncoder(ControlDemoEncoderConfig(d_model=32, heads=4, layers=1, ffn_dim=64, dropout=0.0))
        model.eval()

        output = model(
            context_mel=torch.zeros(2, 600, 160),
            context_dense_timing_v2=torch.zeros(2, 600, 4),
            normalized_difficulty=torch.tensor([-1.0, 1.0]),
            context_padding_mask=torch.tensor([[False] * 600, [False] * 300 + [True] * 300]),
        )

        self.assertEqual(output.value_pred.shape, (2, 100, 1))
        self.assertEqual(output.control_memory.shape, (2, 600, 32))
        self.assertEqual(output.memory_padding_mask.shape, (2, 600))

    def test_demo_loss_uses_density_confidence_only_as_value_weight(self) -> None:
        output = ControlDemoEncoderOutput(
            value_pred=torch.ones(1, 100, 1, dtype=torch.float32),
            control_memory=torch.zeros(1, 600, 8),
            memory_padding_mask=torch.zeros(1, 600, dtype=torch.bool),
        )
        target = torch.zeros(1, 100, 2, dtype=torch.float32)
        valid = torch.ones(1, 100, dtype=torch.bool)
        loss_fn = ControlDemoModelLoss()

        target[:, :, 1] = 0.0
        zero_confidence = loss_fn(output, control_demo_target=target, target_valid_mask=valid)
        target[:, :, 1] = 1.0
        full_confidence = loss_fn(output, control_demo_target=target, target_valid_mask=valid)

        self.assertEqual(float(zero_confidence.total_loss.item()), 0.0)
        self.assertGreater(float(full_confidence.total_loss.item()), 0.0)
        self.assertEqual(float(full_confidence.confidence_loss.item()), 0.0)
        self.assertIn("value/density_level/weighted_smooth_l1", full_confidence.metrics)

    def test_demo_checkpoint_init_copies_backbone_and_density_head_row(self) -> None:
        torch.manual_seed(5)
        full_model = ControlEncoder(ControlEncoderConfig(d_model=32, heads=4, layers=1, ffn_dim=64, dropout=0.0))
        density_index = VALUE_FEATURE_NAMES.index("density_level")
        with torch.no_grad():
            full_model.value_head.weight[density_index].fill_(0.125)
            full_model.value_head.bias[density_index].fill_(0.75)

        demo_model = ControlDemoEncoder(ControlDemoEncoderConfig(d_model=32, heads=4, layers=1, ffn_dim=64, dropout=0.0))
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pt"
            torch.save({"model_state_dict": full_model.state_dict()}, checkpoint_path)

            report = initialize_from_control_checkpoint(demo_model, checkpoint_path)

        self.assertGreater(report["loaded_key_count"], 0)
        self.assertTrue(torch.equal(demo_model.position, full_model.position))
        self.assertTrue(torch.equal(demo_model.value_head.weight[0], full_model.value_head.weight[density_index]))
        self.assertEqual(float(demo_model.value_head.bias[0].item()), 0.75)

    def test_demo_synthetic_smoke_writes_density_only_feature_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with redirect_stdout(io.StringIO()):
                result = run_synthetic_smoke(
                    output_dir=Path(tmpdir),
                    max_steps=1,
                    save_every=1,
                    device_name="cpu",
                    seed=7,
                )

            report = json.loads(result.report_path.read_text(encoding="utf-8"))

        self.assertEqual(report["feature_names"]["value"], ["density_level"])
        self.assertEqual(report["feature_names"]["target"], ["density_level", "density_confidence"])
        self.assertEqual(report["completed_steps"], 1)


def _sample(*, target_start_frame: int, frame_count: int) -> dict[str, object]:
    target = torch.zeros(100, len(MODEL_FEATURE_NAMES), dtype=torch.float32)
    target[:, MODEL_FEATURE_NAMES.index("density_level")] = torch.linspace(0.0, 1.0, 100)
    target[:, MODEL_FEATURE_NAMES.index("density_confidence")] = 0.25
    for name in CONFIDENCE_FEATURE_NAMES:
        if name != "density_confidence":
            target[:, MODEL_FEATURE_NAMES.index(name)] = 1.0
    return {
        "full_mel": torch.zeros(frame_count, 160, dtype=torch.float32),
        "full_dense_timing_v2": torch.zeros(frame_count, 4, dtype=torch.float32),
        "control_v3_target": target,
        "ln_change_n_eff_target": torch.full((100,), 3.0, dtype=torch.float32),
        "target_valid_mask": target_start_frame + torch.arange(100) < frame_count,
        "difficulty": torch.tensor(3.0, dtype=torch.float32),
        "normalized_difficulty": torch.tensor(0.0, dtype=torch.float32),
        "target_start_frame": torch.tensor(target_start_frame, dtype=torch.long),
        "target_start_ms": torch.tensor(target_start_frame * 20, dtype=torch.long),
        "frame_count": torch.tensor(frame_count, dtype=torch.long),
        "beatmap_path": "demo.osu",
        "audio_path": "demo.mp3",
    }


if __name__ == "__main__":
    unittest.main()
