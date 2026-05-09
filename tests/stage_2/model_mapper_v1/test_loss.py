import math
import unittest

import torch
import torch.nn.functional as F

from train.stage_2.model_mapper_v1.grammar import build_grammar_mask
from train.stage_2.model_mapper_v1.replay import empty_ln_carry_state
from train.stage_2.model_mapper_v1.loss import (
    adapter_regularization,
    close_pos_weight,
    density_auxiliary_loss,
    expected_density_from_logits,
    ln_close_aux_loss,
    token_cross_entropy,
)
from train.stage_2.model_mapper_v1.vocab import LaneAction, MapperV1Vocab


class MapperV1LossTests(unittest.TestCase):
    def test_token_cross_entropy_applies_grammar_mask_before_softmax(self) -> None:
        vocab = MapperV1Vocab()
        logits = torch.full((1, 1, vocab.size), -10.0)
        logits[0, 0, vocab.pad_id] = 100.0
        logits[0, 0, vocab.eos_id] = 5.0
        grammar_mask = build_grammar_mask(
            current_ms=torch.tensor([[8000]]),
            open_mask=torch.zeros((1, 1, 4), dtype=torch.bool),
            open_start_ms=torch.full((1, 1, 4), -1, dtype=torch.long),
            open_age_ms=torch.zeros((1, 1, 4), dtype=torch.long),
            write_start_ms=torch.tensor([0]),
            write_end_ms=torch.tensor([8000]),
            ln_carry_in=empty_ln_carry_state(0),
            ln_carry_out=empty_ln_carry_state(8000),
            is_full_chart_start=torch.tensor([True]),
            is_full_chart_end=torch.tensor([True]),
            positions=torch.tensor([[3]]),
            vocab=vocab,
        )

        loss = token_cross_entropy(
            logits,
            torch.tensor([[vocab.eos_id]]),
            grammar_mask=grammar_mask,
            pad_id=vocab.pad_id,
        )

        self.assertTrue(torch.isfinite(loss))
        self.assertLess(float(loss.item()), 1e-6)

    def test_token_cross_entropy_ignores_pad_targets(self) -> None:
        vocab = MapperV1Vocab()
        logits = torch.zeros((1, 2, vocab.size))
        logits[0, 0, vocab.pad_id] = -100.0
        logits[0, 1, vocab.eos_id] = 3.0
        target = torch.tensor([[vocab.pad_id, vocab.eos_id]])

        loss = token_cross_entropy(logits, target, pad_id=vocab.pad_id)
        expected = F.cross_entropy(logits[:, 1].reshape(1, -1), target[:, 1].reshape(-1))

        self.assertTrue(torch.allclose(loss, expected))

    def test_token_cross_entropy_skips_invalid_rows_before_softmax(self) -> None:
        vocab = MapperV1Vocab()
        logits = torch.zeros((1, 2, vocab.size), requires_grad=True)
        target = torch.tensor([[vocab.eos_id, vocab.pad_id]])
        target_mask = torch.tensor([[True, False]])
        grammar_mask = torch.zeros_like(logits)
        grammar_mask[0, 1] = -torch.inf

        loss = token_cross_entropy(
            logits,
            target,
            pad_id=vocab.pad_id,
            target_mask=target_mask,
            grammar_mask=grammar_mask,
        )
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertEqual(float(logits.grad[0, 1].abs().sum().item()), 0.0)

    def test_ln_close_aux_loss_uses_auto_class_balanced_bce(self) -> None:
        close_logits = torch.zeros((1, 1, 4), dtype=torch.float32)
        labels = torch.tensor([[[True, False, False, False]]])
        mask = torch.tensor([[[True, True, True, False]]])

        self.assertEqual(float(close_pos_weight(labels=labels, mask=mask).item()), 2.0)
        loss = ln_close_aux_loss(
            close_logits=close_logits,
            labels=labels,
            mask=mask,
            focal=False,
        )

        expected = torch.tensor((4.0 * math.log(2.0)) / 3.0)
        self.assertTrue(torch.allclose(loss, expected))

    def test_ln_close_focal_loss_downweights_confident_correct_examples(self) -> None:
        close_logits = torch.tensor([[[5.0, -5.0, 0.0, 0.0]]])
        labels = torch.tensor([[[True, False, False, False]]])
        mask = torch.tensor([[[True, True, False, False]]])

        weighted_bce = ln_close_aux_loss(
            close_logits=close_logits,
            labels=labels,
            mask=mask,
            pos_weight=1.0,
            focal=False,
        )
        focal = ln_close_aux_loss(
            close_logits=close_logits,
            labels=labels,
            mask=mask,
            pos_weight=1.0,
            gamma=1.5,
            focal=True,
        )

        self.assertLess(float(focal.item()), float(weighted_bce.item()))

    def test_ln_close_aux_loss_masks_closed_lanes_and_empty_mask_returns_zero(self) -> None:
        close_logits = torch.tensor([[[0.0, 10.0, -10.0, 5.0]]], requires_grad=True)
        labels = torch.tensor([[[True, False, False, True]]])
        mask = torch.tensor([[[True, False, False, False]]])

        loss = ln_close_aux_loss(
            close_logits=close_logits,
            labels=labels,
            mask=mask,
            pos_weight=1.0,
            focal=False,
        )
        expected = F.binary_cross_entropy_with_logits(close_logits[:, :, :1], labels[:, :, :1].float())
        self.assertTrue(torch.allclose(loss, expected))

        empty = ln_close_aux_loss(
            close_logits=close_logits,
            labels=labels,
            mask=torch.zeros_like(mask),
        )
        self.assertEqual(float(empty.item()), 0.0)

    def test_adapter_regularization_sums_mean_squared_biases(self) -> None:
        first = torch.tensor([[1.0, 3.0]])
        second = torch.tensor([[2.0, 4.0]])

        reg = adapter_regularization(first, second)

        self.assertEqual(float(reg.item()), 5.0 + 10.0)

    def test_expected_density_scatters_grammar_masked_onset_mass_to_20ms_frame(self) -> None:
        vocab = MapperV1Vocab()
        event_id = vocab.encode_event((LaneAction.TAP, LaneAction.NONE, LaneAction.NONE, LaneAction.NONE))
        logits = torch.full((1, 1, vocab.size), -20.0)
        logits[0, 0, event_id] = 20.0
        logits[0, 0, vocab.eos_id] = 30.0
        grammar_mask = build_grammar_mask(
            current_ms=torch.tensor([[40]]),
            open_mask=torch.zeros((1, 1, 4), dtype=torch.bool),
            open_start_ms=torch.full((1, 1, 4), -1, dtype=torch.long),
            open_age_ms=torch.zeros((1, 1, 4), dtype=torch.long),
            write_start_ms=torch.tensor([0]),
            write_end_ms=torch.tensor([8000]),
            ln_carry_in=empty_ln_carry_state(0),
            ln_carry_out=empty_ln_carry_state(8000),
            is_full_chart_start=torch.tensor([True]),
            is_full_chart_end=torch.tensor([True]),
            positions=torch.tensor([[0]]),
            vocab=vocab,
        )

        density = expected_density_from_logits(
            logits_final=logits + grammar_mask,
            current_ms=torch.tensor([[40]]),
            write_start_ms=torch.tensor([0]),
            target_mask=torch.tensor([[True]]),
            vocab=vocab,
        )

        self.assertGreater(float(density[0, 2, 0].item()), 0.99)
        self.assertAlmostEqual(float(density.sum().item()), float(density[0, 2, 0].item()), places=5)

    def test_density_auxiliary_uses_confidence_and_backpropagates_through_logits(self) -> None:
        vocab = MapperV1Vocab()
        logits = torch.zeros((1, 1, vocab.size), requires_grad=True)
        target = torch.zeros((1, 400, 1), dtype=torch.float32)
        confidence = torch.zeros((1, 400, 1), dtype=torch.float32)
        confidence[0, 0, 0] = 1.0

        loss = density_auxiliary_loss(
            logits_final=logits,
            current_ms=torch.tensor([[0]]),
            write_start_ms=torch.tensor([0]),
            target=target,
            confidence=confidence,
            target_mask=torch.tensor([[True]]),
            vocab=vocab,
        )
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())

        zero_confidence = density_auxiliary_loss(
            logits_final=logits.detach(),
            current_ms=torch.tensor([[0]]),
            write_start_ms=torch.tensor([0]),
            target=target,
            confidence=torch.zeros_like(confidence),
            target_mask=torch.tensor([[True]]),
            vocab=vocab,
        )
        self.assertEqual(float(zero_confidence.item()), 0.0)


if __name__ == "__main__":
    unittest.main()
