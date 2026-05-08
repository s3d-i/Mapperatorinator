import unittest

import torch

from train.stage_2.model_mapper_v1.adapters import (
    LNCloseAdapter,
    StatePriorAdapter,
    gather_local_control,
    project_lane_action_bias_to_event_tokens,
    project_ln_close_bias_to_tokens,
)
from train.stage_2.model_mapper_v1.vocab import LaneAction, MapperV1Vocab


def _actions(*actions: LaneAction) -> tuple[LaneAction, ...]:
    padded = list(actions)
    while len(padded) < 4:
        padded.append(LaneAction.NONE)
    return tuple(padded)


class MapperV1AdapterTests(unittest.TestCase):
    def test_state_prior_projection_only_writes_event_tokens(self) -> None:
        vocab = MapperV1Vocab()
        lane_action_bias = torch.arange(16, dtype=torch.float32).view(1, 1, 4, 4)

        bias = project_lane_action_bias_to_event_tokens(lane_action_bias, vocab)

        event_id = vocab.encode_event(
            (LaneAction.TAP, LaneAction.NONE, LaneAction.HOLD_END, LaneAction.HOLD_START),
        )
        self.assertEqual(float(bias[0, 0, event_id].item()), 1.0 + 4.0 + 11.0 + 14.0)
        non_event_ids = [vocab.pad_id, vocab.bos_id, vocab.eos_id, *vocab.time_shift_token_ids]
        self.assertTrue(torch.equal(bias[0, 0, non_event_ids], torch.zeros(len(non_event_ids))))

    def test_state_prior_adapter_returns_bounded_full_vocab_bias_with_zero_non_events(self) -> None:
        torch.manual_seed(7)
        vocab = MapperV1Vocab()
        adapter = StatePriorAdapter(vocab=vocab, hidden_dim=8, lane_embedding_dim=3, age_embedding_dim=2, max_bias=1.5)

        output = adapter(
            open_mask=torch.tensor(
                [
                    [[False, False, False, False], [True, False, False, True]],
                    [[False, True, False, False], [False, False, True, False]],
                ],
            ),
            open_age_ms=torch.tensor(
                [
                    [[0, 0, 0, 0], [400, 0, 0, 1200]],
                    [[0, 80, 0, 0], [0, 0, 3000, 0]],
                ],
            ),
            remaining_ms=torch.tensor([[8000, 7600], [4000, 1000]]),
        )
        bias = output.vocab_bias

        self.assertEqual(bias.shape, (2, 2, vocab.size))
        self.assertEqual(output.lane_action_bias.shape, (2, 2, 4, 4))
        self.assertLessEqual(float(bias.abs().max().item()), 1.5)
        non_event_ids = [vocab.pad_id, vocab.bos_id, vocab.eos_id, *vocab.time_shift_token_ids]
        self.assertTrue(torch.equal(bias[:, :, non_event_ids], torch.zeros(2, 2, len(non_event_ids))))

    def test_ln_close_projection_adds_close_keep_event_biases_and_optional_skip_penalty(self) -> None:
        vocab = MapperV1Vocab()
        close_logits = torch.tensor([[[2.0, -1.0, 0.5, 3.0]]])
        open_mask = torch.tensor([[[True, False, False, False]]])

        ln_bias, time_shift_bias = project_ln_close_bias_to_tokens(
            close_logits=close_logits,
            open_mask=open_mask,
            vocab=vocab,
            close_scale=1.0,
            skip_scale=1.25,
        )

        close0 = torch.tanh(close_logits[0, 0, 0])
        close_event = vocab.encode_event(_actions(LaneAction.HOLD_END))
        keep_event = vocab.encode_event(_actions(LaneAction.NONE, LaneAction.TAP))
        closed_lane_hold_end = vocab.encode_event(_actions(LaneAction.TAP, LaneAction.HOLD_END))
        self.assertTrue(torch.allclose(ln_bias[0, 0, close_event], close0))
        self.assertTrue(torch.allclose(ln_bias[0, 0, keep_event], -0.25 * close0))
        self.assertEqual(float(ln_bias[0, 0, closed_lane_hold_end].item()), 0.0)
        self.assertTrue(torch.allclose(time_shift_bias[0, 0, list(vocab.time_shift_token_ids)], -1.25 * torch.sigmoid(close_logits[0, 0, 0])))
        self.assertEqual(float(time_shift_bias[0, 0, vocab.eos_id].item()), 0.0)

        _, no_skip_bias = project_ln_close_bias_to_tokens(
            close_logits=close_logits,
            open_mask=open_mask,
            vocab=vocab,
            close_scale=1.0,
        )
        self.assertEqual(float(no_skip_bias.abs().max().item()), 0.0)

    def test_ln_close_adapter_is_context_aware_and_skip_default_is_zero(self) -> None:
        torch.manual_seed(13)
        vocab = MapperV1Vocab()
        adapter = LNCloseAdapter(vocab=vocab, d_model=3, hidden_dim=8, lane_embedding_dim=2, age_embedding_dim=2)
        decoder_hidden = torch.randn(2, 3, 3, requires_grad=True)
        control_memory = torch.randn(2, 400, 3, requires_grad=True)
        density_teacher = torch.randn(2, 400, 1, requires_grad=True)

        output = adapter(
            decoder_hidden=decoder_hidden,
            control_memory_8s=control_memory,
            density_teacher_8s=density_teacher,
            current_ms=torch.tensor([[0, 20, 80], [40, 60, 100]]),
            write_start_ms=torch.tensor([0, 20]),
            open_mask=torch.tensor(
                [
                    [[False, False, False, False], [True, False, False, False], [True, False, True, False]],
                    [[False, True, False, False], [False, True, False, False], [False, False, False, False]],
                ],
            ),
            open_age_ms=torch.tensor(
                [
                    [[0, 0, 0, 0], [20, 0, 0, 0], [80, 0, 40, 0]],
                    [[0, 10, 0, 0], [0, 30, 0, 0], [0, 0, 0, 0]],
                ],
            ),
            remaining_ms=torch.full((2, 3), 7900),
        )

        self.assertEqual(output.close_logits.shape, (2, 3, 4))
        self.assertEqual(output.ln_close_bias.shape, (2, 3, vocab.size))
        self.assertEqual(output.time_shift_bias.shape, (2, 3, vocab.size))
        self.assertEqual(float(output.time_shift_bias.abs().max().item()), 0.0)
        non_event_ids = [vocab.pad_id, vocab.bos_id, vocab.eos_id, *vocab.time_shift_token_ids]
        self.assertEqual(float(output.ln_close_bias[:, :, non_event_ids].abs().max().item()), 0.0)

        output.close_logits.sum().backward()
        self.assertIsNotNone(decoder_hidden.grad)
        self.assertGreater(float(decoder_hidden.grad.abs().sum().item()), 0.0)
        self.assertIsNotNone(control_memory.grad)
        self.assertGreater(float(control_memory.grad.abs().sum().item()), 0.0)
        self.assertIsNotNone(density_teacher.grad)
        self.assertGreater(float(density_teacher.grad.abs().sum().item()), 0.0)

    def test_gather_local_control_uses_write_relative_20ms_frames_with_clamp(self) -> None:
        control = torch.arange(10, dtype=torch.float32).view(1, 5, 2)

        gathered = gather_local_control(
            control_memory_8s=control,
            current_ms=torch.tensor([[0, 20, 80, 100]]),
            write_start_ms=0,
        )

        self.assertTrue(torch.equal(gathered[0, 0], control[0, 0]))
        self.assertTrue(torch.equal(gathered[0, 1], control[0, 1]))
        self.assertTrue(torch.equal(gathered[0, 2], control[0, 4]))
        self.assertTrue(torch.equal(gathered[0, 3], control[0, 4]))


if __name__ == "__main__":
    unittest.main()
