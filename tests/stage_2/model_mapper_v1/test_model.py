import unittest

import torch
from torch import nn

from train.stage_2.data.mapper_v1_windows import collate_mapper_v1_windows
from train.stage_2.model_mapper_v1.model import MapperV1Config, MapperV1Model
from train.stage_2.model_mapper_v1.tokenizer import MapperTimepoint, encode_mapper_window
from train.stage_2.model_mapper_v1.vocab import LaneAction, MapperV1Vocab


class MapperV1ModelTests(unittest.TestCase):
    def test_forward_uses_teacher_forced_alignment(self) -> None:
        torch.manual_seed(11)
        vocab = MapperV1Vocab()
        model = MapperV1Model(_small_config(vocab), vocab=vocab)
        batch = _empty_batch(vocab)

        output = model(**batch)

        self.assertEqual(output.logits_final.shape, (1, 3, vocab.size))
        self.assertTrue(torch.equal(output.decoder_input_tokens, batch["target_tokens"][:, :-1]))
        self.assertTrue(torch.equal(output.loss_target_tokens, batch["target_tokens"][:, 1:]))
        self.assertTrue(torch.equal(output.state_current_ms, batch["teacher_current_ms"][:, :-1]))
        self.assertTrue(torch.equal(output.state_open_mask, batch["teacher_open_mask"][:, :-1]))

    def test_grammar_mask_is_final_authority(self) -> None:
        torch.manual_seed(13)
        vocab = MapperV1Vocab()
        model = MapperV1Model(_small_config(vocab), vocab=vocab)
        with torch.no_grad():
            model.output_head.weight.zero_()
            model.output_head.bias.fill_(1_000_000.0)
        batch = _empty_batch(vocab)

        output = model(**batch)

        self.assertTrue(torch.isneginf(output.logits_final[0, 0, vocab.pad_id]))
        self.assertTrue(torch.isneginf(output.logits_final[0, 0, vocab.bos_id]))
        self.assertTrue(torch.isneginf(output.logits_final[0, 0, vocab.eos_id]))
        self.assertTrue(torch.isneginf(output.logits_final[torch.isneginf(output.grammar_mask)]).all().item())

    def test_skip_scale_defaults_to_zero_time_shift_bias(self) -> None:
        torch.manual_seed(17)
        vocab = MapperV1Vocab()
        config = _small_config(vocab)
        self.assertEqual(config.skip_scale, 0.0)
        model = MapperV1Model(config, vocab=vocab)
        batch = _empty_batch(vocab)

        output = model(**batch)

        self.assertTrue(torch.equal(output.time_shift_bias, torch.zeros_like(output.time_shift_bias)))

    def test_control_encoder_is_frozen_and_kept_eval(self) -> None:
        vocab = MapperV1Vocab()
        control_encoder = _TinyControlEncoder()

        model = MapperV1Model(_small_config(vocab), vocab=vocab, control_encoder=control_encoder)
        model.train()

        self.assertFalse(control_encoder.weight.requires_grad)
        self.assertFalse(control_encoder.training)

    def test_rejects_invalid_initial_replay_state_contract(self) -> None:
        vocab = MapperV1Vocab()
        model = MapperV1Model(_small_config(vocab), vocab=vocab)

        non_bos = _empty_batch(vocab)
        non_bos["target_tokens"] = non_bos["target_tokens"].clone()
        non_bos["target_tokens"][0, 0] = vocab.time_shift_token_id(10)
        with self.assertRaisesRegex(ValueError, r"target_tokens\[:, 0\] must be BOS"):
            model(**non_bos)

        masked_bos = _empty_batch(vocab)
        masked_bos["target_token_mask"] = torch.ones_like(masked_bos["target_tokens"], dtype=torch.bool)
        masked_bos["target_token_mask"][0, 0] = False
        with self.assertRaisesRegex(ValueError, "target_token_mask must mark BOS"):
            model(**masked_bos)

        shifted_current = _empty_batch(vocab)
        shifted_current["teacher_current_ms"] = shifted_current["teacher_current_ms"].clone()
        shifted_current["teacher_current_ms"][0, 0] = 10
        with self.assertRaisesRegex(ValueError, r"teacher_current_ms\[:, 0\] must equal write_start_ms"):
            model(**shifted_current)

        open_initial = _empty_batch(vocab)
        open_initial["teacher_open_mask"] = open_initial["teacher_open_mask"].clone()
        open_initial["teacher_open_mask"][0, 0, 0] = True
        with self.assertRaisesRegex(ValueError, r"teacher_open_mask\[:, 0\] must have all lanes closed"):
            model(**open_initial)

        aged_initial = _empty_batch(vocab)
        aged_initial["teacher_open_age_ms"] = aged_initial["teacher_open_age_ms"].clone()
        aged_initial["teacher_open_age_ms"][0, 0, 0] = 10
        with self.assertRaisesRegex(ValueError, r"teacher_open_age_ms\[:, 0\] must be zero"):
            model(**aged_initial)

    def test_rejects_non_8s_windows_and_out_of_window_teacher_states(self) -> None:
        vocab = MapperV1Vocab()
        model = MapperV1Model(_small_config(vocab), vocab=vocab)
        batch = _empty_batch(vocab, write_start_ms=8000)
        batch["teacher_current_ms"] = batch["teacher_current_ms"].clone()
        batch["teacher_current_ms"][0, 0] = 0

        with self.assertRaisesRegex(ValueError, "teacher_current_ms.*within"):
            model(**batch)

        bad_span = _empty_batch(vocab)
        bad_span["write_end_ms"] = torch.tensor([9000], dtype=torch.long)
        with self.assertRaisesRegex(ValueError, "8000ms write window"):
            model(**bad_span)

    def test_rejects_write_end_state_unless_predicting_eos(self) -> None:
        vocab = MapperV1Vocab()
        model = MapperV1Model(_small_config(vocab), vocab=vocab)
        batch = _empty_batch(vocab)
        batch["teacher_current_ms"] = batch["teacher_current_ms"].clone()
        batch["teacher_current_ms"][0, 0] = 8000

        with self.assertRaisesRegex(ValueError, "write_end_ms is valid only for EOS"):
            model(**batch)

    def test_rejects_control_memory_padding_mask_in_phase_b(self) -> None:
        vocab = MapperV1Vocab()
        model = MapperV1Model(_small_config(vocab), vocab=vocab)
        batch = _empty_batch(vocab)
        batch["control_memory_padding_mask_8s"] = torch.zeros((1, 400), dtype=torch.bool)

        with self.assertRaisesRegex(ValueError, "control_memory_padding_mask_8s is not supported"):
            model(**batch)

    def test_external_control_tensors_must_be_supplied_as_pair(self) -> None:
        vocab = MapperV1Vocab()
        model = MapperV1Model(_small_config(vocab), vocab=vocab)
        missing_density = _empty_batch(vocab)
        missing_density.pop("density_teacher_8s")
        with self.assertRaisesRegex(ValueError, "control_memory_8s and density_teacher_8s must be supplied together"):
            model(**missing_density)

        missing_control = _empty_batch(vocab)
        missing_control.pop("control_memory_8s")
        with self.assertRaisesRegex(ValueError, "control_memory_8s and density_teacher_8s must be supplied together"):
            model(**missing_control)

    def test_target_token_mask_hides_padded_teacher_state_from_window_validation(self) -> None:
        vocab = MapperV1Vocab()
        model = MapperV1Model(_small_config(vocab), vocab=vocab)
        long = _sample(
            encode_mapper_window(
                [MapperTimepoint(1000, _actions(LaneAction.TAP))],
                vocab=vocab,
                write_start_ms=0,
                write_end_ms=8000,
            ),
        )
        short = _sample(encode_mapper_window([], vocab=vocab, write_start_ms=8000, write_end_ms=16000))
        batch = collate_mapper_v1_windows([long, short], pad_id=vocab.pad_id)
        batch["control_memory_8s"] = torch.randn(2, 400, 32, dtype=torch.float32)
        batch["density_teacher_8s"] = torch.zeros(2, 400, 1, dtype=torch.float32)

        output = model(**batch)

        self.assertEqual(output.logits_final.shape[:2], batch["target_tokens"][:, :-1].shape)

        unmasked = dict(batch)
        unmasked.pop("target_token_mask")
        with self.assertRaisesRegex(ValueError, "teacher_current_ms.*within"):
            model(**unmasked)


def _small_config(vocab: MapperV1Vocab) -> MapperV1Config:
    return MapperV1Config(
        vocab_size=vocab.size,
        control_dim=32,
        d_model=32,
        heads=4,
        layers=1,
        ffn_dim=64,
        dropout=0.0,
        max_seq_len=16,
        state_hidden_dim=32,
        lane_embedding_dim=8,
    )


def _actions(*actions: LaneAction) -> tuple[LaneAction, ...]:
    padded = list(actions)
    while len(padded) < 4:
        padded.append(LaneAction.NONE)
    return tuple(padded)


def _empty_batch(vocab: MapperV1Vocab, *, write_start_ms: int = 0) -> dict[str, torch.Tensor]:
    tokenized = encode_mapper_window(
        [],
        vocab=vocab,
        write_start_ms=write_start_ms,
        write_end_ms=write_start_ms + 8000,
    )
    return {
        "target_tokens": tokenized.target_tensor().unsqueeze(0),
        "teacher_current_ms": tokenized.teacher_current_ms.unsqueeze(0),
        "teacher_open_mask": tokenized.teacher_open_mask.unsqueeze(0),
        "teacher_open_age_ms": tokenized.teacher_open_age_ms.unsqueeze(0),
        "write_start_ms": torch.tensor([tokenized.write_start_ms], dtype=torch.long),
        "write_end_ms": torch.tensor([tokenized.write_end_ms], dtype=torch.long),
        "normalized_difficulty": torch.tensor([[0.0]], dtype=torch.float32),
        "control_memory_8s": torch.randn(1, 400, 32, dtype=torch.float32),
        "density_teacher_8s": torch.zeros(1, 400, 1, dtype=torch.float32),
    }


def _sample(tokenized) -> dict[str, torch.Tensor]:
    return {
        "mel_context": torch.zeros(400, 160, dtype=torch.float32),
        "timing_context": torch.zeros(400, 4, dtype=torch.float32),
        "context_padding_mask": torch.zeros(400, dtype=torch.bool),
        "difficulty": torch.zeros(1, dtype=torch.float32),
        "normalized_difficulty": torch.zeros(1, dtype=torch.float32),
        "target_tokens": tokenized.target_tensor(),
        "teacher_current_ms": tokenized.teacher_current_ms,
        "teacher_open_mask": tokenized.teacher_open_mask,
        "teacher_open_age_ms": tokenized.teacher_open_age_ms,
        "close_labels": tokenized.close_labels,
        "close_label_mask": tokenized.close_label_mask,
        "density_target_8s": torch.zeros(400, 1, dtype=torch.float32),
        "density_confidence_8s": torch.ones(400, 1, dtype=torch.float32),
        "write_start_ms": torch.tensor(tokenized.write_start_ms, dtype=torch.long),
        "write_end_ms": torch.tensor(tokenized.write_end_ms, dtype=torch.long),
    }


class _TinyControlEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))

    def forward(self, *args, **kwargs):
        raise AssertionError("not used by this test")


if __name__ == "__main__":
    unittest.main()
