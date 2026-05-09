import unittest

import torch
from torch import nn

from train.stage_2.model_mapper_v1.replay import ln_carry_state_tensors
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
        self.assertTrue(torch.equal(output.decoder_input_tokens, batch["decoder_input_tokens"]))
        self.assertTrue(torch.equal(output.loss_target_tokens, batch["target_fragment_tokens"]))
        self.assertTrue(torch.equal(output.state_current_ms, batch["target_fragment_states"]["current_ms"]))
        self.assertTrue(torch.equal(output.state_open_mask, batch["target_fragment_states"]["open_mask"]))
        self.assertTrue(torch.equal(output.state_open_start_ms, batch["target_fragment_states"]["open_start_ms"]))

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

    def test_rejects_old_target_tokens_teacher_contract(self) -> None:
        vocab = MapperV1Vocab()
        model = MapperV1Model(_small_config(vocab), vocab=vocab)
        batch = _empty_batch(vocab)
        batch["target_tokens"] = batch["target_fragment_tokens"].clone()

        with self.assertRaisesRegex(ValueError, "old target_tokens/teacher_\\* mapper contract"):
            model(**batch)

    def test_rejects_invalid_initial_replay_state_contract(self) -> None:
        vocab = MapperV1Vocab()
        model = MapperV1Model(_small_config(vocab), vocab=vocab)
        shifted_current = _empty_batch(vocab)
        shifted_current["target_fragment_states"] = dict(shifted_current["target_fragment_states"])
        shifted_current["target_fragment_states"]["current_ms"] = shifted_current["target_fragment_states"]["current_ms"].clone()
        shifted_current["target_fragment_states"]["current_ms"][0, 0] = 10
        with self.assertRaisesRegex(ValueError, r"target_fragment_states.current_ms\[:, 0\] must equal ln_carry_in.current_ms"):
            model(**shifted_current)

        mismatched_open = _empty_batch(vocab)
        mismatched_open["target_fragment_states"] = dict(mismatched_open["target_fragment_states"])
        mismatched_open["target_fragment_states"]["open_mask"] = mismatched_open["target_fragment_states"]["open_mask"].clone()
        mismatched_open["target_fragment_states"]["open_mask"][0, 0, 0] = True
        with self.assertRaisesRegex(ValueError, r"target_fragment_states.open_start_ms must be set for open lanes"):
            model(**mismatched_open)

        aged_initial = _empty_batch(vocab)
        aged_initial["target_fragment_states"] = dict(aged_initial["target_fragment_states"])
        aged_initial["target_fragment_states"]["open_age_ms"] = aged_initial["target_fragment_states"]["open_age_ms"].clone()
        aged_initial["target_fragment_states"]["open_age_ms"][0, 0, 0] = 10
        with self.assertRaisesRegex(ValueError, r"target_fragment_states.open_age_ms must be zero for closed lanes"):
            model(**aged_initial)

    def test_rejects_non_8s_windows_and_out_of_window_teacher_states(self) -> None:
        vocab = MapperV1Vocab()
        model = MapperV1Model(_small_config(vocab), vocab=vocab)
        batch = _empty_batch(vocab, write_start_ms=8000)
        batch["target_fragment_states"] = dict(batch["target_fragment_states"])
        batch["target_fragment_states"]["current_ms"] = batch["target_fragment_states"]["current_ms"].clone()
        batch["target_fragment_states"]["current_ms"][0, 0] = 0

        with self.assertRaisesRegex(ValueError, "target_fragment_states.current_ms.*within"):
            model(**batch)

        bad_span = _empty_batch(vocab)
        bad_span["write_end_ms"] = torch.tensor([9000], dtype=torch.long)
        with self.assertRaisesRegex(ValueError, "8000ms write window"):
            model(**bad_span)

    def test_rejects_write_end_state_unless_predicting_eos(self) -> None:
        vocab = MapperV1Vocab()
        model = MapperV1Model(_small_config(vocab), vocab=vocab)
        batch = _empty_batch(vocab)
        batch["target_fragment_states"] = dict(batch["target_fragment_states"])
        batch["target_fragment_states"]["current_ms"] = batch["target_fragment_states"]["current_ms"].clone()
        batch["target_fragment_states"]["current_ms"][0, 0] = 8000

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

    def test_target_fragment_mask_hides_padded_state_from_window_validation(self) -> None:
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
        batch = _collate_samples([long, short], vocab=vocab)
        batch["control_memory_8s"] = torch.randn(2, 400, 32, dtype=torch.float32)
        batch["density_teacher_8s"] = torch.zeros(2, 400, 1, dtype=torch.float32)
        batch["target_fragment_states"]["current_ms"][1, -1] = 99_999
        batch["target_fragment_states"]["open_mask"][1, -1] = True
        batch["target_fragment_states"]["open_start_ms"][1, -1] = 99_999
        batch["target_fragment_states"]["open_age_ms"][1, -1] = 99_999

        output = model(**batch)

        self.assertEqual(output.logits_final.shape[:2], batch["target_fragment_tokens"].shape)

        unmasked = dict(batch)
        unmasked["target_fragment_mask"] = torch.ones_like(batch["target_fragment_mask"], dtype=torch.bool)
        with self.assertRaisesRegex(ValueError, "target_fragment_states.current_ms.*within"):
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
        chart_end_ms=write_start_ms + 8000,
    )
    sample = _sample(tokenized)
    batch = _collate_samples([sample], vocab=vocab)
    batch["control_memory_8s"] = torch.randn(1, 400, 32, dtype=torch.float32)
    batch["density_teacher_8s"] = torch.zeros(1, 400, 1, dtype=torch.float32)
    return batch


def _sample(tokenized) -> dict[str, torch.Tensor]:
    return {
        "mel_context": torch.zeros(400, 160, dtype=torch.float32),
        "timing_context": torch.zeros(400, 4, dtype=torch.float32),
        "context_padding_mask": torch.zeros(400, dtype=torch.bool),
        "difficulty": torch.zeros(1, dtype=torch.float32),
        "normalized_difficulty": torch.zeros(1, dtype=torch.float32),
        "decoder_input_tokens": tokenized.decoder_input_tensor(),
        "target_fragment_tokens": tokenized.target_fragment_tensor(),
        "target_fragment_mask": torch.ones(tokenized.seq_len, dtype=torch.bool),
        "target_fragment_states": {
            "current_ms": tokenized.target_fragment_current_ms,
            "open_mask": tokenized.target_fragment_open_mask,
            "open_start_ms": tokenized.target_fragment_open_start_ms,
            "open_age_ms": tokenized.target_fragment_open_age_ms,
        },
        "ln_carry_in": ln_carry_state_tensors(tokenized.ln_carry_in),
        "ln_carry_out": ln_carry_state_tensors(tokenized.ln_carry_out),
        "close_labels": tokenized.close_labels,
        "close_label_mask": tokenized.close_label_mask,
        "density_target_8s": torch.zeros(400, 1, dtype=torch.float32),
        "density_confidence_8s": torch.ones(400, 1, dtype=torch.float32),
        "write_start_ms": torch.tensor(tokenized.write_start_ms, dtype=torch.long),
        "write_end_ms": torch.tensor(tokenized.write_end_ms, dtype=torch.long),
        "is_full_chart_start": torch.tensor(tokenized.is_full_chart_start, dtype=torch.bool),
        "is_full_chart_end": torch.tensor(tokenized.is_full_chart_end, dtype=torch.bool),
    }


def _collate_samples(samples, *, vocab: MapperV1Vocab) -> dict[str, torch.Tensor]:
    batch_size = len(samples)
    max_len = max(int(sample["target_fragment_tokens"].shape[0]) for sample in samples)
    decoder_input_tokens = torch.full((batch_size, max_len), vocab.pad_id, dtype=torch.long)
    target_fragment_tokens = torch.full((batch_size, max_len), vocab.pad_id, dtype=torch.long)
    target_fragment_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    current_ms = torch.zeros((batch_size, max_len), dtype=torch.long)
    open_mask = torch.zeros((batch_size, max_len, 4), dtype=torch.bool)
    open_start_ms = torch.full((batch_size, max_len, 4), -1, dtype=torch.long)
    open_age_ms = torch.zeros((batch_size, max_len, 4), dtype=torch.long)
    close_labels = torch.zeros((batch_size, max_len, 4), dtype=torch.bool)
    close_label_mask = torch.zeros((batch_size, max_len, 4), dtype=torch.bool)
    for index, sample in enumerate(samples):
        length = int(sample["target_fragment_tokens"].shape[0])
        decoder_input_tokens[index, :length] = sample["decoder_input_tokens"]
        target_fragment_tokens[index, :length] = sample["target_fragment_tokens"]
        target_fragment_mask[index, :length] = sample["target_fragment_mask"]
        states = sample["target_fragment_states"]
        current_ms[index, :length] = states["current_ms"]
        open_mask[index, :length] = states["open_mask"]
        open_start_ms[index, :length] = states["open_start_ms"]
        open_age_ms[index, :length] = states["open_age_ms"]
        close_labels[index, :length] = sample["close_labels"]
        close_label_mask[index, :length] = sample["close_label_mask"]
        if length < max_len:
            current_ms[index, length:] = sample["write_end_ms"]
    scalar_keys = ("write_start_ms", "write_end_ms", "is_full_chart_start", "is_full_chart_end")
    return {
        "decoder_input_tokens": decoder_input_tokens,
        "target_fragment_tokens": target_fragment_tokens,
        "target_fragment_mask": target_fragment_mask,
        "target_fragment_states": {
            "current_ms": current_ms,
            "open_mask": open_mask,
            "open_start_ms": open_start_ms,
            "open_age_ms": open_age_ms,
        },
        "ln_carry_in": {
            key: torch.stack([sample["ln_carry_in"][key] for sample in samples])
            for key in samples[0]["ln_carry_in"]
        },
        "ln_carry_out": {
            key: torch.stack([sample["ln_carry_out"][key] for sample in samples])
            for key in samples[0]["ln_carry_out"]
        },
        "close_labels": close_labels,
        "close_label_mask": close_label_mask,
        "density_target_8s": torch.stack([sample["density_target_8s"] for sample in samples]),
        "density_confidence_8s": torch.stack([sample["density_confidence_8s"] for sample in samples]),
        "write_start_ms": torch.stack([sample["write_start_ms"] for sample in samples]),
        "write_end_ms": torch.stack([sample["write_end_ms"] for sample in samples]),
        "is_full_chart_start": torch.stack([sample["is_full_chart_start"] for sample in samples]),
        "is_full_chart_end": torch.stack([sample["is_full_chart_end"] for sample in samples]),
        "normalized_difficulty": torch.stack([sample["normalized_difficulty"] for sample in samples]),
    }


class _TinyControlEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))

    def forward(self, *args, **kwargs):
        raise AssertionError("not used by this test")


if __name__ == "__main__":
    unittest.main()
