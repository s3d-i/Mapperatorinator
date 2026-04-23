import unittest

import torch
import torch.nn.functional as F

from train.stage1_oracle.models.mapper import Stage1OracleMapper, Stage1OracleMapperConfig


class Stage1MapperModelTests(unittest.TestCase):
    def test_forward_shape_and_parameter_budget(self) -> None:
        config = Stage1OracleMapperConfig(
            vocab_size=392,
            d_model=64,
            heads=4,
            encoder_layers=1,
            decoder_layers=1,
            ffn_dim=128,
            max_decode_len=16,
        )
        model = Stage1OracleMapper(config)
        logits = model(
            packed_audio=torch.zeros(2, 600, 160),
            timing_track=torch.zeros(2, 600, 5),
            difficulty_bucket=torch.tensor([0, 16]),
            decoder_input_ids=torch.ones(2, 6, dtype=torch.long),
        )

        self.assertEqual(logits.shape, (2, 6, 392))
        self.assertLess(model.parameter_count(), 25_000_000)

    def test_forward_rejects_short_encoder_frame_count(self) -> None:
        config = Stage1OracleMapperConfig(
            vocab_size=392,
            d_model=64,
            heads=4,
            encoder_layers=1,
            decoder_layers=1,
            ffn_dim=128,
            max_decode_len=16,
        )
        model = Stage1OracleMapper(config)

        with self.assertRaisesRegex(ValueError, "exactly 600"):
            model(
                packed_audio=torch.zeros(1, 599, 160),
                timing_track=torch.zeros(1, 599, 5),
                difficulty_bucket=torch.tensor([0]),
                decoder_input_ids=torch.ones(1, 6, dtype=torch.long),
            )

    def test_loss_ignores_condition_prefix_and_training_step_reduces_synthetic_loss(self) -> None:
        torch.manual_seed(0)
        config = Stage1OracleMapperConfig(
            vocab_size=32,
            d_model=32,
            heads=4,
            encoder_layers=1,
            decoder_layers=1,
            ffn_dim=64,
            max_decode_len=8,
            dropout=0.0,
        )
        model = Stage1OracleMapper(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
        packed_audio = torch.zeros(2, 600, 160)
        timing_track = torch.zeros(2, 600, 5)
        difficulty_bucket = torch.tensor([0, 1])
        decoder_input_ids = torch.tensor([[1, 3, 20, 5], [1, 4, 21, 6]])
        labels = torch.tensor([[-100, -100, 5, 2], [-100, -100, 6, 2]])

        def loss_value() -> torch.Tensor:
            logits = model(packed_audio, timing_track, difficulty_bucket, decoder_input_ids)
            return F.cross_entropy(logits.reshape(-1, config.vocab_size), labels.reshape(-1), ignore_index=-100)

        before = float(loss_value().detach())
        for _ in range(8):
            optimizer.zero_grad()
            loss = loss_value()
            loss.backward()
            optimizer.step()
        after = float(loss_value().detach())

        self.assertLess(after, before)

    def test_split_encode_decode_matches_forward(self) -> None:
        torch.manual_seed(7)
        config = Stage1OracleMapperConfig(
            vocab_size=48,
            d_model=32,
            heads=4,
            encoder_layers=1,
            decoder_layers=1,
            ffn_dim=64,
            max_decode_len=8,
            dropout=0.0,
        )
        model = Stage1OracleMapper(config)
        packed_audio = torch.randn(2, 600, 160)
        timing_track = torch.randn(2, 600, 5)
        difficulty_bucket = torch.tensor([2, 5])
        decoder_input_ids = torch.tensor([[1, 3, 20, 5], [1, 4, 21, 6]])
        decoder_padding_mask = torch.tensor([[False, False, False, False], [False, False, False, True]])

        logits = model(
            packed_audio=packed_audio,
            timing_track=timing_track,
            difficulty_bucket=difficulty_bucket,
            decoder_input_ids=decoder_input_ids,
            decoder_padding_mask=decoder_padding_mask,
        )
        memory = model.encode_context(
            packed_audio=packed_audio,
            timing_track=timing_track,
            difficulty_bucket=difficulty_bucket,
        )
        split_logits = model.decode_from_memory(
            memory=memory,
            decoder_input_ids=decoder_input_ids,
            decoder_padding_mask=decoder_padding_mask,
        )

        self.assertTrue(torch.allclose(logits, split_logits))


if __name__ == "__main__":
    unittest.main()
