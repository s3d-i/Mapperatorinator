from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class Stage1OracleMapperConfig:
    vocab_size: int = 392
    audio_dim: int = 160
    timing_dim: int = 5
    difficulty_bucket_count: int = 17
    encoder_frame_count: int = 600
    max_decode_len: int = 512
    d_model: int = 256
    heads: int = 4
    encoder_layers: int = 4
    decoder_layers: int = 6
    ffn_dim: int = 1024
    dropout: float = 0.1


class Stage1OracleMapper(nn.Module):
    def __init__(self, config: Stage1OracleMapperConfig = Stage1OracleMapperConfig()) -> None:
        super().__init__()
        self.config = config
        self.audio_projection = nn.Linear(config.audio_dim, config.d_model)
        self.timing_projection = nn.Linear(config.timing_dim, config.d_model)
        self.difficulty_embedding = nn.Embedding(config.difficulty_bucket_count, config.d_model)
        self.fused_projection = nn.Linear(config.d_model * 3, config.d_model)
        self.encoder_position = nn.Parameter(torch.zeros(1, config.encoder_frame_count, config.d_model))
        self.decoder_position = nn.Parameter(torch.zeros(1, config.max_decode_len + 3, config.d_model))
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.heads,
            dim_feedforward=config.ffn_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.encoder_layers,
            enable_nested_tensor=False,
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.heads,
            dim_feedforward=config.ffn_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.decoder_layers)
        self.output_norm = nn.LayerNorm(config.d_model)
        self.token_head = nn.Linear(config.d_model, config.vocab_size)
        self.register_buffer(
            "_decoder_causal_mask",
            _causal_mask(config.max_decode_len + 3),
            persistent=False,
        )
        self._reset_parameters()

    def forward(
        self,
        packed_audio: torch.Tensor,
        timing_track: torch.Tensor,
        difficulty_bucket: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        decoder_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        memory = self.encode_context(
            packed_audio=packed_audio,
            timing_track=timing_track,
            difficulty_bucket=difficulty_bucket,
        )
        return self.decode_from_memory(
            memory=memory,
            decoder_input_ids=decoder_input_ids,
            decoder_padding_mask=decoder_padding_mask,
        )

    def encode_context(
        self,
        *,
        packed_audio: torch.Tensor,
        timing_track: torch.Tensor,
        difficulty_bucket: torch.Tensor,
    ) -> torch.Tensor:
        if packed_audio.ndim != 3 or packed_audio.shape[-1] != self.config.audio_dim:
            raise ValueError(f"packed_audio must have shape [B, T, {self.config.audio_dim}], got {packed_audio.shape}")
        if timing_track.ndim != 3 or timing_track.shape[-1] != self.config.timing_dim:
            raise ValueError(f"timing_track must have shape [B, T, {self.config.timing_dim}], got {timing_track.shape}")
        if packed_audio.shape[:2] != timing_track.shape[:2]:
            raise ValueError("packed_audio and timing_track must share batch/frame dimensions")

        batch_size, frame_count, _ = packed_audio.shape
        if frame_count != self.config.encoder_frame_count:
            raise ValueError(
                f"encoder frame count must be exactly {self.config.encoder_frame_count}, got {frame_count}",
            )

        audio_emb = self.audio_projection(packed_audio)
        timing_emb = self.timing_projection(timing_track)
        diff_emb = self.difficulty_embedding(difficulty_bucket).unsqueeze(1).expand(batch_size, frame_count, -1)
        fused = self.fused_projection(torch.cat([audio_emb, timing_emb, diff_emb], dim=-1))
        return self.encoder(fused + self.encoder_position[:, :frame_count])

    def decode_from_memory(
        self,
        *,
        memory: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        decoder_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if memory.ndim != 3:
            raise ValueError(f"memory must have shape [B, T, {self.config.d_model}], got {memory.shape}")
        if decoder_input_ids.ndim != 2:
            raise ValueError(f"decoder_input_ids must have shape [B, T], got {decoder_input_ids.shape}")
        if memory.shape[0] != decoder_input_ids.shape[0]:
            raise ValueError("memory and decoder_input_ids must share batch dimension")
        if memory.shape[-1] != self.config.d_model:
            raise ValueError(
                f"memory hidden size must match configured d_model={self.config.d_model}, got {memory.shape[-1]}",
            )
        if decoder_input_ids.shape[1] > self.decoder_position.shape[1]:
            raise ValueError(f"decoder length {decoder_input_ids.shape[1]} exceeds configured maximum")

        target = self.token_embedding(decoder_input_ids) + self.decoder_position[:, : decoder_input_ids.shape[1]]
        causal_mask = self._decoder_causal_mask[
            : decoder_input_ids.shape[1],
            : decoder_input_ids.shape[1],
        ]
        hidden = self.decoder(
            target,
            memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=decoder_padding_mask,
        )
        return self.token_head(self.output_norm(hidden))

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.encoder_position, mean=0.0, std=0.02)
        nn.init.normal_(self.decoder_position, mean=0.0, std=0.02)


def _causal_mask(length: int, *, device: torch.device | None = None) -> torch.Tensor:
    return torch.triu(torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1)
