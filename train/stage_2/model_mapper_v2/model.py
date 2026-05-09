from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from train.stage_2.model_control_demo_global.model import (
    ControlDemoGlobalEncoderConfig,
    MIN_GLOBAL_STRIDE,
    _GlobalSongEncoder,
    _mask_hidden,
)
from train.stage_2.model_mapper_v1.grammar import build_grammar_mask
from train.stage_2.model_mapper_v1.model import (
    MapperV1Config,
    MapperV1ForwardOutput,
    MapperV1Model,
    _difficulty_tensor,
    _load_carry_state,
    _reject_old_mapper_contract,
    _require_state_mapping,
    _require_state_tensor,
    _require_tensor,
    _sanitize_padded_fragment_states,
    _time_features,
    _validate_fragment_contract,
)
from train.stage_2.model_mapper_v1.tokenizer import (
    MAPPER_DENSITY_FRAME_MS,
    MAPPER_DENSITY_FRAMES,
    MAPPER_WRITE_MS,
)
from train.stage_2.model_mapper_v1.vocab import MapperV1Vocab


GLOBAL_POSITION_FEATURES = 4


@dataclass(frozen=True)
class MapperV2Config(MapperV1Config):
    use_global_context: bool = True
    global_stride: int = 16
    global_layers: int = 1
    global_ffn_dim: int | None = None
    global_conv_blocks: int = 1
    global_conv_kernel_size: int = 5
    global_gate_init: float = -2.94


@dataclass(frozen=True)
class MapperV2ForwardOutput(MapperV1ForwardOutput):
    global_memory: torch.Tensor | None = None
    global_memory_padding_mask: torch.Tensor | None = None
    global_attention_gates: torch.Tensor | None = None
    global_position_features: torch.Tensor | None = None


MapperV2ModelOutput = MapperV2ForwardOutput


class MapperV2Model(MapperV1Model):
    def __init__(
        self,
        config: MapperV2Config = MapperV2Config(),
        *,
        vocab: MapperV1Vocab | None = None,
        control_encoder: nn.Module | None = None,
    ) -> None:
        _validate_v2_config(config)
        super().__init__(config, vocab=vocab, control_encoder=control_encoder)
        self.config: MapperV2Config = config
        if config.use_global_context:
            self.global_encoder: _GlobalSongEncoder | None = _GlobalSongEncoder(_global_encoder_config(config))
            global_position_projection = nn.Linear(GLOBAL_POSITION_FEATURES, config.d_model)
            nn.init.normal_(global_position_projection.weight, mean=0.0, std=0.01)
            nn.init.zeros_(global_position_projection.bias)
            self.global_position_projection: nn.Module | None = global_position_projection
            self.global_cross_attention_layers = nn.ModuleList(
                [
                    _MapperGlobalCrossAttentionBlock(
                        d_model=config.d_model,
                        heads=config.heads,
                        dropout=config.dropout,
                        gate_init=config.global_gate_init,
                    )
                    for _ in range(config.layers)
                ],
            )
        else:
            self.global_encoder = None
            self.global_position_projection = None
            self.global_cross_attention_layers = nn.ModuleList()

    def forward(
        self,
        batch: Mapping[str, torch.Tensor] | None = None,
        *,
        control_memory_8s: torch.Tensor | None = None,
        density_teacher_8s: torch.Tensor | None = None,
        **kwargs: torch.Tensor | None,
    ) -> MapperV2ForwardOutput:
        if batch is None:
            batch = {key: value for key, value in kwargs.items() if value is not None}
        elif kwargs:
            merged = dict(batch)
            merged.update({key: value for key, value in kwargs.items() if value is not None})
            batch = merged
        if control_memory_8s is None:
            maybe_control_memory = batch.get("control_memory_8s")
            if isinstance(maybe_control_memory, torch.Tensor):
                control_memory_8s = maybe_control_memory
        if density_teacher_8s is None:
            maybe_density_teacher = batch.get("density_teacher_8s")
            if isinstance(maybe_density_teacher, torch.Tensor):
                density_teacher_8s = maybe_density_teacher
        if (control_memory_8s is None) != (density_teacher_8s is None):
            raise ValueError("control_memory_8s and density_teacher_8s must be supplied together")
        if isinstance(batch.get("control_memory_padding_mask_8s"), torch.Tensor):
            raise ValueError("control_memory_padding_mask_8s is not supported in Phase B; supply full 8s control memory")

        _reject_old_mapper_contract(batch)
        decoder_input = _require_tensor(batch, "decoder_input_tokens", ndim=2).to(dtype=torch.long)
        loss_target_tokens = _require_tensor(batch, "target_fragment_tokens", ndim=2).to(
            device=decoder_input.device,
            dtype=torch.long,
        )
        target_fragment_mask = _require_tensor(batch, "target_fragment_mask", ndim=2).to(
            device=decoder_input.device,
            dtype=torch.bool,
        )
        if int(decoder_input.shape[1]) < 1:
            raise ValueError("decoder_input_tokens must contain at least one fragment position")
        if tuple(loss_target_tokens.shape) != tuple(decoder_input.shape):
            raise ValueError("target_fragment_tokens must match decoder_input_tokens shape")
        if tuple(target_fragment_mask.shape) != tuple(decoder_input.shape):
            raise ValueError("target_fragment_mask must match decoder_input_tokens shape")
        input_padding_mask = ~target_fragment_mask

        device = decoder_input.device
        states = _require_state_mapping(batch, "target_fragment_states")
        current_ms = _require_state_tensor(states, "current_ms", ndim=2).to(device=device, dtype=torch.long)
        open_mask = _require_state_tensor(states, "open_mask", ndim=3).to(device=device, dtype=torch.bool)
        open_start_ms = _require_state_tensor(states, "open_start_ms", ndim=3).to(device=device, dtype=torch.long)
        open_age_ms = _require_state_tensor(states, "open_age_ms", ndim=3).to(device=device, dtype=torch.long)
        write_start_ms = _require_tensor(batch, "write_start_ms", ndim=1).to(device=device, dtype=torch.long)
        write_end_ms = _require_tensor(batch, "write_end_ms", ndim=1).to(device=device, dtype=torch.long)
        is_full_chart_start = _require_tensor(batch, "is_full_chart_start", ndim=1).to(device=device, dtype=torch.bool)
        is_full_chart_end = _require_tensor(batch, "is_full_chart_end", ndim=1).to(device=device, dtype=torch.bool)
        ln_carry_in = _load_carry_state(batch, "ln_carry_in", device=device)
        ln_carry_out = _load_carry_state(batch, "ln_carry_out", device=device)
        if tuple(current_ms.shape) != tuple(decoder_input.shape):
            raise ValueError("target_fragment_states.current_ms must align with decoder_input_tokens")
        if tuple(open_mask.shape[:2]) != tuple(decoder_input.shape) or int(open_mask.shape[-1]) != 4:
            raise ValueError("target_fragment_states.open_mask must have shape [B,S,4]")
        if tuple(open_start_ms.shape) != tuple(open_mask.shape):
            raise ValueError("target_fragment_states.open_start_ms must align with target_fragment_states.open_mask")
        if tuple(open_age_ms.shape) != tuple(open_mask.shape):
            raise ValueError("target_fragment_states.open_age_ms must align with target_fragment_states.open_mask")
        valid_input_mask = target_fragment_mask.to(device=device, dtype=torch.bool)
        _validate_fragment_contract(
            decoder_input_tokens=decoder_input,
            target_fragment_tokens=loss_target_tokens,
            target_fragment_mask=valid_input_mask,
            current_ms=current_ms,
            open_mask=open_mask,
            open_start_ms=open_start_ms,
            open_age_ms=open_age_ms,
            write_start_ms=write_start_ms,
            write_end_ms=write_end_ms,
            is_full_chart_start=is_full_chart_start,
            is_full_chart_end=is_full_chart_end,
            ln_carry_in=ln_carry_in,
            ln_carry_out=ln_carry_out,
            bos_id=self.vocab.bos_id,
            eos_id=self.vocab.eos_id,
        )
        current_ms, open_mask, open_start_ms, open_age_ms = _sanitize_padded_fragment_states(
            current_ms=current_ms,
            open_mask=open_mask,
            open_start_ms=open_start_ms,
            open_age_ms=open_age_ms,
            write_end_ms=write_end_ms,
            valid_input_mask=valid_input_mask,
        )

        if control_memory_8s is None:
            control_memory_8s, density_teacher_8s = self._control_teacher_8s(batch)
        control_memory_8s = control_memory_8s.detach().to(device=decoder_input.device, dtype=torch.float32)
        density_teacher_8s = density_teacher_8s.detach().to(device=decoder_input.device, dtype=torch.float32)
        if control_memory_8s.ndim != 3 or int(control_memory_8s.shape[1]) != MAPPER_DENSITY_FRAMES:
            raise ValueError(f"control_memory_8s must have shape [B,{MAPPER_DENSITY_FRAMES},D]")
        if int(control_memory_8s.shape[-1]) != self.config.control_dim:
            raise ValueError(
                f"control_memory_8s last dim must match config.control_dim={self.config.control_dim}, "
                f"got {control_memory_8s.shape[-1]}"
            )
        if tuple(density_teacher_8s.shape) != (decoder_input.shape[0], MAPPER_DENSITY_FRAMES, 1):
            raise ValueError(f"density_teacher_8s must have shape [B,{MAPPER_DENSITY_FRAMES},1]")
        control_memory = self.control_projection(control_memory_8s)
        global_memory, global_memory_padding_mask, global_position_features = self._global_context_memory(
            batch=batch,
            device=device,
            batch_size=int(decoder_input.shape[0]),
            write_start_ms=write_start_ms,
        )

        decoder_hidden, base_logits = self._decode_with_global_context(
            tokens=decoder_input,
            current_ms=current_ms,
            write_start_ms=write_start_ms,
            write_end_ms=write_end_ms,
            difficulty=_difficulty_tensor(batch, device=decoder_input.device, dim=self.config.difficulty_dim),
            control_memory=control_memory,
            input_padding_mask=input_padding_mask,
            global_memory=global_memory,
            global_memory_padding_mask=global_memory_padding_mask,
            global_position_features=global_position_features,
        )
        remaining_ms = (write_end_ms.reshape(-1, 1) - current_ms).clamp_min(0)
        state_prior = self.state_prior_adapter(
            open_mask=open_mask,
            open_start_ms=open_start_ms,
            open_age_ms=open_age_ms,
            remaining_ms=remaining_ms,
            write_start_ms=write_start_ms,
        )
        ln_close = self.ln_close_adapter(
            decoder_hidden=decoder_hidden,
            control_memory_8s=control_memory,
            density_teacher_8s=density_teacher_8s,
            current_ms=current_ms,
            write_start_ms=write_start_ms,
            open_mask=open_mask,
            open_start_ms=open_start_ms,
            open_age_ms=open_age_ms,
            remaining_ms=remaining_ms,
        )
        positions = torch.arange(decoder_input.shape[1], dtype=torch.long, device=decoder_input.device).reshape(1, -1)
        grammar_mask = build_grammar_mask(
            current_ms=current_ms,
            open_mask=open_mask,
            open_start_ms=open_start_ms,
            open_age_ms=open_age_ms,
            write_start_ms=write_start_ms,
            write_end_ms=write_end_ms,
            ln_carry_in=ln_carry_in,
            ln_carry_out=ln_carry_out,
            is_full_chart_start=is_full_chart_start,
            is_full_chart_end=is_full_chart_end,
            vocab=self.vocab,
            positions=positions.expand(decoder_input.shape[0], -1),
        ).to(dtype=base_logits.dtype)
        logits_final = (
            base_logits
            + state_prior.vocab_bias
            + ln_close.event_bias
            + ln_close.time_shift_bias
            + grammar_mask
        )
        return MapperV2ForwardOutput(
            decoder_input_tokens=decoder_input,
            loss_target_tokens=loss_target_tokens,
            state_current_ms=current_ms,
            state_open_mask=open_mask,
            state_open_start_ms=open_start_ms,
            state_open_age_ms=open_age_ms,
            base_logits=base_logits,
            logits_final=logits_final,
            decoder_hidden=decoder_hidden,
            state_prior_bias=state_prior.vocab_bias,
            state_prior_lane_action_bias=state_prior.lane_action_bias,
            ln_close_logits=ln_close.close_logits,
            ln_close_event_bias=ln_close.event_bias,
            ln_close_time_shift_bias=ln_close.time_shift_bias,
            grammar_mask=grammar_mask,
            control_memory_8s=control_memory,
            density_teacher_8s=density_teacher_8s,
            global_memory=global_memory,
            global_memory_padding_mask=global_memory_padding_mask,
            global_attention_gates=self._global_attention_gates(device=device, enabled=global_memory is not None),
            global_position_features=global_position_features,
        )

    def _decode_with_global_context(
        self,
        *,
        tokens: torch.Tensor,
        current_ms: torch.Tensor,
        write_start_ms: torch.Tensor,
        write_end_ms: torch.Tensor,
        difficulty: torch.Tensor,
        control_memory: torch.Tensor,
        input_padding_mask: torch.Tensor | None,
        global_memory: torch.Tensor | None,
        global_memory_padding_mask: torch.Tensor | None,
        global_position_features: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, steps = tokens.shape
        if steps > self.config.max_seq_len:
            raise ValueError(f"decoder sequence length {steps} exceeds max_seq_len={self.config.max_seq_len}")
        token_hidden = self.token_embedding(tokens)
        position_hidden = self.position[:, :steps]
        difficulty_hidden = self.difficulty_projection(difficulty).unsqueeze(1)
        time_features = _time_features(
            current_ms=current_ms,
            write_start_ms=write_start_ms,
            write_end_ms=write_end_ms,
        )
        hidden = token_hidden + position_hidden + difficulty_hidden + self.time_projection(time_features)
        if global_position_features is not None:
            if self.global_position_projection is None:
                raise ValueError("global_position_projection is required when global_position_features are supplied")
            global_position_input = global_position_features.to(device=tokens.device, dtype=token_hidden.dtype)
            global_position_hidden = self.global_position_projection(global_position_input).unsqueeze(1)
            hidden = hidden + global_position_hidden
        if input_padding_mask is not None:
            hidden = hidden.masked_fill(input_padding_mask.unsqueeze(-1), 0.0)
        causal_mask = torch.triu(
            torch.ones((steps, steps), dtype=torch.bool, device=tokens.device),
            diagonal=1,
        )
        for layer_index, layer in enumerate(self.decoder_layers):
            hidden = layer(
                tgt=hidden,
                memory=control_memory,
                tgt_mask=causal_mask,
                tgt_key_padding_mask=input_padding_mask,
                memory_key_padding_mask=None,
            )
            if input_padding_mask is not None:
                hidden = hidden.masked_fill(input_padding_mask.unsqueeze(-1), 0.0)
            if global_memory is not None:
                if global_memory_padding_mask is None:
                    raise ValueError("global_memory_padding_mask is required when global_memory is supplied")
                hidden = self.global_cross_attention_layers[layer_index](
                    hidden=hidden,
                    input_padding_mask=input_padding_mask,
                    global_memory=global_memory,
                    global_memory_padding_mask=global_memory_padding_mask,
                )
        decoder_hidden = self.output_norm(hidden)
        base_logits = self.output_head(decoder_hidden)
        return decoder_hidden, base_logits

    def _global_context_memory(
        self,
        *,
        batch: Mapping[str, Any],
        device: torch.device,
        batch_size: int,
        write_start_ms: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        if not self.config.use_global_context:
            return None, None, None
        if self.global_encoder is None:
            raise ValueError("global context is enabled but global_encoder is missing")
        full_mel = _require_tensor(batch, "full_mel", ndim=3).to(device=device, dtype=torch.float32)
        full_dense_timing_v2 = _require_tensor(batch, "full_dense_timing_v2", ndim=3).to(device=device, dtype=torch.float32)
        padding_mask = _require_tensor(batch, "padding_mask", ndim=2).to(device=device, dtype=torch.bool)
        raw_padded_frame_count = _require_tensor(batch, "frame_count", ndim=1)
        if (
            raw_padded_frame_count.dtype == torch.bool
            or raw_padded_frame_count.dtype.is_floating_point
            or raw_padded_frame_count.dtype.is_complex
        ):
            raise ValueError("frame_count must be an integer tensor")
        raw_source_frame_count = _require_tensor(batch, "source_frame_count", ndim=1)
        if (
            raw_source_frame_count.dtype == torch.bool
            or raw_source_frame_count.dtype.is_floating_point
            or raw_source_frame_count.dtype.is_complex
        ):
            raise ValueError("source_frame_count must be an integer tensor")
        padded_frame_count = raw_padded_frame_count.to(device=device, dtype=torch.long)
        source_frame_count = raw_source_frame_count.to(device=device, dtype=torch.long)
        target_start_frame = _target_start_frame(batch=batch, write_start_ms=write_start_ms, device=device)
        _validate_global_context_inputs(
            full_mel=full_mel,
            full_dense_timing_v2=full_dense_timing_v2,
            padding_mask=padding_mask,
            padded_frame_count=padded_frame_count,
            source_frame_count=source_frame_count,
            target_start_frame=target_start_frame,
            batch_size=batch_size,
            config=self.config,
        )
        global_position_features = _global_position_features(
            source_frame_count=source_frame_count,
            target_start_frame=target_start_frame,
            device=device,
        )
        global_memory, global_memory_padding_mask, _ = self.global_encoder(
            full_mel=full_mel,
            full_dense_timing_v2=full_dense_timing_v2,
            padding_mask=padding_mask,
            frame_count=source_frame_count,
            target_start_frame=target_start_frame,
        )
        return global_memory, global_memory_padding_mask, global_position_features

    def _global_attention_gates(self, *, device: torch.device, enabled: bool) -> torch.Tensor | None:
        if not enabled:
            return None
        return torch.stack(
            [torch.sigmoid(block.gate_logit) for block in self.global_cross_attention_layers],
        ).to(device=device)


class _MapperGlobalCrossAttentionBlock(nn.Module):
    def __init__(self, *, d_model: int, heads: int, dropout: float, gate_init: float) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(d_model)
        self.memory_norm = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.gate_logit = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(
        self,
        *,
        hidden: torch.Tensor,
        input_padding_mask: torch.Tensor | None,
        global_memory: torch.Tensor,
        global_memory_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        memory = self.memory_norm(global_memory)
        cross, _ = self.cross_attn(
            query=self.query_norm(hidden),
            key=memory,
            value=memory,
            key_padding_mask=global_memory_padding_mask,
            need_weights=False,
        )
        hidden = hidden + torch.sigmoid(self.gate_logit).to(dtype=hidden.dtype) * self.dropout(cross)
        return _mask_hidden(hidden, input_padding_mask)


def _global_encoder_config(config: MapperV2Config) -> ControlDemoGlobalEncoderConfig:
    return ControlDemoGlobalEncoderConfig(
        mel_dim=config.mel_dim,
        timing_dim=config.timing_dim,
        target_frames=MAPPER_DENSITY_FRAMES,
        d_model=config.d_model,
        heads=config.heads,
        dropout=config.dropout,
        global_stride=config.global_stride,
        global_layers=config.global_layers,
        global_ffn_dim=config.ffn_dim if config.global_ffn_dim is None else int(config.global_ffn_dim),
        global_conv_blocks=config.global_conv_blocks,
        conv_kernel_size=config.global_conv_kernel_size,
    )


def _target_start_frame(
    *,
    batch: Mapping[str, Any],
    write_start_ms: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    value = batch.get("target_start_frame")
    if value is None:
        return (write_start_ms // MAPPER_DENSITY_FRAME_MS).to(device=device, dtype=torch.long)
    if not isinstance(value, torch.Tensor):
        raise ValueError("target_start_frame must be a torch.Tensor")
    if value.ndim != 1:
        raise ValueError(f"target_start_frame must have shape [B], got {tuple(value.shape)}")
    if value.dtype == torch.bool or value.dtype.is_floating_point or value.dtype.is_complex:
        raise ValueError("target_start_frame must be an integer tensor")
    return value.to(device=device, dtype=torch.long)


def _global_position_features(
    *,
    source_frame_count: torch.Tensor,
    target_start_frame: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    source_count_f = source_frame_count.to(device=device, dtype=torch.float32)
    target_start_f = target_start_frame.to(device=device, dtype=torch.float32)
    denominator = (source_count_f - 1.0).clamp_min(1.0)
    target_center = target_start_f + 0.5 * float(MAPPER_DENSITY_FRAMES)
    song_seconds = source_count_f * (float(MAPPER_DENSITY_FRAME_MS) / 1000.0)
    seconds_to_end = (source_count_f - target_center).clamp_min(0.0) * (
        float(MAPPER_DENSITY_FRAME_MS) / 1000.0
    )
    return torch.stack(
        (
            target_start_f / denominator,
            target_center / denominator,
            seconds_to_end / 300.0,
            torch.log1p(song_seconds) / 6.0,
        ),
        dim=-1,
    ).to(dtype=torch.float32)


def _validate_global_context_inputs(
    *,
    full_mel: torch.Tensor,
    full_dense_timing_v2: torch.Tensor,
    padding_mask: torch.Tensor,
    padded_frame_count: torch.Tensor,
    source_frame_count: torch.Tensor,
    target_start_frame: torch.Tensor,
    batch_size: int,
    config: MapperV2Config,
) -> None:
    if full_mel.ndim != 3 or int(full_mel.shape[-1]) != config.mel_dim:
        raise ValueError(f"full_mel must have shape [B,F,{config.mel_dim}], got {tuple(full_mel.shape)}")
    if full_dense_timing_v2.ndim != 3 or int(full_dense_timing_v2.shape[-1]) != config.timing_dim:
        raise ValueError(
            f"full_dense_timing_v2 must have shape [B,F,{config.timing_dim}], "
            f"got {tuple(full_dense_timing_v2.shape)}"
        )
    if tuple(full_mel.shape[:2]) != tuple(full_dense_timing_v2.shape[:2]):
        raise ValueError("full_mel and full_dense_timing_v2 must share batch/frame dimensions")
    if int(full_mel.shape[0]) != int(batch_size):
        raise ValueError("full-song tensors must share decoder batch size")
    if tuple(padding_mask.shape) != tuple(full_mel.shape[:2]):
        raise ValueError("padding_mask must have shape [B,F]")
    if padding_mask.dtype != torch.bool:
        raise ValueError("padding_mask must be bool")
    if tuple(padded_frame_count.shape) != (batch_size,):
        raise ValueError("frame_count must have shape [B]")
    if tuple(source_frame_count.shape) != (batch_size,):
        raise ValueError("source_frame_count must have shape [B]")
    if tuple(target_start_frame.shape) != (batch_size,):
        raise ValueError("target_start_frame must have shape [B]")
    if bool(torch.any(padded_frame_count <= 0)):
        raise ValueError("frame_count must be positive")
    if bool(torch.any(source_frame_count <= 0)):
        raise ValueError("source_frame_count must be positive")
    if bool(torch.any(source_frame_count > padded_frame_count)):
        raise ValueError("source_frame_count cannot exceed frame_count")
    if bool(torch.any(padded_frame_count > full_mel.shape[1])):
        raise ValueError("frame_count cannot exceed full-song tensor length")
    if bool(torch.any(target_start_frame < 0)):
        raise ValueError("target_start_frame must be non-negative")
    if bool(torch.any(target_start_frame >= source_frame_count)):
        raise ValueError("target_start_frame must be less than source_frame_count")
    frame_index = torch.arange(full_mel.shape[1], device=full_mel.device).unsqueeze(0)
    beyond_source_frame_count = frame_index >= source_frame_count.to(device=full_mel.device).unsqueeze(1)
    if bool((beyond_source_frame_count & ~padding_mask).any()):
        raise ValueError("padding_mask must cover source_frame_count tail")


def _validate_v2_config(config: MapperV2Config) -> None:
    if not isinstance(config.use_global_context, bool):
        raise ValueError("use_global_context must be bool")
    if not config.use_global_context:
        return
    for name in ("global_stride", "global_layers", "global_conv_blocks", "global_conv_kernel_size"):
        value = getattr(config, name)
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{name} must be an integer")
    if config.global_ffn_dim is not None and (
        not isinstance(config.global_ffn_dim, int) or isinstance(config.global_ffn_dim, bool)
    ):
        raise ValueError("global_ffn_dim must be an integer when set")
    if not isinstance(config.global_gate_init, (int, float)) or isinstance(config.global_gate_init, bool):
        raise ValueError("global_gate_init must be numeric")
    if not math.isfinite(float(config.global_gate_init)):
        raise ValueError("global_gate_init must be finite")
    if config.global_stride < MIN_GLOBAL_STRIDE:
        raise ValueError(f"global_stride must be at least {MIN_GLOBAL_STRIDE}")
    if config.global_layers <= 0:
        raise ValueError("global_layers must be positive")
    if config.global_ffn_dim is not None and config.global_ffn_dim <= 0:
        raise ValueError("global_ffn_dim must be positive when set")
    if config.global_conv_blocks < 0:
        raise ValueError("global_conv_blocks must be non-negative")
    if config.global_conv_kernel_size <= 0 or config.global_conv_kernel_size % 2 == 0:
        raise ValueError("global_conv_kernel_size must be a positive odd integer")
