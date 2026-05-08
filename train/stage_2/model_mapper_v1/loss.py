from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import nn

from .vocab import MapperV1Vocab


@dataclass(frozen=True)
class MapperV1LossConfig:
    lambda_density: float = 0.0
    lambda_ln_close: float = 0.05
    lambda_adapter_reg: float = 1e-5
    ln_close_pos_weight: float = 1.0
    ln_close_focal_gamma: float = 1.5


@dataclass(frozen=True)
class MapperV1LossOutput:
    total_loss: torch.Tensor
    token_loss: torch.Tensor
    ln_close_loss: torch.Tensor
    density_loss: torch.Tensor
    adapter_reg_loss: torch.Tensor
    metrics: dict[str, float]
    metric_numerators: dict[str, float] = field(default_factory=dict)
    metric_denominators: dict[str, float] = field(default_factory=dict)


class MapperV1ModelLoss(nn.Module):
    def __init__(self, config: MapperV1LossConfig | None = None, *, vocab: MapperV1Vocab | None = None) -> None:
        super().__init__()
        config = MapperV1LossConfig() if config is None else config
        _validate_config(config)
        self.config = config
        self.vocab = MapperV1Vocab() if vocab is None else vocab

    def forward(self, output: Any, batch: Mapping[str, torch.Tensor]) -> MapperV1LossOutput:
        logits_final = _require_tensor_attr(output, "logits_final")
        target_tokens = _require_batch_tensor(batch, "target_tokens")
        if target_tokens.ndim != 2:
            raise ValueError(f"target_tokens must have shape [B,S], got {tuple(target_tokens.shape)}")
        if logits_final.ndim != 3 or tuple(logits_final.shape[:2]) != tuple(target_tokens[:, 1:].shape):
            raise ValueError(
                "logits_final must have shape [B,S-1,V] matching target_tokens[:, 1:], "
                f"got logits={tuple(logits_final.shape)} target={tuple(target_tokens.shape)}"
            )

        target = target_tokens[:, 1:].to(device=logits_final.device, dtype=torch.long)
        target_mask = _target_loss_mask(batch, target=target, pad_id=self.vocab.pad_id)
        token_loss = token_cross_entropy(
            logits_final,
            target,
            pad_id=self.vocab.pad_id,
            target_mask=target_mask,
        )

        close_logits = _require_tensor_attr(output, "ln_close_logits")
        close_labels = _require_batch_tensor(batch, "close_labels")[:, :-1]
        close_mask = _require_batch_tensor(batch, "close_label_mask")[:, :-1]
        ln_close_loss = ln_close_focal_bce_loss(
            close_logits=close_logits,
            labels=close_labels,
            mask=close_mask,
            pos_weight=self.config.ln_close_pos_weight,
            gamma=self.config.ln_close_focal_gamma,
        )

        input_mask = _input_loss_mask(batch, steps=logits_final.shape[1], device=logits_final.device)
        adapter_reg_loss = adapter_bias_regularization(
            getattr(output, "state_prior_bias", None),
            getattr(output, "ln_close_event_bias", None),
            getattr(output, "ln_close_time_shift_bias", None),
            mask=input_mask,
        )
        density_loss = logits_final.new_zeros(())
        total_loss = (
            token_loss
            + float(self.config.lambda_ln_close) * ln_close_loss
            + float(self.config.lambda_density) * density_loss
            + float(self.config.lambda_adapter_reg) * adapter_reg_loss
        )

        metrics: dict[str, float] = {}
        numerators: dict[str, float] = {}
        denominators: dict[str, float] = {}
        _record_scalar(metrics, "loss/total", total_loss)
        _record_scalar(metrics, "loss/token", token_loss)
        _record_scalar(metrics, "loss/ln_close", ln_close_loss)
        _record_scalar(metrics, "loss/density", density_loss)
        _record_scalar(metrics, "loss/adapter_reg", adapter_reg_loss)
        metrics["phase/lambda_density"] = float(self.config.lambda_density)
        metrics["phase/lambda_ln_close"] = float(self.config.lambda_ln_close)
        metrics["token/valid_count"] = int(target_mask.sum().detach().cpu())
        metrics["ln_close/open_lane_count"] = int(close_mask.to(dtype=torch.bool).sum().detach().cpu())
        metrics["ln_close/positive_count"] = int((close_labels.to(dtype=torch.bool) & close_mask.to(dtype=torch.bool)).sum().detach().cpu())
        numerators["loss/token"] = float((token_loss.detach() * target_mask.to(dtype=token_loss.dtype).sum().clamp_min(1)).cpu())
        denominators["loss/token"] = float(target_mask.sum().detach().cpu())
        numerators["loss/ln_close"] = float(
            (ln_close_loss.detach() * close_mask.to(device=ln_close_loss.device, dtype=ln_close_loss.dtype).sum().clamp_min(1)).cpu()
        )
        denominators["loss/ln_close"] = float(close_mask.sum().detach().cpu())

        return MapperV1LossOutput(
            total_loss=total_loss,
            token_loss=token_loss,
            ln_close_loss=ln_close_loss,
            density_loss=density_loss,
            adapter_reg_loss=adapter_reg_loss,
            metrics=metrics,
            metric_numerators=numerators,
            metric_denominators=denominators,
        )


def token_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    pad_id: int | None = None,
    target_mask: torch.Tensor | None = None,
    grammar_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if logits.ndim != 3:
        raise ValueError(f"logits must have shape [B,T,V], got {tuple(logits.shape)}")
    if tuple(target.shape) != tuple(logits.shape[:2]):
        raise ValueError(f"target must have shape {tuple(logits.shape[:2])}, got {tuple(target.shape)}")
    if grammar_mask is not None:
        if tuple(grammar_mask.shape) != tuple(logits.shape):
            raise ValueError(f"grammar_mask must have shape {tuple(logits.shape)}, got {tuple(grammar_mask.shape)}")
        logits = logits + grammar_mask.to(device=logits.device, dtype=logits.dtype)
    target = target.to(device=logits.device, dtype=torch.long)
    if target_mask is None:
        valid = torch.ones_like(target, dtype=torch.bool) if pad_id is None else target != int(pad_id)
    else:
        if tuple(target_mask.shape) != tuple(target.shape):
            raise ValueError(f"target_mask must have shape {tuple(target.shape)}, got {tuple(target_mask.shape)}")
        valid = target_mask.to(device=logits.device, dtype=torch.bool)
        if pad_id is not None:
            valid = valid & (target != int(pad_id))
    if not bool(valid.any()):
        return logits.sum() * 0.0

    safe_target = target.masked_fill(~valid, 0)
    losses = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        safe_target.reshape(-1),
        reduction="none",
    ).reshape_as(target)
    return losses[valid].mean()


def ln_close_focal_bce_loss(
    *,
    close_logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    pos_weight: float = 1.0,
    gamma: float = 1.5,
) -> torch.Tensor:
    if close_logits.ndim != 3:
        raise ValueError(f"close_logits must have shape [B,T,4], got {tuple(close_logits.shape)}")
    if tuple(labels.shape) != tuple(close_logits.shape) or tuple(mask.shape) != tuple(close_logits.shape):
        raise ValueError("labels and mask must match close_logits shape")
    _require_finite_number(pos_weight, "pos_weight")
    _require_finite_number(gamma, "gamma")
    if pos_weight <= 0.0:
        raise ValueError("pos_weight must be positive")
    if gamma < 0.0:
        raise ValueError("gamma must be non-negative")

    labels_f = labels.to(device=close_logits.device, dtype=close_logits.dtype)
    mask_f = mask.to(device=close_logits.device, dtype=close_logits.dtype)
    if not bool(mask_f.to(dtype=torch.bool).any()):
        return close_logits.sum() * 0.0
    pos_weight_tensor = close_logits.new_tensor(float(pos_weight))
    bce = F.binary_cross_entropy_with_logits(
        close_logits,
        labels_f,
        pos_weight=pos_weight_tensor,
        reduction="none",
    )
    if gamma > 0.0:
        prob = torch.sigmoid(close_logits)
        p_t = torch.where(labels_f > 0.5, prob, 1.0 - prob)
        bce = bce * (1.0 - p_t).clamp_min(0.0).pow(float(gamma))
    return (bce * mask_f).sum() / mask_f.sum().clamp_min(torch.finfo(mask_f.dtype).eps)


def ln_close_aux_loss(
    *,
    close_logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    pos_weight: float | torch.Tensor | None = None,
    gamma: float = 1.5,
    focal: bool = True,
) -> torch.Tensor:
    if close_logits.ndim != 3:
        raise ValueError(f"close_logits must have shape [B,T,4], got {tuple(close_logits.shape)}")
    if tuple(labels.shape) != tuple(close_logits.shape) or tuple(mask.shape) != tuple(close_logits.shape):
        raise ValueError("labels and mask must match close_logits shape")
    mask_bool = mask.to(device=close_logits.device, dtype=torch.bool)
    if not bool(mask_bool.any()):
        return close_logits.sum() * 0.0
    labels_f = labels.to(device=close_logits.device, dtype=close_logits.dtype)
    resolved_pos_weight = (
        close_pos_weight(labels=labels, mask=mask).to(device=close_logits.device, dtype=close_logits.dtype)
        if pos_weight is None
        else torch.as_tensor(pos_weight, device=close_logits.device, dtype=close_logits.dtype)
    )
    bce = F.binary_cross_entropy_with_logits(
        close_logits,
        labels_f,
        pos_weight=resolved_pos_weight,
        reduction="none",
    )
    if focal and float(gamma) > 0.0:
        prob = torch.sigmoid(close_logits)
        p_t = torch.where(labels_f > 0.5, prob, 1.0 - prob)
        bce = bce * (1.0 - p_t).clamp_min(0.0).pow(float(gamma))
    return bce[mask_bool].mean()


def close_pos_weight(
    *,
    labels: torch.Tensor,
    mask: torch.Tensor,
    min_weight: float = 1.0,
    max_weight: float = 20.0,
) -> torch.Tensor:
    if tuple(labels.shape) != tuple(mask.shape):
        raise ValueError(f"labels and mask must have the same shape, got {tuple(labels.shape)} and {tuple(mask.shape)}")
    mask_bool = mask.to(dtype=torch.bool)
    labels_bool = labels.to(dtype=torch.bool)
    positives = (labels_bool & mask_bool).sum().to(dtype=torch.float32)
    negatives = ((~labels_bool) & mask_bool).sum().to(dtype=torch.float32)
    if float(positives.item()) <= 0.0:
        return torch.tensor(float(min_weight), dtype=torch.float32, device=labels.device)
    return (negatives / positives).clamp(min=float(min_weight), max=float(max_weight))


def adapter_bias_regularization(*biases: torch.Tensor | None, mask: torch.Tensor | None = None) -> torch.Tensor:
    present = [bias for bias in biases if isinstance(bias, torch.Tensor)]
    if not present:
        raise ValueError("at least one adapter bias tensor is required")
    total = present[0].sum() * 0.0
    for bias in present:
        assert isinstance(bias, torch.Tensor)
        if mask is None:
            total = total + bias.square().mean()
        else:
            if tuple(mask.shape) != tuple(bias.shape[:2]):
                raise ValueError(f"mask must have shape {tuple(bias.shape[:2])}, got {tuple(mask.shape)}")
            mask_f = mask.to(device=bias.device, dtype=bias.dtype)
            while mask_f.ndim < bias.ndim:
                mask_f = mask_f.unsqueeze(-1)
            denom = mask_f.sum() * float(bias.shape[-1])
            total = total + (bias.square() * mask_f).sum() / denom.clamp_min(torch.finfo(bias.dtype).eps)
    return total


def adapter_regularization(*biases: torch.Tensor | None, mask: torch.Tensor | None = None) -> torch.Tensor:
    return adapter_bias_regularization(*biases, mask=mask)


def adapter_reg(*biases: torch.Tensor | None, mask: torch.Tensor | None = None) -> torch.Tensor:
    return adapter_bias_regularization(*biases, mask=mask)


def token_ce(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    pad_id: int | None = None,
    target_mask: torch.Tensor | None = None,
    grammar_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    return token_cross_entropy(
        logits,
        target,
        pad_id=pad_id,
        target_mask=target_mask,
        grammar_mask=grammar_mask,
    )


def _target_loss_mask(batch: Mapping[str, torch.Tensor], *, target: torch.Tensor, pad_id: int) -> torch.Tensor:
    raw_mask = batch.get("target_token_mask")
    if raw_mask is None:
        return target != int(pad_id)
    if not isinstance(raw_mask, torch.Tensor) or raw_mask.ndim != 2:
        raise ValueError("target_token_mask must have shape [B,S]")
    shifted = raw_mask[:, 1:].to(device=target.device, dtype=torch.bool)
    if tuple(shifted.shape) != tuple(target.shape):
        raise ValueError(f"target_token_mask[:, 1:] must have shape {tuple(target.shape)}")
    return shifted & (target != int(pad_id))


def _input_loss_mask(batch: Mapping[str, torch.Tensor], *, steps: int, device: torch.device) -> torch.Tensor:
    raw_mask = batch.get("target_token_mask")
    if raw_mask is None:
        return torch.ones((int(_require_batch_tensor(batch, "target_tokens").shape[0]), steps), dtype=torch.bool, device=device)
    if not isinstance(raw_mask, torch.Tensor) or raw_mask.ndim != 2:
        raise ValueError("target_token_mask must have shape [B,S]")
    shifted = raw_mask[:, :-1].to(device=device, dtype=torch.bool)
    shifted = shifted & raw_mask[:, 1:].to(device=device, dtype=torch.bool)
    if int(shifted.shape[1]) != int(steps):
        raise ValueError(f"target_token_mask[:, :-1] must have {steps} steps")
    return shifted


def _require_tensor_attr(output: Any, name: str) -> torch.Tensor:
    value = getattr(output, name, None)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"output.{name} must be a torch.Tensor")
    return value


def _require_batch_tensor(batch: Mapping[str, torch.Tensor], name: str) -> torch.Tensor:
    value = batch.get(name)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"batch[{name!r}] must be a torch.Tensor")
    return value


def _record_scalar(metrics: dict[str, float], key: str, value: torch.Tensor) -> None:
    metrics[key] = float(value.detach().cpu())


def _validate_config(config: MapperV1LossConfig) -> None:
    for name in (
        "lambda_density",
        "lambda_ln_close",
        "lambda_adapter_reg",
        "ln_close_pos_weight",
        "ln_close_focal_gamma",
    ):
        _require_finite_number(getattr(config, name), name)
    if config.lambda_density < 0.0:
        raise ValueError("lambda_density must be non-negative")
    if config.lambda_ln_close < 0.0:
        raise ValueError("lambda_ln_close must be non-negative")
    if config.lambda_adapter_reg < 0.0:
        raise ValueError("lambda_adapter_reg must be non-negative")
    if config.ln_close_pos_weight <= 0.0:
        raise ValueError("ln_close_pos_weight must be positive")
    if config.ln_close_focal_gamma < 0.0:
        raise ValueError("ln_close_focal_gamma must be non-negative")


def _require_finite_number(value: float, name: str) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite numeric")
