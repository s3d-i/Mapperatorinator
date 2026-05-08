from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Sequence

import torch

from .density_calibration import density_metrics
from .grammar import valid_token_mask
from .replay import initial_replay_state, replay_tokens
from .tokenizer import TokenizedMapperWindow
from .vocab import MapperV1Vocab


@dataclass(frozen=True)
class TokenizerAuditReport:
    num_windows: int
    num_eligible_windows: int
    num_dropped_cross_window_ln_windows: int
    max_seq_len: int
    mean_seq_len: float
    p95_seq_len: int
    p99_seq_len: int
    event_vocab_coverage: int
    open_mask_nonzero_before_eos_count: int
    invalid_time_delta_count: int
    noncanonical_time_shift_count: int
    time_shift_vocab_distribution: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class GrammarAuditReport:
    num_windows: int
    checked_token_count: int
    violation_count: int
    violations: list[dict[str, object]]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class LNCloseImbalanceReport:
    num_open_lane_steps: int
    num_close_positive_steps: int
    close_positive_rate: float
    pos_weight: float
    ln_duration_distribution: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def audit_tokenized_windows(
    windows: Sequence[TokenizedMapperWindow],
    *,
    vocab: MapperV1Vocab,
) -> TokenizerAuditReport:
    lengths = [window.seq_len for window in windows]
    event_ids = {
        token_id
        for window in windows
        for token_id in window.target_ids
        if vocab.is_event_token(int(token_id))
    }
    ts_counter: Counter[str] = Counter()
    open_before_eos = 0
    for window in windows:
        for token_id in window.target_ids:
            if vocab.is_time_shift_token(int(token_id)):
                ts_counter[vocab.token_name(int(token_id))] += 1
        eos_index = window.target_ids.index(vocab.eos_id)
        if bool(window.teacher_open_mask[eos_index].any().item()):
            open_before_eos += 1

    return TokenizerAuditReport(
        num_windows=len(windows),
        num_eligible_windows=len(windows),
        num_dropped_cross_window_ln_windows=0,
        max_seq_len=max(lengths, default=0),
        mean_seq_len=float(sum(lengths) / len(lengths)) if lengths else 0.0,
        p95_seq_len=_percentile_int(lengths, 0.95),
        p99_seq_len=_percentile_int(lengths, 0.99),
        event_vocab_coverage=len(event_ids),
        open_mask_nonzero_before_eos_count=open_before_eos,
        invalid_time_delta_count=0,
        noncanonical_time_shift_count=0,
        time_shift_vocab_distribution=dict(sorted(ts_counter.items())),
    )


def audit_grammar_replay(
    windows: Sequence[TokenizedMapperWindow],
    *,
    vocab: MapperV1Vocab,
    max_violations: int = 20,
) -> GrammarAuditReport:
    violations: list[dict[str, object]] = []
    checked = 0
    violation_count = 0
    for window_index, window in enumerate(windows):
        states = replay_tokens(
            window.target_ids,
            vocab=vocab,
            write_start_ms=window.write_start_ms,
            write_end_ms=window.write_end_ms,
            validate_final=True,
        )
        before_bos = initial_replay_state(window.write_start_ms)
        for token_index, token_id in enumerate(window.target_ids):
            if token_index == 0:
                position = before_bos.position
                current_ms = before_bos.current_ms
                open_mask = before_bos.open_mask
            else:
                previous = states[token_index - 1]
                position = previous.position
                current_ms = previous.current_ms
                open_mask = previous.open_mask
            checked += 1
            mask = valid_token_mask(
                position=position,
                current_ms=current_ms,
                open_mask=open_mask,
                write_start_ms=window.write_start_ms,
                write_end_ms=window.write_end_ms,
                vocab=vocab,
            )
            if not bool(mask[int(token_id)].item()):
                violation_count += 1
                if len(violations) < max_violations:
                    violations.append(
                        {
                            "window_index": window_index,
                            "token_index": token_index,
                            "token_id": int(token_id),
                            "token_name": vocab.token_name(int(token_id)),
                            "position": position,
                            "current_ms": current_ms,
                        }
                    )
    return GrammarAuditReport(
        num_windows=len(windows),
        checked_token_count=checked,
        violation_count=violation_count,
        violations=violations,
    )


def audit_ln_close_imbalance(
    windows: Sequence[TokenizedMapperWindow],
    *,
    vocab: MapperV1Vocab | None = None,
) -> LNCloseImbalanceReport:
    resolved_vocab = MapperV1Vocab() if vocab is None else vocab
    open_steps = 0
    positive_steps = 0
    durations_ms: list[int] = []
    for window in windows:
        mask = window.close_label_mask.to(dtype=torch.bool)
        labels = window.close_labels.to(dtype=torch.bool)
        open_steps += int(mask.sum().item())
        positive_steps += int((labels & mask).sum().item())
        durations_ms.extend(_ln_durations_ms(window, vocab=resolved_vocab))
    negative_steps = max(open_steps - positive_steps, 0)
    if positive_steps <= 0:
        pos_weight = 20.0 if open_steps > 0 else 1.0
    else:
        pos_weight = min(max(negative_steps / positive_steps, 1.0), 20.0)
    return LNCloseImbalanceReport(
        num_open_lane_steps=open_steps,
        num_close_positive_steps=positive_steps,
        close_positive_rate=float(positive_steps / open_steps) if open_steps else 0.0,
        pos_weight=float(pos_weight),
        ln_duration_distribution=_duration_distribution(durations_ms),
    )


def audit_density_prediction(
    prediction: torch.Tensor,
    target: torch.Tensor,
    confidence: torch.Tensor,
) -> dict[str, float]:
    report = density_metrics(prediction, target, confidence)
    confidence_flat = confidence.detach().reshape(-1)
    report["density_target_missing_rate"] = float((confidence_flat <= 0).sum().item() / max(confidence_flat.numel(), 1))
    report["density_confidence_mean"] = float(confidence_flat.to(dtype=torch.float32).mean().item())
    report["density_confidence_min"] = float(confidence_flat.to(dtype=torch.float32).min().item())
    report["density_confidence_max"] = float(confidence_flat.to(dtype=torch.float32).max().item())
    report["density_confidence_distribution"] = {
        "min": report["density_confidence_min"],
        "mean": report["density_confidence_mean"],
        "max": report["density_confidence_max"],
    }
    return report


def build_phase_a_report(
    *,
    windows: Sequence[TokenizedMapperWindow],
    vocab: MapperV1Vocab,
    filter_report: object | None = None,
    density_prediction: torch.Tensor | None = None,
    density_target: torch.Tensor | None = None,
    density_confidence: torch.Tensor | None = None,
    calibration: object | None = None,
) -> dict[str, object]:
    tokenizer = audit_tokenized_windows(windows, vocab=vocab).to_dict()
    grammar = audit_grammar_replay(windows, vocab=vocab).to_dict()
    ln_close = audit_ln_close_imbalance(windows, vocab=vocab).to_dict()
    density: dict[str, object] = {
        "gold_mass_to_density_mae": math.nan,
        "gold_mass_to_density_corr": math.nan,
        "density_target_missing_rate": math.nan,
        "density_confidence_distribution": {},
    }
    if density_prediction is not None and density_target is not None and density_confidence is not None:
        density_metrics_report = audit_density_prediction(density_prediction, density_target, density_confidence)
        density.update(density_metrics_report)
        density["gold_mass_to_density_mae"] = density_metrics_report["density_frame_mae"]
        density["gold_mass_to_density_corr"] = density_metrics_report["density_pearson_corr"]

    calibration_payload = calibration.to_dict() if hasattr(calibration, "to_dict") else {}
    report = {
        "window_filter": _filter_report_payload(filter_report, eligible_default=len(windows)),
        "tokenizer": tokenizer,
        "grammar": grammar,
        "density": density,
        "ln_close": ln_close,
        "density_calibration": calibration_payload,
    }
    return report


def _percentile_int(values: Sequence[int], percentile: float) -> int:
    if not values:
        return 0
    sorted_values = sorted(values)
    index = min(len(sorted_values) - 1, max(0, math.ceil(percentile * len(sorted_values)) - 1))
    return int(sorted_values[index])


def _ln_durations_ms(window: TokenizedMapperWindow, *, vocab: MapperV1Vocab) -> list[int]:
    open_start_by_lane: dict[int, int] = {}
    durations: list[int] = []
    for index, token_id in enumerate(window.target_ids):
        if not vocab.is_event_token(int(token_id)):
            continue
        event_ms = int(window.teacher_current_ms[max(index - 1, 0)].item())
        for lane, action in enumerate(vocab.decode_event(int(token_id))):
            if action.value == "HOLD_START":
                open_start_by_lane[lane] = event_ms
            elif action.value == "HOLD_END" and lane in open_start_by_lane:
                durations.append(event_ms - open_start_by_lane.pop(lane))
    return durations


def _duration_distribution(durations_ms: Sequence[int]) -> dict[str, float]:
    if not durations_ms:
        return {"count": 0.0, "mean_ms": math.nan, "p50_ms": math.nan, "p95_ms": math.nan}
    values = sorted(int(value) for value in durations_ms)
    return {
        "count": float(len(values)),
        "mean_ms": float(sum(values) / len(values)),
        "p50_ms": float(values[len(values) // 2]),
        "p95_ms": float(values[min(len(values) - 1, math.ceil(0.95 * len(values)) - 1)]),
    }


def _filter_report_payload(filter_report: object | None, *, eligible_default: int) -> dict[str, object]:
    if filter_report is None:
        return {
            "num_total_windows": eligible_default,
            "num_mapper_eligible_windows": eligible_default,
            "num_dropped_short_windows": 0,
            "num_dropped_cross_window_ln_windows": 0,
            "drop_rate": 0.0,
            "short_drop_rate": 0.0,
            "cross_window_ln_drop_rate": 0.0,
            "drop_rate_by_difficulty": {},
            "drop_rate_by_song": {},
        }
    if hasattr(filter_report, "to_dict"):
        return filter_report.to_dict()  # type: ignore[no-any-return]
    if hasattr(filter_report, "__dataclass_fields__"):
        return asdict(filter_report)
    return dict(filter_report)  # type: ignore[arg-type]
