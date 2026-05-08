from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Sequence

import torch

from .density_calibration import (
    density_metrics,
    fit_monotonic_affine_calibration,
    scatter_tokenized_gold_onset_mass,
    smooth_density_mass,
)
from .grammar import valid_token_mask
from .replay import ReplayError, initial_replay_state, replay_tokens, transition_replay_state
from .tokenizer import MAPPER_DENSITY_FRAMES, TokenizedMapperWindow
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
    invalid_event_count: int
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


@dataclass(frozen=True)
class PhaseAAuditGateDecision:
    status: str
    tokenizer_status: str
    grammar_status: str
    failure_reasons: list[str]
    open_mask_nonzero_before_eos_count: int
    invalid_time_delta_count: int
    invalid_event_count: int
    noncanonical_time_shift_count: int
    grammar_violation_count: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def audit_tokenized_windows(
    windows: Sequence[TokenizedMapperWindow],
    *,
    vocab: MapperV1Vocab,
    filter_report: object,
) -> TokenizerAuditReport:
    filter_payload = _filter_report_payload(filter_report)
    num_total_windows = _filter_report_int(filter_payload, "num_total_windows")
    num_eligible_windows = _filter_report_int(filter_payload, "num_mapper_eligible_windows")
    num_dropped_cross_window_ln_windows = _filter_report_int(filter_payload, "num_dropped_cross_window_ln_windows")
    if num_eligible_windows != len(windows):
        raise ValueError(
            "filter_report num_mapper_eligible_windows must match audited windows: "
            f"{num_eligible_windows} != {len(windows)}",
        )

    lengths = [window.seq_len for window in windows]
    event_ids = {
        token_id
        for window in windows
        for token_id in window.target_ids
        if vocab.is_event_token(int(token_id))
    }
    ts_counter: Counter[str] = Counter()
    tokenizer_counts = _TokenizerTokenCounts()
    for window in windows:
        for token_id in window.target_ids:
            if vocab.is_time_shift_token(int(token_id)):
                ts_counter[vocab.token_name(int(token_id))] += 1
        tokenizer_counts += _audit_token_sequence(window, vocab=vocab)

    return TokenizerAuditReport(
        num_windows=num_total_windows,
        num_eligible_windows=num_eligible_windows,
        num_dropped_cross_window_ln_windows=num_dropped_cross_window_ln_windows,
        max_seq_len=max(lengths, default=0),
        mean_seq_len=float(sum(lengths) / len(lengths)) if lengths else 0.0,
        p95_seq_len=_percentile_int(lengths, 0.95),
        p99_seq_len=_percentile_int(lengths, 0.99),
        event_vocab_coverage=len(event_ids),
        open_mask_nonzero_before_eos_count=tokenizer_counts.open_mask_nonzero_before_eos_count,
        invalid_time_delta_count=tokenizer_counts.invalid_time_delta_count,
        noncanonical_time_shift_count=tokenizer_counts.noncanonical_time_shift_count,
        invalid_event_count=tokenizer_counts.invalid_event_count,
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
    filter_report: object,
    density_prediction: torch.Tensor | None = None,
    density_target: torch.Tensor | None = None,
    density_confidence: torch.Tensor | None = None,
    calibration: object | None = None,
) -> dict[str, object]:
    filter_payload = _filter_report_payload(filter_report)
    tokenizer = audit_tokenized_windows(windows, vocab=vocab, filter_report=filter_payload).to_dict()
    grammar = audit_grammar_replay(windows, vocab=vocab).to_dict()
    ln_close = audit_ln_close_imbalance(windows, vocab=vocab).to_dict()
    density: dict[str, object] = {
        "gold_mass_to_density_mae": math.nan,
        "gold_mass_to_density_corr": math.nan,
        "density_target_missing_rate": math.nan,
        "density_confidence_distribution": {},
    }
    confidence_for_density = _density_confidence_or_ones(density_target, density_confidence)
    gold_mass: torch.Tensor | None = None
    resolved_calibration = calibration
    if density_target is not None:
        gold_mass = _gold_onset_mass_batch(windows, vocab=vocab)
        _validate_density_frame_count(
            gold_mass,
            density_target=density_target,
            density_confidence=confidence_for_density,
        )
        if resolved_calibration is None:
            resolved_calibration = fit_monotonic_affine_calibration(
                smooth_density_mass(gold_mass),
                density_target,
                confidence_for_density,
            )

    if density_prediction is not None and density_target is not None:
        density_metrics_report = audit_density_prediction(density_prediction, density_target, confidence_for_density)
        density.update(density_metrics_report)
    if gold_mass is not None and density_target is not None and hasattr(resolved_calibration, "predict"):
        gold_prediction = resolved_calibration.predict(gold_mass)
        gold_metrics = density_metrics(gold_prediction, density_target, confidence_for_density)
        if density_prediction is None:
            density.update(audit_density_prediction(gold_prediction, density_target, confidence_for_density))
        density["gold_mass_to_density_mae"] = gold_metrics["density_frame_mae"]
        density["gold_mass_to_density_corr"] = gold_metrics["density_pearson_corr"]

    calibration_payload = resolved_calibration.to_dict() if hasattr(resolved_calibration, "to_dict") else {}
    gate_decision = build_phase_a_gate_decision(tokenizer=tokenizer, grammar=grammar).to_dict()
    report = {
        "window_filter": filter_payload,
        "tokenizer": tokenizer,
        "grammar": grammar,
        "density": density,
        "ln_close": ln_close,
        "density_calibration": calibration_payload,
        "gate_decision": gate_decision,
    }
    return report


def _gold_onset_mass_batch(
    windows: Sequence[TokenizedMapperWindow],
    *,
    vocab: MapperV1Vocab,
) -> torch.Tensor:
    if not windows:
        return torch.zeros((0, MAPPER_DENSITY_FRAMES), dtype=torch.float32)
    return torch.stack([scatter_tokenized_gold_onset_mass(window, vocab=vocab) for window in windows])


def _density_confidence_or_ones(
    density_target: torch.Tensor | None,
    density_confidence: torch.Tensor | None,
) -> torch.Tensor | None:
    if density_target is None:
        return None
    if density_confidence is not None:
        return density_confidence
    return torch.ones_like(density_target, dtype=torch.float32)


def _validate_density_frame_count(
    gold_mass: torch.Tensor,
    *,
    density_target: torch.Tensor,
    density_confidence: torch.Tensor | None,
) -> None:
    expected = int(gold_mass.numel())
    target_count = int(density_target.detach().numel())
    if target_count != expected:
        raise ValueError(f"density_target must contain {expected} gold density frames, got {target_count}")
    if density_confidence is None:
        return
    confidence_count = int(density_confidence.detach().numel())
    if confidence_count != expected:
        raise ValueError(f"density_confidence must contain {expected} gold density frames, got {confidence_count}")


def _percentile_int(values: Sequence[int], percentile: float) -> int:
    if not values:
        return 0
    sorted_values = sorted(values)
    index = min(len(sorted_values) - 1, max(0, math.ceil(percentile * len(sorted_values)) - 1))
    return int(sorted_values[index])


def build_phase_a_gate_decision(
    *,
    tokenizer: TokenizerAuditReport | Mapping[str, object],
    grammar: GrammarAuditReport | Mapping[str, object],
) -> PhaseAAuditGateDecision:
    tokenizer_payload = tokenizer.to_dict() if isinstance(tokenizer, TokenizerAuditReport) else tokenizer
    grammar_payload = grammar.to_dict() if isinstance(grammar, GrammarAuditReport) else grammar
    open_mask_count = _report_int(tokenizer_payload, "open_mask_nonzero_before_eos_count")
    invalid_time_delta_count = _report_int(tokenizer_payload, "invalid_time_delta_count")
    invalid_event_count = _report_int(tokenizer_payload, "invalid_event_count")
    noncanonical_time_shift_count = _report_int(tokenizer_payload, "noncanonical_time_shift_count")
    grammar_violation_count = _report_int(grammar_payload, "violation_count")

    tokenizer_failures: list[str] = []
    if open_mask_count > 0:
        tokenizer_failures.append("open_mask_nonzero_before_eos_count > 0")
    if invalid_time_delta_count > 0:
        tokenizer_failures.append("invalid_time_delta_count > 0")
    if invalid_event_count > 0:
        tokenizer_failures.append("invalid_event_count > 0")
    if noncanonical_time_shift_count > 0:
        tokenizer_failures.append("noncanonical_time_shift_count > 0")

    grammar_failures = ["grammar violation_count > 0"] if grammar_violation_count > 0 else []
    failure_reasons = tokenizer_failures + grammar_failures
    return PhaseAAuditGateDecision(
        status="PASS" if not failure_reasons else "FAIL",
        tokenizer_status="PASS" if not tokenizer_failures else "FAIL",
        grammar_status="PASS" if not grammar_failures else "FAIL",
        failure_reasons=failure_reasons,
        open_mask_nonzero_before_eos_count=open_mask_count,
        invalid_time_delta_count=invalid_time_delta_count,
        invalid_event_count=invalid_event_count,
        noncanonical_time_shift_count=noncanonical_time_shift_count,
        grammar_violation_count=grammar_violation_count,
    )


@dataclass
class _TokenizerTokenCounts:
    open_mask_nonzero_before_eos_count: int = 0
    invalid_time_delta_count: int = 0
    noncanonical_time_shift_count: int = 0
    invalid_event_count: int = 0

    def __iadd__(self, other: "_TokenizerTokenCounts") -> "_TokenizerTokenCounts":
        self.open_mask_nonzero_before_eos_count += other.open_mask_nonzero_before_eos_count
        self.invalid_time_delta_count += other.invalid_time_delta_count
        self.noncanonical_time_shift_count += other.noncanonical_time_shift_count
        self.invalid_event_count += other.invalid_event_count
        return self


def _audit_token_sequence(window: TokenizedMapperWindow, *, vocab: MapperV1Vocab) -> _TokenizerTokenCounts:
    counts = _TokenizerTokenCounts()
    state = initial_replay_state(window.write_start_ms)
    time_shift_run: list[int] = []

    def flush_time_shift_run() -> None:
        nonlocal time_shift_run
        if not time_shift_run:
            return
        total_delta_ms = sum(time_shift_run)
        try:
            canonical = vocab.decompose_time_shift_delta(total_delta_ms)
        except ValueError:
            counts.noncanonical_time_shift_count += 1
        else:
            if time_shift_run != canonical:
                counts.noncanonical_time_shift_count += 1
        time_shift_run = []

    for position, raw_token_id in enumerate(window.target_ids):
        token_id = int(raw_token_id)
        is_time_shift = vocab.is_time_shift_token(token_id)
        if is_time_shift:
            time_shift_run.append(vocab.time_shift_value(token_id))
        else:
            flush_time_shift_run()

        mask = valid_token_mask(
            position=state.position,
            current_ms=state.current_ms,
            open_mask=state.open_mask,
            write_start_ms=window.write_start_ms,
            write_end_ms=window.write_end_ms,
            vocab=vocab,
        )
        is_known_token = 0 <= token_id < vocab.size
        is_valid = is_known_token and bool(mask[token_id].item())
        if not is_valid:
            if is_time_shift:
                counts.invalid_time_delta_count += 1
            elif vocab.is_event_token(token_id) or not _is_special_token(token_id, vocab=vocab):
                counts.invalid_event_count += 1

        if token_id == vocab.eos_id and any(state.open_mask):
            counts.open_mask_nonzero_before_eos_count += 1

        try:
            state = transition_replay_state(
                state,
                token_id,
                position=position,
                vocab=vocab,
                write_start_ms=window.write_start_ms,
                write_end_ms=window.write_end_ms,
            )
        except ReplayError:
            continue

    flush_time_shift_run()
    return counts


def _is_special_token(token_id: int, *, vocab: MapperV1Vocab) -> bool:
    return token_id in {vocab.pad_id, vocab.bos_id, vocab.eos_id}


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


def _filter_report_payload(filter_report: object | None) -> dict[str, object]:
    if filter_report is None:
        raise ValueError(
            "filter_report is required for Phase A mapper audits; post-filtered windows cannot reconstruct "
            "cross-window LN exclusion counts or drop rates.",
        )
    if hasattr(filter_report, "to_dict"):
        return filter_report.to_dict()  # type: ignore[no-any-return]
    if hasattr(filter_report, "__dataclass_fields__"):
        return asdict(filter_report)
    if isinstance(filter_report, Mapping):
        return dict(filter_report)
    return dict(filter_report)  # type: ignore[arg-type]


def _filter_report_int(filter_payload: Mapping[str, object], key: str) -> int:
    return _report_int(filter_payload, key, report_name="filter_report")


def _report_int(report_payload: Mapping[str, object], key: str, *, report_name: str = "audit report") -> int:
    try:
        value = report_payload[key]
    except KeyError as exc:
        raise ValueError(f"{report_name} is missing required field {key!r}") from exc
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{report_name} field {key!r} must be an integer-compatible value, got {value!r}") from exc
