import unittest
from types import SimpleNamespace

import torch

from train.stage_2.model_mapper_v1.audits import (
    audit_boundary_tokens,
    audit_carry_windows,
    audit_grammar_replay,
    audit_ln_close_imbalance,
    audit_tokenized_windows,
    build_phase_a_gate_decision,
    build_phase_a_report,
)
from train.stage_2.model_mapper_v1.generation import (
    LNCarryState,
    grammar_constrained_window_generation,
    replay_fragment_tokens,
    short_rollout_recovery_ce,
)
from train.stage_2.model_mapper_v1.vocab import LaneAction, MapperV1Vocab


def _actions(*actions: LaneAction) -> tuple[LaneAction, ...]:
    padded = list(actions)
    while len(padded) < 4:
        padded.append(LaneAction.NONE)
    return tuple(padded)


def _filter_report(*, eligible_windows: int, dropped_cross_window_ln_windows: int = 0) -> dict[str, object]:
    total_windows = eligible_windows + dropped_cross_window_ln_windows
    return {
        "num_total_windows": total_windows,
        "num_mapper_eligible_windows": eligible_windows,
        "num_dropped_short_windows": 0,
        "num_dropped_cross_window_ln_windows": dropped_cross_window_ln_windows,
        "num_dropped_unsupported_action_windows": 0,
        "drop_rate": float(dropped_cross_window_ln_windows / total_windows) if total_windows else 0.0,
        "short_drop_rate": 0.0,
        "cross_window_ln_drop_rate": float(dropped_cross_window_ln_windows / total_windows) if total_windows else 0.0,
        "unsupported_action_drop_rate": 0.0,
        "drop_rate_by_difficulty": {},
        "drop_rate_by_song": {},
    }


class MapperV1AuditTests(unittest.TestCase):
    def test_carry_aware_audits_report_zero_grammar_violations_and_ln_imbalance(self) -> None:
        vocab = MapperV1Vocab()
        windows = [
            _window(vocab, _ts(vocab, 8000), write_start_ms=0, write_end_ms=8000),
            _hold_window(vocab),
        ]

        tokenizer_report = audit_tokenized_windows(
            windows,
            vocab=vocab,
            filter_report=_filter_report(eligible_windows=2),
        )
        carry_report = audit_carry_windows(windows, vocab=vocab)
        boundary_report = audit_boundary_tokens(windows, vocab=vocab)
        grammar_report = audit_grammar_replay(windows, vocab=vocab)
        close_report = audit_ln_close_imbalance(windows)

        self.assertEqual(tokenizer_report.num_windows, 2)
        self.assertEqual(tokenizer_report.invalid_time_delta_count, 0)
        self.assertEqual(tokenizer_report.noncanonical_time_shift_count, 0)
        self.assertEqual(tokenizer_report.invalid_event_count, 0)
        self.assertEqual(carry_report.terminal_state_mismatch_count, 0)
        self.assertEqual(boundary_report.non_initial_window_bos_count, 0)
        self.assertEqual(boundary_report.non_final_window_eos_count, 0)
        self.assertEqual(grammar_report.violation_count, 0)
        self.assertGreater(close_report.num_open_lane_steps, 0)
        self.assertEqual(close_report.num_close_positive_steps, 1)

    def test_phase_a_report_contains_v1_carry_generation_and_recovery_sections(self) -> None:
        vocab = MapperV1Vocab()
        windows = [_window(vocab, _ts(vocab, 8000), write_start_ms=0, write_end_ms=8000)]

        report = build_phase_a_report(
            windows=windows,
            vocab=vocab,
            filter_report=_filter_report(eligible_windows=1),
        )

        self.assertEqual(
            set(report),
            {
                "window_filter",
                "carry",
                "boundary_tokens",
                "tokenizer",
                "grammar",
                "generation",
                "recovery",
                "density",
                "ln_close",
                "density_calibration",
                "gate_decision",
            },
        )
        self.assertIn("carry_reconstruction_failure_count", report["carry"])
        self.assertIn("non_final_window_eos_count", report["boundary_tokens"])
        self.assertIn("generated_carry_out_match_rate", report["generation"])
        self.assertIn("recovery_batch_valid_fraction", report["recovery"])
        self.assertEqual(report["gate_decision"]["status"], "PASS")

    def test_phase_a_report_requires_filter_report_to_avoid_fabricated_drop_rates(self) -> None:
        vocab = MapperV1Vocab()
        windows = [_window(vocab, _ts(vocab, 8000), write_start_ms=0, write_end_ms=8000)]

        with self.assertRaisesRegex(ValueError, "filter_report is required"):
            build_phase_a_report(windows=windows, vocab=vocab, filter_report=None)

    def test_boundary_audit_flags_legacy_per_window_eos(self) -> None:
        vocab = MapperV1Vocab()
        legacy = _window(
            vocab,
            [vocab.bos_id, *_ts(vocab, 8000), vocab.eos_id],
            write_start_ms=0,
            write_end_ms=8000,
            is_full_chart_start=True,
            is_full_chart_end=False,
            raw_tokens=True,
        )

        boundary = audit_boundary_tokens([legacy], vocab=vocab)
        decision = build_phase_a_gate_decision(
            tokenizer={
                "open_mask_nonzero_before_eos_count": 0,
                "invalid_time_delta_count": 0,
                "invalid_event_count": 0,
                "noncanonical_time_shift_count": 0,
            },
            grammar={"violation_count": 0},
            boundary=boundary,
        )

        self.assertEqual(boundary.window_start_bos_count, 1)
        self.assertEqual(boundary.window_end_eos_count, 1)
        self.assertEqual(boundary.non_final_window_eos_count, 1)
        self.assertEqual(decision.boundary_status, "FAIL")
        self.assertIn("non_final_window_eos_count > 0", decision.failure_reasons)

    def test_tokenizer_audit_counts_invalid_and_noncanonical_tokens(self) -> None:
        vocab = MapperV1Vocab()
        invalid_hold_end_event = vocab.encode_event(_actions(LaneAction.HOLD_END))
        malformed = _window(
            vocab,
            [
                vocab.time_shift_token_id(100),
                vocab.time_shift_token_id(200),
                invalid_hold_end_event,
                vocab.time_shift_token_id(4000),
                vocab.time_shift_token_id(4000),
                vocab.time_shift_token_id(10),
            ],
            write_start_ms=0,
            write_end_ms=8000,
            allow_invalid=True,
        )

        report = audit_tokenized_windows(
            [malformed],
            vocab=vocab,
            filter_report=_filter_report(eligible_windows=1),
        )

        self.assertEqual(report.noncanonical_time_shift_count, 1)
        self.assertEqual(report.invalid_event_count, 1)
        self.assertEqual(report.invalid_time_delta_count, 1)

    def test_phase_a_gate_fails_design_hard_fail_counters(self) -> None:
        decision = build_phase_a_gate_decision(
            tokenizer={
                "open_mask_nonzero_before_eos_count": 0,
                "invalid_time_delta_count": 1,
                "invalid_event_count": 2,
                "noncanonical_time_shift_count": 3,
            },
            grammar={"violation_count": 4},
            carry={"carry_reconstruction_failure_count": 5, "terminal_state_mismatch_count": 6},
            boundary={"non_initial_window_bos_count": 7, "non_final_window_eos_count": 8},
            generation={
                "generated_invalid_token_count": 9,
                "generated_dead_end_count": 10,
                "generated_carry_out_mismatch_count": 11,
            },
        )

        self.assertEqual(decision.status, "FAIL")
        self.assertEqual(decision.tokenizer_status, "FAIL")
        self.assertEqual(decision.grammar_status, "FAIL")
        self.assertEqual(decision.carry_status, "FAIL")
        self.assertEqual(decision.boundary_status, "FAIL")
        self.assertEqual(decision.generation_status, "FAIL")
        self.assertIn("terminal_state_mismatch_count > 0", decision.failure_reasons)
        self.assertIn("generated_carry_out_mismatch_count > 0", decision.failure_reasons)

    def test_generation_starts_from_carry_in_and_recovery_ce_uses_only_strict_matches(self) -> None:
        vocab = MapperV1Vocab()
        carry_in = LNCarryState.from_open_starts(1000, [500, None, None, None])
        carry_out = LNCarryState.from_open_starts(9000, [500, None, None, None])

        generated = grammar_constrained_window_generation(
            vocab=vocab,
            write_start_ms=1000,
            write_end_ms=9000,
            ln_carry_in=carry_in,
            ln_carry_out=carry_out,
            left_context_tokens=[vocab.time_shift_token_id(500)],
            max_tokens=4,
        )

        self.assertTrue(generated.completed)
        self.assertEqual(generated.states_before[0], carry_in)
        self.assertNotIn(vocab.eos_id, generated.tokens)
        self.assertEqual(generated.terminal_state, carry_out)

        gold_states = [carry_in, LNCarryState.closed(1000)]
        logits = torch.zeros((2, vocab.size), dtype=torch.float32)
        logits[0, vocab.time_shift_token_id(4000)] = 5.0
        logits[1, vocab.time_shift_token_id(10)] = 5.0
        recovery = short_rollout_recovery_ce(
            logits=logits,
            generated_states=[carry_in, LNCarryState.closed(2000)],
            gold_states=gold_states,
            gold_target_tokens=[vocab.time_shift_token_id(4000), vocab.time_shift_token_id(20)],
        )

        self.assertEqual(recovery.matched_count, 1)
        self.assertEqual(recovery.mismatch_reasons["current_ms_mismatch"], 1)
        self.assertGreater(float(recovery.loss.item()), 0.0)

    def test_generation_final_chart_emits_eos_after_carry_completion(self) -> None:
        vocab = MapperV1Vocab()

        generated = grammar_constrained_window_generation(
            vocab=vocab,
            write_start_ms=0,
            write_end_ms=8000,
            ln_carry_in=LNCarryState.closed(0),
            ln_carry_out=LNCarryState.closed(8000),
            is_full_chart_start=True,
            is_full_chart_end=True,
            max_tokens=4,
        )

        self.assertTrue(generated.completed)
        self.assertEqual(generated.tokens[-1], vocab.eos_id)
        self.assertEqual(generated.states_before[-1], LNCarryState.closed(8000))

    def test_carry_audit_accepts_tensor_backed_window_dict(self) -> None:
        vocab = MapperV1Vocab()
        window = _window(vocab, _ts(vocab, 8000), write_start_ms=0, write_end_ms=8000)
        tensor_window = {
            "write_start_ms": torch.tensor(window.write_start_ms, dtype=torch.long),
            "write_end_ms": torch.tensor(window.write_end_ms, dtype=torch.long),
            "target_fragment_tokens": torch.tensor(window.target_fragment_ids, dtype=torch.long),
            "target_fragment_states": {
                "current_ms": window.target_fragment_current_ms,
            },
            "ln_carry_in": {
                "current_ms": torch.tensor(window.ln_carry_in.current_ms, dtype=torch.long),
                "open_mask": torch.tensor(window.ln_carry_in.open_mask, dtype=torch.bool),
                "open_start_ms": torch.full((4,), -1, dtype=torch.long),
                "open_age_ms": torch.tensor(window.ln_carry_in.open_age_ms, dtype=torch.long),
            },
            "ln_carry_out": {
                "current_ms": torch.tensor(window.ln_carry_out.current_ms, dtype=torch.long),
                "open_mask": torch.tensor(window.ln_carry_out.open_mask, dtype=torch.bool),
                "open_start_ms": torch.full((4,), -1, dtype=torch.long),
                "open_age_ms": torch.tensor(window.ln_carry_out.open_age_ms, dtype=torch.long),
            },
        }

        boundary = audit_boundary_tokens([tensor_window], vocab=vocab)
        carry = audit_carry_windows([tensor_window], vocab=vocab)

        self.assertEqual(boundary.non_initial_window_bos_count, 0)
        self.assertEqual(carry.terminal_state_mismatch_count, 0)


def _ts(vocab: MapperV1Vocab, delta_ms: int) -> list[int]:
    return [vocab.time_shift_token_id(value) for value in vocab.decompose_time_shift_delta(delta_ms)]


def _hold_window(vocab: MapperV1Vocab):
    tokens = [
        *_ts(vocab, 1000),
        vocab.encode_event(_actions(LaneAction.HOLD_START)),
        *_ts(vocab, 400),
        vocab.encode_event(_actions(LaneAction.HOLD_END)),
        *_ts(vocab, 6600),
    ]
    return _window(vocab, tokens, write_start_ms=0, write_end_ms=8000)


def _window(
    vocab: MapperV1Vocab,
    tokens: list[int],
    *,
    write_start_ms: int,
    write_end_ms: int,
    ln_carry_in: LNCarryState | None = None,
    ln_carry_out: LNCarryState | None = None,
    is_full_chart_start: bool = False,
    is_full_chart_end: bool = False,
    raw_tokens: bool = False,
    allow_invalid: bool = False,
):
    carry_in = LNCarryState.closed(write_start_ms) if ln_carry_in is None else ln_carry_in
    carry_out = LNCarryState.closed(write_end_ms) if ln_carry_out is None else ln_carry_out
    fragment = list(tokens)
    if raw_tokens and fragment and fragment[0] == vocab.bos_id:
        fragment = fragment[1:]
    if raw_tokens and fragment and fragment[-1] == vocab.eos_id:
        fragment = fragment[:-1]
    try:
        trace = replay_fragment_tokens(
            fragment,
            vocab=vocab,
            write_start_ms=write_start_ms,
            write_end_ms=write_end_ms,
            ln_carry_in=carry_in,
            ln_carry_out=carry_out,
        )
        states_before = trace.states_before
    except ValueError:
        if not allow_invalid:
            raise
        from train.stage_2.model_mapper_v1.generation import transition_carry_state

        states_before = []
        state = carry_in
        for token_id in fragment:
            states_before.append(state)
            try:
                state = transition_carry_state(
                    state,
                    token_id,
                    vocab=vocab,
                    write_start_ms=write_start_ms,
                    write_end_ms=write_end_ms,
                )
            except ValueError:
                pass
    close_labels = torch.zeros((len(fragment), 4), dtype=torch.bool)
    close_label_mask = torch.zeros((len(fragment), 4), dtype=torch.bool)
    for index, state in enumerate(states_before):
        close_label_mask[index] = torch.tensor(state.open_mask, dtype=torch.bool)
        token_id = int(fragment[index])
        if vocab.is_event_token(token_id):
            for lane, action in enumerate(vocab.decode_event(token_id)):
                close_labels[index, lane] = state.open_mask[lane] and action == LaneAction.HOLD_END
    payload = {
        "write_start_ms": write_start_ms,
        "write_end_ms": write_end_ms,
        "ln_carry_in": carry_in,
        "ln_carry_out": carry_out,
        "is_full_chart_start": is_full_chart_start,
        "is_full_chart_end": is_full_chart_end,
        "target_fragment_ids": fragment,
        "target_fragment_current_ms": torch.tensor([state.current_ms for state in states_before], dtype=torch.long),
        "target_fragment_open_mask": torch.tensor([state.open_mask for state in states_before], dtype=torch.bool),
        "target_fragment_open_age_ms": torch.tensor([state.open_age_ms for state in states_before], dtype=torch.long),
        "close_labels": close_labels,
        "close_label_mask": close_label_mask,
        "seq_len": len(fragment),
    }
    if raw_tokens:
        payload["target_ids"] = list(tokens)
    return SimpleNamespace(**payload)


if __name__ == "__main__":
    unittest.main()
