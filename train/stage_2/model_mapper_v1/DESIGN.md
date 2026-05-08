---
pinned_commit: 2d96e6390c82cd65c7eb1eb6530be9b090e4dc72
status: implementation-ready draft
date: 2026-05-08
owner: s3d-i
module: train/stage_2/model_mapper_v1
design_revision: v1.1
depends_on:
  control_encoder: train/stage_2/model_control_demo_global
  control_config: train/stage_2/training/configs/stage2_control_demo_global_mps.yaml
window_contract: 8s mapper write window
token_contract: mapper_event_vocab_v1
time_shift_contract: canonical_relative_ts_v1
ln_contract: carry_ln_state_v1
---

# Mapper V1.1 Design

## 0. Status

This document defines the implementation contract for
`train/stage_2/model_mapper_v1`.

Status: implementation-ready draft, not frozen until the following are
implemented and audited:

1. differentiable density auxiliary loss;
2. context-aware LN close adapter;
3. write-end dead-end grammar guard;
4. density and LN-close evaluation metrics.

The mapper is a control-conditioned autoregressive event generator for
osu!mania 4K. It maps:

- 8s audio, timing, and difficulty context;
- frozen control memory;
- density-level supervision;
- local long-note state;

to a legal sequence of chart event tokens.

The mapper is not a chart oracle. Hard grammar remains the final authority for
legality.

## 1. Core Decisions

### 1.1 Window Policy

The mapper writes one independent 8s window.

Definitions:

```text
write_start_ms
write_end_ms = write_start_ms + 8000
valid chart event times are in [write_start_ms, write_end_ms)
```

The model does not carry LN state across windows in V1.

Training samples with cross-window LNs are excluded unless a future version
implements carry-in and carry-out.

This exclusion must be audited:

```text
num_total_windows
num_mapper_eligible_windows
num_dropped_cross_window_ln_windows
drop_rate
drop_rate_by_difficulty
drop_rate_by_song
```

If the cross-window drop rate is high, reported mapper quality is biased and
the model is not production-valid.

### 1.2 Grammar Policy

The decoder may be wrong.

The adapter may be wrong.

Sampling may be noisy.

The grammar must still make illegal output impossible.

Final logits are:

```text
logits_final =
    logits_base
  + logits_state_prior_adapter
  + logits_ln_close_adapter
  + hard_grammar_mask
```

Invalid tokens receive `-inf`.

No post-hoc repair is allowed in V1.

### 1.3 Control Model Policy

The existing control model is reused as a frozen conditioner.

The mapper does not train the control encoder in V1.

The control encoder provides:

```text
control_memory
density_teacher
```

The training dataset also provides:

```text
density_target
density_confidence
```

Do not collapse these names.

`density_target` is the supervised target.

`density_teacher` is the frozen control model prediction.

## 2. Inputs

### 2.1 Mapper Batch

Each mapper batch item contains:

```text
mel_context              [B, context_frames, mel_dim]
timing_context           [B, context_frames, timing_dim]
difficulty               [B, difficulty_dim]
context_padding_mask     [B, context_frames]
target_tokens            [B, seq_len]
target_token_mask        [B, seq_len]
teacher_states:
    current_ms           [B, seq_len]
    open_mask            [B, seq_len, 4]
    open_age_ms          [B, seq_len, 4]
density_target_8s        [B, 400, 1]
density_confidence_8s    [B, 400, 1]
```

The 400 density frames correspond to 20ms frames across 8s.

### 2.2 Control Encoder Output

The frozen control encoder runs on the same audio, timing, and difficulty
context.

It returns:

```text
control_memory_context   [B, context_frames, D]
density_teacher_2s       [B, 100, 1] per 2s target slice
```

For an 8s mapper window, concatenate four aligned 2s slices:

```text
control_memory_8s        [B, 400, D]
density_teacher_8s       [B, 400, 1]
```

If the control encoder internally emits a larger memory than 400 frames, the
mapper consumes only the aligned 8s write span.

### 2.3 Naming

Use these names consistently:

```text
density_target_8s      ground-truth control_v3 density_level
density_confidence_8s  ground-truth control_v3 density_confidence
density_teacher_8s     frozen control model value prediction
```

Never call `density_teacher_8s` the target.

## 3. Token Vocabulary

### 3.1 Special Tokens

```text
PAD
BOS
EOS
```

No `UNK`.

`PAD` is used only for batching.

### 3.2 EVENT Tokens

Each `EVENT` token represents one simultaneous 4-lane action tuple.

Per-lane action set:

```text
NONE
TAP
HOLD_START
HOLD_END
```

There are:

```text
4^4 = 256 raw tuples
```

The all-`NONE` tuple is not a generatable `EVENT`.

Therefore:

```text
255 non-empty EVENT tokens
```

An `EVENT` occurs at `current_ms`.

An `EVENT` does not advance time.

Chords are first-class `EVENT` tokens.

### 3.3 TIME_SHIFT Tokens

A `TIME_SHIFT` token advances the cursor by a relative amount.

Recommended vocabulary:

```text
TS_10
TS_20
TS_30
...
TS_90
TS_100
TS_200
...
TS_900
TS_1000
TS_2000
TS_3000
TS_4000
```

All target event timestamps must be aligned to 10ms.

Every positive delta must have exactly one canonical encoding.

Use greedy largest-first encoding.

Example:

```text
delta = 3270ms
=> TS_3000 TS_200 TS_70
```

### 3.4 Approximate Vocabulary Size

```text
3 special tokens
255 EVENT tokens
22 TIME_SHIFT tokens
= 280 tokens
```

Exact size depends on the final time-shift vocabulary.

## 4. Canonical Tokenizer

### 4.1 Target Construction

Given all hit objects inside the 8s write window:

1. sort by timestamp;
2. group all lane actions with the same timestamp;
3. emit canonical `TIME_SHIFT` sequence from previous timestamp to current
   timestamp;
4. emit exactly one `EVENT` token for the grouped actions;
5. after the last event, emit `TIME_SHIFT` sequence to `write_end_ms`;
6. emit `EOS`.

The target sequence always starts with `BOS`.

### 4.2 Empty Window

An empty 8s window is encoded as:

```text
BOS TS_4000 TS_4000 EOS
```

### 4.3 LN State Update During Tokenization

The tokenizer must simulate LN state.

State:

```text
open_mask[4]      bool
open_age_ms[4]    int
```

Rules:

- `TAP` requires lane closed before event and remains closed after event.
- `HOLD_START` requires lane closed before event and opens lane after event.
- `HOLD_END` requires lane open before event and closes lane after event.
- `NONE` leaves lane state unchanged.
- `TIME_SHIFT` increments `open_age_ms` for open lanes only.
- `EOS` requires all lanes closed.

Samples that require carry-in or carry-out LN state are excluded in V1.

## 5. Hard Grammar

The hard grammar consumes:

```text
previous token position
current_ms
open_mask
open_age_ms
write_start_ms
write_end_ms
```

and returns a boolean valid-token mask.

### 5.1 Special Token Rules

`PAD`:

```text
never valid during generation
```

`BOS`:

```text
valid only at position 0
invalid after position 0
```

`EOS`:

```text
valid iff:
    position > 0
    current_ms == write_end_ms
    open_mask == 0
```

### 5.2 TIME_SHIFT Rules

`TS_k` is valid iff:

```text
current_ms + k <= write_end_ms
```

Additional dead-end guard:

```text
if open_mask != 0:
    current_ms + k must be < write_end_ms
```

Rationale:

If an LN is open, allowing the cursor to move exactly to `write_end_ms` creates
a dead state because `EVENT` is no longer valid and `EOS` requires no open
lanes.

### 5.3 EVENT Rules

An `EVENT` token is valid iff:

```text
current_ms < write_end_ms
```

and the action tuple is non-empty.

For each lane:

If lane is closed:

```text
valid actions:
    NONE
    TAP
    HOLD_START
invalid:
    HOLD_END
```

If lane is open:

```text
valid actions:
    NONE
    HOLD_END
invalid:
    TAP
    HOLD_START
```

### 5.4 Optional Min-LN Guard

V1 may disable this.

If enabled:

```text
HOLD_START invalid when remaining window time < min_ln_duration_ms
```

This reduces impossible short LNs near the end of a window.

Recommended default:

```text
min_ln_duration_ms = 60
```

This is a style guard, not a legality guard.

## 6. Model Architecture

### 6.1 Frozen Control Encoder

Use the existing `ControlDemoGlobalEncoder`.

Freeze all parameters:

```text
requires_grad = False
eval mode
```

Mapper receives:

```text
control_memory_8s
density_teacher_8s
```

No gradient flows into the control encoder.

### 6.2 Mapper Decoder

Recommended baseline:

```text
d_model       = 384
num_layers    = 4
num_heads     = 8
ffn_dim       = 1536
dropout       = 0.1
max_seq_len   = audit_p99_seq_len + safety_margin
```

The decoder is causal.

It attends to:

- previous target tokens;
- `control_memory_8s`;
- difficulty embedding;
- timing position embedding.

Output:

```text
base_logits [B, seq_len, vocab_size]
decoder_hidden [B, seq_len, d_model]
```

### 6.3 Control Projection

If `control_memory_8s` dimension differs from decoder `d_model`, use:

```text
control_proj = Linear(control_dim, d_model)
```

Do not use a deep MLP unless required by audit.

### 6.4 State Feature Encoding

At each decode step, construct per-lane state features:

```text
open_bit
age_ms
age_norm = min(age_ms / age_cap_ms, 1.0)
age_bucket
lane_id_embedding
remaining_ms
remaining_norm
```

Recommended:

```text
age_cap_ms = 4000
num_age_buckets = 32
```

## 7. Dynamic Adapters

The V1.1 adapter is split into two constrained parts:

```text
StatePriorAdapter
LNCloseAdapter
```

Both adapters output structured biases.

Neither adapter may output arbitrary full-vocabulary logits.

Hard grammar remains final.

## 8. StatePriorAdapter

### 8.1 Purpose

The `StatePriorAdapter` learns soft priors such as:

- long-open LN is more likely to close;
- just-open LN is unlikely to close;
- certain lanes may have different empirical priors;
- closed lanes may prefer `TAP` vs `HOLD_START` under specific local state.

It does not decide musical timing by itself.

### 8.2 Inputs

```text
open_mask              [B, T, 4]
open_age_ms            [B, T, 4]
lane_id                [4]
remaining_ms           [B, T]
```

### 8.3 Output

```text
lane_action_bias       [B, T, 4, 4]
```

The four action channels are:

```text
NONE
TAP
HOLD_START
HOLD_END
```

### 8.4 Projection to EVENT Tokens

For each `EVENT` token `v` with lane actions `a_l`:

```text
event_bias[v] = sum_l lane_action_bias[l, a_l]
```

For non-`EVENT` tokens:

```text
state_prior_bias = 0
```

### 8.5 Initialization

Initialize near zero:

```text
final_weight std = 1e-3
final_bias = 0
adapter_scale initial = 0.05
```

Do not initialize the whole adapter to exact zero if that blocks useful early
hidden-layer learning.

Bound the final projected bias:

```text
bias = max_bias * tanh(raw_bias / max_bias)
max_bias = 3.0
```

## 9. LNCloseAdapter

### 9.1 Problem

LN close timing is not only a legality problem.

It is a timing decision.

A state-only adapter can learn duration priors, but it cannot know that the
current audio and control context indicates release now.

Therefore the LN close adapter must be context-aware.

### 9.2 Inputs

At each decode step:

```text
decoder_hidden_t          [B, T, d_model]
local_control_t           [B, T, d_model]
local_density_teacher_t   [B, T, 1]
open_mask_t               [B, T, 4]
open_age_ms_t             [B, T, 4]
remaining_ms_t            [B, T, 1]
lane_id_embedding         [4, lane_dim]
```

`local_control_t` is gathered from `control_memory_8s` by:

```text
frame_idx = floor((current_ms - write_start_ms) / 20)
```

Clamp only for safety. Invalid current times should be caught earlier.

Optionally pool local control over a small neighborhood:

```text
frames [f - 2, f - 1, f, f + 1, f + 2]
```

### 9.3 Output

The head predicts lane-level close hazards:

```text
close_logit [B, T, 4]
```

Only open lanes are meaningful.

Closed lanes are masked out for close loss.

### 9.4 Projection to Logits

For each `EVENT` token:

```text
for lane l:
    if lane is open and token action is HOLD_END:
        add +close_bias[l]
    if lane is open and token action is NONE:
        add +keep_bias[l]
```

Recommended:

```text
close_bias[l] = close_scale * tanh(close_logit[l])
keep_bias[l]  = -0.25 * close_bias[l]
```

This softly prefers closing when hazard is high but does not force closure.

### 9.5 EVENT-vs-TIME_SHIFT Gate

A close adapter that only boosts `EVENT` tokens may still lose to
`TIME_SHIFT` logits.

Therefore add a constrained scalar skip penalty when an open-lane close hazard
is high:

```text
any_close_hazard = max_l sigmoid(close_logit[l]) over open lanes
time_shift_bias = -skip_scale * any_close_hazard
```

Apply this only to `TIME_SHIFT` tokens and only when `open_mask != 0`.

Recommended:

```text
skip_scale <= 1.5
```

This is not arbitrary full-vocabulary control. It only affects the competition
between "close now" and "move time forward while an LN is open".

### 9.6 Initialization

Start conservative:

```text
close_scale initial = 0.05
skip_scale initial = 0.0
```

Ramp `skip_scale` after the close auxiliary loss starts improving.

Do not let the adapter close all LNs early.

## 10. Density Auxiliary Loss

### 10.1 Goal

Use existing `density_level` supervision to teach the mapper whether its
probability distribution places the right amount of note onset mass over time.

This loss must be differentiable.

Do not compute it from sampled tokens.

Do not compute it from gold tokens except for calibration and metrics.

### 10.2 Targets

Training target:

```text
density_target_8s [B, 400, 1]
```

Weight:

```text
density_confidence_8s [B, 400, 1]
```

Optional weak teacher:

```text
density_teacher_8s [B, 400, 1]
```

Priority order:

1. `density_target_8s`;
2. `density_teacher_8s` only as fallback or weak consistency.

### 10.3 Expected Onset Mass

Use grammar-masked logits under teacher forcing:

```text
p_t = softmax(logits_final_t)
```

For each `EVENT` token `v`, define:

```text
onset_weight(v) =
    number of lanes where action is TAP or HOLD_START
```

Default:

```text
HOLD_END contributes 0
NONE contributes 0
```

Reason:

`density_level` should model note onset density. LN release behavior is
supervised separately by LN close loss.

Then:

```text
expected_onset_mass_t =
    sum_EVENT_v p_t[v] * onset_weight(v)
```

Ignore `PAD` positions.

### 10.4 Scatter to 20ms Density Frames

For each decode step:

```text
frame_idx_t = floor((current_ms_t - write_start_ms) / 20)
```

Scatter-add:

```text
raw_mass[f] += expected_onset_mass_t
```

Only scatter when:

```text
0 <= frame_idx_t < 400
```

### 10.5 Smoothing and Calibration

The raw expected mass will not automatically be on the same scale as
`density_level`.

Use one of two approaches.

Preferred approach:

```text
reuse the same density feature transform as control_v3,
implemented differentiably
```

Fallback approach:

1. compute raw onset mass from gold tokens over the training set;
2. smooth it with a fixed kernel over 20ms frames;
3. fit a monotonic scalar calibration from smoothed mass to `density_level`;
4. freeze that calibration for mapper training.

Recommended smoothing kernel:

```text
triangular or Gaussian
radius = 5 frames
frame step = 20ms
```

Example calibrated prediction:

```text
density_pred_from_mapper =
    soft_clip(a * smooth(raw_expected_mass) + b)
```

where:

```text
soft_clip(x) = sigmoid(x)
```

or another fixed monotonic bounded mapping.

The calibration parameters must be logged.

### 10.6 Loss

Primary density auxiliary:

```text
L_density_target =
    weighted_smooth_l1(
        density_pred_from_mapper,
        density_target_8s,
        weight = density_confidence_8s
    )
```

Add a window-level count consistency term:

```text
L_density_window =
    smooth_l1(
        mean_f density_pred_from_mapper[f],
        mean_f density_target_8s[f]
    )
```

Total density loss:

```text
L_density =
    L_density_target
  + 0.25 * L_density_window
```

Optional weak teacher consistency:

```text
L_density_teacher =
    smooth_l1(
        density_pred_from_mapper,
        stopgrad(density_teacher_8s)
    )
```

Use only when ground-truth density target is unavailable or as a tiny
regularizer:

```text
lambda_density_teacher <= 0.02
```

### 10.7 Loss Schedule

Do not start with a large density loss.

Recommended schedule:

```text
steps 0 - warmup_steps:
    lambda_density = 0
after warmup:
    linearly ramp to lambda_density_max
lambda_density_max = 0.05
```

If token CE remains stable and density metrics improve, allow:

```text
lambda_density_max = 0.10
```

Cap density gradient norm so it does not dominate token CE:

```text
density_grad_norm <= 0.3 * token_ce_grad_norm
```

### 10.8 Density Metrics

Report:

```text
density_frame_mae
density_frame_smooth_l1
density_window_mean_error
density_pearson_corr
density_spearman_corr
expected_onset_count_error
generated_onset_count_error
```

Report both teacher-forced and generated metrics.

## 11. LN Close Auxiliary Loss

### 11.1 Goal

Train the `LNCloseAdapter` to decide when each open lane should close.

This is a lane-level hazard prediction problem.

### 11.2 Labels

For each teacher-forced decode step `t` and lane `l`:

Mask:

```text
close_mask[t,l] = open_mask[t,l]
```

Positive label:

```text
y_close[t,l] = 1
iff
    gold next token is EVENT
    and gold EVENT action for lane l is HOLD_END
```

Negative label:

```text
y_close[t,l] = 0
iff
    lane l is open
    and the gold next token does not close lane l
```

This includes `TIME_SHIFT` steps while the lane remains open.

Closed lanes are ignored.

### 11.3 Loss

Use class-balanced BCE or focal BCE.

Recommended:

```text
pos_weight = num_negative_open_lane_steps / num_positive_close_steps
```

Clamped:

```text
1 <= pos_weight <= 20
```

Loss:

```text
L_ln_close =
    focal_bce_with_logits(
        close_logit,
        y_close,
        mask = close_mask,
        pos_weight = pos_weight,
        gamma = 1.5
    )
```

If focal BCE is unstable, use weighted BCE first.

### 11.4 Optional Duration Ranking Loss

For each LN instance, compare close hazard before and at the gold close step.

Let:

```text
h_before = max close_logit over open steps before gold close
h_close  = close_logit at gold close step
```

Then:

```text
L_duration_rank =
    max(0, margin - h_close + h_before)
```

Recommended:

```text
margin = 0.5
lambda_duration_rank <= 0.05
```

This is optional. Do not enable until BCE or focal loss works.

### 11.5 Loss Schedule

Recommended:

```text
lambda_ln_close initial = 0.05
lambda_ln_close max     = 0.20
```

Ramp over the first few thousand steps.

Cap close-head gradient norm:

```text
ln_close_grad_norm <= 0.5 * token_ce_grad_norm
```

### 11.6 LN Close Metrics

Report teacher-forced metrics:

```text
open_lane_close_precision
open_lane_close_recall
open_lane_close_f1
close_auc
early_close_rate
late_close_rate
close_timing_mae_ms
ln_duration_mae_ms
premature_close_rate
missed_close_before_eos_rate
```

Report generated metrics:

```text
generated_ln_duration_distribution
generated_premature_close_rate
generated_dead_end_rate
forced_end_pressure_rate
```

## 12. Total Training Loss

### 12.1 Main Loss

Token cross entropy is primary:

```text
L_token =
    cross_entropy(
        logits_final[:, :-1],
        target_tokens[:, 1:],
        ignore_index = PAD
    )
```

### 12.2 Total Loss

```text
L_total =
    L_token
  + lambda_density * L_density
  + lambda_ln_close * L_ln_close
  + lambda_density_teacher * L_density_teacher
  + lambda_adapter_reg * L_adapter_reg
```

Recommended:

```text
lambda_density_max        = 0.05
lambda_ln_close_max       = 0.20
lambda_density_teacher    = 0.00 by default
lambda_adapter_reg        = 1e-5
```

`L_adapter_reg` penalizes excessive adapter bias magnitude:

```text
L_adapter_reg =
    mean(square(state_prior_bias))
  + mean(square(ln_close_bias))
```

### 12.3 Training Phases

Phase A: audit and calibration

- build token vocabulary;
- tokenize mapper windows;
- audit legality;
- compute density calibration from gold tokens;
- estimate LN close class imbalance.

Phase B: stable teacher-forced training

- train decoder;
- train `StatePriorAdapter`;
- train `LNCloseAdapter`;
- keep density loss off during warmup.

Phase C: density ramp

- enable density auxiliary;
- ramp `lambda_density`;
- monitor token CE regression.

Phase D: rollout evaluation

- grammar-constrained generation;
- no training from rollout by default;
- measure generated density and LN close quality.

Phase E: optional scheduled-sampling style fine-tuning

- short generated prefixes only;
- 1s to 2s rollout fragments;
- grammar always active;
- apply LN close supervision where generated state overlaps gold LN state.

Do not enable Phase E until the teacher-forced model is stable.

## 13. Forward Pass

### 13.1 Pseudocode

```python
def forward(batch):
    with torch.no_grad():
        control_out = control_encoder(
            mel=batch.mel_context,
            timing=batch.timing_context,
            difficulty=batch.difficulty,
            padding_mask=batch.context_padding_mask,
        )

    control_memory_8s = slice_or_concat_control_memory(control_out.control_memory)
    density_teacher_8s = slice_or_concat_density_teacher(control_out.value_pred)

    decoder_hidden, base_logits = decoder(
        tokens=batch.target_tokens[:, :-1],
        control_memory=control_memory_8s,
        difficulty=batch.difficulty,
    )

    state_prior_bias = state_prior_adapter(
        open_mask=batch.teacher_open_mask[:, :-1],
        open_age_ms=batch.teacher_open_age_ms[:, :-1],
        remaining_ms=batch.teacher_remaining_ms[:, :-1],
    )

    close_logits, ln_close_bias, time_shift_bias = ln_close_adapter(
        decoder_hidden=decoder_hidden,
        control_memory_8s=control_memory_8s,
        density_teacher_8s=density_teacher_8s,
        current_ms=batch.teacher_current_ms[:, :-1],
        open_mask=batch.teacher_open_mask[:, :-1],
        open_age_ms=batch.teacher_open_age_ms[:, :-1],
        remaining_ms=batch.teacher_remaining_ms[:, :-1],
    )

    grammar_mask = build_grammar_mask(
        current_ms=batch.teacher_current_ms[:, :-1],
        open_mask=batch.teacher_open_mask[:, :-1],
        open_age_ms=batch.teacher_open_age_ms[:, :-1],
        write_start_ms=batch.write_start_ms,
        write_end_ms=batch.write_end_ms,
    )

    logits_final = (
        base_logits
        + state_prior_bias
        + ln_close_bias
        + time_shift_bias
        + grammar_mask
    )

    L_token = token_ce(
        logits_final,
        batch.target_tokens[:, 1:],
    )

    L_density = density_aux_loss(
        logits_final=logits_final,
        current_ms=batch.teacher_current_ms[:, :-1],
        target=batch.density_target_8s,
        confidence=batch.density_confidence_8s,
    )

    L_ln_close = ln_close_aux_loss(
        close_logits=close_logits,
        labels=batch.close_labels[:, :-1],
        mask=batch.close_label_mask[:, :-1],
    )

    L_total = (
        L_token
        + lambda_density * L_density
        + lambda_ln_close * L_ln_close
        + lambda_adapter_reg * adapter_reg()
    )

    return {
        "loss": L_total,
        "loss_token": L_token,
        "loss_density": L_density,
        "loss_ln_close": L_ln_close,
    }
```

## 14. Inference

### 14.1 Generation Loop

Initialize:

```text
tokens = [BOS]
current_ms = write_start_ms
open_mask = 0
open_age_ms = 0
```

Loop:

1. run decoder on prefix;
2. compute adapter biases;
3. compute grammar mask;
4. apply final logits;
5. sample or argmax;
6. update token list;
7. update `current_ms` and LN state;
8. stop only on `EOS`.

### 14.2 Sampling

Recommended default:

```text
temperature = 1.0
top_p = 0.95
```

Apply top-p after grammar mask.

Never sample invalid tokens.

### 14.3 Dead-End Handling

Dead-end should be impossible if grammar is correct.

If no legal token exists:

```text
raise RuntimeError
log full state
count as grammar bug
```

Do not silently repair.

### 14.4 Maximum Length

Use audit-derived max length:

```text
max_seq_len = p99_train_seq_len + margin
```

If exceeded during generation:

```text
raise generation failure
```

Do not force `EOS` if `open_mask != 0`.

## 15. Audits

### 15.1 Tokenizer Audits

Report:

```text
num_windows
num_eligible_windows
num_dropped_cross_window_ln
drop_rate_cross_window_ln
max_seq_len
mean_seq_len
p95_seq_len
p99_seq_len
event_vocab_coverage
time_shift_vocab_distribution
invalid_time_delta_count
noncanonical_time_shift_count
open_mask_nonzero_before_eos_count
```

Hard fail if:

```text
open_mask_nonzero_before_eos_count > 0
invalid_event_count > 0
noncanonical_time_shift_count > 0
```

### 15.2 Grammar Audits

Randomly replay tokenized targets through grammar.

Hard fail if any gold token is invalid.

Also test adversarial states:

```text
open LN at write_end
TIME_SHIFT to write_end while open
EOS while open
HOLD_END on closed lane
HOLD_START on open lane
TAP on open lane
all-NONE EVENT
```

### 15.3 Density Audits

Before training:

```text
gold_mass_to_density_mae
gold_mass_to_density_corr
density_target_missing_rate
density_confidence_distribution
```

During training:

```text
teacher_forced_density_mae
teacher_forced_density_corr
generated_density_mae
generated_density_corr
```

### 15.4 LN Close Audits

Before training:

```text
num_open_lane_steps
num_close_positive_steps
close_positive_rate
pos_weight
ln_duration_distribution
```

During training:

```text
close_precision
close_recall
close_f1
close_auc
early_close_ms
late_close_ms
duration_mae_ms
```

### 15.5 Adapter Audits

Report:

```text
state_prior_bias_mean
state_prior_bias_std
state_prior_bias_max_abs
ln_close_bias_mean
ln_close_bias_std
ln_close_bias_max_abs
time_shift_bias_mean_when_open
time_shift_bias_max_abs
```

If adapter bias saturates early, reduce adapter scale or increase
regularization.

## 16. Failure Modes and Mitigations

### 16.1 Event Spam From Density Loss

Symptom:

```text
density improves but token CE/generation quality worsens
too many TAP/HOLD_START events
```

Mitigation:

- lower `lambda_density`;
- improve density calibration;
- exclude `HOLD_END` from `onset_weight`;
- add generated onset-count metric.

### 16.2 Early LN Closure

Symptom:

```text
LN close recall high but duration too short
premature_close_rate high
```

Mitigation:

- lower `close_scale`;
- delay `skip_scale` ramp;
- increase negative weight for keep-open states;
- add duration ranking loss only after BCE stabilizes.

### 16.3 Late LN Closure

Symptom:

```text
missed_close_before_eos_rate high
close recall low
```

Mitigation:

- increase `lambda_ln_close`;
- increase `pos_weight`;
- allow skip penalty to suppress `TIME_SHIFT` near close;
- check whether `local_control_t` is aligned correctly.

### 16.4 Adapter Ignored

Symptom:

```text
close auxiliary improves but generated close timing unchanged
adapter bias near zero
```

Mitigation:

- increase `adapter_scale` slowly;
- verify adapter bias is added before grammar and top-p;
- verify gradients reach adapter;
- check whether base logits overpower bounded bias.

### 16.5 Adapter Dominates Decoder

Symptom:

```text
adapter bias saturates
generation becomes formulaic
token CE worsens
```

Mitigation:

- reduce `max_bias`;
- increase adapter regularization;
- lower `lambda_ln_close`;
- delay `skip_scale`.

## 17. Non-Goals for V1.1

The following are explicitly out of scope:

- cross-window LN carry-in and carry-out;
- multi-difficulty joint mapper;
- full-chart global structure planning;
- post-hoc repair;
- unconstrained adapter full-vocab logits;
- training the control encoder jointly with mapper.

Future versions may add these only after V1.1 metrics are stable.

## 18. Minimal Acceptance Criteria

A V1.1 mapper run is acceptable only if all are true:

1. tokenizer gold replay has zero grammar violations;
2. generation has zero invalid tokens;
3. generation has zero grammar dead-ends;
4. density auxiliary improves density MAE/correlation without increasing token
   CE materially;
5. LN close auxiliary improves close F1 and generated LN duration MAE;
6. adapter bias does not saturate;
7. cross-window LN drop rate is reported;
8. density target and teacher naming is consistent in code and logs.

Recommended minimum report:

```text
token_ce
generated_validity_rate
generated_dead_end_rate
density_frame_mae
density_window_error
density_corr
close_precision
close_recall
close_f1
ln_duration_mae_ms
premature_close_rate
late_close_rate
adapter_bias_stats
cross_window_ln_drop_rate
```

## 19. Final Design Principle

The mapper should be powerful enough to learn musical mapping, but not powerful
enough to bypass the chart grammar.

The density auxiliary should shape how much note onset mass appears over time.

The LN close adapter should shape when open lanes close.

The token decoder should still decide the final musical event.

The grammar should decide what is legal.

The most important implementation change is this: density loss and LN close
loss must be computed from teacher-forced model distributions and lane-level
close hazards, not from reconstructed gold tokens alone. Otherwise the design
looks clean on paper but the two signals that matter will not actually train
the mapper.
