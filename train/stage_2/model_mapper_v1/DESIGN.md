---
pinned_commit: 174da51756075740411333c501a7cd8140157f03
status: frozen
date: 2026-05-08
owner: s3d-i
module: train/stage_2/model_mapper_v1
depends_on:
  control_encoder: train/stage_2/model_control_demo_global
  control_config: train/stage_2/training/configs/stage2_control_demo_global_mps.yaml
window_contract: 8s mapper write window
token_contract: mapper_event_vocab_v1
time_shift_contract: canonical_relative_ts_v1
ln_contract: carry_ln_state_v1
---

# Stage 2 Mapper V1 Design Spec

This document freezes the Stage 2 Mapper v1 contract for implementation.

## Goal

Stage 2 Mapper v1 maps:

```text
audio + dense timing + difficulty + coarse control memory
```

to a playable osu!mania 4K event token sequence.

The model is not a complete chart oracle. It is a coarse
control-conditioned autoregressive event generator:

```text
ControlDemoGlobalEncoder:
  audio / timing / difficulty -> coarse density + control hidden
MapperDecoder:
  control hidden -> legal event token sequence
Grammar:
  hard legality, especially long-note open/close correctness
```

## Generation Unit

Mapper v1 generates one independent 8 second write window at a time.

```text
[write_start_ms, write_end_ms)
write_end_ms = write_start_ms + 8000
```

Each window decodes independently:

```text
BOS ... EOS
```

The v1 demo fast path defines:

```text
EOS valid iff open_mask == 0
```

Therefore every long note opened inside an 8 second window must close before
EOS. V1 does not support cross-window long-note carry-out.

A later proper version may add:

```text
WINDOW_EOS allows open_mask != 0
SONG_EOS requires open_mask == 0
```

That complexity is explicitly out of scope for v1.

## Input Contract

Each mapper sample corresponds to one 8 second write window.

```text
full_mel: FloatTensor[B, T_full, 160]
full_dense_timing_v2: FloatTensor[B, T_full, 4]
padding_mask: BoolTensor[B, T_full]
frame_count: LongTensor[B]
normalized_difficulty: FloatTensor[B]
mapper_write_start_ms: LongTensor[B]
mapper_write_start_frame: LongTensor[B]
```

Frozen geometry:

```text
frame_hop_ms = 20
mapper_window_frames = 400
mapper_window_ms = 8000
```

## Control Encoder Reuse

Mapper v1 reuses the trained `ControlDemoGlobalEncoder` from:

```text
train/stage_2/model_control_demo_global
```

with training config:

```text
train/stage_2/training/configs/stage2_control_demo_global_mps.yaml
```

The control encoder 2 second target output is:

```text
density_pred: [B, 100, 1]
control_memory: [B, 600, D]
```

Because the mapper window is 8 seconds, control memory is built by concatenating
four 2 second target slices:

```python
control_memory_8s = concat([
    control_out_0.control_memory[:, target_offset : target_offset + 100],
    control_out_1.control_memory[:, target_offset : target_offset + 100],
    control_out_2.control_memory[:, target_offset : target_offset + 100],
    control_out_3.control_memory[:, target_offset : target_offset + 100],
], dim=1)
```

The resulting mapper control inputs are:

```text
control_memory_8s: [B, 400, D]
density_pred_8s: [B, 400, 1]
```

The four target starts are:

```text
write_start_ms + 0
write_start_ms + 2000
write_start_ms + 4000
write_start_ms + 6000
```

V1 defaults to freezing `ControlDemoGlobalEncoder` and training only the
mapper decoder plus adapters. Fine-tuning the final control encoder layers is
deferred.

## Token Vocabulary

### TokenType

```python
class TokenType(Enum):
    SPECIAL = 0
    TIME_SHIFT = 1
    EVENT = 2
```

### SPECIAL Tokens

```text
PAD
BOS
EOS
```

There is no `UNK`. Data that cannot be encoded must fail audit directly.

## EVENT Token Contract

An `EVENT` token represents the 4-lane action tuple occurring at one timestamp.

Each lane action is:

```python
class LaneAction(Enum):
    NONE = 0
    TAP = 1
    HOLD_START = 2
    HOLD_END = 3
```

Event token metadata:

```python
@dataclass(frozen=True)
class EventTokenSpec:
    token_id: int
    lane_actions: tuple[LaneAction, LaneAction, LaneAction, LaneAction]
```

The theoretical lane-action space is:

```text
4 ** 4 = 256
```

`EVENT(NONE, NONE, NONE, NONE)` is a no-op and is not generatable in v1. It may
exist in metadata, but grammar must always mark it invalid.

The actual event vocabulary is 255 non-empty lane-action tuples.

Event semantics:

```text
EVENT occurs at current_ms
EVENT does not advance current_ms
```

A chord is a first-class token. Same-time lane actions are not emitted as
multiple lane tokens.

## TIME_SHIFT Token Contract

### Semantics

`TS_k` advances the current decode time cursor forward by `k` milliseconds:

```text
current_ms += k
```

`TS_k` is relative to the current cursor, not absolute relative to the window
start.

Initial decode time:

```text
current_ms = write_start_ms
```

`EVENT` tokens occur at the current `current_ms`.

### TIME_SHIFT_VOCAB_V1

Frozen time-shift values:

```text
TS_10,  TS_20,  ..., TS_90
TS_100, TS_200, ..., TS_900
TS_1000, TS_2000, TS_3000, TS_4000
```

```python
TIME_SHIFT_VALUES_MS = (
    10, 20, 30, 40, 50, 60, 70, 80, 90,
    100, 200, 300, 400, 500, 600, 700, 800, 900,
    1000, 2000, 3000, 4000,
)
```

There are 22 time-shift tokens. The largest token is `TS_4000`.

An empty 8 second window is:

```text
BOS TS_4000 TS_4000 EOS
```

### Canonical Encoding

All timestamp deltas must be 10ms aligned:

```text
delta_ms > 0
delta_ms % 10 == 0
delta_ms <= 8000
```

Encoding uses greedy largest-first decomposition:

```python
TS_VALUES_DESC = (
    4000, 3000, 2000, 1000,
    900, 800, 700, 600, 500, 400, 300, 200, 100,
    90, 80, 70, 60, 50, 40, 30, 20, 10,
)

def encode_ts(delta_ms: int) -> list[int]:
    assert delta_ms > 0
    assert delta_ms % 10 == 0
    assert delta_ms <= 8000
    out = []
    remaining = delta_ms
    for k in TS_VALUES_DESC:
        while remaining >= k:
            out.append(k)
            remaining -= k
    assert remaining == 0
    return out
```

Examples:

```text
8000 -> TS_4000 TS_4000
7990 -> TS_4000 TS_3000 TS_900 TS_90
5000 -> TS_4000 TS_1000
3760 -> TS_3000 TS_700 TS_60
1250 -> TS_1000 TS_200 TS_50
120  -> TS_100 TS_20
10   -> TS_10
```

Each delta has exactly one legal encoding.

## CarryLNState Contract

```python
@dataclass(frozen=True)
class CarryLNState:
    open_mask: int  # 0..15
    open_age_ms_by_lane: tuple[int, int, int, int]
```

Meaning:

```text
open_mask bit lane:
  0 = lane closed
  1 = lane open
open_age_ms_by_lane[lane]:
  closed -> 0
  open   -> current_ms - hold_start_ms
```

Decode state:

```python
@dataclass(frozen=True)
class DecodeState:
    current_ms: int
    carry: CarryLNState
```

V1 window initial state:

```python
CarryLNState(
    open_mask=0,
    open_age_ms_by_lane=(0, 0, 0, 0),
)
```

## State Update Rules

### TIME_SHIFT Update

```text
current_ms += k
for lane in open lanes:
    open_age_ms_by_lane[lane] += k
```

### EVENT Update

`EVENT` is applied atomically across all 4 lanes:

```text
for lane, action in enumerate(event.lane_actions):
    if action == NONE:
        continue
    if action == TAP:
        require lane closed
        state unchanged
    if action == HOLD_START:
        require lane closed
        open lane
        age[lane] = 0
    if action == HOLD_END:
        require lane open
        close lane
        age[lane] = 0
```

`EVENT` does not advance time.

## Grammar Contract

The hard grammar mask is the final legality authority.

Logit order:

```text
logits = base_logits
logits = logits + dynamic_state_logits_adapter(dynamic_state_t)
logits = logits + hard_grammar_mask(decode_state_t)
```

The hard grammar mask is applied last. Invalid tokens must receive `-inf` or an
equivalent large negative value.

### SPECIAL Grammar

```text
PAD:
  never valid during generation
BOS:
  only first token
EOS:
  valid iff open_mask == 0
```

V1 makes EOS invalid before every open long note has been closed.

### TIME_SHIFT Grammar

`TS_k` is valid iff:

```text
k > 0
current_ms + k <= write_end_ms
```

If:

```text
current_ms == write_end_ms
```

then:

```text
TIME_SHIFT invalid
EVENT invalid
EOS valid iff open_mask == 0
```

### EVENT Grammar

`EVENT` is valid iff:

```text
current_ms < write_end_ms
event tuple is not all NONE
lane actions obey open_mask
```

Per-lane legality:

```text
if lane closed:
  valid:
    NONE
    TAP
    HOLD_START
  invalid:
    HOLD_END
if lane open:
  valid:
    NONE
    HOLD_END
  invalid:
    TAP
    HOLD_START
```

Pseudocode:

```python
def is_event_valid(actions, open_mask, current_ms, write_end_ms):
    if current_ms >= write_end_ms:
        return False
    has_action = False
    for lane, action in enumerate(actions):
        lane_open = bool(open_mask & (1 << lane))
        if action != LaneAction.NONE:
            has_action = True
        if lane_open:
            if action in (LaneAction.TAP, LaneAction.HOLD_START):
                return False
        else:
            if action == LaneAction.HOLD_END:
                return False
    return has_action
```

## Mapper Decoder Architecture

```text
prev_tokens
  -> token embedding
  -> positional embedding / rotary / learned causal position
  -> causal Transformer decoder
       self-attention over prefix
       cross-attention to control_memory_8s
  -> base logits over vocab
```

Recommended v1 config:

```text
d_model: 384
decoder_layers: 4
heads: 8
ffn_dim: 1536
dropout: 0.1
max_seq_len: audit-derived, likely 512 or 768 initially
cross_attention_memory: control_memory_8s
```

Decoder input:

```text
prev_token_ids: LongTensor[B, L]
control_memory_8s: FloatTensor[B, 400, D]
control_memory_padding_mask: BoolTensor[B, 400]
```

Decoder output:

```text
base_logits: FloatTensor[B, L, vocab_size]
```

## DynamicStateLogitsAdapter

### Purpose

The dynamic adapter provides a soft bias based on the current long-note state.
It must not become another decoder.

It may only output structured lane/action bias:

```text
lane_action_bias: [B, 4 lanes, 4 actions]
```

The lane/action bias is projected to event-token logits through fixed vocab
metadata.

### Input

At every teacher-forced or decode step:

```text
open_mask_t: LongTensor[B]
open_age_ms_by_lane_t: LongTensor[B, 4]
```

Derived features:

```text
open bits [B, 4]
age norm [B, 4]
age bucket [B, 4]
lane id [4]
```

### Architecture

```text
open_mask + open_age
  -> DynamicStateEncoder
  -> low-rank bottleneck
  -> lane_action_bias[4, 4]
  -> fixed projection to EVENT vocab
```

Recommended configuration:

```text
d_state: 64
rank: 8
age_buckets_ms:
  [0, 40, 80, 120, 200, 400, 800, 1600, 3200, 6400]
init:
  final projection zero-initialized
  adapter scale near zero
```

Pseudocode:

```python
lane_h = (
    lane_embedding[lane]
    + open_embedding[open_bit]
    + age_bucket_embedding[age_bucket]
    + scalar_proj([open_bit, age_norm])
)
lane_h = LayerNorm(lane_h)
raw = Linear(rank, 4)(
    GELU(
        Linear(d_state, rank)(lane_h)
    )
)
lane_action_bias = tanh(raw) * softplus(scale_logit)
```

### Projection to Flat Vocab

For `EVENT` tokens:

```python
event_bias[token] = sum(
    lane_action_bias[:, lane, action_of_token_lane]
    for lane in range(4)
)
```

For non-event tokens:

```text
PAD/BOS/EOS/TIME_SHIFT bias = 0
```

Final flat bias:

```python
flat_bias[:, event_token_ids] = event_bias
```

This design is:

- low-rank;
- action-aware;
- structured;
- unable to directly rewrite arbitrary vocab logits;
- unable to bypass hard grammar.

## Training Tokenization

### Window Tokenization

Given an 8 second window:

```text
write_start_ms
write_end_ms = write_start_ms + 8000
```

Initialize:

```python
current_ms = write_start_ms
state = CarryLNState(0, (0, 0, 0, 0))
tokens = [BOS]
```

Convert beatmap hit objects to timestamp groups:

```python
dict[int, tuple[LaneAction, LaneAction, LaneAction, LaneAction]]
```

For each timestamp `t`:

```text
delta = t - current_ms
emit canonical TS tokens for delta
emit EVENT(tuple lane actions)
update current_ms
update CarryLNState
```

Finally:

```text
require state.open_mask == 0
emit EOS
```

If the window contains a cross-boundary long note that cannot be closed before
EOS, drop the sample for v1 and record an audit counter.

### Required Audits

The mapper dataset builder must output:

```text
window_count_total
window_count_kept
window_count_dropped_cross_window_ln
invalid_event_tuple_count
invalid_ts_delta_count
max_tokens_per_window
mean_tokens_per_window
p95_tokens_per_window
p99_tokens_per_window
event_vocab_coverage
ts_vocab_distribution
open_mask_nonzero_before_eos_count
```

Any nonzero `open_mask_nonzero_before_eos_count` is a hard fail.

## Training Forward Pass

Teacher forcing:

```python
control_memory_8s, density_pred_8s = build_control_memory_8s(...)
gold_tokens = batch["mapper_tokens"]
gold_dynamic_states = batch["dynamic_state_trace"]
base_logits = mapper_decoder(
    prev_tokens=gold_tokens[:, :-1],
    control_memory=control_memory_8s,
)
dynamic_bias = dynamic_state_logits_adapter(
    open_mask=gold_dynamic_states.open_mask[:, :-1],
    open_age_ms_by_lane=gold_dynamic_states.open_age[:, :-1],
)
grammar_mask = hard_grammar_mask_batch(
    decode_states=gold_decode_states[:, :-1],
)
logits = base_logits + dynamic_bias + grammar_mask
target = gold_tokens[:, 1:]
```

Loss:

```text
L_token = cross_entropy(logits, target, ignore_index=PAD)
```

Optional auxiliary density loss:

```text
tokens -> reconstructed event density
compare against density_level target or frozen density_pred_8s
```

V1 default:

```text
L_total = L_token
```

V1.1 may add:

```text
L_total = L_token + 0.1 * L_density_aux
```

Reason: first make grammar and token modeling work, then add auxiliary control
pressure. The first version must not mix failure modes.

## Inference

```python
state = CarryLNState(0, (0, 0, 0, 0))
current_ms = write_start_ms
tokens = [BOS]
for step in range(max_decode_steps):
    base_logits = decoder(tokens, control_memory_8s)
    dynamic_bias = dynamic_state_logits_adapter(state)
    grammar_mask = hard_grammar_mask(state, current_ms)
    logits = base_logits[:, -1] + dynamic_bias + grammar_mask
    token = sample_or_greedy(logits)
    tokens.append(token)
    if token == EOS:
        assert state.open_mask == 0
        break
    current_ms, state = apply_token(token, current_ms, state)
```

Sampling policy v1:

```text
greedy or temperature <= 1.0
top_p after grammar mask
never repair illegal token after sampling; illegal tokens must be impossible
```

## File Layout

```text
train/stage_2/events/schema.py
  TokenType
  LaneAction
  CarryLNState
  DecodeState
  EventTokenSpec
  MapperWindowRecord
train/stage_2/events/vocab.py
  build_mapper_event_vocab_v1()
  token metadata tensors
  TIME_SHIFT_VALUES_MS
train/stage_2/events/quantization.py
  quantize_10ms_half_up()
  encode_ts_canonical()
  decode_ts_token()
train/stage_2/events/carryLN.py
  update_state_for_token()
  build_dynamic_state_trace()
  validate_carry_trace()
train/stage_2/events/grammar.py
  hard_grammar_mask()
  validate_token_sequence()
train/stage_2/events/tokenize.py
  beatmap_window_to_tokens()
  events_to_lane_action_groups()
train/stage_2/events/stitch.py
  stitch_8s_windows()
  validate_song_events()
train/stage_2/model_mapper_v1/model.py
  Stage2MapperV1
  MapperARDecoder
  DynamicStateEncoder
  DynamicStateLogitsAdapter
train/stage_2/training/mapper_v1.py
  dataset
  collate
  training loop
  eval
  reports
```

## Frozen Constants

```python
MAPPER_WINDOW_MS = 8000
MAPPER_WINDOW_FRAMES = 400
FRAME_HOP_MS = 20
TS_QUANTUM_MS = 10
TIME_SHIFT_VALUES_MS = (
    10, 20, 30, 40, 50, 60, 70, 80, 90,
    100, 200, 300, 400, 500, 600, 700, 800, 900,
    1000, 2000, 3000, 4000,
)
LANE_COUNT = 4
LANE_ACTIONS = (
    NONE,
    TAP,
    HOLD_START,
    HOLD_END,
)
SPECIAL_TOKENS = (
    PAD,
    BOS,
    EOS,
)
```

Approximate vocabulary size:

```text
3 special
22 time shift
255 event
= 280 tokens
```

## Non-Goals for V1

V1 explicitly does not handle:

- cross-window long-note carry-out;
- `WINDOW_EOS` vs `SONG_EOS` distinction;
- full-song single-pass autoregressive decoding;
- direct absolute timestamp tokens;
- grid-frame token generation;
- unstructured vocab-wide dynamic adapter;
- same-lane compound exotic events;
- learned time-shift aliasing.

These are deferred until the demo mapper reliably produces legal 8 second
windows.

## Final Frozen Summary

Stage 2 Mapper v1 freezes an 8 second window autoregressive event generator.
It reuses `ControlDemoGlobalEncoder` as a frozen audio, dense-timing, and
difficulty control planner. Four 2 second control outputs are concatenated into
8 second control memory.

The decoder generates:

```text
SPECIAL: PAD/BOS/EOS
TIME_SHIFT: canonical relative TS_k
EVENT: 4-lane action tuple
```

`TIME_SHIFT` is relative to the current decode cursor. `EVENT` occurs at the
current cursor and does not advance time. `EOS` is valid only when
`open_mask == 0`.

Long-note state is explicitly tracked by `CarryLNState`.
`DynamicStateLogitsAdapter` only produces low-rank lane/action bias. The hard
grammar mask is applied last and is the sole legality authority.
