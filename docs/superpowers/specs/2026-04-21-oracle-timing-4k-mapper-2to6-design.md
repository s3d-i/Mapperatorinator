# Oracle Timing 4K Mapper for 2–6* Mania Design

## Goal

Build a small, fast, 4K-only mapper that generates playable 2–6* osu!mania hitobjects from:

- audio
- target difficulty
- oracle timing derived from the reference `.osu`

This Stage 1 system is intentionally narrow:

- 4K only
- 2.0* to 6.0* only
- oracle timing only
- hitobject generation only

It does **not** solve timing inference, planner/control learning, style conditioning, or out-of-range generalization.

## Scope

### Supported Range

- Training: `2.0* <= stars <= 6.0*`
- Validation: `2.0* <= stars <= 6.0*`
- Inference UI: only allow `2.0* <= difficulty <= 6.0*`

### Out-of-Range Behavior

- Training: filter out
- Validation: filter out
- Inference UI: disallow with explicit validation error

### Non-Goals

- `<2.0*` or `>6.0*` quality
- timing prediction
- SV/KIAI/style/mapper/year/descriptor generation
- multi-gamemode support
- reuse of the `osuT5` training/inference pipeline

## Core Decisions

- Difficulty source is the existing mania calculator in [train/stage1_oracle/core/difficulty.py](/Users/l/projects/Mapperatorinator/train/stage1_oracle/core/difficulty.py:517).
- Time is quantized on a **global deterministic 10ms grid before window assignment**.
- Decoder language is a canonical `(TS+ EV)*` event stream with forced condition prefix.
- Window ownership is defined on half-open absolute intervals: `[write_start, write_end)`.
- Decoder target timestamps are relative to `write_start`.
- Export uses oracle **red timing points only**. Model output is `HitObjects` only.

## Difficulty Source

Stage 1 uses the repository's existing official-style mania difficulty computation:

- `compute_mania_star_rating_20241007()` in [train/stage1_oracle/core/difficulty.py](/Users/l/projects/Mapperatorinator/train/stage1_oracle/core/difficulty.py:517)
- `calculate_mania_difficulty()` in [train/stage1_oracle/core/difficulty.py](/Users/l/projects/Mapperatorinator/train/stage1_oracle/core/difficulty.py:557)

No alternate difficulty label source is allowed in Stage 1.

### Difficulty Buckets

- Bucket size: `0.25*`
- Supported stars: `2.0*` through `6.0*`
- Bucket count: `17`
- Bucket id uses deterministic half-up bucketing:

```text
bucket_id = floor(((stars - 2.0) / 0.25) + 0.5)
```

- Bucket id must validate within `[0, 16]`

Training and validation maps must be filtered by true difficulty first. This is **not** implemented by clamping out-of-range maps into the nearest bucket.

## Time Model

### Absolute Event Representation

The canonical preprocessing order is:

1. Parse `.osu` hitobjects into absolute times in milliseconds.
2. Quantize absolute hitobject times to the global 10ms grid.
3. Build canonical quantized events.
4. Assign quantized events to write windows.
5. Convert window-owned absolute times to decoder-relative times.

This order is fixed. Window assignment must never be done on raw unquantized timestamps.

### Deterministic 10ms Quantization

Stage 1 does **not** use Python's built-in `round()` semantics, because banker's rounding is not acceptable for dataset determinism.

Absolute hitobject timestamps are quantized with deterministic half-up rounding:

```text
quantize_10ms_half_up(t_ms) = 10 * floor((t_ms + 5) / 10)
```

This is defined for non-negative timestamps. Negative-time hitobjects are not supported in Stage 1 and are filtered before quantization.

### Canonical Time Grid Rules

- All hitobject timestamps are quantized globally before windowing.
- Event ownership is determined from quantized absolute time.
- Decoder targets use relative time measured from `write_start`.
- `write_end` is exclusive.

Example:

- raw event at `7996ms`
- quantized event at `8000ms`
- ownership: next window, not previous window

### Generation End

Training and inference use a fixed rule for the final writable time.

Define:

```text
ceil_10ms(t_ms) = 10 * ceil(t_ms / 10)
```

For training:

```text
generation_end =
  max(
    ceil_10ms(audio_duration_ms),
    max_quantized_event_time + 10
  )
```

For inference without ground-truth hitobjects:

```text
generation_end = ceil_10ms(audio_duration_ms)
```

Rationale:

- the last quantized event at time `q_t` must belong to some half-open region `[start, end)`
- therefore the writable timeline must extend at least to `q_t + 10ms`

## Data Legality Rules

These rules are fixed before tokenizer construction.

### Negative-Time Hitobjects

Maps containing any hitobject with:

- `start_time < 0`, or
- `end_time < 0`

are filtered out of Stage 1 training and validation. The filtered count must be reported.

### Missing Red Timing

Maps with no red timing point are filtered out. Stage 1 never synthesizes a fake red timing base. The filtered count must be reported.

### Zero-Length Holds

Zero-length holds are defined after global quantization:

```text
q_end <= q_start
```

These are normalized into a `TAP` at `q_start` during canonical event construction and counted in audit/reporting as `zero_length_hold_normalized_count`.

## Canonical Quantized Events

Canonical quantized events are constructed **after**:

- global half-up 10ms quantization
- zero-length hold normalization
- same-timestamp merge

All downstream decisions must use these canonical quantized events.

This includes:

- Event Space Audit
- Token Statistics Audit
- Window Boundary Audit
- open-hold-state computation
- training target generation

## Windowing

### Write Regions

Write regions are fixed to half-open absolute intervals:

```text
window k:
  write_start = k * 8000ms
  write_end   = min(write_start + 8000ms, generation_end)
  write region = [write_start, write_end)
```

An event belongs to window `k` iff:

```text
write_start <= q_t < write_end
```

### Relative Target Time Axis

Decoder target timestamps are relative to `write_start`, not `input_start`.

For an event owned by the window:

```text
t_rel = q_t - write_start
```

It must satisfy:

```text
0 <= t_rel < write_duration
write_duration = write_end - write_start
```

### Fixed 12s Input Window

Every write region uses a fixed 12s model input, including the first and last windows.

For every window:

```text
input_start = write_start - 2000ms
input_end   = write_start + 10000ms
input_duration = 12000ms
```

This stays fixed even when the final write region is shorter than 8000ms.

### First Window

For the first window:

- `write = [0, min(8000, generation_end))`
- `input = [-2000, 10000)`

Padding/extrapolation rules:

- audio before `0ms`: silence
- timing before `0ms`: extrapolate from the first red timing point
- `timing_change_pulse`: only on real timing changes, never synthesized by extrapolation
- `open_hold_mask_at_write_start = 0000`

### Last Window

For the final window:

- `write_end = generation_end`
- `write_duration` may be `< 8000ms`
- `input` is still fixed to 12s by the rule above

Padding/extrapolation rules:

- audio after `audio_duration`: silence
- timing after `audio_duration`: extrapolate from the last red timing point
- no fake `timing_change_pulse` during extrapolation

The decoder must obey:

- `t_rel < write_duration`
- `EOS` may appear before `write_end`

## Open Hold State

`open_hold_mask_at_write_start` is computed from **canonical quantized events**, not raw `.osu` times.

Definition:

- apply all canonical events with `q_t < write_start`
- the resulting lane-open state at `write_start` is the boundary mask

Important boundary rule:

- events at `q_t == write_start` belong to the current window
- therefore they must **not** be folded into the boundary mask

### Source by Mode

- Training: ground-truth boundary mask
- Oracle-boundary evaluation: ground-truth boundary mask
- Stitched-boundary evaluation: boundary mask carried from previously generated windows
- Inference: boundary mask carried from previously generated windows

### Usage

The boundary mask is used in two places:

- forced decoder condition
- constrained decoding legality state

## Sequence Grammar

The decoder is modeled as a finite-state machine, not as free text generation.

### Token Language

```text
sequence :=
  BOS
  condition_prefix
  (TS+ EV)*
  EOS
```

Where:

- `condition_prefix = DIFF_x OPEN_MASK_xxxx`
- `condition_prefix` is forced input
- `condition_prefix` is excluded from cross-entropy loss
- `EV` must never represent an all-empty timepoint

### Loss Convention

Example with events:

```text
decoder input:
  BOS DIFF_4.25 OPEN_0010 TS_120 EV_A TS_80 EV_B

loss target:
  TS_120 EV_A TS_80 EV_B EOS
```

Example with no events:

```text
decoder input:
  BOS DIFF_4.25 OPEN_0010

loss target:
  EOS
```

`OPEN_MASK_xxxx EOS` is legal. It means "no new events in this write region". It does **not** mean all open holds are forcibly closed.

### EOS Rules

`EOS` is legal only:

1. immediately after `condition_prefix`
2. immediately after `EV`

`EOS` is illegal after one or more `TS` tokens with no following `EV`.

Therefore:

- legal: `BOS DIFF OPEN EOS`
- legal: `BOS DIFF OPEN TS_240 EV_A EOS`
- illegal: `BOS DIFF OPEN TS_240 EOS`

### TS_0 Rules

- `TS_0` is legal only before the first `EV` in a window
- after the first `EV`, the next event must have strictly positive delta

Therefore:

- legal: `BOS DIFF OPEN TS_0 EV_A EOS`
- illegal: `BOS DIFF OPEN TS_0 EV_A TS_0 EV_B EOS`

### Canonical TS Decomposition

The time-shift vocabulary is:

- `TS_0`
- `TS_10`
- `TS_20`
- ...
- `TS_1000`

Canonical decomposition is mandatory:

- `delta <= 1000ms`: use exactly one `TS_delta`
- `delta > 1000ms`: use greedy largest-first decomposition

Examples:

- `0ms -> TS_0` only for the first event
- `990ms -> TS_990`
- `1000ms -> TS_1000`
- `1500ms -> TS_1000 TS_500`
- `2700ms -> TS_1000 TS_1000 TS_700`

Non-canonical decompositions are illegal:

- `TS_500 TS_500 EV`
- `TS_700 TS_300 EV`
- `TS_10 TS_10 ...`

This rule is enforced both:

- in tokenization
- in constrained decode masks

When the decoder is in a state that expects `TS`, each legal `TS_x` must satisfy:

```text
0 <= current_time_rel + pending_delta + x < write_duration
```

That is, every emitted `TS` must still leave the potential next `EV` inside the current write region.

If no `TS` token is legal under this bound:

- `EOS` becomes the only legal continuation when grammar allows `EOS`
- otherwise decoding must continue through the non-`TS` legal path defined by the FSM

### Decoder FSM

The constrained decoder must track at least:

- `current_time_rel`
- `pending_delta`
- `has_emitted_event`
- `open_hold_mask`
- `ts_state`

Suggested `ts_state` values:

- `EXPECT_TS_OR_EOS`
- `EXPECT_EVENT_OR_MORE_TS_1000`
- `EXPECT_EVENT_ONLY`

Core transition rules:

1. after `condition_prefix`: `EOS` or `TS`
2. after `TS`: never `EOS`
3. after `EV`: `EOS` or `TS`
4. the first `EV` may occur at `t_rel = 0`
5. later `EV`s must strictly increase time
6. `EV` time must satisfy `t_rel < write_duration`
7. `EV` must respect lane legality under current `open_hold_mask`
8. after `EV`, update `open_hold_mask`
9. TS canonical decomposition must be enforced by the decode mask
10. when a state expects `TS`, the TS legality mask must prevent pending time from pushing the next possible event outside `[0, write_duration)`
11. if no `TS` is legal and grammar permits `EOS`, `EOS` is the only legal continuation

### Max Decode Length Handling

If `max_decode_len` is reached:

- if the current state legally allows `EOS`, emit `EOS`
- if the current state contains dangling pending `TS`, rollback **only the pending TS suffix**, keep all prior completed `TS+EV` groups, then emit `EOS`

This must be recorded as:

- `EOS_forced_after_pending_ts`

The rollback must never remove previously completed events.

## Event Representation

Stage 1 uses timepoint-level event output, not note-level token output.

### Frozen Lane Action Space

Stage 1 uses the 4-state lane-action vocabulary:

- `NONE`
- `TAP`
- `HOLD_START`
- `HOLD_END`

The 6-state vocabulary is explicitly rejected for Stage 1.

Reference audit:

- [docs/superpowers/audits/2026-04-21-event-space-audit-4k-2to6.md](/Users/l/projects/Mapperatorinator/docs/superpowers/audits/2026-04-21-event-space-audit-4k-2to6.md)
- audit implementation/result commit: `438565f6743d6d9eed32928f01892e1e3e455c3e`

### Event Space Audit

Event Space Audit is a hard pre-training gate and must run on **canonical quantized events**.

It must report:

- `END_TAP` frequency
- `END_START` frequency
- same-lane same-timestamp compound-event frequency
- top-K event coverage
- rare-event count

### 6-State vs 4-State Freeze Rule

Stage 1 is frozen to 4-state. The allowed lane actions are:

- `NONE`
- `TAP`
- `HOLD_START`
- `HOLD_END`

### 4-State Compound Event Policy

Unsupported canonical compound same-lane events are **not** normalized heuristically at token time.

Instead:

- maps containing canonical `END_TAP` or `END_START` events are filtered out of the Stage 1 dataset
- maps containing same-lane compounds unsupported by both 4-state and 6-state vocabularies are filtered out of the Stage 1 dataset
- the filtered map count and filtered event count must be reported

### Rejected 6-State Semantics

Stage 1 does not train, tokenize, decode, or export `END_TAP` or `END_START`. Maps containing these actions after canonical quantization are filtered.

## Timing Track

The dense timing track is aligned to 20ms encoder frames and contains:

- `beat_pulse`
- `measure_pulse`
- `timing_change_pulse`
- `beat_phase_sin`
- `beat_phase_cos`
- `measure_phase_sin`
- `measure_phase_cos`
- `local_bpm_log_norm`

Rules:

- pulse channels use Gaussian or triangular support, not one-hot spikes
- extrapolation outside audio bounds follows the nearest red timing point
- extrapolation must not create fake `timing_change_pulse`

## Audio Representation

Stage 1 caches canonical 10ms log-mel features and derives model input frames deterministically from that cache.

### Canonical Cached Features

- sample rate: `16k`
- mel bins: `80`
- hop: `10ms`
- cached feature type: log-mel

The cached 10ms log-mel representation is the only canonical audio feature cache for Stage 1.

### Deterministic 20ms Encoder Frames

The encoder consumes 20ms audio frames produced by deterministic pair packing from the cached 10ms mel sequence.

For adjacent 10ms mel frames `m_t` and `m_{t+1}`:

```text
packed_audio_frame_t = concat(m_t, m_{t+1})
```

Therefore:

- each packed frame covers exactly `20ms`
- each packed frame is `160` dimensions wide
- packing order is fixed and must be consistent across train/val/infer

This is not a learned downsampling layer. It is a deterministic representation transform.

## Model Architecture

Stage 1 uses a shared fused encoder:

- audio frame projection from packed 160-dim audio frames via a learned `Linear(160 -> d_model)`
- timing frame projection
- broadcast difficulty embedding
- fused frame projection
- shared transformer encoder
- autoregressive decoder with cross-attention

Not included in Stage 1:

- separate audio/timing encoders
- beam search
- wide difficulty-range conditioning

### Parameter Budget

- target: `<= 15M`
- hard cap: `<= 25M`

Baseline configuration:

- `d_model = 256`
- `heads = 4`
- `encoder_layers = 4`
- `decoder_layers = 6`
- `ffn_dim = 1024`

If speed is insufficient, the first ablation is to shrink width/depth, not expand scope.

## Caching and Splits

Cache directories must encode config/version identity, for example:

- `cache/mel_sr16000_hop10_mel80_v1/`
- `cache/timing_track_20ms_8ch_sigma30_v1/`
- `cache/mapper_tokens_4k_2to6_ts10_1000_6state_v1/`

Required version/config fields:

- `cache_version`
- `mel_config_hash`
- `timing_render_config_hash`
- `tokenizer_version`

Split keys:

- primary: `audio hash`
- secondary: `beatmapset_id`
- fallback: `artist + title + audio_length`

Hard rule:

- the same `audio hash` must never cross train/validation

## Sampling

Training sampling is balanced over coarse bins:

- `2–3*`
- `3–4*`
- `4–5*`
- `5–6*`

This is used to avoid collapsing toward the most common midrange difficulty.

Empty windows are kept but capped per epoch. The cap is determined from audit and must be reported.

## Audits

All audits are mandatory before training.

### Event Space Audit

Runs on canonical quantized events and decides:

- retain 6-state, or
- freeze to 4-state

### Token Statistics Audit

Per coarse difficulty bin, report:

- tokens per 8s write region: mean / p95 / p99 / max
- event timepoints per second
- note events per second
- LN ratio
- chord-size distribution
- TS distribution
- empty window ratio
- hold-crossing-window ratio

This audit determines:

- `max_decode_len`
- empty-window cap
- whether `TS_1000` is sufficient

### Quantization Audit

Report:

- mean quantization error
- p95 quantization error
- max quantization error
- post-quantization collision rate

### Window Boundary Audit

Report:

- boundary event density
- hold crossing rate at write boundaries
- duplicate/collision risk after stitch

## Evaluation

Metrics must be reported both overall and per difficulty bin:

- `2–3*`
- `3–4*`
- `4–5*`
- `5–6*`

### Core Metrics

- lane-aware onset F1
- lane-agnostic onset F1
- hold boundary F1
- density error
- LN ratio error
- chord-size distribution error
- invalid hold rate
- same-lane collision rate
- empty output rate
- EOS failure rate
- average generated length
- tokens/sec
- RTF
- GPU memory

### Boundary Metrics

- `boundary_open_mask_error_rate`

This must be reported in two evaluation modes:

1. `oracle-boundary eval`
   - each window gets ground-truth `open_hold_mask_at_write_start`
   - isolates decoder/token quality from boundary-state propagation

2. `stitched-boundary eval`
   - windows are decoded sequentially
   - next-window boundary mask comes from the previously generated windows
   - measures real boundary error accumulation

Definition:

```text
boundary_open_mask_error_rate =
  (# window boundaries where predicted/stitched open_hold_mask_at_write_start != ground-truth open_hold_mask_at_write_start)
  / (# evaluated window boundaries)
```

Interpretation:

- under oracle-boundary eval, this metric is `0` by construction and may be reported as `0` or `N/A`
- under stitched-boundary eval, this is a primary boundary-stability metric

### Difficulty Control Diagnostics

For fixed validation audio+timing, sweep:

- `2.0*`
- `3.0*`
- `4.0*`
- `5.0*`
- `6.0*`

Report:

- NPS
- average chord size
- chord rate
- LN ratio
- peak density
- difficulty-density Spearman correlation

Minimum expectation:

- generated NPS must correlate positively with requested difficulty

### Raw vs Repaired Export Metrics

Core model quality metrics are computed on raw generated events first.

If export repair is applied, report separately:

- `repair_dropped_events_count`
- `repair_closed_holds_count`
- `repair_zero_length_holds_count`

Repair must not be allowed to hide raw generation failures.

## Stitch and Export

### Stitch

Stitch is intentionally simple:

1. decode one window into relative events
2. convert to absolute times using `t_abs = write_start + t_rel`
3. append windows in order
4. update global hold state

Stitch does **not** perform heuristic deduplication.

If duplicate events or same-lane collisions appear after stitch, this is treated as:

- a tokenizer bug
- a decode legality bug
- or a raw generation error

### Export

Stage 1 export uses:

- oracle red timing points from the reference map
- model-generated `HitObjects`

Stage 1 export does **not** copy inherited green lines by default.

This keeps evaluation aligned with the actual Stage 1 capability:

- timing is oracle-provided
- hitobjects are model-generated

## Training Phases

### Phase A

- no timing noise augmentation
- verify tokenizer, decoder grammar, legality masks, stitch, and export correctness

### Phase B

- light timing augmentation may be introduced after Phase A passes
- robustness-only phase, not correctness phase

## Pre-Training Gates

Training must not start until all of the following pass:

1. `.osu -> events -> tokens -> events -> .osu` round-trip
2. timing render debug plots
3. window stitch dry-run
4. Event Space Audit
5. Token Statistics Audit
6. Quantization Audit
7. Window Boundary Audit

## Success Criteria

### 32-Map Overfit

- covers all four coarse difficulty bins
- round-trip exactness already at `100%`
- token accuracy rises clearly
- generated density within `±20%` of ground truth
- `time_monotonicity_error = 0`
- `EOS_failure_rate < 5%`
- `empty_output_rate < 5%`

### 1k-Map Pilot

- report per-bin metrics
- `2–3*` not empty
- `5–6*` not obviously under-dense
- `invalid_hold_end_rate < 2%`
- `unclosed_hold_rate < 5%`
- `empty_output_rate < 10%`
- difficulty-density Spearman `> 0`

### 14k Overnight Baseline

Report:

- overall metrics
- per-bin metrics
- difficulty sweep metrics
- params
- train tokens/sec
- inference tokens/sec
- GPU memory
- raw invalid rates
- density error

RTF reporting rules:

- report RTF per difficulty bin
- measure on the same decode path used for export
- if both naive and cached decode exist, report both

The Stage 1 speed target assumes incremental autoregressive decode with cached decoder state. Naive full-prefix decode is acceptable only as a correctness baseline, not as the speed target path.
