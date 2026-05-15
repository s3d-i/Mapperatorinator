---
pinned_commit: 3ba4d409e729815f4029c10ae553713f72ccf1bf
status: sparse real-data training path and initial real-run config implemented
date: 2026-05-15
owner: s3d-i
module: train/stage_2/model_mapper_v2_1
---

# Mapper V2.1

Mapper V2.1 keeps the Mapper V2 global-context architecture goal, but replaces
the V1/V2 dense 4-lane tuple event vocabulary with an osuT5-like sparse mania
event stream.

## Sparse Vocab Contract

- `PAD`, `BOS`, `EOS` stay as special tokens.
- `TS_*` tokens keep the current 10ms relative time-shift contract.
- Lane events are single-lane tokens:
  - `LANE_1_TAP`
  - `LANE_1_HOLD_START`
  - `LANE_1_HOLD_END`
  - same pattern for lanes 2 through 4.
- There is no `LANE_X_NONE` token. A lane with no action at the current time is
  represented by absence of a lane token.
- Same-time chords are encoded as consecutive lane tokens in ascending lane
  order after one time shift. Example: lane 1 tap and lane 3 LN end at 1240ms
  becomes `TS_<delta_to_1240> LANE_1_TAP LANE_3_HOLD_END`.

## Frozen Canonical Representation

For each timestamp, Mapper V2.1 has exactly one canonical sparse form:

1. Emit the canonical relative `TS_*` run from the previous timestamp to the new
   timestamp.
2. Emit non-NONE lane actions in strictly ascending lane order: lane 1, lane 2,
   lane 3, lane 4.
3. Omit lanes whose action is `NONE`.
4. Do not emit the same lane twice before the next `TS_*`.
5. For ordinary windows, do not emit lane-action tokens after `write_end_ms`.
   For terminal padded windows, `chart_end_ms` is the true final oracle
   hitobject timestamp; final lane actions and `EOS` are allowed at
   `chart_end_ms`, not at the padded `write_end_ms`.

That means a same-time chord like `(TAP, NONE, HOLD_END, NONE)` is always
`LANE_1_TAP LANE_3_HOLD_END`, never reversed, duplicated, or padded with a
`NONE` token. Replay and grammar must enforce this order so a chart cannot have
multiple equivalent tokenizations.

Current initialized vocab size is 37:

- 3 specials;
- 22 time shifts;
- 12 lane-action tokens.

## Implementation Surface

- `vocab.py`: sparse lane-action vocabulary and 10ms time-shift helpers.
- `tokenizer.py`: mapper timepoint/hitobject conversion to sparse token
  sequences. It emits one canonical time-shift run per timestamp, then sorted
  non-NONE lane tokens.
- `replay.py`: sparse lane-action replay with same-time bookkeeping through
  `emitted_lane_mask` and `last_lane_index`.
- `grammar.py`: hard masks for sparse token legality, including open-LN
  legality, duplicate same-time lane prevention, terminal carry matching, and
  EOS gating.
- `adapters.py`: state-prior and LN-close adapters that project directly to
  `LANE_X_*` token ids instead of tuple EVENT ids.
- `loss.py`: sparse-token CE helpers, lane-level LN-close auxiliary loss, and
  density expectation using sparse lane-token onset weights.
- `model.py`: Mapper V2 global-context decoder wired to the v2.1 sparse vocab,
  adapters, replay-state tensors, and grammar mask.
- `train/stage_2/data/mapper_v2_1_windows.py`: v2.1 window dataset/collate path
  that reuses the Stage 2 control-window plumbing while preserving sparse
  same-time replay state.
- `train/stage_2/training/mapper_v2_1.py`: v2.1 Phase B real-data training
  entrypoint.
- `__init__.py`: package exports for the v2.1 API.

## Training

The v2.1 trainer is real-data only:

```bash
uv run python -m train.stage_2.training.mapper_v2_1 --config train/stage_2/training/configs/stage2_mapper_v2_1_phase_b_sparse_global_mps.yaml
```

The initial MPS config points at the existing `plus_end` mapper index and the
cached `stage2_control_demo_global_d384_l3_stride16_b6` teacher outputs, but it
uses a v2.1-specific mapper-record cache path. Do not reuse the V1
`window_records` cache path for v2.1 because the sparse tokenizer has a
different validity contract and sequence-length distribution.

Generation/inference is intentionally not part of this module yet. The model
hides the inherited V2 incremental decoder because that path uses the old tuple
event grammar and is invalid for sparse v2.1 lane tokens.

## Expected Impact

The vocab drops from 280 tokens to 37 tokens, but sequence length can grow
because a chord can use up to four lane tokens at one timestamp. In the worst
case, event-token count grows by up to 4x, so `max_seq_len`, batching, and
decoder attention cost need to be rechecked before full training.
