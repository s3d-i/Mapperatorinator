# Event Space Audit: 4K 2.0-6.0 Stage 1

Date: 2026-04-21

Spec: `docs/superpowers/specs/2026-04-21-oracle-timing-4k-mapper-2to6-design.md`

## Scope

This audit evaluates the Stage 1 lane-action vocabulary decision for the oracle-timing 4K mapper.

Input corpus:

- Index: `train/artifacts/indexes/beatmap_index_4k.parquet`
- Beatmap root: `mania-dataset/`
- Difficulty filter: `2.0 <= difficulty <= 6.0`
- Key count: 4K only

Canonical preprocessing used by the audit:

1. Parse osu!mania hitobjects.
2. Require at least one valid red timing point.
3. Reject negative-time hitobjects.
4. Quantize hitobject times with deterministic 10ms half-up rounding.
5. Normalize zero-length holds after quantization into taps.
6. Merge same-timestamp lane actions into canonical timepoint events.
7. Count `END_TAP` and `END_START` events before choosing the lane-action vocabulary.

## Command

```bash
uv run python -c "from pathlib import Path; import pandas as pd; from train.stage1_oracle.events.canonical import LaneAction; from train.stage1_oracle.audits.event_space import audit_osu_event_space; df=pd.read_parquet('train/artifacts/indexes/beatmap_index_4k.parquet'); df=df[(df['difficulty']>=2.0)&(df['difficulty']<=6.0)]; paths=[Path('mania-dataset')/str(row.shard)/row.beatmap_path for row in df.itertuples()]; r=audit_osu_event_space(paths, top_k=20); es=r.event_space; print('eligible_maps', len(paths)); print('total_map_count', r.total_map_count); print('audited_map_count', r.audited_map_count); print('missing_red_timing_map_count', r.missing_red_timing_map_count); print('negative_time_hitobject_map_count', r.negative_time_hitobject_map_count); print('unsupported_compound_map_count', r.unsupported_compound_map_count); print('zero_length_hold_normalized_count', r.zero_length_hold_normalized_count); print('total_timepoints', es.total_timepoints); print('total_non_empty_lane_actions', es.total_non_empty_lane_actions); print('action_counts', {a.value: es.action_counts[a] for a in LaneAction}); print('end_tap_frequency', es.end_tap_frequency); print('end_start_frequency', es.end_start_frequency); print('same_lane_compound_event_count', es.same_lane_compound_event_count); print('same_lane_compound_event_frequency', es.same_lane_compound_event_frequency); print('four_state_unsupported_map_count', es.four_state_unsupported_map_count); print('four_state_unsupported_lane_action_count', es.four_state_unsupported_lane_action_count); print('four_state_unsupported_lane_action_rate', es.four_state_unsupported_lane_action_rate); print('top_20_event_coverage', es.top_k_event_coverage); print('rare_event_count', es.rare_event_count)"
```

## Result

```text
eligible_maps 11047
total_map_count 11047
audited_map_count 11044
missing_red_timing_map_count 0
negative_time_hitobject_map_count 0
unsupported_compound_map_count 3
zero_length_hold_normalized_count 12
total_timepoints 12622282
total_non_empty_lane_actions 20632082
action_counts {'NONE': 0, 'TAP': 14973282, 'HOLD_START': 2829400, 'HOLD_END': 2829399, 'END_TAP': 1, 'END_START': 0}
end_tap_frequency 4.8468205971651336e-08
end_start_frequency 0.0
same_lane_compound_event_count 1
same_lane_compound_event_frequency 4.8468205971651336e-08
four_state_unsupported_map_count 1
four_state_unsupported_lane_action_count 1
four_state_unsupported_lane_action_rate 4.8468205971651336e-08
top_20_event_coverage 0.7769953959196919
rare_event_count 1
```

## Decision

Stage 1 should use the 4-state lane-action vocabulary:

- `NONE`
- `TAP`
- `HOLD_START`
- `HOLD_END`

The 6-state vocabulary is rejected for Stage 1. In the audited 2.0-6.0 4K corpus, `END_TAP` appears once across 20,632,082 non-empty lane actions, and `END_START` does not appear. Keeping 6-state support would add decoder vocabulary and legality complexity for one observed lane action.

The Stage 1 dataset should filter:

- the 1 map containing canonical `END_TAP`
- the 3 maps with unsupported same-lane compounds that are not representable by either the 4-state or 6-state vocabulary

Zero-length holds normalized after quantization are retained as taps and counted separately.
