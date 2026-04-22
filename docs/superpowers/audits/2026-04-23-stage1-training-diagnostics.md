---
date: 2026-04-23
drafted_on: 2026-04-23
effective_on: 2026-04-23
pinned_commit: 63e885f13d82e18814707bb74def2b34bdb4357a
---

# Stage 1 Oracle Training Diagnostics

Date: 2026-04-23

Current Stage 1 evidence is not yet model-selection evidence. It only proves
limited overfit behavior, not held-out generalization or full-song rollout
reliability.

This is a diagnostics note, not a passing training gate. The goal is to prevent
the next expensive run from answering the wrong question before trusting the 1k,
overnight, or ultimate Stage 1 oracle mapper configs.

## Current Evidence

There are no saved reports for:

- `stage1_oracle_1k_18m_mps`
- `stage1_oracle_overnight_18m_mps`
- `stage1_oracle_ultimate_18m_mps`

The only saved training reports are subset overfit probes:

| Run | Device | Maps | Windows | Dropout | Steps | Final loss | Token acc | Density error | Stitched open-mask error |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `overfit_4_dropout0_3000_mps` | mps | 4 | 98 | 0.0 | 3000 | 0.0321 | 0.9921 | 0.0222 | 0.2447 |
| `overfit_32` | cpu | 32 | 701 | 0.1 | 5000 | 0.7065 | 0.7576 | 0.0363 | 0.4873 |

Interpretation:

- The model can memorize a tiny 4-map subset when dropout is disabled.
- The 32-map run is not a clean overfit result. Loss and token accuracy were
  still improving at the final step.
- Stitched boundary open-mask errors remain high even when teacher-forced token
  accuracy is strong. Full-song rollout needs separate attention from
  token-level teacher forcing.

## Dataset And Audit Facts

The no-timing-anomalies training index contains 10,977 maps in the 2.0* to 6.0*
difficulty range:

| Bin | Maps |
| --- | ---: |
| 2-3 | 3638 |
| 3-4 | 3376 |
| 4-5 | 2705 |
| 5-6 | 1258 |

The larger configs use the same training recipe shape:

| Config | Maps per bin | Steps | Eval every | Batch size | LR | Dropout | Approx params |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `stage1_oracle_1k_mps.yaml` | 250 | 12000 | 1000 | 4 | 0.0002 | 0.1 | 18.65M |
| `stage1_oracle_overnight_mps.yaml` | 1000 | 50000 | 5000 | 4 | 0.0002 | 0.1 | 18.65M |
| `stage1_oracle_ultimate_mps.yaml` | all eligible per bin | 150000 | 10000 | 4 | 0.0002 | 0.1 | 18.65M |

The token statistics audit supports the current sequence budget:

- observed max target tokens: 455
- configured max decode length: 512
- max decode length headroom: 57 tokens
- empty-window ratio by bin is roughly 4.16% to 4.79%
- configured empty-window cap is 5% per difficulty bin per epoch

The window boundary audit shows that boundary state is not a rare edge case:

- global hold-crossing boundary rate: 29.53%
- by bin: 30.65%, 32.58%, 28.44%, 23.80%

## Confirmed Problems

### 1. Evaluation Uses The Training Dataset

`run_overfit_32()` builds one `OracleWindowDataset`, then creates both the
training loader and evaluation loader from that same dataset. This is acceptable
for explicit overfit probes, but not for judging 1k, overnight, or ultimate
configs.

### 2. No Beatmap/Audio-Level Split Exists

The README mentions `train/artifacts/splits/`, but no split artifacts are
present. Any split must happen by map/audio group before window expansion. A
window-level split would leak neighboring windows from the same song into both
train and eval.

Default policy: split by audio group for model-selection evidence. Beatmap-level
splits may be used only for explicitly labeled diagnostics.

### 3. Teacher-Forced Accuracy Hides Rollout Problems

Teacher-forced loss answers: "Can the model predict the next oracle token when
fed the oracle prefix?"

Full-song inference asks: "Can the model keep its own generated state coherent
across windows?"

The high stitched open-mask error rates show that these are not equivalent.

### 4. Dropout Is Not Yet Calibrated

The tiny no-dropout run memorized. The 32-map dropout-0.1 run did not. This does
not prove dropout 0.1 is wrong, because device, subset size, steps, and model
size also differed. It does mean dropout should not be accepted blindly for the
larger configs.

### 5. Step Counts Are Not Tied To Epochs Or Tokens

The configs use raw step counts, but the effective amount of training depends on
sampled epoch size, batch size, average sequence length, and train/eval split.
Reports should at least state sampled windows per epoch, batches per epoch, and
approximate epochs completed.

## Next Valid Sequence

Do not run overnight or ultimate as model-selection evidence until these gates
are satisfied in order.

### Gate 1: Target-Architecture Overfit

Purpose: prove optimization before larger runs.

Use:

- 32 or 64 maps
- the same 18.65M-ish config as the 1k run
- fixed subset, LR, batch size, seed, and max decode length
- dropout sweep: `0.0`, `0.05`, `0.1`

Pass condition:

- teacher-forced token accuracy reaches a clear overfit level
- decode health does not fail trivially
- boundary open-mask error is materially better than the current 32-map run

If dropout `0.1` cannot overfit but `0.0` or `0.05` can, use the lower dropout
for the first split-aware 1k run.

### Gate 2: 1k Held-Out Validation

Purpose: prove a real generalization trend.

Use:

- train/eval split before window expansion
- train loader from `train_manifest`
- eval loader from `eval_manifest`
- teacher-forced validation only at the normal `eval_every` cadence

Pass condition:

- train metrics improve
- held-out validation metrics improve
- validation is clearly labeled as held-out and not mixed with train metrics

### Gate 3: Small Rollout Probe

Purpose: catch state and boundary failure before overnight.

Use:

- fixed held-out probe subset from the validation side
- greedy/stitch decode every few thousand steps
- small enough probe size that it does not dominate wall time

Pass condition:

- rollout probe does not diverge while validation improves
- EOS failures, empty outputs, density drift, and stitched boundary errors are
  visible in the report
- boundary metrics are grouped enough to tell whether failures are concentrated
  on active hold boundaries

## Minimal Required Code Changes

### 1. Add Config Keys

Current `RUN_CONFIG_KEYS` rejects unknown YAML fields, so split/probe fields must
be added before split-aware configs can run.

Add:

- `train_manifest`
- `eval_manifest`
- `rollout_probe_manifest`
- `rollout_eval_every`

Keep unknown-key rejection. The point is to support the next required fields, not
to make config parsing permissive.

### 2. Build Separate Datasets And Loaders

The training path should stop using one `OracleWindowDataset` for both training
and evaluation except in explicit overfit mode.

Target shape:

```python
train_dataset = OracleWindowDataset(..., manifest_path=train_manifest)
eval_dataset = OracleWindowDataset(..., manifest_path=eval_manifest)
rollout_probe_dataset = OracleWindowDataset(..., manifest_path=rollout_probe_manifest)

train_loader = sampled_balanced_loader(train_dataset)
eval_loader = unsampled_ordered_loader(eval_dataset)
rollout_probe_loader = unsampled_ordered_loader(rollout_probe_dataset)
```

Training can remain sampled and balanced. Held-out validation and rollout probes
should not be sampled from the train distribution.

### 3. Label Report Metrics By Source

The current report has `history` and `final`. That is tolerable for a pure
overfit run, but ambiguous once train, validation, and rollout probe metrics
exist.

Do not introduce a large report framework yet. First split the final fields:

```json
{
  "history": [],
  "final_train_teacher_forced": {},
  "final_val_teacher_forced": {},
  "final_rollout_probe": {}
}
```

The immediate goal is semantic clarity: a reader should know whether a metric
came from train teacher forcing, held-out teacher forcing, or greedy/stitch
rollout.

## Boundary Diagnostics To Add Now

Add only the two boundary metrics needed for the next decision:

- `active_boundary_exact_match`
  - Count only stitched boundaries where the oracle open-hold mask is non-zero.
  - This answers whether boundary failures are concentrated on active hold
    carryover rather than all boundaries.
- `boundary_error_by_bin`
  - Report the same boundary error metric grouped by difficulty bin.
  - This answers whether the failure is global or concentrated in specific
    difficulty ranges.

Per-lane precision/recall, auxiliary heads, loss weighting, and scheduled
stitched-prefix training may be useful later. They are not required before the
next split-aware 1k run.

## Next Engineering Slice

The next implementation slice should be:

1. accept split/probe config keys;
2. build separate train/eval/rollout probe datasets and loaders;
3. label final report metrics by source;
4. add `active_boundary_exact_match` and `boundary_error_by_bin`;
5. run Gate 1, then Gate 2, then Gate 3.

Before running overnight or ultimate, make the next run split-aware,
source-labeled, and rollout-probed. Do not expand training scale until the target
architecture passes small overfit and 1k held-out validation.
