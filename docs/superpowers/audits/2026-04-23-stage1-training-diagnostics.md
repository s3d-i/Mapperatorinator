---
date: 2026-04-23
pinned_commit: 8b4b448f1929fbc19a4c8adece383d4611e47bc0
---

# Stage 1 Oracle Training Diagnostics

Date: 2026-04-23

This document summarizes the current Stage 1 oracle mapper training evidence,
the problems visible in saved artifacts, and the experiment/code changes needed
before treating the larger training configs as reliable.

## Scope

This is a diagnostics note, not a passing training gate. It covers:

- current Stage 1 training configs under `train/stage1_oracle/training/configs/`
- saved run reports under `train/artifacts/runs/stage1_oracle/`
- audit artifacts under `train/artifacts/reports/audits/`
- training/evaluation behavior in `train/stage1_oracle/training/overfit_32.py`

The goal is to decide what must change before running and trusting the 1k,
overnight, or ultimate Stage 1 oracle mapper runs.

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
- Stitched boundary open-mask errors are high even when teacher-forced token
  accuracy is strong. This is the clearest signal that full-song rollout needs
  separate attention from token-level teacher forcing.

## Dataset And Audit Facts

The no-timing-anomalies training index contains 10,977 maps in the 2.0* to 6.0*
difficulty range:

| Bin | Maps |
| --- | ---: |
| 2-3 | 3638 |
| 3-4 | 3376 |
| 4-5 | 2705 |
| 5-6 | 1258 |

The token statistics audit supports the current sequence budget:

- observed max target tokens: 455
- configured max decode length: 512
- max decode length headroom: 57 tokens
- empty-window ratio by bin is roughly 4.16% to 4.79%
- configured empty-window cap is 5% per difficulty bin per epoch

The window boundary audit shows that boundary handling is common enough to be a
core training/evaluation concern:

- global hold-crossing boundary rate: 29.53%
- by bin: 30.65%, 32.58%, 28.44%, 23.80%

This means boundary state is not a rare edge case. Roughly one quarter to one
third of window boundaries cross an active hold, depending on difficulty bin.

## Current Training Configs

The larger configs use the same training recipe shape:

| Config | Maps per bin | Steps | Eval every | Batch size | LR | Dropout | Approx params |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `stage1_oracle_1k_mps.yaml` | 250 | 12000 | 1000 | 4 | 0.0002 | 0.1 | 18.65M |
| `stage1_oracle_overnight_mps.yaml` | 1000 | 50000 | 5000 | 4 | 0.0002 | 0.1 | 18.65M |
| `stage1_oracle_ultimate_mps.yaml` | all eligible per bin | 150000 | 10000 | 4 | 0.0002 | 0.1 | 18.65M |

These are plausible as first large-run configs, but the saved evidence does not
prove they are good enough. The larger model has not been shown to overfit a
small subset, and there is no held-out validation report.

## Confirmed Problems

### 1. Evaluation Uses The Training Dataset

`run_overfit_32()` builds one `OracleWindowDataset`, then creates both the
training loader and evaluation loader from that same dataset. This measures
training-set fit, not generalization.

This is acceptable for explicit overfit probes, but not for judging the 1k,
overnight, or ultimate configs.

### 2. No Beatmap/Audio-Level Split Exists

The README mentions `train/artifacts/splits/`, but no split artifacts are present.
Any split must happen by map/audio group before window expansion. A window-level
split would leak neighboring windows from the same song into both train and eval.

The correct split unit should be at least beatmap-level, and preferably audio
group-level when multiple difficulties share the same audio.

### 3. Teacher-Forced Accuracy Hides Rollout Problems

Teacher-forced loss answers: "Can the model predict the next oracle token when
fed the oracle prefix?"

Full-song inference asks: "Can the model keep its own generated state coherent
across windows?"

The high stitched open-mask error rates show that these are not equivalent.
Boundary behavior needs dedicated metrics and, likely, dedicated training
pressure.

### 4. Greedy/Stitched Decode Runs Only At Final Step

The current training loop updates greedy decode metrics only at the final step.
That makes it hard to see when rollout quality begins improving, regressing, or
diverging from teacher-forced validation.

Teacher-forced validation can remain frequent and cheap, but rollout diagnostics
need their own cadence on a smaller fixed probe set.

### 5. Dropout Is Not Yet Calibrated

The tiny no-dropout run memorized. The 32-map dropout-0.1 run did not. This does
not prove dropout 0.1 is wrong, because device, subset size, steps, and model
size also differed. It does mean dropout should not be accepted blindly for the
larger configs.

### 6. Step Counts Are Not Tied To Epochs Or Tokens

The configs use raw step counts, but the effective amount of training depends on
sampled epoch size, batch size, average sequence length, and train/eval split.

The reports should state:

- sampled windows per epoch
- batches per epoch
- approximate epochs completed
- approximate non-pad target tokens processed

Without this, comparing 12k, 50k, and 150k steps is too coarse.

## Recommended Design Direction

Separate the training system into four evaluation modes:

1. **Overfit mode**
   - Train and eval on the same small fixed subset.
   - Purpose: prove optimization, model capacity, LR, and dropout.
   - Success criterion: near-perfect teacher-forced token accuracy and low
     greedy/stitch errors on 32 or 64 maps.

2. **Validation mode**
   - Train on train split, evaluate teacher-forced metrics on held-out split.
   - Purpose: track generalization cheaply during training.
   - Success criterion: validation loss and token accuracy improve without
     obvious overfit divergence.

3. **Rollout probe mode**
   - Greedy/stitch decode a small fixed held-out probe set every N steps.
   - Purpose: track generation stability and boundary state.
   - Success criterion: open-mask error, invalid transitions, empty outputs,
     EOS failures, and density error improve together.

4. **Candidate test mode**
   - Run only for selected checkpoints on an untouched test split.
   - Purpose: compare candidate configs without tuning on the test set.

## Required Code Changes

### A. Split Manifest Generation

Add a split builder that reads the clean 4K index and writes a deterministic
manifest under `train/artifacts/splits/`.

Requirements:

- stratify by difficulty bin
- split by audio group or beatmap group before window expansion
- record seed, source index path, counts by bin, map counts, and audio counts
- produce at least `train`, `val`, and optionally `test` partitions
- ensure no audio path appears in more than one partition when audio grouping is
  enabled

Suggested initial split:

- train: 90%
- val: 5%
- test: 5%

For early iteration, a smaller fixed validation/probe subset can be derived from
the validation partition to keep greedy decode cost bounded.

### B. Dataset Split Filtering

Extend the training entrypoint so configs can specify split manifests:

- `train_manifest`
- `eval_manifest`
- `rollout_probe_manifest`

The existing `OracleWindowDataset(manifest_path=...)` hook can be reused, but
the current training path does not expose it.

### C. Separate Train, Eval, And Probe Loaders

The training entrypoint should build:

- a sampled/balanced train loader
- an unsampled validation loader
- an optional small rollout probe loader

The report should identify which split each metric came from. Avoid using a
single field called `final` for mixed train, validation, and rollout metrics
without labels.

### D. Boundary Diagnostics

Add explicit boundary metrics:

- generated final open-hold mask accuracy per window
- exact open-mask match rate at each stitched boundary
- per-lane hold-open precision/recall at boundary
- boundary metrics only on windows where oracle open mask is non-zero
- metrics by difficulty bin
- metrics by whether the previous oracle window had a crossing hold

The current `stitched_boundary_open_mask_error_rate` is useful, but too blunt.
It says the carried model mask differs from oracle, not which lane/action caused
the divergence.

### E. Rollout Evaluation Cadence

Add separate cadences:

- `eval_every`: teacher-forced validation cadence
- `rollout_eval_every`: greedy/stitch probe cadence
- `save_every`: checkpoint cadence

Recommended default for 1k experiments:

- teacher-forced eval every 500 to 1000 steps
- rollout probe every 2000 to 5000 steps
- save every 1000 to 5000 steps, depending on disk budget

For ultimate training, keep teacher-forced eval moderate and rollout probe small.
Full greedy validation over all held-out windows will be too expensive to run
frequently.

### F. Training Progress Accounting

Add report fields:

- train sampled epoch window count
- train batches per epoch
- completed sampled epochs
- estimated target tokens per batch
- estimated target tokens processed
- validation window count
- rollout probe window count

This makes step counts interpretable.

### G. Dropout Sweep Support

Keep dropout configurable, but treat it as an experimental variable:

- `0.0`: optimization control
- `0.05`: conservative regularization candidate
- `0.1`: current large-run default

Do not run expensive full training until the target architecture can overfit a
32/64-map subset at the chosen dropout and LR.

## Experiment Plan

### Experiment 1: Target-Architecture Overfit Probe

Purpose: prove the 18.65M config can optimize the representation.

Run the 32-map or 64-map subset using the same model shape as the 1k config:

- `d_model: 320`
- `heads: 5`
- `encoder_layers: 5`
- `decoder_layers: 7`
- `ffn_dim: 1280`

Sweep:

- dropout: 0.0, 0.05, 0.1
- keep LR fixed initially at 0.0002

Expected pass:

- teacher-forced token accuracy above 98%
- no EOS failures
- low density error
- materially lower boundary open-mask error than current 32-map run

If dropout 0.1 cannot overfit but 0.0 or 0.05 can, use the lower dropout for
the first 1k run.

### Experiment 2: 1k Split Validation Run

Purpose: establish real generalization behavior before overnight training.

Use the 1k config shape, but train on a real train split and evaluate on a
held-out validation split.

Track:

- train teacher-forced loss/accuracy
- validation teacher-forced loss/accuracy
- rollout probe metrics
- boundary diagnostics
- metrics by difficulty bin

Expected pass:

- validation improves steadily
- rollout probe does not regress while teacher-forced validation improves
- boundary metrics improve across checkpoints

### Experiment 3: Boundary-Focused Ablations

Purpose: determine whether open-mask errors are model, loss, data, or decode
issues.

Run ablations after Experiment 2 if boundary metrics remain poor:

- increase sampling weight for windows with non-zero open masks
- add auxiliary final-open-mask prediction head
- add loss weighting for `HOLD_START` and `HOLD_END` event tokens
- add scheduled stitched-prefix training on short two-window sequences
- compare constrained greedy with stricter boundary-aware decoding rules

Prefer training-signal fixes before decode-only fixes unless diagnostics show
the model is already mostly correct and only needs legality cleanup.

### Experiment 4: Overnight Candidate Run

Purpose: scale only after split validation and boundary diagnostics are stable.

Use `stage1_oracle_overnight_mps.yaml` as the base, adjusted for the chosen
dropout and new split/probe fields.

Run only if:

- target-architecture overfit passes
- 1k validation run has useful held-out metrics
- boundary metrics are tracked and improving

### Experiment 5: Ultimate Run

Purpose: train the strongest oracle-conditioned baseline.

This should be the last step, not the next proof step. The ultimate run is too
expensive to use as the first debugging surface.

## Initial Config Guidance

For the next serious run, prefer:

- keep `max_decode_len: 512`
- keep empty-window cap at `0.05`
- start with dropout `0.05` unless the target-architecture overfit sweep says
  `0.1` is safe
- keep LR `0.0002` for the first controlled comparison
- add split manifests before judging validation quality
- add rollout probe metrics before judging full-song quality

Do not change too many variables at once. First isolate optimization, then
generalization, then rollout stability.

## Success Criteria Before Large Training

Before treating the larger configs as good enough:

- target architecture overfits 32/64 maps cleanly
- train/eval split has no audio leakage
- validation metrics are reported separately from train metrics
- rollout probe metrics run during training, not only at the final step
- boundary open-mask diagnostics identify which lane/action failures dominate
- the selected dropout is justified by a small sweep
- step counts are reported with epoch/token context

## Resolved Open Questions

### Primary Split Unit

Use audio-group splits as the default for validation and test manifests. The
split key should be the dataset-relative audio path, including shard, so every
beatmap that uses the same audio lands in the same partition.

This is stricter than beatmap-level splitting and is the right default because
Stage 1 validation should measure generalization to unseen audio/timing context,
not only unseen difficulty files for familiar songs. Beatmap-level splits can be
kept for explicit diagnostics if needed, but they should not be used for the
main train/val/test gate.

### Boundary Error Threshold

Use provisional boundary gates by evaluation mode until the expanded diagnostics
show which lane/action failures dominate:

- overfit gate: `stitched_boundary_open_mask_error_rate <= 0.05` on the fixed
  32/64-map training subset, with zero EOS failures and no max-decode-length
  failures
- 1k validation candidate gate: `stitched_boundary_open_mask_error_rate <= 0.15`
  globally on the fixed held-out rollout probe and `<= 0.25` in every difficulty
  bin
- overnight/ultimate candidate gate: target `<= 0.10` globally and `<= 0.20` in
  every difficulty bin, plus improving per-lane boundary precision/recall once
  those metrics exist

These thresholds are intentionally stricter for overfit mode. If the model
cannot carry open-hold state correctly on a memorized subset, larger held-out
training runs are not yet meaningful.

### Rollout Probe Size On MPS

Use a fixed ordered rollout probe of whole audio groups, not random windows.
Start with the smallest deterministic validation-derived audio-group subset that
produces roughly 150 to 250 windows, then run it every 2,000 to 5,000 training
steps for 1k experiments.

Also keep a larger checkpoint probe of roughly 500 to 1,000 ordered windows for
less frequent checkpoint comparisons. After the first measured MPS run, adjust
probe size or cadence so greedy/stitch rollout stays under roughly 10% to 15%
of total wall time. If rollout exceeds that budget, reduce probe size before
relaxing the cadence enough to lose trend visibility.

## Recommended Next Change Set

Implement one focused change set before further hyperparameter tuning:

1. split manifest builder
2. config fields for train/eval/probe manifests
3. separate train validation metrics in reports
4. rollout probe cadence
5. expanded boundary diagnostics
6. report epoch/token accounting

After that, run the target-architecture overfit sweep and use the result to
choose dropout for the 1k validation run.
