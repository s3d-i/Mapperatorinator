---
pinned_commit: e3300808bcdb53fdc8713d1a5ab4f65b8c88bb8f
status: accepted
date: 2026-05-03
---

# Stage 2 Timing Module Design Note

## Decision

Keep the current Stage 2 timing module:

```text
BeatThisTimingProvider
  -> frame-level beat/downbeat probabilities
  -> GridFitter
  -> fitted timing segments
  -> dense_timing_v2 renderer
```

The current implementation is accurate and performant enough for the BeatThis
provider and the current oracle dataset. It should remain the default unless a
new provider, dataset, or audit shows a concrete regression.

## Current Shape

`GridFitter` does not run an unconstrained exhaustive search over the full tempo
and offset space. It narrows the search with autocorrelation BPM candidates,
expanded local BPM windows, per-BPM offset candidates, downbeat refinement,
bounded split candidates, adjacent-segment merging, and tempo-alias
canonicalization.

Segmentation is intentionally pragmatic:

- fit one whole-audio segment first
- identify likely timing-change candidates from the beat/downbeat evidence
- greedily accept the split with the best weighted score improvement
- merge adjacent segments when their BPM/phase relationship is effectively the
  same
- canonicalize half/double/quarter/quadruple tempo aliases after fitting

This produces timing good enough for mapper conditioning without trying to
reconstruct every osu red timing point exactly.

## Rejected Alternatives

### Coarse-to-fine search

We tried replacing the current bounded candidate search with a coarse-to-fine
search. It is not worth carrying as the default design.

The current candidate generation already gets most of the intended benefit:
autocorrelation proposes a small set of plausible periods, the expanded BPM
window covers nearby tempos, and offset search is capped per segment. A
coarse-to-fine version adds another search policy to tune, but it does not solve
the hard cases better enough to justify the complexity. It can also make close
tempo aliases and phase-sensitive offsets harder to reason about because early
coarse decisions affect the later refinement path.

Performance data does not force this change. In the 500 unique-audio BeatThis
oracle audit, `fit_seconds` was mean `0.992749`, p95 `2.21886`, and max
`24.9247` seconds, while BeatThis prediction itself was mean `2.47256`, p95
`4.76003`, and max `53.939` seconds. The fitter is acceptable for the current
pipeline.

### Boundary plus DP/penalty segmentation

We also tried changing segmentation from the current greedy split flow to a
boundary-selection model with dynamic programming and split penalties. That
does not need to be kept.

The DP/penalty approach moves the problem into penalty calibration. With a low
penalty it accepts noisy BeatThis boundary evidence and over-splits. With a high
penalty it suppresses real timing changes. In the useful middle it mostly
reproduces what the greedy split plus merge pass already does, while adding more
implementation surface and runtime cost.

The current greedy split is easier to inspect: every accepted split must improve
the weighted segment score, and adjacent compatible segments are merged back
together. That behavior is enough for the BeatThis provider and current oracle
comparison target.

## Accuracy Position

The Stage 2 timing output is a dense conditioning signal, not a promise to
exactly clone osu red timing metadata. Oracle red timing can contain fine-grain
or mapper-specific timing structure that is unnecessary for this stage.

The current BeatThis oracle audit supports keeping the implementation:

- `mean_phase_error_ms`: mean `48.5534`, p95 `87.2669`
- `first_bpm_alias_error`: mean `3.78932`, p50 effectively `0`, p95 `34.0125`
- `beat_pulse_mae`: mean `0.137116`, p95 `0.204921`
- `fit_score`: mean `0.72916`, p95 `0.889859`

These numbers are sufficient for the present use case: generating a stable
`dense_timing_v2` track for Stage 2 mapper conditioning from BeatThis frame
probabilities.

## Maintenance Guidance

- Keep the current fitter as the baseline.
- Do not reintroduce coarse-to-fine search or DP/penalty segmentation without a
  new measured failure that the current fitter cannot address.
- Prefer diagnostics, provider-quality improvements, or narrow config tuning
  before replacing the segmentation/search architecture.
- Re-run the BeatThis oracle comparison before changing default timing behavior.
