## Phase 1 — Oracle Upper-Bound Mapper

In Phase 1, the mapper is trained with oracle timing derived from ground-truth `.osu` timing data. The purpose of this phase is not to build the final deployable system, but to establish an upper bound on mapper quality under ideal timing conditions. This isolates the mapping problem from the timing-recovery problem and answers a critical question early: if timing were perfect, would the current mapper architecture, representation, and training setup already be good enough?

This phase provides a clean reference point for later comparison. If mapper quality is weak even with oracle timing, then the main bottleneck is the mapper itself rather than the timing module. Conversely, if the mapper performs well with oracle timing, then later quality drops under predicted timing can be attributed to timing error and train–test mismatch.

The key output of this phase is a strong oracle-conditioned mapper checkpoint together with its evaluation results. This checkpoint should be treated as an analysis baseline and quality ceiling, not as the final inference-time model.

---

## Phase 2 — Timing-Noise Robustness Fine-Tuning

In Phase 2, the oracle-trained mapper is further fine-tuned under controlled timing perturbations. The goal is to reduce over-reliance on perfect timing and improve robustness to realistic timing errors that will appear at inference time. These perturbations may include small phase jitter, local section drift, missing timing confidence, or mildly corrupted timing tracks.

This phase is necessary because a mapper trained only on ground-truth timing will implicitly assume that timing is always clean, stable, and perfectly aligned. In practice, a predicted timing model will introduce noise, ambiguity, and confidence variation. Without robustness training, even a relatively accurate timing model may cause a large downstream quality drop.

The key output of this phase is a timing-aware mapper that remains strong under imperfect timing conditions. This model still uses oracle-derived timing during training, but it is no longer fragile to small deviations from the ideal timing track.

---

## Phase 3 — Fast and Reliable Timing Model

In Phase 3, a standalone timing model is built to predict timing information directly from audio, without access to the reference `.osu` file at inference time. Its purpose is to generate timing that is accurate enough, reliable enough, and fast enough to support the mapper. The emphasis is on practical usability rather than exact reconstruction of every mapper-authored red timing decision.

The timing model should first target the recovery of a dense timing track or equivalent rhythmic prior that is compatible with the mapper input format. A deterministic DSP-based baseline is the preferred starting point, because it is fast, inspectable, and easy to benchmark. More advanced components, such as a small verifier or refinement model, can be added later only if the baseline reveals clear and repeated failure modes.

The key output of this phase is a deployable timing module that converts audio into mapper-consumable timing features, along with confidence signals and evaluation metrics. This phase should be evaluated not only by timing accuracy itself, but also by how well its outputs preserve downstream mapper quality.

---

## Phase 4 — Predicted-Timing Adaptation

In Phase 4, the mapper is adapted from oracle timing to predicted timing. The timing model from Phase 3 is run over the training data to produce predicted timing tracks, and the mapper is then fine-tuned using a mixture of oracle timing and predicted timing. This bridges the distribution gap between the idealized training condition and the actual inference condition.

This phase is the point where the full system becomes realistic. The mapper must learn how to interpret timing predictions that are mostly correct but imperfect, sometimes ambiguous, and sometimes low-confidence. A gradual mixing strategy is recommended, starting with a higher proportion of oracle timing and then increasing the proportion of predicted timing as training progresses.

The key output of this phase is the final inference-ready mapper stack: audio goes through the timing model, the predicted timing is rendered into the mapper input representation, and the mapper generates output under the same conditions it will face at deployment time. This phase should be judged by end-to-end performance rather than by isolated timing or mapper metrics alone.