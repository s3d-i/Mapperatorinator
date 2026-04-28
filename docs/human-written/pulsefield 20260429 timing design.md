---
commit: 144db8f516dea3049c2b9af29e32a316c5b6c7bb
status: valid
date: 20260429
---

currently we have:
dense timing representation:
```
beat_pulse
beat_phase_sin
beat_phase_cos
local_bpm_log_norm
timing_confidence
```

## issues
a)
inconsistency inside the dense timing representation would mislead mapper decoder and control planner
should not train direct from audio to dense timing

solution:
```
BeatThisTimingProvider(https://github.com/CPJKU/beat_this) (use final0 model with dbn=0)
  -> Audio2Frames
  -> beat_prob[t], downbeat_prob[t] // currently we do not use downbeat

GridFitter
(beat_prob[t], downbeat_prob[t]
  -> fitted osu-red-line-like grid)
  
  -> estimate BPM / beat_length
  -> estimate offset
  -> eastimate segment boundaries for different bpms
  -> Score (should handle half/double tempo ambiguity well; represents how well the grid output explain BeatThis outputs)

DenseRenderer
  grid -> dense_timing_v2
```

b)
timing_confidence has no effective supervision target.

solution:
just drop it. reserve it will do nothing good for mapper and planner. we can still researve it in sidecar diagnostics though
  `dense_timing_v2 = [beat_pulse, phase_sin, phase_cos, local_bpm]`
