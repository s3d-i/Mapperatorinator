---
pinned_commit: f4792ff691f8198a60e27aae82b88aded42b8dc3
date: 2026-05-10
device: mps
---

# Stage 2 Inference Runtime Benchmarks

Session cache policy:

- Keep real audio features: `full_mel`, `full_dense_timing_v2`, `padding_mask`, fitted timing grid.
- Keep control model memory: `control_memory_8s`.
- Do not keep training teacher density level in session cache.

Model/runtime setup:

- Mapper checkpoint: `train/artifacts/runs/stage2_mapper_v2/stage2_mapper_v2_phase_b_global_d768_l8_b1/checkpoint.pt`
- Control checkpoint: `train/artifacts/runs/stage2_control_demo/stage2_control_demo_global_d384_l3_stride16_b6/checkpoint.pt`
- Control memory shape per 8s window: `[400,384]`
- Full-song control output shape: `[window_count,400,384]`

## Full-Song Control

Audio is prepared once, then full-song control is generated over 8s windows.

| audio | windows | sequential total | batched max 12 total |
|---|---:|---:|---:|
| `mania-dataset/0/1047817/audio.mp3` | 41 | 20.269s | 14.490s |
| `mania-dataset/0/2183073/audio.ogg` | 10 | 3.036s | 1.244s |
| `mania-dataset/0/275817/Rolling Star.mp3` | 24 | 7.776s | 4.268s |

## Single Batch Size Sweep

Single-batch micro-benchmark on `mania-dataset/0/1047817/audio.mp3` with cached audio features, repeated 4 times.

| batch size | mean batch time | mean per window | windows/s |
|---:|---:|---:|---:|
| 1 | 0.155s | 0.155s | 6.45 |
| 2 | 0.446s | 0.223s | 4.48 |
| 4 | 0.729s | 0.182s | 5.49 |
| 8 | 1.900s | 0.237s | 4.21 |
| 12 | 3.732s | 0.311s | 3.22 |

Current conclusion: use batch size 1 for lowest streaming latency, and batch size 4 as the first default for full-song pre-generation on MPS. Batch size 12 reduces call count but is slower per window in the single-batch benchmark.
