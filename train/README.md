

dataset is at `mania-dataset/` 

## Stage 1 directory layout

`train/stage1_oracle/` contains the implementation for the oracle-timing 4K mapper described in
`docs/superpowers/specs/2026-04-21-oracle-timing-4k-mapper-2to6-design.md`.

- `core/`: shared difficulty logic
- `osu/`: `.osu` metadata, timing, and hitobject parsing
- `events/`: canonical quantized events
- `features/`: audio loading
- `data/`: index building and dataset loading
- `audits/`: implemented pre-training audits
- `models/`: Stage 1 fused encoder / autoregressive decoder model
- `training/`: overfit and training entrypoints

Generated files live under `train/artifacts/`:

- `indexes/`: beatmap indexes, including `beatmap_index_4k.parquet`
  and the Stage 1 training index `beatmap_index_4k_no_timing_anomalies.parquet`.
  The training index is generated from the 4K index by excluding maps that fail
  the same red timing parser gate used by the dense timing audit, so training
  does not see maps with missing or anomalous red timing points. `ManiaBeatmapDataset`
  uses this training index by default.
- `cache/`: mel, timing-track, and token caches
- `reports/`: audit and evaluation reports
- `runs/`: checkpoints and experiment outputs
- `splits/`: train/validation split manifests

Add new Stage 1 packages only when they contain real implementation files.

### Stage 1 oracle training commands

These run configs train with `device: mps` and use the pre-training gate manifest at
`train/artifacts/reports/audits/pretraining_gates_stage1_4k_2to6_2026-04-22.json`.

Balanced 1k-map run:

```bash
uv run python -m train.stage1_oracle.training.overfit_32 --config train/stage1_oracle/training/configs/stage1_oracle_1k_mps.yaml
```

Longer overnight run:

```bash
uv run python -m train.stage1_oracle.training.overfit_32 --config train/stage1_oracle/training/configs/stage1_oracle_overnight_mps.yaml
```

Ultimate run using per-bin caps for every eligible map in
`train/artifacts/indexes/beatmap_index_4k_no_timing_anomalies.parquet`:

```bash
uv run python -m train.stage1_oracle.training.overfit_32 --config train/stage1_oracle/training/configs/stage1_oracle_ultimate_mps.yaml
```

Training writes a resumable latest checkpoint to `<output_dir>/checkpoint.pt` at step 1,
every `eval_every` steps by default, and at the final step. Archived step checkpoints are
stored as `<output_dir>/checkpoints/checkpoint_step_*.pt`. Use `--save-every N` to change
the checkpoint cadence, and resume a run with:

```bash
uv run python -m train.stage1_oracle.training.overfit_32 \
  --config train/stage1_oracle/training/configs/stage1_oracle_overnight_mps.yaml \
  --resume-from train/artifacts/runs/stage1_oracle/stage1_oracle_overnight_18m_mps/checkpoint.pt
```

### Stage 1 oracle inference commands

Stage 1 inference is still oracle-timing inference: pass an audio file and a reference `.osu`
file whose red timing points are rendered into the dense timing track. The current commands
are for inspection and preview, not final beatmap export.

Browser preview GUI:

```bash
uv run python -m train.stage1_oracle.inference.preview_server \
  --checkpoint-path train/artifacts/runs/stage1_oracle/overfit_32/checkpoint.pt \
  --audio-path "mania-dataset/0/1033765/audio.mp3" \
  --beatmap-path "mania-dataset/0/1033765/onumi - REGRET PART TWO (FAMoss) [ETERNAL].osu" \
  --difficulty 4.5 \
  --device cpu \
  --port 5000
```

Open `http://127.0.0.1:5000` after the server starts.

Terminal JSONL stream:

```bash
uv run python -m train.stage1_oracle.inference.stream_probe \
  --checkpoint-path train/artifacts/runs/stage1_oracle/overfit_32/checkpoint.pt \
  --audio-path "mania-dataset/0/1033765/audio.mp3" \
  --beatmap-path "mania-dataset/0/1033765/onumi - REGRET PART TWO (FAMoss) [ETERNAL].osu" \
  --difficulty 4.5 \
  --device cpu \
  --max-windows 1
```

`stream_probe` can also auto-select the newest local Stage 1 checkpoint and a default
eligible 4K map from `train/artifacts/indexes/beatmap_index_4k_no_timing_anomalies.parquet`:

```bash
uv run python -m train.stage1_oracle.inference.stream_probe --device cpu --max-windows 1
```

## Feature

for the control encoder: `train/stage1_oracle/features/control.py`