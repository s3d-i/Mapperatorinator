> 我们采用一个分层条件生成架构。
>
> 上层是 **VAE-based control planner**，输入音频条件与 difficulty，生成整首谱面的 **低频 control track**；这里 latent 只是 planner 的内部压缩表示，真正提供给后续模块使用的是由 latent 解码得到的、按时间展开的 control 序列。
>
> 这条 control track 在加入时间步位置编码后，输入 **control encoder**，被编码为一串可供检索的 **control memory**。
>
> 同时，mel 等音频特征输入 **audio encoder**，得到对应的 **audio memory**。
>
> 下层是 **seq2seq transformer decoder**，它自回归生成谱面 token。decoder 在每一步生成时，基于自身当前 state：
>
> - 对历史已生成 token 做 self-attention
> - 对 audio encoder 输出的 audio memory 做 cross-attention
> - 对 control encoder 输出的 control memory 做 cross-attention
>
> decoder 的 hidden states 上接两个输出分支：
>
> - **token head**：作为主输出头，预测谱面 token
> - **control prediction head**：作为辅助监督头，预测窗级 control feature，用于训练时提供可导的 control supervision，避免直接从离散生成结果反算 control feature 所带来的不可导问题

训练数据集位于 `mania-dataset/` 目录。

## Stage 1 directory layout

`train/stage1_oracle/` contains the implementation for the oracle-timing 4K mapper described in
`docs/superpowers/specs/2026-04-21-oracle-timing-4k-mapper-2to6-design.md`.

- `core/`: shared difficulty logic
- `osu/`: `.osu` metadata, timing, and hitobject parsing
- `events/`: canonical quantized events
- `features/`: audio loading
- `data/`: index building and dataset loading
- `audits/`: implemented pre-training audits

Generated files live under `train/artifacts/`:

- `indexes/`: beatmap indexes, including `beatmap_index_4k.parquet`
- `cache/`: mel, timing-track, and token caches
- `reports/`: audit and evaluation reports
- `runs/`: checkpoints and experiment outputs
- `splits/`: train/validation split manifests

Add new Stage 1 packages only when they contain real implementation files.

## Feature family v1

- A. 强度族：描述整体压力与密度
- B. 占用族：描述资源持续占用状态
- C. 重复/风险族：描述坏模式风险

当前使用的 control feature 包括：

- `density_env`
- `hold_occupancy`
- `chord_rate`
- `jack_risk`
- `hand_balance_ema`
- `repeat_risk`
