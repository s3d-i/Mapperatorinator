## Repository Ownership Notes

- `train/` and `mania-dataset/` are new fork-era content maintained by `s3d-i`. You work on behalf of `s3d-i`.
- Most directories and files should be treated as reference material.
- If uncertain, check code ownership using git.

## Python Execution

- Run Python code in this repository with `uv run` consistently instead of calling `python` or `python3` directly.

## Performance Awareness

- Be performance-aware when writing or modifying Python, PyTorch, training, inference, data-loading, or demo scripts.
- Before introducing expensive work, estimate the likely runtime, GPU/CPU memory use, and dataset/model-size impact. Prefer bounded, demo-ready paths when the goal is to get a working model in time.
- Avoid unbounded full-dataset passes, unnecessary tensor copies, repeated model reloads, excessive logging, and accidental CPU/GPU synchronization in hot paths.
- For PyTorch work, use batching, `torch.no_grad()` or `torch.inference_mode()` for inference, appropriate device placement, and explicit cleanup where large tensors or models would otherwise stay resident.
- When a script may take a long time or consume substantial memory, say so clearly and provide a smaller smoke-test command before the full run.

## Testing

- Run tests with `uv run python -m unittest discover -s tests`.

## Documentation

- When writing documentation content, pin the repository commit used for the analysis in document frontmatter, for example `pinned_commit: <git rev-parse HEAD>`.

## Compatibility and Placeholder Policy

- Do not preserve backward compatibility unless explicitly instructed.
- Do not leave compatibility wrappers, mock interfaces, or placeholder files unless explicitly instructed.
