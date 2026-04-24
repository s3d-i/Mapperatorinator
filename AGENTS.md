## Repository Ownership Notes

- `train/` and `mania-dataset/` are new fork-era content maintained by `s3d-i`. You work on behalf of `s3d-i`.
- Most directories and files should be treated as reference material.
- If uncertain, check code ownership using git.

## Python Execution

- Run Python code in this repository with `uv run` consistently instead of calling `python` or `python3` directly.

## Testing

- Run tests with `uv run python -m unittest discover -s tests`.

## Documentation

- When writing documentation content, pin the repository commit used for the analysis in document frontmatter, for example `pinned_commit: <git rev-parse HEAD>`.

## Compatibility and Placeholder Policy

- Do not preserve backward compatibility unless explicitly instructed.
- Do not leave compatibility wrappers, mock interfaces, or placeholder files unless explicitly instructed.
