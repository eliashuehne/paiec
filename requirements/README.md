# Dependency environments

All model environments use Python 3.12 and an editable install from the repo
root. These locks preserve the tested package versions; moving them here did
not change a dependency. `pyproject.toml` declares the library and optional
extras; the locks record complete environment snapshots.

| File | Environment | Scope |
| --- | --- | --- |
| `mlx.lock` | `.venv-laya-mlx` | macOS Apple Silicon; MLX inference and tracking, no Torch/Transformers |
| `torch.lock` | `.venv-laya` | macOS reference PyTorch/MPS environment, training and tracking |
| `modal-client.lock` | `.venv-modal` | Local Modal SDK and tracking; no GPU framework required |
| `modal-worker.in` | Modal image | Direct Python 3.12 Linux/CUDA requirements |
| `modal-worker.lock` | Modal image | Resolved Linux x86_64/CUDA dependencies, installed by the adapter |
| `archive/laya-pretracking.lock` | Historical only | Earlier Laya snapshot before MLflow dependency adjustments |

For example, from the repository root:

```bash
uv venv --python 3.12 .venv-laya-mlx
uv pip install --python .venv-laya-mlx/bin/python -r requirements/mlx.lock -e .
```

Use `requirements/torch.lock` for the reference environment or
`requirements/modal-client.lock` for the Modal client. The CUDA worker lock is
for its Linux image; do not install it on a Mac. The archived lock is retained
for historical reproduction and is not the current Torch environment.

For core development on a fresh checkout, install `-e '.[test,tracking]'` in a
separate environment. The optional `test` extra enables SciPy comparison checks.
These tests do not need model weights, local datasets, Modal credentials or GPUs.

When dependencies change, resolve them in the intended platform environment,
verify that backend there, and update its lock with the change. Do not combine
the MLX and Torch environments or silently resolve a worker image from a Mac
lock. A dependency update does not require a new input-format version; renderer
or scientific recipe changes do. See [CONTRIBUTING.md](../CONTRIBUTING.md).

Previous names: `requirements-dev.lock` → `torch.lock`,
`requirements-mlx.lock` → `mlx.lock`, `requirements-modal-client.lock` →
`modal-client.lock`, and `requirements-modal-worker.*` → `modal-worker.*`.
The old `requirements-laya.lock` is now `archive/laya-pretracking.lock`.
