# Tests

From the repository root:

```bash
make check PYTHON=.venv-laya-mlx/bin/python
make test PYTHON=.venv-laya/bin/python
```

The core suite needs only tracked files, the initialized `baseline/` submodule,
and core Python dependencies. The optional `test` extra enables SciPy comparison
checks; `tracking` enables temporary MLflow tests. Native MLX/Torch checks run
only when their packages are available. No test downloads weights, trains a
Laya model, allocates a cloud GPU, or reads sealed data.

`fixtures.py` builds small temporary inventories for job transport and prepared
input integrity checks. Its placeholder Parquet paths are inventory-only;
no Parquet reader consumes them. Table behavior tests use their own synthetic
frames. Protocol tests cover splits, acquisition exclusions, metrics and input
boundaries; adapter tests inject fake runtimes rather than loading weights.

Saved checkpoints and real prepared exports are separate integration artifacts.
Checking their compatibility is useful after a relevant change, but is not a
prerequisite for unit tests on a fresh checkout.
