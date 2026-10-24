# Shared data and features

Use this data preparation for logistic regression or other predictors. Share
examples, labels and split assignments; keep model-specific preprocessing separate.
Reading the prepared bundle requires the core Python package, without Laya,
Torch, MLX, a GPU or a running ZenML service.

## Get started

Code is on the `laya` branch. Generated datasets are Git-ignored and must be
transferred separately. The prepared archive is
`runs/shared_data/statistical_starter_v1.tar.gz` (about 78 MiB); its checksum is
in `statistical_starter_v1_receipt.json`. The extracted bundle contains an
artifact hash manifest.

```bash
uv venv --python 3.12 .venv-stats
uv pip install --python .venv-stats/bin/python -e .
mkdir -p runs/shared_data
tar -xzf /path/to/statistical_starter_v1.tar.gz -C runs/shared_data
```

## Available data

| Resource | Contents |
| --- | --- |
| `dataset/systems.jsonl` | Eight supplied system fields, keyed by `system_key` |
| `dataset/tasks.jsonl` | Full question text, supplied attributes and interactors, keyed by `task_key` |
| `dataset/examples.jsonl` | Row keys, system/task joins, role and grouping metadata |
| `dataset/training_targets.jsonl` | Native binary training outcomes, keyed by `row_key` |
| `features/feature_matrix.npy` | **34,718 × 467** numeric/text features; no fitted estimator |
| `training_metadata.jsonl` | Aligned family, scope and task-group metadata for splitting/weighting; exclude from model features |

The bundle has **34,718 training rows across seven families and 170 scopes**.
A scope is a target system/configuration paired with a benchmark, not a distinct
model. It also contains 1,000 monitor and 963 original-development input rows,
without their outcomes. Calibration/confirmation data is excluded.

## Load training inputs and labels

Run from the repository root after extracting the archive:

```python
from pathlib import Path
import hashlib
import json
import numpy as np
from paiec.feature_preparation import read_jsonl, stable_hash

bundle = Path("runs/shared_data/statistical_starter_v1")
manifest = json.loads((bundle / "bundle_manifest.json").read_text())
for name, expected in manifest["artifacts"].items():
    with (bundle / name).open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == expected

rows = [r for r in read_jsonl(bundle / "dataset/examples.jsonl")
        if r["role"] == "training"]
targets = {r["row_key"]: r["native_binary_value"]
           for r in read_jsonl(bundle / "dataset/training_targets.jsonl")}
metadata = list(read_jsonl(bundle / "training_metadata.jsonl"))
contract = json.loads((bundle / "features/feature_contract.json").read_text())
assert [r["row_key"] for r in rows] == [r["row_key"] for r in metadata]
assert stable_hash([r["row_key"] for r in rows]) == contract["row_order_sha256"]
assert set(targets) == {r["row_key"] for r in rows}

X = np.load(bundle / "features/feature_matrix.npy", mmap_mode="r", allow_pickle=False)
y = np.asarray([targets[r["row_key"]] for r in rows], dtype=int)
assert X.shape == (len(y), contract["dimensions"]) == (34718, 467)
assert np.isin(y, [0, 1]).all()
```

## Features and evaluation

[conditional_features.py](../src/paiec/conditional_features.py) produces 256
hashed task-text features, 64 system-text features, 19 numeric/missingness
features and 128 system–task interactions. It preserves full text and uses no
external research facts. The matrix matches the encoder hash in its contract.
The existing [ridge estimator](../src/paiec/conditional_training.py) is not
logistic regression and failed its quality gates; reuse does not establish
predictive quality.

Keep native label meanings: binary outcomes are not universally correctness.
Do not fill missing model identities from private display names or use IDs,
reference answers, grading rules or outcome-derived scores as input features.
Use family/task-group holdouts, not random row splits; the helper
`conditional_analysis.family_training_mask` excludes overlapping task groups.
Fit scaling, vocabularies, feature selection and hyperparameters on fitting
folds only. Agree on sample weighting and coordinate development evaluation.

Optional documented model/harness facts are a separate **research-only** layer;
see [collected inputs](laya-input-contract.md). They are excluded from this
starter. Laya token windows and checkpoints are unnecessary for logistic regression.

For original data, frozen role assignments and reconstruction dependencies, see
[public data](data-notes.md), [frozen splits](head-data-split.md) and
[workflows](workflows.md). The canonical table bundle is
`runs/competition_pipeline/protocol_panel_v2/`. A fresh checkout does not contain
its generated source dependencies. The portable starter can be read directly;
some original provenance helpers require the original preparation paths.
