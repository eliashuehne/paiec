# Shared data and features

For training Laya, I prepared examples from the public measurement database
that combine a question, the target system's recorded setup and its observed
outcome. The idea was to predict whether a particular system would succeed on
a question using information about both the system and the task.

I also collected documented model and harness information, such as architecture,
parameter counts and supported capabilities, where available. These facts are
stored separately from the original dataset, with sources and unknown values
preserved.

## What I prepared

The shared dataset contains **34,718 training examples across seven benchmark
families and 170 system–benchmark combinations**. Several examples can involve
the same question answered by different systems.

| Data | Contents |
| --- | --- |
| System information | Supplied model name, provider, dates, harness and recorded settings |
| Task information | Full question text, supplied task attributes and interaction context |
| Outcomes | Original binary observations, stored separately from the inputs; their meaning depends on the dataset |
| Grouping metadata | Links between examples, systems and tasks, plus family/task groups for validation |
| Statistical features | A **34,718 × 467** matrix derived from the supplied system and task fields |

The bundle also includes 1,000 monitor and 963 original-development input rows,
without their outcomes. Calibration and confirmation data are not included.

## How this could help other approaches

The tables and grouping metadata can provide a common starting point for
logistic regression or other models, without repeating the data preparation.
Each approach can then choose its own representation and preprocessing.

Alongside the Laya inputs, I built a statistical representation with text
features, input lengths, dates, nominal model size, missing-value indicators
and system–task interactions. The 467-feature matrix is ready to load for a
statistical baseline. Laya used a separate text representation, not this matrix.
The feature code is in [conditional_features.py](../src/paiec/conditional_features.py).

The shared matrix uses only the original supplied fields. The additional
model/harness research facts are **not included** in it or the starter bundle;
they are available separately for research, subject to eligibility review.
See [collected information](laya-input-contract.md).

## Access and loading

Code is on the `laya` branch. The data is shared separately as
`statistical_starter_v1.tar.gz` (about 78 MiB), with a checksum receipt and an
artifact manifest. Reading it requires only the core Python package.

```bash
uv venv --python 3.12 .venv-stats
uv pip install --python .venv-stats/bin/python -e .
mkdir -p runs/shared_data
tar -xzf /path/to/statistical_starter_v1.tar.gz -C runs/shared_data
```

The archive contains `dataset/` (system/task tables, example links and training
labels), `features/` (matrix and feature contract), and `training_metadata.jsonl`
(grouping information). This example loads the training inputs and labels in
their verified row order:

<details>
<summary>Python loading example</summary>

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

</details>

The family/task groups support validation without splitting related questions
across fitting and validation folds. The helper
`conditional_analysis.family_training_mask` handles these exclusions. Any
learned scaling, vocabulary or feature selection belongs inside the fitting
fold; grouping identifiers stay outside the model inputs.

Further details: [data sources](data-notes.md), [split assignments](head-data-split.md),
[preparation workflows](workflows.md) and [Laya experiment record](grouped-enriched-training.md).
