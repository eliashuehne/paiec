# Shared data and features: a starting point for Arno

Use the same recorded examples, outcome meanings and split assignments across
models, then add preprocessing appropriate to each model. Logistic regression
can use the structured tables or numeric/text vectors below. Laya's tokenizer,
8k windows and checkpoint format are specific to Laya and are not prerequisites.

## Start here

1. Get the code from the `laya` branch and install the core package. No Torch,
   MLX, downloaded Laya weights, GPU or ZenML service is needed to read prepared
   data and construct statistical features.
2. Obtain the generated **`arno_v1.tar.gz`** starter bundle from Elias. Data is
   deliberately outside Git; fetching this branch does not download it.
3. Extract it under `runs/shared_data/` and run the example below. It loads
   `X`, `y` and separate metadata; it does not fit a model or score development.

```bash
git fetch origin
# Inspect origin/laya or bring the shared code into your own branch.
# No need to switch branches in a checkout running an active experiment.
uv venv --python 3.12 .venv-stats
uv pip install --python .venv-stats/bin/python -e .
mkdir -p runs/shared_data
tar -xzf /path/to/arno_v1.tar.gz -C runs/shared_data
```

Current local archive: `runs/shared_data/arno_v1.tar.gz`; its archive hash and
size are in `runs/shared_data/arno_v1_receipt.json`. The extracted bundle has a
SHA256 manifest for every artifact. These files must be transferred separately
through an agreed data-sharing channel; they are not committed or uploaded by
the Git push.

The prepared archive is 81,718,672 bytes (about 78 MiB), SHA256
`56b09edc2431af143564a75571b81bcb8c0238fec2d636fab364226c071a2dc4`.

## What is reusable

| Layer | Where it is / what it contains | Use for logistic regression |
| --- | --- | --- |
| Dataset-only tables | `dataset/` in the starter bundle; canonical original directory is `runs/competition_pipeline/protocol_panel_v2/` | Recommended common source of examples and labels |
| Numeric/text vectors | `features/feature_matrix.npy` and `feature_contract.json` in the bundle | Ready-to-load starting representation, 34,718 × 467 |
| Training grouping | `training_metadata.jsonl` in the bundle | Row alignment, training-family folds, scope weighting and task-overlap exclusion; not predictor features |
| Reviewed research facts | `research/system_profiles/collection_v3/` plus local `runs/feature_pipeline/offline_evidence_preparation_v1/` tables | Optional separate research enrichment; not assumed competition-eligible |
| Full enriched states | `runs/laya_inputs/collected_v3/` | Useful illustration of collected information and unknowns; not the required input format for logistic regression |
| Laya token corpus/checkpoints | `runs/grouped_enriched/prepared_v1/tokens/`, trained checkpoint folders | Specific to the neural trainer; unnecessary for statistical models |

The starter bundle contains **no fitted model, no external research features,
no development/monitor outcomes and no calibration/confirmation data**. Its
feature matrix is freshly generated from the current feature encoder. Older
cached analysis matrices have their own source fingerprints and must not be
silently treated as an encoding from newer code. Generating this starter does
not rerun historical ridge fits or establish logistic-regression quality.

## Dataset size and scope

This is a reviewed subset of the public measurement database, not the full
release. The original downloaded release has 61 benchmarks; our review groups
them into 57 source families. Fourteen benchmarks / 13 families were admitted
to the frozen experiment allocation: seven training, two development, two
calibration and two confirmation families. The allocation and admission reasons
are in [the frozen split](../configs/head_finetuning/frozen_split_v1.json) and
[source review](../configs/head_finetuning/data_split_v1.json).

The dataset-only preparation contains:

| Role | Rows | Outcomes in this bundle? |
| --- | ---: | --- |
| Training | 34,718 | Yes, separate native binary targets |
| Training-family monitor | 1,000 | No; inputs only |
| Original development | 963 response records / 960 system-items | No; inputs only |

Training covers **170 visible system–benchmark scopes across seven families**.
A scope is a model/harness/configuration paired with a benchmark; it does not
mean a distinct model. Related tasks remain grouped across wrappers. The 34,718
training rows contain 15,383 normalized task groups, so rows are not all unique
questions. Training retains one selected observed trial per system/item;
original development evaluation preserves repeated recorded trials.

## Table contract and joins

All paths below are relative to `arno_v1/dataset/`:

| File | Key and contents |
| --- | --- |
| `systems.jsonl` | `system_key`, `recorded` with the eight supplied subject strings |
| `tasks.jsonl` | `task_key`, `recorded` with supplied task text, attributes and interactors |
| `examples.jsonl` | `row_key`, `role`, `system_key`, `task_key`, anonymous `benchmark_id`, `scope_key`, `task_group`, `source_line` |
| `training_targets.jsonl` | `row_key`, `native_binary_value`; exactly the training rows |
| `manifest.json` | Artifact hashes used by the shared verifier |
| `source_contract.json` | Source revision, source hashes and original preparation provenance |

Join examples to systems on `system_key`, to tasks on `task_key`, and training
labels on `row_key`. Preserve `examples.jsonl` training order when using the
provided matrix. Do not align labels by independently sorting IDs or dictionary
iteration. The feature contract and training metadata bind the row order.

The subject strings are `normalized_name`, `provider`, `release_date`,
`access_date`, `harness`, `harness_version`, `reasoning_effort` and
`subject_features_extra`. Task strings are `item_content`, `item_features` and
`interactors`; `benchmark_id` is the fourth published item field but remains
anonymous join/evidence metadata, outside the base learned features.

Missing values remain empty strings. Literal text such as `None`/`null` is not
silently removed. Missing model identities are not recovered from private
display names. Labels are native 0/1 events: some datasets describe events
other than correctness. Review [outcome semantics](outcome-semantics.md) before
changing the target; do not globally invert labels.

## Load the statistical starter

Run this from the repository root after obtaining the bundle:

```python
from pathlib import Path
import hashlib
import json
import numpy as np

from paiec.competition_analysis import verify_competition_bundle
from paiec.feature_preparation import read_jsonl, stable_hash

bundle = Path("runs/shared_data/arno_v1")
manifest = json.loads((bundle / "bundle_manifest.json").read_text())
for name, expected in manifest["artifacts"].items():
    with (bundle / name).open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == expected

data = bundle / "dataset"
verify_competition_bundle(data)
rows = [r for r in read_jsonl(data / "examples.jsonl") if r["role"] == "training"]
targets = {r["row_key"]: r["native_binary_value"]
           for r in read_jsonl(data / "training_targets.jsonl")}
metadata = list(read_jsonl(bundle / "training_metadata.jsonl"))
contract = json.loads((bundle / "features/feature_contract.json").read_text())
assert [r["row_key"] for r in rows] == [r["row_key"] for r in metadata]
assert stable_hash([r["row_key"] for r in rows]) == contract["row_order_sha256"]
assert set(targets) == {r["row_key"] for r in rows}

X = np.load(bundle / "features/feature_matrix.npy", mmap_mode="r", allow_pickle=False)
y = np.asarray([targets[r["row_key"]] for r in rows], dtype=int)
assert X.shape == (len(y), contract["dimensions"]) == (34718, 467)
assert np.isin(y, [0, 1]).all()
print(X.shape, y.shape)  # (34718, 467), (34718,)
```

`X` contains only predictor features. `y` is the loss target. Keep family,
scope, task-group and row identifiers in `metadata`, outside `X`.

## What the 467 features represent

The shared [encoder](../src/paiec/conditional_features.py) is deterministic and
does not fit a vocabulary or truncate the task:

| Block | Dimensions | Contents |
| --- | ---: | --- |
| Task text | 256 | Hashed word/bigram and Unicode character features from task, attributes and interactors |
| System text | 64 | Hashed features from the eight supplied subject fields |
| Numeric / missingness | 19 | Log input lengths, digit/non-ASCII proportions, parsed dates, nominal dense capacity from supplied names and known/missing flags |
| System–task interactions | 128 | Products of folded system and task vectors |

Nominal model size from a name is a proxy, not a verified parameter count.
Unknown values have indicators; MoE names are not converted into invented
active/total sizes. The existing estimator in
[conditional_training.py](../src/paiec/conditional_training.py) is **ridge
regression, not logistic regression**. Its saved predictions and fitted weights
are not included in this starter or recommended as features. Earlier ridge
controls failed their quality gates; a reusable representation is not proof
that it is the best one for logistic regression.

For a custom representation, load `recorded` dictionaries from the system/task
tables and call `feature_vector(subject, item)`, or implement another versioned
encoder. A fresh representation should have its own name/source hash rather
than claiming compatibility with `conditional_fields_v1`.

## Shared protocol, model-specific preprocessing

Keep the data revision, raw labels, family/task assignments, scored examples,
sampling seeds and weighting explicit and shared. Fit scaling, imputation,
learned text vocabularies, feature selection and hyperparameters only within
the fitting part of each training fold. Unknowns and documentation-supported
capabilities must remain distinguishable from recorded enabled settings.

Use training-only family folds while developing logistic regression. The
existing helper excludes the held family and any task groups overlapping it:

```python
from paiec.conditional_analysis import family_training_mask

families = np.asarray([r["validation_family"] for r in metadata])
task_groups = np.asarray([r["task_group"] for r in metadata])
fit_mask, validation_mask = family_training_mask(families, task_groups, "gsm8k_math")
# Fit preprocessing and the estimator on X[fit_mask], y[fit_mask] only.
```

This is not a random row split. Keep repeated/shared tasks together. Do not add
private IDs, anonymous benchmark IDs, outcome-derived family scores, reference
answers or grading rules as input features. Agree on sample weights too: the
current Laya pilot weights families then scopes equally; the historical ridge
control weights scopes equally. Those existing experiments are not identical
recipes merely because they use the same rows.

The original-development evaluation is coordinated separately; it is not a
new tuning set to open automatically. Calibration/confirmation remain sealed.
Keep budget-zero models and any adaptation using revealed labels as explicit
separate evaluation modes. The common competition score weights pairs equally
and preserves trial counts; ordinary pooled row accuracy is a different metric.

## Research enrichment and reproduction

The optional research layer has documented model/base architecture facts,
harness-supported capabilities, recovered actual run settings, provenance,
unknowns and scope qualifiers. It has no validated teacher-generated task
difficulty/ability labels. See [collected-data delivery](laya-input-contract.md)
and [the profile catalog](../research/system_profiles/README.md). Outside-source
facts remain research-only until eligibility is reviewed; they are not part of
the dataset-only starter or the 467 features above.

To inspect the canonical dataset workflow without executing anything:

```bash
.venv-pipeline/bin/python scripts/run_pipeline.py data \
  --config configs/pipelines/competition_data_protocol_v2.json
```

That recipe expects the separately generated
`runs/competition_sources/protocol_panel_v1/` source bundle. A fresh checkout
does not contain it. Rebuilding the entire preparation additionally needs the
pinned public source tables and earlier frozen/expanded preparations; see
[workflows](workflows.md), [protocol panel](protocol-panel-and-policy.md) and
[public-data setup](data-notes.md). Receiving the portable starter avoids those
dependencies. Some original provenance helpers retain the author's absolute
source paths: do not edit hashed manifests to make them appear portable. The
table verifier/loader above and the included grouping metadata work from the
copied bundle without those original paths.
