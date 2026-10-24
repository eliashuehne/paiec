# PAIEC: predicting system outcomes

Predict the probability that a target AI system succeeds on a task, using its
supplied system/task information and permitted revealed outcomes. This project
contains shared data preparation, feature builders, probability predictors,
evaluation and experiment tracking for the
[Predictive AI Evaluation Competition](https://aims.stanford.edu/competition).
The organizers' reference code is the pinned `baseline/` submodule.

## Current direction

**Laya work is on hold.** The completed enriched multilingual pilot scored
**0.282439 Brier ALC**, worse than constant 50% predictions (**0.25**), on the
original development set. Within-pair AUROC was **0.5457**. The checkpoint and
input checks passed, but the experiment did not deliver the intended predictive
improvement. No further Laya or GPU run is planned.

Focus now is reusing the data, features and evaluation protocol for other
approaches. Start with [current status](docs/current-status.md) and
[shared data and features](docs/shared-data-start.md). Prior negative results
and model-specific limitations remain documented; data preparation alone does
not establish predictive quality.

## Reuse the data

The portable starter has **34,718 training rows**, a **467-feature matrix**,
separate native binary labels and family/task grouping metadata. Logistic
regression or another statistical model can read it using the core package;
no Laya weights, Torch, MLX or GPU are required.

Local archive: `runs/shared_data/statistical_starter_v1.tar.gz`. Generated data
is Git-ignored and needs a separate transfer or reconstruction. Optional
model/harness research facts are separate and are not included in this
dataset-only bundle; their competition eligibility remains unreviewed.

Use the [workflow map](docs/workflows.md) for ZenML orchestration and
[data lifecycle](docs/data-lifecycle.md) for retained artifacts. Recipes,
checkpoints, source snapshots and saved scores stay intact for reproducibility.
The [completed Laya record](docs/grouped-enriched-training.md) contains exact
results and artifact paths; the default checkpoint is not silently replaced.

## Set up a fresh checkout

From the repository root, initialize the pinned reference code and create a
Python 3.12 environment. The core tests need no downloaded dataset or weights.

```bash
git submodule update --init --recursive
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e '.[test,tracking]'
make check PYTHON=.venv/bin/python
```

Keep model environments separate. Exact dependency snapshots and their platform
scope are documented in [requirements/README.md](requirements/README.md).

| Purpose | Setup and use |
| --- | --- |
| Statistical data and features | [Shared starter](docs/shared-data-start.md), core package only |
| Preparation workflows | [ZenML with MLflow](docs/zenml.md), separate `.venv-pipeline` environment |
| Public data and EDA | [Data provenance](docs/data-notes.md) |
| Experiment dashboard | [MLflow](docs/mlflow.md) |
| Retained Laya runtimes | [MLX/Torch reference](docs/laya-backends.md), separate native locks |
| Retained cloud execution | [Modal reference](docs/modal.md), separate client/worker locks |

## Inspect workflows

Use the shared data directly for model-specific feature engineering and
training-only validation. Inspect the dataset workflow without launching it:

```bash
.venv-pipeline/bin/python scripts/run_pipeline.py data \
  --config configs/pipelines/competition_data_protocol_v2.json
```

The plan needs only the recipe; execution additionally needs its generated
source dependencies. See [shared data](docs/shared-data-start.md) and
[workflows](docs/workflows.md). Training, development assessment and release are
separate actions. Historical Laya commands are retained for reproducibility;
they are not the next recommended workflow.

## Repository map

| Location | Contents |
| --- | --- |
| `src/paiec/` | Scoring, prediction, input builders, training helpers and execution adapters |
| [scripts/](scripts/README.md) | Stable command-line entrypoints, grouped by purpose in the catalog |
| [configs/](configs/README.md) | Suites, frozen splits, versioned recipes, policies and schemas |
| [research/system_profiles/](research/system_profiles/README.md) | Sourced, versioned model/harness catalogs |
| [requirements/](requirements/README.md) | Environment locks and worker dependency inputs |
| [tests/](tests/README.md) | Synthetic protocol, integrity and adapter tests |
| [docs/](docs/README.md) | Current guides, experiment findings and historical notes |
| `baseline/` | Unmodified organizers' submodule at a pinned revision |
| `payload/`, `runs/` | Ignored local data, outputs, reports and tracking stores |

See [CONTRIBUTING.md](CONTRIBUTING.md) for checks and versioning rules, and
[AGENTS.md](AGENTS.md) for experiment constraints and the working handoff.

## Evaluation boundary

Predictors receive the published eight subject and four item fields, anonymous
benchmark IDs, and only budget-permitted revealed examples. Missing model names
stay missing. Reference answers, grading definitions and evaluation outcomes
never enter the prediction state. Models predict the recorded target event;
the historical Laya prompt and probability interface are retained for reproducibility.

The evaluator keeps all trials of an item together in a 50/50 acquisition/evaluation
split. Budgets are nested at 0, 1, 3, 7, 15 and 31; subject–benchmark pairs are
weighted equally. Raw binary labels retain their recorded meaning, which is not
always correctness. Details: [input boundary](docs/benchmark-questions.md),
[outcome semantics](docs/outcome-semantics.md), and
[frozen family split](docs/head-data-split.md).
