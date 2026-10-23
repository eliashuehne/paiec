# PAIEC / Laya

Predict how likely a target AI system is to complete a task successfully, using
its model and harness details, the task, and a small budget of revealed outcomes.
This repository contains a local evaluator, interchangeable Laya runtimes, and
head fine-tuning experiments for the
[Predictive AI Evaluation Competition](https://aims.stanford.edu/competition).
The organizers' reference implementation is the pinned `baseline/` submodule.

## Current work

The latest prepared experiment compares recorded inputs (`compact_v3`) with
the same tasks plus sourced model/harness facts (`compact_enriched_v3`), at
2,048 tokens. **It has not been trained.** See
[compact inputs and development coverage](docs/compact-inputs-and-development.md)
for the recipe, actual information delivery, and remaining missing identities.

An earlier enriched head completed training and scored **0.249080 Brier ALC**
on the original development set; a separate update using revealed labels scored
**0.238145** on its saved predictions. Neither is a result for compact-v3.
Always predicting 50% gives **0.25**; lower is better. See
[training results](docs/modal-head-pilot.md) and
[the evidence control](docs/task-feature-pipeline.md).

MLX is the default inference runtime on Apple Silicon. PyTorch supports local
MPS and Modal CUDA execution. Head training uses PyTorch. Model weights, public
payloads, MLflow stores, and prepared inputs are local artifacts outside Git.
External research profiles are not yet approved for competition use; hosted
Codabench compatibility is also unverified.

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
| Apple MLX inference | [Runtime setup](docs/laya-backends.md), `requirements/mlx.lock` |
| PyTorch inference and head training | [Local workflow](docs/laya.md), `requirements/torch.lock` |
| Modal client and CUDA workers | [Cloud execution](docs/modal.md), separate client/worker locks |
| Public data and EDA | [Data provenance and preparation](docs/data-notes.md) |
| Experiment dashboard | [MLflow](docs/mlflow.md) |

## Run and prepare experiments

Evaluation requires a prepared public payload in `payload/`. After setting up
the MLX environment, a small pretrained check is:

```bash
.venv-laya-mlx/bin/python scripts/run_eval.py --suite smoke --predictor constant empirical_mean laya
```

Local and Modal placement share `scripts/run_job.py`. Inspect a plan before
dispatching it; this command does not allocate a GPU:

```bash
.venv-modal/bin/python scripts/run_job.py --profile modal-l4 --task evaluate --suite smoke --out runs/example_plan --dry-run
```

Head recipes live in `configs/head_finetuning/`. Preparation, preflight and
training are separate steps; training requires `--train`. Existing prepared
directories and checkpoints are immutable. Use the
[current compact workflow](docs/compact-inputs-and-development.md) rather than
restarting a historical interrupted run or overwriting its output.

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
never enter the prediction state. The shared question asks about the target
system's success; Laya already returns its yes-probability.

The evaluator keeps all trials of an item together in a 50/50 acquisition/evaluation
split. Budgets are nested at 0, 1, 3, 7, 15 and 31; subject–benchmark pairs are
weighted equally. Raw binary labels retain their recorded meaning, which is not
always correctness. Details: [input boundary](docs/benchmark-questions.md),
[outcome semantics](docs/outcome-semantics.md), and
[frozen family split](docs/head-data-split.md).
