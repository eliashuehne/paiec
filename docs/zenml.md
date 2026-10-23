# ZenML feature preparation and MLflow

The pipeline now includes `prepare_collection_review`, before publication.
It exports a training-only 14-task review set, seven agent-authored candidate
annotations with exact evidence spans, a task-assessment JSON Schema, and a
prioritized collection queue. The plan is
`configs/pipelines/feature_collection_v1.json`; annotation seeds and limitations
are in `research/task_features/README.md`. ZenML exposes a `collection_review`
HTML artifact with original task text and candidate claims. These estimates
are not independently reviewed and do not enter Laya training. No new external
source collection, teacher API calls or empirical capability fitting is performed.

The local ZenML 0.98.0 stack is `paiec_local`, with the `paiec_mlflow`
experiment tracker. It uses the existing MLflow 3.17.2 SQLite database at
`runs/mlflow/mlflow.db`, under **PAIEC Feature Preparation**. Historical
model runs remain in their existing experiments.

This integration prepares research feature sidecars. It does not launch
training, model inference, annotation APIs, downloads or Modal jobs, or change
existing renderers, checkpoints, frozen recipes or family roles. These new
tables are not a replacement for an existing checkpoint's input contract.

## Set up and open

Use a separate Python 3.12 environment. ZenML's dependencies differ from the
model environments; `requirements/pipeline.lock` records the tested macOS
Apple Silicon environment without native model frameworks.

```bash
uv venv --python 3.12 .venv-pipeline
uv pip install --python .venv-pipeline/bin/python -r requirements/pipeline.lock -e '.[pipeline,test]'
.venv-pipeline/bin/python scripts/run_feature_pipeline.py setup
.venv-pipeline/bin/python scripts/run_feature_pipeline.py ui
```

Setup is idempotent and checks the named tracker before reusing it. ZenML's
configuration, database and artifacts live under `runs/zenml/config`, with
repository configuration in ignored `.zen/`. Other projects' global settings
are not changed. Analytics and automatic browser opening are disabled.
The dashboard listens at <http://127.0.0.1:8237>.

Run the existing MLflow dashboard separately at <http://127.0.0.1:5001>:

```bash
.venv-pipeline/bin/python scripts/mlflow_ui.py
```

MLflow's server is not required for logging: the pipeline writes directly to
the existing SQLite store. Both dashboards are local, with no hosted account
or container requirement.

## Execute preparation

The default recipe, `configs/pipelines/feature_preparation_v1.json`, verifies
the existing expanded compact-v3 bundle: 30,000 training records, 1,000
task-disjoint monitor records, and the original 963 development records.
No old preparation is regenerated.

```bash
.venv-pipeline/bin/python scripts/run_feature_pipeline.py run
.venv-pipeline/bin/python scripts/run_feature_pipeline.py status
```

Each execution creates a timestamped folder in `runs/feature_pipeline/`.
Use `--out runs/feature_pipeline/NAME` for an explicit fresh directory.
Existing directories and the immutable source bundle cannot be overwritten.
Step caching is disabled so unchanged paths cannot bypass hash checks.

The stages are frozen-input verification, system-identity matching,
model-profile preparation, harness/run preparation, task preparation, and
validation/publication through the MLflow tracker. Model, harness and task
stages depend on the system table; publication depends on all stages.

Outputs include separate system/model/harness/run/task JSONL tables, example
links, source evidence, a hash manifest, an HTML report, and a feature plan.
ZenML exposes the system table, model/harness fact tables, the first 100 task
previews, and the HTML report. MLflow records counts, fingerprints, reports
and a source snapshot. Counts are preparation statistics, not model scores.

`expected` labels are ignored and are not exported to feature tables. Matching
uses only visible attributes and the original bundled profile policy. Missing
historical development fields become empty strings; private IDs/display names
do not restore missing identities. Cross-role normalized task overlap is
rejected. Calibration/confirmation are not accepted pipeline roles and their
outcome files are not opened. This exact-content check does not establish
translation/paraphrase independence.

## Feature readiness

| Bucket | Current output | Remaining work |
| --- | --- | --- |
| Model | Existing sourced facts and evidence/status | Review conflicts and missing specialization/language/limit facts |
| Demonstrated abilities | Pending | Training-only/out-of-fold capability estimates |
| Harness/run | Existing facts and recorded settings | Typed enabled tools, workflows and budgets |
| Task requirements | Full allowed inputs and deterministic text observations | Reviewed skill/language/modality/reasoning annotations; learned difficulty |
| Compatibility | Pending | Interactions from verified system/task features |
| Revealed evidence | Pending in this pipeline | Training-only support/query episodes |

Character/script/number counts are observations, not difficulty or proficiency
estimates. Unsupported annotations remain null; unknown settings are not treated
as disabled. A new versioned renderer/input contract, matched controls and
native token/visibility qualification are required before new feature-based
Laya training. There is no training-dispatch step in this pipeline.

## Checks and shutdown

```bash
make check PYTHON=.venv-laya-mlx/bin/python
.venv-pipeline/bin/python -m unittest discover -s tests -p 'test_*.py'
ZENML_CONFIG_PATH="$PWD/runs/zenml/config" ZENML_ANALYTICS_OPT_IN=false .venv-pipeline/bin/zenml logout --local
```

Stopping ZenML preserves its database/artifacts; reopen with `ui`. Stop a
foreground MLflow dashboard with Ctrl+C. Never commit `runs/`, `.zen/`, or the
pipeline environment.

References: [installation](https://docs.zenml.io/getting-started/installation),
[MLflow integration](https://docs.zenml.io/stacks/stack-components/experiment-trackers/mlflow).
