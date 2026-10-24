# ZenML feature preparation and MLflow

Commit checkpoint, 2026-10-23: offline collection machinery is ready for review;
the full feature dataset is incomplete. All scoped metadata fields have source
reviews, but only 14 of 16,909 tasks have candidate annotations and none has an
independent quality review. Training-only folds and evidence links cover all
30,000 training records without exported outcomes or fitting. Compatibility,
learned abilities/difficulty and a new Laya input contract remain unfinished.
The user chose offline preparation. Resume from the verified run documented in
[evidence links](feature-evidence-links.md); do not submit the prepared pilot
or launch fitting as part of committing this checkpoint.

Current preparation recipe: `configs/pipelines/feature_preparation_offline_v10.json`
(the CLI default), with `feature_sources_v6.json` and an explicit identity-review
ledger. It audits **all selected roles**:
176 visible systems, 39 model profiles and eight harness profiles. Earlier
training-only collection queues did not cover every system in this preparation.
All 617 requested static fields, 1,584 run-setting fields and ten unresolved
identity requests now have scoped reviews. Unknowns remain unknown; task
annotation and the other feature builders remain incomplete. See the
[source-review status](feature-source-review.md) for the measured gaps and the
user-approved acceptance rule: complete review, retaining evidenced unknowns.

Primary-source supplementation is now applied in `prepare_model_profiles`.
The assertion/source ledgers live in `research/feature_collection/`; archived
text is under `runs/feature_sources/primary_v1` and `provider_v2`. The registry
`configs/pipelines/feature_sources_v1.json` selects the first two source bundles. Fresh
checkouts must restore both pinned archives using the commands in the collection
README. Source bytes
and quotations are verified before additive model facts enter the new tables.
Existing documented conflicts are retained for review, not overwritten.
Collection also exports `typed_run_features.jsonl`, matching recorded-run
facts only by identical visible subject descriptions. Completeness remains
unproven: successful pipeline execution is not evidence of 100% field coverage.

`prepare_collection_review` retains the earlier queue and seven v1 starter
candidates. The subsequent `prepare_task_annotation_requests` stage prepares
outcome-blind requests for all 16,909 distinct full task inputs, 14 training-only
manual v2 candidates with exact quotes, and an unsent 14-question Batch pilot.
All supplied text is preserved. There are 16,895 pending annotations and zero
independent reviews. These sidecars do not enter Laya training. The offline
importer checks request binding, provider errors, response completeness and
quote grounding. See [task annotations](task-annotation-collection.md) for
commands and limits. The user chose offline collection; no teacher calls or
empirical capability fitting are performed.

`prepare_training_evidence_links` also prepares outcome-blind collection links
for all 30,000 training rows: five task-group folds, 850 same-scope support
pools, and 170 training-only 50/50 evidence splits with nested budgets.
No outcomes are exported or fitted. See [evidence links](feature-evidence-links.md).

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

The current recipe, `configs/pipelines/feature_preparation_offline_v10.json`, verifies
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

The nine stages are frozen-input verification, system-identity matching,
model-profile preparation, harness/run preparation, task preparation,
collection review, task annotation preparation, training evidence links, and validation/publication
through the MLflow tracker. Publication depends on all stages.

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
| Model | Scoped metadata review complete; confirmed values and evidenced unknowns | Applicability and competition eligibility remain separate |
| Demonstrated abilities | Five-fold collection links only; no estimates | Training-only/out-of-fold capability estimates |
| Harness/run | Scoped recorded-setting review complete; unknowns retained | Keep supported capabilities separate from actual settings |
| Task requirements | Full request plan and 14 unreviewed v2 candidates | Complete and quality-review annotations; learn difficulty separately |
| Compatibility | Pending | Interactions from verified system/task features |
| Revealed evidence | Training-only support/query links, without outcomes | Materialize eligible evidence and validate its later use |

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
