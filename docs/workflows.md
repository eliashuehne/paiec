# Workflows and data

Current modeling focus is model-independent data/features; Laya is on hold
after a completed negative result. See [current status](current-status.md) and
[shared data](shared-data-start.md). Historical training recipes are retained
for reproducibility and are not queued work.

Use `scripts/run_pipeline.py` as the canonical command. ZenML coordinates
repeatable workflows; scientific functions stay in `src/paiec/`, MLflow records
metrics and provenance, and existing local/Modal adapters execute native jobs
in their separate model environments.

| Command | ZenML graph | Responsibility |
| --- | --- | --- |
| `research` | `laya_feature_preparation` | Reviewed research facts, task requests and evidence links |
| `data` | `competition_dataset_preparation` | Supplied fields, separate targets and evidence episodes |
| `inputs` | `laya_collected_input_preparation` | Join collected facts to full tasks and audit native token/target conversion; no model inference |
| `evidence` | `competition_evidence_selection` | Fixed controls and whole-training-family policy selection |
| `training` | `laya_native_execution` | Verify an explicit native job, dispatch and retain provenance |
| `validation` | `competition_validation_release` | Review a pinned assessment, or explicitly assess a frozen policy |
| `release` | `competition_validation_release` | Reject failing candidates; package and qualify passing ones locally |

All graphs are implemented. New training and validation/release orchestration
has been compiled without execution and tested with synthetic inputs/mocked
native dispatch. This migration did not run a real-data experiment or GPU job.
Hardware/runtime and model quality remain properties of the selected run, not
guarantees supplied by ZenML.

## One command, explicit execution

Inspect a workflow first. These commands read recipes, not outcome datasets,
and do not initialize ZenML, create output directories or launch jobs:

```bash
.venv-pipeline/bin/python scripts/run_pipeline.py data
.venv-pipeline/bin/python scripts/run_pipeline.py evidence
.venv-pipeline/bin/python scripts/run_pipeline.py training
.venv-pipeline/bin/python scripts/run_pipeline.py validation
```

Execution requires `--execute`, an appropriate versioned `--config`, and a fresh
`--out runs/workflows/NAME/RUN`. A new development assessment additionally needs
`--allow-development-evaluation`; reviewing pinned saved results does not.
No workflow uploads a submission or automatically continues into another
workflow. Step caching is disabled so path reuse cannot skip integrity checks.

Recipes live in `configs/pipelines/`. Native jobs additionally select an
immutable scientific recipe in `configs/head_finetuning/` and a placement
profile in `configs/execution/profiles.json`. The default training-workflow
recipe illustrates a **new enriched compact-5k head run**, not a restart, a
newly recommended experiment or an authorized launch. It contains research
enrichment; do not mistake it for a competition-eligible training recipe.
Encoder qualification/training and ordinary evaluation use the same native
graph with their existing validated `JobSpec` tasks. Encoder training still
requires its exact passed qualification. Existing trainers retain preflight,
checkpoint/export checks and their own detailed MLflow runs.

The validation/release examples reference the already assessed strength-0.5
policy. It **failed** development validation. The release graph will record
rejection and build no ZIP. There is no automatic fallback/refitting: a changed
candidate needs its own declared policy and assessment. A passing candidate
must pass the pinned starting-kit checker and isolated exact-package runtime
qualification before a local-release report is emitted. Hosted compatibility
and hidden-test performance remain unverified.

## Data ownership and versions

The subsequent [Laya input audit](laya-input-contract.md) connects the latest
collected tables to versioned Laya-native rows. It also measures multilingual
capacity up to 8,192 tokens. Qualification and the grouped pilot subsequently
completed. The export remains research-only; the negative development result
does not establish competition eligibility or a reason to continue training.

The catalog is `configs/data/data_versions_v1.json`; details are in
[data lifecycle](data-lifecycle.md). It separates original/frozen sources,
competition-only preparations, research-enriched Laya inputs, question-only
supplements and retired untrained previews. A newer version does not silently
replace a checkpoint's bundled input contract.

```text
payload/                     Original public source tables; preserve
research/                    Reviewed facts, sources and annotation candidates
configs/data/                Data roles and lifecycle catalog
configs/pipelines/           Workflow recipes
configs/head_finetuning/     Immutable scientific recipes and frozen split
runs/workflows/              New outputs, grouped by workflow/run
runs/head_finetuning/        Existing pinned inputs/checkpoints; keep paths
runs/competition_*/          Existing dataset-only sources and preparations
runs/data_retirement/        Verified, restorable retired-preview archives
runs/mlflow/, runs/zenml/    Local tracking/artifact stores; preserve
```

Research facts are not used by competition-only preparations. Training outcomes
stay separate from inputs. Calibration/confirmation remain sealed; the extra
development supplement remains question-only. Native labels retain their
recorded semantics. Workflow success is not a model-quality or data-eligibility
claim.

## Utilities and compatibility

The enriched multilingual route uses the same `inputs` and `training` workflow
families, with `grouped_enriched_preparation_v1.json` and
`grouped_enriched_training_v1.json`. Offline grouped preparation is complete;
the single enriched training arm and original-development evaluation completed
(Brier ALC 0.282439 versus constant 0.25). No further Laya run is planned.
See [grouped enriched training](grouped-enriched-training.md) for the shared
encoder/aggregation, bundled facts and explicit predictor loader.

Older `run_feature_pipeline.py` and `run_competition_pipeline.py` interfaces
delegate execution to the same routing code. Individual-stage commands remain
debug/replay tools. Libraries and core tests work without ZenML installed.
Reports, source downloads, audits, dashboards, retirement/restoration and tests
remain ordinary utilities. Historical recipe/renderer paths and checkpoint
contracts were not moved or renamed to achieve this cleanup.

## Full-population feature preparation

The `features` route inventories all downloaded benchmark, subject, item and
response rows, preserves frozen roles, and builds outcome-independent input
features. It uses no scorer admission threshold or legacy sample cap. Its graph
is `verify_feature_sources → prepare_feature_tables → publish_feature_report`.
The core builder and validator work without tracking packages.

```bash
PYTHONPATH=src .venv-pipeline/bin/python scripts/run_pipeline.py features \
  --source-root /absolute/path/to/source-checkout \
  --out runs/feature_foundation/v1
```

Add `--execute` to build locally through ZenML and MLflow. Outputs must be fresh;
source paths and hashes stay bound to the recipe. Protected roles retain
metadata-only rows, and non-text assets receive partial-support status.
See [feature foundation](feature-foundation.md) for the table contracts, review
method, independent validator and measured gaps.
