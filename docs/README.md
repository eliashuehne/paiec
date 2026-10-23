# Documentation

Start with [compact inputs and development coverage](compact-inputs-and-development.md)
for the current prepared experiment. Preparation is complete; compact-v3 has
not been trained. [AGENTS.md](../AGENTS.md) records active constraints, and
[CONTRIBUTING.md](../CONTRIBUTING.md) covers repository checks and versioning.

## Inputs and research data

| Guide | Scope |
| --- | --- |
| [Compact inputs and development](compact-inputs-and-development.md) | Current compact-v3 recipe, retained task text and missing identities |
| [Predictor input boundary](benchmark-questions.md) | Exact permitted fields and shared question |
| [Original input audit](input-context-audit.md) | Recorded settings and context-v1 |
| [Task feature pipeline](task-feature-pipeline.md) | Task-v2 predecessor, omissions and measured evidence control |
| [Profile catalog](../research/system_profiles/README.md) | Homogeneous research schema, provenance and collection versions |
| [Initial collection](system-profile-collection.md) | Collection-v1 findings and ambiguities |
| [Expanded coverage](system-profile-coverage.md) | Collection-v2 scope and information delivery audit |
| [Capability-learning design](capability-learning-design.md) | Proposed research; not an implemented architecture |

## Training and evaluation

| Guide | Scope |
| --- | --- |
| [Fine-tuning plan](decision-head-finetuning-plan.md) | Decision head, holdouts and gates |
| [Frozen data split](head-data-split.md) | Family roles, exports and integrity checks |
| [Training smoke](head-training-smoke.md) | Historical bounded machinery validation |
| [Development results](head-development-results.md) | Original smoke-head comparison |
| [Enriched pilot](enriched-head-pilot.md) | Earlier matched local preparation and interrupted MPS attempts |
| [Modal pilot](modal-head-pilot.md) | Completed cached enriched training and historical cloud attempts |
| [GPU performance](gpu-training-performance.md) | Measured 1,024-token cache/resource probes |
| [Approach review](head-approach-review.md) | Audit of the completed earlier head |
| [Experiment log](experiment-log.md) | Chronological work, measured findings and untried proposals |

## Running the project

| Guide | Scope |
| --- | --- |
| [Laya backends](laya-backends.md) | MLX/Torch interface and measured runtime checks |
| [Pretrained Laya](laya.md) | Environments, sampling and local evaluation |
| [Modal execution](modal.md) | Job planning, transport and cloud setup |
| [MLflow](mlflow.md) | Tracking, metrics and artifacts |
| [ZenML preparation](zenml.md) | Local feature pipeline, structured tables, coverage and MLflow integration |
| [Dependency locks](../requirements/README.md) | Platform-specific environments and historical lock |
| [Command catalog](../scripts/README.md) | Script purposes and side effects |
| [Configuration index](../configs/README.md) | Current and historical recipes |

## Public data and history

- [Data notes](data-notes.md): preparation, provenance, Parquet tables and EDA.
- [Outcome semantics](outcome-semantics.md): why recorded binary events differ.
- [Project handoff](project-status.md): historical pre-MLX state, explicitly dated.
- [Archived session notes](history/README.md): dated working notes, superseded by current guides.

Paths under `runs/` describe local reports, inputs, weights or tracking stores.
Those artifacts are intentionally absent from Git and from a fresh checkout.
Historical documents retain the results and limitations of their own versions;
use this index and the current recipe rather than treating every old command as
the next experiment to run.
