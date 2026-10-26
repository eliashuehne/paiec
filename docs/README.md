# Documentation

Start with [collected model and harness information](shared-data-start.md)
for the main shared handoff: facts, recorded run settings, sources and system
mappings. Each approach can choose its own representation. Feature matrices
and Laya inputs remain separate experiment artifacts.

[Current status](current-status.md) records the completed negative Laya result
and the decision to focus on other approaches. [Workflow organization](workflows.md)
maps the reusable preparation/evaluation graphs; [data lifecycle](data-lifecycle.md)
records retained versions. [AGENTS.md](../AGENTS.md) preserves execution and
holdout constraints. Historical Laya guides are reference material, not queued work.

## Inputs and research data

| Guide | Scope |
| --- | --- |
| [Feature foundation PR target](feature-foundation-pr-target.md) | Proposed deliverables, table contracts, source scope, review cases and merge criteria for the next feature layer |
| [Feature foundation work plan](superpowers/plans/2026-10-25-feature-foundation.md) | Testable implementation sequence for the PR target; no feature implementation yet |
| [Laya input contract](laya-input-contract.md) | Upstream format audit, delivery of collected facts and measured multilingual 8k capacity |
| [Multilingual qualification](multilingual-qualification.md) | Frozen 8k recipe, grouped overflow policy and bounded learning/hardware gates |
| [Compact inputs and development](compact-inputs-and-development.md) | Historical compact-v3 recipe, retained task text and missing identities |
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
| [Grouped enriched training](grouped-enriched-training.md) | Completed pilot: Brier ALC 0.282439, weak ranking; Laya work on hold |
| [Compact enriched training](compact-enriched-training.md) | Completed enriched-only Modal run and 2,048-token qualification |
| [Strategy toward ALC 0.15](brier-alc-strategy.md) | Saved-prediction diagnosis, fixed evidence control and proposed encoder/data experiments |
| [Enriched 30k and encoder training](enriched-30k-encoder.md) | Expanded training data, learning controls, upper-layer comparison and L4 profiling |
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
| [Competition pipeline](competition-pipeline.md) | Dataset-only route, training evidence diagnostics and remaining runtime requirements |
| [Protocol panel and policy](protocol-panel-and-policy.md) | Full 170-scope coverage, nested evidence selection and locally checked candidate ZIP |
| [Conditional control](conditional-control.md) | Whole-family negative result, portable predictor and candidate rejection checks |
| [CatBoost base](catboost-base.md) | Structured supplied-field features, whole-family and unseen-benchmark results, interval-checked ablations, the additive two-block control, the research-only profile block, and its scrambled single-draw control with the permutation distribution that decides it |
| [Pair matrix and rate model](competition-pairs.md) | Whole-corpus subject–benchmark pairs, a competition-shaped Brier-ALC scorer that keeps cross-pair evidence, the fixed controls, the two-way ability/difficulty model and its gate |
| [Item axis](competition-item-axis.md) | Whether item difficulty is shared or model specific, the measured ordering prize, the coverage limit of revealed labels, and why a label-driven Rasch arm is not worth building |
| [Training-data expansion](training-data-expansion.md) | Measured row-count versus unused-benchmark diversity A/B; no admission decision |
| [Task annotations](task-annotation-collection.md) | Full-input plans, offline imports, candidate coverage and quality limits |
| [Feature evidence links](feature-evidence-links.md) | Training-only collection folds and support/query indices; no fitting |
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
