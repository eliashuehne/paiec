# Command catalog

Run scripts from the repository root in the intended environment. Entry points
stay here because existing commands, source snapshots and imports use these
paths. Shared model/scoring logic belongs in `src/paiec/`. Use `--help` to inspect
options; no command in this catalog is authorization to launch a run.

## Evaluation, jobs and tracking

| Script | Purpose / effects |
| --- | --- |
| `run_eval.py` | Score selected predictors; Laya selection loads weights and runs inference |
| `run_job.py` | Local/Modal job launcher; `--dry-run` plans only, dispatch may allocate a GPU |
| `compare_runs.py` | Compare saved completed predictions without new model inference |
| `mlflow_ui.py` | Serve the local tracking dashboard |
| `fetch_modal_job.py` | Recover existing cloud results and import tracking; no GPU allocation |

## Data and input inspection

| Script | Purpose / effects |
| --- | --- |
| `eda.py` | Local public-table statistics and reports; does not load trace content |
| `audit_outcomes.py` | Selective semantic audit; can inspect archived native records |
| `audit_input_context.py` | Audit original recorded settings and preservation |
| `preview_context_inputs.py` | Export context-v1 previews and native token audit |
| `measure_tokens.py` | Tokenize real sampled inputs; no model weights |
| `disk_report.py` | Report disk use; `--cleanup` explicitly removes duplicate download cache |

## Preparation and head training

| Script | Purpose / effects |
| --- | --- |
| `prepare_head_data.py` | Review/freeze family roles and export data; requires careful split boundaries |
| `prepare_compact_pilot.py` | Current matched compact-v3 inputs; tokenizes, does not fit or score |
| `review_development_profiles.py` | Outcome-blind identity review and separate question-only supplement |
| `prepare_task_pilot.py` | Prepared task-v2 predecessor |
| `prepare_enriched_head_pilot.py` | Earlier enriched-v1 paired preparation |
| `run_head_pilot.py` | Verify/preflight a prepared recipe; `--train` starts fitting and development inference |
| `report_head_pilot.py` | Render saved pilot progress/results; no fitting |
| `train_head_smoke.py` | Historical bounded training machinery check; applies optimizer updates |
| `check_head_checkpoint.py` | Inference on saved training-only fixtures for an explicit checkpoint |
| `benchmark_head_batches.py` | Bounded CUDA performance probe; GPU work, no optimizer updates |

## Research catalogs and saved-result audits

| Script | Purpose / effects |
| --- | --- |
| `validate_system_profiles.py` | Offline schema/provenance validation |
| `seed_system_profiles.py` | Reproduce original subject-only seed records in a fresh directory |
| `fetch_system_profile_sources.py` | Download the catalog's pinned source snapshots |
| `build_system_profile_collection.py` | Rebuild catalogs from reviewed assertions and cached evidence |
| `audit_system_profile_coverage.py` | Identity/fact coverage and optional token visibility |
| `audit_capability_training.py` | Training-role breadth and crossed system/task audit |
| `audit_head_approach.py` | Audit earlier head predictions and allowed features |
| `evaluate_saved_head_evidence.py` | Score a revealed-evidence control from saved probabilities; no Laya inference |

## Historical prompt experiment

`run_input_experiments.py` runs a controlled prompt/format experiment and performs
model inference. `report_input_experiments.py` renders its saved results.
Their configurations/results describe the earlier experiment, not the current
compact pilot. See [the experiment log](../docs/experiment-log.md).
