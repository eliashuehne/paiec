# Command catalog

Run scripts from the repository root in the intended environment. Entry points
stay here because existing commands, source snapshots and imports use these
paths. Shared model/scoring logic belongs in `src/paiec/`. Use `--help` to inspect
options; no command in this catalog is authorization to launch a run.

Current focus is shared data/features and alternative predictors; see
[shared data](../docs/shared-data-start.md) and [current status](../docs/current-status.md).
Laya commands remain reproducibility utilities. No training/inference job or
continuation is queued.

## Evaluation, jobs and tracking

| Script | Purpose / effects |
| --- | --- |
| `run_pipeline.py` | Canonical ZenML entry point for research/data/evidence/training/validation/release; plans by default, `--execute` launches |
| `export_system_enrichment.py` | Package the full versioned model/harness/run collection, supplemental research and reviewed JSONL/CSV views; no matrix, labels, inference or fitting |
| `manage_data.py` | Data-role audit and verified, reversible retirement/restoration of marked untrained previews |
| `check_laya_inputs.py` | Cached native tokenizer/target conversion and context-coverage audit, including multilingual; never loads a model |
| `qualify_multilingual.py` | Local grouped-input preparation with `--prepare`, or a bounded CUDA qualification; disposable training controls, no full pilot |
| `train_grouped_enriched.py` | Native stage utility for enriched multilingual grouped preparation or explicit CUDA training; normally dispatched through ZenML |
| `predict_grouped.py` | Explicit checkpoint-bound Torch grouped prediction from published subject/item JSONL; loads weights and runs inference |
| `evaluate_grouped.py` | Separate original-development preparation, outcome-free GPU inference and trial-preserving local scoring |
| `fetch_grouped_checkpoint.py` | Download missing files for a declared fitted grouped checkpoint and verify hashes; no GPU allocation |
| `complete_grouped_pilot.py` | Complete one already-launched, explicitly authorized pilot: recover checkpoint, evaluate once, stop GPU apps; never starts/restarts fitting |
| `run_eval.py` | Score selected predictors; Laya selection loads weights and runs inference |
| `run_baseline_eval.py` | The preserved benchmark-fold evaluator for the constant and empirical-mean reference predictors; its modules live in `paiec/baseline_harness/`, separate from the submission harness |
| `download_data.py` | Download the eligible Measurement DB tables into `payload/data`; needs the `data` extra and a Hugging Face login |
| `run_job.py` | Local/Modal job launcher; `--dry-run` plans only, dispatch may allocate a GPU |
| `compare_runs.py` | Compare saved completed predictions without new model inference |
| `mlflow_ui.py` | Serve the local tracking dashboard |
| `run_feature_pipeline.py` | Local ZenML/MLflow setup and feature sidecars; no training, APIs or GPU allocation |
| `run_competition_pipeline.py` | ZenML competition workflows: `--workflow data` (default) or training-only `--workflow evidence`; `--plan` has no execution side effects |
| `prepare_competition_panel.py` | Outcome-blind training-only augmentation to at least 80 normalized tasks per admitted scope |
| `select_evidence_policy.py` | Select smoothing outside whole held training families; original development unused |
| `build_competition_submission.py` | Deterministic allowlisted local ZIP; never uploads |
| `qualify_evidence_submission.py` | Isolated exact-package checks on every training scope with global revealed evidence |
| `analyze_competition_data.py` | Training-only fixed numerical evidence controls; no fitted model or native inference |
| `analyze_conditional_predictor.py` | CPU ridge fits outside whole training families; no native Laya inference, APIs or GPU |
| `run_training_data_expansion.py` | CPU ridge A/B of row count versus unused-benchmark diversity; admits no data, fits no candidate, opens no sealed role |
| `run_catboost_base.py` | CPU CatBoost whole-family base experiment with ablations or a declared feature-group recipe, including the research-only profile recipe and its scrambled control; no native Laya inference, APIs or GPU |
| `analyze_catboost_uncertainty.py` | Paired scope bootstrap and family jackknife over a finished run's saved predictions; refits nothing |
| `run_catboost_unseen_benchmark.py` | Scores panel fold models on unused public benchmarks with name-matched subjects; fits nothing on evaluation rows |
| `run_catboost_permutations.py` | Fits the declared scrambled-profile permutations and reports the null distribution, path-only effect and residual gain; resumable, `--aggregate-only` re-reads hash-verified replicates |
| `run_additive_logit.py` | CPU additive two-block logistic control on the same folds, episodes and unseen rows; no native Laya inference |
| `build_competition_pairs.py` | Build subject–benchmark pair cells over the whole public corpus at the platform's 80-item threshold; records direction, keeps native labels, no fitting |
| `run_competition_controls.py` | Score constant, revealed-label, smoothed and same-benchmark-evidence controls on pair draws; fits nothing |
| `run_rate_model.py` | Benchmark-fold two-way ability/difficulty fit with task-text difficulty regression and an optional CatBoost item term; writes a fail-closed gate verdict |
| `build_rate_model_submission.py` | Self-contained ZIP inputs for a gate-clearing rate model; refuses a rejected gate, never uploads |
| `qualify_conditional_control.py` | Isolated standard-library loading/parity checks and initial candidate release audit |
| `run_task_annotations.py` | Offline request export and checked result import; no network calls or training |
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
| `check_head_checkpoint.py` | Historical smoke-checkpoint inference on saved training-only fixtures |
| `check_encoder_checkpoint.py` | Native Torch/MLX inference and exact rendering on training-only fixtures from an explicit encoder-experiment checkpoint |
| `benchmark_head_batches.py` | Bounded CUDA performance probe; GPU work, no optimizer updates |
| `prepare_expanded_training.py` | Training-role-only 30k unique system/item expansion, collected facts and original development |
| `run_encoder_experiment.py` | Separate local token preparation, CUDA qualification/learning controls and matched frozen/upper-layer training |

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
| `audit_head_strategy.py` | Audit compact head ranking/confidence and fixed revealed-evidence controls; no model inference; label-using oracles are diagnostics only |
| `evaluate_saved_head_evidence.py` | Score a revealed-evidence control from saved probabilities; no Laya inference |
| `fit_evidence_adaptation.py` | Fit a separate correction from training-only monitor episodes; compare saved 30k heads on original development, no GPU |
| `report_encoder_experiment.py` | Analyze saved matched-head/evidence results and measured resources; no fitting or inference |

## Historical prompt experiment

`run_input_experiments.py` runs a controlled prompt/format experiment and performs
model inference. `report_input_experiments.py` renders its saved results.
Their configurations/results describe the earlier experiment, not the current
compact pilot. See [the experiment log](../docs/experiment-log.md).

`run_job.py --task qualify-head-pilot` runs a separate bounded CUDA check for
one verified pilot variant. It uses the selected `--pilot-config`, a physical
batch of two, the first frozen optimizer window and the longest training inputs.
It saves `qualification.json` under the fresh output directory, verifies that
model weights did not change, and applies zero optimizer updates. Training is
a separate `train-head-pilot` job after that result is reviewed.
