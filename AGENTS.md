# Working on PAIEC / Laya

## Start here

Repository layout, checks and versioning: `CONTRIBUTING.md`; documentation index:
`docs/README.md`. Dependency locks now live in `requirements/`; the old dated
`memory/` notes are archived in `docs/history/`. Preserve hash-pinned renderer
paths and immutable recipes when reorganizing unrelated project files.

Latest: `docs/compact-inputs-and-development.md`. `compact_v3` /
`compact_enriched_v3` are prepared at 2,048 tokens using
`configs/head_finetuning/compact_pilot_v3.json`; **no training launched**.
Use `paiec.predictor_factory.make_laya_predictor` for these formats. Old pinned
renderer sources remain unchanged. Prepared inputs:
`runs/head_finetuning/compact_pilot_v3_release/`; earlier compact working previews
are explicitly superseded. All available facts are delivered: training model/
base 4,584/5,000 and harness 3,958/5,000; original development model/base 603/963,
harness 963/963. This is not 100% model identity coverage: three original C-Eval
systems (360 records) still have no allowed model identifier. There are 1,145
long training task/context omissions and 652 oversized question/user sections;
both arms retain identical task text and all detected options. `collection_v3`
adds ERBench harness-scoped reported/base-family interpretations from primary
paper/code evidence. Actual private fine-tuned targets stay unbound. Four extra
development systems have separate question-only inputs (480 evaluation / 160
acquisition) in `runs/head_finetuning/development_identity_review_v3_release_final/`;
no outcomes were loaded/scored. Preserve original evaluation/frozen roles.
237 tests pass in each model environment and a source-only core export after
repository cleanup. Earlier full native Torch token/render preflight is unchanged.
No new quality result, sealed-set access or GPU performance qualification follows.

Read `docs/laya-backends.md` for the current interchangeable runtimes and
validation, and `docs/project-status.md` for the earlier handoff, important
findings, saved results, and interrupted-run history. Then read the documentation relevant to the task:

- `docs/benchmark-questions.md`: exact predictor fields and current Laya input.
- `docs/input-context-audit.md`: original field coverage, preserved attributes,
  missing-identity limitations and extra-settings decoding. `context_v1` is now
  available explicitly; JSON stays the default for the existing checkpoint.
- `docs/experiment-log.md`: latest reviewed runs, limitations, and proposed
  experiments before fine-tuning; keep attempted work and untried ideas distinct.
- `docs/decision-head-finetuning-plan.md`: planned head-only training, data roles,
  checkpoint integration, and decision gates. A 10-update training smoke check
  has run; see `docs/head-training-smoke.md` for its scope and validation.
- `docs/head-development-results.md`: first frozen-development Brier ALC and
  raw/calibrated controls; confirmation is still untouched.
- `docs/head-approach-review.md`: engineering audit of the completed enriched
  pilot. Predictions cluster around 50%; 654 training rows belong to groups
  where distinct full tasks share a retained input. Input preservation, small
  regularized controls and revealed-label adaptation are proposed next, not
  implemented or trained. Audit artifacts: `runs/head_finetuning/approach_audit_v1/`.
- `docs/task-feature-pipeline.md`: subsequent implemented `task_v2` /
  `task_enriched_v2` inputs, explicit omissions, Unicode/structured control and
  optional numeric evidence. Prepared recipe is one epoch/79 updates, not run.
  Old renderer contracts remain unchanged. Saved-head-only evidence control
  improves development ALC from 0.249080 to 0.238145 at untuned k=5; this is not
  a new-input or retrained-head result. Numeric evidence evaluations pool labels
  across all sampled pairs, matching only visible profiles and anonymous IDs.
- `docs/capability-learning-design.md`: subsequent proposed research direction:
  attribute-derived system abilities, text-conditioned task difficulty and
  compatible-example retrieval, followed by episodic/encoder training. Not an
  implemented architecture or launchable recipe. Training-only breadth audit
  finds 170 candidate pairs/35 canonical-name strings, versus 24 sampled pairs;
  only four name strings recur across training benchmarks. Candidates still
  need subject/input review. Frozen roles remain unchanged; web research stays
  separate from competition eligibility.
- `docs/head-data-split.md`: frozen pilot families and exports, preparation
  checks and truncation limits; use `configs/head_finetuning/frozen_split_v1.json`.
- `docs/laya.md`: environment, sampling, context limits, and historical results.
- `docs/mlflow.md`: tracking, artifacts, and interrupted-run limitations.
- `docs/outcome-semantics.md`: native binary labels are not always correctness.
- `docs/data-notes.md`: dataset provenance, tables, and EDA.

- `docs/modal.md`: execution profiles, authenticated client, cloud data/tracking
  layout and remaining GPU qualification. The cached enriched run is complete;
  see `docs/modal-head-pilot.md`. No cloud job is currently authorized by cleanup.

- `research/system_profiles/README.md`: homogeneous model/harness/run research
  profiles, architecture fields, provenance and source eligibility. Default
  prediction stays unchanged; `enriched_v1` is an explicitly authorized
  research-only experiment. `collection_v1` contains a primary-source
  first pass for the 32 exported training/development systems. See
  `docs/system-profile-collection.md` for identity ambiguity and coverage;
  external sources remain `unreviewed` for competition eligibility. Never
  promote candidate matches, documentation defaults, or configured position
  limits into verified run settings.
- `research/system_profiles/collection_v2/coverage.md`: expanded collection for
  the user-approved 170 training candidates plus eight development systems.
  There are 44 model/artifact/family profiles, eight harness profiles and 178
  recorded runs. Sourced model facts are available for 169/178, harness facts
  for 171/178, and both for 162/178. All 178 have some sourced enrichment;
  this is **not** 100% model-and-harness coverage. Nine unresolved models and
  seven missing harness names are in `collection_v2/unresolved.jsonl`.
  New opt-in policy: `configs/head_finetuning/enrichment_catalog_v2.json`.
  Prior catalogs/policies and renderer sources remain unchanged. Tokenizer-only
  task-v2 audit: model facts reach 3,185/5,000 training inputs, despite being
  available for 4,584. No new preparation bundle, fitting, inference or GPU job
  was launched. See `docs/system-profile-coverage.md` and
  `runs/head_finetuning/system_profile_coverage_v2_final/report.html`.
- `docs/enriched-head-pilot.md`: prepared local control/enriched pilot, actual
  coverage and launch command. The user subsequently gave the go-ahead for
  PyTorch/MPS: both variants were launched via `run_head_pilot.py --train`.
  The first attempt was stopped at 11 updates for growing MPS allocation and is
  KILLED, not completed. Memory-bounded recovery was stopped at the user's
  request after 11 completed updates and zero epochs in
  `runs/head_finetuning/enrichment_pilot_v1_training_mpsbounded_v2/`;
  MLflow `1187918c63604de29a3d0e0bcf0364a6` is KILLED. Do not restart it.
  The user requested Modal preparation with enriched first; see
  `docs/modal-head-pilot.md`. `train-head-pilot` defaults to enriched alone;
  baseline is a separate follow-up. The user subsequently authorized launching
  it: Modal job `job_bf242b6e53874dcebb6cd6dd44738625`
  stopped after client disconnection at 139 updates / one completed epoch.
  Its recovered history is in `runs/head_finetuning/enrichment_pilot_v1_modal_interrupted_recovery`.
  User authorized a fresh cached retrain: detached Modal job
  `job_e5548e95fcec4aba88d110056a0e4f71`, call `fc-01M5K8SQNQ19CSW1Z8GFTGAB2T`.
  Inspect `runs/modal_cached_head_launch.json` and `.log` before any launch;
  do not duplicate or restart the interrupted jobs. The cached run is now complete (see results below), bounded
  to 14 GiB, FP32, same batch size/recipe.
  Earlier uncached observations follow:
  CUDA preflight passed; initial updates take about 8 seconds, not a quality
  result. Remote MLflow run `3a15460e13bc4bc6933b93693712126a`. Check
  `runs/modal_enriched_pilot_launch.json` and `.log` before further action;
  never launch a duplicate or restart either local attempt.
  Both variants use 5,000 records,
  four epochs and 1,024 tokens, separate from the historical JSON/512 smoke.
  Later prediction must use the bundled input contract, not the old evaluator.

- `docs/gpu-training-performance.md`: measured batch/TF32/frozen-encoder-cache
  probes. Cache prototype is about 6x faster on one warmed window and estimates
  15 GB full-corpus features. Production cache is integrated in `encoder_cache.py`,
  with weight/mode guards, LRU byte cap and equivalence checks before training.
  The cached retrain completed: 316 updates, best epoch 1 Brier ALC 0.24908013,
  later epochs worse. Local output is
  `runs/head_finetuning/enrichment_pilot_v1_modal_cached/output/`; best weights
  are downloaded in `enriched/best`. See `docs/modal-head-pilot.md` for results.
  Cache: 12.14 GB, zero evictions, warm updates about 1.4 seconds. No GPU active
  at final check; do not start another run without new authorization.
  Earlier profiling pauses did not
  change weights. Do not interpret pause-inflated step timings as computation.

## Current user decisions

- The user explicitly authorized the MLX migration after the earlier handoff.
  It is implemented behind a shared interface. MLX is now the default;
  `--laya-backend torch` selects the preserved reference runtime. Do not
  interpret older deferred-migration notes as overriding this authorization.
- The 10,000-input Laya run was stopped at the user's request after 3,000
  completed evaluation records. The user subsequently canceled the proposed
  rerun. Do not restart, recompute, or relabel that interrupted run as completed.
- The canceled-run constraint does not prohibit small validation checks needed
  for authorized runtime integration, or ordinary lightweight unit tests.
- Explain changes and findings in plain language. Distinguish measured results
  from proposals, smoke checks, historical experiments, and unverified claims.

## Preserve the prediction task and input boundary

- Predict the **target system's** success, not Laya's ability to answer the task.
- The active question is `configs/laya_question.json`, prompt version
  `task_correctness_v1`: “Will the target AI system, using the supplied settings,
  solve this task correctly on a single attempt?” Laya already returns a
  probability. Do not reintroduce generic criteria or extra probability
  instructions without a new user-directed experiment.
- `src/paiec/competition.py` projects records to the published eight subject and
  four item fields. Missing values stay empty strings at that interface.
  Do not enrich missing model names with `display_name`, or supply reference
  answers, grading rules, target outcomes, or benchmark metadata to prediction.
- Benchmark IDs are anonymous at the predictor boundary. The Laya renderer
  omits IDs and empty fields from its text; IDs remain in code to match evidence.
  Preserve the contents of supplied fields, including benchmark mentions that
  naturally occur in harness names or task text.
- `configs/benchmarks/*.json` are retained public-data research notes. They are
  not an active question registry. Do not restore prompt selection by benchmark
  name or infer the identities of hidden competition benchmarks.
- Revealed examples are optional and budget-limited. Default Laya inference
  ignores them. When enabled, only matching subject/benchmark examples enter
  the state, with their recorded numeric labels. Never use evaluation labels.
- Preserve raw labels. Some public tables encode attack success or a decision,
  rather than correctness; do not silently invert or reinterpret them.

## Environment and verification

- Branch at handoff: `laya`. Check Git status for the current commit and pending
  changes; the initial handoff preceded the session commit. Preserve existing
  work. Do not reset, discard, or commit changes unless the user's task calls
  for that action.
- Use `.venv-laya-mlx/bin/python` for native MLX (Laya-MLX 0.5.0, MLX 0.32.4),
  or `.venv-laya/bin/python` for Laya 0.4.13 / PyTorch MPS. Both use Python 3.12.
  Keep the environments separate; the MLX environment has no torch/transformers.
  The original `.venv` uses Python 3.14. Preserve the PyTorch reference environment.
- The MLX/tracking environment is recorded in `requirements/mlx.lock`; the
  PyTorch/tracking reference is recorded in `requirements/torch.lock`.
  `requirements/archive/laya-pretracking.lock` predates MLflow installation
  and has dependency differences; use the combined lock when tracking is needed.
- A bounded 256-row, 10-update head-only training smoke check has run.
  This is machinery validation, not a model-quality result or a full pilot.
  Cached model weights, payload, and run artifacts live outside Git. Avoid loading the roughly 23.4 GB trace data for
  routine inspection; the scoring path does not need it.
- Usual lightweight check, when relevant to authorized code changes:

  ```bash
  .venv-laya-mlx/bin/python -m unittest discover -s tests -p 'test_*.py'
  git diff --check
  ```

  See `runs/repo_cleanup_mlx_tests.log`, `runs/repo_cleanup_torch_tests.log` and
  `runs/repo_cleanup_clean_export_tests.log` for current full test results. Documentation-only
  work does not require new model inference or evaluation runs.

## Evaluation and experiment integrity

- A subject is the model plus harness/configuration. `n_pairs` counts
  subject–benchmark combinations; `n_eval_inputs` counts scored subject–item
  records; `forward_items` counts actual Laya inference inputs after caching.
  Do not equate those counts or treat repeated budgets as new questions.
- Brier ALC is lower-is-better; always returning 0.5 gives exactly 0.25.
  Preserve the 50/50 item split, trial grouping, acquisition exclusion, nested
  budgets 0/1/3/7/15/31, and equal weighting of subject–benchmark pairs.
- Use identical seeds, dataset/checkpoint revisions, and sample manifests for
  direct prompt/backend comparisons. The `sample_10k` suite generates a fresh
  seed unless `--seed NUMBER` is supplied; the smaller existing suites are fixed
  by default. Do not launch the large suite merely because it exists.
- Prior/text training in `scripts/run_eval.py` excludes the entire evaluation
  suite and keys caches by training selection and corpus manifest. Preserve
  that boundary. Lower-level cache helpers and `SubjectPrior` alone do not
  guarantee holdouts if called with contaminated data.
- Local trained artifacts require explicit `checkpoint_path` /
  `--laya-checkpoint`, with verified `paiec_training.json` hashes. The default
  remains the original pinned checkpoint. Smoke checkpoints are uncalibrated.
- `src/paiec/laya_runtime.py` owns backend loading; `LayaPredictor` owns the
  shared state/question/cache. Preserve this separation. Both engines default to the
  same original pinned upstream weights; never substitute community checkpoint
  weights silently. Native libraries are imported only for the selected backend.
- MLX defaults to FP16; FP32 is available for closer probability fidelity.
  PyTorch retains its automatic precision policy. Log actual precision and
  compare probabilities as well as token counts; speed alone is insufficient.
- Log exact prompts, actual backend/device/precision, package and checkpoint
  versions, seed/sample IDs, truncation, and source/config state for new work.
  Do not overwrite historical metrics to make a changed experiment appear
  equivalent to an older run.
- The runner currently saves final predictions/scores only after a predictor
  completes. Running progress counters are **not** prediction checkpoints.
  Graceful interruption, resumable checkpoints, and partial-score finalization
  are outstanding work, not implemented features. Never promise them as present.
- Local input conformity does not establish Codabench packaging/runtime
  compatibility. Hosted execution has not been verified.

<!-- CODEGRAPH_START -->
## CodeGraph

In repositories indexed by CodeGraph (a `.codegraph/` directory exists at the
repo root), reach for it BEFORE grep/find or reading files when you need to
understand or locate code:

- **MCP tool** (when available): `codegraph_explore` answers most code questions
  in one call: relevant symbols' verbatim source and call paths. Name a file or
  symbol to read its current line-numbered source.
- **Shell**: `codegraph explore "<symbol names or question>"` prints the same
  output.

If there is no `.codegraph/` directory, skip CodeGraph entirely. Indexing is the
user's decision. There was no `.codegraph/` directory at this handoff.
<!-- CODEGRAPH_END -->
