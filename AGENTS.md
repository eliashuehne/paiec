# Working on PAIEC / Laya

## Start here

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
- `docs/head-data-split.md`: frozen pilot families and exports, preparation
  checks and truncation limits; use `configs/head_finetuning/frozen_split_v1.json`.
- `docs/laya.md`: environment, sampling, context limits, and historical results.
- `docs/mlflow.md`: tracking, artifacts, and interrupted-run limitations.
- `docs/outcome-semantics.md`: native binary labels are not always correctness.
- `docs/data-notes.md`: dataset provenance, tables, and EDA.

- `docs/modal.md`: execution profiles, authenticated client, cloud data/tracking
  layout and remaining GPU qualification. No paid job has been run.

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
- The MLX/tracking environment is recorded in `requirements-mlx.lock`; the
  PyTorch/tracking reference is recorded in `requirements-dev.lock`.
  `requirements-laya.lock` predates MLflow installation and has dependency
  differences; use the combined lock when tracking is needed.
- A bounded 256-row, 10-update head-only training smoke check has run.
  This is machinery validation, not a model-quality result or a full pilot.
  Cached model weights, payload, and run artifacts live outside Git. Avoid loading the roughly 23.4 GB trace data for
  routine inspection; the scoring path does not need it.
- Usual lightweight check, when relevant to authorized code changes:

  ```bash
  .venv-laya-mlx/bin/python -m unittest discover -s tests -p 'test_*.py'
  git diff --check
  ```

  See `runs/mlx_tests.log` and `runs/torch_tests.log` for current full test results. Documentation-only
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
