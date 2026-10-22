# Local experiment tracking

MLflow is installed in `.venv-laya`. Evaluation runs are logged automatically
when that optional dependency is available. Everything uses the project-local
SQLite database and artifact directories under `runs/mlflow/` by default.
Logging does not require the web dashboard to be running.

## Open the dashboard

From the project root:

```bash
.venv-laya/bin/python scripts/mlflow_ui.py
```

Open http://127.0.0.1:5001, select **PAIEC**, then **Model training → Runs**.
The GenAI view is for a different MLflow workflow; our probability-model
experiments are listed in Model training. Stop the server with Ctrl+C. Restarting
it reads the saved records. Port 5001 is used to avoid macOS services on 5000;
choose another port with `--port` if necessary.

The server binds to localhost. No hosted account is required. Backend usage
telemetry defaults to disabled in the project launcher and tracking client.

## Record an experiment

```bash
.venv-laya/bin/python scripts/run_eval.py --suite smoke --predictor constant empirical_mean laya
.venv-laya/bin/python scripts/run_eval.py --suite smoke --predictor laya --laya-model multilingual
```

Each predictor gets its own immutable MLflow run with a descriptive name. Repeat
commands to create new runs rather than replacing earlier MLflow records. The
existing CSV output still works as before and can overwrite a predictor in its
output directory; MLflow retains that run's independent artifact copies.

The smoke suite has only 16 distinct evaluation inputs across two pairs. Its
scores verify operation, not relative model quality. It includes image-referencing
AI2D examples while Laya receives stored text only; see the EDA report.

Useful options:

- `--experiment NAME`: organize runs in a separate experiment.
- `--no-tracking`: evaluate without MLflow. If MLflow is not installed, the
  runner prints a notice and still evaluates normally.
- `--suite sample_10k`: sample 100 pairs × 100 evaluation questions with a fresh
  saved seed, balanced across the default suite's benchmarks.
- `--seed NUMBER`: repeat a sample; `--seed random` explicitly generates a fresh
  seed for any suite. All predictors in a command share one sample.
- `--out PATH`: choose the existing CSV output folder.
- `--tracking-uri URI`: explicitly override the local store. Keep the default
  for the dashboard started by this project's launcher. An explicit external
  URI may send run records and artifacts to that destination.

## What is recorded

- Predictor, suite, seed, pool caps, benchmark selection, and dataset revision.
- Suite and sample hashes, the resolved random seed, and `sample_manifest.csv`
  containing every acquisition/evaluation item ID and position. Use the same
  seed and dataset revision to reproduce a sample.
- For Laya: checkpoint commit, English/multilingual choice, actual device,
  context limit, batch size, revealed-example mode, prompt version, and question
  definition/hash. The current version is `task_correctness_v1`.
- Brier ALC, Brier at each budget, per-benchmark Brier ALC, evaluation time,
  number of scored pairs, `n_eval_inputs` (system–question records at budget
  zero), and Laya truncation counts. Laya also logs `completed_pairs` and running
  inference/truncation counts during execution.
- `predictions.csv`: probability for every scored subject-item-budget input,
  with the number of recorded trials and positive outcomes. Those counts allow
  reconstructing response-weighted Brier scores; acquisition items are excluded.
- `scores/`: per-pair/per-budget, per-pair ALC, and per-benchmark tables.
- `input_examples.json`: up to eight actual subject/item inputs (up to four
  from each pair), without grading criteria or reference answers. These are
  illustrations, not an exhaustive input archive.
- `competition_input_examples.jsonl`: sample budget-zero calls containing only
  `input` (subject and item) and `labeled` (empty), also saved beside local scores.
  This interface sample retains anonymous IDs and empty fields; Laya state
  examples show the rendered text after omitting them.
- `question_definition.json`, `config.json`, and `run_details.json`.
- New runs carry `input_contract=codabench_v1`. Laya saves exact state/question
  examples and one shared correctness question without added criteria.
  `laya_prompt_version=task_correctness_v1` distinguishes these from earlier
  prompts. No benchmark-specific definition is loaded; see [benchmark-questions.md](benchmark-questions.md).
- `reproducibility/`: source/config/test snapshot, package versions, Git commit,
  branch/dirty flag, and data download manifest. The source snapshot includes
  uncommitted source files; a commit alone would not describe this working tree.

Source snapshots contain the small project source/config/test files, not the
payload, model weights, environment directory, credentials, or upstream submodule.
The official baseline submodule version remains pinned by the parent repository.
The snapshot is not a standalone submission package or a replacement for the
original data release. All traces/model weights remain outside the tracking store.

Failed or interrupted predictor runs are retained with FAILED/KILLED status
and an error artifact. Configuration errors are rejected before expensive model
initialization. Data-loading failures before a predictor run begins are reported
by the command and do not create a run.

**Interruption does not currently save partial predictions or scores.** The
runner holds those in memory and writes them after a predictor completes.
`completed_pairs`, `forward_items`, and `truncated_items` are progress counters,
not resumable checkpoints. A sample manifest identifies tasks but contains no
predicted probabilities, so it cannot reconstruct a missing Brier score.

The large Laya run `9004df1086904bc9bfa7bf70cd3e2988` was intentionally stopped
after 30 completed pairs / 3,000 evaluation records. It remains KILLED, with its
original 10,000-input plan and an added stop summary/completed-ID manifest.
No partial Laya Brier ALC exists, and the user canceled recomputation. The
constant/empirical-mean runs from the same command completed all 10,000 inputs;
their metrics must not be presented as metrics on the 3,000-record subset.
See [project-status.md](project-status.md) for IDs and evidence locations.

## ECE names

`ece_budget_mean` preserves the evaluator's existing unweighted diagnostic.
`ece_alc` applies the competition's budget weights to per-budget, equal-pair
ECE. Brier ALC and the original scoring logic are unchanged. This avoids calling
the legacy unweighted mean the official ECE summary.

## Pydantic validation

The runner validates suite/model settings with Pydantic before loading pairs:
benchmark and predictor lists must be nonempty and unique, unknown options
are rejected, acquisition caps must cover 31 labels, and sizes, context budgets,
and shrinkage weights must be valid. Duplicate benchmarks would change scoring
weights; this validation catches that mistake early.

The shared Laya question is also validated with Pydantic. The old benchmark
registry remains a set of public-data research notes and is not used during
prediction. All predictors receive only the published input fields with
anonymous benchmark IDs. Prior/text controls exclude the complete evaluation
suite during training and use caches keyed by training selection and manifest.
Earlier runs retain their original inputs and metrics; they predate this
boundary and must not be interpreted as corrected results.

## Install/reproduce

```bash
uv pip install --python .venv-laya/bin/python -e '.[tracking]'
```

For the exact combined development environment tested here (Laya, EDA, and
tracking), install `requirements-dev.lock` together with the editable project:

```bash
uv venv --python 3.12 .venv-laya
uv pip install --python .venv-laya/bin/python -r requirements-dev.lock -e .
```

MLflow 3.17.2 requires dependency versions that differ slightly from the original
Laya-only environment; use the combined lock for this workflow. Each run records
its actual installed versions as well.

Storage:

```text
runs/mlflow/mlflow.db       run metadata, parameters, metrics
runs/mlflow/artifacts/      per-run artifacts and source snapshots
```

Back up the database and artifacts together while no evaluation or server is
writing. Git ignores `runs/`; these records are local, not committed to Git.

Upstream references: [MLflow Tracking](https://mlflow.org/docs/latest/ml/tracking/tracking-api/),
[local tracking server](https://mlflow.org/docs/latest/self-hosting/architecture/tracking-server/).
