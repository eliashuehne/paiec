# PAIEC predictors

Local harness and predictors for the [Predictive AI Evaluation Competition](https://aims.stanford.edu/competition).
The task is to estimate the probability that an AI system answers an item
correctly, from the system's attributes and the item, and to adapt as a small
number of labels are revealed.

This repository holds the offline scoring harness and the reference predictors
that any new method has to beat. The competition code itself is in `baseline/`,
pulled in as a submodule at a pinned commit.

## Current status

Read [docs/laya-backends.md](docs/laya-backends.md) for the current selectable
MLX/PyTorch runtimes, and [docs/project-status.md](docs/project-status.md) for the
earlier handoff, measured results, known limitations, and interrupted-run history. Agent
instructions are in [AGENTS.md](AGENTS.md).

The large Laya run was intentionally stopped after 3,000 completed evaluation
records. It has no saved partial predictions or Brier ALC; the proposed rerun
was canceled. The current short-prompt smoke result did not beat the 50%
baseline, and its 16 inputs are too few for broad conclusions. The user has
since authorized MLX integration: it is now the default runtime, with PyTorch
still selectable. The canceled run has not been restarted or recomputed.

## Layout

```
baseline/            submodule: aims-foundations/paiec_baseline
configs/suites.json  benchmark lists and per-pair item caps for each suite
src/paiec/           harness library
  schema.py          public tables -> the evaluator's [subject, item] dictionaries
  pairs.py           subject-benchmark pairs, with the 80-item rule
  split.py           50/50 item split, nested acquisition order
  metrics.py         Brier, ECE, Brier ALC
  evaluate.py        the scoring loop
  predictors.py      constant, empirical mean, subject prior, shrunk mean
  stats.py           cross-benchmark subject counts
  textmodel.py       hashed-text ridge control
  laya_predictor.py  pretrained English/multilingual Laya adapter
scripts/run_eval.py  command line runner
tests/               unit tests for the protocol, metrics, and predictors
```

Downloaded data lives in `payload/` and is not tracked. See
[docs/data-notes.md](docs/data-notes.md) for its provenance and size.

## Running

The harness uses `numpy`, `pandas`, and `pyarrow`; the runner also validates
settings with `pydantic`. These are installed in the existing environments.

```bash
PYTHONPATH=src .venv/bin/python scripts/run_eval.py --predictor constant empirical_mean
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -p 'test_*.py'
```

Results are written to `runs/eval_<suite>/`: one scores table per predictor plus
a `summary.csv` that accumulates across runs. Use `--suite longctx` for the
long-context suite and `--k` to change the shrinkage weight.

## Experiment dashboard

Evaluation commands automatically log to local MLflow when it is installed
(already configured in `.venv-laya`). Open the dashboard with:

```bash
.venv-laya/bin/python scripts/mlflow_ui.py
```

Visit http://127.0.0.1:5001 and open **PAIEC → Model training → Runs**. Each
completed predictor run retains its configuration, dataset/checkpoint revision, scores,
predictions, and source snapshot. Logging also works with the dashboard closed.
Use `--no-tracking` to disable it. See [docs/mlflow.md](docs/mlflow.md) for details
and [requirements-dev.lock](requirements-dev.lock) for the tested combined environment.

## Pretrained Laya

The separate `.venv-laya-mlx` environment runs pretrained Laya through native
MLX on the Mac GPU. The original `.venv-laya` PyTorch environment is preserved.
Start with a small check:

```bash
.venv-laya-mlx/bin/python scripts/run_eval.py --suite smoke --predictor constant empirical_mean laya
```

Use `--suite default` for the existing 32-pair evaluation, or add
`--laya-model multilingual` to compare the other checkpoint. No fine-tuning is
performed. Select ordinary Laya with `.venv-laya/bin/python` and
`--laya-backend torch`; select MLX with `--laya-backend mlx` (the default).
`--laya-dtype float32` enables MLX FP32; its default is FP16. See
[backend setup and measured checks](docs/laya-backends.md). The runtime caps
MLX's Metal buffer cache at 2 GiB (override: `PAIEC_MLX_CACHE_LIMIT_GB`) so
long runs stay near 1.5 GB resident instead of growing into swap.
By default Laya sees only the subject profile and target item, so its
scores stay the same at every label budget. Revealed examples are optional with
`--laya-use-labels`.

Laya asks one shared yes/no question from `configs/laya_question.json`:
“Will the target AI system, using the supplied settings, solve this task
correctly on a single attempt?” Its output already supplies the probability;
no separate criteria or probability instructions are added. Laya's text contains
available system/task details and optional revealed examples, with empty fields
and benchmark IDs omitted. Anonymous IDs remain in code to match evidence.
The files in `configs/benchmarks/` are public-data research notes, not runtime
inputs. See [docs/benchmark-questions.md](docs/benchmark-questions.md).

See [docs/laya.md](docs/laya.md) for setup, context budgets, output locations,
and reproducibility details.

For a larger random sample (100 system–benchmark pairs × 100 questions):

```bash
.venv-laya-mlx/bin/python scripts/run_eval.py --suite sample_10k --predictor constant empirical_mean laya --laya-batch-size 8
```

This suite chooses a fresh seed per run, samples systems across the eight
benchmarks in the default suite, and scores all predictors on the same 10,000
system–question records. The seed and complete acquisition/evaluation item IDs
are saved locally and in MLflow. Pass `--seed NUMBER` to repeat the sample for
controlled comparisons. Scoring still averages system–benchmark pairs equally.

## Dataset exploration

```bash
uv pip install --python .venv-laya/bin/python -e '.[eda]'
.venv-laya/bin/python scripts/eda.py
```

This scans all non-trace tables across the 61 downloaded benchmarks and writes
`runs/eda/report.html`, a summary figure, full schemas, and CSV statistics.
Trace counts/sizes come from Parquet metadata; trace content is not loaded.
The report checks missing outcomes, profile completeness, repeated trials,
table links, item lengths, and benchmark coverage. Inspect the outcome meanings
before treating a benchmark's binary value as correct/incorrect.

## Scoring protocol

The harness follows the competition's rules where they affect a local number:

- A pair is one subject on one benchmark, scored only if it has at least 80
  distinct items.
- Items are split 50/50 into acquisition and evaluation pools. All trials of an
  item stay on the same side; acquisition items are never scored.
- The acquisition order is shuffled once per pair and seeded, so labels at
  budget 3 are a prefix of those at budget 7.
- Budgets are 0, 1, 3, 7, 15, 31. Brier is computed per pair and budget over
  responses, then averaged over pairs with equal weight.
- Brier ALC is the trapezoid of that curve on x = log2(n + 1), which reproduces
  the published weights 0.1 / 0.2 / 0.2 / 0.2 / 0.2 / 0.1. `tests/test_metrics.py`
  asserts this.

Local item caps (120 evaluation and 40 acquisition items per pair) keep runs
to minutes; they change which items are scored, not the structure. Public
benchmark names remain available to the scorer for splitting and reporting,
but predictors receive stable synthetic anonymous IDs. These local IDs do not
identify or reproduce private Codabench IDs.

## Reference results

`default` suite: 32 pairs across bbq, editbench, devbench, cmmlu, afrimedqa,
cruxeval, ai2d_test, and helm_afr. Lower is better.

| predictor | Brier ALC | Brier@0 | Brier@1 | Brier@31 | mean ECE |
| --- | --- | --- | --- | --- | --- |
| shrunk_mean | 0.2349 | 0.2466 | 0.2491 | 0.2188 | 0.134 |
| subject_prior | 0.2466 | 0.2466 | 0.2466 | 0.2466 | 0.170 |
| constant 0.5 | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.180 |
| text_ridge | 0.2706 | 0.2706 | 0.2706 | 0.2706 | 0.217 |
| empirical_mean (baseline) | 0.2864 | 0.2500 | 0.4545 | 0.2194 | 0.199 |

Three things are worth reading off this table. The organizers' empirical mean
is the worst predictor on the metric, because an unsmoothed single label is a
0-or-1 prediction and Brier ALC weights budget 1 heavily. Shrinking that label
toward a prior fixes most of the damage. And item text alone, without any
subject information, does worse than a constant, so a text model has to carry
the system profile as well.

The subject prior is weak because the public tables name systems
inconsistently across benchmarks, and most systems appear in only one. Identity
lookups are therefore a poor adaptation strategy; the state Laya sees has to
make the profile informative, not just name the model.

These numbers are historical exploratory public-data results, predating the
`codabench_v1` input boundary. Earlier subject-prior fallback counts included
the evaluation benchmark, and text-training cache names omitted the benchmark
selection. The runner now excludes the entire selected evaluation suite from
prior/text training and keys those caches by training benchmarks and data
manifest. Earlier model-name fallbacks and benchmark-specific Laya questions
also used information unavailable in the published interface. Old results are
retained for reference, not presented as results of the corrected setup.
Laya pretraining overlap with public benchmarks remains unknown; future
fine-tuning still requires benchmark-level holdouts.

## Adding a predictor

A predictor is any callable `(subject, item, labeled) -> float` returning a
finite probability in [0, 1]. `labeled` is a list of `[[subject, item], label]`
entries, exactly as the evaluator supplies them. Register a new name in
`scripts/run_eval.py` and add a unit test that pins its behaviour.
