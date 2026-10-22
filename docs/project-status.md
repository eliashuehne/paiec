# Project status and handoff

**Historical pre-MLX handoff.** The user subsequently authorized the runtime
integration. It is now implemented with MLX as default and PyTorch selectable;
see [laya-backends.md](laya-backends.md) for current setup and validation. The
canceled large run remains canceled. Deferred-migration statements below
record the earlier state, not the current authorization.

Recorded **2026-10-21, America/Los_Angeles**. This document records the current
working tree and saved experiment evidence, not a new evaluation. No model
inference, training, installation, or migration was performed to write it.

## User direction and immediate state

The user plans to explore switching to
[Laya-MLX](https://github.com/mizorewww/laya-mlx), but will explicitly say when
to start. No MLX migration has begun. The canceled inference run must not be
restarted or recomputed without a new request. A proposed rerun to obtain
partial Brier scores was explicitly canceled before it started.

The current backend remains **PyTorch/MPS**, using the M4 MacBook GPU rather
than the Apple Neural Engine. The large inference process exited after the
requested stop; the MLflow dashboard is independent of that process.

At the initial, pre-commit handoff, work was on branch `laya`, HEAD
`6886ebfaa6a0bf1edf02c6c7c381bcc455cc9ade`, with substantial uncommitted changes.
The baseline submodule remained pinned at
`cc88a959c1e5e7462f31a9215baba066ce6d6e2b`. The user subsequently requested a
local session commit without pushing; inspect Git log/status for the resulting
revision rather than treating the pre-commit HEAD above as current. Use MLflow
source snapshots to identify the working tree actually used by earlier runs.

## Implemented work

| Area | Current implementation |
| --- | --- |
| Pretrained inference | `src/paiec/laya_predictor.py`: English/multilingual checkpoints, MPS/CUDA/CPU selection, batch prediction, bounded state/question cache, probability validation, truncation counters |
| Official input fields | `src/paiec/competition.py`: strict field projection, empty missing values, anonymous local benchmark IDs, same projection for revealed evidence |
| Shared prompt | `configs/laya_question.json`, validated by `LayaQuestion` in `src/paiec/config.py` |
| Experiment tracking | `src/paiec/tracking.py`, `scripts/mlflow_ui.py`: local SQLite MLflow, exact configs/prompts, predictions and score tables for completed runs, source snapshots, package/data/checkpoint provenance |
| Random samples | `src/paiec/pairs.py`, `scripts/run_eval.py`, `configs/suites.json`: seeded random subject selection, benchmark balancing, sample manifests/hashes, fresh-seed large suite |
| EDA | `scripts/eda.py`, local report and statistics in `runs/eda/` |
| Outcome audit | `scripts/audit_outcomes.py`, `docs/outcome-semantics.md`, local evidence in `runs/outcome_audit/` |
| Holdouts and caches | Runner excludes evaluation-suite benchmarks before prior/text training; cache names include training selection and manifest fingerprint |

There is no fine-tuning, no fitted Laya adaptation, and no verified hosted
Codabench submission. The optional revealed-example mode only changes input
text; it does not update weights.

## Current input to Laya

The active prompt version is **`task_correctness_v1`**. It asks:

> Will the target AI system, using the supplied settings, solve this task correctly on a single attempt?

The question schema contains only `type: "noul"` and `instructions`. Laya
already supplies a probability, read from `answers.correct.noul`; `correct`
is the adapter's output key. No generic criteria, recorded-outcome phrasing,
extra probability instructions, or benchmark-specific prompt selection remain.

The rendered JSON text has `target_system` and `target_item`, containing their
available permitted fields. Empty fields and benchmark IDs are omitted from
that text. The task includes its stored context/options/features/tools; nested
task JSON remains a string rather than being rewritten. When examples are
explicitly enabled, matching revealed tasks and numeric outcomes are appended
under `revealed_examples`; the shared system profile is not repeated.

The competition-facing dictionaries still contain exactly eight subject fields
and four item fields, with empty strings for missing values. Anonymous IDs
remain in code for evidence matching. Real benchmark names remain scorer-side
for selecting data, splits, and reports. Allowed field contents are preserved:
for example, a supplied harness name can itself say `BBQ`.

See [benchmark-questions.md](benchmark-questions.md) for the complete field list.
`configs/benchmarks/*.json` are research notes from an earlier approach; they
are not loaded by the active runner. The retired `--benchmark-registry` and
`--laya-question-mode` flags must not be restored accidentally.

## Important findings

### Benchmark names and grading rules are unavailable as dedicated test fields

The published [competition interface](https://aims.stanford.edu/competition)
supplies system details, task text, item features, interactors, an anonymous
benchmark ID, and budget-permitted observed examples. It does not supply a
dedicated real benchmark name or grading definition. An earlier implementation
selected prompts using real public names; that approach was removed.

The private test pool uses held-out benchmarks. Public benchmark metadata can
support research and training-data review, but cannot be assumed available at
prediction time. Never infer hidden benchmark identities for development.

The competition-page review also recorded an eight-hour runtime including
setup, bundled/prefetched model weights rather than runtime model downloads,
organizer-controlled extra dependencies, and competition-specific training
restricted to the published inventory. Network access was limited to approved
inference APIs, with web search/browsing prohibited. Recheck those policies
before packaging a submission; local MPS execution has not validated them.

### Binary labels are not universally correctness

The public Data Bank preserves native recorded events. Examples include attack
success, harmful classification decisions, and detected alignment-faking
reasoning. The audit found no evidence that our downloader accidentally inverted
labels. AdvPrompter's 100 records match archived native scores; BraveGuard's
536 records match native harmful decisions, not necessarily correctness.

The current simple correctness prompt follows the user's chosen starting point
and the competition's stated task. It does **not** resolve the event meaning of
every public table. Preserve labels and do not silently reverse them. Whether
the hidden competition pool normalizes heterogeneous events remains unresolved;
no organizer message was sent. See [outcome-semantics.md](outcome-semantics.md).

### Context and missing information affect interpretation

- AI2D and DevBench require images for some tasks; the current adapter uses
  stored text only. Assets are not supplied to Laya.
- The eight-benchmark larger suite includes multilingual content even when
  `--laya-model english` is selected.
- Model identity and other system fields are often missing. The removed
  `display_name` fallback must not fill official inputs.
- The current default context limit is 512 tokens. Long tasks and especially
  revealed examples can be truncated substantially. Counters report this;
  successful execution does not imply the model saw the complete task.
- Earlier upstream runtime warnings concerned the `choice:11+` temperature.
  This adapter uses the `noul` output. No calibration settings were changed.

### Scoring, sampling, and cache details matter

Brier ALC is lower-is-better; zero is perfect and returning 0.5 always gives
0.25. Budgets are 0, 1, 3, 7, 15, 31, with ALC weights
0.1/0.2/0.2/0.2/0.2/0.1. Responses are weighted within each pair; pairs are
then weighted equally. Acquisition items are never scored; all trials of an
item remain on one side of the 50/50 split. ECE is diagnostic; `ece_alc` uses
ALC weights while `ece_budget_mean` retains an older unweighted summary.

`n_pairs` is the number of system–benchmark combinations. `n_eval_inputs` counts
scored system–question records at one budget. `forward_items` counts actual
model inputs after caching. Different records can have identical visible text;
the six budgets do not imply six new sets of zero-shot questions.

The `sample_10k` suite samples 100 pairs × 100 evaluation items, with at least
200 available binary-graded items per selected pair. It draws a fresh seed,
samples without replacement, and balances across the existing eight benchmarks
subject to available systems. BBQ has only seven eligible systems. All
predictors in a command share one sample. Seeds and complete ordered acquisition/
evaluation IDs are recorded; reuse them for direct backend/prompt comparisons.

Earlier prior fallback counts and text-cache naming weakened holdouts. The
runner now excludes the entire evaluation suite from training and includes
training selection/data-manifest hashes in cache names. This protection lives
in runner construction: lower-level helpers are not independently safe if
called with contaminated inputs. `SubjectPrior` still has a global fallback
over whatever statistics the caller supplies.

## Saved experiment evidence

### Completed runs with the current short prompt

These are only **16 evaluation records across two pairs**, not a model-quality
study. All used the unchanged English checkpoint, 512-token limit, and checkpoint
revision `7b928d828b7b0e022f929d9bd2e44165aa270148`.

| Predictor/mode | Brier ALC | Actual Laya forwards | Truncated forwards | MLflow run |
| --- | --- | --- | --- | --- |
| Constant 50% | 0.250000 | — | — | `660327d89f5944a3a2e41c0da6dce2f4` |
| Empirical mean | 0.292460 | — | — | `0af12ae29d09474498413bec61059517` |
| Laya, no examples | 0.455436 | 16 | 0 | `df69354078094d5e8f2417997191aa5b` |
| Laya, revealed examples | 0.386943 | 96 | 73 | `aa5ceb1583e54d6b9ab69f80a5e402a4` |

Both Laya modes produced the same budget-zero score. The current short prompt
did not beat 50% on this tiny sample. No broader conclusion follows from it.
Older legacy/benchmark-specific/generic-criteria runs used different inputs;
do not mix their scores with this prompt version.

The stand-alone 50% smoke run requested afterward is
`238062a171a34ec1acf728d18031aa43`, with Brier ALC 0.25 at every budget.

### Large sample stopped at 3,000 completed records

| Field | Recorded value |
| --- | --- |
| Laya MLflow run | `9004df1086904bc9bfa7bf70cd3e2988` |
| Status | `KILLED`, intentional user-requested stop |
| Suite / seed | `sample_10k` / `1207735414` |
| Original plan | 100 pairs, 10,000 evaluation records |
| Completed evaluation | 30 pairs, 3,000 records |
| Last logged actual forwards | 2,908; some visible states reused predictions |
| Last logged truncated forwards | 1,008 |
| Backend / batch / context | MPS / 8 / 512 |
| Model / examples | English / no revealed examples |
| Final/partial Laya Brier ALC | **Not saved; unavailable** |

The full planned sample covered 7,479 distinct benchmark/question IDs across
10,000 system–question records. That is a property of the full planned sample,
not a count of completed work. A subsequent pair had begun when interrupted;
the table records completed pairs and the last persisted counters.

The baseline predictors completed the **full 10,000-record plan**:

| Predictor | Brier ALC | MLflow run |
| --- | --- | --- |
| Constant 50% | 0.250000 | `df8432d4999b450d98ee07aa5c5c12b8` |
| Empirical mean | 0.285879 | `f2cfc41512a548cbaa5e0230aeacc874` |

Those baseline scores are not scores on only the stopped 3,000-record subset.
Do not present a Laya comparison for that subset: its predictions were not
persisted and no partial Brier score was finalized.

Relevant local evidence:

- `runs/sample_10k.log`: execution and interruption log.
- `runs/sample_10k_stop.json`: stop monitor's last observations.
- `runs/eval_sample_10k/stopped_at_3000_seed_1207735414/stop_summary.json`.
- The same directory's `completed_sample_manifest.csv`: acquisition and
  evaluation IDs for the 30 completed pairs; it contains no probabilities.
- MLflow: original full `sample_manifest.csv`, added completed manifest and stop
  summary, and progress counters. The original 10,000-input parameters remain
  historical facts; the run was not rewritten as a completed 3,000-input run.

**Important failure mode:** predictions and per-pair scores remain in process
memory until a predictor completes. Progress counters and sample manifests are
not prediction checkpoints. An interrupt retains the MLflow record and error,
but does not save partial predictions or finalize Brier ALC. Checkpointing,
resuming, and graceful partial finalization were discussed but not implemented.
The user declined recomputation. No inference has been rerun to repair it.

## Environment, data, and local artifacts

- `.venv-laya`: Python 3.12.13, Laya 0.4.13, PyTorch 2.14.1,
  Transformers 5.20.2, MLflow 3.17.2, Pydantic 2.14.1, PyArrow 25.0.1.
  Use `requirements-dev.lock` for the tested combined environment. The earlier
  Laya-only lock predates dependency changes needed for MLflow.
- `.venv`: separate original Python 3.14 environment.
- Public data: `aims-foundations/measurement-db`, pinned revision
  `54074453da7b9911be0f48e4c560be547bc84d21`, under `payload/data/`.
- EDA: 61 benchmarks; 8,163 subject configuration rows; 1,356,713 items;
  16,423,758 response rows, of which 137,549 are missing outcomes;
  6,870 eligible binary-observed pairs. Subject rows are not distinct base LLMs.
- 304 Parquet files total roughly 24.22 GB. Traces account for roughly
  23.42 GB (96.7%); routine scoring/EDA does not load their content. The
  AdvPrompter/BraveGuard audit inspected selected archived native fields.
- `runs/eda/report.html` and CSVs contain detailed EDA; `runs/outcome_audit/`
  contains semantic checks. `runs/` and `payload/` are ignored by Git.
- MLflow backend: `runs/mlflow/mlflow.db`; artifacts:
  `runs/mlflow/artifacts/`; dashboard: `http://127.0.0.1:5001`, experiment
  `PAIEC`. Direct SQLite tracking works without the dashboard process.
- Model weights use the normal Hugging Face cache outside Git. Both English
  and multilingual checkpoints were cached; model revision is logged per run.

The latest recorded complete unit-test execution passed **105 tests**:
`runs/sampling_tests.log`. It was not rerun for this documentation task. The
later progress-logging change also executed during the stopped real run. No
hosted Codabench execution or MLX compatibility test has been completed.

## Deferred Laya-MLX work: reference only

[Laya-MLX](https://github.com/mizorewww/laya-mlx) is a separate community native
Apple MLX runtime, not the backend used by the installed `laya` package. Its
README inspected during this session described version 0.5.0 targeting upstream
v0.4.12-era APIs, including `Router.predict_batch`. Our installed upstream
package is 0.4.13. Compatibility is not established merely by matching API names.
MLX is not installed in `.venv-laya`.

Published speed figures concern that port's own hardware and short fixtures.
They are not measured speedups on this M4 or these long benchmark inputs. The
current run used PyTorch/MPS; GPU acceleration does not imply MLX or Neural
Engine use.

When the user explicitly requests migration, the likely work is:

1. Pin and inspect the port; use a separate environment and keep the existing
   PyTorch environment as a reference.
2. Establish weight provenance/equivalence to the exact checkpoint revision.
   Do not assume a community model repository or a similarly named fine-tuned
   checkpoint has identical weights, calibration, or tokenizer behavior.
3. Verify the same rendered state, simple `noul` question, output probability
   interpretation, truncation accounting, and context limits. Check probability
   differences, not only selected yes/no answers; Brier uses the probabilities.
4. Compare fixed examples with matched precision where possible; record any
   FP16/FP32 differences. Separate prompt changes from backend changes.
5. Measure warmed-up, synchronized latency/throughput and memory on this M4 at
   realistic lengths and batch sizes. Do not use headline short-input timings
   as an estimate for the full benchmark mix.
6. Adapt tracking to log actual backend, dtype, MLX/port version, weight and
   tokenizer provenance, and conversion revision. `scripts/run_eval.py`
   currently logs a PyTorch-specific package list and uses the upstream Hugging
   Face checkpoint resolver.
7. Preserve the competition input boundary, dataset holdouts, budget semantics,
   recorded prompt version/hash, and same-seed comparison.

Relevant integration points are `src/paiec/laya_predictor.py` (runtime/router,
device, request/result handling), `scripts/run_eval.py` (model construction,
provenance, artifacts), `src/paiec/config.py`, dependencies/locks, and adapter
tests. The injected `router` constructor parameter already supports testing
the adapter without loading a model.

Graceful interruption and incremental prediction/score saving remain a separate
open engineering issue. Hosted competition packaging, dependency policy, model
prefetching, and allowed runtime hardware also need validation before a real
submission; a local MLX backend must not be assumed deployable on Codabench.

### MLX memory finding (post-handoff)

The MLX backend's default cache limit equalled its memory limit at `16.32 GB`,
above this 16 GB machine's `12.71 GB` recommended working set. Because MLX
caches freed Metal buffers per distinct tensor shape and Laya states vary in
length, the cache grew with every new batch shape (~0.78 GB each at batch
eight) with no eviction. This is the plausible cause of the earlier memory
exhaustion, and it is specific to the MLX path; PyTorch/MPS evicts under
`PYTORCH_MPS_HIGH_WATERMARK_RATIO`.

The runtime now sets a 2 GiB cache limit and a 6 GiB memory limit before the
checkpoint loads and clears the cache between scoring pairs. Bounded probes on
real tables held peak resident memory near 1.5 GB over 216 items, with the
cache plateauing at 2.15 GB inside an untrimmed 100-item pair. Trimming cost
about 2% throughput on a repeated 100-item pair. No evaluation suite was
launched for this change.

This list is a handoff, not authorization to begin migration or restart runs.
