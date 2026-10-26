# Feature lift loop

A fast, fixed test of whether a feature improves P(correct) on **benchmarks the
model has never seen**. Every feature idea (items, subjects) is scored the same
way against the same baseline, in under a minute. Winners are then confirmed
with the official harness, `scripts/run_eval.py`.

By default it scores every official budget (0, 1, 3, 7, 15, 31) on the same
held-out rows and combines them with the official weights, so the headline
number is a proxy for Brier ALC; see [Budget mode](#budget-mode).

## Quick start

```bash
# 1. Once per data revision (~40 s each)
python scripts/lift/build_lift_sample.py      # runs/lift/sample.parquet + manifest
python scripts/lift/build_subject_fields.py   # runs/features/subject_fields.parquet

# 2. Score a feature table on top of KM (all budgets, a few minutes)
python scripts/lift/run_lift.py --base KM --features runs/features/subjects.parquet \
    --columns release_days --name release_days

# 3. Compare two baselines
python scripts/lift/run_lift.py --base A --candidate K --name k_vs_a

# 4. Quick check at chosen budgets, or the original budget-0 loop on all rows
python scripts/lift/run_lift.py --base KM --features runs/features/subjects.parquet --columns release_days --budgets 1 7
python scripts/lift/run_lift.py --budgets --base A --abilities normalized_name
```

Each run prints a report and saves it to `runs/lift/<time>_<name>/`; every run
is appended to `runs/lift/history.jsonl`.

## Adding a feature

1. Write a table keyed by `item_id` (plus `benchmark`) or by `subject_id`, one
   row per key, under `runs/features/`.
2. Build it **from predictor-visible fields only**: `item_content`,
   `item_features`, `interactors`, and the subject fields in
   `schema.SUBJECT_FIELDS`. Never `display_name`, `grading_criterion`,
   `verifier`, labels, or `benchmarks.parquet` metadata.
3. Compute each row on its own. Anything fitted across rows (vocabularies,
   PCA, scalers) or learned from labels belongs inside the folds, like
   `lift.in_fold_ability`. `lift_report.is_label_free` checks a builder.
4. Run `scripts/lift/run_lift.py` and read the report.

## Reading the report

```
verdict: NO CLEAR EFFECT   (screening rule: |mean| >= 0.001 and >= 4 folds agree; ...)
brier  base 0.2523  candidate 0.2528  delta +0.0005 ± 0.0010  (better in 2/5 folds)
fold  bench    rows   base    cand    delta   macro Δ  logloss Δ
   0     25   259141  0.2467  0.2461  -0.0006  -0.0004  -0.0012
   ...
most improved / most worsened:  per benchmark
fingerprint check:  log_chars  filled 100.0%  fingerprint 0.87  WARNING
```

- **delta** is candidate minus base *in the same fold* (negative = better).
  Compare it with its ± spread, not with the raw fold scores: those swing with
  each fold's base rate.
- **verdict** is `HELPS` / `HURTS` only if the mean moves by at least 0.001 and
  at least 4 of 5 folds agree. A pure noise column costs a steady +0.0003.
- **macro Δ** weights every benchmark equally; the plain delta weights rows.
- **fingerprint** is the share of a column explained by which benchmark a row
  is from. At ≥ 0.8 the model can recognise training benchmarks and memorise
  their difficulty, which never transfers. Express such features relative to
  their benchmark, or expect them to hurt.

## How it stays honest

| Guard | Where |
| --- | --- |
| Whole benchmarks held out; families share a fold | `folds.grouped_benchmark_folds`, `configs/lift/benchmark_groups.json` |
| 31 benchmarks sealed, never read while choosing features | `fold == "sealed"`; tested by flipping their labels |
| Fixed sample, seed and model settings | `lift_sample`, `lift.PARAMS`, `lift.ROUNDS` |
| Label-based features rebuilt per fold, leaving out the row's family | `lift.in_fold_ability` |
| Ability enters the model alone and monotone, so it cannot act as a benchmark ID | `lift.run_lift` |
| Paired deltas, a minimum effect size, a fingerprint warning | `lift_report` |
| Every run logged with code, sample, config and feature-table hashes | `runs/lift/history.jsonl` |

**Families** are benchmarks that are not independent: they share items (BBQ
inside `helm_bbq`) or come from one evaluation run (the ten `medhelm_*`). Each
group in the config lists its `reasons` and `evidence`. On a new data revision,
re-run `scripts/lift/scan_benchmark_families.py`, update the config, and rebuild the
sample. A name prefix alone is not evidence.

## Budget mode

The competition reveals labels during evaluation. Each (system, benchmark)
pair's items are split 50/50 with `split.make_split`: at budget k the first k
acquisition items are revealed, and the evaluation half is scored at every
budget. The revealed labels include other systems on the same anonymous
benchmark (`baseline/empirical_mean/model.py` filters `labeled` by subject),
so from budget 1 the benchmark's level can be estimated. `lift_budget` copies
this for training and test benchmarks and adds four columns to each scored
row: `bench_revealed_rate`, `bench_revealed_n`, `pair_revealed_rate`,
`pair_revealed_n`. A revealed item is never scored.

Baselines in budget mode:

| Name | Prediction |
| --- | --- |
| A | training correct rate |
| B | A + subject ability |
| E | the organizers' empirical mean: the pair's revealed rate, or 0.5 |
| K | the pair's revealed rate pulled toward the benchmark's (5 pseudo-labels), itself pulled toward the training rate (20) |
| KM | LightGBM that starts from K's log-odds and learns corrections from feature columns; with no columns it is K. **Add features here.** |

The revealed rates are not tree inputs: each is one value per training
benchmark, and trees on them memorise training benchmarks. On real data that
version matched K with more fold-to-fold noise.

## Baselines (data revision `54074453`)

ALC-weighted Brier over all six budgets, paired against A:

| | Brier | Δ vs A | Verdict |
| --- | ---: | ---: | --- |
| A: training correct rate | 0.2523 | | |
| E: organizers' empirical mean | 0.2457 | −0.0067 ± 0.0185 | NO CLEAR EFFECT |
| **K / KM: shrunk revealed rate** | **0.2034** | **−0.0489 ± 0.0159** | **HELPS (5/5 folds, 100 of 125 benchmarks)** |

Per budget, K goes from 0.2523 (budget 0) to 0.2122, 0.2009, 0.1952, 0.1895
and 0.1864 (budget 31). E is unstable at budget 1 (0.3595: one label gives 0
or 1).

The budget-0 loop on all rows (`--budgets` with no values) gives A 0.2523 and
B 0.2528 (+0.0005 ± 0.0010, no clear effect). Knowing each test benchmark's
true rate would give about 0.199: **at budget 0 the unknown is benchmark
difficulty**, subject strength alone does not help, and raw text features
mostly act as benchmark IDs (text length: fingerprint 0.87, +0.0027).

## Limits

- Labels are revealed in the default random order; a custom acquisition
  function is not modelled.
- The revealed labels of other systems on the same item are not used yet
  (70% of rows have an item that another system also answered).
- Thresholds (0.8, 0.001, 4 of 5) are judgment calls; the fingerprint score
  looks at one column at a time.
- Row-weighted Brier is a proxy for the official pair-weighted Brier ALC.

## Files

Lift work lives under `scripts/lift/`, `tests/lift/` and `configs/lift/`; the
library code is in `src/paiec/` (the grouped folds extend the shared
`folds.py`).

| Path | Role |
| --- | --- |
| `scripts/lift/build_lift_sample.py`, `src/paiec/lift_sample.py` | capped sample (20k responses per benchmark) with folds |
| `scripts/lift/build_subject_fields.py`, `src/paiec/subject_fields.py` | visible subject fields keyed by `subject_id` |
| `src/paiec/lift.py` | joins, folds, LightGBM, in-fold ability, baselines |
| `src/paiec/lift_budget.py` | budget mode: official split, revealed columns, baselines E, K, KM, ALC weighting |
| `src/paiec/lift_report.py` | paired comparison, verdict, fingerprints, per-budget report, run log |
| `scripts/lift/run_lift.py` | the command |
| `scripts/lift/scan_benchmark_families.py` | evidence for family groups |
| `configs/lift/benchmark_groups.json` | family groups (`benchmark_groups_v2`) |
| `tests/lift/` | tests for all of the above (`pytest tests/lift`) |
