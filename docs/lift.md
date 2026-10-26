# Feature lift loop

A fast, fixed test of whether a feature improves P(correct) on **benchmarks the
model has never seen**. Every feature idea (items, subjects) is scored the same
way against the same baseline, in under a minute. Winners are then confirmed
with the official harness, `scripts/run_eval.py`.

It measures the **budget-0** case: the test benchmark contributes no labels.
That is 10% of the Brier ALC weight; see [Limits](#limits).

## Quick start

```bash
# 1. Once per data revision (~40 s each)
python scripts/lift/build_lift_sample.py      # runs/lift/sample.parquet + manifest
python scripts/lift/build_subject_fields.py   # runs/features/subject_fields.parquet

# 2. Score a feature table against baseline A
python scripts/lift/run_lift.py --features runs/features/items.parquet \
    --columns n_chars has_code --name text_basics

# 3. Score subject ability (baseline B against A)
python scripts/lift/run_lift.py --base A --abilities normalized_name --name b_vs_a
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

## Baselines (data revision `54074453`)

| | Brier | Δ vs A (paired) | Verdict |
| --- | ---: | ---: | --- |
| A: training correct rate | 0.2523 | | |
| B: A + subject ability (`normalized_name`) | 0.2528 | +0.0005 ± 0.0010 | NO CLEAR EFFECT |

Knowing each test benchmark's true correct rate would give about 0.199. **At
budget 0 the unknown is benchmark difficulty**: subject strength alone does not
help, and raw text features mostly act as benchmark IDs (text length has
fingerprint 0.87 and hurts by +0.0027).

## Limits

- **Budget 0 only.** At budgets 1–31 the predictor also sees revealed labels,
  apparently including other systems' labels on the same anonymous benchmark
  (`baseline/empirical_mean/model.py` filters `labeled` by subject). Planned
  next: a budget-k mode that reveals k labels per system on the test benchmark.
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
| `src/paiec/lift_report.py` | paired comparison, verdict, fingerprints, run log |
| `scripts/lift/run_lift.py` | the command |
| `scripts/lift/scan_benchmark_families.py` | evidence for family groups |
| `configs/lift/benchmark_groups.json` | family groups (`benchmark_groups_v2`) |
| `tests/lift/` | tests for all of the above (`pytest tests/lift`) |
