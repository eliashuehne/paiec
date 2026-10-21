# PAIEC predictors

Local harness and predictors for the [Predictive AI Evaluation Competition](https://aimslab.stanford.edu/competition).
The task is to estimate the probability that an AI system answers an item
correctly, from the system's attributes and the item, and to adapt as a small
number of labels are revealed.

This repository holds the offline scoring harness and the reference predictors
that any new method has to beat. The competition code itself is in `baseline/`,
pulled in as a submodule at a pinned commit.

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
scripts/run_eval.py  command line runner
tests/               unit tests for the protocol, metrics, and predictors
```

Downloaded data lives in `payload/` and is not tracked. See
[docs/data-notes.md](docs/data-notes.md) for its provenance and size.

## Running

The harness needs only `numpy`, `pandas`, and `pyarrow`; the existing `.venv`
has them.

```bash
PYTHONPATH=src .venv/bin/python scripts/run_eval.py --predictor constant empirical_mean
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -p 'test_*.py'
```

Results are written to `runs/eval_<suite>/`: one scores table per predictor plus
a `summary.csv` that accumulates across runs. Use `--suite longctx` for the
long-context suite and `--k` to change the shrinkage weight.

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

Two deviations are deliberate. Local item caps (120 evaluation and 40
acquisition items per pair) keep runs to minutes; they change which items are
scored, not the structure. And the public tables use benchmark names where the
evaluation supplies anonymous ids, so `benchmark_id` is the public name here.

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

These numbers are not a cold-start estimate. Every predictor here either
ignores the evaluation benchmark or excludes it from its statistics, but the
suite's benchmarks are still public data. A learned model must be scored with
benchmark-level holdouts, which is what the Laya setup is for.

## Adding a predictor

A predictor is any callable `(subject, item, labeled) -> float` returning a
finite probability in [0, 1]. `labeled` is a list of `[[subject, item], label]`
entries, exactly as the evaluator supplies them. Register a new name in
`scripts/run_eval.py` and add a unit test that pins its behaviour.
