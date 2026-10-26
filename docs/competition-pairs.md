# The pair matrix, the competition-shaped scorer, and the rate model

2026-10-25. This rebuilds the measurement unit around what the competition
actually scores — a subject–benchmark pair — instead of the 170-scope protocol
panel. CPU only, public corpus only. No Laya inference, no GPU job, no sealed
role, no upload.

Result: the competition-faithful measurement works, and the first rate model on
it does **not** beat the fixed controls. Cross-pair revealed labels are the real
lever; the structured model adds nothing over smoothing them. See
[Rate model](#rate-model) for the numbers and the gate verdict.

## Why the panel could not answer the question

The panel that carried the earlier CatBoost work holds 170 scopes, **170 distinct
systems, and no system in two benchmarks**, with a median of exactly 80 rows per
scope. A held-out scope is therefore a new system and a new benchmark at once, and
the two effects cannot be separated. Brier is dominated by the pair's overall
success rate, so what the panel actually measured was a level shift — `bird` is
120 of its 170 scopes at a 9% success rate — not transfer.

The public corpus is not thin. Measured from `payload/data` at revision
`54074453…`: 61 benchmark directories, 16,423,758 response rows, 1,356,713 item
rows. This work uses all of it.

## What was built

| Piece | Responsibility |
| --- | --- |
| `src/paiec/competition_pairs.py` | Admit benchmarks by declared rules, form subject–benchmark cells at the platform's 80-item threshold, keep native labels and direction |
| `src/paiec/competition_scorer.py` | 50/50 item split, nested budgets, cross-pair evidence, response-weighted Brier, pair-equal ALC |
| `src/paiec/rate_model.py` | Two-way ability × difficulty fit, task-text difficulty regression, optional CatBoost item term, evidence posterior |
| `src/paiec/competition_gate.py` | Fail-closed assessment against the constant, the revealed-label mean and the strongest fixed control |
| `src/paiec/competition_submission.py` | Self-contained `model.py`/`labeling.py`/`state.json`, refused unless the gate clears |
| `src/paiec/competition_tracking.py` | Optional MLflow receipt; a missing client never invalidates the run files |

Scripts: `build_competition_pairs.py`, `run_competition_controls.py`,
`run_rate_model.py`, `build_rate_model_submission.py`. Recipes:
`configs/pipelines/competition_pairs_v1.json`, `…_controls_v1.json`,
`rate_model_v1.json`.

## The matrix

| Measure | Value |
| --- | ---: |
| Benchmark directories read | 61 |
| Admitted (binary, item-level, text-only, ≥1 cell at 80 items) | 41 |
| Excluded: carries image/audio/video | 13 |
| Excluded: no subject reaches 80 items | 9 |
| Subject–benchmark cells | 6,252 |
| Labelled responses in those cells | 14,713,067 |
| Cells with a named model | 4,702 |
| Distinct named models / present in ≥2 benchmarks | 194 / 45 |
| Recorded directions: higher / lower / unordered | 1,936 / 36 / 4,280 |
| Mean cell rate | 0.471 |
| Mean `rate × (1 − rate)` — the score of knowing each cell's rate | 0.1676 |

Three properties matter downstream.

- Admission is a rule, not a judgement: `lower_is_better` benchmarks are kept and
  labelled, never flipped, and are excluded from the ability fit because pooling
  an inverted label would corrupt it.
- One benchmark, `alignment_faking`, supplies 4,280 of the cells at an
  `unordered` recorded direction. It is retained as data and excluded from the
  scored scope, so the numbers below are not driven by a single directory.
- Only 45 named models appear in two or more admitted benchmarks, so a system's
  ability is observable across benchmarks for a minority of cells. That ceiling is
  measured, not assumed.

Artifacts: `runs/competition_pipeline/pair_matrix_v1/` — `cells.jsonl`,
`benchmarks.jsonl`, `cell_items.parquet`, `items.parquet`, `item_lengths.json`,
`source_contract.json`, hashed `manifest.json`.

## The scorer

Each pair's items are split 50/50 into acquisition and evaluation pools, fixed
across budgets, with all recorded responses of an item staying together. Budgets
0/1/3/7/15/31 reveal acquisition items in a seeded arrival order; an item that
would overshoot a budget is skipped rather than partially revealed. Brier is
weighted by recorded responses using `n p² − 2 p S + S`, which is exact for any
mix of responses behind an item. The curve is weighted
`0.1/0.2/0.2/0.2/0.2/0.1` and averaged equally over pairs.

`labeled` is built **across pairs**: a prediction for one subject sees the
revealed outcomes of other subjects on the same `benchmark_id`. This is what the
interface supplies, and it is the only cheap estimator of a new benchmark's
difficulty. The earlier pipeline required an identical subject and scored each
scope in isolation, discarding that channel by construction.

## Fixed controls

Three draws of 1,000 pairs each, higher-is-better cells only, 41 benchmarks.
Lower is better.

| Arm | all | shared subject | cold subject |
| --- | ---: | ---: | ---: |
| Constant 0.5 | 0.2500 | 0.2500 | 0.2500 |
| Acquired-label mean | 0.2430 | 0.2131 | 0.2392 |
| Fixed beta prior, strength 5 | 0.2157 | 0.1960 | 0.2104 |
| Same-benchmark labels only | 0.2100 | 0.1872 | 0.2145 |
| Same-benchmark labels plus own | **0.2049** | **0.1801** | **0.2027** |
| Oracle: each cell's true rate | 0.1831 | 0.1543 | 0.1780 |

Two findings changed the rest of the plan.

- **The revealed-label mean is not the binding control here.** On the old panel it
  scored 0.1491 and beat every model, because that panel's scopes had extreme rates
  (median 0.175). With pair rates near 0.5 and budgets 1 and 3 carrying 0.4 of the
  weight, an unsmoothed mean predicts 0 or 1 from a single label and barely beats
  the constant. Shrinkage is what the metric rewards, so the gate also compares
  against the strongest fixed control.
- **Cross-pair evidence is the largest single lever available without any model.**
  Using only other subjects' labels on the same benchmark scores 0.2100 against
  0.2430 for the pair's own labels, and combining both reaches 0.2049. This was
  being thrown away.

## Rate model

`logit p = μ + θ(subject) + b(benchmark) + δ(item)`.

θ is an empirical-Bayes ability per canonical model name, shrunk over the pair
matrix. b for a *new* benchmark is a ridge meta-regression over the task-text
statistics of its own items (15 features: length, script mix, code/table/option
structure, digits, questions), fitted across training benchmarks. δ is the
existing CatBoost task-structure model, refit as a within-pair term around the
pair prior. Evidence is applied as a log-odds shift so within-pair ordering is
preserved.

Five benchmark-level folds; every fitted object for a fold comes from the other
benchmarks. 1,916 pairs scored in total, higher-is-better cells only. One fold
at a time is written to `folds.jsonl`, so an interrupted run resumes instead of
starting over.

| Arm | Brier ALC | sd over folds | budget 0 | budget 1 | budget 31 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Constant 0.5 | 0.2500 | 0.0000 | 0.2500 | 0.2500 | 0.2500 |
| Acquired-label mean | 0.2368 | 0.0657 | 0.2500 | 0.3129 | 0.1855 |
| Fixed beta prior, strength 5 | 0.2112 | 0.0385 | 0.2500 | 0.2327 | 0.1842 |
| Same-benchmark labels plus own | **0.2004** | 0.0521 | 0.2500 | **0.1981** | 0.1930 |
| Ability only | 0.2012 | 0.0513 | 0.2526 | 0.2002 | 0.1929 |
| Ability plus text difficulty | 0.2061 | 0.0550 | 0.2902 | 0.2025 | 0.1929 |
| Hierarchical rate (model) | 0.2036 | 0.0584 | 0.2544 | 0.2026 | 0.1946 |

| Fold | Held benchmarks | Training cells | Pairs | Named ability | Model | Same-benchmark plus own |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 10 | 1,450 | 486 | 28 | 0.2575 | 0.2479 |
| 1 | 7 | 916 | 1,000 | 47 | 0.1891 | 0.1883 |
| 2 | 5 | 1,841 | 95 | 34 | 0.2418 | 0.2363 |
| 3 | 15 | 1,727 | 209 | 63 | 0.2334 | 0.2252 |
| 4 | 4 | 1,810 | 126 | 88 | 0.0960 | 0.1042 |

Gate verdict: **`no_gain_over_fixed_controls`** (`runs/competition_analysis/rate_model_v1/summary.json`).
No submission package was built; the packager refuses a run whose gate did not
clear. Every fold improved on the revealed-label mean (five of five); the model
beat the strongest fixed control in one fold of five, by 0.0082, and lost the
other four by 0.0008 to 0.0096.

Three things the numbers say, in order of how much they matter.

- **The gain is in the evidence, not in the model.** The best arm on the board is
  model-free: shrinking the same-benchmark revealed labels plus the pair's own
  labels, 0.2004. The full ability × difficulty × item model reaches 0.2036. The
  structured model contributes nothing that smoothing does not already give.
- **Cold start is where the model is supposed to help, and it is the one place it
  loses.** At budget 0 the model scores 0.2544 against the constant's 0.2500. The
  text-difficulty block alone costs 0.0402 there (0.2902), and the CatBoost item
  term is what pulls it back. The ridge predicts an unseen benchmark's difficulty
  from 15 task-text counts with a residual of roughly 1.6–1.9 logits; the level
  it adds is honest noise, and budget 0 carries 0.1 of the metric, so this is not
  recoverable from evidence later in the curve.
- **A named-ability term can only fire for a minority of pairs.** 260 of 1,916
  pairs (13.6%) have an ability estimate for the target's canonical name. Fold 4,
  where 88 of 126 pairs did (70%), is the fold the model wins. Fold 1, with 47 of
  1,000 (4.7%), is a tie with the model-free arm. That is a coverage ceiling in
  the corpus, not a fitting mistake — 194 named models exist, 45 appear in two or
  more admitted benchmarks.

Cost: 85–132 s per fold on eight CPU threads, about ten minutes for the whole
five-fold run. No GPU job was used or needed; 0.5 M rows × 300 trees of tabular
CatBoost is not a GPU-shaped workload, and a Modal round trip would cost more
than the run itself.

What this rules in for a next attempt, without claiming any of it works yet:
calibrate or drop the text-difficulty block and let budget 0 fall back to 0.5
where ability is unknown; pool ability over base model families rather than exact
name strings, to raise coverage from 13.6%; and treat `pooled_plus_own` as the
arm a model has to beat, since it is the current best.

## Reproduce

```bash
PYTHONPATH=src .venv-pipeline/bin/python scripts/build_competition_pairs.py \
  --config configs/pipelines/competition_pairs_v1.json \
  --out runs/competition_pipeline/pair_matrix_v1

PYTHONPATH=src .venv-pipeline/bin/python scripts/run_competition_controls.py \
  --config configs/pipelines/competition_controls_v1.json \
  --out runs/competition_analysis/NEW_CONTROLS_RUN

PYTHONPATH=src .venv-catboost/bin/python scripts/run_rate_model.py \
  --config configs/pipelines/rate_model_v1.json \
  --out runs/competition_analysis/NEW_RATE_RUN

PYTHONPATH=src .venv-pipeline/bin/python scripts/build_rate_model_submission.py \
  --run runs/competition_analysis/NEW_RATE_RUN --out runs/competition_submission/NEW
```

The pair-matrix builder refuses an existing output directory and writes its own
hashed manifest; the submission builder refuses any run whose gate did not clear.
