# Is item difficulty shared, or model specific?

2026-10-25. CPU only, public corpus only, no model promoted and no sealed role
opened. This answers one question, because that question decides where item-level
work should go next.

The rate model estimates a pair's overall success rate and reached within-pair
AUROC **0.5027** using fifteen surface counts. That is a statement about the
features we chose, not about the world: it cannot tell us whether item difficulty
is empty or merely unmodelled. The two possibilities imply opposite investments.

- If difficulty is **shared**, a hard question is hard for everyone, so other
  systems' outcomes on it predict this system's outcome — and item descriptions
  become a learnable target.
- If difficulty is **model specific**, one system fails code and another fails
  arithmetic, and no description of a question can rank it for a system. Only
  that system's own revealed outcomes on similar questions would help.

## What was measured

A Rasch fit per benchmark, `logit p = theta(subject) - b(item)`, penalised on both
blocks. Benchmarks of the judged direction only, since pooling a
`lower_is_better` benchmark would invert its labels.

| Measure | Value |
| --- | ---: |
| Benchmarks with item-level data | 37 |
| Items with a split-half estimate | 517,191 |
| Split-half reliability of item facility | **0.670** |
| Benchmarks with a positive estimate | 24 of 24 |
| Held-out transfer AUC, pooled by items | **0.765** |
| Transfer AUC range across benchmarks | 0.587 – 0.965 |
| Benchmarks where transfer beats chance | **24 of 24** |
| Median sd of item difficulty | 3.68 logits |
| Median sd of subject ability | 0.94 logits |
| Median residual Brier with a perfect Rasch fit | 0.0774 |

Two of those numbers carry the verdict. Split the systems on a benchmark into two
disjoint halves; the same questions look hard to both, at r = 0.670. Then rank one
half's items using difficulty learned **only** from the other half, and ask whether
that ordering predicts individual outcomes for the held-out systems: AUC 0.765,
and above chance in every single benchmark measured. Item difficulty belongs to
the item.

The spread is large: median sd of item difficulty is 3.68 logits, against 0.94 for
system ability. Within one benchmark, items run from almost always solved to
almost never solved. A single number per pair throws all of that away.

## What the ordering prize is worth

Oracle ceilings on the same five benchmark folds and draws the scorer uses. Lower
is better; the constant is 0.2500 and the best real arm is 0.2004.

| Knowing… | Brier ALC | What it means |
| --- | ---: | --- |
| Each pair's true rate | 0.1764 | the level prize the evidence rule already chases |
| Each evaluation item's facility, from **other** subjects' revealed items | 0.1723 | what item evidence can reach at current coverage |
| Each evaluation item's own recorded rate | **0.0657** | the ordering prize |

So ordering is worth roughly **0.11 ALC** — six times the 0.017 still available on
the level axis. It is the only large prize left.

## Why revealed labels cannot reach it

Coverage is the wall. An evaluation item is only predictable from item evidence if
some other sampled subject revealed *that same* item, and a budget of 31 responses
reveals about 31 items out of each subject's pool.

Share of evaluation items with any sibling revealed data, per fold:

| Fold | budget 1 | budget 3 | budget 7 | budget 15 | budget 31 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.00 | 0.01 | 0.02 | 0.04 | 0.07 |
| 1 | 0.02 | 0.07 | 0.16 | 0.30 | 0.50 |
| 2 | 0.01 | 0.02 | 0.05 | 0.08 | 0.11 |
| 3 | 0.00 | 0.00 | 0.00 | 0.01 | 0.01 |
| 4 | 0.21 | 0.48 | 0.68 | 0.76 | 0.76 |

Median coverage at the largest budget is about one item in nine, and budgets 1 and
3 together carry 0.4 of the metric. That is why the sibling-facility ceiling
(0.1723) is barely better than the pair-rate ceiling (0.1764): perfect estimation
of facility buys almost nothing when facility is only observable for a tenth of the
items.

**A Rasch model on revealed labels is therefore not worth building.** Its mechanism
is validated and its reach is not there. This is a change of plan, and it comes from
measurement rather than preference.

## What this funds instead

Item difficulty is a stable property of the question, and it is measurable offline
for every training item from the public corpus. That makes it a **learnable target
for item content**, with no coverage limit at prediction time.

The test is cheap and honest: fit facility for training items from one pool of
systems, then predict it for held-out systems' items from text alone, and score
with the same split-half harness. A TF-IDF model answers whether content carries
this signal at all; a frozen sentence encoder is only worth its GPU hour if TF-IDF
clears roughly AUC 0.60 against a 0.765 ceiling. The existing surface-feature
CatBoost item block scored 0.5027 against individual outcomes, which is the
baseline that has to be beaten.

## Reproduce

```bash
PYTHONPATH=src .venv-pipeline/bin/python scripts/analyze_item_axis.py \
  --config configs/pipelines/item_axis_v1.json \
  --out runs/competition_analysis/NEW_ITEM_AXIS_RUN
```

Report: `runs/competition_analysis/item_axis_v1/report.json`. Thresholds for the
verdict are declared in the recipe before the numbers are read, so the
shared-versus-model-specific call is not chosen after the fact.

## Limits

Both ceilings use evaluation outcomes by construction; they bound a score and are
never predictors. Coverage depends on draw size, the number of pairs per benchmark
and the pairing, so a larger draw would raise it — the folds above already sample
the platform's 1,000-pair cap. Reliability covers 37 of the 41 admitted benchmarks;
the four with no item-level cells of the judged direction are absent rather than
silently included. Everything here is the public corpus, not the competition's
held-out pairs.
