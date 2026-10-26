# CatBoost base predictor: structured features, whole-family and unseen-benchmark validation

2026-10-24. This implements the CPU CatBoost base predictor from
[PLAN.md](../PLAN.md) on the complete protocol panel, then tests what its
whole-family result actually means.

Summary of the four measurements:

| Measurement | Result |
| --- | --- |
| All declared feature groups, whole-family holdout | **0.2639** cold-start Brier, worse than the 0.25 constant |
| Task-structure-only recipe, whole-family holdout | **0.2393**, first score below the constant |
| Same model, unseen public benchmarks with seen model names | **0.2528**, no transfer beyond the constant |
| Additive two-block logistic, same folds | **0.3325**, severe extrapolation on unseen systems |
| Documented profile facts, 20 scrambled permutations | Level does not improve; ordering gain is inside the null except in the cross-fold-averaged estimand |

The panel result does not survive a genuinely new benchmark, and the additive
model fails outright. No final model, calibration, submission package, GPU job
or Laya inference was produced.

## What is implemented

| Piece | Responsibility |
| --- | --- |
| `src/paiec/catboost_features.py` | 51 declared columns from the twelve supplied fields, plus the declared profile group, with a feature catalog and ablation groups |
| `src/paiec/catboost_training.py` | CPU log-loss fitting, equal-scope sample weights, predeclared grid, inner training-family selection |
| `src/paiec/catboost_analysis.py` | Leave-one-training-family-out folds, ablations, fixed controls, diagnostics and reports |
| `src/paiec/catboost_gate.py` | Fail-closed release assessment for any declared base model |
| `src/paiec/budget_evaluation.py` | Shared full-protocol episode validation and nested-budget scoring |
| `src/paiec/catboost_uncertainty.py` | Paired scope bootstrap and family jackknife for ablation margins |
| `src/paiec/catboost_unseen_benchmark.py` | Competition-shaped evaluation: new benchmark, model name already seen |
| `src/paiec/system_profile_features.py` | Declared profile columns from documented catalog facts, with barred-evidence and missingness rules |
| `src/paiec/catboost_permutation.py` | Permutation distribution for the scrambled control, path-only versus residual gain, paired per-scope check |
| `src/paiec/additive_logit.py` | Additive two-block logistic model and its panel/unseen evaluation |
| `scripts/run_catboost_base.py` | Tracked CLI with fresh-output enforcement and optional MLflow record |
| `scripts/analyze_catboost_uncertainty.py` | Interval analysis over a finished run's saved predictions |
| `scripts/run_catboost_unseen_benchmark.py` | Competition-shaped evaluation runner |
| `scripts/run_additive_logit.py` | Additive model runner |
| `scripts/run_catboost_permutations.py` | Resumable permutation runner; `--aggregate-only` re-reads verified replicates and refits nothing |
| `configs/pipelines/catboost_base_v1.json`, `..._v2.json`, `..._v3.json` | Predeclared panel recipes |
| `configs/pipelines/catboost_unseen_benchmark_v1.json`, `configs/pipelines/additive_logit_v1.json` | Predeclared follow-up recipes |
| `configs/pipelines/catboost_permutation_control_v1.json` | Declared permutation study: seeds, reused replicate and reference arms |

The encoder is dataset-only. It reads the eight subject and four item strings,
and nothing else: no model cards, research profiles, benchmark names, reference
answers, grading rules or outcomes. `benchmark_id` is never a feature. Task text
is measured, never tokenized into a learned vocabulary, embedded, annotated with
skills/difficulty, or clipped. All 16,909 retained task inputs are read in full,
including 162,121-character texts.

Derived values stay transparent:

- A single plain name label (`8B`) is read as a nominal size class. `8x7B` is
  flagged as an expert label and is **not** multiplied into a fabricated total or
  active count. Several plain labels are ambiguous and stay unknown.
- Missing supplied values become an explicit `__missing__` category or a
  presence flag; no value is inferred from identity, defaults or source tables.
- `subject_features_extra` contributes only its setting count and character
  count. Model paths and setting values are not parsed into features.

Fitting gives every visible system/benchmark scope equal total sample weight and
does not rebalance classes. Depth, learning rate and L2 are chosen from a
predeclared six-configuration grid by mean inner training-family log loss, with
early stopping on the inner held family. The configuration, its iteration count
and every stopping decision come from training-side families only.

## Recipe versions

`catboost_base_v1.json` set `min_iterations: 50`. Early stopping chose fewer than
50 trees in every inner fold, so the floor bound **all seven** outer folds: the
iteration count was imposed rather than selected, and the cold-start Brier was
0.289335. That run is preserved as evidence; it is not a reported result.

`catboost_base_v2.json` changes only `min_iterations` to 1, which lets inner
validation shrink the model toward the prior when that is what it prefers. The
selected models use 4-20 trees.

`catboost_base_v3.json` declares `feature_groups: ["task_structure"]`. A recipe
that declares groups fits exactly those columns and sets `ablations_run: false`;
re-running the default ablation suite relative to a reduced base would not
describe the same experiment. The group choice came from the v2 ablation table,
so its outer score is optimistic relative to a blind choice, and it is one of
several margins examined. It is reported as measured, not as a blind estimate.

## Panel result, whole-family holdout

**Cold start, equal-scope Brier over all 34,718 held rows / 170 scopes** (lower
is better):

| Predictor | v2, all groups | v3, task structure only |
| --- | ---: | ---: |
| Constant 0.5 | 0.250000 | 0.250000 |
| Training base rate, per held fold | 0.417488 | 0.417488 |
| CatBoost | 0.263850 | **0.239313** |
| minus system attributes | 0.247501 | — |
| minus system missingness flags | 0.252485 | — |
| minus task structure | 0.285417 | — |
| minus supplied item extras | 0.249860 | — |
| plus system identity (ablation) | 0.270553 | — |

Ranking and calibration of v3 on the held rows: mean within-scope AUROC
**0.5041**, ECE 0.1368, log loss 0.7181. The improvement is a level effect, not
better ordering.

**Brier ALC over all 170 eligible scopes** (identical episodes and query rows for
every row; base predictions repeat across budgets):

| Predictor / update | v2 | v3 |
| --- | ---: | ---: |
| Constant 0.5 | 0.250000 | 0.250000 |
| Acquired-label mean | 0.149105 | 0.149105 |
| Fixed beta mean, strength 5 | 0.160365 | 0.160365 |
| No update | 0.263626 | 0.238951 |
| Beta mean | 0.165144 | 0.156123 |
| Log-odds offset | 0.161884 | **0.154659** |

No scope was excluded: all 170 retained scopes cover all six budgets. For v3 the
gate therefore reports the base as clearing the cold-start check and the two
adaptation variants as clearing both predictive checks. That is a statement about
the panel protocol only; the unseen-benchmark evaluation below is the reason not
to read it as readiness.

### Where the v3 gain comes from

| Held family | Held rate | Trees | v2 | v3 |
| --- | ---: | ---: | ---: | ---: |
| abstract_reason | 0.46 | 1 | 0.2565 | 0.2453 |
| benger | 0.82 | 18 | 0.3538 | 0.4060 |
| bird | 0.09 | 22 | 0.2462 | 0.2010 |
| cellverse | 0.43 | 8 | 0.2445 | 0.2514 |
| clasheval | 0.31 | 4 | 0.2753 | 0.2752 |
| fewshot_ttt_bbh | 0.53 | 4 | 0.2498 | 0.2510 |
| gsm8k_math | 0.66 | 2 | 0.2597 | 0.2501 |

`bird` owns 120 of 170 scopes and its observed rate is 0.09. Task text alone
pushes predictions below 0.5 there, which pays on an extreme base rate, while
`benger` (rate 0.82) degrades by 0.05. On equal-scope weighting the first effect
dominates by construction. This is why the unseen-benchmark evaluation exists.

## Ablation margins with intervals

`scripts/analyze_catboost_uncertainty.py --out runs/competition_analysis/catboost_base_family_v2_uncertainty`
recomputes each v2 margin as a paired per-scope difference and resamples whole
scopes (2,000 draws, seed 20261024, 95% percentile interval). Scopes share
systems and questions, so this interval is a lower bound on uncertainty. A
seven-family leave-one-out jackknife gives the coarse family view: a margin is
called distinguishable only when the interval excludes zero **and** every
jackknife value keeps its sign.

| Ablation (difference = ablation minus full) | Difference | 95% interval | Jackknife | Reading |
| --- | ---: | --- | --- | --- |
| minus system attributes | −0.01635 | [−0.01850, −0.01421] | all negative | distinguishable |
| minus item extras | −0.01399 | [−0.01577, −0.01237] | all negative | distinguishable |
| minus system missingness | −0.01136 | [−0.01340, −0.00930] | all negative | distinguishable |
| minus task structure | +0.02157 | [+0.01655, +0.02621] | `bird` −0.0171 | not distinguishable |
| plus system identity | +0.00670 | [+0.00264, +0.01123] | `benger` −0.0033 | not distinguishable |

Negative means the ablation was better than the full model. Removing the supplied
system block, its missingness flags, or the item extras each improved the score
on 146-156 of 170 scopes with the same sign in all seven family jackknives.
Removing task structure hurts on 131 scopes overall but helps strongly inside
`bird`, so the family view does not agree; the same is true, in reverse, for the
identity column.

The three distinguishable margins are what v3 combines. The v3 result is
therefore consistent with this analysis, and is also a selection made after
seeing these margins, which the interval analysis cannot correct.

## Competition-shaped evaluation: new benchmark, known model name

The panel has 170 family-system pairs and **none** span more than one benchmark.
Holding out a training family therefore also holds out every one of its systems,
while the competition is a benchmark-level cold start whose subject coverage may
differ between pools ("The evaluation is a benchmark-level cold start: held-out
items are drawn from benchmarks assigned to the private test pool").

`configs/pipelines/catboost_unseen_benchmark_v1.json` scores the v3 fold models on
unused public benchmarks. Selection is a predeclared rule and never reads an
outcome: unused directory, no non-training frozen role, text-only,
`higher_is_better`, binary item-level, at least one subject sharing a supplied
model name with the panel, and **zero** shared task content with the panel. The
last condition matters: nine unused-looking directories share 15-100% of their
tasks with the panel (`cellverse`, `bird_sql`, `benger`, `fewshot_ttt_bbh`,
`critic_discernment_game`, `abstract_reason`, `clasheval`, `adaptivestep`,
`felm`), and admitting them would have scored rows from benchmarks the model was
already trained on. An earlier draft with a 5% tolerance did exactly that and was
discarded before being reported.

Sixteen benchmarks were admitted: `aegis`, `afrimedqa`, `agi_elo`, `algotune`,
`babilong`, `beavertails`, `bridging_gap`, `chi_bench`, `cruxeval`, `editbench`,
`egoschema`, `ehrflowbench`, `engdesign`, `faithcot`, `financebench`, `helm_afr`.
One hashed response per subject-item, capped at 100 per subject, sampled 37,241
rows across 402 subjects. After fold eligibility, 8,506 rows from 92 subjects and
15 benchmarks were actually scored; `bridging_gap` matched no fold's training
systems.

A row is eligible for a fold only when its supplied model name, or its full
supplied identity, is in that fold's training systems. Identity is eligibility
metadata only and never a feature. The panel score is recomputed in the same run
and must reproduce the recorded value exactly (it does: 0.239313).

| Measure | Value |
| --- | ---: |
| Equal-scope Brier, mean over the seven folds | **0.252828** |
| Pooled row-level equal-scope Brier | 0.250267 |
| Pooled row Brier | 0.250418 |
| Mean within-scope AUROC | 0.5077 |
| ECE | 0.0540 |
| Constant 0.5 | 0.250000 |
| Mean panel base rate applied to these rows | 0.323002 |
| Rows with a fully identical supplied subject | 0 |

The model is calibrated and uninformative on unseen benchmarks: predictions sit
near the right level and carry almost no ordering. **No scored row has both an
unseen benchmark and a fully identical supplied subject**, so the competition's
"shared subject" case cannot be tested with this evaluation set at all; only the
weaker "same model name, different harness" case exists.

Per benchmark, equal-scope Brier ranges from 0.208 (`beavertails`, one scope,
rate 0.99) to 0.265 (`algotune`, rate 0.83). The per-scope observed-rate oracle,
which uses the evaluation labels and is never a predictor, ranges from 0.010 to
0.242, so the achievable headroom is real but not reachable from supplied
attributes and task text alone.

Scoring these rows makes them development evidence. They are no longer pristine,
and a recipe should not be tuned repeatedly against them.

## Additive two-block model

`configs/pipelines/additive_logit_v1.json` fits `logit p = b0 + f_system + f_task`
with no cross-block term: weighted L2 logistic regression, Newton/IRLS, one-hot
categoricals from the fitting fold only, standardized numerics, and block
penalties from a predeclared four-configuration grid chosen by inner families.
There is no per-system offset, so an unseen system's contribution can only come
from its supplied attributes.

| Measure | Additive | CatBoost v3 |
| --- | ---: | ---: |
| Cold-start equal-scope Brier | 0.3325 | 0.2393 |
| Mean within-scope AUROC | 0.554 | 0.504 |
| ECE | 0.317 | 0.137 |
| Brier ALC, no update | 0.3332 | 0.2390 |
| Brier ALC, beta mean | 0.1892 | 0.1561 |
| Brier ALC, log-odds offset | 0.2248 | 0.1547 |
| Unseen-benchmark equal-scope Brier | 0.3302 | 0.2528 |

The failure is extrapolation, not noise. Held-family mean predictions leave the
plausible range: 0.032 for `benger` (observed 0.82), 0.826 for `cellverse` (0.43),
0.818 for `abstract_reason` (0.46). A linear block over one-hot harness/provider
and continuous size/date features extrapolates for unseen systems, whereas trees
saturate at seen values. Its ranking is slightly better than CatBoost's, but the
level error dominates.

The fail-closed gate rejects every additive variant, correctly. This is
consistent with the ablation finding that removing the supplied system block
improves the tree model: supplied attributes do not carry system strength across
families, and forcing them into an unbounded linear block makes it worse.

## Enriched profile features (research-only)

`configs/pipelines/catboost_base_v4.json` adds the declared `system_profile`
group: 17 columns computed by `src/paiec/system_profile_features.py` from the
documented facts in `research/system_profiles/collection_v2`, attached through
the same fail-closed alias policy the Laya enrichment uses. Columns are size,
active share, layers, hidden size, heads, key/value-head share, expert count,
family, model type, parameterization, modalities, harness workflow, and explicit
found/missingness flags. Barred sources are treated as unknown and unresolved
systems get `__missing__`, never a guess.

Coverage is the reason this was worth measuring. On the panel, 169 of 176
systems resolve to a documented model profile and 169 to a harness profile, but
only 104 have at least one documented numeric architecture fact, and 52.9% of
rows have none - the closed API models dominate the row counts. On the unseen
set it is thinner: 42 of 92 subjects resolve to a model profile and **none** to a
harness profile, so the evaluation only exercises model-side facts.

| Arm | Cold Brier | Within-scope AUROC | ECE | ALC log-odds | Unseen Brier | Unseen AUROC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v3 task structure only | **0.2393** | 0.5041 | 0.1368 | **0.1547** | 0.2528 | 0.5077 |
| v4 + documented profiles | 0.2543 | **0.5364** | 0.1444 | 0.1589 | **0.2510** | **0.5530** |
| v5 profiles scrambled | 0.2603 | 0.5157 | 0.1831 | 0.1609 | 0.2523 | 0.5300 |

`configs/pipelines/catboost_base_v5.json` is the control: the same 17 columns and
the same per-system constancy, with the system-to-fact link permuted. It exists
to separate capability information from the mere presence of 17 extra columns in
the boosting path.

**The enriched block does not earn its place.** Cold Brier gets worse on the
panel (0.2393 to 0.2543) and the ALC follows (0.1547 to 0.1589); the unseen
improvement of 0.0018 sits inside the control's own 0.2523. The eye-catching
ordering gain is not attributable: this one scramble keeps 36% of the panel
AUROC gain and 49% of the pooled unseen gain, and per fold the unseen delta is
+0.0065 against the control's -0.0016. The gain is also larger where **no**
profile resolves (+0.058) than where one does (+0.030), which a capability
explanation does not predict. One draw cannot show whether those fractions are
typical, which is what the permutation study below measures; that draw turned
out to overstate the panel side substantially.

The pooled unseen AUROC gain (+0.045, scope bootstrap 95% interval
[+0.019, +0.072], 59 of 91 scopes improved) is real as a number and still not
attributable: averaging seven fold models raises AUROC by itself, and the control
shows how much of it the columns alone produce. An enriched additive block would
be constant within a pair and could not move within-scope ordering at all; a
tree can, through profile-by-task interactions, which is why this had to be fit
rather than argued.

### Permutation distribution (20 scrambles)

`configs/pipelines/catboost_permutation_control_v1.json` repeats the control
nineteen more times, so one draw becomes a distribution: seeds 2-20 fitted fresh
and the existing seed-1 run folded in. Each replicate runs the v4 recipe and the
v2 evaluation verbatim with only `profile_permutation_seed` changed.
`scripts/run_catboost_permutations.py` refuses any replicate whose resolved
recipe is not exactly the declared permutation for that seed, re-verifies every
artifact hash, and asserts that all 22 arms - observed, task-only and the 20
permutations - scored the identical 8,506 evaluation rows. The primary metric was
declared before any replicate was fitted; the paired scope check was declared
before fitting finished.

| Metric | Observed | Task-only | Permuted mean (sd) | Permuted range | Path-only | Residual | p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Panel cold Brier | 0.2543 | 0.2393 | 0.2483 (0.0090) | 0.2308-0.2610 | -0.0089 | -0.0061 | 0.71 |
| Panel within-scope AUROC | 0.5364 | 0.5041 | 0.5288 (0.0165) | 0.4821-0.5511 | +0.0247 | +0.0076 | 0.38 |
| Panel ALC, log-odds offset | 0.1589 | 0.1547 | 0.1569 (0.0029) | 0.1515-0.1609 | -0.0023 | -0.0019 | 0.71 |
| Unseen Brier, equal scope | 0.2510 | 0.2528 | 0.2516 (0.0008) | 0.2499-0.2527 | +0.0013 | +0.0006 | 0.24 |
| Unseen AUROC, one fold model per row | **0.5162** | 0.5097 | 0.5132 (0.0036) | 0.5081-0.5224 | +0.0036 | +0.0030 | 0.19 |
| Unseen AUROC, fold-averaged predictions | 0.5530 | 0.5077 | 0.5281 (0.0056) | 0.5173-0.5370 | +0.0204 | +0.0249 | 0.048 |

Path-only is the permuted mean minus the task-only arm in the better direction;
residual is the observed value minus the permuted mean. `p` counts permutations
at least as good as the observed arm with the usual +1 correction, so with 20
draws its floor is 1/21 = 0.048: a p at the floor means *above every
permutation*, not a precise probability.

**The panel ordering gain is mostly the extra columns, not the facts.** Of the
+0.0323 panel AUROC gain over task-only, +0.0247 is reproduced by scrambled
facts; the residual +0.0076 is reached by 7 of 20 permutations (p 0.38). The
single-draw estimate of 36% was unrepresentative - averaging the draws puts the
path share near 76%. The level is worse than the single draw suggested too: every
profile model, real or scrambled, costs about 0.009 on panel cold Brier relative
to task-only, and the real facts cost another 0.006, so the observed arm is worse
than 14 of the 20 permutations.

**The primary metric is inside the null.** Unseen per-fold AUROC gains +0.0030
over the permuted mean, and 3 of 20 permutations match or beat it, so the
predeclared verdict is `residual_ordering_gain_within_permuted_control_range`.
The unseen Brier difference is smaller than the null's own spread (+0.0006 against
sd 0.0008): enrichment does not make the competition-shaped predictions better
calibrated.

**One estimand does survive, and it is not the deployable one.** When each row
is scored by the average of its qualifying fold models instead of by a single
fold model, the observed ordering exceeds every one of the 20 permutations
(+0.0249, p at the floor); the paired check agrees, with a mean +0.0249 per scope
(scope bootstrap 95% interval +0.0116 to +0.0388), 57 of 91 scopes above the
permuted mean, and 9 scopes above every permutation. The task-only arm is worse
than the permuted mean on the same scopes (-0.0204, interval -0.0404 to
-0.0046), which is the signature of a column-count effect rather than of carried
information. Averaging up to seven fold models is a different estimator from the
one a submission would run; the predeclared primary uses a single model per row
for that reason, and it is the one that fails. The honest summary is that
documented facts may help an ensemble of family-held-out trees order tasks, and
that this is not the same claim as the facts carrying capability information a
single model can use.

Two properties of the null are worth keeping in mind. A permutation preserves
every column's marginal distribution and every within-scope constancy, and only
removes the system-to-fact link, so a system can still receive an identical
profile record by chance when profiles repeat - which makes the null
conservative. And the primary metric's p-value is limited by 20 draws; a
sharper answer needs more permutations, not a different test.

Artifacts: `runs/competition_analysis/catboost_permutation_control_v1/`
(`permutation_distribution.json`, `scope_ordering.jsonl`, per-seed panel and
evaluation folders, hashed manifest). MLflow runs
`6150767129754fec9aa4ec8c92df9b9d` (fitting) and
`5413598dbe8e45e0bae5a9aff84f65b0` (re-aggregation), both FINISHED with the same
`comparison` digest; re-aggregating from the saved replicates reproduced every
reported metric exactly.

The research policy is enforced, not advisory: `source_policy` becomes
`provided_plus_research_profiles`, and `assess_research_only_catboost_base`
records the predictive checks but marks every candidate
`external_research_profiles_not_competition_eligible`. No enriched variant is a
release candidate.

## How to read it

- The whole-family panel gate is necessary but not sufficient. The task-only
  model clears it at 0.2393 largely because `bird` is 70% of scopes and has a 9%
  success rate; on unseen benchmarks the same model scores 0.2528.
- The binding difficulty is a new benchmark's overall difficulty level, which
  must come from task text alone out of family. Task-text structure carries some
  level information within the panel and essentially none across benchmarks.
- Supplied system attributes hurt the tree model and wreck the linear model.
  There is no cross-benchmark system-strength signal to learn in this panel
  because no system appears twice.
- The identity column is rejected as a feature, as designed; it does not help
  even as a diagnostic.
- All reported values are CPU, training-only, with calibration, confirmation,
  monitor and original-development outcomes untouched.

## Reproduce

```bash
uv venv --python 3.12 .venv-catboost
uv pip install --python .venv-catboost/bin/python -r requirements/catboost.lock -e .

PYTHONPATH=src .venv-catboost/bin/python scripts/run_catboost_base.py \
  --config configs/pipelines/catboost_base_v3.json \
  --out runs/competition_analysis/NEW_PANEL_RUN

PYTHONPATH=src .venv-pipeline/bin/python scripts/analyze_catboost_uncertainty.py \
  --run runs/competition_analysis/catboost_base_family_v2 \
  --out runs/competition_analysis/NEW_UNCERTAINTY_RUN

PYTHONPATH=src .venv-catboost/bin/python scripts/run_catboost_unseen_benchmark.py \
  --config configs/pipelines/catboost_unseen_benchmark_v1.json \
  --out runs/competition_analysis/NEW_UNSEEN_RUN

PYTHONPATH=src .venv-catboost/bin/python scripts/run_additive_logit.py \
  --config configs/pipelines/additive_logit_v1.json \
  --out runs/competition_analysis/NEW_ADDITIVE_RUN

# Research-only profile block and its scrambled control (same commands).
PYTHONPATH=src .venv-catboost/bin/python scripts/run_catboost_base.py \
  --config configs/pipelines/catboost_base_v4.json \
  --out runs/competition_analysis/NEW_PROFILE_RUN

PYTHONPATH=src .venv-catboost/bin/python scripts/run_catboost_unseen_benchmark.py \
  --config configs/pipelines/catboost_unseen_benchmark_v2.json \
  --out runs/competition_analysis/NEW_PROFILE_UNSEEN_RUN

# Permutation distribution for the scrambled control. Resumable: finished
# replicates are hash-verified and reused, and --aggregate-only reports without
# fitting anything.
PYTHONPATH=src .venv-catboost/bin/python scripts/run_catboost_permutations.py \
  --config configs/pipelines/catboost_permutation_control_v1.json \
  --out runs/competition_analysis/NEW_PERMUTATION_STUDY
```

The base CLI refuses an existing output directory, keeps it separate from the
prepared bundle, writes a resolved recipe and hashed artifact manifest, and
records MLflow unless `--no-tracking` is passed. Repeated runs reproduced every
reported score to floating-point precision, and the unseen run verifies its own
panel reproduction before reporting anything.

The shared episode scorer was extracted while building the first experiment.
`scripts/analyze_conditional_predictor.py` still reproduces the documented
conditional-control numbers exactly (cold-start 0.361193; ALC 0.199843 beta
mean, 0.195096 log-odds offset), so that refactor did not change the result.

## Limits and next gates

- This is training-family and public-unused-benchmark validation, not
  hidden-competition or original development performance. No calibration was
  fitted; calibration, confirmation and supplemental development outcomes stay
  unused.
- No final model was fitted on all training families, and no submission package
  was built. Base predictions repeat across budgets and ignore revealed evidence
  except in the explicitly scored adaptation modes.
- The v3 recipe is one of several margins examined in v2. Its score is optimistic
  relative to a blind choice and the interval analysis cannot correct that.
- The competition-shaped set has no exact-subject rows, so it measures only the
  weaker name-level sharing case. It is also now development evidence.
- The next recipe must beat 0.25 on the unseen-benchmark set, not just on the
  panel. The available levers are a representation that separates benchmark-level
  difficulty from task content, or a deliberate, row-matched admission of unused
  benchmarks to give the model cross-benchmark system-strength signal. Neither has
  been run.
- Documented profile facts are now measured and rejected as a standalone block:
  they worsen the cold level, and 20 scrambled permutations reproduce most of
  their panel ordering gain and all of their per-row unseen ordering gain. The
  one surviving effect needs the fold-model average to appear, which is a
  different estimator from a single submission model. They remain research-only
  regardless, since their sources are unreviewed for competition eligibility.
- The permutation test cannot currently reach p below 1/21. Reaching a
  conventional threshold would need at least 40 fresh seeds, about 3 CPU-hours
  at the measured 4-5 minutes per replicate.

## Where this went next

The 170-scope panel that produced every number above holds 170 distinct systems
and no system in two benchmarks, so a held-out scope changes system and benchmark
at once and the two effects cannot be separated. Every result here therefore
measures a level shift, not transfer. [The pair matrix and rate model](competition-pairs.md)
rebuild the unit over all 61 public benchmark directories: 6,252 subject-benchmark
cells, 14.7M labelled responses, a competition-shaped scorer that keeps the
cross-pair evidence this pipeline discarded, and a two-way ability × difficulty
model with benchmark-level folds. The task-structure model built here survives as
that model's within-pair term; the panel itself is retained unchanged as the
historical record of these measurements.
