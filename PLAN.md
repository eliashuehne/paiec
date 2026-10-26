# Plan: CatBoost for system–task success prediction

Status: proposed, 2026-10-24. This branch starts from the completed Laya work.
The primary implementation on this branch is CatBoost with structured system/task
features and revealed-outcome adaptation. Logistic regression is a comparison
baseline and can be developed separately against the shared contract.
Writing this plan does not launch experiments. Laya remains on hold; preserve
existing data versions, checkpoints, frozen roles and historical results.

## Objective

Predict the probability of the recorded binary outcome for a target system
(model plus harness/configuration) on a supplied task. Use permitted revealed
outcomes to improve predictions at positive label budgets. The predictor must
handle unfamiliar tasks, benchmark families and system configurations.

The public tables contain heterogeneous events, not exclusively correctness.
Preserve native labels and review their meanings; do not silently invert labels
or assume every outcome of 1 means a correct answer.

Primary metric: competition Brier ALC, lower is better, equally weighted across
subject–benchmark pairs:

`0.1 B0 + 0.2 B1 + 0.2 B3 + 0.2 B7 + 0.2 B15 + 0.1 B31`.

Positive budgets contribute 90% of this score. A predictor that ignores revealed
outcomes must therefore be compared with simple adaptive controls.

## Starting point

| Asset / result | Meaning for this plan |
| --- | --- |
| Dataset-only panel: `runs/competition_pipeline/protocol_panel_v2/` | 34,718 training observations, 16,909 distinct full task inputs, seven training families and 170 visible scopes |
| Enrichment: `runs/shared_data/system_enrichment_v2/` | Full catalog of 46 model/base/family profiles, eight harness profiles and 189 configuration records, plus supplemental research and provenance |
| Latest Laya pilot | Development Brier ALC 0.282439 versus constant 0.25; enrichment's contribution was not isolated |
| Existing lexical ridge controls | Failed transfer/quality checks; the historical 467-column representation is a control, not the required representation |
| Fixed strength-5 evidence control | Saved original-development ALC 0.239531; retain as a matched comparator, not an untouched test result |
| Training-selected strength-0.5 policy | Failed original development at 0.264185; do not promote its candidate package |

Scopes are configurations, not independent model identities. The response graph
is sparse across domains. The current training panel samples one observation
per original system/item; it does not retain every source trial. Keep that
sampling choice explicit, and preserve all trials in the scoring protocol.

## 1. Agree on the shared contract and source eligibility

- Freeze source revisions/hashes, joins, task groups, family assignments,
  acquisition/query splits, seeds and scoring weights before comparing models.
- Predictor inputs remain the published eight subject and four item fields,
  plus explicitly permitted revealed examples. Missing supplied values remain
  empty strings at that boundary. Private identities, reference answers,
  benchmark metadata and query outcomes are excluded.
- Anonymous benchmark IDs scope evidence; they are not learned input features
  or a way to recover hidden benchmark identities.
- Keep a dataset-only competition route. External model/harness enrichment
  remains research-only until its use is established as permitted under the
  inventory-source rules. Record eligibility per source/feature; do not assume
  that earlier collection or a successful pipeline established eligibility.
- Keep documented capabilities separate from settings enabled in a recorded
  run. Preserve unknowns, source qualifications, identity confidence and
  exact-model versus base/family matches.
- Maintain a small feature catalog: definition, entity, units, source,
  prediction-time availability, missing-value policy, code location and status.
  Maintain an experiment log linking changes to configurations and results.

Deliverable: a versioned data/feature contract and common evaluation recipe.
No new collection or spreadsheet service is required to begin.

## 2. Build the first CatBoost predictor

Start with a CPU CatBoost binary classifier using structured numeric and
categorical features. This is the first modeling deliverable; it does not depend
on completing a separate logistic-regression implementation.

Add feature groups incrementally:

1. Supplied system/run fields, parseable recorded dates/settings and missingness.
2. Task length, script/language indicators, code/table/option structure and
   supplied item attributes. Record how each estimate is obtained.
3. Eligible model/harness facts and identity/source-confidence indicators.
   Keep externally researched facts in a separate research arm until permitted.

Represent missing categories consistently. CatBoost can learn system–task
interactions; inspect whether they transfer rather than memorizing identities.
Predeclare a small search over depth, learning rate and regularization, with
selection and early stopping restricted to inner training-family validation.
Use binary log loss for fitting and competition Brier/ALC for model selection.
Log sample weights; do not automatically rebalance classes and distort the
probability target without an explicit correction and evaluation.

Evaluate on identical examples, folds and acquisition trajectories against
constant 0.5, acquired-label mean and fixed strength-5 smoothing. Include the
shared logistic-regression baseline when available, and preserve the rejected
ridge results as historical controls. Do not reuse the 467-column matrix as a
mandatory CatBoost input representation.

Deliverable: out-of-family CatBoost predictions, per-budget scores, feature-group
ablations and missingness/coverage diagnostics. Add an unfamiliar-system stress
test using training-role data only. Feature importance is not causal evidence.

## 3. Add revealed-outcome adaptation to CatBoost

Evaluate adaptation alongside the controls, starting with a regularized
per-system/benchmark log-odds offset around the base predictions. The existing
offset implementation is a starting point; it did not rescue the failed ridge
model and is not a newly demonstrated improvement.

Support labels should update the estimate relative to how difficult the base
model predicted those support tasks to be. Preserve query-level differences
instead of replacing every prediction with one group-wide success rate.

Fit adaptation settings on training-only support/query episodes. Match the
published nested budgets 0/1/3/7/15/31 and acquisition exclusion. Query outcomes
are scoring targets only. Budget zero must reproduce the base predictor.
Check that unrelated evidence, query duplicates and call order cannot affect
predictions. Treat legitimately repeated observations according to the protocol.

Only after this works, test similarity-weighted support or active acquisition
against random acquisition. Optimize for all budgets, especially one and three,
rather than only performance after 31 labels.

## 4. Improve task representation only if the structured model warrants it

Compare task TF–IDF and a pinned frozen multilingual Sentence Transformers
encoder as optional additions to CatBoost. Fit vocabulary or dimensionality
reduction inside each training fold. Cache embeddings by full input hash,
checkpoint revision and encoding configuration; audit long inputs and declare
chunking/aggregation explicitly. Semantic similarity is not task difficulty.

Keep this extension behind the first structured-feature and adaptation results.
No encoder fine-tuning, large AutoML search, EmbedLLM or IrtNet implementation is
in the initial scope. Any model download or GPU work needs a separate execution
decision.

## Evaluation and selection

- Develop with leave-one-training-family-out outer folds. Exclude overlapping
  normalized query tasks from fitting and keep repeated trials together.
  Review near duplicates separately; do not claim normalization catches them all.
- Fit preprocessing, target encodings, feature selection, calibration and all
  hyperparameters using only the fitting side, with inner folds where needed.
  A calibration stage uses out-of-fold training predictions, never outer-query
  outcomes. Preserve the reserved calibration role.
- Match cohorts, acquisition trajectories and seeds across candidates; use a
  predeclared small set of repeated trajectories. Report every budget, equally
  weighted pair ALC, per-family variation, within-pair AUROC and calibration.
  Account for dependence between shared tasks/scopes when summarizing uncertainty.
- The original development set has been inspected repeatedly. Use it only for
  a separately agreed frozen-candidate assessment, not iterative tuning or a
  claim of pristine test performance. Calibration, confirmation and supplemental
  development outcomes remain unused without a separate decision.
- A learned base must improve on constant 0.25 in matched cold-start validation;
  an adaptive candidate must improve on the matched fixed smoothing control.
  Prefer a simple fallback when gains are inconclusive or concentrated in one
  family. Do not promote a model because training or packaging checks pass.
- Freeze the final feature set and recipe before any reserved assessment.
  Qualify the exact submission package with the starting-kit checker and an
  isolated runtime; local checks do not establish hosted compatibility.

## Team workflow and later research

This branch owns CatBoost and its adaptation experiments. Shared feature and
evaluation work, and a separately developed logistic-regression baseline, should
use the same data contract and scorer; avoid duplicating those implementations.
Model-specific preprocessing stays with its model. Use Python modules/scripts
in Git and consistent MLflow records. Reuse existing ZenML workflows where
helpful; adopting ZenML is not a prerequisite for contributing a feature.

Ability/difficulty models such as EmbedLLM or IrtNet remain later research
options, not tasks on this branch. Revisit them only after assessing the CatBoost
results and limitations.

## References

- [Competition interface, source rules and metric](https://aims.stanford.edu/competition)
- [Current results](docs/current-status.md), [outcome semantics](docs/outcome-semantics.md)
- [Dataset-only preparation](docs/competition-pipeline.md), [protocol and evidence results](docs/protocol-panel-and-policy.md)
- [Rejected conditional control](docs/conditional-control.md), [shared enrichment](docs/shared-data-start.md)
- [scikit-learn pipelines](https://scikit-learn.org/stable/modules/compose)
- [CatBoost](https://github.com/catboost/catboost), [Sentence Transformers](https://www.sbert.net/)
- [EmbedLLM](https://github.com/richardzhuang0412/EmbedLLM), [IrtNet](https://github.com/JianhaoChen-nju/IrtNet)
