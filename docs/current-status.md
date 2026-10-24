# Current status and reusable work

Updated 2026-10-24. **Laya is no longer the active modeling focus.** Preserve
its code, inputs, checkpoints and results for reproducibility; do not launch
further Laya training, inference sweeps or GPU jobs without a new decision.

## Completed result

The enriched multilingual pilot completed one epoch: 34,718 training examples,
543 updates, final two encoder blocks plus the decision head trainable.

| Original-development measure | Result |
| --- | ---: |
| Brier ALC, lower is better | **0.282439** |
| Constant 50% prediction baseline | **0.250000** |
| Difference from baseline | **+0.032439**, worse |
| Mean within-pair AUROC | **0.545742** |
| Evaluated responses / system-items / pairs | 963 / 960 / 8 |

Training loss improved, but held-out probability predictions did not beat the
constant baseline. Ranking was weak. This establishes a negative result for
this recipe, not that the collected information is useless. No recorded-only
control was trained, so enrichment's contribution is not isolated.

Input parity, frozen-parameter checks, checkpoint hashes and native reload
passed (maximum probability drift 0.0000275824). One development evaluation
completed; revealed outcomes were ignored, so predictions were shared across
budgets. No calibration was fitted. Calibration/confirmation and supplemental
development outcomes were unused. Both GPU apps are stopped and the completed
run's monitor was removed. See [the experiment record](grouped-enriched-training.md)
for exact provenance and the cleanup-only recovery.

## What other approaches can reuse

| Asset | Reusable purpose |
| --- | --- |
| Dataset-only tables | Supplied system/task fields, stable joins and separate native training labels |
| Collected enrichment handoff | Model/base facts, harness capabilities, actual run settings, source evidence and original-system mappings |
| Historical statistical starter | Optional experiment artifact; not the main sharing recommendation |
| Frozen assignments and grouping helpers | Shared comparisons with whole-family and overlapping-task exclusions |
| Scoring utilities | Trial-preserving, equally weighted pair Brier/ALC and calibration diagnostics |
| ZenML / MLflow | Preparation, explicit execution, integrity manifests and saved experiment evidence |
| Model/harness research tables | Optional sourced enrichment with unknowns and scope qualifiers; research-only |

The main handoff is [collected model/harness information](shared-data-start.md),
not the representation used by a failed experiment. These records let other
approaches choose their own features. Prior matrices, selected training rows
and trained weights are preserved separately as experiment artifacts.

## Protocol for the next approach

Keep the data revision, raw target meanings, split assignments and scored
examples explicit. Develop using training-only family/task-group folds; fit
preprocessing and hyperparameters inside each fitting fold. Agree on weighting
before comparing models: historical recipes used different weighting schemes.

The original development set has already been evaluated repeatedly and is not
a pristine test set. Coordinate further assessments; do not tune directly on
its saved labels. Calibration/confirmation remain sealed. Keep reference
answers, private identities, scorer IDs and outcome-derived statistics out of
predictor features. Any revealed-evidence adaptation needs its own explicit mode.

Data and features may support different model-specific preprocessing. Outside
research facts need a separate eligibility review and must not silently enter
the dataset-only starter. Laya's token windows and checkpoint loaders are
model-specific infrastructure, not a shared preprocessing requirement.

Nothing has been deleted or automatically retrained. Generated data and model
weights remain outside Git; see [data lifecycle](data-lifecycle.md) and
[workflow organization](workflows.md).
