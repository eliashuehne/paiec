# Conditional predictor control and rejection result

For the subsequent complete 170-scope comparison and selected evidence-policy
candidate, see [protocol panel and policy](protocol-panel-and-policy.md).
Both conditional control variants still fail the predictive quality checks;
the earlier coverage limitation described below is repaired in that new source.

2026-10-24. This implements an executable, dataset-only statistical control for
the [competition pipeline](competition-pipeline.md). It is not a replacement
Laya checkpoint or a qualified competition release. The first whole-training-
family experiment rejects both fitted variants as default predictors.

## Implemented representation and runtime

The encoder consumes every supplied token in `item_content`, `item_features`
and `interactors`, including Unicode character bigrams. No text is clipped.
It separately hashes the eight supplied system fields. Numeric features cover
input sizes, character proportions, parseable recorded dates, explicitly
supplied nominal capacity text and missingness. Nominal `8B` is a transparent
name-derived proxy; `8x7B` is not multiplied into a fabricated total or active
parameter count. No model cards, aliases, benchmark score tables, reference
answers or external task annotations are read.

The 467-dimensional vector combines task, system and numeric features with
128 system/task interaction features. Those interactions can express different
system strengths on different tasks. They do not establish named abilities,
semantic understanding or causal architectural effects. The additive ablation
zeros exactly the interaction block and otherwise uses identical inputs.

CPU ridge fitting uses equal visible system/benchmark scope weight and fixed
L2=100. Fitted state is JSON, with an explicit feature version, dimensions,
source policy, center, intercept, coefficients and training provenance. Loading
requires a matching artifact SHA-256. Runtime modules use only Python's standard
library; they never fit, download or load native model weights.

`ConditionalPredictor.predict(input, labeled)` returns a finite native float.
Base predictions omit anonymous benchmark IDs from features and cache keys.
Revealed evidence must match the complete visible system and nonempty anonymous
benchmark ID. The exact query input is excluded. Repeated legitimate revealed
trials retain their multiplicity. Only base probabilities are cached; labels
are supplied independently on every call, so budget order cannot leak state.

Three evidence modes are implemented: unchanged base probability, the fixed
beta-mean update with strength five, and a regularized log-odds offset inferred
from the revealed support outcomes and their base probabilities. The offset
preserves relative item difficulty instead of assigning every query the same
probability. Its penalty is fixed at one; it is not a separately trained
evidence correction. Zero labels preserve the exact base probability.

## Validation protocol

```bash
.venv-pipeline/bin/python scripts/analyze_conditional_predictor.py \
  --out runs/competition_analysis/NEW
```

Recipe: `configs/pipelines/conditional_control_v1.json`. All 30,000 training
records receive predictions from a model fitted outside their whole training
family. All occurrences of held query tasks in other fitting families are also
excluded. This sample had zero such cross-family overlaps. Family attribution
is read from hash-pinned scorer provenance, never embedded as a predictor
feature. Original development, monitor, calibration and confirmation outcomes
are unused. Seven folds fit two variants each; fixed controls share the same
acquisition splits and query records.

Cold-start Brier averages all held rows equally across 170 visible scopes.
Full-budget ALC uses the common 50 scopes with at least 80 retained normalized
questions and enough acquisition supports. These denominators differ; the
120 smaller retained scopes cannot establish the full protocol's ALC. The
source retained one chosen trial per original system/item, not all native
trials. No hyperparameters were searched in this initial comparison.

## Measured results

Execution: `runs/competition_analysis/conditional_family_v1/`; MLflow
`398b0aca37f94f6499e68f454c5c6b7c` (FINISHED), **PAIEC Competition Analysis**.

| Base predictor | Cold-start Brier, all 170 scopes |
| --- | ---: |
| Constant 0.5 | 0.250000 |
| Additive text/system ridge | 0.375872 |
| Conditional ridge | 0.362303 |

| Predictor/update | Full-budget Brier ALC, common 50 scopes |
| --- | ---: |
| Constant 0.5 | 0.250000 |
| Acquired-label mean | 0.243859 |
| Fixed smoothed evidence, k=5 | **0.213164** |
| Additive, no update | 0.339623 |
| Additive, beta mean | 0.244551 |
| Additive, log-odds offset | 0.242735 |
| Conditional, no update | 0.336483 |
| Conditional, beta mean | 0.242950 |
| Conditional, log-odds offset | 0.241588 |

The interaction changes predictions, but the fixed control remains better.
The text models' cold-start estimates transfer poorly to held-out families.
Positive-budget adaptation repairs some error without establishing value over
the simple fixed prior. This is a training-family validation study, not a
hidden-competition or original-development result. Do not promote these
checkpoints because the machinery works.

## Release gate and portable qualification

`paiec.conditional_gate.assess_conditional_controls` rejects candidates with
worse cold-start error than the matched constant, no ALC improvement over the
matched smoothed control, or incomplete full-budget scope coverage. It also
requires isolated training-family validation and finite matched-control scores.
These are necessary initial checks; passing would not establish statistical
robustness, unseen-system performance or hosted compatibility.

All six fitted variant/update candidates fail the current checks. The audit is
`runs/competition_analysis/conditional_family_v1_qualification/candidate_assessment.json`.
Future experiment runs include this assessment in their published artifact
manifest; the existing first experiment is preserved unchanged.

```bash
.venv-pipeline/bin/python scripts/qualify_conditional_control.py \
  --prepared runs/competition_pipeline/dataset_only_v1 \
  --analysis runs/competition_analysis/conditional_family_v1 \
  --out runs/competition_analysis/NEW_QUALIFICATION
```

An isolated `-I -S` interpreter loaded all 14 JSON fold models using only the two
portable runtime modules and an empty package initializer. It checked the
shortest, median and longest supplied input for each family/variant: 42 real
fixtures, maximum probability drift **1.11e-16** from saved matrix predictions.
No source dataset or site packages were available to that child process.
This demonstrates portable loading and numerical parity, not a passed quality
gate or hosted Codabench execution. No deployment, paid API or GPU work occurred.

The source/lineage audit verified every training and held-family row-order hash,
30,000 out-of-family predictions, all 14 fitted model artifacts and 20 artifact
hashes logged to MLflow. Evidence:
`runs/competition_analysis/conditional_family_v1_qualification/verification.json`.
Both native environments ran 303 unit tests successfully (12 MLX-environment
skips, one Torch-environment skip). Four profile catalogs and whitespace checks
passed. No native Laya model was loaded by this control study.

## Next work

The competition pipeline still needs better full-protocol data coverage,
unseen-system stress tests, a complete selection/fallback policy, final fitting
of any accepted learned candidate, submission packaging and resource checks.
A pretrained semantic representation is an untested alternative to this lexical
control. No new native Laya inference or canceled GPU job was launched here.
The failed quality result is retained so later feature/model choices must earn
their complexity through matched comparisons.
