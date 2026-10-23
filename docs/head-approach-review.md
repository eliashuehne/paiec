# Engineering review of the enriched decision-head pilot

2026-10-22. This reviews **saved training inputs, development predictions and
training logs** from the completed cached enriched pilot. No new model training,
inference, calibration fit, or access to the sealed calibration/confirmation
roles was performed. Proposed experiments below have **not** been run.

## Assessment

The training and evaluation machinery works, but the resulting predictor has
mostly learned to retreat toward the training average. We have not yet shown
useful, generalizable prediction of which system will solve which task.

The most urgent work is to preserve task-specific information, use the limited
outcomes available at prediction time, and establish smaller, stronger controls.
Adding more architecture descriptions or training more epochs is not yet
supported by the evidence. Enrichment remains a hypothesis requiring a matched
control, rather than an established improvement.

Saved analysis: `runs/head_finetuning/approach_audit_v1/report.html`, with
`summary.json`, `per_pair.csv`, `coverage.csv`, `input_collisions.json`, real
examples, plots and source hashes. Reproduce into a **new** directory:

```bash
.venv-laya-mlx/bin/python scripts/audit_head_approach.py --out runs/head_finetuning/approach_audit_NEW
```

## What the scores actually establish

| Predictor or diagnostic | Development Brier ALC | Interpretation |
| --- | ---: | --- |
| Original frozen parent, same enriched inputs | 0.33525644 | Poorly suited raw probabilities |
| Constant 50% | 0.25000000 | Stronger than the original parent |
| Training-only, equal-pair global success prior: 49.865% | 0.25001980 | A fitted global average offers no useful improvement |
| Trained head, selected epoch 1 | **0.24908013** | Improvement of only 0.00091987 over 50% |
| Replace each pair's predictions with their own mean prediction | **0.24842928** | Item-specific variation makes the score worse on this sample |

The last row uses the model's own predictions, **not evaluation labels**, to
choose each constant. It is an explanatory diagnostic requiring the sampled
target inputs, not a claim about an already implemented online predictor.

All probabilities lie between **46.8% and 55.6%**. Mean within-pair AUROC is
**0.450**, where 0.5 represents chance ordering. In plain terms, the head's small
changes between questions are not reliably assigning higher probabilities to
successful outcomes. Calibration alone cannot create missing task information
or fix an ordering that a monotonic recalibration preserves.

| Development pair (reporting names only) | Sampled outcome rate | Mean prediction | Brier | Within-pair AUROC |
| --- | ---: | ---: | ---: | ---: |
| C-Eval / Baichuan2-7B | 55.8% | 51.4% | 0.24442 | 0.613 |
| C-Eval / Qwen-7B | 55.0% | 53.9% | 0.24785 | 0.504 |
| C-Eval / Yi-34B | 83.3% | 52.2% | 0.23840 | 0.377 |
| C-Eval / Llama-7B | 25.8% | 51.9% | 0.26217 | 0.437 |
| ERBench / gpt35 | 33.1% | 48.2% | 0.24419 | 0.498 |
| ERBench / gemini | 13.1% | 48.0% | 0.23598 | 0.425 |
| ERBench / llama | 48.3% | 48.3% | 0.25251 | 0.382 |
| ERBench / gpt_finetune_2 | 90.8% | 48.0% | 0.26713 | 0.363 |

Five pairs beat constant 50%, but most gains are a small average shift in the
right direction, rather than useful differentiation between questions. Baichuan
is the promising ranking slice in this sample; 120 questions in one pair do
not establish a repeatable capability. Table rates use different sampled task
sets and repeated outcomes where recorded, not a matched-task system ranking.

The improvement over 50% is uncertain. A 4,000-replicate item-cluster bootstrap
gives a 95% interval for `Brier - 0.25` of **[-0.002539, +0.000687]**, including
no improvement. Items are resampled jointly across systems within each fixed
benchmark, repeated trials stay together, and pairs retain equal weight. This
is conditional on the selected eight pairs/two development benchmarks; it does
not measure uncertainty on unseen families or correct the optimism from
selecting the best epoch on these same development data.

There are **963 response outcomes, 960 system–item inputs and eight pairs**,
not 5,760 independent examples. The six budgets reuse identical predictions.

## What is working

- The audit reproduces the recorded macro Brier score exactly. Pair weighting
  and repeated-response counts agree with the saved prediction files.
- Frozen-family roles and provenance are explicit. There is no overlap between
  training and development retained-input hashes in this audit. This is an
  exact-input check, not a proof against all semantic duplicates or upstream
  pretraining overlap.
- Encoder and auxiliary-head hashes stayed fixed; decision weights changed.
  Cache equivalence checks and exact export/reload checks passed. There is no
  evidence that the memory optimization caused the quality limitation.
- The first epoch's useful checkpoint was retained instead of using epoch four.
- Training settings, source profiles and the input contract are bundled for
  later prediction. Missing identity remains missing instead of being guessed.

## Problems supported by the data

### 1. Truncation can remove the actual distinction between tasks

**2,042/5,000 training records (40.84%)** are truncated at the current 1,024-token
limit. There is no truncation in the selected development sample, so input
quality also differs between fitting and evaluation.

More seriously, **654 training records** belong to 17 retained-input groups
that each contain more than one distinct full task: **625 BenGER records and
29 ClashEval records**. The head cannot learn a difference it never receives.
Ordinary identical trials and stochastic outcomes are not themselves corrupt
data; these counts specifically require distinct full-task hashes.

One BenGER group contains **209 distinct tasks with the same retained input**.
For example, zero-based training rows 14 and 41 concern a constitutional
complaint and criminal liability, respectively. Their full states are about
13,257 and 12,164 characters, but begin with the same long German instructions
and target system. The retained token sequence is identical. Longer generic
instructions have crowded out the distinguishing case facts.

| Training benchmark | Records | Truncated | Distinct retained inputs |
| --- | ---: | ---: | ---: |
| BenGER | 833 | 691 (83.0%) | 211 |
| BIRD SQL | 833 | 566 (67.9%) | 833 |
| ClashEval | 834 | 489 (58.6%) | 819 |

Unique retained inputs do not prove a complete task is visible: a partially
visible SQL schema can differ between rows while still omitting needed tables.
Likewise, increasing the limit alone is not a guarantee that important content
survives. Token-retention tests need to inspect semantic fields.

### 2. Some target systems are indistinguishable at the allowed interface

Three selected C-Eval systems have identical supplied profiles: OpenCompass,
five shots, minimum-perplexity selection and the same source-study setting.
Their canonical model names are empty. The display names in the source tables
identify Baichuan, Yi and Llama, but those names are **not predictor inputs**.

Their sampled outcome rates are about **56%, 83% and 26%**. For an identical
task, our predictor necessarily supplies the same probability for all three.
Their sampled item sets also differ, so these rates are not a controlled
capability comparison. There are eight development pairs but only **six
distinct supplied system profiles**.

Feature engineering must not repair this using private IDs or display-name
fallbacks. Mark identity uncertainty explicitly; report performance with and
without unresolved profiles. This is also a concrete metadata question to ask
the organizers, if the user later authorizes contacting them.

### 3. The head has ample capacity to fit this small, narrow sample

Head-only training still updates **26,248,193 parameters**. The 5,000 training
outcomes cover only **24 pairs, seven families and 3,573 distinct tasks**. They
are not 5,000 independent model configurations.

| Epoch | Mean logged update cross-entropy | Development Brier ALC |
| --- | ---: | ---: |
| 1 | 0.7486 | **0.24908** |
| 2 | 0.6368 | 0.25615 |
| 3 | 0.5719 | 0.26679 |
| 4 | 0.5489 | 0.27266 |

Training loss falls while development error grows: consistent with overfitting
and domain mismatch. Longer training is unlikely to solve the observed problem.
Cross-entropy is a proper probability loss; choosing it is not, by itself, a
scoring bug. A Brier-loss ablation can wait until the stronger problems are
addressed.

### 4. Current enrichment supplies little verified model information

Published model-property blocks occur in **1,429/5,000 training rows (28.6%)**
and **120/963 development rows (12.5%)**. All development model-property blocks
belong to Qwen. All development rows have harness research, much of it describing
general capabilities rather than verified settings of the recorded run.
No selected row has the dedicated reasoning-effort field populated; extra fields
can still contain other reasoning/generation settings.

We have not completed a matched full `context_v1` versus `enriched_v1` training
comparison. Comparing the original parent with the trained enriched head changes
both weights and inputs relative to a dataset-only experiment; it does not tell
us whether architecture facts helped. More facts repeated across a few models
do not provide more independent model examples.

### 5. Language coverage is a plausible representation bottleneck

**479/480 C-Eval task texts contain Chinese characters**. Training includes
119 ClashEval rows containing Chinese characters and substantial German legal
text, but the frozen parent is the ModernBERT-large English checkpoint. Upstream
also offers a multilingual variant. This motivates a controlled language/encoder
comparison; it does not establish language as the sole cause, particularly
because English ERBench discrimination is poor too.
[Laya model card](https://huggingface.co/convaiinnovations/laya)

The existing cheap text ridge is not an adequate Chinese-text control as written:
its `[a-z0-9]+` tokenizer drops Chinese characters. Use Unicode-aware tokens or
character n-grams, with the same equal-pair training weights and family holdouts.
No tokenizer or checkpoint was changed during this review.

### 6. The current ALC ignores the available adaptation signal

The pilot ignores revealed outcomes, so every budget has the same Brier score.
The published ALC weights put **90% of the total weight at positive budgets**.
An evidence-aware control is therefore a high-priority experiment.
[Competition scoring](https://aims.stanford.edu/competition)

Even a simple smoothed estimate is useful to test:

```text
updated_probability = (k * initial_probability + revealed_successes) / (k + revealed_count)
```

Tune `k` using training-family folds. Start with a constant or system-level
initial prior, then compare an item-specific prior. The latter assumes observed
outcomes can usefully shift the target probability; it is a baseline to measure,
not a guarantee of improvement. The repo already has shrunk-mean machinery.

Match evidence using only visible system fields and the anonymous benchmark ID.
Identical visible profiles remain ambiguous: an internal subject ID cannot be
invented for prediction. Never include target/evaluation outcomes, and keep all
trials of the target item out of evidence when constructing training episodes.

## Feature engineering proposal

Implement this as a versioned feature builder shared by preparation and
prediction. Preserve the allowed raw fields and provenance alongside derived
features. Unknown values stay unknown; absent tools are not automatically
disabled tools. Separate recorded settings from documented capabilities.

| Feature group | Concrete examples | Information source and guardrail |
| --- | --- | --- |
| Input visibility | Full/retained task tokens, retained question/options/schema flags, omitted fields, missing context | Our tokenizer/renderer; missingness is explicit, not a fabricated task property |
| Task structure | Language/script, prompt length, answer-option count, MCQ/open answer/code/SQL format, supplied category, number of examples | Only supplied text/features; deterministic extraction with confidence/unknown flags |
| Model | Recorded identity/family/version, known total and active parameters on log scale, dense/MoE, modalities, documented languages/domain tuning | Exact supplied identifiers; externally researched facts remain a separate research feature set |
| Harness/run | Shot count, actual token limit, recorded tools, reasoning setting, retry/turn budget | Recorded values only; never substitute a documentation default for a measured configuration |
| Interactions | Task language × model language coverage; task modality × support and supplied content; prompt size × verified context limit; code/SQL × domain tuning; task complexity × actual reasoning budget | Only when each ingredient is available; avoid assuming that more parameters always means higher success |
| Revealed evidence | Number observed, successes, smoothed rate, task-similarity summaries and their support counts | Only the labels allowed at the current budget, with target/trial exclusion |

Do not encode real benchmark names, task IDs, reference answers, correctness
verifiers or outcomes as predictive shortcuts. A category explicitly recorded
in the supplied item features is usable; a category invented from hidden
benchmark metadata is not.

For long inputs, parse nested messages/question/context/options instead of
passing escaped JSON inside an opaque string. Reserve space for task-specific
content and the target system; avoid spending the entire budget on generic
instructions. Where the context cannot fit, compare field-aware chunks or
multiple views, with explicit omissions, against truncation. Summarization is
not automatically lossless and needs its own controlled evaluation.

Structured features need not all be additional prose for Laya. A small numeric
model can combine frozen text features, structured system/task features and
evidence statistics. This directly tests whether the 26M-parameter decision head
is necessary for the available data.

## Ordered experiment plan — not yet executed

| Priority | Experiment | What it resolves | Success evidence |
| --- | --- | --- | --- |
| 1 | Versioned task-preserving input builder; retained-field audit on all training rows | Whether the predictor actually receives the case/question/options/schema | Remove unintended multi-task collisions; report retained semantic fields, not token counts alone |
| 2 | Constant/train-only priors, Unicode character/text + structured linear model, small frozen-feature probe | Whether simple regularized predictors learn more than the current head | Pair-weighted Brier, within-pair AUROC and consistent improvements across training-family folds |
| 3 | Numeric revealed-label update at budgets 0/1/3/7/15/31; then similarity-conditioned evidence | Whether a simple adaptation mechanism improves the metric we compete on | Lower paired ALC with correct nested acquisition/evaluation exclusion |
| 4 | Matched dataset-only versus enriched training, plus structured-feature ablations | Whether external model architecture information adds value | Improvement survives identical inputs/samples/seed/recipe except the tested feature group |
| 5 | Smaller/lower-rate/stronger-regularization decision head; language-aware encoder control if needed | Capacity and representation bottlenecks | Gain beyond simple controls; no repeat of late-epoch development deterioration |

Items 2 and 3 can begin as small CPU/frozen-feature controls before another
large cloud run. Cache reuse is valid only for unchanged encoder weights and
identical retained inputs; the proposed renderer change requires new features.
Adding more independent training pairs/families is likely more informative than
duplicating outcomes for the same few configurations, but expansion requires
the existing semantics/input-completeness review and preserved family roles.

Use leave-one-family-out folds **within the seven training families** for tuning
and feature selection; do not move frozen held-out families into training. Track
language, input length, identity resolution and profile coverage as slices.
Add an unseen-model stress test where identity coverage permits, separately
from benchmark-family generalization. Treat the current development role as
already used for checkpoint selection and this audit. Keep calibration and
confirmation sealed until the corresponding recipe/decision gate is frozen.

For decisive comparisons, use identical sample manifests, original checkpoint,
input contract, pair weighting and nested budgets. Separate each change into
an ablation; repeat promising comparisons across seeds and report paired
uncertainty. Do not select recipes only by the best of many tiny development
gains. Report identity-missing and complete-input subsets alongside the full
score rather than quietly dropping hard rows.

## Competition eligibility of researched features

Current rules restrict competition-specific data sources to the inventory and
prohibit runtime web browsing. The web-researched metadata is not yet cleared
under that restriction. Keep a dataset-only path, preserve the research catalog's
`unreviewed` eligibility status, and resolve the interpretation before using it
in a submission. No organizer contact was made.
[Official rules](https://aims.stanford.edu/competition)

## Recommended next action

Implement the task-preserving feature builder and its visibility audit first,
then compare simple feature/evidence controls before retraining the large head.
There is meaningful information to recover, but the current results do not yet
justify unfreezing the encoder or promising a particular target Brier score.

The hindsight diagnostic in `summary.json` (Brier 0.1866 from each pair's actual
development outcome rate) is deliberately **not** a baseline, feature or promised
target: it uses labels unavailable at prediction time. It only illustrates that
different sampled pairs have different rates which the current near-50%
predictions do not capture.
