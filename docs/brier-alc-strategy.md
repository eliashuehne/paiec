# Engineering assessment toward Brier ALC 0.15

Status as of 2026-10-24: **Laya work is on hold** after the completed grouped
pilot failed to beat the constant baseline. This is a retained experiment or
implementation reference; proposals and commands below are not a current work
queue. See [current status](current-status.md) and [shared data](shared-data-start.md).

2026-10-23. Follow-up to the completed compact enriched head. The user asked
for an engineering assessment, including confidence and encoder training.
This work audits saved predictions and runs a fixed numeric evidence control.
No new Laya inference, gradient training, cloud job, calibration/confirmation
access, or default prediction change occurred.

## Diagnosis from the completed head

Audit: `scripts/audit_head_strategy.py`; artifacts:
`runs/head_finetuning/compact_strategy_audit_v1/`. The raw saved score is
reproduced exactly on 960 original development system-item inputs / 963 response
rows / eight pairs. All budgets and response/pair weights are preserved.

| Measurement | Value | Interpretation |
| --- | ---: | --- |
| Raw head Brier ALC | 0.24864261 | Very little skill over constant 0.5 |
| Mean within-pair AUROC | 0.42803711 | Weak/reversed question-level ranking on these eight pairs |
| Pooled response AUROC | 0.55359947 | Between-pair/domain differences can hide weak within-pair ranking |
| Predicted probability range | 0.465174–0.560411 | Narrow forecasts |
| Double deviations from 0.5 | Brier 0.24916408 | More confidence worsens the score |
| Quadruple deviations from 0.5 | Brier 0.25584363 | Worse than constant 0.5 |
| Best development-fitted linear stretch | 0.24859612 | Label-using diagnostic only; cannot close the gap |
| Best constant-per-pair development fit | 0.18660171 | Label-using optimistic oracle, not an eligible predictor |

Stretching uses `clip(0.5 + factor * (p - 0.5), 0, 1)`. The analytic best
unclipped factor is 1.222452 on these same outcomes. It is deliberately fitted
on development only to diagnose the scale of a calibration opportunity, and
must not be used as a production transformation or claimed held-out result.

The pair oracle assigns each scorer-side pair its empirical evaluation success
rate. It is the minimum Brier among constant-per-pair predictions on this
sample, even with access to evaluation outcomes/private pair identity. It is
not the overall Bayes/noise floor: task-specific predictions could do better.
It shows that estimating only system/benchmark average success cannot reach
0.15 on this sample. Most cells have one observation, so per-item label fits
would not establish irreducible stochastic noise either.

Three C-Eval systems have identical permitted subject descriptions, while their
sampled outcome rates range from 25.83% to 83.33%. Their mean predictions are
almost identical near 53%. Missing information cannot be repaired by training
or filled with audit-only display names/private IDs. Uncertainty can be honest
when the available input cannot distinguish systems. Language mismatch may
also matter, but ERBench's English questions show poor ranking too; Chinese
coverage is not a complete explanation.

## Confidence must follow information

For a genuine success probability q and forecast p, expected binary Brier is:

`q * (1-q) + (p-q)^2`.

If q is 0.5, forecasting 0.9 scores 0.41 in expectation versus 0.25 at 0.5.
If q is 0.9, a forecast of 0.9 scores 0.09. The goal is to learn which cases
justify those probabilities, not reward confidence itself. Penalizing entropy,
forcing forecasts away from 0.5, or choosing a threshold for accuracy can worsen
a proper probability score. Existing weighted binary cross-entropy is already
a proper probabilistic objective; replacing it or imitating RLCD is not by
itself a remedy for absent information, weak representations or bad supervision.

## The measured value and limits of revealed outcomes

The untuned k=5 update reuses saved query probabilities and original acquisition
responses, with nested budgets, the original seed/50-50 split, trial grouping,
acquisition exclusion and all-sampled-pair evidence. Matching uses allowed
visible profiles and anonymous benchmark IDs. Identical visible subjects are
not secretly separated by private scorer IDs. No inference or fitting is used.

| Predictor | ALC | B0 | B1 | B3 | B7 | B15 | B31 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Current head | .248643 | .248643 | .248643 | .248643 | .248643 | .248643 | .248643 |
| Current head + beta k=5 | .237534 | .248643 | .257250 | .256870 | .225116 | .214361 | .219502 |
| Constant 0.5 + beta k=5 | .239531 | .250000 | .261199 | .260346 | .226604 | .214664 | .219682 |

This modest control already matters more than the latest head-only gain, but
worsens budgets one and three on this trajectory. It does not approach 0.15.
It is one fixed previously used acquisition trajectory, not a generalization
claim or a hyperparameter search.

The [competition metric](https://aims.stanford.edu/competition) weights budget
zero by 0.1 and positive budgets by a total of 0.9. An illustrative curve of
B0=.20, B1=.18, B3=.16, B7=.14, B15=.12, B31=.10 has ALC exactly .15. These
are planning targets, not forecasts. With B0 still .248643, the positive-budget
weighted average would need to be approximately .139040 to achieve .15.
Success at budget 31 alone is insufficient because that point has weight .1.

Our beta update `(k*p_i + successes)/(k+n)` shrinks all question-level differences
by `k/(k+n)`, becoming a pair-wide average with much evidence. A stronger design
should learn from support outcomes relative to their predicted difficulties,
retain useful query ordering, and adapt capability/difficulty estimates.

## What to change

### 1. Qualify the learning machinery before enlarging a run

Use a standalone, trivially learnable synthetic positive control, then a tiny
training-only overfit test. Verify class/option mapping, retained markers,
gradients and actual weight changes, and whether training loss/ranking can
improve strongly on that tiny slice. Repeat a bounded setup across seeds if
learning stays near the prior. A toy success proves machinery, not real-task
generalization. No target outcome/reference answer should be injected into any
competition input; synthetic controls stay separate from competition exports.

The current [upstream fine-tuning guide](https://nandhakishorm.github.io/laya/finetune/)
explicitly discusses chance-collapse diagnostics and uses encoder updates by
default. Our custom `freeze_for_head` freezes every encoder parameter each
window; our head-only outcome is not a test of that broader training recipe.

### 2. Broaden behavioral supervision using existing training families

The 5,000-row pilot covers 24 system-benchmark pairs. The prior training-only
breadth audit identifies 170 candidates / 35 canonical-name strings and
151,356 distinct system-item combinations before additional admission/overlap
checks. These are candidates/configurations, not 170 independent base LLMs.
Only four name strings recur across training benchmarks; connectivity is weak.

Review and admit candidates, emphasizing the same questions attempted by
several systems and the same system across question difficulties/domains.
A first expanded 30k-50k sample is a proposal, not an existing recipe. Preserve
trials, avoid treating repeated trials as independent task diversity, balance
families/pairs and keep original development/calibration/confirmation families
out of training. More SQL configurations must not dominate every minibatch.
Use all verified enrichment already collected; additional generic web prose
is lower priority than broader outcome supervision and meaningful interactions.

### 3. Make partial encoder adaptation the next major training ablation

Compare on the same expanded enriched inputs and training-family folds:

- Frozen encoder with a regularized head as the control.
- Encoder LoRA/adapters plus the same head. Rank 8/16 is an initial hypothesis,
  not a qualified configuration. Keep encoder learning rates lower than head
  rates and use early stopping/regularization.
- If justified, train only the upper two to four encoder blocks alongside the
  head, then consider full encoder training after reproducible gains.

[LoRA](https://arxiv.org/abs/2106.09685) adapts transformer computations with a
small trainable update while keeping pretrained base weights fixed. It does
not promise 0.15. Full encoder training adds substantial capacity to a model
whose existing head already has 26.25M parameters for only 5,000 observations.
The earlier four-epoch head pilot overfit development; unfreezing everything
without a better test/data plan could amplify that problem.

Test an explicitly versioned multilingual base/preparation as a language
ablation, keeping sample identities fixed. English and multilingual checkpoints
have different tokenizer/head contracts; do not swap them under existing
weights. Upstream's [model card](https://huggingface.co/convaiinnovations/laya)
describes the English and multilingual encoders separately.

Any encoder-training policy must replace the current repeated freeze policy,
log which layers changed, and qualify gradients/memory/exports on the actual
GPU. The existing full-frozen-encoder feature cache becomes invalid as soon as
encoder parameters change. New adapters need a save/load/merge contract for
later Torch and MLX inference. The prior ~3 GB usage is not a resource estimate
for encoder backpropagation. No such implementation or GPU run was launched.

### 4. Teach the actual budget-dependent prediction task

Build training-only episodes with disjoint support and query items, keeping all
trials of each item together. Query labels are loss targets, never inputs.
Sample nested evidence counts 0/1/3/7/15/31 and weight query probability losses
with the competition's ALC weights. Keep the original acquisition/evaluation
boundary, and avoid retrieving a query's own item or other held-out outcomes.

Represent system capability from model/harness/run attributes, task difficulty
and skill demands from task content, and an evidence correction from compatible
support tasks plus their observed outcomes. A useful proposed form is:

`p = sigmoid(dot(system_ability, task_requirements) - task_difficulty + evidence_correction)`.

These are learned quantities, not manually assigned skill scores or private
model/item ID embeddings. [EmbedLLM](https://arxiv.org/abs/2410.02223) and
[ability/difficulty item-response models](https://arxiv.org/abs/2510.00844)
motivate the factorization, not a proven result for our competition setting.

Task embeddings should learn predictive difficulty/skill relationships from
outcomes. Generic semantic similarity alone can match easy/hard questions with
the same topic. Combine task similarity with system compatibility; a strong
model's success is weak evidence about a weaker target. Separate compact
support summaries/embeddings from target-state text to preserve the question.
With a reliable difficulty model, later test budget-aware acquisition of
informative/diverse questions; current chance-like uncertainty is insufficient
to justify an uncertainty-selection heuristic.

## Decision gates and scope

Keep all new arms enriched; the important control for encoder adaptation is
the same enriched data with the encoder frozen. Changes to data breadth,
architecture, evidence objective and encoder training need separable ablations,
not one conflated run. Search using held-out families within the training role;
original development stays a comparable, repeatedly inspected development set.
Keep final calibration/confirmation sealed until the recipe is frozen.

Measure within-pair AUROC, proper Brier/calibration, system-rate bias, low-budget
curves, identity/language/visibility slices and training-versus-held-out gaps.
A ranking improvement above chance is useful evidence, not a universal AUROC
threshold guaranteeing a Brier score. Calibrate after a useful predictor exists,
using training-only holdout/out-of-fold predictions during development; never
fit a temperature on the same development data used to report its improvement.

0.15 is a research target, not yet an established achievable value from these
allowed inputs. Eight pairs/two development families cannot establish hidden
competition performance. External facts remain research-only until their
eligibility under the inventory restriction is reviewed. No default, checkpoint,
question wording, renderer, frozen role or historical metric was changed.
