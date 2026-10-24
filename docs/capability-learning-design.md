# Training Laya to model capabilities, difficulty and similarity

Status as of 2026-10-24: **Laya work is on hold** after the completed grouped
pilot failed to beat the constant baseline. This is a retained experiment or
implementation reference; proposals and commands below are not a current work
queue. See [current status](current-status.md) and [shared data](shared-data-start.md).

2026-10-22. Research and training-design follow-up to
[task-feature-pipeline.md](task-feature-pipeline.md). The user authorized broad
web research and wants a stronger learned predictor of target-system success.
The proposal below is **not an implemented or trained architecture**. The work
performed here is primary-source research and a training-role data-breadth audit.
No embedding model was downloaded/run, and no GPU/API training job was launched.

## Recommendation

Train a predictor that learns three connected things:

1. What abilities the supplied model and harness are likely to provide.
2. Which abilities the supplied task requires, and how difficult it is.
3. How actual outcomes on similar tasks should change the estimate.

Web documentation supplies explanatory features and priors. Observed outcomes
supply the main supervision connecting those features to success. More text
about models does not automatically supply more behavioral evidence. A larger
catalog and a broader, crossed system–task response matrix serve different
purposes; both matter.

The first experimental model should be small and explicitly conditional on
attributes, with a later stage adapting Laya's encoder. Avoid an ID-only model
embedding: it cannot represent an unseen or unidentified target system.

## Research grounding and the adaptations we need

- [EmbedLLM](https://arxiv.org/html/2410.02223v2) learns model representations
  by predicting correctness from model and question vectors. Its implementation
  identifies models by index and uses a question-level random split. We can
  borrow the behavioral representation objective, but need an attribute-based
  system encoder and whole-family evaluation rather than adopt that setup.
- [IrtNet / Learning Compact Representations of LLM Abilities via Item Response
  Theory](https://arxiv.org/html/2510.00844v1) factors success into a multi-skill
  model vector, a query discrimination vector and a difficulty scalar. Our
  design adapts this idea to model-plus-harness descriptions, missing identities,
  and observed-label budgets. Their majority-vote treatment of repeated answers
  is not suitable for our recorded-trial probability task.
- [Observational Scaling Laws](https://arxiv.org/html/2405.10938v2) studies
  low-dimensional capability structure across models. It motivates looking for
  a small capability representation; it does not establish that parameter counts
  or architecture descriptions alone will predict success on our data.

These are related methods and evidence of plausibility, not measured gains on
our frozen evaluation.

## New measured finding: the data can be broader, but is weakly connected

Audit: `scripts/audit_capability_training.py`; saved report and source hashes:
`runs/head_finetuning/capability_data_audit_v1/`.

The audit reads only the eight training-role benchmarks, their minimal original
subject/item/response tables, and existing training-export provenance. It applies
binary-outcome/text linkage and the 80-item threshold, plus BenGER's recorded
prompt scope restriction. Candidate status is not automatic subject admission.

| Training benchmark | Current sampled pairs | Candidate pairs | Candidate nonempty canonical names |
| --- | ---: | ---: | ---: |
| abstract_reason | 1 | 1 | 1 |
| adaptivestep | 2 | 2 | 0 |
| benger | 4 | 25 | 12 |
| bird_sql | 4 | 120 | 15 |
| cellverse | 1 | 1 | 1 |
| clasheval | 4 | 6 | 6 |
| critic_discernment_game | 4 | 4 | 2 |
| fewshot_ttt_bbh | 4 | 11 | 2 |
| **Total pairs** | **24** | **170** | **35 unique name strings overall** |

BIRD's 120 configurations cover the same 360 questions, making it especially
useful for learning differences between model and prompting configurations.
Shot count, instruction size and serving/model identifiers are recorded. They
are not 120 independent LLMs. BenGER's 25 candidates cover 744 eligible items
with varying coverage; actual run settings and identity resolution need review.

In our current sample, 2,729 benchmark–item groups appear with only one sampled
system, while 447 appear with two, 369 with three and 48 with four. For learning
ability versus difficulty, overlapping observations are particularly valuable.

Only four canonical-name strings recur across multiple training benchmarks:
OpenAI GPT-3.5, Anthropic Claude 3 Opus, Anthropic Claude 3 Sonnet, and Meta
Llama 3.1 8B. Equal names do not verify equal historical snapshots/settings.
This is weak connectivity between domains, not an established common-panel
measurement of general ability. Ability and difficulty can trade off in
disconnected response graphs. Shared attribute encoders provide statistical
coupling, but do not create missing observational links.

Consequently, prioritize more independent systems and shared questions, followed
by cross-domain connections where provenance establishes them. Increasing trial
counts for the same items is not equivalent to increasing that coverage. For
example, AdaptiveStep has 931,328 candidate binary response rows but only 3,638
distinct system–item combinations after these filters.

Keep frozen family roles unchanged. Reviewing/admitting additional systems from
training families is a new training-data version, with its own completeness,
semantics and overlap checks. Do not absorb the development/calibration/
confirmation families into training to create a denser matrix.

## 1. A system representation derived from supplied attributes

Create separate representations for the base model and the actual harness/run,
then combine them. Useful fields, in descending initial priority:

| Component | Features to collect or derive |
| --- | --- |
| Model identity and lineage | Exact reported identifier/version; base model; instruction/domain adaptation; identity confidence and missingness |
| Functional capabilities | Documented modalities/languages, context/output capacities, tool-call format support, domain-specialized training |
| Run behavior | Actual shots, prompting/selection method, output/reasoning limits, enabled tools, candidate selection, verification and retry/turn budgets |
| Architecture | Total and active parameters, dense/MoE, backbone and attention family; keep undisclosed internals unknown |

Combine semantic embeddings with numeric/categorical features. A sentence
embedding should not be expected to encode exact token ceilings or parameter
ratios reliably. Model-capacity and harness features need interactions with the
task; a tool-capable model with no tool enabled is a different condition from
one actually running Python.

Collect once per resolved entity/version, not once per response. Prefer model
cards/configuration, technical reports, original harness repositories and
recorded run files. Preserve date, source commit/snapshot, evidence passage,
confidence and whether the fact is available at inference. Unsupported or
ambiguous identity matches stay unresolved.

Existing homogeneous static facts stay in `research/system_profiles/`. Measured
capability observations need a **separate** table: system/version, source-task
family, protocol, metric definition, numerator/denominator if known, and source
provenance. Do not insert leaderboard scores or subjective strength rankings
into static architecture facts. Marketing claims and LLM-generated assessments
are weaker evidence than actual recorded outcomes.

## 2. Difficulty as a learned, system-relative property

Represent a task by its question, context, options and required interaction,
with separate views for long context and task-specific instructions. Derive
domain/format/language and resource-demand features where the input supports
them. Token count alone is not difficulty.

The proposed small head has this form:

```text
ability = system_encoder(model facts, recorded harness settings)
requirements, difficulty = task_encoder(task text and features)

initial_probability = sigmoid(dot(ability, requirements) - difficulty)
```

Start with a 16-dimensional latent representation as a hypothesis, and a modest
MLP projection budget below one million new trainable parameters. Tune dimensions
and regularization using training-family folds. Latent coordinates are not
automatically named skills; numerical identifiability/scale constraints and
auxiliary labels would be needed for stronger interpretability claims.

Learn from actual binary observations, retaining trials. The model must predict
difficulty from new task text, not retrieve a fitted item-ID parameter at test
time. Raw failure rate is a useful diagnostic, but is confounded by which systems
attempted the question. Joint modeling and crossed observations help address
that confounding.

For an algebra task, a model's mathematical ability should matter more than its
ability to use a browser. For a long-context SQL task, schema understanding and
the recorded context/workflow limits can matter. These are desired learned
patterns, not hand-assigned correctness labels.

Optional auxiliary objectives include matched-question comparisons across
systems, task-skill annotations, and model-fact entailment. Give weak/synthetic
annotations an explicit confidence and low influence. Stochastic outcomes do
not justify asserting a permanent ranking from one success versus one failure.

## 3. Two kinds of similarity, with observed outcomes attached

We need task similarity and **system compatibility**. A successful answer by
a much stronger model is weak evidence about a weaker target, even when the
questions are near-identical.

For a first retrieval control, compare lexical retrieval with a pinned frozen
multilingual embedding encoder. [BGE-M3](https://huggingface.co/BAAI/bge-m3) is a
candidate: its card documents multilingual retrieval and an 8,192-token limit.
This is not proof it measures difficulty. No embedding weights were downloaded
or evaluated for this design.

Retrieve on task-specific content, not shared wrappers or benchmark names.
Use separate views for required operations, context/schema and options. A
question about proving a theorem and one about stating it can be semantically
close while differing sharply in difficulty. Inspect such retrieval failures.

The retrieval feature can summarize:

- Compatible observed tasks, their similarities, and support counts.
- Observed success/failure and residuals relative to their own system priors.
- Whether neighbors are the same system, a related documented system, or an
  uncertain match.
- Whether enough applicable evidence exists to depart from the initial prior.

Prefer learning a correction from these features to blindly averaging labels.
A residual such as `observed_outcome - prior_for_that_system_and_task` helps
separate surprisingly good/bad outcomes from a model that simply succeeds often.
This correction is a proposal and must be fitted out of fold; a residual based
on in-sample fitted priors can otherwise encode overfitting.

At budget zero, memory contains only training-role observations. At positive
budgets, add only allowed revealed outcomes. Never retrieve held-out labels or
the query's own item/trials, and audit exact/near duplicates across roles.
Anonymous benchmark IDs may scope evidence; they must not identify hidden tasks.
When visible profiles are identical, do not disambiguate them using private IDs.

Train the use of memory episodically: hide query outcomes, select a disjoint
support set, vary the allowed evidence count, and optimize query predictions.
All trials of an item stay together. Begin with the existing 50/50 pool and
nested 0/1/3/7/15/31 budget discipline. Use compact embedding/statistical features
or a separate evidence module, so examples cannot crowd the target out of its
context window.

## 4. How this becomes Laya training

**Phase A: representation/control experiment.** Reuse Laya's pinned frozen
encoder to extract separate system and task features, then fit the small
attribute-conditioned ability/difficulty head. Keep the existing trained Laya
probability and the numeric update as controls. Add a separate retrieval encoder
only as an explicitly tracked feature experiment; ordinary pooled ModernBERT
features should not be assumed to be retrieval-calibrated sentence embeddings.

This is a new Laya-derived head and needs explicit model/checkpoint/runtime
support. It is not a flag already supported by the native decision-head trainer.
Separate system/task encoding is also a new cache layout; do not reuse the old
joint-state feature cache. Pin model revision, tokenizer, pooling, chunking,
input hashes and split membership in every cache key.

**Phase B: evidence-conditioned training.** Add episodes and a small evidence
correction/gating network. Teach it to fall back to the initial estimate with
little compatible evidence. Compare with the untuned numeric update at every
budget; improvement only at 31 labels is insufficient for selection.

**Phase C: encoder adaptation.** If frozen representations limit the small
head, train adapters or selected upper encoder layers at a lower rate alongside
the head. This lets representations learn relationships such as model lineage,
prompting changes and task demands, instead of relying only on their original
generic encoding. It requires a broader response panel than the original 5,000
records. Keep an encoder-frozen ablation and check language coverage separately.
Do not unfreeze all 421M parameters as the initial experiment.

**Optional distillation.** A stronger offline teacher could consume verified
facts and training-only examples and provide task-skill/difficulty annotations
or soft probabilities. It must not see the query outcome/reference answer;
teacher scores are weak labels, not ground truth. Compare with outcomes-only
training. Distilling retrieval-assisted predictions into Laya could reduce
inference cost, but does not remove the need for a versioned catalog when new
model facts arrive. No paid teacher API calls are authorized or made by this
design alone.

## Data plan and comparison sequence

1. Review the 170 candidate training pairs, emphasizing exact identities,
   complete recorded tasks, shared questions and real configuration differences.
   Preserve pair/family balance and avoid duplicating thousands of trials as
   if they were new tasks. Choose the expanded sample size after this review.
2. Extend profiles for the reviewed entities and measure architecture, run and
   domain-feature coverage. Keep missingness and source quality as features.
   More SQL configurations must not dominate every minibatch.
3. Compare the small difficulty/capability head with the existing head and
   Unicode/structured ridge, using the same expanded sample and family folds.
4. Add retrieval as one ablation, then budget-aware episodes, then encoder
   adaptation. An improvement in this sequence identifies what actually helps.
5. Freeze the recipe before calibration and confirmation. Test an unseen-model
   or whole-model-family split within the training role as a separate stress
   test, in addition to benchmark-family holdout.

Use weighted proper losses for actual outcomes and Brier ALC for selection.
Require within-pair discrimination gains, not just benchmark-average shifts.
Report calibrated probability quality, uncertainty, identity coverage,
truncation/visibility and language slices. A useful sanity check shuffles
outcomes within each training pair while preserving rates: surviving gains
would suggest shortcut reliance. Identity dropout and descriptor ablations
can test whether the system representation generalizes beyond memorized names.

We should not claim named latent skills, causal architectural effects or a
specific achievable Brier score without those validations.

## External data and evaluation integrity

The user's permission covers a broad research experiment. Keep a dataset-only
control because [competition rules](https://aims.stanford.edu/competition)
currently restrict competition-specific data sources to the inventory and
prohibit runtime browsing. Web-derived features remain research-only until
eligibility is established; this does not block designing the broader predictor.

External response matrices such as the one released with EmbedLLM could supply
useful behavioral bridges. Before using them, map their source-task families,
remove overlap with reserved/held-out tasks, review label semantics and preserve
actual trial information. A public release or an embedding index does not make
held-out outcomes valid training features. Never import outside benchmark score
tables into profiles for the evaluation families or use public benchmark names
to infer anonymous competition identities.

## Status

Completed: primary-source research; reproducible training-only breadth audit;
this proposed training design. The existing task-v2 input builder, numeric
evidence control and one-epoch pilot remain available and unchanged.

Not completed: expanded-data subject admission, new profile collection,
embedding extraction, capability/difficulty model implementation, episodic
training, teacher distillation, encoder adaptation, or new quality results.
