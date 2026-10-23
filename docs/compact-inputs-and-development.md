# Compact inputs and development identity review

Completed 2026-10-22 as preparation, research and native-token validation. No
training, forward inference, Brier evaluation or GPU job ran. The original
evaluation and frozen roles are unchanged.

## What 100% means

The earlier collection reached 100% for **some model or harness context per
system**, not complete model coverage. Compact-v3 now delivers **every safely
available model/base-family and harness fact block** on the prepared corpus.
Missing identities and unknown flags never count as documented facts.

| Existing frozen records | Previous facts rendered | Compact facts available / delivered |
| --- | ---: | ---: |
| Training model/base-family | 3,185 | 4,584 / 4,584 |
| Training harness | 2,304 | 3,958 / 3,958 |
| Development model/base-family | 120 | 603 / 603 |
| Development harness | 963 | 963 / 963 |

All 5,000 training and 963 development records receive some sourced context.
Model identity coverage remains incomplete: 416 training records reference two
unverified private CDG checkpoints; 360 development records belong to three
C-Eval systems with no model identifier in the allowed subject fields. A larger
context window or a catalog lookup cannot recover an absent identifier. Audit
display names and private subject IDs remain excluded from prediction.

Model context includes explicitly labeled **base-family properties** for private
fine-tuned GPT variants. It does not establish their private weights, exact base
snapshot, tuning behavior or capabilities gained during tuning.

## Shared input design

`compact_inputs.py` constructs the control/enriched pair; `compact_profiles.py`
supplies documented facts and scoped alias evidence. `compact_predictor.py`
uses the same builder later. Native loading, prediction and caching remain
separate, and the shared correctness question is unchanged.

- Recorded model/harness identifiers, reasoning, tools, shots and output budgets
  take priority. Other supplied settings retain omission flags if too large.
  This corpus has zero system/item-attribute omissions.
- Compact facts give model and harness an initial complete fact before adding
  deeper dimensions. Sizes remain approximate; language-component counts stay
  labeled. Supported workflows are distinguished from tools actually enabled.
- Both arms reserve the same fixed **192-token** fact allocation and retain
  **identical task payloads**. The control contains recorded inputs only; its
  allocation does not encode the lengths of individual researched facts.
- Explicit questions and detected options get priority. Leading marked
  `Question: ... Context: ...` templates are split before budgeting, preserving
  the short question while shortening a huge retrieved passage. This uses
  supplied syntax/structure, not benchmark identities.
- Detected options are mandatory and kept whole. If options or essential system
  fields cannot fit, `InputCapacityError` stops that input. No native prefix clamp
  silently removes them. Large user messages/context still have explicit middle
  omissions; short surrounding instructions retain an allocation too.

The recipe uses **2,048 total tokens**, versus task-v2's 1,024. English only is
qualified. State capacity is measured after the unchanged native question head.
This combines formatting and context changes; no Brier gain is established.

Preparation finds zero native overflow, zero shortened detected options, no
additional task removal between experimental arms, and no identical effective
inputs crossing training/development roles. All 480 development records with
detected options retain them completely. There remain **1,145 training records
with task/context omissions**, including 652 with an oversized question/user
section. A user section can contain both a question and extensive background;
this counter does not certify which semantic information was lost.

## Development research and scoped aliases

Reviewed all 19 subjects in the existing C-Eval and ERBench families. C-Eval
provides a model identifier only for Qwen 7B. Its other ten subject rows have
indistinguishable allowed model metadata: three are original evaluation systems,
seven are additional candidates. None is reidentified from audit-only fields.

The [ERBench paper, model and fine-tuning sections](https://arxiv.org/html/2403.05266v3)
and [pinned published harness](https://github.com/DILAB-KAIST/ERBench/blob/2eca983dbe79d18cf3b2bf205114999764d061a3/binary/utils.py)
support the following reviewed **family** interpretations:

| Supplied alias | Context |
| --- | --- |
| `claude` | Claude 3 Sonnet, named in code |
| `gemini` | Gemini Pro text family, named in code |
| `gpt35` | GPT-3.5 family; private deployment snapshot unknown |
| `gpt4` | GPT-4 text API family; snapshot unknown |
| `llama` | Llama 2 70B Chat family |
| `mistral` | Mistral 7B Instruct family; version-specific attention/context not assumed |
| `gpt_finetune`, `gpt_finetune_2` | GPT-3.5 **base** family only; actual private tuned model unbound |

Rules require the supplied `harness=ERBench` and one matching `source_model`.
Conflicting identifiers/canonical names and explicitly different harness
versions fail closed. No global `gemini`, `llama` or `mistral` alias is added.
No study performance, reference answer or grading definition enters the state.
The paper's documented temperature is not promoted into a recorded run setting.

`collection_v3` preserves earlier collections and adds this research: 46
model/artifact/family profiles, eight harness profiles, 189 recorded systems
(170 training candidates plus all 19 development subjects), 88 source records
and 78 hashed snapshots. Private fine-tuned run profiles keep a null actual-model
link and a separate base-family link in the subject review. External eligibility
remains unreviewed. Proprietary sizes and unestablished specifications stay null.

## Separate outcome-blind supplement

Four additional identifiable ERBench systems are selected: Claude 3 Sonnet,
GPT-4, Mistral 7B Instruct and the other private GPT fine-tune's base family.
Selection uses subject metadata, item text and response linkage IDs. It reads
no response values, reference answers, grading definitions or traces.

Each has 120 evaluation and 40 acquisition system-item inputs, using the existing
50/50 split, fixed seed and disjoint pools: **480 evaluation / 160 acquisition**
inputs. These are not necessarily distinct questions across models. All have
matched control/enriched states and both model/base-family and harness facts.
The 80-linked-text-item threshold passes; binary-outcome scoring eligibility
has not yet been checked during this outcome-blind selection.

The supplement has no `expected` labels or scores and is not training data.
It remains separate from the original eight-system evaluation: 963 response
records / 960 system-item inputs. Original model/base-family coverage is now
603 records / 600 system-item inputs. Future supplemental scores must not replace
or masquerade as the original evaluation.

## Artifacts and use

- Recipe: `configs/head_finetuning/compact_pilot_v3.json`.
- Policy/aliases: `configs/head_finetuning/compact_profiles_v3.json`.
- Frozen preparation: `runs/head_finetuning/compact_pilot_v3_release/`.
- Question-only supplement: `runs/head_finetuning/development_identity_review_v3_release_final/`.
- Combined report: `runs/head_finetuning/compact_workflow_v3/report.html`.
- Durable research/review summaries: `research/system_profiles/collection_v3/`.

```bash
.venv-laya-mlx/bin/python scripts/prepare_compact_pilot.py --config configs/head_finetuning/compact_pilot_v3.json
.venv-laya-mlx/bin/python scripts/review_development_profiles.py --render-inputs --out runs/NEW_DEVELOPMENT_REVIEW
.venv-laya/bin/python scripts/run_head_pilot.py --config configs/head_finetuning/compact_pilot_v3.json
```

Preparation needs a fresh output folder; the saved recipe is already prepared.
The last command is **preflight only** without `--train`. Both arms plan one
epoch / 79 updates from the original pinned parent; neither has been launched.
Earlier GPU speed/cache qualification covered 1,024 tokens; the new 2,048-token
recipe needs fresh resource/performance qualification before an actual GPU job.

Prediction uses `paiec.predictor_factory.make_laya_predictor` with `compact_v3`
or `compact_enriched_v3`. Both checkpoint variants bundle their policy/catalog
because the control needs the same reservation contract. Policy, catalog,
question, context and renderer hashes are checked before weight loading. Both
evaluation and execution-job interfaces expose these formats. Old defaults,
catalogs, pinned renderer files, v1/v2 preparations and the completed old head
remain valid. Earlier untrained compact working previews are superseded; use
the `release` preparation only.

Both lightweight suites pass **235 tests** (MLX nine expected skips, Torch one).
Native PyTorch preflight verifies every prepared token sequence, label alignment,
paired rendering and frozen manifest without loading weights or running inference.
