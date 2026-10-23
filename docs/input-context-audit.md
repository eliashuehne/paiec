# Audit: which system and task context can Laya receive?

## Clarification: identity availability versus enrichment coverage

Rechecked 2026-10-22 after the user asked whether every record should identify
both the LLM and harness. All **32 selected system profiles** match their
original subject-table permitted fields exactly; the exporter is not dropping
recorded identities. The published interface has the keys in every dictionary,
but allows empty values. A present key does not guarantee an established value.

| Source information in frozen pilot exports | Training (5,000 rows) | Development (963 rows) |
| --- | ---: | ---: |
| Nonempty harness name | 3,958 (79.2%) | 963 (100%) |
| Nonempty canonical model name | 4,583 (91.7%) | 120 (12.5%) |
| Canonical name **or** recorded model/base/checkpoint identifier in extras | 5,000 (100%) | 603 (62.6%) |

The last row counts recorded identifier strings, including incomplete/private
aliases. It does **not** establish an exact model/version or a safe external
profile match. Three selected C-Eval systems (360 development rows) have neither
a canonical model name nor a usable identifier in supplied extras. Their
source-table display names are outside the published predictor interface.

The earlier **28.6% training / 12.5% development** figures count rendered
**additional model-property blocks**, not model-name availability. Such a block
requires a resolved documented fact in the selected research catalog and spare
input capacity. Missing blocks can reflect unresolved identity, catalog/fact
coverage or token limits; they do not necessarily mean a missing model name.
An available harness name likewise does not imply known enabled tools or limits.

These counts are response-row weighted for the frozen pilot. The whole-corpus
subject-row percentages below use a different denominator and must not be
substituted for this sample's coverage.

2026-10-22. The user requested inspection of the original data behind our
training states, with particular attention to model identity, harness details,
and reasoning effort. This is a data/input audit, not a new prompt experiment
or model-quality result. No training or inference was performed.

## What was inspected

Read all 61 downloaded `subjects.parquet` tables at pinned measurement-db
revision `54074453da7b9911be0f48e4c560be547bc84d21`; inspected the schemas of
subjects/items/responses/benchmarks through Parquet metadata. Checked the
original AdaptiveStep subject and item behind smoke example 151. Compared every
nonempty published subject field with the rendered state in all 256 smoke rows,
5,000 training rows, and 963 development response rows. All comparisons matched.
Existing labeled exports were loaded for their input states; their labels were
not used. No trace tables, response outcome columns, or confirmation exports
were loaded.

Reproduction: `scripts/audit_input_context.py --out NEW_DIRECTORY` using the MLX
Python environment. Saved audit: `runs/input_context_audit_v1/`, with coverage
CSV files, schemas, source table hashes, original examples, preservation checks,
and `report.html`.

## Availability is uneven

These are **8,163 subject rows**, not 8,163 distinct LLMs. They include repeated
models/configurations across benchmarks and non-LLM/human subjects in parts of
the public inventory. The percentages are subject-row weighted, not response
weighted and not representative of the hidden test pool.

| Recorded attribute | Nonempty rows | Present |
| --- | ---: | ---: |
| Harness | 8,120 | 99.47% |
| Canonical model name | 5,350 | 65.54% |
| Provider | 5,350 | 65.54% |
| Model release date | 3,303 | 40.46% |
| Access date | 78 | 0.96% |
| Harness version | 500 | 6.13% |
| Dedicated reasoning effort | 14 | 0.17% |
| Extra system settings | 8,125 | 99.53% |

Dedicated reasoning effort appears in chi_bench (5 subjects), editbench (6), and
engdesign (3). Extra settings may also contain reasoning-related options; the
14 count describes the dedicated column only. No smoke-training, full-training,
or selected development row has a nonempty dedicated reasoning-effort field.

The normalized-name column is present in 221/256 smoke rows, 4,583/5,000 full
training rows, and only 120/963 selected development rows. This does not mean
model identity is absent everywhere else: some extras carry usable model names.
It does mean a missing canonical name cannot be treated as a reliable complete
system description.

## The short AdaptiveStep example is already complete for its published fields

Original subject `50e726a69a130d06` has:

- `display_name`: `MetaMath-Mistral` (outside the predictor interface).
- `normalized_name`, provider, dates, harness version, reasoning effort: null.
- `harness`: `ASPRM`.
- `subject_features_extra`: `model_identifier=MetaMath-Mistral`.

The exported state contains both nonempty published subject fields verbatim.
The item contains the original question and its complete recorded item_features.
There is no missing recorded reasoning setting to append to this particular
example. The other original item columns contain identifiers, reference grading
and verifier details; the latter are not extra task context for prediction.

A clearer *proposed* rendering of the same information is saved as
`proposed_adaptivestep_state.txt`. It labels MetaMath-Mistral as a **recorded model
identifier from supplied extras**, without pretending a canonical model name,
provider or reasoning effort was supplied. It adds no facts, has not been scored,
and is not the active renderer.

## Useful information is often inside the extra fields

Actual recorded settings include:

- `base_model=Llama-3.1-8B-Instruct;method=ICL_10` (Fewshot-TTT).
- `n_shots=5;selection=minimum_perplexity` (C-Eval, plus a source-study field).
- `prompting=cot` (CRUXEval, with other supplied extras).
- `max_tokens=4096` (Alpha-SQL, with other supplied extras).
- `reported_task_generation_size=512` versus `32768`, checkpoint identifiers and
  reported model paths (Critic Discernment Game).
- URL-encoded JSON `agent_configuration` with model IDs, harness kwargs,
  reasoning settings, retry/turn limits and other configuration (Chi-Bench).

These values are already passed if their fields are present. The current JSON
renderer leaves semicolon settings and URL-encoded JSON as opaque strings.
The earlier `text_v1` experiment expands JSON-valued strings, but does not fully
parse semicolon-separated fields or URL-encoded nested configurations. A focused
extra-settings renderer would therefore test a distinct change. Previous
readable-text experiments did not show a score improvement; readability alone
must not be reported as a measured gain.

## A genuine identity gap in the original interface

Three selected C-Eval models have names `baichuan2-7b-base-hf`, `Yi-34B-200k`,
and `llama_7b` in display_name, but all three normalized_name values are null.
Their complete published system profiles are identical: OpenCompass with
`n_shots=5;selection=minimum_perplexity;source_study=arXiv:2310.17589v3`.
Their other published system fields are empty.

For the same task at budget zero, a predictor using this interface necessarily
receives identical inputs for these three systems. Formatting cannot restore
the missing identity. Full examples are saved in
`ceval_missing_identity_example.json` for auditing only; display_name is not
passed to the predictor. This is a candidate metadata-quality question for the
organizers, not a reason to add a local display-name fallback.

## Boundary and proposed next work

The current [published predictor interface](https://aims.stanford.edu/competition#competition)
provides eight subject fields and four item fields, with empty strings for
missing values. Our renderer preserves every nonempty permitted subject field;
it does not arbitrarily select only a model and harness. It also preserves the
supplied item text/features/interactors, while omitting the anonymous benchmark
ID from text as previously agreed.

Other raw columns do exist. For example, Critic Discernment Game response rows
have `test_condition=temperature=0`. `test_condition` is not a published input
field. Likewise display_name, asset_manifest, reference grading/verifiers and
benchmark descriptions are not supplied as dedicated predictor fields. Adding
those during local training would create features unavailable to the submitted
predictor. Settings already appearing in the permitted extras remain usable.

Recommended next input experiment:

1. Decode valid URL-encoded JSON and render recorded model identifiers,
   base models/checkpoints, harness settings, few-shot settings, token limits,
   reasoning effort and tools under clear labels. Preserve their source and
   precise values; do not infer missing identities or settings.
2. Preserve the remaining supplied extras. Validate any parser against quoted
   values, embedded semicolons, malformed JSON and literal encoded strings;
   avoid silently dropping unfamiliar fields or treating task text as code.
3. Audit retained tokens for the full training set. The 54.22% truncation rate
   means adding prose can remove task information. Keep model/harness details
   and the actual question/options visible, and compare a compact rendering
   with the existing one using identical samples and a distinct format version.
4. Evaluate on the existing development role. Leave the frozen exports and
   historical checkpoint/results intact; version any new training inputs.

No active renderer, data split, checkpoint, prediction input boundary, or
historical score was changed by this audit. No organizer message was sent.


## Subsequent implementation

The user subsequently authorized implementation. `context_v1` is available as
an explicit state format; see [benchmark-questions.md](benchmark-questions.md)
for parser behavior, usage and measured token visibility. The audit findings
above preceded this implementation; the current default remains JSON.
