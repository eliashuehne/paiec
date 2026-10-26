# PR target: a model-independent feature foundation

Status: implementation target; full-population scope explicitly approved by the user.
Date: 2026-10-25. Branch: `feature-foundation`, based on `catboost` at `6f78ac6`.
Proposed PR base: `catboost` (a stacked, feature-only PR). Targeting `main`
directly would include substantial pre-existing branch work outside this scope.

## Proposed title

**Add versioned feature tables for supplied settings, sourced enrichment and training-task observables**

## Problem and resulting behavior

Useful information is currently spread across raw supplied dictionaries,
Laya-specific preparation bundles, multiple profile collections and experimental
feature encoders. Some recorded settings are represented only by string lengths
and key counts. The full shared enrichment export contains functional facts that
older feature projections omit. Missingness, uncertain identity and capability
versus actual run configuration are easy to collapse during encoding.

This PR adds a reproducible, outcome-independent preparation layer. It keeps
supplied inputs, parsed observations, resolved settings, sourced facts and task
observables in separate linked tables. Every value has a definition and source;
unknowns, conflicts and unresolved parsing remain visible. A coverage report and
review ledger explain which information is usable and which gaps remain.

A later predictor can consume these tables without adopting Laya's rendering,
CatBoost's encodings or another model's assumptions. This PR makes no claim that
the features improve prediction.

The detailed rationale is in the
[approved feature design](superpowers/specs/2026-10-25-feature-inventory-design.md).

## Required scope and boundaries

The population is the **entire downloaded public release**, not the old Laya
sample or the scorer-admitted matrix. The pinned release has 61 benchmarks,
8,163 source subject rows, 1,356,713 source item rows and 16,423,758 recorded
response rows. Every source record is accounted for, including short/non-text
benchmarks, unsupported input formats and protected roles. The older 6,252-cell
matrix and 15,436-task pilot are comparison views only.

A versioned manifest assigns whole benchmark/source families to training,
validation and test using a deterministic seed, preserving every existing
frozen family role. Existing development maps to validation; calibration and
confirmation map to test with their original protected-role names retained.
Their contents/outcomes remain unopened: metadata/IDs account for their rows,
with explicit withheld-feature status. Other public validation/test inputs may
receive deterministic features without exposing outcomes or entering manual
review. Prior exposure is recorded; a new split is not called a pristine test.
Known source-family groups stay together. Exact normalized-content overlaps in
processed inputs are flagged; nontraining overlap blocks training use rather
than moving frozen assignments. Unmeasured overlap with protected content is
reported, not presumed absent.

Source response tables are projected to **non-outcome** linkage fields only:
subject/item IDs, response ID/trial and supplied interactors. Retain legitimate
trials and distinct interactor variants. Do not deserialize response labels,
grading rules, reference answers, test conditions or traces. Raw file hashes
may bind sources without interpreting their outcome columns. Missing foreign
keys and unsupported modalities remain counted with explicit dispositions.

Include all 46 model/base/family profiles, eight harness profiles and 189 catalog
configuration mappings in shared export v2; keep its 176-description preparation
view distinct. Preserve the eight supplied subject and four item fields. IDs,
roles, source benchmark names and hashes are audit/join metadata, not predictors.

**Outside this PR:** model fitting, outcome-derived abilities/difficulties,
revealed-label updates, scorer/submission repairs, embeddings, model/tokenizer
downloads, Laya changes, new web collection, annotation API calls, bulk semantic
annotation and GPU jobs. The legacy pilot is a compatibility check, never the
population definition or training eligibility gate.

## User-facing workflow

Add a `features` route to the existing canonical workflow command. Planning is
the default and must succeed from a source-only checkout without datasets,
weights, credentials, ZenML initialization or model imports:

```bash
PYTHONPATH=src .venv-pipeline/bin/python scripts/run_pipeline.py features \
  --config configs/pipelines/feature_foundation_v1.json \
  --out runs/feature_foundation/v1
```

Execution is explicit:

```bash
PYTHONPATH=src .venv-pipeline/bin/python scripts/run_pipeline.py features \
  --config configs/pipelines/feature_foundation_v1.json \
  --out runs/feature_foundation/v1 --execute
```

An optional `--source-root /absolute/path/to/original/checkout` lets an isolated
worktree read the pinned local artifacts without copying large data. The option
is valid only for `features`. Recipes contain portable relative source paths;
the receipt records the resolved physical root separately. Every declared file
is verified against the recipe's source hashes before use. Source paths are
allowlisted, not discovered by recursively scanning `runs/`.

A separate lightweight validator must work without the original datasets:

```bash
PYTHONPATH=src .venv-pipeline/bin/python scripts/validate_feature_foundation.py \
  --bundle runs/feature_foundation/v1
```

All commands above are proposed interfaces. They do not exist yet.

Execution uses the existing local ZenML/MLflow conventions through thin steps;
the extraction library and validator must remain callable without those optional
packages. No new model runtime or core dependency is needed. Never overwrite an
existing bundle, write into an input directory or follow a path outside the
declared input/output roots. A failed build must not publish a final manifest.

## Required output bundle

Large tables, task text and generated reports stay under ignored `runs/`.
The PR tracks code, schemas, policies, tests, compact source inventories and
review records; it does not commit downloaded/generated data.

| File | Key/grain | Required content |
| --- | --- | --- |
| `systems.parquet` | One `system_key` per exact visible subject dictionary | All eight raw subject fields; no display-name fallback or canonicalization that merges distinct inputs |
| `population.parquet` | One downloaded benchmark | Original row counts, split/family, frozen role, exposure, modalities/direction and processing dispositions |
| `subject_links.parquet` | `(benchmark, subject_id)` | All source subjects linked to exact visible dictionaries; audit-only source keys |
| `setting_observations.parquet` | `(system_key, source_field, source_path, occurrence)` | Raw keys/values, duplicate occurrences, container/path, parse disposition and canonical-field candidate; preserve unrecognized text |
| `run_settings.parquet` | `(system_key, field)` | One resolved value or explicit unknown/conflict for every declared run-setting field, with links to all supporting observations |
| `profile_matches.parquet` | `(system_key, entity_type)` | Profile ID or explicit unresolved status, identity basis, matching-policy version/hash, conflict reason and checkpoint-verification flag |
| `facts.jsonl` | `(profile_id, field)` | Full shared-export fact inventory, including unknowns; value/status, source evidence, review disposition, validity qualifiers and research eligibility |
| `catalog_mappings.jsonl` | One source catalog configuration | All 189 original mappings, with their original identities and qualifiers; distinct from the preparation view |
| `research_system_features.parquet` | `(system_key, field)` | Applicable fact values or explicit withholding reasons; matching basis and research-only status remain attached |
| `tasks.parquet` | One `task_key` per exact three-field visible task | Complete `item_content`, `item_features`, `interactors`; content/group hashes remain audit columns |
| `task_features.parquet` | One `task_key` | Deterministic observable features with feature/contract versions; semantic candidates are excluded from the default projection |
| `items.parquet`, `item_input_links.parquet` | All source items and their visible variants | Source item IDs, content or withheld status, task keys and interactor hashes; unsupported/withheld rows retained |
| `response_input_links.parquet` | One recorded response row | Non-outcome subject/item/trial/interactor linkage, original split and explicit orphan/withheld disposition; no targets |
| `sources.jsonl` | Namespaced source ID | Evidence registry needed to resolve fact/review references locally, with original-source hashes and eligibility preserved |
| `feature_dictionary.json` | One feature name/version | Definition, type/unit, grain, mechanism hypothesis, source fields, null/conflict semantics, availability, eligibility, quality status and implementation status |
| `coverage.json`, `report.html` | Explicit populations and feature families | Coverage, gaps, parsing quality, annotation audit and source scope; no predictive scores |
| `reviewed_examples.jsonl` | Stable review case ID | Source hashes/pointers, expected observations or facet assessments, reviewer/method and dispositions |
| `manifest.json`, `receipt.json` | Bundle/run | Schema/recipe/input/code hashes, logical table digests, artifact byte hashes, counts, output status and actions performed |

Use explicitly typed nullable Arrow schemas. Heterogeneous fact/setting values
use a declared `value_type` and canonical JSON strings in `value_json`; unknown
values have an Arrow null in `value_json` plus an explicit non-known status.
Do not rely on unstable Python object columns. A typed false, numeric zero or
known empty list is known and distinct from unknown. Canonical JSON must reject nonfinite numbers.
Output row order is deterministic by the declared keys.

## Settings that v1 must handle

The dictionary and parser cover, at minimum: `few_shot_count`,
`prompting_method`, `selection_method`, `reasoning_effort`, `max_input_tokens`,
`max_output_tokens`, `max_reasoning_tokens`, `max_turns`, `max_tool_calls`,
`max_retries`, `candidate_count`, `temperature`, `top_p`, `enabled_tools`,
`quantization` and `serving_configuration`.

- Reuse `state_rendering.split_settings` and the explicit settings-container
  decoding behavior through adapters; leave the existing hash-pinned renderer
  files unchanged. Quotes, nested JSON, escaped delimiters and duplicate keys
  must survive extraction. Decode only explicitly named configuration containers.
- A versioned interpretation policy lists recognized paths, aliases, types,
  units and source qualifications. `n_shots=5` can provide a shot count;
  `method=ICL_10` does not establish ten shots. Generic `max_tokens` requires
  established semantics before becoming an output limit. Reported task-generation
  size is not automatically a hard limit.
- Retain `source_model_label`, `source_model_identifier`, base/checkpoint names
  and adaptation metadata as reported observations. Parsing a name does not
  extend the approved identity policy or certify the fine-tuned target.
- Identical repeated claims may resolve to one value while preserving every
  occurrence. Contradictory claims yield `conflict` and no silently selected
  scalar. Dedicated fields and extras must be reconciled explicitly.
- Distinguish absent, malformed, unrecognized, conflicting and known values.
  Unknown settings and unreadable containers are normal recorded dispositions,
  not a reason to discard a system. Malformed source schemas or hash mismatches
  fail the build.
- Preserve full raw strings. Bound recursive decoding and resource use; reaching
  a limit produces a visible disposition and no silently trusted partial parse.

## Enrichment that v1 must expose

Use the **full shared v2 export**, including supplemented facts and review
decisions. Do not substitute the older renderer-selected field list.

At minimum expose identity/lineage, total and active parameters, parameterization,
documented languages, specialization/tuning domains, modalities, input/output/
combined-context limits, supported reasoning settings, harness workflow, tools,
context management and documented defaults. Preserve every other source fact in
the inventory even if its feature projection is deferred.

Matching uses only the permitted visible subject fields and the existing pinned
conservative policy. Facts joined through base-family relations cannot become
verified properties of an exact fine-tuned model. Reviewed unknowns/conflicts
take precedence over less-qualified catalog assertions; all original assertions
remain traceable. Historical applicability may be unknown and must be reported.
Supported tools, documented defaults and verified run settings have different
field names and must never overwrite each other.

Provide explicitly separate supplied-only and research-enriched projections.
The supplied-only projection cannot contain external values or features derived
from whether an external match was available. In the research projection,
external-source eligibility remains unreviewed/research-only by default.

As an optional legacy compatibility diagnostic, the unchanged matching policy should reproduce the old primary-scope
baseline: 225 model matches, 204 harness matches and 170 with both out of 1,936
cells. The full export contains known language facts reaching 88 cells and
context-management facts reaching 128 under this policy. These are source audit
checks, not mandatory feature-use approvals or predictive performance gates.

## Training-task observables and taxonomy review

Read projected subject/item tables for the full population and projected non-outcome response linkage. Never open protected content or outcome columns. Stream the population in bounded batches; the old Laya JSONL is optional compatibility evidence only. The small manual/taxonomy review sample comes from the declared training split only. Do not read the 23 GB trace data.

The deterministic task features include full character/word/line/paragraph
counts, numeric spans, Unicode script counts, code/math/table/option markers,
structured-content/message/section indicators, and presence of supplied item
metadata/interactors. Reuse existing extractors where their contracts match.
Heuristics have names ending in `_hint` or equivalent explicit quality metadata;
script presence is not certified language and an option count does not certify
the number of valid answers. Any clipping affects only a declared derived view;
complete visible fields and full-input lengths remain retained.

Keep the existing eight-facet semantic taxonomy: language, domain, skills,
reasoning structure, modalities, answer format, information needs and input
issues. Numeric task difficulty stays null. Do not create reference answers,
solutions, benchmark identity predictions or subjective model-strength ratings.

The initial audit contains **48 cases**: 24 system/settings cases and 24 training
task cases. Select training tasks with fixed seed `20261025`, using outcome-blind diversity in
source families, length quartiles, scripts and structural formats. Existing 14 candidates may be checked where they fall in the training split, but do not determine the sample population.
Select system cases to cover known/absent values, identity ambiguity, supported
versus enabled capabilities, duplicate/conflicting settings and unfamiliar
containers when present. Report any unavailable stratum; cover it synthetically
instead of inventing a real example. The selection algorithm and sample hashes
are versioned and reproducible.

Each review records source location/hash, expected extraction or facet decision,
supporting field/span, agreement/disagreement/unknown, defect class and reviewer
method. Do not commit full task text. Separate implementer review, agent review
and independent human/domain review. An agent-authored candidate cannot become
"human reviewed" or "independently validated" through a status rename. Unverified
language/domain judgments stay unknown. Report facet-level coverage, objections
and taxonomy changes needed; do not extrapolate sample agreement to the corpus.

Parser defects found in the sample must be fixed and tested, or withheld with an
explicit reason. Semantic labels remain candidates outside the default feature
projection until an independent quality gate is separately specified. Completing
this audit does not require approving every semantic annotation.

## Required report and dictionary

For every implemented feature, report known/unknown/conflict/invalid counts and
the precise denominator, grouped by benchmark, native direction, unique visible
system and cell population where applicable. Task coverage is reported against all source item rows, separately by training/validation/test and processing status. Protected content is explicitly withheld; unsupported input is separately partial/unsupported. Unique visible tasks, item rows and response trials use different denominators.

Separate identity-match coverage, fact availability, applicability, parse
coverage and feature inclusion. Show the full-export versus older-catalog gap
and why matched facts are withheld. Do not weight a system coverage percentage by
response rows without labeling that denominator explicitly.

The gap register has `gap_id`, affected feature/population, evidence, affected
count, cause, priority/rationale, next action, cost category, eligibility and
closure criterion. Causes distinguish missing source data, parser limits,
unresolved identity, unreviewed semantics, historical applicability and source
eligibility. Closure may be a justified unknown or deferral; it is not fabricated
completeness.

The dictionary also includes proposed-but-deferred compatibility, embedding and
behavioral features from the design. Their status is explicit; they must not
appear as all-zero implemented columns. The report prioritizes information
already collected before proposing new collection.

## Upstream reuse and expected diff

Port/adapt main's feature-table join checks and visible-subject conflict handling,
with source attribution in the implementation notes. Reuse the evidence-described
family grouping as audit metadata without redefining frozen folds. Do not run the
LightGBM feature-lift loop or import its model dependency into this library.

From logreg, reuse the separation of raw feature encoding and model state plus
relevant synthetic test cases. Its simple semicolon splitter, first size-token
parameter heuristic and episode/evaluator code are not drop-in implementations.
Retain the architecture-heavy current feature encoders for historical runs.

Expected code is a focused package under `src/paiec/feature_foundation/` with
contracts, source projection, settings, enrichment, task observables, joins,
reporting and build/validation modules; a thin `feature_foundation_steps.py`
workflow adapter; one pinned recipe; one settings interpretation policy; one
feature catalog; synthetic tests; a small review ledger and documentation.
Existing orchestration/index files receive only the additions needed to expose
the new workflow. Do not wholesale replace README, dependencies, folds, scorers,
predictor registries, existing recipes or hash-pinned renderers.

## Merge acceptance checklist

- [ ] A source-only plan lists exact populations, files, hashes and actions, with all model/API/outcome-access actions false.
- [ ] The pinned local build completes with the declared populations and validates all output schemas, keys, links and evidence references, including every population row or an explicit metadata-only withheld disposition.
- [ ] Every supplied field and encountered setting is retained and has a typed/literal/unresolved disposition.
- [ ] Unknown, known zero/false, conflict, unsupported applicability and absent evidence are distinguishable in tables and reports.
- [ ] Supplied-only and research-enriched projections obey separate provenance and availability policies.
- [ ] Every feature is defined, versioned, measurable and traceable; deferred features are identified explicitly.
- [ ] All 48 training-only audit cases have recorded dispositions; selection is reproducible and review limitations are stated.
- [ ] Rebuilding identical inputs produces identical logical tables and feature values. Nonvolatile logical digests match; timestamps, absolute roots and execution IDs may differ only in receipts.
- [ ] Synthetic outcome changes do not change feature values or logical feature digests; changed raw-source hashes may change provenance receipts.
- [ ] Tests prove no reads of protected content/outcome columns and no model loading, fitting, encoder inference, network access or job dispatch.
- [ ] Bad hashes, duplicate join keys, mismatched foreign keys, source/output overlap and incompatible schemas fail before final publication.
- [ ] Targeted feature-foundation tests and the existing source-only suite pass; existing native runtime/renderer/scorer behavior is unchanged.
- [ ] `git diff --check` passes. Generated data and unrelated facility-text edits are excluded from the PR.

## Evidence the eventual PR description must contain

State the before/after behavior, exact source and feature versions, population
counts, unresolved gaps and review method. Link a tracked compact audit receipt
with counts, schema versions, hashes and test commands; keep full local artifact
paths and ZenML/MLflow IDs in the execution receipt. Do not attach task text,
labels or a generated feature matrix to the main shared enrichment handoff.

Provide the actual build/validation commands, logical artifact digests and test
results. List any applicability/eligibility issues that prevent downstream use.
Explicitly state that no predictive quality claim is made. The PR is ready when
the foundation is reproducible and auditable, not when a model score improves.
