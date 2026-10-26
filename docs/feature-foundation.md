# Full-population feature foundation

The feature foundation inventories the whole downloaded public release and
exposes reusable inputs before choosing another predictive model. It separates
supplied system descriptions, recorded settings, researched facts and observable
task structure. It does not fit a model, inspect response outcomes, infer task
answers, or certify that any feature improves prediction.

The main shared research handoff remains [the collected system information](shared-data-start.md).
This preparation is a separate, local artifact. Its target and boundaries are in
[the PR target](feature-foundation-pr-target.md); measured counts and validation
evidence are in [the compact audit](../research/feature_foundation/audit_v1.json).
The [coverage and gap report](../research/feature_foundation/coverage-and-gaps-v1.md)
and [dictionary snapshot](../research/feature_foundation/feature_dictionary_v1.json)
are checked in as compact review metadata; large feature tables remain local.

## Population and splits

The recipe pins release `54074453da7b9911be0f48e4c560be547bc84d21`:

| Source unit | Count |
| --- | ---: |
| Benchmark directories | 61 |
| Subject rows | 8,163 |
| Item rows | 1,356,713 |
| Response trials | 16,423,758 |

These counts are different units. A response is a system's recorded trial on an
item; repeated trials remain separate. Exact visible task variants and system
descriptions are deduplicated feature entities linked back to every source row.
Neither the old 30,000-record training sample nor the scorer's admitted cells
defines this population. No minimum-response or higher-is-better filter applies.

Whole benchmark/source families receive a deterministic 80/10/10 assignment
unless a frozen role already determines their split. The actual assignment is
42 training, nine validation and ten test benchmark directories; unequal family
sizes mean row counts do not follow those percentages. Frozen training remains
training, development remains validation, and calibration/confirmation retain
their separate protected role names within test. The historical training-family
monitor hash bucket remains validation at item level.

Protected subjects/items retain IDs and census metadata only. Their fields,
task content and interactors are not opened. Other validation/test inputs can
receive deterministic features; their outcomes remain unopened and their inputs
do not enter the review sample. Prior public-data exposure is recorded, so this
partition is not a claim of a pristine test set.

Normalized-content groups flag exact overlaps across processed splits. Training
items in an overlapping group remain visible but ineligible. This is a
conservative text screen, not paraphrase/translation detection. Overlap with
protected content remains unmeasured.

The public source also records human participants and private/tuned artifacts.
For example, 1,125 visible dictionaries contain the literal `subject_kind=human`.
These rows stay in the population. An unresolved model-profile match can mean
non-applicability or an intentionally unbound private artifact. Item-level
`training_eligibility` does not decide which subject populations or native label
events should enter a later model-training recipe; that decision remains separate.

## Build and validate

Use the existing Python 3.12 pipeline environment. Planning needs no downloaded
data, tracking initialization, credentials, model weights or native runtime:

```bash
PYTHONPATH=src .venv-pipeline/bin/python scripts/run_pipeline.py features \
  --source-root /absolute/path/to/source-checkout \
  --out runs/feature_foundation/v1
```

Add `--execute` to run the local three-step ZenML graph. Omit `--source-root`
when the current checkout holds the source artifacts. The recipe is
`configs/pipelines/feature_foundation_v1.json`; paths in it are relative to the
physical source root. All declared hashes are verified before extraction. The
source release and full enrichment export must already exist; the builder does
not download them. A fresh checkout without those ignored sources can plan and
run synthetic tests but cannot reproduce the real bundle from Git alone.

Use a new output directory for each run. The builder refuses existing outputs
and overlap with immutable inputs. Interrupted/failed output retains an
`INCOMPLETE` marker and no accepted final manifest. Inspect or remove that
specific failed output before retrying with a fresh path.

The standalone validator requires the bundle, PyArrow and Pydantic; it does not
need original datasets, ZenML, MLflow or model libraries:

```bash
PYTHONPATH=src .venv-pipeline/bin/python scripts/validate_feature_foundation.py \
  --bundle runs/feature_foundation/v1 --require-review
```

`--require-review` requires completed dispositions for the selected audit cases.
It does not turn the agent audit into independent human or domain validation.
Reproduction compares `manifest.json`'s `logical_tables` between fresh builds;
timestamps and physical roots in receipts can differ. Byte and logical hashes,
explicit schemas, row accounting, references and protected boundaries are
checked independently after publication. Source outcome mutation tests prove
that changing fixture outcomes changes source hashes without changing features.

For embedding in another local preparation, the pure entry points are
`FeatureFoundationConfig`, `build_bundle(config, source_root=..., out=...)` and
`validate_bundle(path)` in `paiec.feature_foundation`. Tracking wraps this core.
MLflow receives only compact metadata/report artifacts; raw task text and large
tables stay in the local bundle. Tracking IDs are written to a sibling receipt.

## Linked tables and safe consumption

All tables use explicit nullable Arrow schemas and carry their version through
the bundle manifest. Original source order and seeded selection are stable.
Join metadata is separate from prediction fields.

| Tables | Grain and purpose |
| --- | --- |
| `population`, `subject_links`, `items` | Every source benchmark/subject/item, including protected and partial rows |
| `systems` | Exact eight-field visible subject dictionary, keyed by content hash |
| `setting_observations`, `run_settings` | Lossless setting claims and one resolved status/value per system/declared field |
| `profile_matches`, `research_system_features` | Qualified, research-only model/harness joins |
| `tasks`, `task_features` | Exact three-field visible task variant and deterministic observables |
| `item_input_links`, `response_input_links` | Item/interactor variants and every non-outcome response trial |
| `training_eligibility`, `content_groups` | Split eligibility and conservative overlap exclusions |

Table filenames have `.parquet` suffixes. Facts, evidence sources, 189 catalog
mappings and 176 preparation descriptions remain separate JSONL files. The
export includes all 46 model/base/family profiles and eight harness profiles,
plus hash-bound original provenance files. The 176 descriptions are not the
whole research collection.

A training input join starts from `training_eligibility` rows with
`eligible=true`, then follows item/variant and task keys. A response links to its
variant by `(item_key, interactor_key)` and to its subject by
`(benchmark, subject_id)`. Original response trial order is retained. Orphan
source references receive explicit dispositions; do not silently inner-join
them away. `safe_left_join` supplies strict composite many-to-one join checks
for consumers. IDs, hashes, benchmark names, split roles and eligibility are
audit/selection metadata, not default predictor features.

The supplied-only system projection is the eight raw fields plus parsed
`run_settings`; task features use the three published visible task strings.
Research matches, match availability and all research-derived values belong
only in a separately chosen research projection. `feature_projection(...,
"supplied_only")` excludes every external feature, including match signals.
This PR does not wire any projection into an existing predictor or change
Laya's hash-pinned rendering, scorer or submission package.

Response extraction explicitly projects only subject/item/response IDs, trial
and supplied interactors. No response value, test condition, grading rule,
reference answer or trace is deserialized. Benchmark response direction is
inventory metadata; native event meanings are never inverted or renamed.

## Settings and enrichment qualifications

`settings_policy.json` records policy `recorded_settings_v3`: 18 typed fields,
aliases, explicit containers and ambiguity rules. Full raw strings survive;
unfamiliar keys, duplicate occurrences, malformed containers and invalid claims
have observation rows. Known zero, false and an empty list are distinct from
unknown. Contradictory/invalid claims cannot silently select a trusted scalar.
Duplicate JSON members are separate observations; quoted containers decode with
bounds. Unresolved containers block trusted scalar resolution. Nested examples,
defaults and unrelated components remain literal observations; only the explicit
branch allowlist can establish run settings.

`released_configuration` values retain that scope; their presence is not an
independent verification of the historical invocation. Generic `selection`
retains an unspecified scope. Generic `method` is a reported label and never
establishes shots from a suffix. Generic `max_tokens` stays ambiguous. Supported
tools and documented defaults do not fill actual enabled tools or limits.

Research matching uses the existing conservative pinned policy. Base-family
relations expose qualified architecture only. Reviewed unknowns/rejections,
missing evidence and denied sources withhold values. Every joined fact keeps
its identity basis, source IDs, review disposition, research-only eligibility
and unverified historical applicability. No new aliases or web facts are added.

## Task observables and reviewed examples

Observables include full lengths, lines/paragraphs, numeric/script counts,
code/math/table/line-option hints, structured option/section/message counts,
visible asset-reference hints and presence of supplied metadata/interactors.
They do not certify language, skill, semantic difficulty or asset availability.
Full input strings are retained; counting does not tokenize, truncate or embed.
Non-text benchmark inputs remain `partial_nontext` because assets were not read.

The seeded audit contains 24 systems across harness strata and 24 training tasks
across benchmark, length bands, scripts and formats. Bounded reservoirs and an
8,192-character complete-input review cap keep inspection manageable; this is
not a representative corpus quality estimate. Longer tasks are fully extracted,
but outside this initial semantic audit. Synthetic fixtures cover duplicate,
conflict and malformed settings not guaranteed to occur in the small sample.

[The compact ledger](../research/feature_foundation/reviews_v1.jsonl) records all
48 input hashes, exact short evidence spans, checks and dispositions. The
implementation agent inspected them; there are zero independent human reviews.
Task facet judgments remain explicitly candidate-only, outside feature tables.
No task answer, solution or numeric difficulty was generated.

The audit found released settings and structured options that needed mechanical
extraction fixes, both covered by regression tests. It also found referenced
assets, role-prefix-only prompts, ambiguous execution contracts, mismatched
option instructions and native events that are not generic correctness. The
existing eight-facet taxonomy needs decisions on time-series inputs, data
analysis/multiple-artifact output and adversarial intent before bulk annotation.

## Coverage and next collection decisions

`feature_dictionary.json` defines emitted fields and deferred candidates with
type, grain, source, missingness, mechanism hypothesis, availability and quality.
Deferred semantic, compatibility, embedding and outcome-derived families have
explicit status and no all-zero stand-in columns.

`coverage.json` and `report.html` distinguish original rows, visible systems,
unique task variants and response trials. Every supplied source column receives
a read/withheld/excluded disposition in `source_fields.json`; `accesses.json`
records actual projected reads. Unsupported assets, protected roles, unknown
settings and unresolved identities remain in the declared denominator.

The first priorities are auditing already supplied identity/settings text and
documenting justified unknowns. Next are modality handling, independent semantic
quality policy, historical applicability and source eligibility. Those are
information-quality priorities, not measured feature importance. Broad new
research, embeddings and model comparisons remain separate decisions.

## Upstream reuse

The family groups and strict many-to-one join principles come from `main` at
`b202afd`. Composite source keys and exact visible-dictionary hashes preserve
cross-benchmark identity ambiguity and raw empty strings. The upstream subject
collector's global-ID/cleaned-string representation is not substituted for this
raw input contract. Existing Laya and CatBoost encodings remain available for
their historical runs.

The remote refresh on 2026-10-26 also found `features/subjects` at `dfe0091`.
Its new budget-k evaluation operates on revealed outcomes and fitted models;
it adds no source-only extraction required here. It was inspected, not run or
merged. The feature PR remains based on `catboost` at `6f78ac6` so its diff stays
focused.
