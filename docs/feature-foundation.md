# Feature foundation

Build model-independent feature tables over the entire downloaded public release.
Supplied inputs, recorded settings, task measurements and researched facts remain
separate and linked. Preparation does not train a model or read response outcomes.

## Build and validate

Use Python 3.12 with the [pipeline environment](zenml.md). The default command
plans the workflow without loading data or initializing tracking:

```bash
PYTHONPATH=src .venv-pipeline/bin/python scripts/run_pipeline.py features \
  --source-root /absolute/path/to/source-checkout \
  --out runs/feature_foundation/v1
```

Add `--execute` to build through ZenML and MLflow. The recipe is
`configs/pipelines/feature_foundation_v1.json`; its source paths are relative to
`--source-root`. Omit that option when the current checkout holds the sources.
The pinned public release and shared enrichment export must already exist;
preparation does not download them. Generated tables stay outside Git.

Use a fresh output directory. Inputs must match the declared hashes and cannot
overlap outputs. Failed builds retain `INCOMPLETE` without an accepted final
manifest. Validate a completed bundle independently:

```bash
PYTHONPATH=src .venv-pipeline/bin/python scripts/validate_feature_foundation.py \
  --bundle runs/feature_foundation/v1 --require-review
```

The validator needs PyArrow and Pydantic, but no original data, tracking or model
libraries. `--require-review` checks that the selected audit cases have completed
dispositions; it does not establish independent human review. Compare
`manifest.json` → `logical_tables` between fresh builds for reproducibility;
receipt timestamps and physical paths can differ.

The Python API is `FeatureFoundationConfig`, `build_bundle(config,
source_root=..., out=...)` and `validate_bundle(path)` under
`paiec.feature_foundation`. Tracking wraps this core; only compact metadata and
reports enter MLflow.

## Tables

All 14 tables use explicit nullable Arrow schemas and `.parquet` filenames.
Source order is retained; consumers must join by keys rather than row position.

| Table | One row represents |
| --- | --- |
| `population` | One downloaded benchmark collection and its counts, roles and support status |
| `subject_links` | One original subject within a benchmark, linked to its visible system description |
| `systems` | One distinct combination of the eight supplied system fields |
| `setting_observations` | One raw setting occurrence, with its key, value and source location |
| `run_settings` | One of 18 declared settings per visible system, including unknown/invalid/conflict status |
| `items` | One original item within a benchmark, with its split and processing status |
| `tasks` | One exact visible input: content, supplied item features and interactors |
| `task_features` | Deterministic measurements of one visible task input |
| `item_input_links` | One source item and distinct interactor variant, linked to its task |
| `response_input_links` | One original response trial, retaining IDs and links without its outcome |
| `content_groups` | One normalized-content group and the splits where it occurs |
| `training_eligibility` | One item/input variant's permission for training and the reason |
| `profile_matches` | One model or harness matching disposition per visible system |
| `research_system_features` | One available, qualified research-profile field per matched system |

Supporting JSONL files retain facts, evidence sources, catalog mappings and
preparation descriptions. The full enrichment collection contains 46 model/base/
family profiles, eight harness profiles and 189 catalog mappings. Its separate
176-description preparation view is not the full collection. The main shared
handoff remains [the system-enrichment collection](shared-data-start.md).

## Join and input contracts

Start a training-input join from `training_eligibility` where `eligible=true`,
then follow item/variant and task keys. Responses link to input variants by
`(item_key, interactor_key)` and to subjects by `(benchmark, subject_id)`.
`safe_left_join` checks composite many-to-one joins. Preserve explicit orphan
and protected dispositions instead of silently dropping them with inner joins.

The supplied-only inputs are the eight raw subject fields, parsed run settings
and features from the three visible task strings. IDs, hashes, benchmark names,
splits and eligibility are selection/audit metadata, not predictor features.
Research values and match-availability signals belong only in a separately
chosen research projection; `feature_projection(..., "supplied_only")`
excludes them. Existing predictors, renderers and scorers are unchanged.

## Interpretation rules

- **Settings:** `recorded_settings_v3` preserves raw strings, unfamiliar keys,
  duplicate occurrences and source paths. Known zero, false and empty lists
  differ from unknown. Conflicting, malformed or unresolved containers cannot
  silently establish a trusted scalar. Only allowlisted nested branches can
  establish settings; examples, defaults and unrelated components remain raw
  observations. Generic `method`, `selection` and `max_tokens` retain their
  ambiguity. Released configurations are not independently verified invocations.
- **Research:** matching uses the pinned conservative identity policy. Each fact
  retains its source IDs, identity basis, review disposition, research-only
  eligibility and unverified historical applicability. Base-family matches do
  not certify an exact checkpoint. Rejected, denied or unsupported facts withhold
  values; documentation defaults do not fill actual run settings.
- **Tasks:** measurements cover full text lengths, lines, scripts, structural
  hints, supplied metadata and asset references. They do not certify language,
  skills, difficulty or asset contents. Non-text inputs remain partial. Semantic
  annotations, embeddings and outcome-derived features are not materialized.

## Population and protected roles

Release `54074453da7b9911be0f48e4c560be547bc84d21` contains 61 benchmark
directories, 8,163 subject rows, 1,356,713 item rows and 16,423,758 response
trials. No scorer threshold or legacy sample limits this inventory. Item rows,
exact visible inputs and repeated response trials are different units.

Whole benchmark/source families receive seeded 80/10/10 assignments unless a
frozen role determines their split. The result is 42 training, nine validation
and ten test collections; unequal collection sizes and the preserved historical
item-level monitor bucket mean row counts do not follow those percentages.
Frozen development maps to validation; calibration and confirmation retain their
protected roles within test. Protected rows expose IDs and census metadata only.
Their content, interactors and overlap remain unexamined.

Exact normalized-content overlap blocks affected training inputs without moving
frozen assignments. This does not detect paraphrases or establish pristine test
data. Human participants and private/tuned artifacts remain in the population;
unresolved model matches are not necessarily missing research. Item eligibility
does not select a future subject population or reinterpret native outcome labels.
No response outcome, test condition, grading rule, reference answer or trace is
read into the feature layer.

## Reference and quality evidence

- [Feature dictionary](../research/feature_foundation/feature_dictionary_v1.json):
  definitions, types, source, missingness and implemented/deferred status.
- [Coverage and gaps](../research/feature_foundation/coverage-and-gaps-v1.md):
  full-population denominators, supported inputs and prioritized information gaps.
- [Audit](../research/feature_foundation/audit_v1.json) and
  [source inventory](../research/feature_foundation/source_inventory_v1.json):
  source versions, logical table hashes and validation receipts.
- [Review ledger](../research/feature_foundation/reviews_v1.jsonl): 24 system
  and 24 training-task cases, with evidence spans and dispositions. These are
  agent reviews, with zero independent human reviews. The seeded sample caps
  complete reviewed inputs at 8,192 characters; longer inputs are extracted but
  outside this review. Candidate facets do not establish semantic accuracy.

Generated bundles also include `source_fields.json` (every column's disposition),
`accesses.json` (actual projected reads), `settings_policy.json`, `coverage.json`
and `report.html`. Feature usefulness and competition eligibility of external
research remain unestablished.
