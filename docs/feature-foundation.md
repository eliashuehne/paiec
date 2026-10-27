# Feature foundation

Prepare model-independent features for the entire downloaded public release.
The command preserves supplied inputs and separates recorded settings, task
measurements and researched facts. It does not train models or read outcomes.
This is the first public v1 of the tables, settings policy and lift
export. Source archive names and evidence revisions retain their original names.

## Build and validate

Install the optional preparation dependency:

```bash
pip install -e '.[features]'
```

For development tests, install `.[dev,features]` and run `python -m pytest`.

The default command prints a plan without opening source data or creating output:

```bash
python scripts/prepare_feature_foundation.py \
  --source-root /path/to/source-data \
  --out runs/features/foundation_v1
```

Add `--execute` to build. The source root must contain the pinned public release
under `payload/data/` and the extracted `system_enrichment_v2` collection under
`runs/shared_data/system_enrichment_v2/`, including its provenance files. These
are existing external inputs, not files included in a fresh checkout. The
command does not download data or require a model checkpoint, ZenML or MLflow.

`configs/features/foundation_v1.json` pins source hashes, reviewed matching rules,
family groups and protected roles. Use a fresh output directory. Changed inputs,
escaping paths and input/output overlap are refused. Failed builds retain an
`INCOMPLETE` marker without an accepted final manifest.

Validate a completed bundle without the original source data:

```bash
python scripts/validate_feature_foundation.py \
  --bundle runs/features/foundation_v1 --require-review
```

`--require-review` requires dispositions for the selected review cases; it does
not establish independent human review. Compare the `logical_tables` entries in
`manifest.json` between preparations to check logical reproducibility. Physical
Parquet bytes can differ with library versions.

## Tables and joins

All 14 tables have explicit nullable Arrow schemas and `.parquet` filenames.

| Table | One row represents |
| --- | --- |
| `population` | A source benchmark collection with counts, roles and input support |
| `subject_links` | An original subject within a benchmark, linked to its system description |
| `systems` | A distinct combination of the eight supplied subject fields |
| `setting_observations` | A raw setting occurrence and its source location |
| `run_settings` | A declared setting per visible system, including unknown/invalid/conflict status |
| `items` | An original item, including split and processing status |
| `tasks` | An exact visible input: content, item features and interactors |
| `task_features` | Deterministic measurements of a visible task input |
| `item_input_links` | An item/interactor variant linked to its task |
| `response_input_links` | An original response trial's IDs and input links, without its outcome |
| `content_groups` | A normalized-content group and the splits where it occurs |
| `training_eligibility` | An item/input variant's training eligibility and reason |
| `profile_matches` | A model or harness matching disposition per visible system |
| `research_system_features` | An available, qualified research field per matched system |

For a recipe using the frozen preparation roles, start a training-input join
from `training_eligibility` where `eligible=true`.
Follow item/variant and task keys; response inputs join by
`(item_key, interactor_key)` and subjects by `(benchmark, subject_id)`.
`safe_left_join` checks composite many-to-one joins. Preserve orphan/protected
statuses rather than silently dropping them. IDs, hashes, benchmark names and
split roles are audit/selection metadata, not default prediction features.

Supporting JSONL files preserve facts, sources, catalog mappings and preparation
descriptions. The existing collection contains 46 model/base/family profiles,
eight harness profiles and 189 catalog mappings; its 176-description preparation
view is separate. Unmatched profiles remain in the collection.

## Interpretation and boundaries

- Settings retain raw values, duplicate claims and source paths. Zero, false and
  empty lists differ from unknown. Unfamiliar or ambiguous fields are not guessed.
  Only explicit allowed configuration branches establish settings; examples,
  defaults and unrelated components remain observations. Released configurations
  do not independently verify a historical invocation.
- Research stays research-only. Each value keeps its source, identity basis,
  review status and unverified historical applicability. Base-family matches do
  not establish exact checkpoint capabilities. Supplied-only projections exclude
  research values and match-availability signals.
- Task features measure full visible text and structural hints. They do not
  certify language, skill, difficulty or asset contents. Non-text inputs remain
  partial; semantic annotations and embeddings are not materialized.
- The frozen release has 61 collections, 8,163 subject rows, 1,356,713 item rows
  and 16,423,758 response trials. Every source row is counted, including human
  participants, private artifacts, unsupported inputs and protected populations.
  Repeated trials and distinct visible input variants are different units.
- Family assignments preserve frozen training/development roles, the historical
  training-family monitor bucket and protected calibration/confirmation roles.
  Protected content and interactors stay unopened. Exact visible-content overlap
  blocks affected training inputs; it does not detect paraphrases or prove a
  pristine test set. Subject selection and native label meanings remain separate
  decisions for a later training recipe. These roles do not replace the lift
  runner’s own folds or sealed holdout; consumers must apply those boundaries too.
- Response outcomes, grading rules, reference answers, test conditions and traces
  are not read. Existing scoring, subject-feature and prediction logic is unchanged.

## Use with the lift loop

Keep the [existing lift sample and subject-feature builders](lift.md#quick-start).
Export a supplied-only view for that sample after validating the bundle:

```bash
python scripts/lift/build_foundation_features.py \
  --bundle runs/features/foundation_v1 --sample runs/lift/sample.parquet \
  --out runs/features/foundation_lift_v1 --execute
```

Omit `--execute` to print the plan without opening inputs. Export uses a fresh
directory and reads only the sample's `benchmark`, `subject_id`, `item_id` and
`trial` columns. It checks the consumed bundle artifacts against their manifest.
The separate full validator checks all tables and relationships.

The resulting `features.parquet` works alongside the existing subject features:

```bash
python scripts/lift/run_lift.py --base KM \
  --features runs/features/subjects.parquet runs/features/foundation_lift_v1/features.parquet \
  --columns recorded_few_shot_count task_content_characters --name recorded_and_task
```

This second command runs a predictive experiment; export alone does not. No
improvement has been measured. Select columns explicitly: `recorded_*` contains
18 settings with separate `*_status` columns; `task_*` contains the 19 task
measurements. Known zero, false and empty lists stay distinct from unknown.
Lists use their JSON text as categorical values. The adjacent dictionary records
types and definitions; coverage records missing/ambiguous/protected counts, and
the manifest pins the sample, consumed bundle artifacts, code and outputs.

Task resolution uses each exact subject/item/trial's recorded interactor variant.
Conflicting variants at the same key produce null task features. Identical
repeated trials share one feature row and remain separate scoring rows. Protected
inputs remain null; the export never reads research or adds its availability as
a feature. Missing-input reasons stay in the coverage report, outside model columns.

The export neither selects rows using preparation splits nor changes the lift
sample. The lift loop remains responsible for folds, sealed holdout, acquisition
budgets and in-fold estimates. Its identity/family/date builders remain the
source of those features; research profile matching has a different purpose.

## Reference

The bundle includes `feature_dictionary.json`, `coverage.json`, `report.html`,
`source_fields.json`, `accesses.json`, `settings_policy.json`, source/code hashes
and a manifest. Compact checked-in references are the
[dictionary](../research/feature_foundation/feature_dictionary_v1.json),
[coverage and gaps](../research/feature_foundation/coverage-and-gaps-v1.md),
[audit](../research/feature_foundation/audit_v1.json) and
[review ledger](../research/feature_foundation/reviews_v1.jsonl).

The 48 review cases are agent audits, with zero independent human reviews.
Complete reviewed inputs are capped at 8,192 characters; longer inputs are
extracted but outside this review. Semantic accuracy, predictive improvement
and competition eligibility of external research are not established.
