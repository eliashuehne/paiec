# Feature Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish auditable, model-independent tables of supplied settings, full sourced enrichment and training-task observables with a feature dictionary, reviewed examples and a gap report.

**Architecture:** Pure extraction/validation modules produce linked typed tables and retain raw observations and provenance. Thin local workflow steps expose them through the existing plan-first orchestration; no model, label-based estimator or online service is involved.

**Tech Stack:** Python 3.12+, standard library, existing Pydantic/pandas/PyArrow dependencies; unittest fixtures; existing optional ZenML/MLflow workflow environment.

**Spec:** [Feature foundation PR target](../../feature-foundation-pr-target.md) and [source-grounded design](../specs/2026-10-25-feature-inventory-design.md).

## Global Constraints

- Branch `feature-foundation`, based on `catboost` at `6f78ac6`; no `codex/` prefix and no unrelated facility-text work.
- Version `feature_foundation_v1`: full downloaded release, 61 benchmarks, 8,163 subject rows, 1,356,713 item rows, 16,423,758 response rows. Old admitted-cell counts are compatibility views only.
- Full shared v2 scope: 46 model/base/family profiles, eight harness profiles, 189 catalog configurations; preserve the separate 176-description preparation view.
- All source items belong to the population. Versioned training/validation/test family assignments preserve frozen roles; protected content/outcomes stay unopened. Response reads project only non-outcome links and interactors.
- Preserve all eight raw subject and four item fields, null/empty semantics, target directions, source/hash contracts and frozen roles. IDs/hashes are audit keys, never default predictors.
- No fitting, embedding/tokenizer inference, web/API calls, cloud jobs, native model loading, scorer changes or submission changes.
- Full generated data belongs under ignored `runs/`; code/tests/configs/docs and compact reviewed/audit records belong in Git.
- Keep sourced facts, supported/default behavior and actual run settings distinct; external facts remain research-only.
- This plan is not implementation status. Commit steps below describe future implementation commits; this planning turn does not commit or open a PR.

## Review Focus

- Valid-looking nested or quoted settings with duplicates must preserve every observation and withhold contradictory resolved values.
- Model/base/family ambiguity and historical version uncertainty must not silently certify exact-target capabilities.
- Multiple systems with identical permitted descriptions and source IDs reused across benchmarks must not create false distinguishability or multiply joins.
- Worktree input roots, symlink escapes, changed input hashes and partial builds must not overwrite inputs or publish a complete manifest.
- Mixed-role legacy containers and apparently harmless target fields must never leak outcomes or excluded task content into extraction or review selection.

Each condition is pinned by an owning task below. An implementation check marked
complete requires its command and observed result, not an inferred pass.

## Proposed file structure

| File | Responsibility |
| --- | --- |
| `src/paiec/feature_foundation/__init__.py` | Minimal public exports; no optional-runtime imports |
| `src/paiec/feature_foundation/contracts.py` | Config, table schema registry, status/value contracts and canonical hashes |
| `src/paiec/feature_foundation/sources.py` | Hash-bound allowlisted reads and immediate visible-field projection |
| `src/paiec/feature_foundation/joins.py` | Safe keys, deduplication and cardinality-checked joins adapted from upstream |
| `src/paiec/feature_foundation/settings.py` | Raw setting observations, typed resolutions and conflicts |
| `src/paiec/feature_foundation/enrichment.py` | Full export facts, existing matching policy, applicability and source eligibility |
| `src/paiec/feature_foundation/tasks.py` | Deterministic training-task observables; candidate annotation linkage |
| `src/paiec/feature_foundation/review.py` | Deterministic case selection and review-ledger validation |
| `src/paiec/feature_foundation/report.py` | Coverage, dictionary, gap register and HTML rendering |
| `src/paiec/feature_foundation/build.py` | Assembly, deterministic writers, publication and bundle validation |
| `src/paiec/feature_foundation_steps.py` | Thin optional ZenML/MLflow graph |
| `scripts/validate_feature_foundation.py` | Dataset-independent output validator |
| `configs/pipelines/feature_foundation_v1.json` | Exact portable source paths, hashes, scope and versions |
| `configs/data/feature_foundation_settings_v1.json` | Path-aware settings interpretation rules |
| `configs/data/feature_foundation_catalog_v1.json` | Feature definitions, projection policy and implementation/deferred status |
| `research/feature_foundation/reviews_v1.jsonl` | Compact selected-case pointers, judgments and review provenance |
| `research/feature_foundation/source_inventory_v1.json` | Compact reproducible source/audit counts and versions; no source data |
| `tests/feature_foundation_fixtures.py` | Synthetic sources and tiny bundle fixtures |
| `tests/test_feature_foundation_*.py` | Top-level unittest modules, discovered by the existing command |
| `docs/feature-foundation.md` | Usage, schemas, review/coverage interpretation and limitations |

Modify `scripts/run_pipeline.py`, `src/paiec/pipeline_workflows.py` and relevant
workflow/index tests/docs only to add the `features` route and its source-root
option. Preserve the existing route defaults and project dependency metadata.

## Task 1: contracts, strict source projection and joins

**Files:** Create `contracts.py`, `sources.py`, `joins.py`, `__init__.py`, config/catalog skeletons, fixture helper, and `tests/test_feature_foundation_contracts.py`.

**Interfaces:**
- `FeatureFoundationConfig`: strict Pydantic config with `version`, `dataset_revision`, explicit source file paths/hashes, expected population counts, policy/catalog versions and review seed. Physical `source_root` is supplied separately.
- `SourceInventory`: dataclass with `systems`, `cells`, `training_items`, `training_links`, `profiles`, `catalog_mappings`, `sources`, `candidate_annotations`, `source_receipts`.
- `project_subject(subject: Mapping[str, str]) -> dict[str, str]` preserves exactly eight visible string fields; missing fields become empty, unexpected/non-string fields fail.
- `load_sources(config: FeatureFoundationConfig, source_root: Path) -> SourceInventory` verifies allowlisted inputs and immediately projects permitted state; no outcome values enter the return object.
- `safe_left_join(base: pd.DataFrame, features: pd.DataFrame, keys: tuple[str, ...]) -> pd.DataFrame` preserves base rows/order and rejects nonunique feature keys and column collisions.

- [ ] Write `test_raw_subject_fields_and_indistinguishable_inputs`: different private IDs/display names cannot change the same eight-field key; unknown raw columns are not silently adopted.
- [ ] Write `test_composite_source_keys_and_duplicate_join_rejection`: identical item IDs in two benchmarks stay distinct; duplicate feature keys fail and no join multiplies rows.
- [ ] Write `test_source_projection_never_reads_excluded_roles_or_targets`: spy on file opens and record access; only declared training state/cell subject fields reach downstream modules. Changing fixture targets affects receipts only.
- [ ] Write `test_input_hash_root_and_schema_fail_closed`: wrong hash, missing required source, non-string visible fields and root/symlink escape fail explicitly.
- [ ] Run `PYTHONPATH=src .venv-pipeline/bin/python -m unittest discover -s tests -p 'test_feature_foundation_contracts.py'`; observe the intended failures.
- [ ] Implement the contracts, projection and safe join. Attribute the adapted upstream behavior without importing LightGBM or editing the renderer.
- [ ] Rerun the same tests; all pass. Commit this coherent unit with its config/schema/tests.

## Task 2: recorded-setting observations and typed resolutions

**Files:** Create `settings.py`, settings interpretation policy and `tests/test_feature_foundation_settings.py`.

**Interfaces:**
- `SettingParse`: dataclass with `observations: list[dict]`, `resolved: list[dict]` and `diagnostics: dict` using Task 1's schema/status contracts.
- `parse_run_settings(subject: Mapping[str, str], policy: Mapping) -> SettingParse` returns all raw observations and one resolution per canonical setting. It does not change identity matching.

- [ ] Write a known-value fixture with `n_shots=5;temperature=0;max_retries=0;enabled_tools=[]`; assert typed 5, 0.0, 0 and an empty list remain known.
- [ ] Write quoted/nested/URL-container cases and duplicate `n_shots=5;n_shots=2`; assert occurrence retention and a null conflict resolution. Add a dedicated-field/extras conflict case.
- [ ] Write ambiguous `max_tokens`, `reported_task_generation_size` and `method=ICL_10` cases; assert no uncertified output limit or shot count.
- [ ] Write malformed JSON, duplicate JSON keys, unknown keys, nonfinite numerics and bounded-decoding cases; assert raw preservation and explicit disposition without trusted partial extraction.
- [ ] Run the settings test module and observe failures, then implement via adapters around existing parsing without editing pinned renderer sources.
- [ ] Rerun to pass; verify dictionary entries for all 16 canonical fields listed in the PR target. Commit.

## Task 3: full sourced enrichment with applicability and eligibility

**Files:** Create `enrichment.py` and `tests/test_feature_foundation_enrichment.py`; extend catalog definitions.

**Interfaces:**
- `resolve_enrichment(subjects: Sequence[Mapping], inventory: SourceInventory, policy: Mapping) -> dict[str, list[dict]]` returns `profile_matches`, `facts`, `catalog_mappings`, `research_system_features`, `sources`.
- `feature_projection(rows: Sequence[Mapping], mode: Literal['supplied_only', 'research']) -> list[dict]` returns only fields allowed by the declared projection, with its version and qualifiers.

- [ ] Write `test_full_export_fields_and_unknown_precedence`: a supplemented language/context fact is available while a reviewed unknown overrides an older documented assertion; original claims remain linked.
- [ ] Write `test_base_family_conflict_and_historical_qualification`: base architecture may carry a base qualifier; fine-tuning domains/languages are not certified as exact-target properties. Conflicting identity matches withhold values.
- [ ] Write `test_supported_default_and_enabled_are_separate`: documented tool support/default limits never fill actual run fields.
- [ ] Write `test_supplied_only_has_no_external_value_or_match_signal`: changing external catalogs cannot change this projection, including missingness flags.
- [ ] Write `test_all_source_refs_resolve_and_eligibility_is_preserved`: namespaces prevent registry collisions; unreviewed/denied sources are not promoted by serialization.
- [ ] Run and observe failures; implement, then rerun to pass. Keep the existing matching policy/aliases unchanged. Commit.

## Task 4: training-task observables and reproducible review cases

**Files:** Create `tasks.py`, `review.py`, review ledger, and `tests/test_feature_foundation_tasks.py` / `tests/test_feature_foundation_review.py`.

**Interfaces:**
- `task_observables(item: Mapping[str, str]) -> dict[str, int | float | bool | str | None]` consumes only the three visible content fields.
- `select_review_cases(inventory: SourceInventory, observations: Sequence[Mapping], task_features: Sequence[Mapping], *, seed: int = 20261025) -> list[dict]` returns 24 system and 24 task pointers, from the full training split rather than the old pilot.
- `validate_review_cases(cases: Sequence[Mapping], inventory: SourceInventory) -> dict` checks membership, hashes, supporting spans, dispositions and reviewer method.

- [ ] Write structure fixtures for Unicode, long text, messages/SQL JSON, options and missing fields; assert full lengths/raw preservation and explicitly named heuristic signals, with no language certification or numeric difficulty.
- [ ] Write `test_only_training_inputs_and_candidate_hashes_join`: all source links retain their role/grain in the real build; synthetic monitor/development records are not opened or selected.
- [ ] Write `test_review_selection_is_reproducible_and_outcome_invariant`: fixture target mutation leaves selection unchanged; insufficient/unavailable strata are reported.
- [ ] Write `test_review_provenance_and_quotes`: missing quotes, foreign task hashes, self-promoted review status and solution/difficulty fields fail validation.
- [ ] Run both modules to observe failures; implement extraction and selection, then rerun to pass.
- [ ] Audit the selected real 48 cases from allowed inputs; record expected parses/facet judgments, errors and abstentions. Correct extractor bugs with regression tests or withhold unsupported values. Keep unvalidated semantic candidates outside default features. Commit the module/tests and compact review ledger.

## Task 5: deterministic publication, coverage and validation

**Files:** Create `report.py`, `build.py`, validator script, and `tests/test_feature_foundation_bundle.py` / `tests/test_feature_foundation_report.py`.

**Interfaces:**
- `FeatureBundle`: dataclass with `directory: Path`, `manifest: dict`, `receipt: dict`.
- `build_bundle(config: FeatureFoundationConfig, *, source_root: Path, out: Path) -> FeatureBundle` publishes the required artifacts after schema/reference validation.
- `validate_bundle(path: Path) -> dict` checks hashes, schemas, grains, foreign keys, logical digests, projections and evidence references using bundle files only.
- `coverage_report(bundle: FeatureBundle) -> dict` derives denominator-specific known/unknown/conflict/withheld coverage, task scope, review status and prioritized gaps.

- [ ] Write `test_bundle_rebuild_has_same_logical_features`: two fresh output paths yield equal canonical logical digests; only allowed receipt metadata differs.
- [ ] Write `test_mutated_outcomes_change_only_source_provenance`: regenerated fixture input hashes validate, but feature values/digests and review selection stay unchanged.
- [ ] Write duplicate-key, orphan-source, omitted-field, invalid-null, partial-build, existing-output and input/output-overlap fixtures; assert rejection before final publication.
- [ ] Write `test_coverage_distinguishes_population_and_reason`: system matching, fact presence, applicability, extraction and inclusion use distinct counts; task populations outside training are `not_measured`.
- [ ] Write `test_candidate_dictionary_and_gap_register_complete`: every emitted feature has a definition; every deferred candidate has a status; every missing-data class has a next action or justified deferral. No numeric all-zero stand-ins for deferred families.
- [ ] Run targeted modules to observe failures; implement publication/reporting and standalone validation, then rerun to pass. Commit.

## Task 6: workflow entry, pinned build and PR evidence

**Files:** Create `feature_foundation_steps.py`, finalize the recipe, add workflow wiring/source-root option and `tests/test_feature_foundation_workflow.py`; add usage/index docs and compact source inventory.

**Interfaces:**
- `feature_foundation_preparation(root: str, config: dict, out: str, source_root: str)` is the thin optional local graph: verify sources, prepare tables, validate/publish.
- `workflow_plan(..., source_root: str | None = None)` accepts the new root only for `features`; existing routes preserve their public behavior.
- `scripts/run_pipeline.py features` defaults to plan, `--execute` explicitly runs; the standalone validator reads the completed bundle without original sources.

- [ ] Write plan-only tests with all data paths absent and network/model imports blocked; assert no execution and all prohibited action flags false.
- [ ] Write source-root and existing-route regression tests, including refusal of `--source-root` for other workflows and optional-dependency-free library import.
- [ ] Run the workflow tests to observe failures; implement the minimal route/graph and rerun to pass.
- [ ] Run all targeted tests with `PYTHONPATH=src .venv-pipeline/bin/python -m unittest discover -s tests -p 'test_feature_foundation_*.py'`.
- [ ] Execute the pinned local `features` workflow against the original artifact root into a fresh output; run standalone bundle validation and a second fresh build to compare logical digests. This is feature preparation, not a predictor experiment.
- [ ] Verify full 61-benchmark/8,163-subject/1,356,713-item/16,423,758-response accounting, all split/status dispositions, 46/eight profiles and 189 mappings. Report legacy matching only as a separately scoped compatibility check.
- [ ] Run the complete source-only suite and `git diff --check`. Confirm the actual diff excludes generated data, old renderer/scorer/recipe changes and unrelated unfinished work.
- [ ] Finish `docs/feature-foundation.md` and compact audit receipt with actual commands, counts, digests, 48-case dispositions and limits; update indexes. Commit.
- [ ] Prepare the actual PR title/body against the agreed base, using the target checklist and measured verification evidence; attach any created PR to the chat. Do not claim a quality gain or promote research eligibility.

## Self-review and execution handoff

Tasks 1–3 cover raw/source/run/fact contracts; Task 4 covers task scope and review;
Task 5 covers deterministic artifacts/dictionary/coverage/gaps; Task 6 covers
orchestration, real preparation, regression checks and delivery. The five Review
Focus conditions have named tests above. No new training recipe, data-role
assignment or predictive metric is introduced.

Recommended execution is native implementation in this isolated working copy,
using the available `executing-plans` skill, with the existing approved scope.
The current deliverable is the PR target and work plan. The proposed final PR
targets `catboost` so its diff contains this feature only. A PR directly to
`main` would include pre-existing CatBoost/Laya changes and requires a separate
repository integration decision before publication.

## Execution ruling: full population

The user explicitly superseded the old pilot-limited scope. Task 1 now owns full-corpus metadata inventory and a family-group split manifest preserving frozen roles; tasks 4–5 stream all permitted item inputs and non-outcome response linkage. Protected rows remain in the census with withheld status. No scorer admission threshold filters this population. The old pilot is optional compatibility evidence only. Source helpers and tests are adapted to this binding correction rather than retaining contradictory legacy constants.
