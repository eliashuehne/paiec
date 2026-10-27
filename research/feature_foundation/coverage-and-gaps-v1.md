# Feature coverage and next information gaps

This is a source-quality audit of release `54074453da7b9911be0f48e4c560be547bc84d21`, not a model result. The full [machine-readable audit](audit_v1.json) records all logical hashes, source versions and verification evidence.

| Original unit | Count |
| --- | --- |
| benchmarks | 61 |
| items | 1,356,713 |
| responses | 16,423,758 |
| subjects | 8,163 |

All 61 downloaded benchmark directories are included. The 60 trace files are explicitly excluded from feature reads. Protected content and every response outcome remain unopened. Full processing does not imply support for all modalities.

| Item disposition | Original item rows |
| --- | --- |
| partial_nontext | 114,817 |
| supported_text | 1,144,822 |
| withheld_protected | 97,074 |

## Splits and training use

| Split | Original item rows |
| --- | --- |
| test | 325,803 |
| training | 947,646 |
| validation | 83,264 |

| Training-use disposition | Item/input variants |
| --- | --- |
| cross_split_content | 85 |
| eligible | 862,754 |
| nontraining_split | 281,983 |
| partial_nontext | 114,969 |
| withheld_protected | 97,074 |

Variants include distinct supplied interactors, so their denominator differs from original item rows. The 692,987 exact visible task variants receive deterministic observables; semantic facets remain unmaterialized. Prior exposure and protected content prevent calling these new splits pristine.

## Recorded settings

Denominator: 7,160 distinct processed visible system dictionaries, one slot per field. Known means a qualified recorded claim, not independent verification of the historical run. Zero, false and empty lists are known; ambiguous or unsupported paths cannot establish actual settings.

| Field | Known | Unknown | Invalid | Conflict |
| --- | --- | --- | --- | --- |
| candidate_count | 0 | 7149 | 11 | 0 |
| enabled_tools | 0 | 7149 | 11 | 0 |
| few_shot_count | 122 | 7027 | 11 | 0 |
| max_input_tokens | 3 | 7146 | 11 | 0 |
| max_output_tokens | 85 | 7064 | 11 | 0 |
| max_reasoning_tokens | 0 | 7149 | 11 | 0 |
| max_retries | 0 | 7149 | 11 | 0 |
| max_tool_calls | 0 | 7149 | 11 | 0 |
| max_turns | 0 | 7149 | 11 | 0 |
| prompting_method | 34 | 7115 | 11 | 0 |
| quantization | 0 | 7149 | 11 | 0 |
| reasoning_effort | 14 | 7135 | 11 | 0 |
| reported_method | 11 | 7138 | 11 | 0 |
| sampling_enabled | 74 | 7075 | 11 | 0 |
| selection_method | 2 | 7147 | 11 | 0 |
| serving_configuration | 0 | 7149 | 11 | 0 |
| temperature | 31 | 7067 | 62 | 0 |
| top_p | 9 | 7066 | 85 | 0 |

The frequent untyped keys below remain literal, with source paths and raw values retained. Many are identity or task-specific configuration rather than missing scalar run settings; their counts do not establish parsing defects.

| Raw key | Distinct visible systems |
| --- | --- |
| model_identifier | 5024 |
| experiment_name | 4278 |
| initialised_from | 4278 |
| intervention | 4278 |
| training_step | 4278 |
| subject_kind | 1126 |
| education | 1125 |
| education_description | 1125 |
| gender | 1125 |
| source_participant | 1125 |
| task_cohort | 1125 |
| geographic_region | 1117 |
| age | 1088 |
| programming_experience | 1087 |
| socioeconomic_status | 1087 |
| source_model_identifier | 418 |

## Researched facts

All 46 model/base/family profiles, eight harness profiles and 189 configuration mappings are retained; the 176-description preparation view remains separate. Full collection scope is different from matching coverage.

| Entity | Matched systems | Unresolved systems |
| --- | --- | --- |
| harness | 212 | 6948 |
| model | 247 | 6913 |

Per-field research coverage, missing-profile-field slots and withholding reasons are in generated `coverage.json`, including language, specialization, context and limits. Every value remains research-only and historically unverified; base-family matches do not certify exact-target capabilities or actual enabled settings.

The supplied `subject_kind=human` marker occurs on 1,125 visible dictionaries. Private/tuned artifacts also remain unbound. An unresolved public-model match is not automatically a missing research fact: distinguish non-applicability and private identities before extending aliases. Training eligibility here is at item/input level; future label-bearing recipes must separately define subject populations and native event meanings. All source rows remain in this inventory.

## Task taxonomy audit

All 48 selected examples (24 systems, 24 tasks) have dispositions. They are implementation-agent reviews, with zero independent human reviews and no established semantic accuracy. Longer complete inputs remain outside this bounded initial review. Candidate facet counts describe only the 24 reviewed tasks.

| Facet | Candidate | Unknown | Validated |
| --- | --- | --- | --- |
| answer_format | 13 | 11 | 0 |
| domain | 13 | 11 | 0 |
| information_needs | 11 | 13 | 0 |
| input_issues | 15 | 9 | 0 |
| language | 1 | 23 | 0 |
| modalities | 1 | 23 | 0 |
| reasoning_structure | 0 | 24 | 0 |
| skills | 2 | 22 | 0 |

The audit flags time-series assets, multi-artifact data-analysis outputs, adversarial intent, role-prefix-only prompts and ambiguous execution contracts. These need a separately reviewed annotation policy; no task answers or numeric difficulty were generated.

## Prioritized gap register

Priorities reflect source reach and contract completeness, not measured feature importance. A justified unknown or deferral can close an accounting gap.

| Priority | Cause | Affected population | Count | Next action |
| --- | --- | --- | --- | --- |
| P0 | unresolved_identity | distinct_processed_visible_systems | 6913 | Review supplied reported/base identifiers and applicability before extending aliases |
| P0 | absent_ambiguous_or_unparsed_settings | system_setting_cells | 128495 | Review frequent raw keys and containers; distinguish missing source data from interpretation gaps |
| P0 | protected_role | original_item_rows | 97074 | Keep metadata-only census and frozen roles |
| P1 | unsupported_nontext_assets | original_item_rows | 114817 | Design modality handling separately; current text counts are partial |
| P0 | cross_split_normalized_content | item_input_variants | 85 | Keep blocked training rows visible and review grouping before future modeling |
| P1 | unreviewed_semantic_annotations | unique_processed_visible_tasks | 692987 | Use the small training audit to refine the eight-facet taxonomy before any bulk annotation |
| P1 | unverified_historical_applicability | known_research_feature_cells | 2117 | Bind facts to recorded versions where evidence permits; retain other cases as qualified |
| P1 | external_source_eligibility_unreviewed | research_profiles | 54 | Resolve competition-source permission separately; keep research view segregated |

## Verification

The standalone main-branch preparation reproduces all 14 logical table hashes,
schemas and row counts from the original feature foundation. All 256 tests pass
on Python 3.10 and 3.12. The [audit](audit_v1.json) records current code hashes and
validation evidence. No model was trained or evaluated by preparation.
