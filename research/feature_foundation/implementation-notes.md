# Feature foundation implementation decisions

This records the implementation choices and independent code-review disposition
for the full-population feature PR. Source counts and logical digests are in
[the measured audit](audit_v1.json); usage is in [the guide](../../docs/feature-foundation.md).

## Rulings made during execution

| Choice and reason | Cost or limitation |
| --- | --- |
| The user's whole-release instruction supersedes the legacy pilot/sample constants. Source helpers therefore stream full projected tables. | More local preparation time/storage; no sampled coverage claim can stand in for the population. |
| Family splits preserve frozen roles; calibration and confirmation are metadata-only protected test populations. | Their content features and overlap remain unmeasured. |
| Preserve the historical training-family monitor hash bucket at item level, in addition to whole-family assignments. | Training row proportions differ from a simple family-only 80/10/10 split. |
| Read response IDs, trial and supplied interactors for linkage; protected responses expose only IDs/trial. | No response-derived ability, target or behavioral feature is produced. |
| Preserve deterministic original source order, sorting variants/groups explicitly. | Consumers join by keys and cannot assume every table is globally key-sorted. |
| Reuse the existing external Python 3.12 pipeline environment. The plan's missing companion helper scripts are replaced by a local task/test ledger. | Validation depends on that recorded environment; the missing helpers supply no automatic bookkeeping. |
| Versioned Python contracts generate the settings policy and feature dictionary rather than maintaining duplicate policy JSON. Implementation and inherited-helper hashes are recorded. | Editing interpretation rules requires a code revision. |
| Sparse research tables are accompanied by explicit unmatched/missing-profile-field coverage, including original subject-row denominators per benchmark. | Additional report metadata; absent fact rows must still be interpreted through the coverage contract. |
| Seeded bounded training reservoirs and fixed length bands replace exact length-quartile balancing. Complete manual review input is capped at 8,192 characters. | The 48-case audit is not representative and cannot establish corpus semantic quality; longer tasks are extracted but outside this review. |
| Predictive usefulness and semantic annotation accuracy remain outside this preparation. Candidate task judgments are agent-only. | Downstream users must establish quality independently. |
| Inherited research facts keep their source/review/applicability qualifications; no new independent research was performed. | Even a sourced assertion may be inaccurate and is not competition-approved. |
| A fresh reviewer inspected the branch and synthetic reproductions; the implementer performs the final full-population rebuilds and standalone validation. | There is no second independent large-data execution. |
| The requested PR is opened against `catboost`; there is no extra integration menu because the user already chose a PR. | This is a stacked feature PR; unrelated base-branch integration remains separate. |

## Fresh review and one fix pass

One fresh-context reviewer found five Important issues and no Critical or Minor
issues. All five were reproduced before correction, then covered by regression
tests. The initial review said the branch needed these fixes; no second reviewer
verdict is claimed.

| Finding | Correction and regression |
| --- | --- |
| Nested duplicate/quoted claims could leave an outer scalar known. | Preserve duplicate JSON members, decode bounded quoted containers, and block trusted values on unresolved containers. `test_nested_duplicates_quoted_containers_and_unresolved_containers_block_false_certainty` |
| Alias names under examples/defaults/unrelated components became actual settings. | Explicit nested-branch policy; unsupported paths remain observations. `test_aliases_under_examples_defaults_and_other_components_do_not_establish_run_settings` |
| Mid-build input changes could publish stale source hashes. | Revalidate source hashes before publication; failed output keeps `INCOMPLETE` and no final manifest. `test_source_change_after_verification_cannot_publish_a_completed_bundle` |
| Optional enrichment could bypass file-level confinement/pins. | Every consumed export, policy and catalog file must be confined and hash-bound. `test_unpinned_or_escaping_optional_enrichment_is_refused` |
| Response variant joins were not independently checked. | Validate unique composite item/interactor variants and response references, retaining protected/orphan exceptions. `test_validator_checks_composite_response_variant_reference` |

The five finding groups first failed in 11 subcases. An additional typed-object
claim case also failed before its correction; it now rejects nested object values
without leaving a valid outer scalar certified as known. After the single fix pass, the full
source-only suite passed: 566 tests, including 18 optional-native skips, plus all
four profile-catalog validations and whitespace checks. No minors were deferred.
The final audit records the fresh-build and standalone-validation evidence.
