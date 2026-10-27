# Remaining model metadata review

This bundle completes 420 field reviews across the remaining 28 selected model
profiles. Together with prior model/harness reviews, every one of the 617
requested static fields now has a scoped source review. This is **not** full
completion of task annotations or the six-bucket feature pipeline.

`sources.jsonl` and `snapshots.json` pin 73 primary metadata/launch documents.
Existing model cards/configuration snapshots were copied after hash verification;
four additional native release pages were downloaded. Files live under
`runs/feature_sources/model_review_v5/`, outside Git. No weights or model code
were executed. `profile_scopes_v1.json` records the checked document set per
profile, and `reviews_v1.jsonl` records every field-level decision and value hash.

`assertions_v1.jsonl` adds 35 facts, including explicit parameter counts,
release dates, language metadata, declared modalities and selectable thinking
modes. Names alone are not parameter counts. A teacher model is not a base
checkpoint; a repository update, training date or snapshot is not automatically
a release date. Examples/recommendations/configured positions are not actual run
budgets or hard output/reasoning limits.

Ten qualified observations preserve PDF input formats separately from sensory
modalities, the Gemini Flash distillation relationship, a published language
coverage count, and context qualifications. The 70B Code Llama context exclusion
is kept distinct from the broader original-family claim.

Reviews are by the agent, not independent human approval or an exhaustive proof
that a value is unavailable anywhere. Unknowns reflect the checked sources and
profile identity scope. Eight known raw values across all review bundles remain
unconfirmed because of precision/taxonomy issues. The pipeline's new
`reviewed_static_features.jsonl` retains only confirmed values and emits null
for these conflicts, while preserving original profiles and reviews separately.

New output evidence/source IDs include document hashes. This prevents an older
source and a later snapshot with the same short ID from becoming ambiguous.
Raw ledgers retain their local source IDs, resolved against their own source
catalogs and hashes; immutable historical catalogs remain unchanged.

Restore this archive with:

```bash
.venv-pipeline/bin/python scripts/fetch_system_profile_sources.py \
  --catalog research/feature_collection/model_review_v5 \
  --cache runs/feature_sources/model_review_v5
```
