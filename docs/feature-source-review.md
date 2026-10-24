# Feature collection and source review

The user approved **complete source review with evidenced unknowns retained**
on 2026-10-23. This changes the acceptance rule, not the values: a reviewed
unknown stays null. Review coverage, known-value coverage and preparation
readiness are separate measurements. The scoped metadata review is complete;
task-feature collection and the remaining builders are still incomplete.

## Current scope and evidence

The current CLI default is
`configs/pipelines/feature_preparation_offline_v10.json`, selecting
`configs/pipelines/feature_sources_v6.json` plus
`research/feature_collection/identity_reviews_v1.jsonl`. It includes all systems present in
the selected training, monitor and development inputs. The task annotation
starter remains training-only. Calibration/confirmation are not opened.

The latest preparation uses 30,000 training records, 1,000 task-disjoint monitor
records and 963 original development records: 176 distinct permitted subject
descriptions, 39 selected model profiles, eight harness profiles and 16,909
distinct task inputs. Previous collection queues covered training systems only;
their smaller denominators must not be compared as if they covered this scope.

| Requirement | Current evidence | Incomplete work |
| --- | --- | --- |
| Static model/harness fields | All 617 scoped reviews complete; 213 raw values, 205 confirmed in review | 404 raw unknowns plus eight unconfirmed precision/taxonomy values |
| Actual run settings | 1,584 fields checked against published records; 149 values recovered | 1,435 remain unknown; public defaults cannot establish actual settings |
| Visible identity | All ten unresolved-identity requests reviewed against supplied descriptions | Ten stay unresolved; no private checkpoint or missing identity was invented |
| Task requirements | Full annotation request plan for 16,909 visible inputs | 14 v2 candidates, zero independent reviews; 16,895 pending |
| Demonstrated abilities/difficulty | Five-fold collection links prepared; no fitting | Actual empirical estimates and difficulty model remain incomplete |
| Compatibility/evidence | 170 training-only support/query link splits, without outcomes | Compatibility interactions, evidence materialization and quality remain incomplete |

The run-setting review is mechanical and limited to the published subject and
its exactly matching recorded-run profile. It does not establish that settings
are unavailable everywhere. Unrecognized settings containers or unresolved
explicit values remain pending; missing tools are never encoded as disabled.

## New collection and representation

Three additional primary launch pages cover Claude 3 and DBRX, alongside a
second extraction pass over the pinned open-model cards. Relative to the prior
two-bundle collection, 27 additional exact/typed facts are applied. Eight
nominal context claims are retained separately, as strings such as `128k`:
`qualified_model_observations.jsonl`. A rounded claim is not automatically
converted to an exact maximum token count, and is never a verified run budget.
The model fact table displays these under `qualifications.*`.

The current source registry applies 100 facts in total (97 model, three harness),
finds 39 already present and
retains four existing-value/taxonomy differences for review. It does not modify
old profile catalogs, policies, renderer sources or checkpoint contracts. PDF
is treated as a document format in the new assertion ledger; incompatible old
modality lists are retained as conflicts rather than silently overwritten.

Source reviews live in the `historical_v3`, `harness_v4`, `provider_review_v4` and `model_review_v5`
bundles under `research/feature_collection/`; schema:
`configs/schemas/source-review-v1.schema.json`. Each records the field's value
fingerprint, disposition, rationale, reviewer, checked source hashes and scope.
These are agent reviews, not independent human approval. A changed value makes
the review stale; changed archived source bytes fail validation. Negative
reviews establish only what was not found in those reviewed sources.

All eight harnesses now have their four requested fields reviewed. Eleven
additional harness observations preserve supported prompting modes, inference
backends, optional assistance and candidate handling without converting them to
actual run settings. Seven named provider models have 105 field reviews, plus
four newly sourced first-public-release dates. The final 28 model profiles add
420 field reviews and 35 sourced facts, completing all 617 static reviews.
Ten additional qualified model observations retain context, distillation,
language-count and document-format details without forcing them into incorrect
exact-value fields. This is a review of the cited primary metadata, not an
exhaustive proof of absence across all possible sources or independent human
approval of each statement.

There are 205 confirmed known values and eight reviewed precision/taxonomy
issues: four Claude context/output limits, two DeepSeek V4 context claims and
two Google input-modality lists. `value_confirmed` in `source_review_status.jsonl`
distinguishes these from confirmed values. The new
`reviewed_static_features.jsonl` contains the 205 confirmed values and 412 nulls;
unconfirmed raw values are not silently fed into the reviewed feature table.
The older profiles remain available separately. Review issues are distinct
from raw supplementation conflicts described above. This review does not
rewrite immutable historical profiles or certify a recorded run's limits.

New output document IDs include their hashes. Older catalogs and newer
snapshots with the same short source ID can coexist without rebinding evidence
to different bytes. All references in the reviewed/qualified output tables
resolve against the original and supplemented source ledgers.

## Reproduce

Restore all selected pinned archives before running. The original two bundles
use the commands in `research/feature_collection/README.md`. The historical
bundle additionally needs:

```bash
.venv-pipeline/bin/python scripts/fetch_system_profile_sources.py \
  --catalog research/feature_collection/historical_v3 \
  --cache runs/feature_sources/historical_v3
.venv-pipeline/bin/python scripts/fetch_system_profile_sources.py \
  --catalog research/feature_collection/harness_v4 \
  --cache runs/feature_sources/harness_v4
.venv-pipeline/bin/python scripts/fetch_system_profile_sources.py \
  --catalog research/feature_collection/provider_review_v4 \
  --cache runs/feature_sources/provider_review_v4
.venv-pipeline/bin/python scripts/fetch_system_profile_sources.py \
  --catalog research/feature_collection/model_review_v5 \
  --cache runs/feature_sources/model_review_v5
.venv-pipeline/bin/python scripts/run_feature_pipeline.py run
```

The open-model second pass reuses the original `primary_v1` archive. Source
pages pinned by content hash may change online; the downloader rejects changed
bytes instead of silently substituting them. Preserve the existing local
archive when a live page changes.

The complete metadata preparation is
`runs/feature_pipeline/complete_metadata_review_v1/`,
ZenML run `8723f162-be11-4885-aa71-8d14373de209`, MLflow run
`8088623992674c838ae78d9cab8c9a2e` (FINISHED). Its detailed review records,
supplemented model/source tables and qualified observations are verified in
MLflow artifacts. Checks: 271 unit tests passed (12 environment skips), all four
historical profile catalogs validated, and whitespace checks passed. Logs:
`runs/zenml/complete_metadata_review_checks.log` and
`runs/zenml/complete_metadata_review_pipeline.log`.
Every execution uses a fresh output directory; earlier outputs remain intact.

No model inference, empirical fitting, GPU job or training was performed for
this collection. Passing pipeline steps and source-integrity tests do not
establish semantic annotation quality or completion of the full six buckets.
