# Feature-source collection, 2026-10-23

Goal: full evidence-backed coverage of the six planned feature buckets.
This goal is **incomplete**. Unknowns are not counted as collected values.

Latest recipe: `configs/pipelines/feature_preparation_offline_v10.json`, with
the v6 source registry, identity-review ledger and all selected systems in its
review scope. All 617 static fields are reviewed; full task-feature collection
is still incomplete. See
[the source-review guide](../../docs/feature-source-review.md) for the current
617-field denominator, reviewed unknowns and qualified context observations.
The smaller counts below describe the earlier collection passes.

`source_plan_v1.json` identifies 18 named open-model repositories for metadata
collection. `sources.jsonl` and `snapshots.json` pin 31 retrieved text files by
repository revision and SHA-256. Four requests failed (DBRX metadata and three
gated Meta configuration files). No weights are downloaded or remote code run.
Run the collector into a fresh directory to inspect newer source revisions:

```bash
.venv-pipeline/bin/python scripts/collect_feature_sources.py --out runs/feature_sources/NEW
```

Reproduce the currently pinned source archive with the existing downloader:

```bash
.venv-pipeline/bin/python scripts/fetch_system_profile_sources.py \
  --catalog research/feature_collection --cache runs/feature_sources/primary_v1
.venv-pipeline/bin/python scripts/fetch_system_profile_sources.py \
  --catalog research/feature_collection/provider_v2 --cache runs/feature_sources/provider_v2
```

`assertions_v1.jsonl` holds 30 manually reviewed, quote-grounded statements.
Statements describe the named model documentation, not verified run settings.
The first pipeline collection applied 16 previously missing facts; 12 were
already present and two text-category differences were retained for review.
No existing catalog, policy or renderer was changed. Every added fact is
checked against archived source bytes and typed using the profile schema.
Source benchmark scores and marketing performance claims are not imported.

`provider_v2/` adds ten archived provider documents and 41 assertions for named
OpenAI, Anthropic and Google models. The registry at
`configs/pipelines/feature_sources_v1.json` selects both bundles. The combined
pass applies 31 new facts, finds 36 already present and retains four conflicts
for review. Provider pages are content-hash pinned; a changed live page cannot
silently replace the saved evidence.

The same pipeline now recovers 145 recorded settings by exact equality of
permitted subject descriptions against the original recorded-run profiles.
Private identifiers do not participate in that join. Conflicting visible
matches remain unknown. There are still 1,385 unestablished run fields and
454 missing static fields after both source bundles. These counts are fields,
not systems. The successful combined preparation is
`runs/feature_pipeline/provider_collection_v1`; successful execution does not
mean the collection goal is complete.

The v2 stage prepares all 16,909 task requests and 14 manual candidates, with
zero independent reviews. The user chose offline preparation; the teacher
pilot stays unsent. Candidate coverage is incomplete. The pipeline now prepares
training-only task-group folds and support/query links without reading labels.
Learned capability/difficulty, system–task compatibility, and outcome-conditioned
evidence remain incomplete. Collection
does not authorize a training job; calibration/confirmation stay sealed.
