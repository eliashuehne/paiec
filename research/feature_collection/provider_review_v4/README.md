# Named provider-model review

This bundle reviews 15 requested static fields for seven named API model
profiles: 105 reviews, using seven pinned model references plus four primary
release announcements. It adds four first-public-release dates. These are
different from access dates and from default or recorded snapshot dates.
No benchmark scores or undocumented proprietary parameter estimates are added.

Four Claude context/output-limit precision issues remain explicitly unresolved:
the references give rounded K/M values, while older extracted facts contain
decimal integers. The review does not confirm those integers as exact limits.
`source_review_status.jsonl` distinguishes a reviewed conflict from a confirmed
value; a completed review alone must not be used as confirmation of a value.

Other unknowns stay null with field-specific reasons and checked source hashes.
These agent reviews establish only what those primary references support;
they are not independent human reviews or verified settings of recorded runs.

Archive: `runs/feature_sources/provider_review_v4/`. Restore it with:

```bash
.venv-pipeline/bin/python scripts/fetch_system_profile_sources.py \
  --catalog research/feature_collection/provider_review_v4 \
  --cache runs/feature_sources/provider_review_v4
```

Live pages can change; the downloader requires the recorded content hashes.
