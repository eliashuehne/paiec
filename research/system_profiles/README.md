# Model, harness and run profiles

Versioned research data for the system-knowledge layer. The current researched
catalog is **collection_v3**; the original `v1/` contains subject-only seed
records. All collections retain schema **1.0**, with JSONL records and
Pydantic/JSON Schema validation. Collection numbers and schema versions are
independent; earlier collections remain immutable for old input contracts.
It is separate from prediction input rendering, frozen training exports, and
model checkpoints. Default prediction uses recorded fields only. The explicit
[`enriched_v1` experiment](../../docs/enriched-head-pilot.md) now adds selected
documented facts under a pinned policy, with the same builder at training and
prediction time. Eligibility remains research-only.

```
research/system_profiles/v1/
  profiles.jsonl        one model, harness or recorded-run profile per line
  sources.jsonl         one evidence source per line
  collection_plan.json field priorities and collection boundaries
src/paiec/system_profiles.py
configs/schemas/system-profile-v1.schema.json
configs/schemas/system-profile-source-v1.schema.json
scripts/validate_system_profiles.py
scripts/seed_system_profiles.py
```

All profiles share these top-level keys: `schema_version`, `profile_id`,
`entity_type`, `identity`, `valid_from`, `valid_until`, `specifications`, and
`usage`. The specification block is typed for model, harness or recorded run.
Names are descriptive stable IDs; they are not automatic identity matches or
aliases. An unresolved reported identifier must not be treated as an exact
checkpoint or merged with a similarly named model.

Every specification uses the same fact format, for example:

```json
{
  "value": null,
  "status": "unknown",
  "evidence": [],
  "note": "Not established from reviewed sources."
}
```

Known values use `recorded`, `documented` or `inferred` status and require
supporting evidence. Each evidence entry has `source_id`, `locator`, and
`excerpt`; its source must exist in sources.jsonl. A recorded run setting is
not a universal model capability or harness default. Inferred values must stay
identified as inferences. `null` means unknown, not zero, false or an empty
capability list. Values have strict types, and undocumented fields are rejected.

Architecture fields cover model family, backbone, attention direction, dense
versus mixture-of-experts parameterization, total and active parameter counts,
layers, hidden size, attention/KV heads, attention mechanisms, expert counts,
positional encoding and tokenizer ID. Collection priorities distinguish the
first useful pass from additional architectural detail. Predictive usefulness
has not yet been measured.

Models also cover input/output/context limits, modalities, reasoning controls,
model variant/base model, release information, tuning domains and documented
language coverage. Harnesses cover supported tools, answer workflows, context
management and documented execution defaults. Recorded runs cover what was
actually enabled, prompting, sampling, budgets, limits, candidate selection and
serving configuration, plus a verbatim copy of the supplied subject fields.
Run records reference model/harness profiles rather than duplicating their specs.

Sources record location, revision, retrieval date, optional content hash, and
eligibility (`inventory_source`, `unreviewed`, `not_permitted`) with a rationale.
Dates are represented as ISO dates; profile validity dates describe applicability,
not when a page was retrieved. New documentation must not silently be applied
to an older recorded model/harness version. Units are in field names: integer
parameters/tokens/counts, seconds for time limits.

## Current contents and collection status

Latest: [`collection_v3`](collection_v3/) preserves earlier data and adds primary
ERBench paper/code evidence for harness-scoped reported/base-family context.
It reviews all 19 development subjects alongside the 170 training candidates.
See [compact inputs and development](../../docs/compact-inputs-and-development.md)
for actual delivery, missing C-Eval identities, the separate question-only
supplement and the explicitly unbound private fine-tuned target profiles.

The user subsequently requested all **178 training/development candidates**.
The expanded [`collection_v2`](collection_v2/coverage.md) preserves this schema
and earlier catalogs. It supplies model facts for 169/178 and harness facts for
171/178; every system has at least one available, while 162 have both. This is
not 100% model-and-harness coverage. See
[coverage and input visibility](../../docs/system-profile-coverage.md) for the
nine unresolved model identities, seven missing harness names, actual token
visibility and the new opt-in catalog policy. No training was launched.

**The user subsequently authorized web collection.** Its first pass is saved
in [`collection_v1`](collection_v1/), using the same schema version 1.0:
18 model profiles, 8 harness profiles, and 32 recorded-run profiles covering
every exported training/development subject in `data_v2`. There are 42 external
source snapshots and 10 original subject-table sources. See the
[collection report](../../docs/system-profile-collection.md) for coverage,
architecture findings, unresolved identities and exact reproduction steps.
This does not cover the entire public corpus or the sealed evaluation splits.

`profiles.jsonl` and `sources.jsonl` are the homogeneous catalog. The companion
`subjects.jsonl` records matching decisions, candidates and verbatim published
fields; `recipe.json` and `assertions.jsonl` reproduce the reviewed extraction.
`snapshots.json` pins downloaded evidence by SHA-256. Full source documents
are archived under ignored `runs/system_profile_sources_20261022/`; they may
contain unrelated material and are never predictor input. External sources are
marked `unreviewed`, and all profiles remain `research_only`.

The original seed catalog below is preserved as a separate historical artifact.

Three **real seed profiles** are saved: the reported MetaMath-Mistral model,
reported ASPRM harness, and AdaptiveStep subject `50e726a69a130d06` connecting
them. Evidence comes only from the pinned original subject table. Its supplied
model identifier and harness name are recorded; the other capabilities and
architecture remain explicitly unknown. No architecture, parameter count,
context-window value, proprietary detail, or model-card claim has been guessed
from the model's name. No external web collection was performed for this seed.

This records the schema and collection plan plus grounded starting records;
the seeds themselves are not an enriched knowledge base. The collected external
metadata still needs the permitted-source question resolved for competition use.
Every profile has `usage: research_only`, including facts from inventory
sources. Saving a profile does not approve it as a training feature. No
benchmark scores, outcomes, grading definitions or subjective strength rankings
belong in this catalog. Per-task derived features belong in a later feature
builder with explicit estimation/provenance, not in static model specifications.

## Usage

```bash
.venv-laya-mlx/bin/python scripts/validate_system_profiles.py
.venv-laya-mlx/bin/python scripts/validate_system_profiles.py --catalog research/system_profiles/collection_v3
```

Validation checks types, unknown/evidence consistency, architecture counts,
entity/specification agreement, duplicate IDs, source references and run links.
To reproduce the seeds in a new directory:

```bash
.venv-laya-mlx/bin/python scripts/seed_system_profiles.py --out runs/NEW_PROFILE_SEED
```

The seed generator reads the original subject table, never outcomes or traces,
and refuses an existing output directory. Add reviewed facts by updating their
fact objects and source records, then run validation. Preserve unresolved and
conflicting version/source information in notes rather than inventing a value.
A schema change gets a new version and an explicit migration; do not change
field meanings under an existing version. No new training or model evaluation
has been run to create these files.
