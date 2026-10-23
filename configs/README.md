# Configuration index

Local feature preparation: `pipelines/feature_preparation_v1.json` selects the
existing expanded bundle and unsealed roles for the preparation-only ZenML
pipeline. See [ZenML](../docs/zenml.md). It does not dispatch training.

Keep scientific configurations immutable once prepared or used. A changed recipe
gets a new descriptive version; do not rename historical artifacts merely to
make their number match the latest input format. None of the files below starts
a job by itself.

## Active shared settings

| Path | Purpose |
| --- | --- |
| `laya_question.json` | Shared target-system correctness question (`task_correctness_v1`) |
| `suites.json` | Public-data evaluation suites, caps and seeds |
| `execution/profiles.json` | Placement and resources; separate from the model runtime |
| `head_finetuning/frozen_split_v1.json` | Frozen family roles, dataset/export hashes and original input contract |
| `schemas/system-profile-v1.schema.json` | Version 1.0 model/harness/run profile schema |
| `schemas/system-profile-source-v1.schema.json` | Version 1.0 provenance-source schema |

## Head recipes and enrichment policies

| Path within `head_finetuning/` | Status and role |
| --- | --- |
| `compact_pilot_v3.json` | Current prepared control/enriched recipe, 2,048 tokens; not trained |
| `compact_profiles_v3.json` | Compact facts and scoped alias policy for collection-v3 |
| `task_features_pilot_v2.json` | Prepared task-v2 predecessor; not trained |
| `enrichment_catalog_v2.json` | Expanded collection-v2 research policy |
| `enrichment_pilot_v1.json` | Historical enriched pilot recipe, 1,024 tokens; completed cached enriched run uses it |
| `enrichment_v1.json` | Collection-v1 policy used by that earlier head |
| `smoke_v1.json` | Historical bounded 256-row, ten-update training check |
| `data_split_v1.json` | Reviewed source specification for the frozen family allocation |

The frozen split's original `status` and input settings describe its preparation
at that time; later experiments have separate records and recipes. Preserve
its contents and hashes. Sealed calibration/confirmation roles stay sealed.
See [the data split](../docs/head-data-split.md) and
[compact preparation](../docs/compact-inputs-and-development.md).

`experiments/input_design.json` records the earlier prompt/format experiment.
`benchmarks/*.json` are historical public-data research notes, retained for
provenance and tests. They are **not** an active benchmark-specific question
registry; runtime prediction uses the shared question and anonymous IDs.

## Version meanings

- Profile schema `1.0`: homogeneous record structure, independently versioned.
- `collection_v1`, `collection_v2`, `collection_v3`: research-data snapshots.
- `enriched_v1`, `task_v2`, `compact_v3`: distinct input contracts, with matching enriched variants where defined.
- Pilot recipe versions: fixed scientific settings, artifact paths and policies.

A training checkpoint bundles its input policy/catalog and renderer hashes.
Use those bundled files at inference; a newer policy must not silently replace
what an older head saw during training.
