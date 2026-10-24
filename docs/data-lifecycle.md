# Data lifecycle

The authoritative role/lifecycle catalog is
`configs/data/data_versions_v1.json`. It lists principal retained datasets,
not every historical report. Original sources, checkpoint contracts, frozen
exports, sealed roles, tracking stores and research provenance are protected.
Keep historical scientific recipe and renderer paths unchanged. Laya work is
on hold; retained artifacts support reproducibility, not automatic continuation.
See [current status](current-status.md) and [shared data](shared-data-start.md).

| Data version under `runs/` | Use |
| --- | --- |
| `shared_data/system_enrichment_v1` | Main shared collection: model/harness facts, run settings, sources and system mappings |
| `shared_data/statistical_starter_v1` | Retained statistical experiment artifact; not the primary handoff |
| `grouped_enriched/prepared_v1/tokens` | Full grouped training tokens and exact collected-fact snapshot |
| `workflows/grouped_enriched_training_v1/job/output/final` | Completed fitted checkpoint; preserve hashes and native configuration |
| `workflows/grouped_development_v1/scores` | Completed negative result and saved predictions; no need to rerun inference |
| `head_finetuning/data_v2` | Original frozen exports and development provenance |
| `head_finetuning/enriched_30k_v1` | Research-enriched 30k Laya inputs and token preparation |
| `head_finetuning/compact_pilot_v3_release` | Retained compact-5k inputs used by the completed checkpoint |
| `competition_sources/protocol_panel_v1` | Outcome-blind training-only augmentation; 34,718 records |
| `competition_pipeline/protocol_panel_v2` | Dataset-only tables and training targets for all 170 scopes |
| `feature_pipeline/offline_evidence_preparation_v1` | Research tables and collection plans; not a model input contract |
| `head_finetuning/development_identity_review_v3_release_final` | Question-only supplement; outcomes remain unused |

Version numbers count different contracts, not one interchangeable sequence.
A later dataset-only bundle cannot replace research-enriched inputs expected by
an existing Laya checkpoint. There is not yet a newly qualified competition-only
Laya training recipe. No such follow-up is planned; model-independent reuse
starts from the dataset-only bundle, without changing historical contracts.

## Retired working previews

The old `compact_pilot_v3`, `compact_pilot_v3_ready`, and
`compact_pilot_v3_final` folders were explicitly marked **untrained, superseded
input-development previews**. No current recipe or checkpoint contract referenced
those exact paths. They were retired into separate compressed archives under
`runs/data_retirement/workflow_migration_v1/`. Every archived file was checked
byte for byte before its original directory was removed. Receipts retain original
paths, file hashes, sizes, archive hashes and the retained release path.

No checkpoint, source table, completed experiment, old research catalog or frozen
export was deleted. Historical tracked configurations remain because past runs
and checkpoint/source hashes can depend on them. Other large old folders should
not be removed merely because a newer number exists.

## Audit and restore

```bash
.venv-pipeline/bin/python scripts/manage_data.py audit
.venv-pipeline/bin/python scripts/manage_data.py restore \
  --archive runs/data_retirement/workflow_migration_v1/compact_pilot_v3
```

Restoration verifies archive/file hashes, rejects unsafe members, and never
overwrites a data version. `retire` accepts only an exact catalog entry marked
`retirable_preview`, requires its superseded marker and a retained replacement,
checks recipes/checkpoint references, and writes a verified archive before
removal. Protected entries cannot be retired with this tool.

Datasets, archives, checkpoints and generated reports remain ignored by Git.
A fresh checkout contains the catalog and machinery; restore/download the
required local artifacts before executing a workflow.
