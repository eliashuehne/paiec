# Collected model and harness information

For the Laya experiments, I researched the models, harnesses and recorded
configurations in our selected dataset systems. I collected architecture,
parameter counts, dates, documented limits, specialization and harness
capabilities, together with source evidence and notes about uncertain values.
These records can also support logistic regression or other approaches.

## Collection scope

The full catalog covers **189 selected configurations**: 170 training candidates
and 19 systems from the development families. It contains **46 model/base/family
profiles and eight harness profiles**. Model or base-family information is
available for 177 configurations, harness information for 182, and both for 170.
These are profile and coverage counts, not counts of verified exact checkpoints.
Some profiles describe candidates or base families; unknown identities remain
unresolved. The collection does not cover every system in the public corpus.

`system_enrichment_v2.zip` includes the full catalog and later supplemental
research, including material that was not used in Laya inputs. It replaces the
narrower v1 handoff. Earlier collection versions and source-review records are
preserved alongside the current views.

## Files

| File | Contents |
| --- | --- |
| `models.jsonl`, `harnesses.jsonl` | All 46 model/base/family profiles and eight harness profiles, with collected facts and evidence |
| `facts.csv` | Those facts in a table, including values, status, review decisions and source references |
| `catalog_systems.jsonl`, `catalog_runs.jsonl` | All 189 original configuration mappings and recorded-run profiles |
| `systems.jsonl`, `runs.jsonl` | The smaller preparation view: 176 distinct supplied descriptions and their typed run settings |
| `system_mapping.csv`, `run_settings.csv` | Tabular versions of that preparation view |
| `sources.jsonl`, `observations.jsonl` | Combined preparation source registry and qualified observations |
| `provenance/research/` | Complete versioned catalogs, supplemental assertions, observations, source registries, review ledgers and unresolved records |
| `provenance/source_tables/` | Original preparation tables; collection audits are retained under `provenance/runs/` |
| `manifest.json` | Counts and checksums for every exported file |

## Using the information

Profile IDs connect system mappings to model and harness records. The supplied
system fields are preserved so these mappings can be joined to the original
dataset. The 189 catalog rows and 176 distinct preparation descriptions use
different scopes and identifiers; they should not be matched by row order.

Each fact retains its value, status, evidence and qualifications. Reviewed facts
and reviewed unknowns take precedence in the combined profiles; other catalog
facts are marked `catalog_only`. Earlier or conflicting claims remain in the
versioned provenance, whose source IDs resolve within each bundle's own registry.
Unknown is not zero or disabled. A documented harness capability does not mean
it was enabled in a recorded run. Base-model facts do not establish the exact
properties of a private fine-tune.

For logistic regression, we could select model size, architecture, dates,
domain specialization or recorded settings, then choose our own encoding.
Questions and labels can come from the original dataset. This collection does
not contain question samples, outcome labels, feature matrices or trained weights.
Source URLs, excerpts and hashes are included; full downloaded web pages remain
in the local source archive. External-source competition eligibility is unreviewed.

## Access and reproduction

The archive, guide and checksum receipt are in the
[shared Drive folder](https://drive.google.com/drive/folders/1nFZn_2OwsX_RyV1vSA4fGg2395gfNS28).
JSONL files contain one JSON object per line; CSV files can be opened as tables.
Generated exports remain outside Git at `runs/shared_data/system_enrichment_v2/`.

Reproduce with `scripts/export_system_enrichment.py --config
configs/data/enrichment_handoff_v2.json`. The recipe pins all collection files
and packages existing data without new collection or training.
