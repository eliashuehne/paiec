# Collected model and harness information

For the Laya experiments, I collected additional information about the models,
harnesses and recorded configurations in the public dataset. This includes
architecture, parameter counts, dates, documented limits and harness
capabilities where I could find supporting sources.

I’m sharing those records so we can choose which information to use in other
approaches. The handoff contains the collected facts and their sources, rather
than a feature matrix or a particular model’s training inputs.

## What is included

`system_enrichment_v1.zip` contains information for **39 model/base profiles,
eight harness profiles and 176 recorded system descriptions**. This covers the
systems reviewed for these experiments, rather than the entire public dataset.
Some identities and fields remain unresolved.

| File | Contents |
| --- | --- |
| `models.jsonl` | Model/base identities and collected facts, including architecture, capacity and documented limits |
| `harnesses.jsonl` | Harness identities and documented capabilities |
| `runs.jsonl` | Settings recovered from the recorded system descriptions |
| `systems.jsonl` | Original supplied descriptions and their mappings to model/harness profiles |
| `observations.jsonl` | Additional sourced observations, such as nominal context claims or optional tool support |
| `sources.jsonl` | Source URLs, revisions, retrieval dates and source hashes |
| `facts.csv` | The model/harness facts in a table that can be inspected or filtered |
| `run_settings.csv`, `system_mapping.csv` | Tabular versions of run settings and system mappings |
| `provenance/` | Original source tables and review records, preserved for reference |

There are no question samples, outcome labels, feature vectors or trained
weights in this bundle. The earlier statistical starter is a separate experiment
artifact, not the collection handoff.

## How to use the records

The `recorded` object in `systems.jsonl` contains the eight supplied system
fields. Matching that description connects an original dataset system to its
`system_key`, `model_profile_id` and `harness_profile_id`. Those profile IDs join
to `models.jsonl` and `harnesses.jsonl`; `system_key` joins to `runs.jsonl`.

The mapping also records whether the model identification is exact, refers to
a base/family, or remains unresolved. A documented base-model property is not
necessarily verified for the evaluated fine-tune.

Each fact keeps its value, status, source evidence and notes. `unknown` values
remain null. Inferred values are marked as inferred. The export distinguishes
scoped source reviews from additional previously documented facts. Harness
support is separate from tools/settings recorded as enabled in a run.

For logistic regression, we could select fields such as model size, dates,
architecture or recorded settings and decide how to encode them. Another model
could use a different representation of the same records. The choice of
features and preprocessing is left to each approach.

## Access

The archive, this guide and a checksum receipt are in the
[shared Drive folder](https://drive.google.com/drive/folders/1nFZn_2OwsX_RyV1vSA4fGg2395gfNS28).
The local export is `runs/shared_data/system_enrichment_v1/`; generated data is
not stored in Git. JSONL files have one record per line and can be read with
Python’s standard library; the CSV files can be opened directly as tables.

```python
import json
from pathlib import Path

folder = Path("system_enrichment_v1")  # after extracting the ZIP
models = {r["profile_id"]: r for r in
          map(json.loads, (folder / "models.jsonl").read_text().splitlines())}
systems = list(map(json.loads, (folder / "systems.jsonl").read_text().splitlines()))
model_id = systems[0]["resolution"]["model_profile_id"]
model = models.get(model_id)  # None when no model profile is established
```

The export recipe is `configs/data/enrichment_handoff_v1.json`; it packages
existing reviewed tables using `scripts/export_system_enrichment.py`, without
collecting new data or training a model. The full research catalog remains in
[research/system_profiles](../research/system_profiles/README.md).
External-source competition eligibility has not been reviewed.
