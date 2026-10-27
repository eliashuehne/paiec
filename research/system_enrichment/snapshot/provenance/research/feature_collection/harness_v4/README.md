# Harness source review

Eight primary READMEs are copied from previously pinned, hash-verified archives.
`sources.jsonl` preserves their original source records and repository commits;
`snapshots.json` identifies the archive under `runs/feature_sources/harness_v4/`.
The model/harness catalogs and old recipes remain unchanged.

`assertions_v1.jsonl` adds three typed facts: ERBench's optional Wikipedia
retrieval, its context augmentation, and Petavue's question/schema assembly.
These describe supported behavior of the pinned code/documentation revision.
They do not establish the actual harness version, enabled tools, supplied
context, or budgets in any recorded run.

`observations_v1.jsonl` preserves eleven additional capabilities, including
prompting modes, inference backends and candidate handling. These appear in
the pipeline's `qualified_harness_observations.jsonl` and the ZenML harness
fact table. They never populate enabled-tool or actual-run fields.

`reviews_v1.jsonl` reviews all four requested fields for all eight harnesses:
32 field reviews. Where the README does not establish a global default or tool
registry, the field stays unknown. The reviews are by the agent and are not
independent human reviews or proofs of absence in all possible sources.

Restore the archived sources with:

```bash
.venv-pipeline/bin/python scripts/fetch_system_profile_sources.py \
  --catalog research/feature_collection/harness_v4 \
  --cache runs/feature_sources/harness_v4
```
