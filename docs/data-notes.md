# Public training data

The competition's public training release is
[`aims-foundations/measurement-db`](https://huggingface.co/datasets/aims-foundations/measurement-db),
downloaded with the baseline's own tool so the layout matches what the
submission expects:

```bash
python baseline/tools/prepare_data.py --out payload --no-embeddings
```

## What is on disk

- Revision `54074453da7b9911be0f48e4c560be547bc84d21`, downloaded 2026-10-07.
- 61 eligible benchmarks of 102 listed. Eligibility is the competition's:
  `response_type == "binary"` and `granularity == "item"`.
- 304 parquet files, 24.22 GB, of which 23.4 GB is `traces.parquet` and 17 GB of
  that is `alignment_faking` alone. The tables used for training and scoring
  total under 1 GB. Routine scoring and EDA do not load trace content; the
  outcome audit inspected selected archived native fields for semantic checks.
- Roughly 1.36M items and 16.4M responses. The largest benchmarks by response
  count are bridging_gap, alignment_faking, babilong, afrieval, and cruxeval.
- 6,870 subject-benchmark pairs have at least 80 distinct items with observed
  binary outcomes. Counting items whose outcomes are missing gives 6,895;
  those unknown outcomes cannot be scored by the local harness.

The full local audit is reproducible with `scripts/eda.py`. See
`runs/eda/report.html` for its report and `runs/eda/` for CSV statistics,
schemas, examples, and plots. The audit found that binary outcomes do not
universally mean correctness: consult each benchmark's `response_scale`.

## Disk footprint and the duplicated cache

`prepare_data.py` downloads every table with `hf_hub_download` and then copies it
into `payload/` with `shutil.copyfile`. Both copies persist, so a full build
costs about twice the payload: on 2026-10-22 `payload/data` was 22.6 GiB and the
HuggingFace dataset cache held the same 22.6 GiB, against 33 GiB of free space.

The cache is safe to drop once `payload/data/corpus_manifest.json` exists,
because every scoring path reads `payload/` (the default `--data-root`). It is
not safe to drop before then: at that point the cache is the only copy.

```bash
.venv-laya-mlx/bin/python scripts/disk_report.py            # sizes, read-only
.venv-laya-mlx/bin/python scripts/disk_report.py --cleanup  # reclaim the copy
```

`scripts/disk_report.py` only ever deletes the dataset cache. It never touches
`models--convaiinnovations--laya`: those weights are the only copy of the
checkpoint and share no blobs with the dataset. Deleting the dataset cache does
not affect tokenizer loading, because `measure_tokens.py` fetches the tokenizer
from the model repo.

Two smaller notes, so size numbers can be read correctly:

- The cache lists 102 benchmark directories but only 61 are eligible and
  downloaded. The other 41 are filtered out by the binary/item check.
- `runs/` is small (~40 MB) even after many evaluations. MLflow stores metrics
  and CSV artifacts, not trace tensors. It is not worth pruning.

Per-table schema:

| table | columns used here |
| --- | --- |
| `subjects.parquet` | subject_id, display_name, normalized_name, provider, release_date, access_date, harness, harness_version, reasoning_effort, subject_features_extra |
| `items.parquet` | item_id, content, item_features |
| `response.parquet` | subject_id, item_id, response, trial, interactors |

`traces.parquet` is not used by predictors. See
[outcome-semantics.md](outcome-semantics.md) for the selective native-record audit.

## Two data quirks that shape the code

`normalized_name` is empty for many subjects. The competition-shaped input
keeps it empty; `display_name` is not used to fill it. The same system is also named differently across
benchmarks (`CMMLU / gemma3_27b_it`, `UnifiedQA-T5-11B (ARC prompt)`), so
`canonical_identity` reduces names before any cross-benchmark lookup. Even after
that reduction most systems appear in a single benchmark, which limits how much
a subject prior can help.

Items and responses are separate tables keyed by `item_id` and `subject_id`.
There can be several responses per item, one per trial; the split keeps all
trials of an item together, and acquisition reveals one recorded response per
item.

## Embeddings

`payload/` has no `embeddings/` directory. The companion release
`measurement-db-embed` contains three files (`real_webagents`,
`researchcodebench`, `swe_rebench`), and none of those benchmarks are in the
eligible public set, so `prepare_data.py` fails if asked for embeddings at this
revision. Only the keyword/table retrieval tools in the baseline are usable;
the semantic-retrieval tool cannot be built.

## Checking for updates

The repository has more than 100,000 files, so `repo_info` truncates its file
listing and file counts prove nothing. Compare the commit instead:

```python
from huggingface_hub import HfApi
HfApi().repo_info("aims-foundations/measurement-db", repo_type="dataset").sha
```

If that equals the revision recorded in `payload/data/corpus_manifest.json`,
the payload matches the live release. As of 2026-10-20 it does.
