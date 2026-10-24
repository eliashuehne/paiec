# Laya input contract and collected-data delivery

2026-10-24. This work prepares and inspects inputs only: no model forward pass,
optimizer update, GPU job, paid teacher call, or held-out outcome scoring.
Revealed-example learning is outside this preparation.

## Upstream contract, checked against installed code

The [Laya README](https://github.com/NandhaKishorM/laya#quickstart),
[training implementation](https://github.com/NandhaKishorM/laya/blob/main/laya/train.py),
and installed `laya==0.4.13` were inspected. States can be text, dictionaries or
lists. Dictionaries are serialized using `json.dumps(..., ensure_ascii=False)`;
they are not separate numeric feature tensors. There is no upstream guarantee
that JSON or particular section headings produce better learning.

Training records use `{state, questions, expected}` (or a gold distribution).
Our question remains the existing `correct`/`noul` question, unchanged. Native
conversion maps false to `[1, 0]` and true to `[0, 1]`; actual examples of each
label in both arms passed this check. No label enters the state.

The native sequence is question/options first, then state:

```text
[CLS] noul question: <unchanged question> [SEP]
[MASK] false: no, the statement does not hold
[MASK] true: yes, the statement holds [SEP]
<serialized state> [SEP]
```

Upstream also documents English `noul` label/criteria sensitivity. This is a
reason for later controlled learning checks, not evidence that it caused our
particular scores. No criteria, labels, probability instructions or shared
question wording were changed in this work.

## Gap in the earlier route

The existing compact/30k renderer loads its frozen `profile_bundle` and
`compact_profiles_v3` policy. It does not automatically consume the newer
reviewed static features, typed recovered run settings and qualified source
observations produced by feature preparation. Those historical contracts stay
unchanged. A green collection pipeline did not establish delivery to Laya.

## Prepared bridge

```bash
.venv-pipeline/bin/python scripts/run_pipeline.py inputs
```

The default recipe is `configs/pipelines/laya_collected_inputs_v3.json`.
Planning is the default; execution requires `--execute` and a fresh output.
This graph exports matched recorded-only and collected-information arms,
checks native tokenization in the separate Torch environment, and publishes
hashes/reports to MLflow. It does not train or evaluate predictions.

The actual export `runs/laya_inputs/collected_v3/` contains:

- 34,718 training rows; 1,000 monitor and 963 original development input rows.
  Only training rows have `expected`; other outcome files are not opened.
- All 205 confirmed values from the 617-field scoped review, all 149 recovered
  recorded run-setting values, and 29 documented observation entries.
- 105 additional previously documented architecture facts from the existing
  profiles, kept distinguishable from the scoped review. Two inferred extras
  are excluded. Twenty meaning-changing source qualifiers are retained, including
  rounded counts and language-component counts that exclude a vision encoder.
- Exact supplied system/task strings. Anonymous benchmark IDs remain audit/join
  metadata, not state features. Reference answers and grading metadata are not
  added. Provenance/source quotations remain in pinned collection snapshots.
- Explicit unknowns; documented harness support remains separate from enabled
  run settings. Base-family facts do not establish an exact evaluated checkpoint.
  Unreviewed task annotations are excluded; no difficulty/ability scores are invented.

Each state has `target_system_recorded`, `target_task` and (in the collected
arm) `collected_information`. Compact groups store values by source status,
unknown/unconfirmed field names, separate documented model/base and harness
facts, actual settings, additional observations and applicable source qualifiers.
The task is identical between arms; there is no shortening during export.

The actual cartoonist record is in `collected_cartoonist_example.json` with its
recorded label outside the state. In particular, the raw field remains
`reported_task_generation_size=512`; it is not promoted to a verified output
limit. The typed maximum-output setting is unknown.

The earlier v1 diagnostic used verbose value/status pairs. V2 compacted the same
reviewed information and added prior documented architecture facts; v3 preserves
scope qualifiers. All recipes/renderers are retained. These are input audits,
not trained or scored model versions, and no format has demonstrated a learning
gain yet. Main v3 MLflow run: `9340939fa1a147988c2b7b27c1e3185c`.

## Multilingual long context

The pinned original checkpoint revision is
`7b928d828b7b0e022f929d9bd2e44165aa270148`. Its multilingual encoder and tokenizer
configs both permit 8,192 positions/tokens. The saved agent default is 1,024,
with a 256-token head cap. The native head for our actual question occupies 45
tokens, leaving **8,146 state tokens** at a total limit of 8,192. Do not subtract
the entire head cap as though it were the actual head length.

The whole v3 export was checked with the pinned multilingual tokenizer and
native sequence/target conversion, without loading weights:

| Multilingual total budget | Complete collected training states | Fraction |
| --- | ---: | ---: |
| 2,048 | 19,489 / 34,718 | 56.1% |
| 4,096 | 32,759 / 34,718 | 94.4% |
| 8,192 | 34,197 / 34,718 | 98.5% |

At 8,192, 521 training states and 17 monitor states still overflow; all 963
original development states fit. No development outcomes were read. These are
coverage results with that tokenizer, not correctness or runtime-memory results.
English and multilingual tokenizers differ, so compare within the relevant
tokenizer. The English v3 audit fits 24,169 / 34,718 collected training states
at 2,048 (69.6%).

Audit: `runs/laya_inputs/collected_v3_multilingual_8192_audit/`, MLflow
`14351b06b34d4f78823ae2430b4b44d9`. Reproduce into a new directory with:

```bash
.venv-laya/bin/python scripts/check_laya_inputs.py \
  --prepared runs/laya_inputs/collected_v3 --model multilingual \
  --max-len 8192 --head-max-len 256 --out runs/laya_inputs/NEW_AUDIT
```

## Subsequent outcome

The [multilingual qualification](multilingual-qualification.md), grouped trainer,
checkpoint-bound inference and one full enriched pilot subsequently completed.
Every task token was retained, but original-development Brier ALC was
**0.282439**, worse than constant **0.25**. See [the result](grouped-enriched-training.md).
Laya work is on hold; this input audit is retained as evidence of data delivery,
not evidence that this representation improves predictions. The structured
source tables can still be reused independently of Laya; see [shared data](shared-data-start.md).
