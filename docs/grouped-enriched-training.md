# Enriched multilingual grouped pilot: completed result

**Completed 2026-10-24; Laya work is on hold.** This guide preserves the
experiment and implementation. It does not authorize further training or
inference. See [current status](current-status.md) for the modeling decision and
[shared data](shared-data-start.md) for reuse by other approaches.

## Result

| Measure | Result |
| --- | ---: |
| Training examples / optimizer updates | 34,718 / 543 |
| Training, export and reload phase | 6,239.54 seconds |
| Original-development Brier ALC | **0.2824393058** |
| Constant-50% baseline | **0.2500000000** |
| Mean within-pair AUROC | **0.5457424346** |
| Pair-weighted ECE, diagnostic | 0.2788073398 |
| Development responses / system-items / pairs | 963 / 960 / 8 |
| Native export/reload maximum probability drift | 0.0000275824 |

The recipe did not beat the constant baseline. Weak ranking and poor probability
calibration limit usefulness despite improved training loss. The experiment
used two held-out benchmark families and no recorded-only control; it does
not isolate enrichment's effect or establish that the underlying data lacks
signal. No calibration was fitted. Revealed outcomes were ignored, so the same
predictions apply at every budget. Calibration/confirmation and supplemental
development outcomes remained unused.

Frozen-parameter checks, exact training/inference token parity and all checkpoint
hashes passed. Both GPU apps are stopped with no active containers. The monitor
was removed. The initial completion helper exited before evaluation because an
already-stopped app returned a nonzero cleanup code; cleanup is now idempotent
and regression-tested. One authorized evaluation then completed, without
restarting training or duplicating inference.

## Artifacts and provenance

| Artifact | Location / identity |
| --- | --- |
| Fitted checkpoint | `runs/workflows/grouped_enriched_training_v1/job/output/final/` |
| Checkpoint manifest SHA256 | `a4e9a8ac4be347811896478130abd87b8e00ebb45f7a64eb9aa34bdf5fe4518c` |
| Training report | `runs/workflows/grouped_enriched_training_v1/job/output/report.json` |
| Development scores / predictions | `runs/workflows/grouped_development_v1/scores/` |
| Completion and cleanup receipts | `runs/workflows/grouped_pilot_completion_v1/` |
| Training ZenML / local native MLflow | `77bc46d6-9a22-4aba-a734-f10310eb33bb` / `6274cd35121049ce827f76a0906c84b1` |
| Development ZenML / MLflow | `eea88677-2242-4e09-ae7f-aa969723d7f4` / `7b31e61a71ce474797401770ef98f4cb` |
| Training Modal job | `job_93b4b7030f4e4a1a8911bb4faf46a462` |
| Development Modal job | `job_ae2bfc15636b4ee28b29af58e551568d` |

The native parent is the original multilingual checkpoint at revision
`7b928d828b7b0e022f929d9bd2e44165aa270148`. Training updated the final two encoder
blocks and decision head, with frozen prefix/action head, 8,192 total tokens,
FP16 autocast and FP32 scorer. Scientific recipes and renderer hashes remain
unchanged. The downloaded checkpoint is retained, not promoted to the default.
It supports inference and supplies learned weights for possible future work;
exact optimizer resumption is not implemented.

The remaining sections describe the retained implementation and historical
preparation checks. Commands are references, not queued work.

## Data delivered to the model

`configs/head_finetuning/grouped_enriched_v1.json` binds the exact collected-v3
export and passed multilingual qualification. Preparation rebuilt every one of
the **34,718 training states** using the inference state builder and retained
every original example. This includes the reviewed static facts, documented
architecture facts, recovered recorded settings, observations, meaningful
source qualifiers and explicit unknowns described in [the input audit](laya-input-contract.md).
Unreviewed task annotations are excluded. Research enrichment is not thereby
approved for competition use.

The unchanged shared question and native labels stay separate from the state.
Family and anonymous scope identifiers control sampling weights only; they do
not enter the model. Raw binary outcomes retain their original semantics,
including tables where the event is not correctness. No evaluation outcomes,
reference answers, private identities or revealed labels are added to inputs.

The prepared corpus is `runs/grouped_enriched/prepared_v1/tokens/`:

| Property | Prepared value |
| --- | ---: |
| Original fitting records | 34,718 |
| Training families / scopes | 7 / 170 |
| Examples requiring multiple views | 514 |
| Maximum views per example | 6 |
| Maximum native sequence | 8,192 tokens |
| Stored tokens, including repeated shared prefixes | 72,904,232 |
| Task tokens omitted | 0 |

`tokens.bin` stores int32 native tokens; `groups.jsonl` stores offsets, option
markers, per-view weights and separate loss metadata. Read-only memory mapping
avoids holding the corpus as millions of Python integers. The bundle includes
the seven feature tables, a pinned snapshot manifest, questions, reload fixtures
and a complete artifact/renderer fingerprint manifest. Monitor and development
labels are neither prepared nor selected by this trainer.

ZenML preparation run `c6c12be2-486b-45ce-b840-67fa2ceb955b` completed in 53.32s;
MLflow `49d27161ae6946a19048b23ddbc8f9ca`. The initial routing failure, before any
token output, is retained in `runs/zenml/grouped_enriched_preparation_v1.log`
and `runs/grouped_enriched/prepared_v1_failed_routing/`. The corrected run uses
`grouped_enriched_preparation_v1_retry.log`; a regression test covers the route.

## One objective for each complete example

`multilingual_recipe.encode_group` is the unchanged, qualified encoder shared
by preparation and prediction. Fitting states use the original native encoding.
Oversized states repeat all shared system/feature information and cover the
original question in overlapping task-token windows (128-token overlap).

Both trainer and predictor take the mean of the two logits across views,
weighted by newly covered task tokens, then apply softmax. Training computes
one binary loss for that original example. It does not treat fragments as new
independently labelled examples, average their probabilities, or shorten the
question silently. The qualified RNG-replay backward holds one view graph at
a time and has the same gradient as the joint grouped loss. Averaging logits
still does not guarantee reasoning across distant windows.

Loss weights give equal mass to training families, then to scopes within each
family, then to records within a scope. Weights are normalized across each
optimizer window, as in the existing weighted-loss path. Classes are not
artificially balanced. The completed recipe used one epoch, 543 optimizer windows
of up to 64 original examples, decision-head learning rate 1e-4 and final-two
encoder-block learning rate 1e-5. Encoder prefix and action head remain frozen.
Its measured held-out result is the negative result above.

The trainer uses qualified CUDA FP16 autocast, FP32 scorer, gradient scaling,
one physical view at a time and head/encoder checkpointing. It starts from the
original pinned multilingual parent and checks qualification, package versions,
tokenizer, parent tensors, source and data hashes before fitting. There is only
one enriched arm. It saves a final checkpoint after the fixed epoch; no implicit
baseline, development selection, evidence fitting or follow-on job occurs.

## ZenML entry points

```bash
# Offline preparation. Use a new version/path for a changed or repeated recipe;
# the actual output below already exists and must not be overwritten.
.venv-pipeline/bin/python scripts/run_pipeline.py inputs \
  --config configs/pipelines/grouped_enriched_preparation_v1.json \
  --out runs/grouped_enriched/prepared_v1 --execute

# Training PLAN ONLY; this command allocates no GPU and starts no fitting.
.venv-pipeline/bin/python scripts/run_pipeline.py training \
  --config configs/pipelines/grouped_enriched_training_v1.json \
  --out runs/workflows/grouped_enriched_training_v1
```

Training requires a separately authorized explicit `--execute`. The declared
profile is L4, two CPU cores, 12 GiB host RAM, four-hour timeout. The completed
qualification establishes bounded fixtures, not full-epoch throughput or
generalization. The verified upload contains only the prepared training bundle,
recipe and qualification report, not original trace tables, sealed outcomes,
recorded-only inputs or local model tensors. A dry transport plan is saved in
`runs/zenml/grouped_training_transport_plan_v1.json`.

`scripts/train_grouped_enriched.py` is the native stage utility: `--prepare`
tokenizes only; `--train --out FRESH_PATH` explicitly starts qualified CUDA
training. The ZenML workflow remains the main orchestration route.

## Checkpoint-bound prediction

Use the new explicit loader, rather than a historical English factory:

```python
from paiec.grouped_predictor import make_grouped_laya_predictor

predictor = make_grouped_laya_predictor(
    checkpoint_path="path/to/completed/output/final",
    backend="torch", model="multilingual", device="cpu", dtype="float32",
)
probability = predictor(subject, item)  # published string dictionaries
predictor.close()
```

The retained checkpoint is at the path above. The loader requires a valid local
`grouped_enriched_checkpoint_v1` manifest, exact artifact/renderer hashes,
8,192/256 context limits, identical question and the bundled feature snapshot.
It never picks a newer catalog or original checkpoint silently. Known subjects
join on all eight supplied fields. An unseen complete run stays explicitly
unresolved and cannot inherit another run's settings. Conflicting enrichment
for identical visible subjects is rejected. Anonymous benchmark IDs are absent
from the state and probability cache. Revealed examples are ignored, and
enabling evidence mode is rejected. MLX is not supported by this new route.

Probabilities use **raw grouped softmax**; no serving-temperature transform or
calibration is fitted. CUDA FP16 is supported explicitly; CPU uses FP32.
`scripts/predict_grouped.py` provides the same wrapper for JSONL requests with
exactly `subject` and `item` keys and a required checkpoint. It writes one
probability per original request; invalid labels/private fields are rejected.

The completed export checks rebuilt training-only single/multi-view fixtures
through this wrapper and verified identical token IDs, markers and view weights,
probability drift <=.005 across native FP16 save/reload, and frozen-parameter hashes. This does not score held-out outcomes,
establish competition packaging compatibility, or provide resumable training.

## Integration verification

The final source-only handoff passes **352 tests** in each native environment
(16 skips without Torch; one Torch skip), without datasets or weights;
all four catalogs validate. Added tests cover weighted gradients across
original examples, task-token coverage, label/input separation, native synthetic
model fitting/export/reload through the wrapper, snapshot tampering, context
overrides, anonymous IDs, private-field rejection and ZenML routing.

All 34,718 rebuilt training states also match the original serialized states
byte-for-byte, including key order. An independent tokenizer-only wrapper check
compares cached training tokens/markers/weights against all 514 multi-view groups
and a seeded sample: 614 original groups / 1,140 views in total. Receipt:
`runs/zenml/grouped_integration_parity_v1.json`. These are integration checks,
not a new trained-model or evaluation result.
