# Enriched multilingual grouped training and prediction

## Authorized pilot execution

The training stage has now completed **543 updates / 34,718 examples** in
6,239.54 seconds. Frozen-parameter checks and exact training/inference token
parity passed; maximum native export/reload probability difference is
**0.0000275824**. The fitted checkpoint is downloaded and all hashes verify
locally at `runs/workflows/grouped_enriched_training_v1/job/output/final/`.
Checkpoint manifest SHA256:
`a4e9a8ac4be347811896478130abd87b8e00ebb45f7a64eb9aa34bdf5fe4518c`.

The original completion helper then exited before evaluation because the
Modal stop command returned code 1 for an already-stopped training app. The
app's stopped state and empty GPU-container inventory were verified. The
cleanup operation now accepts that exact already-satisfied result, with a
regression test; genuine errors still fail. Original failure artifacts are
retained under the completion directory. Training was not restarted.

The first authorized evaluation was subsequently launched once through ZenML
`eea88677-2242-4e09-ae7f-aa969723d7f4`, using the already generated bound recipe.
See `runs/workflows/grouped_development_v1/` and
`runs/workflows/grouped_pilot_completion_v1/recovery_receipt.json` for live/final
status. No development quality result is claimed until its score report exists.
The old completion process has exited; do not start a duplicate evaluation.

The user subsequently authorized one full enriched-only epoch, checkpoint
verification and one original-development evaluation. Training is launched in
`runs/workflows/grouped_enriched_training_v1/`, ZenML
`77bc46d6-9a22-4aba-a734-f10310eb33bb`, Modal
`job_93b4b7030f4e4a1a8911bb4faf46a462`, app `ap-wnqQfjWc36PqoccJN3lkO9`.
The remote MLflow run is `832f1fb44be94ae9bf0aa7b6f10d185e`.
There are 543 planned updates; observed training is underway at launch.

Current progress: `runs/workflows/grouped_enriched_training_v1/native_job.log`.
A durable, already-started completion process waits for the existing run,
downloads missing checkpoint tensors without a GPU, verifies all hashes, then
uses the same ZenML native execution route for one original-development
prediction/scoring workflow. It does not start another training run, select
another checkpoint, or retry a failure. Its current status is
`runs/workflows/grouped_pilot_completion_v1/status.json`; launch receipt is
`runs/zenml/grouped_completion_v1_launch.json`. Inspect these before further
action. The process stops both declared GPU apps after completion. A
`STOP_AFTER_TRAINING` file in the training workflow directory cancels deferred
stages while preserving active training.

Development preparation is `runs/grouped_development/prepared_v1/`: the original
963 response trials, 960 system-items and eight pairs, matched to the exact
frozen input/label/provenance hashes. Only `requests.jsonl` and its manifest are
uploaded for prediction. Native labels and scorer identities remain local and
separate from the predictor. Predictions repeat across budgets because revealed
evidence is ignored. Existing scoring preserves trial counts and equal pair
weighting, with Brier ALC, constant-.25 control, within-pair AUROC and diagnostic
calibration bins. No calibration is fitted. Calibration/confirmation sets and
the supplemental development outcomes stay unused.

The checkpoint-bound evaluation recipe and workflow are generated only after
the fitted checkpoint manifest is recovered. Expected final evaluation report:
`runs/workflows/grouped_development_v1/scores/report.json`. Training is not yet a
quality result. Both native suites pass 351 tests after this addition; saved logs
are `runs/zenml/grouped_completion_{mlx,torch}_tests_final.log`.

The preparation-stage account below predates this launch.

2026-10-24. The grouped input contract is integrated into a new enriched-only
trainer and an explicit Torch predictor. Offline preparation is complete. No
full training, new real-model inference, development scoring or GPU job was
launched for this integration. The existing English trainers, factories and
hash-pinned renderers remain unchanged.

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
artificially balanced. The prepared recipe has one epoch, 543 optimizer windows
of up to 64 original examples, decision-head learning rate 1e-4 and final-two
encoder-block learning rate 1e-5. Encoder prefix and action head remain frozen.
This policy is a proposed full experiment, not a measured generalization gain.

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

There is no trained grouped checkpoint yet. The loader requires a valid local
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

After a future training run, export checks rebuild training-only single/multi-
view fixtures through this wrapper, require identical token IDs/markers/view
weights and maximum probability drift <=.005 across native FP16 save/reload.
Frozen-parameter hashes must also match. This does not score held-out outcomes,
establish competition packaging compatibility, or provide resumable training.

## Integration verification

Both native suites pass **344 tests** (16 skips without Torch; one Torch skip),
including in a clean 451-file source-only export without datasets or weights;
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
