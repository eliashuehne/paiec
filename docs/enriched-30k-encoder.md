# Expanded enriched data and upper-encoder experiment

Status as of 2026-10-24: **Laya work is on hold** after the completed grouped
pilot failed to beat the constant baseline. This is a retained experiment or
implementation reference; proposals and commands below are not a current work
queue. See [current status](current-status.md) and [shared data](shared-data-start.md).

2026-10-23. The user authorized the sequence of learning checks, expanding to
30,000 records, comparing frozen and partially trained encoders, and later
learned evidence adaptation. This version implements those as separate stages;
quality results require completing the corresponding jobs. **Subsequent user
steering: finish the active frozen arm, report Brier, and hold all further runs.
The upper-two arm and evidence fitting are deferred; do not launch them.** Original
development, calibration and confirmation roles are preserved.

## Data

`configs/head_finetuning/enriched_30k_data_v1.json` prepares
`runs/head_finetuning/enriched_30k_v1/` with the existing pinned compact-v3
renderer and collected profile catalog. No renderer or historical recipe changed.

The eight training benchmarks / seven families admit all 170 candidate pairs:
source hashes match the earlier full-item review; binary outcomes link to
nonempty tasks; BenGER retains its recorded-prompt-only filter; every pair has
at least 80 items; supplied tools do not vary between trials of an item. Missing
model/harness identities remain missing. Private IDs/display names never enter
the predictor state. Calibration/confirmation outcomes are not read.

Sampling selects one uniformly random recorded trial per system/item, then
cycles through families and pairs without replacement, independently of label
values. All 30,000 records are distinct system/item pairings, covering all 170
pairs. Equal family weight and equal pair weight within a family prevent the
120 BIRD configurations from dominating training. This changes the training
sample/weighting from the old pilot; do not attribute a difference from that
pilot solely to encoder updates.

| Training family | Records |
| --- | ---: |
| Abstract reasoning | 4,882 |
| BenGER | 4,883 |
| BIRD | 4,882 |
| CellVerse | 707 |
| ClashEval | 4,882 |
| Fewshot-TTT BBH | 4,882 |
| GSM8K/MATH (AdaptiveStep + Critic) | 4,882 |

Training includes 15,383 distinct normalized tasks. Sourced model/base facts
reach 28,542/30,000 records; harness facts reach 20,236/30,000. All detected
answer options survive. There are 7,286 explicit task/context omissions,
including 2,490 oversized question/user sections. No different full tasks share
an identical complete retained training input. This does not establish complete
visibility or remove every semantic ambiguity from long inputs.

A deterministic 5% normalized-task-hash bucket within training families is
reserved for an internal monitor. All systems/trials of a task inherit its
bucket. The exported monitor has 1,000 records and is task-disjoint from
training. It is an item holdout inside training families, not an unseen-family
quality claim. Frozen original development remains the same 963 response rows /
960 system-item evaluation inputs / eight pairs. Original full-item overlap
review is reused only with unchanged source hashes; exact reserved provenance
hashes and effective-input cross-role conflicts are checked again. These are
not exhaustive translation/paraphrase checks.

## Learning checks and adaptation

New training implementation: `src/paiec/encoder_training.py`; orchestration:
`scripts/run_encoder_experiment.py`. The old head-only pilots remain unchanged.
Recipe: `configs/head_finetuning/enriched_30k_encoder_v1.json`.

Both main arms restart the pinned original English parent and use the same
enriched data, seed, effective batch 64, one epoch / 469 optimizer updates,
head LR 1e-4, weight decay .01 and clipping 1.0. The adapted arm also trains
the final two ModernBERT encoder blocks and final normalization at LR 1e-5.
The encoder prefix and auxiliary action head stay frozen. The control freezes
the whole encoder. Native `detach_encoder` is false for the adapted arm; tests
check actual gradients and layer-specific training/evaluation modes.

Qualification first tests throughput/numerical fidelity on real training inputs.
It then trains disposable synthetic controls and tiny real training-only
overfit controls for both policies. Synthetic data deliberately exposes a
YES/NO success marker; it is isolated from competition exports and quality
evaluation. Passing requires >=95% synthetic validation accuracy and Brier
<=.08 within 80 updates. The 32-short-input real control requires a training
Brier <=.10 and at least a 40% reduction within 60 updates. These controls
prove learnability of those fixtures, not generalization. Their models are
discarded; main arms restart original weights. A failed control blocks the
main launcher.

There is no checkpoint selection on original development. The fixed one-epoch
checkpoint is scored once per arm; internal monitor statistics are saved.
Raw-head scores repeat across label budgets because this experiment does not
yet learn from revealed outcomes. Do not imply these are evidence-conditioned
heads. Final calibration/confirmation and the supplemental development set
remain untouched.

## Resource optimization

The GPU is NVIDIA L4 (24 GB nominal, about 23 GB visible), separate from system
RAM. Qualification requests two physical CPU cores and 8 GiB system RAM instead
of the old 16 GiB. Its measured peak RSS was 8.72 GiB, so the main training
profile requests 10 GiB for headroom. Modal charges CPU/memory at the higher of
requested or actual usage; these are minimum allocations, not hard limits. Peak process RSS is
recorded. See [resource billing](https://modal.com/docs/guide/resources).

Native tokenization, full render parity and all-record option/token checks run
locally before GPU rental. A compressed native-token cache is keyed by the
prepared manifest, checkpoint weight hash, token budgets and exact
Laya/Transformers/tokenizer versions. Remote loading verifies cache hashes and
re-encodes representative and longest fixtures. This caches **tokens**, not
encoder representations; it remains valid while encoder weights change.

The L4 probe compares FP32 batch two in original order, FP32 batches four/eight
with within-window length grouping, and FP16 autocast batches 4/8/16/32 with
grouping. Earlier qualification also tested BF16, which failed the numerical
gates.
The mixed-precision path keeps the final scorer FP32: large nearly equal option
scores can lose their difference when rounded to BF16. The wrapper changes
compute context only, preserving all native parameter names and export layout.
Optimizer windows keep identical membership. Weights/AdamW state stay FP32;
TF32 stays off. Loss accumulation stays on GPU until the optimizer-window
boundary, avoiding a host synchronization per microbatch. CUDA fused AdamW
reduces optimizer dispatches. Padding is rounded to 64 tokens. Actual dropout
remains enabled during throughput timing; it is disabled for numerical checks.

Both policies must pass a common operational setting: 64 mixed-length fixture
probabilities, sampled clipped-gradient cosine >.99, and longest-input capacity
with peak allocation below 85% of visible VRAM. BF16 probability drift must
stay below .03 maximum/.01 mean versus FP32; these are bounded numerical gates,
not equivalence or quality guarantees. The fastest combined measured setting
is selected. Zero-LR probe optimizer steps allocate real AdamW state without
numerical weight changes. Fused AdamW can normalize negative zero to positive
zero; the disposable-probe check canonicalizes only signed zero and verifies
every nonzero bit. Main-training frozen parameter checks retain strict byte
hashes. A failed/OOM candidate is rejected.
No frozen-encoder feature cache is used during encoder adaptation.

BF16 autocast is motivated by [PyTorch mixed-precision guidance](https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html),
but must pass our measured checks. Memory occupancy alone is not a throughput
objective. Current [Modal rates](https://modal.com/pricing) are $0.000222/s for
L4, $0.0000131/core/s and $0.00000222/GiB/s. Two CPU cores plus 8 GiB requested
RAM gives an approximate $0.9575/hour compute base rate, versus $1.0214/hour
at 16 GiB, assuming actual usage stays within the request. Job timeouts do not
cap all billed setup/tail time or storage.

## Checkpoints and recovery

Both arms export ordinary native Laya weights/config/tokenizer and the bundled
profile/input contract, with exact file hashes. Upper-layer adaptation adds no
adapter-specific inference dependency; existing Torch/MLX loaders can use the
ordinary architecture. CUDA export/reload checks verify tensor identity and
probabilities under the selected policy. Native MLX probability validation is
separate; export compatibility alone does not establish numeric equivalence.

The job is detached, no automatic retries, and input inventories contain only
the prepared training/monitor/development bundle and selected qualification.
No traces, calibration/confirmation labels or old checkpoints are uploaded.
Progress is saved per optimizer update, but optimizer resumption is not
implemented. A completed frozen arm survives if the later adapted arm fails.
Do not restart/duplicate submitted jobs without checking their receipts.

## Separate learned evidence control

`scripts/fit_evidence_adaptation.py` fits one small regularized logistic
correction for each completed head. The base head has never fitted the 1,000
training-family monitor queries. Their outcomes become supervision for this
separate correction, with support outcomes drawn only from matching visible
profiles and anonymous benchmark IDs in the 30k training sample. Query tasks
are disjoint from every support. Twenty random nested support sets per query
and the positive-budget ALC weights train the correction; episode rows repeat
the same 1,000 supervised queries and are not additional independent records.

Features are query log-odds, smoothed observed log-odds, evidence count/reliability
and interactions. This first learned numeric control does not yet learn task
similarity or support-task difficulty. Budget zero stays exactly the trained-head
prediction. Regularization is fixed (C=1), with no development fitting or search.
Original development compares raw head, fixed beta-k=5 and learned correction
using the same shared acquisition trajectory and acquisition exclusion. This
is 30k head-fitting records plus 1k additional training-role records for evidence
fitting. It does not unseal the separate calibration role. Coefficients use
versioned homogeneous JSON, with an optional explicit predictor wrapper;
default prediction and historical checkpoints do not change.

The script uses saved monitor/development probabilities, so it needs no new
Laya inference or GPU work after the heads complete. Native event labels stay
unchanged. These are separate postprocessing results, not raw checkpoint scores.

## Commands

Preparation and local CPU token checks:

```bash
.venv-laya-mlx/bin/python scripts/prepare_expanded_training.py
.venv-laya/bin/python scripts/run_encoder_experiment.py --stage prepare-tokens
```

Separate bounded qualification, then training only after a passed report:

```bash
.venv-modal/bin/python scripts/run_job.py --profile modal-l4-encoder-qualify \
  --task qualify-encoder --out runs/head_finetuning/enriched_30k_qualification_v3
.venv-modal/bin/python scripts/run_job.py --profile modal-l4-encoder \
  --task train-encoder \
  --qualification runs/head_finetuning/enriched_30k_qualification_v3/output/qualification.json \
  --out runs/head_finetuning/enriched_30k_encoder_training_v1
```

Use `--dry-run` to inspect the inventory/command without allocating a GPU.
Receipt IDs, measured settings and actual results will be recorded after the
corresponding stages complete. Preparation and implementation are not completed
training results.

First qualification: `job_9b8e295996874ce98fc99d80319b7442`, detached call
`fc-01M5N4AJCTG5CYVECDSDQG3VJ9`, app `ap-VefUShbGu5n8dXrbFnZrl6`.
It ran only the frozen policy before stopping on the signed-zero byte-hash
check. CPU regression reproduces -0 to +0 with exactly zero numerical change.
All BF16 candidates also exceeded the original probability-drift gate: maximum
.0360-.0445, mean .0045-.0049. Their gradients had sampled cosine >.99999.
FP32 batch two took 12.03 seconds/window; grouped FP32 batch eight 10.73;
BF16 batches 4/8/16/32 took 3.41/3.75/4.40/5.00 seconds. Larger memory use did
not imply better throughput. This is not a completed qualification or training
result. The app is stopped. Logs:
`runs/enriched_30k_qualification_launch_v1.log` and
`runs/enriched_30k_qualification_worker_v1.log`.

Revised qualification: `job_7724d8673d424285bfd6875835545fe3`, detached call
`fc-01M5N4ZW3JWWETNES72DEBHF50`, app `ap-77R2Ry6IZ7G3VDqU7B87A7`.
It tested the FP32 scorer boundary and signed-zero-only probe check, preserving
the same numerical drift gates. Its report is superseded by v3 because trainer
source hashes changed for FP16 support. No main arm had started at this entry.

V2 completed successfully with grouped FP32 batch eight. The synthetic controls
reached 100% after 10 updates; tiny real training Brier fell from .426783 to
.033126 frozen and .029997 upper-two after 50 updates. Both were discarded;
these are in-sample machinery checks, not held-out performance. BF16 remained
outside the numerical limit for every batch. The app was explicitly stopped.

Qualification v3 tests FP16 autocast with proper gradient scaling, initial scale
1024 and unscale-before-clipping. Nonfinite gradients reject a candidate rather
than silently skipping a main optimizer update. The max probability-drift limit
is .005, with the same >.99 sampled gradient-cosine and longest-input headroom
checks. Candidates include FP32 batches 2/4/8 and FP16 batches 4/8/16/32. FP16
retains the FP32 scoring boundary. Source hashes require using a report from
this updated code; the old v2 report cannot launch the updated trainer.

Submitted v3: `job_e21bed934f1b4a9283454393c2e6137e`, detached call
`fc-01M5N5P3YTXM18B56Q7GX36TPR`, app `ap-GsqvojiNWvqjNlTcXksQ6h`.
Log: `runs/enriched_30k_qualification_launch_v3.log`. No main training launched
at this entry. Use the passed v3 report for the main experiment.

V3 passed with FP16, FP32 scorer, microbatch four, effective batch 64 and
within-window length grouping. Frozen / upper-two window medians are 3.437 /
3.783 seconds, versus 12.080 / 13.292 for original-order FP32 batch two. Max
fixture probability drift is .003625, mean .000394; gradient cosine 1.0. Longest
upper-two peak allocation is 4,096,672,256 bytes (3.82 GiB). Microbatch eight
also passed but was slower; 16/32 were slower and exceeded the .005 drift gate.
Synthetic controls reached 100% accuracy in ten updates. Tiny real training
Brier fell from .427079 to .062162 frozen (60 updates) / .063530 upper-two
(50 updates). Controls were discarded. Qualification app is stopped.

Main training submitted: `job_22bab18c0ed84ae1bd90a81377078cb3`, app
`ap-rHJVtBX7GZkIrPKrUYtC9d`, call `fc-01M5N6ENR8NGHG800W9HKSQNR1`.
Log: `runs/enriched_30k_encoder_training_launch_v1.log`. It runs frozen then
upper-two sequentially on one L4. Do not launch a duplicate.

## L4 versus T4

Pricing rechecked 2026-10-23 at [Modal](https://modal.com/pricing). With two
cores and 10 GiB requested RAM, L4 is approximately $0.9734/hour and T4
$0.7646/hour, before storage and actual-usage increases. T4 must complete the
same job within 1.273 times the L4 runtime to be cheaper. For the measured
3.437-second frozen L4 window, that means about 4.375 seconds on T4; for the
3.783-second adapted window, about 4.816 seconds. No T4 timing was measured,
so lower hourly price does not establish lower experiment cost. Keep the
qualified L4 for this matched comparison. Low VRAM occupancy is compatible
with highest measured throughput; batch 32 used much more memory and ran slower.

A full-corpus padding simulation preserves the exact 469 optimizer windows.
Grouped microbatch four processes 31,983,616 padded tokens versus 29,931,388
real tokens (6.42% padding). Grouped microbatch 32 processes 43,183,104 tokens
(30.69% padding), about 35% more work by this token-count proxy. Attention and
kernel costs also vary, so this is an explanation supported by the timing, not
a complete performance model. Original-order microbatch two has 28.20%
padding. Artifact: `runs/enriched_30k_padding_analysis_v1.json`. One live training
snapshot showed 100% GPU utilization, 3,134 MiB device memory and 72.94 W; an
instantaneous sample does not establish average utilization over the job.

## Using the resulting predictor

After checkpoints and evidence fits finish, the usual predictor factory loads
the checkpoint's pinned input format, maximum length and bundled research facts:

```python
import json
from pathlib import Path
from paiec.predictor_factory import make_laya_predictor
from paiec.learned_evidence import LearnedEvidencePredictor

base = make_laya_predictor(
    backend="mlx", dtype="float32",
    checkpoint_path="runs/head_finetuning/enriched_30k_encoder_training_v1/output/upper_2/best",
)
fitted = json.loads(Path(
    "runs/head_finetuning/enriched_30k_evidence_v1/upper_2/learned_evidence.json"
).read_text())
predictor = LearnedEvidencePredictor(base, fitted)
# subject/item use the published predictor fields; revealed contains only
# outcomes acquired within the current budget.
probability = predictor(subject, item, revealed)
```

Choose the raw base or explicit learned wrapper according to the measured
comparison. The wrapper rejects a correction bound to a different checkpoint.
Torch is selected with `backend="torch"`; selecting a backend does not change
the model weights or input contract. Cross-backend probabilities require the
separate local inference checks; training/export alone does not prove parity.

`scripts/check_encoder_checkpoint.py` verifies each downloaded checkpoint through
the native Torch/MPS and MLX FP32 predictors using up to five training-only
fixtures, including short/long inputs. It checks exact reconstruction from the
checkpoint's bundled facts, artifact identity and valid native probabilities.
An optional cross-backend reference requires matching fixtures/weights and max
FP32 probability difference <=.001. This is integration validation, not a new
quality evaluation or a guarantee over all possible inputs.

## Outcome-blind breadth diagnostics

The expanded training sample has 170 distinct allowed visible profiles. Of
15,383 normalized tasks, 3,565 appear with multiple profiles, covering 18,181
fitting records. This counts genuine differences in supplied system fields,
not private scorer IDs used as model features. A text-only script diagnostic
finds CJK Unicode characters in 738/30,000 training records, 24/1,000 monitor
records and 479/963 development records. This does not classify languages or
measure understanding; it flags a substantial transfer difference for the
English parent. No outcomes were used for these diagnostics. Artifacts:
`runs/head_finetuning/enriched_30k_data_diagnostics_v1/`.

Evidence evaluation uses `src/paiec/development_evidence.py` to verify the frozen
source-table hashes and select only the original eight development systems.
The full eligible item universe is read without returning outcome values, to
reproduce the original split; outcome values are materialized only for the
120 evaluation items and first 31 acquisition items per pair. Supplemental
systems and other item outcomes are excluded by Parquet predicates. A synthetic
regression verifies identical scores, repeated-response weighting and revealed
prefixes versus the ordinary full-table scorer. No historical loader changed.

## Subsequent user stop boundary

While the frozen arm was active, the user requested its Brier when finished and
no new run. The queued upper-two arm and separate evidence fitting are now held.
An external controller polls for the completed frozen `report.json`, then sends
an interrupt to the parent training subprocess. No training code or weights
are changed by that controller. The first arm completes training, scoring and
export/reload verification before the interrupt; the two-arm parent job is
intentionally interrupted. Its completed frozen score must be distinguished
from the unrun encoder/evidence comparisons. Controller intent/outcome are
recorded in `user_stop_after_frozen.json` in the job output and
`runs/enriched_30k_stop_after_frozen_v1.log`. No new run should be started to
finish the original two-arm plan without another user instruction.

After the scope change, completion means: frozen arm has 469 actual updates /
30,000 presentations, a finished raw development score, verified frozen
encoder/action weights and exported checkpoint; parent interruption prevents
further training. A cloud tracking status interrupted during final logging
should be reported separately from these completed scientific artifacts.
Do not present a frozen-versus-encoder result or fit the prepared evidence
correction merely to finish the superseded plan.

## Completed frozen arm: 2026-10-23

| Predictor | Original development Brier ALC |
| --- | ---: |
| Constant 50% | 0.250000 |
| Earlier compact enriched head, 5k records | 0.248643 |
| Expanded enriched head, 30k records; encoder frozen | **0.272163** |

The completed arm has 469 updates / 30,000 presentations / one epoch. Its exact
ALC is 0.2721625032217301; C-Eval mean .2541253564, ERBench .2901996501.
The score is worse by .0221625 versus the constant baseline and .0235199 versus
the prior 5k head on the unchanged original development questions. Training
weighting and numerical policy also changed; this comparison does not isolate
data quantity or enrichment. Raw predictions ignore labels and repeat across
all budgets. No further evidence fit or inference/evaluation run was started.

The training-family task-held-out monitor scores .1686245256. This is a different
role/distribution from original unseen-family development. Development
probabilities now span .227071 to .879776; greater confidence alone did not
improve the held-out score. The two-family development scope remains narrow.
No calibration/confirmation or supplemental development outcomes were used.

Encoder/action tensors stayed unchanged; decision weights changed. Native CUDA
export/reload fixture drift is zero. Downloaded checkpoint artifact hashes and
identity were verified without additional model inference:
`local-8821de1fd7271a91cd46c139fe56593730a5ad8ea5f84c7a980e371070c255db`.
Local path: `runs/head_finetuning/enriched_30k_encoder_training_v1/output/frozen/best`.
The same folder's `report.html`, `report.json`, `completed_summary.json` and
`download_verification.json` record the result.

Measured median update 3.354 seconds, peak CUDA allocation 2.773 GiB, process RSS
9.621 GiB; training/development/export phase 28.146 minutes. This excludes initial
parent/data loading and initial monitor prediction. L4/two cores/10 GiB RAM
remains the measured configuration; there is no T4 timing comparison.

The stop controller interrupted the parent during final metric logging after
the finished report, before any upper-two training. Parent return code -2 is an
intentional stop of the two-arm plan. The sole completed scientific arm remains
valid. Its local MLflow run `9448ce30b7c34f6581900633de5ff40d` was completed from
verified saved scores/artifacts in the same run record; the original imported
KILLED logging status and reason are retained in tags and
`completed_frozen_tracking_receipt.json`. No new training/tracking experiment
was created for this correction. App stopped; final container listing is empty.
The encoder arm, learned evidence fitting and local native inference checks
remain held.
