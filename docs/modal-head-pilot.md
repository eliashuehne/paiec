# Enriched head pilot on Modal

This guide records the earlier enriched-v1 pilot. The subsequent compact-v3
enriched head completed separately; see [compact enriched training](compact-enriched-training.md).

Prepared 2026-10-22 at the user's request. **The local run is stopped. The user
subsequently authorized the first cloud launch.** The submitted job selects **enriched alone**;
baseline is available separately. No automatic second run or retry is scheduled.

## Completed cached retrain

The enriched cached run completed all four epochs / 316 updates / 20,000
training presentations. Best epoch 1: development Brier ALC **0.24908013**.
Original raw parent on the same enriched inputs: **0.33525644**; constant 0.5:
**0.25000000**. Epoch scores: 0.24908013, 0.25615209, 0.26679374, 0.27265947.
Later epochs worsened held-out development despite decreasing training loss,
consistent with overfitting. This is only two development families; no
confirmation scoring, calibration fit or paired baseline retraining was run.

Training/development phase: 1,013.86 seconds (16.9 minutes), excluding startup,
initial raw-parent evaluation, cache validation and downloads. Median full-window
update times by epoch: 7.05, 1.41, 1.40, 1.40 seconds. Warm updates are roughly
5x faster than cold epoch-one updates; this is not a 5x end-to-end speed claim.
Cache held 5,215 unique features / 12.14 GB with zero evictions. Maximum
allocated GPU memory was 14.58 GB, peak allocated 15.09 GB and reserved 15.31 GB.

Encoder and auxiliary action-head hashes stayed unchanged; decision weights
changed. Export/reload tensor hashes match and fixture probability drift is
zero. Best checkpoint was downloaded locally, every file hash verified, and
the bundled enriched_v1/1,024-token input contract validated without inference.

Local output: `runs/head_finetuning/enrichment_pilot_v1_modal_cached/output/`.
Report: `output/report.html`; best model: `output/enriched/best/`.
Performance summary: `output/enriched/performance_summary.json`.
Local MLflow experiment **PAIEC Modal**, run `6f174af602994e3d9f15bc0131351ca8`
is FINISHED with all four development curve points imported. No GPU container
remained active at the final check.

## Cached retrain (implementation and launch history)

The user authorized a fresh enriched retrain with feature caching (now completed). Prior job
`job_bf242b6e53874dcebb6cd6dd44738625` stopped at 139 updates after its client
disconnected. Its first-epoch results and history are recovered under
`runs/head_finetuning/enrichment_pilot_v1_modal_interrupted_recovery/`; latest
weights remain on the volume. It is not a completed four-epoch result.

New detached job: `job_e5548e95fcec4aba88d110056a0e4f71`, function call
`fc-01M5K8SQNQ19CSW1Z8GFTGAB2T`.
[Live cached run](https://modal.com/apps/elias-huehne/main/ap-qpuYpW9bljFIY7WwDgyba2).
Remote MLflow: `dedd67f8fc0247c28020573f1157e32c`.
Local receipt/log: `runs/modal_cached_head_launch.json` and `.log`.
Fresh output: `runs/head_finetuning/enrichment_pilot_v1_modal_cached`.
Do not launch a duplicate.

Same original parent, FP32, 5,000 records, four epochs, 316 updates, micro-batch
2/accumulation 32, same order and token budgets. Only the frozen encoder outputs
are cached, capped at 14 GiB or available memory minus 4 GiB, whichever is lower.
No TF32, mixed precision or length sorting was introduced.

208 tests pass in each environment (PyTorch one skip, MLX nine). New CUDA checks:
33 mixed-length training batches show maximum probability drift 7.90e-7 and
head-gradient drift 1.68e-8; all development predictions/Brier scores match
exactly. Cache capacity/eviction, masks, frozen weights, RNG preservation and
state-dict integrity are tested. Raw model export/reload bypasses the cache.
Cache identity/bytes/hits/misses are logged; features are never checkpoint tensors.

First epoch fills features lazily, so its cold updates remain near the original
speed. Subsequent epochs should reuse them. Actual warmed full-run performance
requires measurement. New launches use a detached app and durable spawned call
so local client loss does not intentionally terminate training. Result retrieval
and MLflow import can be recovered with the existing fetch script.

Example launch (already running; do not repeat):

```bash
.venv-modal/bin/python scripts/run_job.py --profile modal-l4-pilot \
  --task train-head-pilot --variant enriched --encoder-cache-gib 14 \
  --out runs/head_finetuning/enrichment_pilot_v1_modal_cached
```

## First cloud launch (historical)

Submitted Modal job `job_bf242b6e53874dcebb6cd6dd44738625` on the prepared L4
profile. [Live Modal app](https://modal.com/apps/elias-huehne/main/ap-LrMzFk0CpOsTcAQj28MLGD).
Controller PID 17867, with idle-sleep prevention while attached. Local launch
log: `runs/modal_enriched_pilot_launch.log`; receipt:
`runs/modal_enriched_pilot_launch.json`. Setup/model downloads and CUDA preflight completed. Every prepared training
and development state/token check passed; parent and preparation hashes match.
The worker reports NVIDIA L4 (23,034 MiB). Remote MLflow run:
`3a15460e13bc4bc6933b93693712126a`.
The earlier uncached attempt was running at this historical check. First 15 updates: median 7.67 seconds,
peak allocated GPU tensor memory 2.43 GB, and 2.58 GB reserved.
These are early training measurements, not full-run timings or quality results.
Local initial baseline updates were about 35 seconds; differing variants/inputs
mean this is an indicative comparison, not a controlled hardware benchmark.
Progress snapshot: `runs/modal_enriched_pilot_progress.json`. The attached
controller downloads results and imports local MLflow on completion. Do not
submit a duplicate. No epoch or completed score is established by these updates.

## Stopped local attempts

Both local attempts remain interrupted, with no complete epochs, trained
checkpoints or completed quality scores:

| Attempt | Completed updates | MLflow status |
| --- | ---: | --- |
| Initial MPS allocation growth | 11 | KILLED, `be2aec0b45bb441fad3f881f79719d25` |
| Memory-bounded MPS, stopped at user request | 11 | KILLED, `1187918c63604de29a3d0e0bcf0364a6` |

The second attempt's log and history are preserved under
`runs/enriched_pilot_mpsbounded_training.log` and
`runs/head_finetuning/enrichment_pilot_v1_training_mpsbounded_v2/`.
Its GPU driver allocation after cleanup stayed near 3.24 GB; updates took about
35 seconds. This was stable but slow. Neither history is an optimizer checkpoint.
Cloud training starts from the pinned original parent, not those partial heads.
The older canceled 10,000-input evaluation remains untouched.

## Prepared first job

| Setting | Value |
| --- | --- |
| Task / variant | `train-head-pilot` / `enriched` only |
| Execution profile | `modal-l4-pilot`: one L4, two CPU cores, 16 GiB RAM |
| Runtime | Same shared PyTorch trainer, explicit CUDA, FP32, autocast and TF32 disabled |
| Frozen recipe | `configs/head_finetuning/enrichment_pilot_v1.json` |
| Training | Same 5,000 records, seed 0, four epochs, 316 updates |
| Batch | Micro-batch 2, accumulation 32, effective batch 64 |
| Inputs | Same `enriched_v1` states, active question, 1,024 / 192 token limits |
| Parent | Original English Laya, revision `7b928d828b7b0e022f929d9bd2e44165aa270148` |
| Selection | Best raw development Brier ALC among epochs 1–4; original raw parent also scored |
| Development | 963 outcomes / 960 subject–item inputs / eight pairs / two families |
| Calibration and confirmation | Not uploaded or opened |
| Timeout | Four hours, no automatic retries |

Only device and output location may override the prepared configuration.
Learning rate, sample selection, labels, seed, weighting, token limits and
optimizer settings remain verified against the preparation. Masked padding in
multiples of 128 is shared with the stopped memory-bounded run. CUDA keeps its
allocator cache and logs allocated/reserved/peak memory; MPS-specific allocator
calls are never used on CUDA. Backend numerical differences remain possible;
this is a new CUDA experiment, not a continuation or a completed comparison.

## Launch after go-ahead

Authentication is already configured in `.venv-modal`; it need not be repeated.
Inspect the complete plan without creating cloud resources:

```bash
.venv-modal/bin/python scripts/run_job.py \
  --profile modal-l4-pilot --task train-head-pilot --variant enriched \
  --out runs/head_finetuning/enrichment_pilot_v1_modal_enriched --dry-run
```

The reviewed plan is saved as `runs/modal_enriched_pilot_plan.json`.
To start the first paid cloud run, use the same command without `--dry-run`:

```bash
.venv-modal/bin/python scripts/run_job.py \
  --profile modal-l4-pilot --task train-head-pilot --variant enriched \
  --out runs/head_finetuning/enrichment_pilot_v1_modal_enriched
```

This builds the locked Linux image, stages inputs, downloads only the pinned
parent files into the persistent model cache, performs native preflight, and
starts enriched training. Missing CUDA or failed input checks cause failure;
there is no local/CPU fallback. A fresh output directory is required. Keep the
launcher attached: detached execution and optimizer resumption are not supported.

`modal-l4` retains its 15-minute timeout for smoke work. Use `modal-l4-pilot`
for this experiment. The worker mirrors progress into Modal logs and retains
`console.log`; initial development scoring and epoch exports also take time.
Early CUDA throughput is now measured above. Full training, checkpoint export,
result download and quality assessment remain pending. Do not infer model quality
from update speed or local preparation.

If a baseline control is later requested, use `--variant baseline` and a fresh
output such as `runs/head_finetuning/enrichment_pilot_v1_modal_baseline`. Run it
on the same profile and scientific recipe. The first enriched result alone
cannot establish that enrichment helped.

## Data, checkpoints and tracking

The 20-file input inventory is 75,980,393 bytes (20,633,944 compressed here).
It includes every manifest-pinned prepared artifact and original training and
development JSONL needed to reconstruct inputs. Baseline artifacts are included
to verify the complete paired manifest; baseline training does not run.
It excludes calibration/confirmation label exports, payload tables, traces,
credentials, environments, Git state and unrelated runs. Source/configuration
files enter the image separately. Profile facts are bundled; no runtime browsing
is used for enrichment.

Prepared input hashes and renderer hashes remain unchanged. Both variants keep
the same raw outcomes and active success question. External profile-source
eligibility for the competition is still unreviewed; this remains research-only.

Each epoch writes development scores and a `latest` checkpoint; completion
exports `best` and checks reload hashes/probabilities. Large weights, including
`best_decision.safetensors`, stay on `paiec-jobs`. Result downloads include
history, scores, predictions, reports and tracking artifacts. Model bundles
retain the enrichment input contract and profiles for later local inference.

After completion, local results appear under the selected output, with training
artifacts in its `output/` subdirectory. `output/report.html` supports a single
completed enriched run. Local MLflow imports into **PAIEC Modal**, retaining
metric steps/timestamps and interrupted statuses. The remote run IDs remain
attached. Checkpoint paths are listed in `modal_receipt.json` and `status.json`
as `checkpoint_directories`, typically:

```text
job_<id>/work/runs/head_finetuning/enrichment_pilot_v1_modal_enriched/enriched/best
job_<id>/work/runs/head_finetuning/enrichment_pilot_v1_modal_enriched/enriched/latest
```

Use Modal's volume download command with a listed path to retrieve a complete
checkpoint when needed. Hash verification and the existing `LayaPredictor`
interface then support local inference with the checkpoint's bundled inputs.
Ordinary failures/timeouts retain logs/results; hard platform termination may
leave partial files. No automatic resumption or partial Brier finalization is
promised.

## Cost and completed preparation checks

At the published rates checked 2026-10-22, L4 + the requested CPU/RAM costs
approximately **$1.02 per billed hour**. Four hours would be about $4.09 at those
allocations, excluding storage and other charges. Actual usage, model loading
and container lifecycle contribute to billing; the timeout is not a dollar cap
or a runtime estimate. Account credit balance was not checked. See
[Modal pricing](https://modal.com/pricing) and [the pricing calculation](modal.md).

Completed locally without cloud resources or model inference:

- Enriched-only dry run and frozen input/recipe/hash verification.
- Full bundle round trip into an isolated worker workspace, followed by native
  preflight on MPS: every token sequence and rendered state matched. This
  validates bundle completeness, not CUDA execution.
- Modal app/image/function definitions constructed locally with the L4 profile.
- Unit checks for placement, configuration drift, complete bundle transport,
  model-file exclusion, failed/timed-out worker output and MLflow curve import.

Evidence is in `runs/modal_pilot_preparation_validation.json`,
`runs/modal_pilot_isolated_preflight.log`, `runs/modal_enriched_pilot_plan.json`,
`runs/modal_pilot_torch_tests.log` and `runs/modal_pilot_mlx_tests.log`.
These checks preceded the authorized first launch. Read its live status above;
no CUDA speed or quality claim follows from the preparation checks alone.
