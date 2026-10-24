# Compact enriched head training

Status as of 2026-10-24: **Laya work is on hold** after the completed grouped
pilot failed to beat the constant baseline. This is a retained experiment or
implementation reference; proposals and commands below are not a current work
queue. See [current status](current-status.md) and [shared data](shared-data-start.md).

Authorized 2026-10-23, enriched variant first. The recorded-data control is
not launched. The family split, prepared inputs, catalog, question and scientific
recipe remain immutable. Calibration/confirmation and the separate question-only
development supplement are not used by this run.

## GPU qualification

A separate bounded Modal L4 job tested the exact compact-v3 enriched input
contract before training. Both a mixed 64-record training window and the longest
training inputs passed at 2,048 tokens, FP32 with TF32 off, physical batch two.
One warmed window took 13.004 seconds. Peak allocated memory was 2.57 GB;
reserved memory 2.71 GB, with 23,689,363,456 bytes total GPU memory. These are
short probe measurements, not a whole-run speed claim. No optimizer steps were
applied and all three model-module hashes stayed unchanged.

Qualification job: `job_f5095d9c721e465c87dd6571fea14830`.
Report: `runs/head_finetuning/compact_pilot_v3_modal_qualification/output/qualification.json`.
The new `qualify-head-pilot` job uses verified, training/development-only pilot
inputs and stores a separate report. It does not silently start training.

## Completed enriched run

Modal job: `job_58c10a1eb5a54753af8b3afc0f19c855`.
Detached call: `fc-01M5MSWBXW80BBK8Y9B9FTP0SB`.
[Modal app](https://modal.com/apps/elias-huehne/main/ap-gLoOfFmYcx5zeQOuXx3PGX).
Receipt/log: `runs/compact_v3_enriched_launch.json` and `.log`.
Output: `runs/head_finetuning/compact_pilot_v3_modal_enriched/`.
Completed 79 updates / 5,000 presentations. Do not relaunch this experiment.
The app was stopped after result retrieval; no GPU container remained active.

| Setting | Value |
| --- | --- |
| Variant | `compact_enriched_v3` only |
| Parent | Original English Laya, revision `7b928d828b7b0e022f929d9bd2e44165aa270148` |
| Recipe | `configs/head_finetuning/compact_pilot_v3.json` |
| Prepared inputs | `runs/head_finetuning/compact_pilot_v3_release/` |
| Training | 5,000 records, one epoch, 79 updates |
| Batches | Physical 2, accumulation 32, effective 64 |
| Native context / question head | 2,048 / 192 tokens |
| Precision / device | FP32, TF32 off, NVIDIA L4 CUDA |
| Frozen modules | Encoder and auxiliary action head |
| Training modules | Decision head, type embeddings and scorer |
| Seed / learning rate / decay | 0 / 0.0001 / 0.01 |
| Feature cache | Disabled for this single-pass run |

The prior cache benefit came from reusing encoder features across four epochs.
This recipe has one epoch. A simple full-corpus FP32 estimate is 18.99 GB for
training features plus 1.02 GB for development, before model, gradients,
optimizer and temporary allocations. The existing uncached recipe avoids
assuming that whole cache fits or helps this pass. No precision, sample order,
physical batch or optimizer change was made to speed the experiment.

Startup verifies source/artifact hashes, all native tokens and shared rendering
on the remote environment. The initial original-parent score and post-epoch
score use the original frozen development records. A new completed Brier result
must be read from the finished report, not inferred from progress counters.

## Validation and interpretation

238 tests pass in each model environment (MLX nine skips, Torch one). Prepared
hashes remain valid. Training follows the same workflow as earlier pilots:
unchanged frozen-module hashes, changed decision weights, best/latest exports,
and native reload checks before reporting completion. Checkpoint tensors remain
on the Modal volume until explicitly downloaded; routine result retrieval
excludes them.

Compare against constant 0.5 (Brier ALC 0.25) and the raw original parent on
these same compact enriched states. The older 0.249080 trained head used a
different 1,024-token enriched-v1 input contract. A comparison with that number
combines formatting, context, catalog and training-duration changes. Without a
matched recorded-data head, this run cannot isolate enrichment's contribution.


## Completed results

| Predictor on the original frozen development set | Brier ALC |
| --- | ---: |
| Original raw parent on these compact enriched inputs | 0.41407047 |
| Constant 50% | 0.25000000 |
| Earlier trained enriched-v1 head (historical different inputs) | 0.24908013 |
| New compact enriched-v3 head | **0.24864261** |

The new head improves on constant by 0.00135739 and the historical trained
head by 0.00043752. This is a small exploratory gain, not proof of a meaningful
quality improvement or of enrichment's isolated benefit. Both development
families improve slightly over 0.25: C-Eval 0.24814649, ERBench 0.24913873.
Predictions remain narrow (46.52–56.04%, mean 50.27%). The encoder is still frozen.
No revealed-label adaptation was used; the Brier curve is flat across budgets.
No confirmation/calibration or supplemental development score was computed.

Training/development/export phase: 1,151.60 seconds (19.2 minutes), excluding
startup, full input preflight and the initial parent evaluation. Median full
64-record update: 13.109 seconds. Maximum sampled peak allocation: 2.777 GB;
maximum reserved memory: 3.020 GB. Both encoder and action-head hashes stayed
unchanged; decision hashes changed. Native export/reload probability drift was
zero. Local best weights, all file hashes, the bundled compact policy/catalog,
renderer hashes and one exact prepared training render were verified without
additional local model inference.

Best checkpoint: `runs/head_finetuning/compact_pilot_v3_modal_enriched/output/enriched/best/`.
Report: `output/report.html`; detailed checks: `output/enriched/local_checkpoint_check.json`.
Performance summary: `output/enriched/performance_summary.json`.
Local MLflow experiment **PAIEC Modal** (ID 4), run
`57987fd441494993babee451c882f1c6`, FINISHED. Remote run:
`2f6c34e8dbfb407bb8b7b1cd777a845f`. The client timed out waiting for final
app log streaming after completion; the job return code was zero, results and
weights were retrieved, and completion checks passed. No retry was launched.

Later inference must explicitly select the checkpoint; defaults still use the
original parent. The predictor factory reads `compact_enriched_v3` and 2,048
from its bundled contract. Native Torch/CUDA reload was qualified here; this
run does not establish a new MLX probability-fidelity or speed measurement.
