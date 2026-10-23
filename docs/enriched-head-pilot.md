# Ready-to-launch local head experiment

Prepared 2026-10-22 at the user's request. The user subsequently authorized
PyTorch/MPS training and the paired workflow was launched. Live progress is in
`runs/enriched_pilot_training.log` and the per-variant history files under
`runs/head_finetuning/enrichment_pilot_v1_training/`. Do not launch a duplicate.
The preparation checks below describe the completed pre-training stage;
training results must be read from the actual run reports.

The first attempt was interrupted after 11 updates because GPU driver allocation
grew from 6.45 GB (update 3) to 8.84 GB (update 11), with increasing update time.
MLflow run `be2aec0b45bb441fad3f881f79719d25` remains KILLED. It has no completed
epoch, finished checkpoint or quality result. Its log/history are preserved.

The recovery starts from the original parent in a fresh directory:
`runs/head_finetuning/enrichment_pilot_v1_training_mpsbounded_v2/`, with log
`runs/enriched_pilot_mpsbounded_training.log`. It uses a 0.7 MPS allocator fraction,
masked padding in multiples of 128 to reduce distinct GPU shapes, and unused-cache
trimming after optimizer updates and every 16 development batches. Real token
values, positions, masks, labels, 1,024-token budget and optimizer recipe stay
the same. The padding policy is recorded because stochastic training can depend
on tensor shapes. All 198 PyTorch tests pass, including masked-padding checks.
This is a fresh attempt, not an optimizer resumption or continuation of the
earlier canceled 10,000-input experiment.

## Local stop and Modal preparation

The user requested stopping local training and preparing Modal with the enriched
variant first. The memory-bounded attempt stopped at 11 completed updates and
zero epochs. MLflow `1187918c63604de29a3d0e0bcf0364a6` is KILLED; its history, log
and interruption record remain intact. GPU driver allocation stayed near 3.24 GB
after cleanup, and updates took about 35 seconds. No completed checkpoint or
development result exists. Neither local attempt may be restarted implicitly.

The CUDA pilot is now prepared; see [modal-head-pilot.md](modal-head-pilot.md).
It starts from the original weights, with the same input hashes and recipe.
The first cloud job selects enriched alone, with baseline a later separate job.
No cloud training has been launched. The local launch instructions below record
the earlier recipe and must not be interpreted as instructions to restart it.

## Saved recipe

Two independent head-only fine-tunes start from the original English checkpoint,
revision `7b928d828b7b0e022f929d9bd2e44165aa270148`:

| Setting | Both variants |
| --- | --- |
| Training | Same 5,000 frozen response records, labels and seed 0 |
| Development | Same 963 outcomes, 960 subject–item inputs, 8 pairs across 2 benchmarks |
| State | `context_v1` control; `enriched_v1` adds documented profile facts |
| Question | Unchanged `task_correctness_v1`; no revealed examples |
| Sequence / head budgets | 1,024 / 192 tokens |
| Added metadata | At most 96 spare tokens; complete facts; original text stays intact |
| Epochs / updates | Four epochs, 316 updates per variant, 632 total |
| Batching | Micro-batch 2, accumulation 32; effective batch 64, final window 8 |
| Optimizer | AdamW, LR 0.0001, weight decay 0.01, clipping norm 1 |
| Placement | Local PyTorch/MPS, FP32, `.venv-laya`; sequential variants |
| Trainable | Decision head, type embeddings and scorer |
| Frozen | Encoder and auxiliary action head |
| Selection | Lowest raw development Brier ALC among epochs 1–4; latest saved separately |
| Calibration / confirmation | Remain unopened and unscored |

The new 1,024-token limit gives more tasks room for metadata: at 512, none of
the sampled SQL-system states fits completely. Both variants use the same
larger limit, so the comparison isolates enrichment. Historical JSON/512
results remain separate. Runtime and peak memory for this full recipe have
not yet been measured; the 256-row smoke timing is not a reliable estimate
for these two full runs.

## Prepared data and measured coverage

Source data remains `runs/head_finetuning/data_v2`. New exports and the
[readable preview](../runs/head_finetuning/enrichment_pilot_v1/report.html)
are in `runs/head_finetuning/enrichment_pilot_v1/`:

| Role | Records | Any metadata | Model properties | Harness capabilities | Truncated, both variants |
| --- | ---: | ---: | ---: | ---: | ---: |
| Training | 5,000 | 2,732 | 1,429 | 2,278 | 2,042 |
| Development | 963 | 963 | 120 | 963 | 0 |

Model/harness counts overlap. Enrichment never adds truncation: it is appended
only when complete facts fit after the original state. The 2,042 long training
states still truncate identically in both versions. This experiment does not
recover their removed task text.

Only the Qwen development subject receives model properties under the current
identity policy. The other development subjects receive harness capabilities.
Consequently, this is limited evidence about architecture-feature generalization.
Any metadata does not imply that the model identity has been resolved.

Artifacts include paired per-role JSONL files, provenance, input contracts,
`token_audit.csv`, `coverage_by_pair.csv`, actual states with evidence in
`examples.json`, source/data hashes in `manifest.json`, and
`local_preflight.json`. Preparation has a separate MLflow run named
`enriched_v1 / input preparation / no training`, with no model scores.

## Shared enrichment and prediction

`src/paiec/profile_enrichment.py` resolves explicit supplied model IDs, reviewed
canonical aliases, base-model identifiers and harness names. It never resolves
from benchmark names, subject IDs, audit display names, item content, reference
answers, grading rules or outcomes. Unknown/private explicit IDs block fallback
to a conflicting canonical model; there is no fuzzy matching.

`configs/head_finetuning/enrichment_v1.json` defines reviewed aliases and fields.
Only `documented` facts enter the state; inferred facts are excluded. Allowed
fields cover architecture, rounded parameters, dimensions, modalities and model
type. Current API context/output defaults and harness execution defaults are
excluded because applicability to recorded versions is unverified. Base-model
matches add architecture only. Harness workflows are labeled as capabilities,
not confirmed enabled behavior.

Exports and `LayaPredictor` use the same builder. New checkpoints bundle their
catalog and policy. Their input contract pins the question, format, token limit,
renderer source hashes and profile hashes. Later predictions restore omitted
format/limit options from that contract. Conflicting overrides or changed
bundles fail before inference. Upstream and old JSON checkpoints retain their
existing defaults.

This is explicitly **research-only**. External competition-source eligibility
remains unreviewed. There is no runtime browsing or change to the published
subject/item interface or default predictor mode.

## Next launch

The recipe is `configs/head_finetuning/enrichment_pilot_v1.json`. The user has
stopped local training; use the Modal instructions above for the next experiment.
The following commands describe local capabilities. Check readiness
without loading model weights, inference or training:

```bash
.venv-laya/bin/python scripts/run_head_pilot.py
```

After the user's go-ahead, launch both variants:

```bash
.venv-laya/bin/python scripts/run_head_pilot.py --train
```

That original output now exists. The stopped recovery was launched with
`--out runs/head_finetuning/enrichment_pilot_v1_training_mpsbounded_v2`.
Do not run either command again while training is active.

An explicit `--variant baseline` or `--variant enriched` limits the launch if
requested; default is both. A fresh training output directory is required.
Results go to `runs/head_finetuning/enrichment_pilot_v1_training/` and **PAIEC
Head Fine-tuning** in MLflow. No Modal resources are requested.

Each variant records losses, gradients, raw development Brier ALC per epoch,
best/latest checkpoints, per-pair scores, predictions, package/source provenance,
frozen-module hashes and an export/reload check. The original parent is also
scored with raw temperature 1 on that variant's inputs. The 50% reference is
0.25. Without revealed examples, every label budget has the same score.
Checkpoint weights stay local; MLflow receives identities and smaller artifacts.

`latest` is saved after complete epochs; `best` is exported at the end. Optimizer
resumption is not implemented. Update history is progress, not a resumable
checkpoint. Interrupted runs must not be relabeled complete or restarted
implicitly. Calibration/confirmation remain separate later steps.

## Later inference

After training, this automatically applies the bundled input setup:

```python
from paiec.laya_predictor import LayaPredictor

predictor = LayaPredictor(
    checkpoint_path="runs/head_finetuning/enrichment_pilot_v1_training/enriched/best",
    backend="mlx",
    dtype="float32",
)
probability = predictor.predict([subject, item], [])
predictor.close()
```

PyTorch is also available using `backend="torch"` in `.venv-laya` and its usual
`dtype="auto"`. Input construction is shared; numerical fidelity across backend
and precision choices remains a separate measurement.

`run_eval.py --laya-checkpoint PATH` and execution jobs also restore the input
contract. The historical `evaluate_head_development.py` rejects new pilot
checkpoints because its JSON/512 inputs would differ. Pilot development scoring
is built into the new trainer. Low-level runtimes expect already-rendered states;
use `LayaPredictor` for subject/task prediction with automatic enrichment.

## Preparation checks

Local preflight passed with MPS available and roughly 50 GiB free. Every paired
record passed native PyTorch-versus-MLX token checks. Rebuilding each enriched
state through the runtime builder reproduced the prepared text exactly. Labels,
questions and original text are unchanged. These checks establish input and
environment readiness; training convergence, peak GPU memory and quality are
not yet measured. Parent weights were hashed on disk, not loaded into Laya.

The full lightweight suite passes in both environments: 197 tests discovered,
four expected skips in MLX and one in PyTorch. Logs are
`runs/enrichment_mlx_tests.log` and `runs/enrichment_torch_tests.log`.
