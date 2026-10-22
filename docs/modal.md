# Local and Modal execution

The user authorized Modal integration on 2026-10-22 and authenticated the client
against workspace **elias-huehne**. The SDK verified the credentials and saved
them outside the repository in `~/.modal.toml`. No credentials are included in
images, data bundles, source snapshots, logs or Git.

## Architecture and supported work

A runtime and an execution location are separate choices:

| Profile | Location | Runtime | Supported jobs |
| --- | --- | --- | --- |
| `local-mlx` | Apple Silicon Mac | MLX, FP32 in this launcher | Evaluation |
| `local-torch` | Mac | PyTorch/MPS, upstream automatic inference precision | Evaluation; head smoke training |
| `modal-l4` | Modal NVIDIA L4 | PyTorch/CUDA, upstream automatic inference precision | Evaluation; head smoke training |

MLX is Apple-specific and is not an NVIDIA training backend. Head training uses
FP32 on either MPS or CUDA. CUDA performance, inference precision differences,
probability fidelity, and model quality still need measurement on real cloud
hardware. The existing **bounded 256-row/10-update** smoke trainer is supported;
a full pilot, encoder adaptation, and resumable training are not implemented
by this integration. It does not silently expand the training budget.

```
src/paiec/
  competition.py, laya_predictor.py, state_rendering.py  input and prediction logic
  laya_runtime.py                                     PyTorch / MLX engines
  head_training.py, head_evaluation.py                training/scoring helpers
  execution/
    jobs.py          typed profiles, job commands and minimal input inventories
    transport.py     hashed input bundles, safe extraction, local MLflow import
    modal_adapter.py optional SDK, image, volumes and cloud dispatch
    modal_worker.py  isolated whole-job subprocess, logs and result packaging
configs/execution/profiles.json    placement and resource settings
scripts/run_job.py                 common launcher (thin CLI)
scripts/fetch_modal_job.py         recover completed cloud results without a GPU
```

The execution package calls existing scientific scripts; it does not copy the
prediction/scoring/training algorithms. Framework and Modal imports stay lazy.
The existing direct script commands remain available. New job types can be added
to the typed plan without adding another inference backend or mixing cloud
transport into LayaPredictor.

## Setup and commands

The optional client environment is separate from both model environments:

```bash
uv venv --python 3.12 .venv-modal
uv pip install --python .venv-modal/bin/python -r requirements-modal-client.lock
.venv-modal/bin/python -m modal setup
```

Authentication is already complete on this machine. Use `python -m modal`
because the executable is not on the shell's normal PATH. No need to rerun setup
for each experiment. `requirements-modal-client.lock` records the installed
SDK (1.6.2) and local tracking dependencies. The worker has its own Linux/CUDA
resolution in `requirements-modal-worker.lock`, compiled for Python 3.12 Linux
x86_64 from `requirements-modal-worker.in`; the image installs that lock.
The package `cloud` extra is available for a fresh client resolution.

Inspect a plan without creating cloud resources or allocating a GPU:

```bash
.venv-modal/bin/python scripts/run_job.py --profile modal-l4 \
  --task evaluate --suite smoke --state-format context_v1 \
  --out runs/modal_smoke_NEW --dry-run

.venv-modal/bin/python scripts/run_job.py --profile modal-l4 \
  --task train-head-smoke --out runs/modal_head_NEW --dry-run
```

Remove `--dry-run` only when ready to run the job. Select `local-mlx` or
`local-torch` to run the same evaluation locally. Head smoke training requires
`local-torch` or `modal-l4`; choosing MLX fails explicitly. The frozen smoke
trainer still uses its original JSON inputs. A context_v1 training request is
rejected rather than silently pairing a changed format with the old manifest.

`--checkpoint runs/PATH/checkpoint` explicitly selects a verified local trained
artifact for evaluation; otherwise the original pinned weights are used. Local
checkpoint files are staged to Modal only when explicitly selected. Use the same
suite, seed, state format and checkpoint for comparisons. GPU evaluation uses
upstream CUDA automatic precision; local MLX uses FP32 in this launcher. Those
policies differ, so the profiles do not promise identical predictions.

## Data, persistence and tracking

Jobs execute as whole batches, rather than one remote call per question. Modal
runs an ephemeral app named `paiec-laya`, with one worker, no automatic retries,
2 CPU cores, 16 GiB memory and the selected GPU. The default timeout is 900
seconds. Validated profiles can change GPU and timeout (60–3,600 seconds).
A timeout is a runtime bound, not a dollar spending limit. GPU, CPU, memory and
storage are separately billable; do not assume a particular credit balance.

Input inventories are explicit:

- Evaluation uploads only the requested suite's subject/item/response tables
  and corpus manifest. The current launcher evaluates Laya, not cross-benchmark
  prior/text controls that require additional training tables.
- Smoke training uploads only `training.jsonl` and `provenance.parquet` from the
  frozen export (about 25.5 MB here). Provenance includes split metadata for all
  roles, but no target outcome column. No calibration/confirmation label export
  is uploaded. Config/source files are included separately in the image.
- Neither job uploads trace tables, the full repository, `.git`, virtual
  environments, browser state, tokens, or unrelated historical runs.

Source hashes, input hashes, checkpoint provenance, seed, actual device/precision
and execution metadata are retained. The base checkpoint revision is unchanged.
Input and output paths are validated, and extraction rejects traversal/symlinks.

Two named persistent volumes are used: `paiec-jobs` for per-job data/results and
`paiec-model-cache` for downloaded pinned model files. Every job gets an isolated
working tree and MLflow database. Volumes are committed after execution,
including failures. Normal process failures and subprocess timeouts produce
logs/status/result archives; a hard platform termination may leave only partial
files on the volume. No automatic job resume or partial-score finalization is
provided. Existing evaluation/training interruption limitations still apply.

Small result archives are downloaded automatically. Remote MLflow runs are
re-logged into the existing local store under **PAIEC Modal**, with the remote
run IDs attached. This avoids copying a cloud SQLite database whose artifact
paths refer to another machine. Model weights stay on the cloud volume and are
excluded from routine result downloads; use the checkpoint location recorded in
`modal_receipt.json` to download them explicitly through Modal's volume CLI.
A downloaded checkpoint can then use the ordinary `--checkpoint` path and hash
verification.

The launcher writes a job receipt under `runs/modal_jobs/` before submission and
prints the job ID. To recover a completed job into a new local folder:

```bash
.venv-modal/bin/python scripts/fetch_modal_job.py \
  --job-id job_REPLACE_WITH_32_HEX_DIGITS --out runs/recovered_NEW
```

This downloads results and imports tracking without allocating a GPU. Cloud
checkpoints and caches persist until deliberately removed through Modal volume
management. The launcher stays attached; closing it can terminate an ephemeral
app. Detached submission is not implemented.

## Verification and remaining qualification

Completed locally:

- Modal SDK installed in its separate environment; browser authentication and
  token verification succeeded for elias-huehne.
- Linux worker dependencies resolved; the SDK constructs the locked image,
  app and function definitions locally without hydrating/deploying them.
- Evaluation and training dry runs produce correct CUDA commands and explicit
  input inventories, without cloud calls.
- Unit coverage includes backend/location validation, unsupported training,
  payload exclusions, bundle integrity/traversal checks, and failed/timed-out
  subprocess result packaging. Full suites: 174 discovered, 171 pass/3 skipped
  in MLX and 173 pass/1 skipped in PyTorch.

No image has been built remotely, no GPU job has run, and CUDA numerical fidelity,
real volume transport, result download and MLflow import have not been verified
end to end. The next bounded cloud validation is the 16-input smoke evaluation,
followed by the 10-update head smoke only if desired. Cloud execution does not
establish Codabench runtime compatibility.

## Official documentation checked

- [Apps and invocation](https://modal.com/docs/guide/apps)
- [GPU configuration](https://modal.com/docs/guide/gpu)
- [Images and dependencies](https://modal.com/docs/guide/images)
- [Volumes and persistence](https://modal.com/docs/guide/volumes)
- [Project structure](https://modal.com/docs/guide/project-structure)
- [Current pricing](https://modal.com/pricing)
