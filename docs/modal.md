# Local and Modal execution

Status as of 2026-10-24: **Laya work is on hold** after the completed grouped
pilot failed to beat the constant baseline. This is a retained experiment or
implementation reference; proposals and commands below are not a current work
queue. See [current status](current-status.md) and [shared data](shared-data-start.md).

Historical expanded work: [enriched 30k and upper-encoder training](enriched-30k-encoder.md).
The new `qualify-encoder` / `train-encoder` tasks preserve historical pilots and
use `modal-l4-encoder-qualify` / `modal-l4-encoder` profiles (2 cores; 8 GiB RAM for qualification, 10 GiB for training).
They add selected encoder-block training, local native-token caching and
measured batch/precision qualification. GPU memory is separate from system RAM.
Read the current-status guide for the completed result; older limitations below describe the
earlier head-only implementation.

The user authorized Modal integration on 2026-10-22 and authenticated the client
against workspace **elias-huehne**. The SDK verified the credentials and saved
them outside the repository in `~/.modal.toml`. No credentials are included in
images, data bundles, source snapshots, logs or Git.

## Architecture and supported work

A runtime and an execution location are separate choices:

| Profile | Location | Runtime | Supported jobs |
| --- | --- | --- | --- |
| `local-mlx` | Apple Silicon Mac | MLX, FP32 in this launcher | Evaluation |
| `local-torch` | Mac | PyTorch/MPS, upstream automatic inference precision | Evaluation; head smoke and pilot training |
| `modal-l4` | Modal NVIDIA L4 | PyTorch/CUDA, upstream automatic inference precision | Evaluation; head smoke training (900 seconds) |
| `modal-l4-pilot` | Modal NVIDIA L4 | PyTorch/CUDA, FP32 pilot training | Head pilot (four-hour timeout) |

The separate `qualify-head-pilot` task uses a CUDA profile and the same verified
pilot input inventory, then measures the selected variant at its actual context
limit. It does not train or export weights. For compact-v3, use the 900-second
`modal-l4` profile, `--variant enriched`, and
`--pilot-config configs/head_finetuning/compact_pilot_v3.json`. The saved report
must pass the requested batch and unchanged-weight checks before training.

MLX is Apple-specific and is not an NVIDIA training backend. Head training uses
FP32 on either MPS or CUDA. CUDA performance, inference precision differences,
probability fidelity, and model quality still need measurement on real cloud
hardware. The **bounded 256-row/10-update** smoke trainer and prepared
**5,000-row/four-epoch** head pilot are supported. The user requested preparing
the enriched pilot first; see [modal-head-pilot.md](modal-head-pilot.md).
Those historical pilot tasks keep the encoder frozen. The separate current
`train-encoder` task implements upper-block adaptation; optimizer resumption
remains unavailable. The cached
enriched L4 pilot is complete; see [modal-head-pilot.md](modal-head-pilot.md)
for results and interrupted-run history. The pilot task does not silently start
the baseline variant. Compact-v3 subsequently passed a separate 2,048-token
qualification and completed enriched-only training; see
[compact-enriched-training.md](compact-enriched-training.md).

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
uv pip install --python .venv-modal/bin/python -r requirements/modal-client.lock
.venv-modal/bin/python -m modal setup
```

Authentication is already complete on this machine. Use `python -m modal`
because the executable is not on the shell's normal PATH. No need to rerun setup
for each experiment. `requirements/modal-client.lock` records the installed
SDK (1.6.2) and local tracking dependencies. The worker has its own Linux/CUDA
resolution in `requirements/modal-worker.lock`, compiled for Python 3.12 Linux
x86_64 from `requirements/modal-worker.in`; the image installs that lock.
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
seconds; `modal-l4-pilot` explicitly uses 14,400 seconds. Validated profiles can
change GPU and timeout (60 seconds–24 hours).
A timeout is a runtime bound, not a dollar spending limit. GPU, CPU, memory and
storage are separately billable; do not assume a particular credit balance.

The pilot uploads a verified 20-file inventory (75,980,393 bytes; roughly
20.6 MB compressed): paired prepared artifacts, bundled research profiles, and
original training/development JSONL for parity checks. Calibration and
confirmation label exports and payload tables are excluded. Baseline artifacts
are needed for complete manifest verification even for an enriched-only job;
only the selected variant trains.

Other input inventories are explicit:

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
excluded from routine result downloads, including pilot best/latest and
decision-only weights. Use `checkpoint_directories` in `modal_receipt.json` or
`status.json` to download them explicitly through Modal's volume CLI. Training
metric steps/timestamps survive local MLflow import; worker output is mirrored
into Modal logs.
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
management. The launcher now submits a detached app and a durable spawned function call,
then waits to download/import results. Closing the local client no longer
intentionally stops training. App/call IDs are saved before waiting; use
`fetch_modal_job.py` to recover results if the waiting client disconnects.

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

The pilot isolated input-bundle preflight passed locally on MPS: all native
tokens and runtime states matched. Modal app/image definitions construct locally
without creating resources. Tests cover device routing, recipe integrity, bundle
transport, weight exclusions and MLflow curve import. Latest logs are
`runs/modal_pilot_torch_tests.log` and `runs/modal_pilot_mlx_tests.log`.

The first enriched pilot image has now built and its L4 worker is training.
Input volume transport and native CUDA input checks passed. Numerical fidelity
against MPS, full result download and MLflow import remain unverified end to end. The next bounded cloud validation is the 16-input smoke evaluation,
followed by the 10-update head smoke only if desired. Cloud execution does not
establish Codabench runtime compatibility.

## GPU recommendation and cost estimates

Pricing checked **2026-10-22**. Start with the **Starter plan and one NVIDIA
L4**, using a normal Modal Function through the existing `modal-l4` profile.
Starter has no subscription fee and includes $30/month of compute credits;
the account's remaining credit balance has not been checked. Team is not
needed for the current workflow.

Current GPU-only base rates, converted from the published per-second charges:

| GPU | USD per second | Approx. USD per hour |
| --- | ---: | ---: |
| T4 | 0.000164 | 0.59 |
| **L4** | **0.000222** | **0.80** |
| A10 | 0.000306 | 1.10 |
| A100 40 GB | 0.000583 | 2.10 |

Source: [Modal pricing](https://modal.com/pricing). CPU, system RAM and storage
are additional. These are base Function rates, without region selection or
non-preemptible premiums.

The L4 has [24 GB of GPU memory](https://www.nvidia.com/en-us/data-center/l4/).
It is expected to fit the current head-only experiment comfortably, but actual
CUDA memory consumption and throughput have not been measured. This is a
starting recommendation, not a demonstrated cost/performance winner.

For our configured **2 physical CPU cores and 16 GiB system RAM**, estimated
compute charges are:

| Component | Approx. USD per billed hour |
| --- | ---: |
| L4 GPU | 0.7992 |
| 2 CPU cores | 0.0943 |
| 16 GiB RAM | 0.1279 |
| **Total** | **1.0214** |

That is approximately **$0.26 for 15 billed minutes**. Calculation:
`3600 × (0.000222 + 2 × 0.0000131 + 16 × 0.00000222)`.
Modal bills CPU/RAM at the greater of requested or actual usage, so this
estimate assumes usage stays within those allocations. Model loading and the
billed container tail contribute to billed time; it is not just time spent
performing optimizer updates. Storage and any separately billable services
are excluded. The 900-second function timeout is not a total spending cap.

The proposed first hardware checks are the existing 16-input evaluation and
10-update training smoke check on L4. Measure runtime, GPU memory, numerical
fidelity and cost per completed experiment before selecting hardware for a
longer pilot. These cloud checks have not yet run.

At the same CPU/RAM allocation, A10 costs approximately $1.3238 per billed hour.
It would need about **30% higher throughput** than L4 to break even on compute
cost for the same workload. No such speedup has been measured. Benchmark an
A100 later if encoder training or longer contexts warrant it; hourly price
alone does not determine experiment cost. There is no need to reserve a
continuously running machine for the present batch workflow.

## Official documentation checked

- [Apps and invocation](https://modal.com/docs/guide/apps)
- [GPU configuration](https://modal.com/docs/guide/gpu)
- [Images and dependencies](https://modal.com/docs/guide/images)
- [Volumes and persistence](https://modal.com/docs/guide/volumes)
- [Project structure](https://modal.com/docs/guide/project-structure)
- [Current pricing](https://modal.com/pricing)
