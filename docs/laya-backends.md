# Interchangeable Laya backends

The user authorized this integration on 2026-10-21. **MLX is now the default**;
the original PyTorch Laya remains selectable. The canceled 3,000-record run
was not restarted, recomputed, or relabeled.

## Shared interface

```text
competition fields → shared state/question/cache → selected runtime
                   → shared probability validation → scoring / MLflow
```

`LayaPredictor` remains the public prediction interface. State rendering, the
short correctness question, optional revealed evidence, deduplication, and
scoring are independent of the backend. `src/paiec/laya_runtime.py` defines a
small `LayaRuntime` protocol, a factory, and a common native-Agent adapter.
Both native libraries implement `predict_batch(states, questions, ...)`, so the
adapter normalizes our request dictionaries without changing their contents.

The factory imports only the selected framework. The MLX environment has no
PyTorch or Transformers installation. An unavailable/unsupported backend raises
an actionable error; there is no silent fallback to another framework.

```python
from paiec.laya_predictor import LayaPredictor

predictor = LayaPredictor(backend="mlx", dtype="float32")
probability = predictor.predict([subject, item], labeled)
predictor.close()
```

Use `backend="torch"` in the reference environment to select ordinary Laya.
Its precision policy is managed by upstream Laya, so leave `dtype="auto"`.
The injected `router` argument remains available for lightweight unit tests;
production instances are backed by the selected native Agent.

## Environments and commands

The original `.venv-laya` environment was preserved. The new
`.venv-laya-mlx` environment uses Python 3.12.13, Laya-MLX 0.5.0, MLX/MLX Metal
0.32.4, and MLflow 3.17.2. Both environments share local data and tracking files.

Recreate the tested MLX/tracking environment:

```bash
uv venv --python 3.12 .venv-laya-mlx
uv pip install --python .venv-laya-mlx/bin/python -r requirements-mlx.lock -e .
```

For a fresh dependency resolution, install the `mlx` and `tracking` extras
instead. This is a new environment, not an upgrade of the PyTorch reference:

```bash
uv pip install --python .venv-laya-mlx/bin/python -e '.[mlx,tracking]'
```

The same evaluator selects either backend:

```bash
# New default: native MLX GPU, FP16.
.venv-laya-mlx/bin/python scripts/run_eval.py --suite smoke --predictor constant empirical_mean laya --laya-backend mlx --laya-batch-size 8

# Original runtime: PyTorch MPS, upstream automatic precision.
.venv-laya/bin/python scripts/run_eval.py --suite smoke --predictor laya --laya-backend torch --laya-batch-size 8

# MLX FP32 for closer probability agreement with the reference.
.venv-laya-mlx/bin/python scripts/run_eval.py --suite smoke --predictor laya --laya-backend mlx --laya-dtype float32
```

`--laya-model english|multilingual`, the prompt, label mode, seed, and context
limit work with either backend. `--laya-device auto` selects MLX GPU or the
usual PyTorch device; `mps` maps to MLX GPU too. `cpu` is selectable in both;
`cuda` is valid only with PyTorch. MLX requires an Apple Silicon Mac and is not
an Apple Neural Engine backend.

## Same original weights and explicit precision

Both backends load the **original** `convaiinnovations/laya` checkpoint, pinned
by default to `7b928d828b7b0e022f929d9bd2e44165aa270148`:

- English uses the repository root; multilingual uses `multilingual/`.
- The shared resolver downloads/selects only the relevant checkpoint files.
- The MLX port loads these upstream safetensors directly and converts parameter
  names/dtypes in memory. We do not substitute a community Hub checkpoint or
  silently change to a fine-tuned model.
- SHA-256 hashes of weights, agent config/calibration, encoder config, and
  tokenizer JSON are recorded. The validation compared those hashes, not just
  repository names.
- `--laya-dtype auto` means FP16 for MLX. FP32 is available explicitly.
  Ordinary Laya retains its native policy: FP32 stored weights and automatic
  mixed precision, with MPS autocast gated by batch size. Tracking records this
  as `upstream_auto`, along with autocast dtype and threshold, not a promise
  that every operation uses one fixed dtype.

## Validation on the user's M4

All **116 shared unit tests pass in both environments**. Real smoke evaluations
completed for PyTorch English, MLX English, and MLX multilingual. No large
evaluation was launched for this integration.

The English probability check used eight saved budget-zero interface examples
(four BBQ and four AI2D) plus one synthetic long input. Requests, prompt, source
weight/config/tokenizer hashes, and 512-token context limit matched. Reference
probabilities used state batches of one; this avoids PyTorch's batch-size-gated
MPS autocast. Native libraries report probabilities rounded to four decimals.

| Backend | Largest difference from reference, 8 dataset examples | Largest difference including synthetic long input |
| --- | --- | --- |
| PyTorch reference | 0 | 0 |
| MLX FP32 | 0 | 0 |
| MLX FP16 | 0.0023 | 0.0121 |

Input/state token counts, truncation flags, and dropped-state-token counts
matched across these English checks. Exact reported FP32 agreement on these
fixtures is not a guarantee of bitwise or arbitrary-input equivalence.

A separate nine-input multilingual-checkpoint validation used the same eight
dataset records plus a synthetic long task. MLX FP32 again matched every
reported reference probability and all checked token/truncation counters.
Thus FP32 agreement was checked on 18 model/input combinations across the two
checkpoints. The table above reports English FP16 differences only.

Warmed timings used the eight dataset examples, batch size eight, three timed
repeats, a warm-up, and device synchronization. They bypassed the predictor's
answer cache and excluded model initialization/hashing:

| Runtime / precision policy | Median seconds per 8-input batch | Inputs/second |
| --- | --- | --- |
| PyTorch / upstream automatic | 1.2788 | 6.26 |
| MLX FP32 | 1.3983 | 5.72 |
| MLX FP16 | 1.0362 | 7.72 |

MLX FP16 delivered about **1.23× throughput** in this small measurement; FP32
was slower here. These are different precision policies, not a pure matched-
precision speed comparison. Do not extrapolate to all task lengths, batch
sizes, or the full benchmark mix. Backend selection does not improve the
pretrained predictor's underlying ability to estimate another AI's success.

Local evidence is in `runs/backend_validation_*.json`,
`runs/backend_validation_summary.json`, and the corresponding logs. The worker
is `runs/backend_validation_worker.py`; it records request hashes, probabilities,
usage, runtime provenance, and timings. Complete smoke scores remain exploratory
16-input checks, not a model-quality study.

## Memory: why MLX needed a cache cap

The MLX backend is the default but had a memory regression the PyTorch backend
did not. MLX keeps freed Metal buffers in a size-classed cache instead of
returning them to the OS, and **each distinct tensor shape gets its own
blocks**. Laya states vary in token length, so nearly every batch is a new
shape and the cache grows monotonically over a run.

Its defaults made this worse: both the cache limit and the memory limit were
`16.32 GB`, which is the whole machine, and above the `12.71 GB` GPU
`max_recommended_working_set_size` reported on this 16 GB Mac. Nothing evicted
until well past the point where the OS starts swapping. PyTorch is fine because
`torch.mps` uses a caching allocator bounded by `PYTORCH_MPS_HIGH_WATERMARK_RATIO`
(default 1.7), which does evict under pressure.

Measured growth on real data before the fix, batch size eight: cache
`0.78 → 1.57 → 2.35 → 3.03 → 3.72 GB` over 48 BBQ items, with only ~31 distinct
item lengths. Repeating an identical shape costs nothing; a new length costs
roughly one 0.78 GB block at this batch size. Extrapolated to a realistic
benchmark mix that reaches 10–12 GB in the low hundreds of items, which matches
the earlier memory exhaustion.

The runtime now caps both limits in `create_runtime` before the checkpoint
loads, and `LayaPredictor.trim_cache()` releases cached buffers after each
scoring pair in `scripts/run_eval.py`:

| Setting | Value | Override |
| --- | --- | --- |
| `mx.set_cache_limit` | 2 GiB | `PAIEC_MLX_CACHE_LIMIT_GB` |
| `mx.set_memory_limit` | 6 GiB | `PAIEC_MLX_MEMORY_LIMIT_GB` |

Both are recorded in `runtime.json` and runtime metadata, so a run's actual
limits are auditable. The intent is to fail loudly on a genuinely oversized
allocation rather than degrade the whole machine into swap. Overrides are
refused if unparseable or non-positive, so a typo cannot silently disable the
guard.

Measured after the fix, using real benchmark tables and no evaluation suite:

| Scenario | Peak resident | MLX cache |
| --- | --- | --- |
| One 100-item pair, no trim | 1.47 GB | plateaus at 2.15 GB, then flat |
| 216 items across three benchmarks, trim between pairs | 1.53 GB | 0.00 GB after each pair |

Resident memory stayed flat while the cache cap held, so the guard and the
eviction path both work. Eviction cost was then measured directly: the same
100-item BBQ pair at batch size eight, run twice each way, with a warm-up
excluded.

| Cache handling | Items/second | MLX cache after |
| --- | --- | --- |
| No trim | 15.3 / 15.2 | 2.15 GB (capped) |
| Trim after each batch | 15.0 / 14.8 | 0.00 GB |

That is roughly a **2% throughput cost**, within run-to-run noise but
consistently slower. It buys flat resident memory, which is the trade worth
making here. These are memory/throughput probes on real tables, not evaluation
runs, and they do not change the interrupted-run record.

## Tracking and limits

Run names, parameters, and default output folders now include backend and
precision policy, separating MLX, PyTorch, and historical runs. `runtime.json`
records actual device, source revision/subfolder, artifact hashes, backend
package versions, precision policy, and initialization seconds. Evaluation
timing synchronizes the selected device and excludes initialization; old
timings that included lazy model loading are not directly comparable.

The competition input boundary and `task_correctness_v1` prompt are unchanged.
The known public-event-label ambiguity, missing image inputs, text truncation,
unknown pretrained overlap, and unverified hosted Codabench compatibility remain.
There is still no incremental prediction checkpointing or graceful partial
score finalization. The previously interrupted run remains untouched.

Upstream port: [Laya-MLX](https://github.com/mizorewww/laya-mlx).
