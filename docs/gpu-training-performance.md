# L4 training performance probes

Measured 2026-10-22 during the first enriched Modal pilot, at the user's request.
The actual GPU is NVIDIA L4, not T4. Training remains on the existing FP32
recipe. No optimizer updates or scientific configuration changes were applied
by these probes, and no additional GPU was rented.

## Results

Each isolated probe loads its own original parent and times forward/backward
work for the same first frozen 64-record training window. No optimizer step is
applied. The active trainer is briefly paused, then resumed, with a separate
watchdog. The probes preserve the effective 64-record update, but compare
different physical batches and padding. Single-window timings are exploratory,
not full-run throughput estimates. They exclude model loading, cache population
and checkpoint export. Update timers spanning a profiling pause are inflated;
do not interpret those outliers as slow computation.

| Method | Physical batch | Seconds per warmed window |
| --- | ---: | ---: |
| Existing FP32, original order | 2 | 7.78 |
| FP32, original order | 64 | 8.49 |
| FP32, similar lengths grouped within window | 8 | 6.60 |
| TF32, similar lengths grouped | 8 | 3.79 |
| FP32, cached encoder, original order | 2 | 1.35 |
| FP32, cached encoder, similar lengths grouped | 8 | 1.08 |

The ordinary 64-input batch allocated about 13.8 GB versus 2.2 GB at batch two,
but padded more tokens and was slower. Sampled device busy time was 96–100%.
That metric is not proof of maximum useful throughput or tensor-core occupancy.

TF32 retained FP32 weights but changed probabilities on a 64-record check by
up to 0.009154 (mean absolute difference 0.000575). It was not enabled for the
active experiment. Faster arithmetic is a separate numerical-policy experiment.

## Best memory opportunity: cache the frozen encoder

The encoder is frozen and evaluated without dropout, while its output feeds
the trainable type embeddings and decision head. Its repeated outputs can be
reused across epochs without caching trainable head activations or predictions.
The prototype stores per-input, unpadded FP32 encoder features on the GPU and
reassembles padded batches. Keys are exact native token IDs; labels are not
inputs to the cache. Sequential layout and a fully frozen encoder are required.

For the 1,024-wide encoder, prepared token lengths plus native question overhead
estimate these cache sizes:

| Role | Estimated FP32 feature bytes |
| --- | ---: |
| Training | 14,033,408,000 |
| Development | 937,967,616 |
| Total | 14,971,375,616 (about 15.0 GB / 13.9 GiB) |

These are estimates, not a full-corpus allocation measurement. Model, gradients,
optimizer, temporary activations and allocator buffers need additional space.
The L4 reports 23,034 MiB total. A production cache needs an explicit byte cap,
headroom and controlled eviction rather than assuming every record fits.

The two-record correctness fixture matched probabilities exactly. With identical
CUDA RNG restored, maximum absolute head-gradient difference was 1.86e-9. The
timed cached window contained 64 records. This does not yet establish full-corpus
equivalence across padding/grouping combinations or sustained cache hit rate.

Before making caching a training option, implement a bounded cache behind the
encoder interface, verify representative mixed-length probabilities and gradients,
preserve RNG and frozen-state checks, record cache hits/bytes, test exports, and
measure cache-population cost. Cache identity must include the frozen checkpoint,
tokenization and input contract. Clear it if those change. Encoder fine-tuning
cannot use this frozen-feature cache.

The running process cannot pick up this new execution path safely. It has already
completed its first epoch; no restart or silent mid-run numerical change was made.
The first epoch's raw development Brier ALC was 0.249080, a partial training
result rather than a completed four-epoch experiment.

Other candidates remain unmeasured: reducing host/GPU synchronization for loss
logging, compiled kernels with bounded shapes, and BF16 training. Prioritize the
measured caching opportunity over these additional precision/compilation changes.

## Production integration and completed measurement

After the prototype, the user authorized retraining. A bounded production
cache is implemented in `src/paiec/encoder_cache.py` and the head pilot.
The fresh cached run completed 316 updates. All-development probabilities
matched the native pre-update model exactly; training fixture probability
drift was 7.90e-7 and gradient drift 1.68e-8. Model export/reload passed.
Actual cache: 12.14 GB / 5,215 unique entries, zero evictions; max reserved
GPU memory 15.31 GB. Median warm update time was 1.40 seconds, versus 7.05
seconds for the cold first epoch. Full training/development phase took
16.9 minutes, excluding setup and pre-training validation. See
[completed results](modal-head-pilot.md#completed-cached-retrain).

## Evidence

Reusable probe: `scripts/benchmark_head_batches.py` (optional `--tf32` or
`--cache-encoder`, existing CUDA worker required). Raw logs:
`runs/modal_batch_benchmark.log`, `runs/modal_batch_benchmark_tf32.log`,
`runs/modal_batch_benchmark_cache.log`. Structured audit:
`runs/modal_gpu_utilization_audit.json`. These are performance probes, not
training-quality runs. The separate probe never modifies the active model.
