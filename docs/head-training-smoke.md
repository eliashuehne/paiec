# Head-only training smoke check

Authorized on 2026-10-22. This is a bounded machinery check, not a model-quality
experiment. Its result and timings are recorded in
`runs/head_finetuning/smoke_v1/report.json` and `history.json` when complete.

## Recipe and boundaries

- Pinned original English checkpoint; unchanged JSON state and correctness question.
- 256 deterministic training-role responses from the frozen `data_v2` export,
  selected without consulting outcomes. Every selected state and question fits
  the 512-token sequence; native tokenization rechecks this before training.
- Round-robin sampling across eligible system–benchmark pairs, with equal total
  loss weight per selected pair. This short-input subset represents fewer
  families than the full pilot and is unsuitable for judging generalization.
- 10 AdamW updates, learning rate 0.0001, weight decay 0.01, gradient clipping 1.
  Micro-batch 4 with accumulation 16: 64 presentations per update, 640 total.
  The 256 selected responses repeat as the sampler cycles.
- PyTorch/MPS FP32. Only `head`, `type_emb`, and `scorer` train: 26,248,193 of
  421,293,827 parameters. Encoder and auxiliary action head remain frozen.
  Encoder dropout stays disabled. No option shuffling, smoothing, RLCD,
  acquired examples, temperature fitting, or held-out scoring.
- Weighted supervised cross-entropy uses one denominator per optimizer window;
  accumulation does not accidentally give small micro-batches extra weight.

The wrapper is `scripts/train_head_smoke.py`, with validated settings in
`configs/head_finetuning/smoke_v1.json`. It refuses more than 256 rows or 10
updates. A full pilot needs a separate training configuration and development
selection loop; it is not launched by this script.

```bash
.venv-laya/bin/python scripts/train_head_smoke.py --out runs/head_finetuning/NEW_DIRECTORY
```

The output directory must not exist. Each run logs to **PAIEC Head Fine-tuning**
in MLflow. A smoke loss is cross-entropy on training examples, not Brier ALC.
Checkpoints stay in the ignored local run directory, with their location,
identity, and provenance in MLflow; the full weights are not copied into the
MLflow artifact store. Optimizer resumption is not implemented.

## Checkpoint integrity and explicit loading

The project saves FP32 safetensors to avoid the upstream training helper's
implicit FP16 conversion. Before and after training it hashes all encoder,
auxiliary-head, and decision tensors separately. The check requires the first
two groups to match exactly and the decision group to change. It then reloads
all saved tensors and compares predictions on the shortest and longest selected
training inputs. No held-out examples are needed for this test.

Saved config explicitly uses temperature 1 and clears inherited per-option
calibration. This produces raw trained probabilities; calibration remains a
later, independently held-out step.

The runtime now accepts `checkpoint_path` on `LayaPredictor`, or
`--laya-checkpoint DIRECTORY` in `scripts/run_eval.py`. It requires
`paiec_training.json`, verifies every listed artifact hash, and checks the
requested model. Runtime metadata, tracked run names, and default evaluation
output folders identify the local checkpoint separately. Omitting this option
continues to load the original pinned checkpoint. The shared prediction task
and input boundary are unchanged.

`scripts/check_head_checkpoint.py` compares the explicit local runtime with
saved direct predictions, then compares MLX FP32/FP16 with PyTorch on the same
two training inputs and one synthetic long input. This qualifies loading and
limited numerical fidelity; it does not establish equivalent scores across a
new evaluation suite.

## Next decision

After the machinery check succeeds, inspect which task fields survive among
the 54.22% of full training responses that truncate. Fix or explicitly accept
that limitation before a separate 5,000-response pilot with development scoring.
The confirmation split remains locked.

## Completed result

All 10 updates completed in **131.10 seconds**. All logged losses and gradient
norms were finite. Training cross-entropy ranged from 1.061 to 1.827; these are
different sampled/dropout batches and do not establish a quality gain.
The highest logged MPS driver allocation was **6.38 GB**; process peak resident
memory logged during updates was 2.72 GB. These are different measurements of
unified memory and must not be added together.

Direct comparison with the parent found **31 changed tensors**, all in the
allowed decision modules. Every encoder and auxiliary-head tensor matched the
parent exactly after promotion from the original stored FP16 to training FP32.
Reloaded FP32 predictions agreed exactly on two selected training inputs.
The original parent artifact hashes also remained unchanged.

The first run (`d47d2cab75314be3acba0d3128919528`) remains **FAILED** because,
after completing training and the tensor/reload assertions, upstream load-time
tokenizer normalization changed `tokenizer_config.json` and invalidated the
artifact manifest. The export now applies that normalization before recording
hashes, with a regression test. No additional gradient updates were performed.
The recovered artifact was independently verified in
`18d5895e73c94bc5b54dd706d84b98d7` (**FINISHED**). The initial manifest and recovery
script are retained with the verification artifacts. This preserves the failed
packaging history instead of rewriting it into a successful training run.

At the observed average of 13.11 seconds per update, the proposed 316-update
pilot would take roughly **69 minutes of training**, excluding initialization,
validation and saving. This is a rough extrapolation: the full dataset contains
many more long states than this smoke subset, so it may be slower. No full pilot
has started.

The full lightweight suite passed in both environments: 160 tests discovered,
159 passed/1 skipped under PyTorch and 157 passed/3 skipped under MLX. The skips
are environment-specific checks. No confirmation or development scores were
computed, and no calibration was fitted.


Explicit runtime qualification passed on two training inputs and a synthetic
truncated input. MLX FP32 matched all three reported PyTorch probabilities;
MLX FP16 differed by at most **0.0005** (0.05 percentage points). Input/state token
counts and truncation counters matched across all three runtimes. These are
three fixtures, not broad accuracy or precision equivalence. Saved outputs are
`runtime_torch_auto.json`, `runtime_mlx_float32.json`,
`runtime_mlx_float16.json`, and `runtime_qualification.json` alongside the report.


## Subsequent development evaluation

The user subsequently authorized Brier ALC measurement. It is now recorded in
[head-development-results.md](head-development-results.md): tuned raw 0.394842,
original raw 0.459007, original with shipped calibration 0.379140, 50% control
0.250000. Earlier statements above describe the smoke-check stage before this
evaluation. Confirmation and the dedicated calibration split remain untouched.
