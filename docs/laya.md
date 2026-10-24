# Running pretrained Laya

Status as of 2026-10-24: **Laya work is on hold** after the completed grouped
pilot failed to beat the constant baseline. This is a retained experiment or
implementation reference; proposals and commands below are not a current work
queue. See [current status](current-status.md) and [shared data](shared-data-start.md).

This setup uses the released, unchanged Laya checkpoints. It performs no
fine-tuning, paid API calls, or public-corpus retrieval. The first run downloads
the selected weights from Hugging Face; later runs reuse the local cache.

The runner asks one shared question from `configs/laya_question.json`:
“Will the target AI system, using the supplied settings, solve this task
correctly on a single attempt?” Laya already returns a probability, so there
are no added criteria or probability instructions. See
[benchmark-questions.md](benchmark-questions.md) for the exact input contract.

See [laya-backends.md](laya-backends.md) for the current shared MLX/PyTorch
interface, setup, and measured probability/performance checks. The earlier
[project handoff](project-status.md) records pre-MLX history. The user authorized
this integration; the canceled large run remains canceled.

## Environments

The default backend is native MLX. Run it from the separate MLX environment:

```bash
uv venv --python 3.12 .venv-laya-mlx
uv pip install --python .venv-laya-mlx/bin/python -r requirements/mlx.lock -e .
```

For a fresh dependency resolution, use `-e '.[mlx,tracking]'` instead of the
lock. The tested environment uses Python 3.12.13, Laya-MLX 0.5.0, MLX/MLX Metal
0.32.4, and MLflow 3.17.2. It has no PyTorch or Transformers dependency.

The original `.venv-laya` environment is preserved for ordinary Laya 0.4.13,
PyTorch 2.14.1, Transformers 5.20.2, and tracking. Recreate it with
`requirements/torch.lock` plus the editable project if needed. Select it with
`.venv-laya/bin/python` and `--laya-backend torch`. The older
`requirements/archive/laya-pretracking.lock` predates the MLflow dependency adjustments.

MLX `--laya-device auto` uses the Mac GPU and `--laya-dtype auto` means FP16.
Use `--laya-dtype float32` for closer numerical agreement. Ordinary Laya's auto
selection prefers MPS on this Mac, then CUDA, then CPU, retaining its upstream
automatic precision policy. `--laya-device cpu` is available in both backends.
There is no silent framework fallback and neither Mac path uses the Neural
Engine.

MLX defaults to caching up to 16.32 GB of freed Metal buffers, which exceeds
this machine's 12.71 GB recommended working set and let long runs climb into
swap. The runtime caps the MLX cache at 2 GiB and its memory limit at 6 GiB
(`PAIEC_MLX_CACHE_LIMIT_GB`, `PAIEC_MLX_MEMORY_LIMIT_GB` override both) and
clears the cache between scoring pairs. See `docs/laya-backends.md` for the
measurements. Peak resident memory on real data stayed near 1.5 GB over 216
items after the change.

Both runtimes load the same original Hugging Face files, with default revision
`7b928d828b7b0e022f929d9bd2e44165aa270148`. Weights remain cached outside Git;
model/config/tokenizer hashes are recorded for provenance.

## First test

```bash
.venv-laya-mlx/bin/python scripts/run_eval.py --suite smoke --predictor constant empirical_mean laya
```

This scores eight evaluation items for each of two subject-benchmark pairs,
with all six label budgets. It checks that the pipeline works; it is too small
to decide model quality. The unchanged model receives the subject's visible
profile and the target item, and answers that one `noul` question. The adapter
reads the returned probability of yes, not its confidence in an action.
Only available subject/task details appear in the rendered state; empty fields
and benchmark IDs are omitted. Anonymous IDs remain in code to match evidence,
and real benchmark names remain scorer-side for splitting and reports.

By default, revealed labels are ignored by Laya. Its Brier curve is therefore
flat across budgets, and identical inputs reuse their predictions. Empirical
mean uses the revealed labels and can change across budgets. Compare Brier@0
for a comparison without revealed examples, and Brier ALC for the complete
competition-shaped comparison.

## Larger experiments

The `sample_10k` suite selects 100 system–benchmark pairs and 100 evaluation
questions per pair. Subject selection is random and balanced across benchmarks
as far as eligible subject availability allows. BBQ has only seven eligible
systems, so its unused pair allocation is redistributed. Each selected pair
must have at least 200 distinct binary-graded items, supporting a 50/50 split
and 100 evaluation items. Acquisition pools remain disjoint and capped at 40.

```bash
.venv-laya-mlx/bin/python scripts/run_eval.py --suite sample_10k --predictor constant empirical_mean laya --laya-batch-size 8
```

A fresh seed is generated each run. The same subjects, questions, split, and
acquisition order are reused across all predictors within that run. Repeat
`--seed NUMBER` from a previous run when comparing prompt or model changes;
otherwise sampling variation can confound the comparison. The original smoke
and default suites retain their fixed configured seeds unless overridden.

MLflow records `seed`, `sample_sha256`, `n_eval_inputs`, and the full
`sample_manifest.csv` of acquisition/evaluation IDs (including their order).
`n_eval_inputs` counts system–question records, not unique question texts.
`forward_items` counts actual Laya inference inputs; identical observable states
can reuse a cached prediction. Zero-shot predictions are reused at later label
budgets. `n_pairs` counts system–benchmark combinations and remains the unit
of equal-weight scoring. During Laya execution, `completed_pairs`,
`forward_items`, and `truncated_items` update in MLflow.

Random-suite output folders end in `seed_NUMBER`, preserving separate samples.
These remain public-data experiments; the competition's hidden split is fixed.
The suite retains image-dependent AI2D/DevBench items and multilingual tasks,
while this adapter reads stored text only. Inspect per-benchmark results and
truncation before interpreting an aggregate score as model quality.

The `default` suite has 32 pairs across eight benchmarks and up to 120 evaluation
items per pair:

```bash
.venv-laya-mlx/bin/python scripts/run_eval.py --suite default --predictor constant empirical_mean laya --laya-model english
.venv-laya-mlx/bin/python scripts/run_eval.py --suite default --predictor constant empirical_mean laya --laya-model multilingual
```

Omitting `--laya-max-len` uses each checkpoint's own trained length, read from
its `rl_agent_config.json`: **512 for English, 1024 for multilingual**. A shared
512 default truncated multilingual states below the length that checkpoint was
fine-tuned at, so the flag is now optional and the resolved ceiling appears in
the run title, tracked parameters, and output folder.

Set `--laya-max-len` explicitly only for a deliberate comparison. A larger
limit can increase runtime and memory: the harness pads each item to the
ceiling, so cost is roughly linear in `max_len`. Inputs exceeding the limit are
truncated by Laya; `run.json` records the number of truncated forward inputs
and dropped state tokens. A bigger budget does not guarantee better
predictions, and going far past the trained length moves inference outside the
length the checkpoint was optimized for.

### Measured token distribution

Character counts mislead here; the state is tokenized as JSON. Rebuild the
exact zero-shot states for a suite's evaluation split with:

```bash
.venv-laya-mlx/bin/python scripts/measure_tokens.py --config runs/configs/sample_1k.json --suite sample_1k --seed 20261021 --json runs/token_distribution_sample_1k.json
```

This loads the pinned tokenizer, not model weights. On the `sample_1k` suite
(1,000 scored states, seed 20261021) the result was:

| Benchmark | Items | Median tokens | Max | Over 512 |
| --- | --- | --- | --- | --- |
| editbench | 100 | 1556 | 11443 | 89 |
| helm_afr | 100 | 804 | 3562 | 100 |
| cruxeval | 100 | 448 | 515 | 2 |
| afrimedqa | 100 | 283 | 536 | 1 |
| cmmlu | 200 | 308 | 509 | 0 |
| devbench | 100 | 286 | 313 | 0 |
| bbq | 100 | 204 | 272 | 0 |
| ai2d_test | 200 | 288 | 337 | 0 |
| **Suite** | **1000** | **296** | **11443** | **192 (19.2%)** |

Truncation at 512 tokens is concentrated in two benchmarks, editbench and
helm_afr. Half the suite never truncates at all. Shares over larger ceilings:
1024 → 9.6%, 2048 → 4.1%, 4096 → 1.8%, 8192 → 0.8%.

The truncated-forward counter in `run.json` is not this number. It counts
forward *records*, so a state shared across label budgets is counted once per
budget; the `sample_1k` run reported 221 truncated forwards against 192
distinct over-length states.

Batch size defaults to one for conservative memory use. Try
`--laya-batch-size 2` for batching. Progress is printed for each pair.

To test adaptation through examples, without changing the pretrained weights:

```bash
.venv-laya-mlx/bin/python scripts/run_eval.py --suite smoke --predictor constant empirical_mean laya --laya-use-labels --laya-max-len 2048
```

This appends available task details and observed 0/1 labels from the same
subject and anonymous benchmark. The subject profile is not repeated in every
example. The target comes first; long targets or many examples may exceed the
context budget. This is an experimental input format, not a fine-tuned
adaptation method.

## Results and reproducibility

Outputs are separated by suite, backend/precision, checkpoint, context length,
evidence mode, and prompt version:

```text
runs/eval_smoke/laya_mlx_float16_codabench_v1_english_512_zeroshot_task_correctness_v1/
  summary.csv             comparison across predictors, lower Brier is better
  scores_laya.csv         scores for each pair and label budget
  scores_constant.csv
  scores_empirical_mean.csv
  suite.json              exact suite configuration
  run.json                checkpoint commit, package versions, device, timing,
                          and truncation counts
```

`run.json` records the exact Hugging Face commit resolved before loading either
checkpoint. To reproduce those weights, pass `--laya-revision` with that commit.
Use `--out runs/my_experiment` for a separate output directory when changing
the suite configuration or comparing other settings. Repeating a run in the
same output directory replaces that predictor's saved scores and run metadata.

The English checkpoint currently emits an upstream temperature warning for
`choice:11+`. This experiment uses the separate `noul:2` temperature; the
checkpoint's temperature values are left unchanged by this project.

These are public-data exploratory experiments. The pretrained model's overlap
with these benchmarks is unknown. The existing subject-prior and text-ridge
paths are not used by these commands. Before fine-tuning, create explicit
benchmark-level training/validation/test partitions.

## Checks

```bash
.venv-laya-mlx/bin/python -m unittest discover -s tests -p 'test_*.py'
```

The adapter tests use a fake router and need no model download. The smoke
commands above verify real checkpoint inference on the downloaded payload.

The latest backend-integration test runs are recorded in `runs/mlx_tests.log`
and `runs/torch_tests.log`. The earlier sampling implementation passed 105
tests; the current suite adds runtime-interface checks, including the MLX
memory-limit setup order and the trim hook, plus the per-checkpoint context
default. It now passes 119 tests in both environments.
Before MLX integration, `task_correctness_v1` English MPS smoke runs completed
with and without revealed examples at a 512-token limit. The zero-shot run
processed 16 inputs without truncation; the example-enabled run processed 96
inputs, of which 73 were truncated. Both modes produced identical budget-zero
scores. These checks establish execution, not predictive quality.

Historical results from the earlier input/prompt setup are retained below.
These are not results of `task_correctness_v1`. Real MPS smoke runs completed
with the following Brier ALC:

| Predictor | Context limit | Brier ALC |
| --- | --- | --- |
| Constant 0.5 | — | 0.2500 |
| Empirical mean | — | 0.2925 |
| English Laya, no revealed examples | 512 | 0.2909 |
| Multilingual Laya, no revealed examples | 512 | 0.2067 |
| Multilingual Laya, with revealed examples | 2048 | 0.2804 |

These runs used checkpoint commit
`7b928d828b7b0e022f929d9bd2e44165aa270148`. Neither run without examples
truncated its 16 inputs. The run with examples used batch size two and truncated
8 of 96 forward inputs. This tiny sample verifies execution, not comparative
model quality; repeat on the larger suite before drawing conclusions.

Upstream API reference: [Laya repository](https://github.com/NandhaKishorM/laya).
