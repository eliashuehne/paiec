# Running pretrained Laya

This setup uses the released, unchanged Laya checkpoints. It performs no
fine-tuning, paid API calls, or public-corpus retrieval. The first run downloads
the selected weights from Hugging Face; later runs reuse the local cache.

The runner asks one shared question from `configs/laya_question.json`:
“Will the target AI system, using the supplied settings, solve this task
correctly on a single attempt?” Laya already returns a probability, so there
are no added criteria or probability instructions. See
[benchmark-questions.md](benchmark-questions.md) for the exact input contract.

See [project-status.md](project-status.md) for the current handoff and measured
results. The large run was stopped at the user's request and must not be
restarted without a new instruction. A switch to the separate community
[Laya-MLX runtime](https://github.com/mizorewww/laya-mlx) is deferred until the
user explicitly requests it. The current environment uses PyTorch/MPS; no MLX
installation or migration has been performed.

## Environment

Run from the project root. `.venv-laya` is separate from the original `.venv`:

```bash
uv venv --python 3.12 .venv-laya
uv pip install --python .venv-laya/bin/python -r requirements-laya.lock -e .
```

The lock file records the package versions tested on Apple Silicon with Python
3.12.13, Laya 0.4.13, PyTorch 2.14.1, and Transformers 5.20.2. For a fresh
dependency resolution instead, install the optional extra:

```bash
uv pip install --python .venv-laya/bin/python -e '.[laya]'
```

Automatic device selection prefers MPS on this Mac, then CUDA, then CPU. Force
CPU with `--laya-device cpu` if needed. Only one checkpoint is resident per run.
Weights live in the normal Hugging Face cache under `~/.cache/huggingface/`,
outside Git. English weights are approximately 843 MB, excluding tokenizers
and configuration; multilingual downloads separately.

## First test

```bash
.venv-laya/bin/python scripts/run_eval.py --suite smoke --predictor constant empirical_mean laya
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
.venv-laya/bin/python scripts/run_eval.py --suite sample_10k --predictor constant empirical_mean laya --laya-batch-size 8
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
items per pair. Run both checkpoints at the same context budget:

```bash
.venv-laya/bin/python scripts/run_eval.py --suite default --predictor constant empirical_mean laya --laya-model english --laya-max-len 512
.venv-laya/bin/python scripts/run_eval.py --suite default --predictor constant empirical_mean laya --laya-model multilingual --laya-max-len 512
```

To test longer inputs, repeat with `--laya-max-len 2048`. A larger limit can
increase runtime and memory. Inputs exceeding the limit are truncated by
Laya; `run.json` records the number of truncated forward inputs and dropped
state tokens. A bigger budget does not guarantee better predictions.

Batch size defaults to one for conservative memory use. Try
`--laya-batch-size 2` for batching. Progress is printed for each pair.

To test adaptation through examples, without changing the pretrained weights:

```bash
.venv-laya/bin/python scripts/run_eval.py --suite smoke --predictor constant empirical_mean laya --laya-use-labels --laya-max-len 2048
```

This appends available task details and observed 0/1 labels from the same
subject and anonymous benchmark. The subject profile is not repeated in every
example. The target comes first; long targets or many examples may exceed the
context budget. This is an experimental input format, not a fine-tuned
adaptation method.

## Results and reproducibility

Outputs are separated by suite, checkpoint, context length, evidence mode,
and prompt version:

```text
runs/eval_smoke/laya_codabench_v1_english_512_zeroshot_task_correctness_v1/
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
.venv-laya/bin/python -m unittest discover -s tests -p 'test_*.py'
```

The adapter tests use a fake router and need no model download. The smoke
commands above verify real checkpoint inference on the downloaded payload.

The latest recorded full unit-test execution passed 105 tests; see
`runs/sampling_tests.log`. It was not repeated for the documentation handoff.
For `task_correctness_v1`, English MPS smoke runs completed
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
