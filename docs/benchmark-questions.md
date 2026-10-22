# Competition inputs and Laya questions

The evaluation runner enforces the published
[competition predictor interface](https://aims.stanford.edu/competition#competition).
Every predictor receives only these fields, all as strings; missing values are
empty strings:

| Subject | Item |
| --- | --- |
| normalized_name | item_content |
| provider | item_features |
| release_date | interactors |
| access_date | benchmark_id (anonymous) |
| harness | |
| harness_version | |
| reasoning_effort | |
| subject_features_extra | |

`display_name` does not fill missing model names. Benchmark metadata, reference
answers, grading definitions, and target outcomes are not joined into inputs.
We preserve the supplied contents of the permitted fields, even if their text
mentions a benchmark. This is a field boundary, not an attempt to redact the task.

Real benchmark names remain inside the scorer for dataset selection, repeatable
splits, and reports. At the predictor boundary, local names become stable
synthetic IDs such as `benchmark_4fa1943e83611e20`. Target items and acquired
examples use the same mapping. These IDs do not reproduce private test IDs;
the code does not infer hidden benchmark identities or select prompts by ID.

## One shared question

The only active question definition is `configs/laya_question.json`:

```json
{
  "type": "noul",
  "instructions": "Will the target AI system, using the supplied settings, solve this task correctly on a single attempt?"
}
```

Laya returns the probability of “yes” directly from `answers.correct.noul`.
`correct` is our output key. There are no separate criteria or instructions
about estimating probabilities, recorded outcomes, or hidden grading rules.
Pydantic validates the question's type and nonempty instructions and rejects
additional fields such as criteria.

The state supplies two things: the available target AI identity/configuration
and the complete stored task, including context, options, features, and tools.
Empty fields are omitted from this rendered text; they remain empty strings in
the competition interface dictionaries. Model identity is never filled from
unavailable fields.

Anonymous benchmark IDs are retained by the code solely to match evidence.
They are omitted from both target and example text sent to Laya. The shared
question is independent of the benchmark identifier, real or anonymous.

The earlier `configs/benchmarks/*.json` definitions are public-data research
notes; the evaluator and Laya do not load them. `--benchmark-registry` and
`--laya-question-mode` are no longer accepted. There is one active question.

This prompt targets correctness/success. Some public tables instead record
other events, so it must not be treated as an established predictor of every
native public label. We preserve source grades and document that unresolved
issue in [outcome-semantics.md](outcome-semantics.md).

## Revealed labels

At budget zero, `labeled` is empty. At budgets 1, 3, 7, 15, and 31 the local
scorer supplies only the corresponding nested acquisition examples. Evaluation
outcomes stay scorer-side. All evidence uses the same field projection and
anonymous IDs as the target.

Laya is zero-shot by default. Add `--laya-use-labels` to include revealed
examples from the matching subject and anonymous benchmark in its state. Each
example includes its available task fields and a numeric `outcome`, copied
from the revealed label. The shared subject profile appears once in the target
system section. IDs and empty fields are omitted from examples too. The adapter
does not consult a public dataset to interpret the labels. Long evidence can
exceed the checkpoint context limit; inspect the logged truncation counters.

`LayaPredictor.predict(input, labeled)` accepts the published two-dictionary
input layout. The CLI still uses a batch helper for efficient local scoring.
The pretrained checkpoint is unchanged. The same shared predictor supports
`--laya-backend mlx` (default, `.venv-laya-mlx`) and `--laya-backend torch`
(reference, `.venv-laya`). See [laya-backends.md](laya-backends.md).

## Run a comparison

```bash
.venv-laya-mlx/bin/python scripts/run_eval.py --suite smoke --predictor constant empirical_mean laya
.venv-laya-mlx/bin/python scripts/run_eval.py --suite smoke --predictor laya --laya-use-labels
```

Pass `--laya-revision COMMIT` to pin the pretrained weights when comparing runs.

## Tracking and historical runs

New runs carry `input_contract=codabench_v1` as an MLflow parameter/tag and in
saved run metadata. Laya output folders include that contract name, separating
them from earlier experiments. The prompt is separately marked
`laya_prompt_version=task_correctness_v1` and output folders include that version.
MLflow records the exact shared question/hash and up to eight actual
state/question examples at budget zero. `competition_input_examples.jsonl` contains the corresponding budget-zero
interface calls and is also saved beside local scores. Scorer-side benchmark
names in reports and artifact annotations are not passed to the predictor.
The interface JSONL retains anonymous IDs; the rendered Laya state omits them.

Earlier benchmark-specific and generic-outcome/criteria runs remain historical
records. Their scores used different input
assumptions and should not be treated as results of this corrected setup.
Public-data local tests are still not a hosted Codabench validation or evidence
of generalization to the hidden benchmark pool.

Prior and text controls are trained on public benchmarks outside the entire
selected evaluation suite. Cache names include the training selection and data
manifest fingerprint; text cache names also include the seed. Old caches are
not reused by the corrected runner.

## Larger random samples

Use `--suite sample_10k` to score 10,000 system–question records from 100 random
system–benchmark pairs. This suite draws a fresh seed per command and saves
that seed and complete acquisition/evaluation item IDs in `sample_manifest.csv`.
Within a command, every predictor uses exactly the same sample. Pass the saved
`--seed NUMBER` for controlled prompt/model comparisons. See
[laya.md](laya.md) for sampling, inference-count, and text-only limitations.
