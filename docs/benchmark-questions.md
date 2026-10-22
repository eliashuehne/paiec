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

## Opt-in input-design experiments

On 2026-10-22 the user authorized testing state formatting and short question
variants, including a system-description ablation. These experiments are run
separately by `scripts/run_input_experiments.py`, with definitions in
`configs/experiments/input_design.json`. They do not change the default question
or the ordinary evaluation CLI.

The shared predictor accepts a validated experimental question only with a
distinct prompt version. `state_format="text_v1"` renders projected fields as
readable text and expands nested JSON without adding information.
`omit_system=True` is a diagnostic ablation, not a production recommendation;
evidence matching still uses the original subject. Both options are explicit
and logged by the experiment script. The original JSON renderer remains the
default. Anonymous IDs and unavailable metadata stay out of every rendering.

The `audit` phase saves exact retained token sequences and decoded states without
loading weights; `prompts` compares the baseline with three wordings;
`states --question NAME` holds a selected question fixed while comparing JSON,
readable text, and text without system details. `evidence` compares JSON/text
with revealed examples on a smaller matched sample. `confirm` uses 120 items
per pair instead of the 20-item development screen. Every phase requires a new
output directory to protect historical artifacts. See
[experiment-log.md](experiment-log.md) for findings and run IDs.

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

## Recorded system context: `context_v1`

Implemented after the original-table audit in
[input-context-audit.md](input-context-audit.md). Use
`--laya-state-format context_v1` with `scripts/run_eval.py`, or
`LayaPredictor(state_format="context_v1")`. Both native backends share it.
JSON remains the default for existing checkpoints and historical comparisons.
The correctness question and published field projection are unchanged.

The renderer keeps the supplied canonical model/provider/harness/version/effort
fields. It groups settings from `subject_features_extra` into recorded model
identifiers/checkpoints, harness/agent settings, reasoning/generation settings,
prompting/few-shot settings, and other recorded settings. These are descriptive
groups, not additional facts or fallbacks for missing canonical fields. Dates
remain in the system block; task content/features/tools follow. Task text is
literal, including code, JSON strings, encoded strings and answer options.

A conservative key=value parser handles quoted strings and nested JSON. It
retains duplicate keys, empty values and unfamiliar settings; malformed input
falls back to the complete original literal. Explicit configuration fields such
as `agent_configuration` support one pass of percent decoding when the result
is valid JSON. The rendering labels this as URL-decoded JSON. Plus signs and
leaf strings stay literal. Duplicate JSON keys and invalid configurations fall
back to their original value. No arbitrary task text is URL-decoded or executed.

Example from the real AdaptiveStep record:

```text
Target AI system
harness: ASPRM
subject_features_extra (recorded settings):
  Recorded model identifiers and checkpoints:
    model_identifier: MetaMath-Mistral

Target task
item_content: Simplify $(-k + 4) + (-2 + 3k)$.
item_features: dataset=math500;prompt_origin=released_question
```

Evaluation configs, MLflow parameters/run names, actual input-example artifacts,
and default output folders distinguish the format. Input-design audit/state/
evidence comparisons now include `context_v1`. No new inference or quality
comparison has been run for it; the existing smoke checkpoint was trained on
JSON. Use identical model/sample/precision settings for a direct comparison.

### Versioned training-input preview and token audit

```bash
.venv-laya-mlx/bin/python scripts/preview_context_inputs.py \
  --source runs/head_finetuning/data_v2/training.jsonl \
  --out runs/NEW_CONTEXT_EXPORT
```

This writes a separate `inputs_context_v1.jsonl` with the same row order,
questions and expected labels, plus source/export hashes, per-row token audits,
summary and representative text examples. It refuses an existing output folder.
It uses only the pinned tokenizer; it does not load weights, train, score, or
read confirmation data. Original provenance row ordering can identify records,
but original state/effective-input hashes and truncation flags describe JSON
and must not be reused for the new format. This is an input export, not a full
pilot training manifest or resumable training artifact.

Saved previews:

| Source | Format | Rows | Truncated at 512 | Complete tasks | Median state tokens |
| --- | --- | ---: | ---: | ---: | ---: |
| 256-row smoke | JSON | 256 | 0 | 256 | 291 |
| 256-row smoke | context_v1 | 256 | 0 | 256 | 251 |
| Full training | JSON | 5,000 | 2,711 | 2,289 | 763.5 |
| Full training | context_v1 | 5,000 | 2,687 | 2,313 | 722.5 |

Artifacts: `runs/context_v1_smoke_preview/` and
`runs/context_v1_training_preview/`. All system blocks fit in these checks.
This is a modest visibility improvement; 53.74% of the full training inputs
still truncate. It does not establish improved Brier ALC. No original frozen
exports, checkpoints, or previous scores were modified.
