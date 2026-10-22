# Experiment log and ideas before fine-tuning

Updated: **2026-10-22, America/Los_Angeles**.

This is the working record of completed comparisons, their limitations, and
proposed next experiments. The user requested documentation of the run review
and brainstorming. The initial version recorded proposals only. The user then
authorized the input-visibility and short-question experiments; their results
are recorded below. Other proposals remain untried. No Laya fine-tuning has
been performed.

For runtime setup, see [laya-backends.md](laya-backends.md). For historical
experiments and the canceled 10,000-input run, see
[project-status.md](project-status.md). That canceled run must not be restarted
or presented as completed.

## Goal and scoring

Predict the target AI system's outcome from competition-visible system and task
attributes, plus the outcomes revealed within the current label budget.
Improving Laya itself and improving the final predictor are related but distinct
goals: a statistical predictor might improve the final score even if pretrained
Laya remains weak.

Brier ALC is lower-is-better. Always predicting 50% gives exactly 0.25. The
budget weights are:

| Revealed labels | 0 | 1 | 3 | 7 | 15 | 31 |
| --- | --- | --- | --- | --- | --- | --- |
| ALC weight | 10% | 20% | 20% | 20% | 20% | 10% |

Thus budgets 1, 3, and 7 account for 60% of ALC; effective adaptation from a few
examples is a major opportunity. Zero-shot Laya ignores revealed labels by
default, so its predictions and scores are unchanged across budgets.
See the [competition interface and scoring](https://aims.stanford.edu/competition).

## Completed comparisons reviewed on October 22

These are saved local results, not hidden Codabench results. The review used
stored predictions and did not rerun model inference. Within each comparison
group, the comparison tool verified matching evaluation item sets.

### Default suite: 3,840 evaluation inputs, 32 system–benchmark pairs

Eight benchmarks, four systems per benchmark, 120 evaluation inputs per pair;
seed 0. English Laya uses MLX FP16, 512-token context, and the unchanged
`task_correctness_v1` question, without revealed examples.

| Predictor | Brier ALC | MLflow run |
| --- | ---: | --- |
| Constant 50% | 0.250000 | `32345256e5594fc6ad01ffa9ccffb389` |
| English Laya, no examples | 0.320424 | `2c83ed48d86d47d49de803409fcf9619` |
| Shrunk mean, prior strength 5 | 0.235098 | `073951c70cb248adbf28912a8bb80db7` |

- Laya minus constant: **+0.070424** Brier, paired bootstrap 95% interval
  **[+0.037635, +0.102382]**. Laya wins on 8 pairs and loses on 24.
- Shrunk mean minus constant: **−0.014902**, interval
  **[−0.029883, −0.000115]**. The paired t-test has p=0.065; the average gain
  is promising but uncertainty remains. It wins on 16 pairs and loses on 16.
- Shrunk mean improves from **0.246988 at budget 0** to **0.218958 at 31**.
- Laya performs 3,603 actual forwards, with 1,145 truncated (31.8%). Reused
  visible inputs explain why forwards are fewer than scored records.
- Laya beats 50% on only one of the eight benchmark averages: CMMLU.
- Calibration examples: EDITBench mean predicted success is about 80%, versus
  42% recorded success; AI2D is about 37%, versus 73%. A single global shift
  cannot necessarily fix errors in opposite directions.

Intervals resample pairs, which share benchmarks and sometimes related systems.
They describe this local comparison; they do not establish hidden-benchmark
generalization.

The text-ridge run `c4b017dd8a2643cbb6e8ac6603bc37c0` was interrupted during
the disk-cleanup work. Its stale MLflow status is RUNNING, but it has no final
predictions or score and had no active evaluation process at review. Exclude it
from performance comparisons; restarting it is a new action.

### Matched 200-input sample: 10 pairs, seed 20261021

| Variant | Brier ALC | Truncated forwards | MLflow run |
| --- | ---: | ---: | --- |
| English, 512 | 0.281100 | 47 / 200 | `1a0c3457312c4b8ea71c0929bb29035e` |
| English, 1024 | 0.275751 | 25 / 200 | `142e93d887b24139af4edc0ffad46178` |
| English, 2048 | 0.274100 | 13 / 200 | `9282c84c98de4a46a85ef69b734f710b` |
| Multilingual, 1024 | 0.381807 | 21 / 200 | `4b294f78fe3a465f8694e7351aebda11` |
| English, 1024, revealed examples | 0.261340 | 755 / 1200 | `8174714ea71343f1b4b4640e4171bf56` |

- Longer context reduces truncation, but these accuracy differences do not
  establish a clear gain. English 2048 minus 1024 is −0.001651.
- Multilingual 1024 minus English 1024 is +0.106056, with paired bootstrap
  interval [+0.029743, +0.180372]. English is the more promising reference on
  this sample; language alone is not the only checkpoint difference.
- Revealed examples minus no examples at 1024 is **−0.014411**, interval
  **[−0.054039, +0.025774]**; 4 pair wins and 6 losses. Improvement is
  inconclusive. Batch size also differs (1 with examples versus 8 without),
  causing a small budget-zero probability difference.
- The examples mode uses six times as many forwards and truncates 62.9% of
  them. It took roughly 338 seconds versus 33 seconds without examples.

Do not directly attribute differences between the default and 200-input scores
to a code change: those groups contain different systems and items. Duplicate
runs on identical samples are not independent evidence.

### Verification and local artifacts

Recomputing response-weighted Brier from saved predictions reproduced the
reported default Laya, shrunk-mean, and revealed-example scores to floating-point
precision. No invalid probabilities were found. This does not suggest a scoring
arithmetic explanation for the performance gap.

Derived review tables are local ignored artifacts:

- `runs/review_latest/default/{summary,comparisons,per_benchmark}.csv`
- `runs/review_latest/sample_200/{summary,comparisons,per_benchmark}.csv`

Use `scripts/compare_runs.py` for paired comparisons. The MLflow database is
`runs/mlflow/mlflow.db`; completed runs retain predictions and source snapshots.

## Input diagnosis from code inspection

`src/paiec/laya_predictor.py:build_state` renders the system, target task, then
revealed examples. Nested task JSON remains an escaped string. For string
states, the installed MLX tokenizer keeps the beginning when truncating
(`laya_mlx/common.py:finish_sequence`). Therefore a long target task can push
some or all appended examples beyond the retained input.

This is a plausible limitation of the current examples experiment, not a
measurement of exactly how many examples survived. A future token audit should
inspect actual retained question/options/system/evidence, not just total token
counts.

## Experiment backlog

The following IDs were initially proposals. P03 and P04 have now been screened
as recorded below; the others remain proposals. Calibration or fitting a small
statistical predictor leaves Laya's weights frozen; it is distinct from
fine-tuning Laya.

| ID | Proposal | Question and design | Priority |
| --- | --- | --- | --- |
| P01 | Blend saved Laya scores with the shrunk mean | Does Laya add useful item-level information? Fit a blend weight on separate development data; allow zero weight. Initially reuse saved predictions. | First |
| P02 | Improve the prior and numerical label update | Compare 50% and system-informed priors; tune prior strength, preserving evaluation-benchmark exclusion. Emphasize budgets 1/3/7. | First |
| P03 | Compact state rendering | Compare current JSON with labeled plain text and readable nested task content, preserving supplied meaning. Inspect retained tokens before inference. | First |
| P04 | Short question variants | Compare a small set of target-system correctness questions while holding state fixed. Then hold the chosen wording fixed for state experiments. | First |
| P05 | Conservative probability calibration | Test shrinkage toward 50% and a low-parameter level/spread correction. Fit on separate benchmarks; calibration cannot manufacture useful ranking. | Early, alongside P01 |
| P06 | Compact evidence summary | Supply counts of acquired numeric outcomes instead of all full examples; compare against a numerical update outside Laya. | After P02/P03 |
| P07 | Relevant revealed examples | Retrieve a few similar tasks only from already acquired examples, optionally alongside outcome counts. All revealed labels still inform the external estimate. | After P06 |
| P08 | Small system-and-task predictor | Compare system features alone, task features alone, and both. Include harness/settings and missingness without inventing absent values. | Early baseline work |
| P09 | Frozen embeddings / capability–difficulty model | Predict unseen-item difficulty from task content and system capabilities; adapt with acquired labels. Test transfer across entire benchmarks. | Later |
| P10 | Ensembles and selective weighting | Average useful prompt/state variants or learn when to downweight Laya, using visible truncation/missingness features. Test whether multilingual adds complementary signal. | Later |
| P11 | Online label acquisition | Compare diversity/disagreement policies with random selection, respecting sequential candidates and nested budgets. Account for selection bias in success-rate estimates. | After a predictor uses random labels well |
| P12 | Stronger auxiliary model for task features | Derive capability requirements or task descriptions from permitted input. Validate their value; verify competition data/runtime policies before use. | Exploratory |

### Concrete starting designs

For P01, use `p_final = (1 - w) * q_n + w * p_laya`, where `q_n` is the
statistical estimate at the current budget. Example: q_n=0.60, p_laya=0.90,
w=0.20 gives 0.66. Learn w on development benchmarks rather than the benchmark
being evaluated. A weight of zero is a valid and informative result.

For P02, `q_n = (successes + k * q_0) / (n + k)`. The current implementation
uses k=5. With q_0=0.60 and one acquired success, this gives about 0.67 rather
than 1.00. At budget zero it returns the prior. Only budget-permitted revealed
responses count toward n; do not read additional trials from source tables.

For P04, candidate wordings include:

- “Will the target AI system answer the supplied task correctly?”
- “Will the AI system described above successfully complete the supplied task
  on its first attempt?”
- “Given this AI system’s model and settings, will its response to the supplied
  task be correct?”

These are candidate variants only; the active question remains unchanged in
`configs/laya_question.json`. Do not add generic criteria, probability-output
instructions, benchmark-specific prompt selection, or hidden grading metadata.

Diagnostic ablations could remove system details or task text separately to
test whether Laya uses both. Use real multi-system observations where possible
to examine whether predictions respond meaningfully to system differences.

For P05, prefer one- or two-parameter adjustments initially. A curve fitted and
scored on the same labels can look excellent without transferring. Existing
session notes already record a flexible isotonic calibration attempt that
failed a held-out-benchmark check; do not present in-sample results as evidence
that calibration will solve the problem. With acquired labels, a strongly
regularized pair-specific level adjustment is another proposal, not a feature
currently implemented.

For P08, the existing `TextRidge` uses item text only and tokenizes with
`[a-z0-9]+`. It ignores the system configuration and poorly represents Chinese
text. Its unfinished run is not evidence against richer small predictors.
Model identity and harness should have separate representations where supplied;
the current prior pools by canonical model identity rather than modeling every
harness/configuration distinction.

For P11, always retain random acquisition as a control. Selecting only uncertain
or difficult items can bias the observed success rate. The hosted interface
offers candidates sequentially, so an offline policy that inspects the entire
pool before selecting 31 items would not be equivalent.

## Evaluation rules for the next experiments

- Preserve the official input boundary and anonymous benchmark IDs. Never
  infer hidden benchmark identities. No reference answers, target outcomes,
  grading definitions, or unavailable metadata enter predictions.
- Preserve raw labels. Some public tables describe events other than task
  correctness; review training targets before pooling them. See
  [outcome-semantics.md](outcome-semantics.md). Missing image content remains a
  limitation; do not invent images or claim a text predictor saw them.
- Treat the current eight benchmarks as development evidence: they have already
  informed decisions. Reserve other eligible benchmarks for confirmation, with
  the permitted data inventory and target semantics checked first.
- Split by whole benchmarks, and consider related benchmark families, rather
  than random response rows. Fit prior/calibration/blend settings without the
  evaluation benchmark. Reproduce any model-selection procedure inside folds;
  do not tune a weight on the same scores used to claim improvement.
- Compare variants on identical seeds, subjects, items, acquisition order,
  checkpoint, precision, and batch size where possible. Screening on a small
  fixed sample is useful; confirm finalists across more independent benchmarks
  and acquisition trajectories rather than repeatedly searching the same sample.
- Report Brier at every budget and ALC, including paired changes and uncertainty.
  Score all recorded responses correctly; squared error against an item's mean
  label alone omits within-item variability. Keep trial grouping, acquisition
  exclusion, and equal pair weighting intact.
- Log exact states/questions, source/config versions, sample IDs, runtime,
  truncation, and any fitted parameters in MLflow. Distinguish actual forwards
  from scored inputs. Do not count repeated budgets as new questions.
- Current reference scores are 0.2500 for constant and 0.2351 for shrunk mean on
  the default sample; they are not promised scores on new data. A useful Laya
  addition should beat the strongest matched baseline, not merely raw Laya.
- Hosted Codabench packaging/runtime is still unverified. Local MLX success
  does not establish hosted compatibility. No new experiments are authorized
  merely by appearing in this log.

## October 22: input visibility and question experiments (P03/P04)

The user authorized the first two ideas from the brainstorming response:
inspect what Laya sees and compare short question wordings/state formatting.
No statistical blending, calibration, fine-tuning, external inference APIs, or
benchmark-specific questions were introduced.

### Protocol

- Definitions: `configs/experiments/input_design.json`.
- Development screen: the default eight benchmarks and four systems each,
  capped at 20 evaluation items per pair: **640 inputs, 32 pairs, seed 0**.
  These are prefixes of the same default-suite evaluation splits.
- Original English checkpoint revision `7b928d828b7b0e022f929d9bd2e44165aa270148`,
  MLX FP16, batch size 8, max_len 512 for every inference variant.
- Compare original JSON with three short questions first. Select the lowest
  observed ALC for a subsequent format screen; the original question won.
  Keep that question fixed for JSON, readable text, and readable text without
  system metadata. This is exploratory selection on development data.
- `text_v1` uses labeled fields and renders stored JSON structurally. It does
  not summarize the task, add criteria, invent model information, or introduce
  benchmark identities. JSON embedded as literal text inside a decoded task
  remains literal text rather than being recursively reinterpreted.
- The original JSON and `task_correctness_v1` remain defaults. Custom questions
  require an explicit, distinct prompt version; state format and ablation mode
  are separately recorded. Runs save predictions, source snapshots, configs,
  runtime/checkpoint provenance, sample manifests, and actual input examples
  in MLflow. Local outputs refuse to overwrite existing variant directories.

### Visibility audit (tokenization only, no model inference)

For the first system in each benchmark, select the shortest, middle, and longest
state among its 20 screened items: **24 deliberately selected cases**. Audit
all six budgets at 512 and 1024 tokens for JSON and text. These are diagnostic
cases, not an estimate of how often truncation occurs across the dataset.

The audit calls the native question-head builder and sequence truncation code,
including the question head in the token limit. It saves full rendered states,
retained decoded states, complete sequence token IDs, per-task-field visibility,
and whether each numeric revealed outcome survived. Tests check its sequence
against native sequence preparation; token visibility does not establish
semantic understanding.

| Format / limit | Complete tasks at budget 0 | Visible outcomes at budget 1 | Visible outcomes at budget 31 |
| --- | ---: | ---: | ---: |
| JSON / 512 | 18 / 24 | 4 / 24 | 4 / 744 |
| Text / 512 | 19 / 24 | 12 / 24 | 12 / 744 |
| JSON / 1024 | 21 / 24 | 18 / 24 | 47 / 744 |
| Text / 1024 | 21 / 24 | 18 / 24 | 55 / 744 |

The current ordering can lose the numeric outcomes entirely. Formatting helps
some cases, but full examples still vastly exceed the context allowance at
larger budgets. This finding does not prove a compact numerical summary would
help Laya; that remains P06. The initial audit was retained in `audit/`; the
final renderer audit is in `audit_final/` (same aggregate counts).

### Question screen: no revealed examples

| Question version | Brier ALC | Delta versus original | MLflow run |
| --- | ---: | ---: | --- |
| Original `task_correctness_v1` | 0.319734 | — | `a68e4fa2a75245c2b03c4bdd6647b145` |
| `short_v1` | 0.331943 | +0.012208 | `8d723b8ed8854eefb59aecbc9cf33140` |
| `first_attempt_v1` | 0.343049 | +0.023315 | `51a7509e5a244e2da8642e5dae407542` |
| `model_settings_v1` | 0.349455 | +0.029720 | `c1d5912d00904a7fb3ef703236d85918` |

No alternative beats the original on this screen. Paired bootstrap intervals
for all three differences include zero; this does not establish universal
superiority of the original wording. Exact alternatives are in the config.

### State screen: original question held fixed, no revealed examples

| Format | Brier ALC | Truncated forwards | MLflow run |
| --- | ---: | ---: | --- |
| Original JSON | 0.319734 | 195 / 630 | `7da8f26fd67045eda8fe41db93538948` |
| Readable text | 0.340497 | 168 / 630 | `0eb498c83ded4c2080e0aa78fc2be4c0` |
| Readable text, system omitted | 0.358528 | 140 / 625 | `4b550f3fa9054b49926e8cc42437cc7b` |

Readable text reduces truncation but does not improve the observed score.
Its paired delta versus JSON is +0.020762, bootstrap interval
[−0.024479, +0.064613]. Omitting the system has delta +0.038794 versus JSON,
interval [+0.005194, +0.071923]. These are exploratory local comparisons with
correlated pairs, not proof of transfer to unseen benchmarks.

The system description has a substantial effect: removing it changes predicted
probabilities by **25.7 percentage points on average**. Removing metadata also
frees task context, so the report separately examines the **472 inputs across
28 pairs whose supplied task fields fit in both conditions**. Their average
absolute change is 30.3 points; Brier is 0.314677 with the system versus 0.332654
without it. These subset scores have a different task mix. The experiment
supports sensitivity to the profile, not a claim that Laya correctly understands
every model or setting.

### Smaller matched evidence check

Use the same 32 pairs and split/acquisition trajectories, capped at four
evaluation items each: **128 inputs**. Compare JSON and readable text with all
budget-permitted full examples appended, using the original question and the
same 512-token ceiling. This tests formatting with evidence, not the untried
numerical-summary proposal. The sample differs from the older 200-input,
1024-token evidence experiment and from the 640-input screen above.

| Format with examples | Budget-zero Brier | Brier ALC | Truncated forwards | MLflow run |
| --- | ---: | ---: | ---: | --- |
| JSON | 0.275858 | 0.254278 | 654 / 767 | `6038502853784df2a4d40bad5c7262e7` |
| Readable text | 0.299773 | 0.298689 | 607 / 767 | `e76bf48e2f4d491b8e7b517bbeb5d8e2` |

Reusing each run's own saved budget-zero predictions at every budget gives a
matched frozen zero-shot control without new inference. JSON's ALC change
relative to that control is −0.021580, with paired bootstrap interval
[−0.055612, +0.011214]. Text's change is −0.001084, interval
[−0.030053, +0.028642]. Neither establishes a reliable benefit from examples.
For JSON, probabilities are bit-identical at budgets 3, 7, 15, and 31,
consistent with later appended content being cut off. Do not assume score
changes demonstrate use of numeric outcomes: example text can also change an
input even when its outcome is truncated.

### Decision after this screen

Keep the original question and JSON as defaults. No tested alternative has a
lower observed ALC on its matched format/question comparison. The repeated
JSON control reproduces every saved probability exactly, so the format screen
has a consistent reference. A full 3,840-input confirmation was not launched
because there was no improving candidate to confirm. This does not rule out
other wordings, layouts, or better evidence handling.

The practical finding is that full revealed examples often do not fit; a
readable representation alone does not solve either that problem or prediction
quality. A compact outcome summary or an external numerical update is a
reasonable next proposal (P06/P02), not work performed in this experiment.

### Reproduction and artifacts

Use a fresh output directory per phase:

```bash
.venv-laya-mlx/bin/python scripts/run_input_experiments.py --phase audit --out runs/NEW/audit_final
.venv-laya-mlx/bin/python scripts/run_input_experiments.py --phase prompts --out runs/NEW/prompts
.venv-laya-mlx/bin/python scripts/run_input_experiments.py --phase states --question baseline --out runs/NEW/states
.venv-laya-mlx/bin/python scripts/run_input_experiments.py --phase evidence --question baseline --out runs/NEW/evidence
.venv-laya-mlx/bin/python scripts/report_input_experiments.py runs/NEW
```

Local ignored artifacts are under `runs/input_design_20261022/`, including
`report.html`, `audit_final/retained_inputs.jsonl`, per-variant predictions and
details, paired comparison CSVs, and `system_ablation.json`. No historical
experiment was overwritten. Tests: 148 pass in MLX; 148 run in the reference
environment with the native-MLX audit test skipped (147 pass).

## Planned head-only fine-tuning

The user subsequently requested a plan for decision-head fine-tuning. See
[decision-head-finetuning-plan.md](decision-head-finetuning-plan.md) for the
dataset/holdout design, frozen-encoder pilot, runtime/checkpoint changes,
calibration, matched controls, and decision gates. This is planning only: no
training, export, or new evaluation has been performed for that plan.

## Head-pilot data preparation (no training)

After discussing family-first splits and random sampling within roles, the user
authorized preparation. All 61 downloaded benchmarks were reviewed; 14 admitted
benchmarks form 13 families. The frozen allocation is 7 training / 2 development
/ 2 calibration / 2 confirmation families, with 5,000 training responses and
1,000 calibration responses. Development and confirmation each have 960 distinct
evaluation system–item inputs; development retains 963 response targets because
trials are preserved. Confirmation JSONL has no target labels.

See [head-data-split.md](head-data-split.md) for names, exclusions, the rejected
initial length-imbalanced draft, overlap checks, and substantial remaining
training truncation. Current exports are in `runs/head_finetuning/data_v2/`;
`configs/head_finetuning/frozen_split_v1.json` records their fingerprints. No
model inference, training, new Brier score, or encoder adaptation occurred.

## How to update this log

For each attempted proposal, append its ID, date, hypothesis, exact change,
run IDs, sample/config fingerprints, baseline, per-budget Brier/ALC, uncertainty,
runtime/truncation, and conclusion. Record interrupted or inconclusive work
explicitly. Keep unsuccessful experiments; do not overwrite historical runs or
turn proposals into findings without evidence.

## Background references

- [Competition rules, interface, acquisition, and scoring](https://aims.stanford.edu/competition)
- [Laya model card](https://huggingface.co/convaiinnovations/laya)
- [Probability calibration: scikit-learn guide](https://scikit-learn.org/stable/modules/calibration.html)
- [On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html)
- [tinyBenchmarks](https://arxiv.org/abs/2402.14992): inspiration for efficient
  evaluation, not a drop-in solution to anonymous benchmark cold starts.
- [Contextual Multidimensional IRT](https://arxiv.org/abs/2608.22295): task-text
  generalization and its limitations under cross-scenario shifts.


## 2026-10-22: head-only training smoke check

User authorized the bounded check after data preparation. See
[head-training-smoke.md](head-training-smoke.md) for the recipe, saved artifact,
packaging repair, and runtime qualification. Ten FP32 PyTorch/MPS updates on
256 fully fitting training-only responses completed in 131.10 seconds. Only
the allowed decision modules changed; encoder and auxiliary head match the
original parent exactly. The model reloads without prediction drift on two
training fixtures. This is machinery validation, not Brier ALC evidence.

Training run `d47d2cab75314be3acba0d3128919528` remains failed because its final
manifest check caught upstream tokenizer normalization. The export was fixed;
verification run `18d5895e73c94bc5b54dd706d84b98d7` passed on the already trained
weights, with zero additional optimizer updates. Outputs are in
`runs/head_finetuning/smoke_v1/`. No held-out scoring, calibration, full pilot,
encoder unfreezing, or commit occurred. Next: inspect retained task fields in
the full training set's truncated inputs before selecting the pilot recipe.


## 2026-10-22: first head development Brier ALC

See [head-development-results.md](head-development-results.md) for matched data,
run IDs, controls and limitations. On frozen development (960 inputs, 963
responses, eight pairs/two families), the 10-update head scored **0.394842**.
Original Laya with shipped calibration scored **0.379140**, and constant 50%
scored **0.250000**. Because tuned output is uncalibrated, we also measured
original Laya at temperature 1: **0.459007**. The tuned head improves raw Brier
on all eight pairs and both benchmark averages, but its current end-to-end
pipeline remains worse than the calibrated original and 50% control.

All Laya inference used matched MLX FP32, 512 tokens and the same question/state.
No gradients, calibration fitting, or confirmation scoring occurred. The
separate raw-control report reuses the three finished initial runs and leaves
their historical metrics intact. Do not interpret this as a full pilot or
cross-family confirmation result.


## 2026-10-22: original input-context audit

See [input-context-audit.md](input-context-audit.md). All 61 original subject
tables were inspected; all nonempty published subject attributes are retained
in the 256/5,000/963 smoke/training/development exports. Dedicated reasoning
settings are recorded in only 14 of 8,163 public subject rows; extra fields
sometimes contain more detailed configurations. Three selected C-Eval models
have identical published system profiles because their names exist only in the
excluded display_name column. A focused decoded-extra-settings renderer is
proposed; no default change, inference, training, or new score occurred.

## 2026-10-22: implement recorded-context rendering

User authorized the code changes after the original input audit. Added the
explicit `context_v1` format, conservative settings parsing, selectively decoded
configuration JSON, shared CLI/config tracking and input-experiment integration.
Missing names/settings remain missing; hidden metadata stays excluded. JSON is
retained as the default for the already trained checkpoint and old comparisons.
See [benchmark-questions.md](benchmark-questions.md#recorded-system-context-context_v1).

Prepared separate 256/5,000-row training-input previews with unchanged question,
labels and row order, and native-tokenizer audits only. Full-training truncation
falls from 2,711 to 2,687 rows; median state tokens from 763.5 to 722.5. There is
no new inference, training, Brier ALC result, or calibration. All 169 discovered
tests pass with the relevant environment skips (MLX: 3, PyTorch: 1).


## 2026-10-22: Modal execution integration and authentication

Added typed job plans and local PyTorch/MLX versus Modal CUDA placement, with
minimal hashed input bundles, persistent per-job results/model cache, CUDA smoke
training support and local MLflow result import. Installed the client in a
separate environment and resolved a separate Linux worker lock. The user then
requested browser authentication, which verified successfully for elias-huehne.
SDK definition checks, dry runs and 174 lightweight tests pass with environment
skips; no cloud image build, GPU invocation or new model scores occurred. See
[modal.md](modal.md) for exact support and unverified end-to-end steps.
