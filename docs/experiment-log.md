# Experiment log and ideas before fine-tuning

Updated: **2026-10-24, America/Los_Angeles**.

Latest scientific result: the enriched multilingual grouped pilot completed
543 updates and one original-development evaluation. Brier ALC **0.2824393058**
versus constant **0.25**, mean within-pair AUROC **0.5457424346**. Checkpoint
integrity/reload passed; both GPU apps are stopped. **Laya work is now on hold**;
reuse data, features and evaluation pipelines for other approaches. No new model
or additional run is authorized by this status update. See [current status](current-status.md)
and [the complete result](grouped-enriched-training.md). Earlier proposals below
are historical hypotheses, not a current work queue.

This is the working record of completed comparisons, their limitations, and
proposed next experiments. The user requested documentation of the run review
and brainstorming. The initial version recorded proposals only. The user then
authorized the input-visibility and short-question experiments; their results
are recorded below. Later fine-tuning and evidence experiments are recorded in the linked guides
and subsequent entries; the original October 22 comparisons used frozen weights.

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


## 2026-10-22: versioned system-profile research format

Saved typed JSONL profiles with homogeneous fact/evidence objects and explicit
model/harness/run separation. Architecture includes backbone, attention direction,
dense/MoE, total/active parameters and optional layer/head/expert/tokenizer details.
A collection plan and JSON Schemas accompany real MetaMath-Mistral/ASPRM/run
seed records grounded in the original subject table. Architecture and other
unreviewed facts remain unknown; no external collection, predictor integration,
training or evaluation occurred. See `research/system_profiles/README.md`.

## 2026-10-22: primary-source profile collection

The user explicitly authorized actual collection after the schema-only seed.
Saved `research/system_profiles/collection_v1` using schema 1.0: 18 model,
8 harness and 32 recorded-run profiles covering all exported training/development
subjects in `data_v2`. Collected 42 external document/configuration snapshots,
plus 10 original subject-table sources. There are 121 documented specification
facts, five inferred classifications and 20 recorded run settings; identity
facts are separate. Architecture configs are pinned and quoted; current provider
defaults are qualified rather than assumed to describe historical runs.

Nineteen subjects have name-level model matches; thirteen retain unresolved
identity/version status, including private CDG checkpoints, missing C-Eval names,
ERBench aliases and historical GPT-3.5-16k. MetaMath-Llama's candidate card is
inconsistent; unofficial CodeLlama config assertions were excluded. Configured
position limits are not promoted into supported context limits. External
sources remain unreviewed for competition eligibility, with no predictor join,
training, inference, cloud job or score changes. See
[system-profile-collection.md](system-profile-collection.md) for evidence,
coverage and reproducibility.

Verification: offline rebuild and both catalog validators pass; the full MLX
lightweight suite passes 182 tests with three expected skips. Source snapshot
checks and quoted/recorded evidence checks pass. No model evaluation was added.

## 2026-10-22: enriched head pilot prepared; training awaits go-ahead

Implemented research-only `enriched_v1`, a conservative identifier/field policy,
and one shared renderer for training exports and later `LayaPredictor` calls.
Documented properties fit only in spare input room, capped at 96 tokens; no
original task text is removed. Unknown/private/conflicting models remain
unresolved. Harness capabilities are not claimed to be enabled run behavior.
Current API context/output defaults and inferred facts are excluded.

Prepared matched `context_v1` control and enriched versions of the same frozen
5,000 training responses and 963 development outcomes. Both use 1,024 tokens:
2,042 training inputs truncate identically, development inputs do not truncate.
2,732 training records receive some metadata (1,429 model, 2,278 harness;
counts overlap). All development rows receive harness capabilities; only the
120 Qwen rows receive model properties. This limits architecture generalization
claims. The original JSON/512 experiments and frozen source exports are intact.

Added a separate local MPS/FP32 pilot launcher with an explicit `--train` gate,
four epochs/316 updates per variant, raw development Brier ALC selection,
best/latest export, frozen-module checks and reload checks. New checkpoint
input contracts bundle and pin profiles, policy, question, format, context and
renderer hashes. Later API/CLI inference restores those settings automatically.
The historical JSON/512 evaluator rejects new pilot checkpoints.

Native PyTorch token checks and runtime-render reconstruction passed on every
prepared record, with no cross-role effective-input overlap. Local MPS and disk
checks pass; no Laya weights were loaded, no model inference or training started,
and calibration/confirmation remain unopened. Preparation is logged separately
in MLflow with no quality metrics. See [enriched-head-pilot.md](enriched-head-pilot.md)
and `runs/head_finetuning/enrichment_pilot_v1/report.html`.

Final checks pass: 197 tests in each environment (MLX four expected skips;
PyTorch one), checkpoint-bundle export/drift tests, native input/render parity,
and `git diff --check`. No actual Laya model forward pass or gradient update
was performed during preparation.

## 2026-10-22: local PyTorch/MPS pilot authorized and launched

The user chose the prepared PyTorch/MPS workflow after discussing MLX training.
Launched `scripts/run_head_pilot.py --train`: baseline then enriched, same
prepared inputs and seed, four epochs/316 updates each. Progress is in
`runs/enriched_pilot_training.log`. Completion and scores are not established
by launch; read per-variant reports and the final comparison when available.
No duplicate run, calibration fit or confirmation scoring is authorized by
this launch.

### MPS memory correction and fresh recovery

The initial baseline reached 11 updates with finite losses/gradients, but GPU
driver allocation increased from 6.45 GB at update 3 to 8.84 GB at update 11,
and update time rose from about 30 to 43 seconds. Interrupted it before further
memory pressure. MLflow marks `be2aec0b45bb441fad3f881f79719d25` KILLED; original
log/history plus `interruption.json` are preserved. No epoch, trained checkpoint
or quality result completed. This is unrelated to the older canceled 10k run.

Added a 0.7 allocator fraction, 128-token masked padding buckets and unused-cache
trimming after each update/every 16 development batches. Padding preserves real
tokens, attention/target masks and marker positions; a synthetic attention check
passes. All 198 PyTorch tests pass (one expected skip). Recorded memory policy
in training provenance. The shared renderer and frozen prepared data are unchanged.

Launched a fresh paired attempt from the original parent using `--out
runs/head_finetuning/enrichment_pilot_v1_training_mpsbounded_v2`. New log:
`runs/enriched_pilot_mpsbounded_training.log`. Recovery changes only output
location and memory handling; same samples, seed, optimizer and four-epoch budget.

## 2026-10-22: local run stopped; enriched-first Modal pilot prepared

The user requested stopping local training and preparing Modal, with enriched
first. Sent SIGINT to PID 12623 and verified exit. The memory-bounded attempt
ended at 11 completed updates and zero epochs. MLflow
`1187918c63604de29a3d0e0bcf0364a6` is KILLED, with reason tagged and
`interruption.json` saved. No completed development score or checkpoint exists.
Its driver allocation stayed near 3.24 GB after cleanup; update times were
about 35 seconds. The interrupted histories are retained, not resumed or relabeled.

Extended the shared pilot trainer to explicit PyTorch/CUDA placement and the
typed execution launcher to `train-head-pilot`. The task defaults to enriched
alone. `modal-l4-pilot` specifies one L4, two CPU cores, 16 GiB RAM and a
four-hour timeout. The first job retains all 5,000 records, four epochs,
316 updates, seed, weighting, optimizer and input hashes. FP32 is explicit,
with CUDA TF32/autocast disabled. Device/output are the only allowed overrides
of the prepared recipe; baseline remains a separate requested follow-up.

The explicit 20-file inventory totals 75,980,393 bytes (20,633,944 compressed).
Round-tripped it into an isolated source workspace and ran native preflight on
MPS: all token sequences and runtime-rendered states matched. This establishes
input completeness, not actual CUDA execution. Modal definitions constructed
locally without hydration or cloud resources. The dry-run plan is saved in
`runs/modal_enriched_pilot_plan.json`; validation evidence is in
`runs/modal_pilot_preparation_validation.json` and the isolated preflight log.

Cloud results preserve metric steps/timestamps during local MLflow import.
The worker mirrors console progress, keeps model tensors/best/latest on the
persistent volume, and records their paths in result receipts/status. The HTML
report supports a single completed enriched variant without fabricating a
baseline comparison. No cloud GPU has been launched, no remote image built,
and no calibration/confirmation exports uploaded or scored. See
[modal-head-pilot.md](modal-head-pilot.md) for the reviewed next launch.

Final verification: 204 tests pass in each environment (PyTorch one expected
skip; MLX six), isolated bundle/native preflight passes, and `git diff --check`
passes. No remote compute or model inference was performed for preparation.

## 2026-10-22: first enriched Modal job submitted

The user asked to start training on Modal after preparation. Launched the
reviewed enriched-only command with the original frozen recipe and L4 profile.
Modal accepted job `job_bf242b6e53874dcebb6cd6dd44738625`, app
`ap-LrMzFk0CpOsTcAQj28MLGD`. The first image build is running; model download,
preflight, CUDA updates and quality results are not established by submission.
Controller PID 17867, log `runs/modal_enriched_pilot_launch.log`, receipt
`runs/modal_enriched_pilot_launch.json`. Idle-sleep prevention follows the
controller process. Do not start a duplicate. Baseline remains unlaunched.

### First CUDA updates verified

The L4 worker started; pinned parent download and all native token/runtime-state
preflight checks passed. Remote MLflow run `3a15460e13bc4bc6933b93693712126a`.
First 15 updates completed with finite losses/gradients; median 7.67 seconds
per update, maximum allocated tensor peak 2.43 GB and 2.58 GB reserved.
This is early throughput, not a completed epoch/quality result. The local
baseline used different inputs, so the roughly 35-second local observation is
only an indicative speed comparison. Progress snapshot retained in
`runs/modal_enriched_pilot_progress.json`. Controller remains attached to fetch
results and import local MLflow automatically on completion.

## 2026-10-22: GPU utilization and frozen-encoder caching probes

User requested using GPU/memory efficiently. Existing L4, no new GPU: bounded
isolated probes with watchdog-protected trainer pauses. Larger plain batches
used more memory but slowed the single 64-record window (7.78s at batch two,
8.49s at batch 64). Length-grouped batch eight took 6.60s. TF32 took 3.79s
but shifted probabilities by up to 0.009154 on 64 records; not enabled live.

Frozen-encoder GPU feature caching took 1.35s at batch two, 1.08s with grouped
batch eight, after warming the cache. FP32 retained. Two-record probability
check exact; maximum head-gradient drift 1.86e-9 with identical RNG. Estimated
full training/development cache: 14.97 GB; not fully allocated or validated.
Production integration, memory cap and full-corpus verification remain pending.
The active run resumed unchanged (115 updates observed; epoch-one raw
development Brier ALC 0.249080). No optimizer updates applied by the probes.
Profiling pauses inflate step elapsed times; original metrics are retained.
See docs/gpu-training-performance.md and runs/modal_gpu_utilization_audit.json.

## 2026-10-22: cached enriched retraining authorized

User requested retraining after the feature-cache prototype. The prior Modal
app was already stopped after local client disconnection: recovered 139 update
records, one completed epoch (raw development Brier ALC 0.249080), predictions
and per-pair scores. Latest checkpoint remains on the volume; no final four-epoch
result exists. Small files recovered to
`runs/head_finetuning/enrichment_pilot_v1_modal_interrupted_recovery/`.

Implemented a process-local frozen-encoder FP32 feature cache with byte-bounded
LRU eviction, exact-token keys, mode/weight-version guards and explicit cleanup.
Only frozen sequential FP32 encoders are supported. New runs compare cached
and native probabilities/gradients on mixed-length training fixtures and all
development inputs before updates. Cache buffers are not checkpoint tensors.
Same 5,000 records, seed, micro-batch 2, accumulation 32, 316 updates, original
parent, no TF32 or precision change. Feature reuse fills lazily during epoch one;
later epochs should gain more. End-to-end speed remains to be measured.

208 PyTorch tests pass (one skip), covering cache reuse/eviction/repadding,
probabilities/gradients/RNG, state dict integrity and detached submission.
Submitted fresh enriched-only L4 job `job_e5548e95fcec4aba88d110056a0e4f71`,
app `ap-qpuYpW9bljFIY7WwDgyba2`, call `fc-01M5K8SQNQ19CSW1Z8GFTGAB2T`,
14 GiB feature cap. Now uses `app.run(detach=True)` plus durable `spawn`,
with call/app IDs written to the local receipt before waiting. Training can
continue after local client loss; retrieval/import remains available afterward.
No duplicate or baseline run was launched. Live log:
`runs/modal_cached_head_launch.log`; receipt: `runs/modal_cached_head_launch.json`.

Cached job CUDA checks passed: 33 training fixture batches, maximum probability
drift 7.90e-7 and head-gradient drift 1.68e-8; full-development predictions and
Brier match exactly. Remote MLflow `dedd67f8fc0247c28020573f1157e32c`.
Observed update 30 with 5.28 GB stored features, 7.43 GB allocated total.
Cold epoch-one updates still fill the cache; warmed throughput is pending.

### Cached retrain completed and recovered locally

The enriched cached run completed all four epochs / 316 updates / 20,000
training presentations. Best epoch 1: development Brier ALC **0.24908013**.
Original raw parent on the same enriched inputs: **0.33525644**; constant 0.5:
**0.25000000**. Epoch scores: 0.24908013, 0.25615209, 0.26679374, 0.27265947.
Later epochs worsened held-out development despite decreasing training loss,
consistent with overfitting. This is only two development families; no
confirmation scoring, calibration fit or paired baseline retraining was run.

Training/development phase: 1,013.86 seconds (16.9 minutes), excluding startup,
initial raw-parent evaluation, cache validation and downloads. Median full-window
update times by epoch: 7.05, 1.41, 1.40, 1.40 seconds. Warm updates are roughly
5x faster than cold epoch-one updates; this is not a 5x end-to-end speed claim.
Cache held 5,215 unique features / 12.14 GB with zero evictions. Maximum
allocated GPU memory was 14.58 GB, peak allocated 15.09 GB and reserved 15.31 GB.

Encoder and auxiliary action-head hashes stayed unchanged; decision weights
changed. Export/reload tensor hashes match and fixture probability drift is
zero. Best checkpoint was downloaded locally, every file hash verified, and
the bundled enriched_v1/1,024-token input contract validated without inference.

Local output: `runs/head_finetuning/enrichment_pilot_v1_modal_cached/output/`.
Report: `output/report.html`; best model: `output/enriched/best/`.
Performance summary: `output/enriched/performance_summary.json`.
Local MLflow experiment **PAIEC Modal**, run `6f174af602994e3d9f15bc0131351ca8`
is FINISHED with all four development curve points imported. No GPU container
remained active at the final check.

## 2026-10-22: engineering audit of the current approach

Reviewed saved inputs, predictions and training logs, without new fitting,
inference, GPU work or calibration/confirmation access. The trained head's
0.24908013 Brier ALC is just 0.00091987 better than constant 50%. Probabilities
span 46.8–55.6%; mean within-pair AUROC is 0.450. Replacing each pair's
predictions by its own average prediction improves Brier to 0.24842928 without
using evaluation labels to choose the constants: item variation is adding net
error on this sample. A paired item-cluster uncertainty interval includes no
gain over 50%, conditional on the selected two development benchmarks, without
correcting best-checkpoint selection optimism.

2,042/5,000 training records are truncated. 654 rows belong to 17 retained-input
groups containing distinct full tasks; one BenGER group merges 209 different
tasks. Three C-Eval systems also have identical allowed system profiles because
model identity is missing. These are distinct input-loss and missing-information
problems; source display names must not be substituted into prediction.

Training updates 26.2M parameters on 24 pairs/seven families. Later training
loss decreases while development error increases. Web enrichment has no matched
completed dataset-only training control, and verified model-property coverage
is only 28.6% of training rows / 12.5% of development rows. The English frozen
encoder's suitability for the largely Chinese C-Eval slice is an untested
representation question. Current predictions ignore revealed labels at all
budgets.

Recommendations, **not run**: task-preserving inputs and visibility checks;
small regularized Unicode-aware text/structured controls; numeric updates from
allowed evidence; matched enrichment ablations; language/capacity experiments
after those controls. External profile eligibility remains unreviewed under the
current competition inventory restriction. No production input or model changed.

Full review and ordered feature plan: [head-approach-review.md](head-approach-review.md).
Reproduction: `scripts/audit_head_approach.py --out NEW_DIRECTORY`.
Saved report, per-pair/coverage tables, real collision/error examples, cluster
interval and source hashes: `runs/head_finetuning/approach_audit_v1/`.

## 2026-10-22: task-v2 implementation and saved-prediction evidence control

Implemented a label-blind, native-token-budgeted task parser/builder, structured
feature/visibility export, separate task-v2 inference adapter, one-epoch matched
head preparation, and an optional numeric evidence wrapper. Original renderers
and historical input contracts are unchanged. The new Unicode/structured ridge
is implemented but has only synthetic unit-test fitting, no competition score.

Prepared all 5,000 training and 963 development responses under
`runs/head_finetuning/task_features_pilot_v2/`. Whole-input multi-task collision
rows fall from 654 to zero; BenGER retained inputs increase from 211 to 833.
Explicit task omissions remain in 2,040 training rows; zero native truncation
is not a claim of full context retention. Ignoring item metadata/IDs leaves 34
ClashEval rows with content collisions, and zero in BenGER. Frozen roles/labels
are unchanged, and calibration/confirmation remain unopened.

New numeric evidence evaluations use mixed observations across sampled pairs,
rather than the historical pair-local label list, preventing private scorer
subject grouping from disambiguating identical visible profiles. The predictor
matches only permitted fields and anonymous benchmark IDs. Prior strength is
explicit and untuned, not a fitted calibration parameter.

Scored a separate control with the already saved best enriched head
probabilities, same frozen development targets and acquisition trajectory:
0.249080 unchanged head; **0.238145** head + beta_mean k=5; **0.239531** constant
0.5 + beta_mean k=5. No new Laya inference or gradient training. Early budgets
1/3 worsen while 7/15/31 improve on this trajectory. This does not establish
new-input quality or confirmation generalization. No strength search was run.
Artifacts: `runs/head_finetuning/evidence_control_v1/`.

Full design, upstream sources, measured limitations and usage:
[task-feature-pipeline.md](task-feature-pipeline.md).

## 2026-10-22: capability/difficulty/similarity research design

User requested focusing retraining on model/harness capabilities, task difficulty
and similarity, authorizing broad web research. Reviewed primary research:
EmbedLLM, IrtNet and observational scaling laws; examined BGE-M3 as a possible
multilingual retrieval control. None was imported as training data or run.

New training-role-only audit finds **170 candidate pairs / 35 nonempty canonical
model-name strings**, versus the current 24 sampled pairs. BIRD has 120
configurations/15 names over the same 360 questions; BenGER has 25 candidates/
12 names. These require subject/input review and are not 170 independent LLMs.
Only four canonical strings recur across training benchmarks, with historical
version equivalence unresolved. This limits disentangling ability and difficulty.
No frozen family was reassigned or expanded automatically.

Proposed model: attribute-conditioned system representation + text-predicted
task requirements/difficulty + a learned compatible-evidence correction.
Begin with a small head on frozen Laya representations; then episodic training
and encoder adapters if validated. Use outcomes as primary supervision and web
facts as explanatory inputs. ID-only embeddings, guessed difficulty scores,
outside evaluation labels and unreviewed benchmark data are not accepted as
shortcuts. This architecture has not been implemented/trained.

Design: [capability-learning-design.md](capability-learning-design.md).
Audit: `runs/head_finetuning/capability_data_audit_v1/`;
reproduce with `scripts/audit_capability_training.py --out NEW_DIRECTORY`.
No embedding inference, new quality scoring, paid API calls, calibration/
confirmation outcomes or GPU jobs were used for this work.

## 2026-10-22: compact-v3 delivery and development identity investigation

Implemented opt-in `compact_v3` / `compact_enriched_v3`, shared across preparation
and prediction through a new factory. Original pinned renderer sources, catalogs,
prepared v1/v2 data and the trained head remain valid. The new recipe uses 2,048
tokens and reserves a fixed 192-token fact block in both arms. Retained task text
is identical between arms; detected options are never clipped. Marked question/
context templates now separate the question from large retrieved passages.

On the same frozen 5,000 training / 963 development records: all available facts
reach input. Training model/base-family facts: 4,584 available/delivered, versus
3,185 previously delivered; harness: 3,958 versus 2,304. Development model/base:
603 versus 120; harness: all 963. All records receive some sourced context,
**not** a verified model identity for every record. Three original C-Eval systems
still lack identifiers (360 records); two private CDG targets remain unverified.

There are zero native overflows, system/item-attribute omissions or shortened
detected options; all 480 development MCQ records keep their options. Task/context
omissions remain in 1,145 training rows; 652 have oversized question/user sections.
Formatting and context length both changed, so these counts are not a quality
gain or an isolated formatting comparison. No new Brier score exists.

Reviewed all 19 development-family subjects. Primary ERBench paper/code evidence
supports scoped reported model families, with explicitly labeled base-only facts
for private fine-tunes. `collection_v3` keeps actual fine-tuned model links null.
Four extra identifiable systems have a separate outcome-blind question-only
supplement: 480 evaluation / 160 acquisition inputs. Response values/reference
answers were not loaded; binary-outcome eligibility is not yet checked. Original
evaluation records, labels, acquisition trajectories and frozen roles stay intact.

Artifacts: `runs/head_finetuning/compact_pilot_v3_release/`,
`runs/head_finetuning/development_identity_review_v3_release_final/`,
`runs/head_finetuning/compact_workflow_v3/report.html`.
Details: [compact-inputs-and-development.md](compact-inputs-and-development.md).
Both suites pass 235 tests (MLX nine skips, Torch one). Full PyTorch preflight
confirms native token/render parity without loading weights. The one-epoch /
79-update paired recipe is prepared, not launched; GPU resources/performance
at 2,048 tokens still need qualification. No model inference, scoring, training,
paid API call, sealed-set access or GPU job was performed.

## 2026-10-22: expanded profile collection and coverage audit

User requested 100% coverage and selected all 170 training candidates plus eight
development systems. Added versioned `collection_v2`, preserving prior catalogs,
policies, renderer sources, prepared artifacts and the completed trained head.
There are now 44 model/artifact/family profiles, eight harness profiles and 178
recorded runs; 240 documented specification facts, six explicit inferences and
273 recorded settings. Sources/excerpts/config values are checked offline;
subject fields and frozen roles are checked against original subject tables.

Across the same 178 systems, model-fact availability rises from 44 to 169;
harness-fact availability remains 171. Both are available for 162; some sourced
context is available for all 178. **Full model-and-harness coverage is not
achieved:** three missing model identities, four generic study aliases/private
deployments, two unverified private CDG checkpoints and seven missing harness
names remain in a recorded resolution queue. No guessed dimensions or forbidden
display-name recovery were used. Thirty-four model matches intentionally use
family-only properties; no evaluated weight snapshot is independently verified.

New opt-in policy `configs/head_finetuning/enrichment_catalog_v2.json` prioritizes
size/type/modalities before deeper dimensions, with the existing 96-token cap.
Tokenizer-only audit on the frozen task-v2 pilot: model facts available for
4,584/5,000 training rows, actually rendered for 3,185; some enrichment rendered
for 3,599. Development: 120/963 model-fact rows and 963/963 harness-fact rows.
Metadata availability does not establish input visibility. Existing task omissions
remain in 2,040 training records. No new preparation bundle or training was run.

Report: `runs/head_finetuning/system_profile_coverage_v2_final/report.html`.
Durable metrics/queue: `research/system_profiles/collection_v2/`.
Details: [system-profile-coverage.md](system-profile-coverage.md).
Both lightweight suites pass 225 tests (MLX nine skips, Torch one); prior prepared
bundles and the completed best-head contract still validate. No new inference,
Brier evaluation, paid API calls, sealed-set access or GPU job.


## 2026-10-23 — Repository organization and clean-checkout checks

Repository maintenance only; no Laya inference, fitting, evaluation score or
cloud GPU job. The compact-v3 recipe remains prepared and untrained.

Moved the six dependency files into `requirements/`, with runtime-specific
names and the earlier pre-tracking lock under `requirements/archive/`. Their
bytes and package versions are unchanged. Updated Modal image/provenance paths
and MLflow source capture; snapshots now include nested execution modules,
test helpers and dependency input files. Dated `memory/` notes now live in
`docs/history/`; historical README baseline results are archived there too.
Local root download metadata/logs were preserved under ignored
`runs/setup_archive_20261023/`; the generated root Finder metadata was removed.

Rewrote the root README around current work and fresh-checkout setup. Added
indexes for docs, configs, commands, dependencies and tests, plus contribution
and versioning rules, a `make check` entrypoint, LF checkout rules and explicit
credential/cache/model-output ignores. Historical recipes, profile collections,
source schemas and hash-pinned renderers retain their names and contents.

An isolated export exposed three unit tests depending on ignored local runs or
payload files. Those checks now build synthetic temporary inventories and still
exercise artifact hashes, transport exclusion and configuration-drift rejection.
The baseline reference remains the real pinned submodule. New focused tests
cover relocated Modal worker dependencies and complete MLflow source capture.

Verification: 237 tests pass in the MLX environment (9 skips), Torch environment
(1 skip), and a source-only export using the core/Modal-client environment
(10 skips). That export includes the pinned baseline and no local payload,
prepared runs, model weights or credentials. All four checked-in catalogs
validate. Prepared enriched-v1, task-v2 and compact-v3 bundles and the completed
enriched checkpoint pass their existing integrity checks. CLI help, dependency
byte equality and Markdown links also pass; no renderer regeneration was needed.

Logs: `runs/repo_cleanup_mlx_tests.log`, `runs/repo_cleanup_torch_tests.log`,
and `runs/repo_cleanup_clean_export_tests.log`. A GitHub Actions core-check
workflow was added; it has been inspected locally, not executed on GitHub.
Changes are left uncommitted and unpushed for review.


## 2026-10-23 — Compact enriched head completed on Modal

The user authorized the next training experiment with the enriched variant as
the focus. The recorded-data control remains deferred. Added an independent
`qualify-head-pilot` execution task: same verified pilot bundle, selected
variant, bounded physical-batch-two CUDA window and longest training inputs,
zero optimizer steps, unchanged-weight checks and a saved JSON report. The
compact 2,048-token L4 check passed (13.004 seconds per warmed 64-record window).
It is not a batch-size optimization sweep. Preparation, input renderers, frozen
roles and scientific recipe were unchanged.

Enriched job `job_58c10a1eb5a54753af8b3afc0f19c855` completed one epoch,
79 updates and 5,000 presentations from the pinned original English parent,
FP32/TF32 off. Native preflight verified all paired inputs remotely. Final
original-development Brier ALC: **0.2486426089**, versus constant **0.25**,
raw original parent on the same compact enriched states **0.4140704701**,
and the historical different-input head **0.2490801324**. Difference from the
older trained head: -0.0004375236. This is a small exploratory gain; no matched
recorded-data training control was run, so it does not isolate enrichment.
Both original development families are near 0.25 (C-Eval 0.24814649, ERBench
0.24913873); predictions span 46.52–56.04%, mean 50.27%. No label adaptation,
calibration/confirmation scoring or supplemental development scoring occurred.

The training/development/export phase took 1,151.60 seconds; median full update
13.109 seconds. Peak sampled CUDA allocation 2.777 GB, reserved memory 3.020 GB.
The user's 16 GB memory question concerns the worker's system RAM, separate
from the L4's roughly 23 GB usable VRAM. Live utilization snapshots were 95–100%
with about 3 GB used. This does not establish maximum useful GPU throughput.
Full-corpus encoder features estimate about 20 GB before other allocations,
and the prior cache gain came from repeated epochs; this single-pass run used
the unchanged uncached recipe. Larger 2,048-token batches remain unmeasured.

Encoder/action-head hashes stayed unchanged and decision weights changed.
Export/reload probability drift was zero. Best checkpoint downloaded and every
file hash, policy/catalog hash, renderer hash and one exact training render
verified locally without new model inference. Local MLflow experiment
`PAIEC Modal` (ID 4), FINISHED run `57987fd441494993babee451c882f1c6`;
remote run `2f6c34e8dbfb407bb8b7b1cd777a845f`. Final app-log streaming timed
out after computation; return code and exported verification files confirm
success. No retry was launched. Completed app explicitly stopped; no GPU
container remained active.

Artifacts: `runs/head_finetuning/compact_pilot_v3_modal_enriched/output/`,
including `enriched/best`, `report.html`, `enriched/performance_summary.json`,
and `enriched/local_checkpoint_check.json`. Receipt/log:
`runs/compact_v3_enriched_launch.json` and `.log`. The HTML report now derives
coverage from preflight counts rather than claiming only one enriched system.
See `docs/compact-enriched-training.md`. Integration checks: 238 tests in each
model environment, plus all four research catalogs; no pinned renderer edits.

## 2026-10-23 — Diagnose the gap to Brier ALC 0.15

The user requested an engineering assessment of weak compact enriched results,
confidence near 0.5, and encoder training. `scripts/audit_head_strategy.py`
reproduces saved raw Brier exactly and audits original development only:
960 system-item inputs, 963 response rows, eight pairs. Mean within-pair AUROC
is 0.428037, versus pooled response AUROC 0.553599. Doubling deviations from
0.5 worsens Brier from 0.248643 to 0.249164; quadrupling yields 0.255844.
Forcing confidence is therefore unsupported by this sample.

A fixed previously used beta strength k=5, using only original acquisition
outcomes with shared allowed-profile matching, improves the saved head's ALC
to 0.237534. The constant-0.5-plus-evidence control scores 0.239531. Both worsen
budgets one/three on this frozen acquisition trajectory, and neither approaches
0.15. These are postprocessing controls, not retrained checkpoint results.

Two deliberately label-using diagnostics are kept separate: optimal scalar
stretch scores 0.248596 on its fitting labels; best constant-per-scorer-pair
fit scores 0.186602. Neither is a valid predictor/held-out result. The latter
is not a global irreducible-noise estimate; it shows that pair-wide constants
alone cannot reach 0.15 on this sample. Three indistinguishable allowed C-Eval
profiles still span observed rates 25.83–83.33%.

Proposed next work: synthetic/tiny-training learning checks, broaden admitted
training-role system/task coverage, compare partial encoder adaptation against
a matched enriched frozen-encoder control, and learn budget-dependent updates
from disjoint support/query episodes. Preserve frozen roles and input contracts;
invalidate frozen-encoder caches for any encoder updates. Upstream fine-tuning
guidance and primary adaptation/ability-difficulty research are linked in
`docs/brier-alc-strategy.md`. No new model inference, gradient training, cloud
job, supplemental scoring or calibration/confirmation access occurred.

Artifacts: `runs/head_finetuning/compact_strategy_audit_v1/report.html`,
`summary.json`, per-pair/sharpness/evidence tables and `diagnostics.png`.

## October 23: 30k enriched data, learning controls and encoder experiment

User-authorized expansion uses the existing eight training benchmarks / seven
families. All 170 candidates passed source/input admission, yielding 151,356
unique system/item candidates. The outcome-blind balanced selection exports
30,000 unique fitting records, 1,000 task-disjoint training-family monitor
records and the unchanged original development (963 responses / eight pairs).
There are 15,383 distinct normalized fitting tasks. Model/base facts reach
28,542 fitting records and harness facts 20,236. All detected options survive;
7,286 task/context omissions include 2,490 oversized question/user sections.
No calibration/confirmation labels or supplemental development outcomes read.
Full local native token/render parity passed for all 31,963 exported records.

Qualification v3 passed on L4 with FP16 autocast, FP32 scorer, gradient scaling
and within-window grouping: microbatch four / effective batch 64. Frozen /
upper-two medians 3.437 / 3.783 seconds per 64 records; original-order FP32
12.080 / 13.292 seconds. Larger microbatches used more VRAM but ran slower.
Both synthetic controls reached 100% validation accuracy; tiny real training
Brier .427079 -> .062162 frozen / .063530 upper-two. Controls discarded; these
are machinery checks, not generalization results. BF16 candidates failed the
numerical drift gate; v1 also failed a signed-zero-only disposable probe check
now reproduced and covered by CPU regression. No quality metric was rewritten.

Matched final experiment submitted as `job_22bab18c0ed84ae1bd90a81377078cb3`:
identical enriched fitting inputs, seed, optimizer windows and precision; frozen
encoder then final-two-block adaptation, one epoch / 469 updates per arm.
Original development is scored once per fixed-epoch checkpoint; it does not
select a checkpoint. Separate learned evidence fitting uses only training-role
monitor queries/supports after heads complete. Guide, receipts and limitations:
[enriched 30k and encoder training](enriched-30k-encoder.md). At this entry,
main training is active and no new development score is available.

The user subsequently requested finishing only the active run, reporting Brier
and starting no new run. The frozen arm remains active; the queued upper-two
arm, evidence fitting and extra inference/evaluation are deferred. An external
controller stops the parent after the frozen arm's finished report and preserves
its checkpoint. This changes the scope from a completed paired encoder study
to a single 30k enriched frozen-encoder run. The original plan remains prepared,
not automatically authorized for continuation.

Completed frozen arm: 469 updates / one epoch / 30,000 records, original
development **Brier ALC 0.2721625032** (constant .25; previous compact 5k
.2486426089). C-Eval .2541254; ERBench .2901997. Training-family monitor
.1686245 is not the unseen-family result. More confident development predictions
(.2271-.8798) did not improve this score. Encoder/action bytes unchanged; native
export/reload fixture drift zero; downloaded artifact hashes verified.
The parent was intentionally interrupted during final logging after the finished
arm, before any upper-two optimizer update. Local MLflow
`9448ce30b7c34f6581900633de5ff40d` records the completed arm with the parent-stop
distinction. GPU app stopped / zero containers. Encoder training, evidence
fitting and extra inference/evaluation remain held. No paired encoder result
exists. See the [full result and scope](enriched-30k-encoder.md).
