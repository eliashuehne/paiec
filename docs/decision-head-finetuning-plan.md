# Plan: fine-tune Laya's decision head

Subsequent preparation: [enriched-head-pilot.md](enriched-head-pilot.md) records
the separately authorized control/enriched experiment using shared 1,024-token
inputs. It is ready for a new user go-ahead; training has not started. The
JSON/512 recipe below is the historical starting plan, not that new experiment.

Date: **2026-10-22**. Status: **data prepared; bounded head-only smoke training performed**.
See [head-training-smoke.md](head-training-smoke.md). No pilot quality result yet.

Data preparation has since been completed and checked; see
[head-data-split.md](head-data-split.md) and the frozen assignment in
`configs/head_finetuning/frozen_split_v1.json`. There are 13 admitted families,
allocated 7/2/2/2, with 5,000 sampled training responses. This completes the data
preparation portion below. Explicit checkpoint loading and a 10-update smoke
check are now implemented; the full pilot and held-out evaluation remain pending.

## Objective and first experiment

Test whether the existing encoder already represents useful system/task
information that a task-specific decision head can turn into better success
probabilities. Continue from the pinned English checkpoint, rather than
initializing a new model. Keep the encoder frozen throughout this experiment.

The first pilot predicts from system and task attributes only: no revealed
examples in the model state. Its budget curve will therefore be flat. This
isolates head adaptation from the separate problem of teaching evidence use.
Evaluate all six budgets anyway, alongside label-aware statistical controls.
Budget-aware head training is a possible second experiment, not part of the
initial pilot.

Keep the current JSON state, `task_correctness_v1` question, 512-token limit,
192-token head budget, tokenizer, and sequential option layout unchanged.
The question/format screen did not find a better starting point; see
[experiment-log.md](experiment-log.md). Missing image content and truncation
remain limitations. Fine-tuning cannot recover removed information.

## 1. Freeze a reviewed dataset and benchmark split

Produce an explicit eligibility manifest before building training rows. Use
only the existing competition-permitted inventory and pinned dataset revision
`54074453da7b9911be0f48e4c560be547bc84d21`.

Review benchmark metadata to decide inclusion; metadata never enters the model
state. For the first pilot, require a reviewed correctness/task-success target
and useful text inputs. Exclude unresolved native-event labels, missing required
modalities, missing outcomes, and unusable task content. A higher-is-better flag
alone is insufficient evidence of correctness. Preserve original labels and
record each exclusion reason; do not invert or relabel them. Do not silently
train on every downloaded binary table.

Create four separate roles, grouped by whole benchmarks and related source
families:

| Role | Purpose | Permitted use |
| --- | --- | --- |
| Training | Update head parameters | Gradient updates |
| Development | Choose learning rate, stopping point, and checkpoint | Inspect predictions/scores during development |
| Calibration | Fit the final probability temperature | Used after checkpoint selection; no gradient updates to the head |
| Confirmation | Assess the locked procedure on new benchmarks | Score once after choices are fixed |

Keep all eight previously used default-suite benchmarks out of training,
development selection, and calibration. They remain a historical monitoring
suite, not an untouched confirmation test. Select additional role assignments
from reviewed eligible benchmarks, using metadata and coverage before inspecting
new model scores. Aim for multiple benchmark families in every held-out role;
exact names and counts must be fixed after the eligibility audit. If too few
families qualify, use grouped development folds and retain a separate final
holdout rather than falling back to random response-row splits.

All subjects and all trials of an item stay in the same role. Audit exact and
near-duplicate task content across benchmark families, and identical effective
tokenized inputs across roles. Resolve cross-role leakage before training.
Unknown future subjects should have a separate diagnostic; benchmark cold start
is the primary evaluation setting.

Deliverables: eligibility/exclusion table, role manifest, dataset/config hashes,
subject/item/trial counts, missing-field and truncation summaries.

## 2. Export competition-shaped training cases

Build each state through the existing competition projection and JSON renderer.
Use only the published eight subject and four item fields. Omit anonymous IDs
from rendered text as today. Missing model names remain missing. Do not supply
reference answers, grading definitions, target outcomes, trace outputs, or
unavailable benchmark metadata as features.

Keep labels separate from input state. A training row uses the upstream JSONL
shape `{state, questions, expected}`, with `expected.correct` equal to the
observed binary outcome as a Boolean. The question remains `type: noul` and
uses the current instructions. Keep provenance in a sidecar indexed by row ID:
benchmark/family, subject, item, trial, split role, raw label, rendered-state
hash, effective token hash, and sampling weight.

Repeated trials are legitimate observations, not duplicate errors to discard.
Train from observed outcomes; no teacher-generated probabilities are needed.
If identical inputs are aggregated later, retain their response counts and
outcome totals rather than selecting an arbitrary label.

The pilot target is **approximately 5,000 response records**, distributed over
diverse training pairs and reviewed families. This is a proposed cap, not a
claim that 5,000 eligible records have already been exported. Use pair-balanced
sampling or weights so huge response tables do not dominate. Within a pair,
responses retain their natural outcome distribution. Do not force a 50/50 class
balance, which would change the probability target.

The upstream loop averages items uniformly. A project-owned training wrapper
must implement and verify the chosen pair-balanced sampler or weighted loss;
passing an unbalanced JSONL to the stock command is not equivalent.

Deliverables: role-specific JSONL/Parquet exports, provenance sidecar, sampling
manifest, tokenizer-only visibility audit, and input/label separation checks.

## 3. Add explicit trained-checkpoint evaluation

The current runtime factory resolves only the original pinned upstream
checkpoint. Before a meaningful pilot, add an explicit local checkpoint option
through the shared predictor/runtime configuration. Preserve the original
checkpoint as the default and untouched reference. Never silently substitute a
trained artifact or mutate the cached base weights.

Record the trained artifact's tensor/config/tokenizer hashes, parent checkpoint,
training run ID, dataset/split manifest hashes, and runtime packages. Include the
artifact identity in output locations and any persistent model-dependent caches.
Keep lazy backend imports and the shared state/question boundary intact.

Train with PyTorch in `.venv-laya`, initially on MPS. The existing native MLX
integration is an inference backend, not a verified training path. Verify that a
saved head-only checkpoint reloads correctly in PyTorch, then qualify its MLX
loading before using MLX for reported trained-model scores.

Deliverable: the ordinary evaluator can explicitly select either the immutable
base or a named trained artifact, with provenance visible in MLflow.

## 4. Establish a bounded training smoke check

Inspect and log actual parameter counts before training. In the installed
version, the decision side includes two transformer layers, type embeddings,
and a probability scorer; it is not just one small linear layer.

- Freeze every encoder parameter and keep the encoder in evaluation mode to
  disable its dropout during head training.
- Update the existing `head`, `type_emb`, and `scorer` parameters. Freeze the
  auxiliary `act_head`, which is outside our probability-prediction objective.
- Use raw two-option logits with supervised soft cross-entropy and observed
  binary targets, equivalent to binary cross-entropy for this decision.
- Disable RLCD, label smoothing, option shuffling, and encoder adaptation for
  the first pilot. These would introduce additional experimental variables.

Use a tiny training-only subset for up to **10 optimizer updates**, measuring
memory, throughput, gradient norms, and loss. Treat its output as a disposable
smoke artifact, not a quality result. Verify that trainable tensors change and
that encoder tensors remain exactly unchanged, using tensor-level hashes rather
than the hash of the whole checkpoint file. Check for non-finite values.

The Mac's feasibility and runtime must be measured here. If memory is too high,
reduce the micro-batch while preserving the effective batch by accumulation.
Do not move work to paid/cloud hardware automatically. Cache encoder features
only if profiling justifies the storage cost; a frozen encoder still costs a
forward pass in the initial implementation.

Deliverables: freeze/integrity check, measured memory/time per update, optimizer
budget and estimated pilot duration. No multi-hour training commitment before
this measurement.

## 5. Run the first supervised pilot

Proposed starting settings, subject to the smoke check:

| Setting | Initial value |
| --- | --- |
| Parent | English, revision `7b928d828b7b0e022f929d9bd2e44165aa270148` |
| Model input | Existing JSON, `task_correctness_v1`, no examples |
| Sequence / head budgets | 512 / 192 |
| Trainable modules | Decision transformer, type embeddings, probability scorer |
| Frozen modules | Entire encoder and auxiliary action head |
| Loss | Supervised cross-entropy, with pair-balanced sampling/weighting |
| Optimizer | AdamW |
| Head learning rate | `1e-4` |
| Weight decay / gradient clipping | `0.01` / norm `1.0` |
| Micro-batch / accumulation | 4 / 16, effective batch up to 64 |
| Epoch cap / seed | 4 / 0 |
| Device / precision | PyTorch MPS, FP32 initially; record actual behavior |
| Checkpoint selection | Lowest uncalibrated development Brier, not training loss |

For exactly 5,000 records, this batching gives 316 optimizer updates over four
epochs; actual updates depend on the sampler's exported epoch size. Log both
micro-steps and optimizer updates, including the final partial accumulation
window. Validate at each epoch. Save the best and latest states separately.

Use independent development benchmarks for stopping/selection. Keep the
calibration and confirmation roles out of this process. Begin with one learning
rate and one seed. If there is a useful development signal, test `3e-5` as one
additional learning rate and repeat the selected recipe with seeds 0, 1, 2.
Do not expand to a broad sweep before the first result.

If loss remains near a constant predictor, distinguish insufficient updates,
optimization problems, missing signal, and inadequate encoder features. A
failed head-only pilot does not by itself establish that full encoder training
will help. Extend the update budget only for a documented learning trend;
otherwise diagnose the setup before increasing cost.

## 6. Preserve explicit calibration and checkpoint semantics

The installed `finetune()` wrapper randomly takes calibration items from its
input rows. That does not enforce our benchmark/family split. Use its lower-level
training, calibration, and checkpoint helpers through the project wrapper,
with explicit manifests; do not rely on its automatic split or treat its
calibration metrics as held-out quality evidence. Do not edit installed package
files to accomplish this.

After checkpoint selection, fit only the `noul` temperature on the dedicated
calibration role. Report both raw and calibrated probabilities. Use a compatible
temperature range and config precedence in both runtimes, removing conflicting
inherited overrides for this trained artifact. Never fit temperature on
development-selected predictions and then call those same rows a confirmation
test. Calibration may fail to transfer; evaluate it rather than assuming a win.

Save model/config/tokenizer together. Preserve the base artifact untouched.
Implement epoch-level training state with optimizer, scheduler, RNG, sampler
position and manifest identity if resumption is supported. Upstream's latest
weights alone do not provide exact training resumption. Validate resume behavior
before promising it. Avoid retaining a full encoder copy for every epoch:
keep best/latest plus the logs needed for audit.

## 7. Evaluate against matched controls

Run all predictors on identical system–item samples and nested acquisition
trajectories, first on development and finally on the locked confirmation role:

1. Constant 50%.
2. Subject prior and shrunk mean, trained outside every evaluated benchmark.
3. Original pretrained Laya.
4. Head-tuned Laya, raw probabilities.
5. The same selected head-tuned checkpoint with its independent calibration.

For the head-only zero-shot predictor, Brier ALC equals its budget-zero Brier;
the label-aware statistical controls may improve across budgets. This difference
must be explicit. Existing local scores (0.3204 pretrained, 0.2351 shrunk mean)
are historical references, not acceptance thresholds on a different sample.

Primary metric: response-correct Brier ALC with equal pair weighting. Also save
per-budget and per-benchmark curves, calibration plots, within-pair ranking
diagnostics, truncation, throughput, and prediction distributions. Retain all
recorded trials for evaluation; squared error against an item-average label
alone is not the competition Brier. Use paired comparisons, benchmark/family
level sensitivity checks, and uncertainty across seeds; related pairs are not
fully independent.

Check runtime fidelity on both short and truncated held-out fixtures. Compare
PyTorch reference with MLX FP32 first, then FP16; log probability differences
and score changes, not just token counts or throughput. Investigate material
disagreement before treating the runtimes as equivalent for the trained model.

Decision gates:

- **Worth expanding:** consistent development improvement over pretrained Laya
  and 50%, without one benchmark accounting for the result.
- **Worth using in the competition predictor:** beats the strongest matched
  statistical control, or adds validated value to a separately selected blend.
  Blending would be a new experiment, not part of this pilot.
- **Worth unfreezing the encoder:** valid training learns the training task but
  the frozen representation appears insufficient on held-out benchmarks. First
  rule out target semantics, leakage, truncation, imbalance, and optimization
  failures. Escalation requires a new experiment rather than automatic full
  fine-tuning.

The locked confirmation result is reported even if disappointing. Changes made
in response to it turn that split into development data and require a fresh
confirmation set.

## Implementation order and completion record

| Step | Proposed work | Done when |
| --- | --- | --- |
| A | Eligibility and grouped role manifests | Label meanings reviewed; roles disjoint; exclusions and duplicate checks saved |
| B | Exporter and weighted sampling | Only permitted fields in state; correct labels/weights/trial grouping; deterministic hashes |
| C | Explicit local checkpoint loading | Base and trained artifacts selectable, independently identified and reloadable |
| D | Freeze and training smoke check | Head changes; encoder does not; finite gradients; measured memory/runtime |
| E | Pilot trainer, checkpointing, MLflow logs | Best/latest saved, development metrics and optimizer budget recorded |
| F | Independent calibration and matched evaluation | Raw/calibrated outcomes compared with controls on correct splits |
| G | Repeat/confirm or diagnose | Decision documented with uncertainty; no automatic encoder unfreezing |

Training logic belongs in project-owned modules/scripts, with Pydantic settings
for data roles, parent checkpoint, trainable modules and optimization. Save
resolved configs, environment/source snapshots, training curves, tensor hashes,
checkpoints, and final predictions under a new MLflow experiment group or clear
head-tuning tags. Keep weights/exports outside Git. Before long evaluations,
add or explicitly account for the evaluator's current lack of incremental
prediction checkpointing; progress counts are not saved predictions.

No training, dataset export, new evaluation, installation, or commit was run to
write this plan. Hosted Codabench runtime compatibility remains a separate
qualification task.

## Sources checked

- Local Laya 0.4.13: `train.py` (`TrainConfig`, `train_model`, `finetune`) and
  `common.py` (`DecisionModel`). Installed code is the implementation reference.
- [Upstream fine-tuning guide](https://nandhakishorm.github.io/laya/finetune/):
  freeze-encoder mode, supported losses, calibration and saved artifacts.
- [Outcome semantics](outcome-semantics.md),
  [competition input boundary](benchmark-questions.md),
  [runtime integration](laya-backends.md), and
  [completed experiment log](experiment-log.md).
