# Frozen data split for the head-only pilot

Prepared **2026-10-22**. Split ID: `head_family_split_v1`.
Status: **data prepared; no training or model inference performed**.

The source decisions are in `configs/head_finetuning/data_split_v1.json` and the
frozen assignments/fingerprints in `configs/head_finetuning/frozen_split_v1.json`.
The current local exports are under `runs/head_finetuning/data_v2/`.

## Actual allocation

The review admitted 14 benchmarks representing 13 source families. With at least
two families in each held-out role, the closest feasible allocation to the
proposed 60/20/10/10 is **7/2/2/2 families**, or approximately **54/15/15/15%**.
Family percentages are not response-row percentages.

| Role | Families | Included public benchmarks |
| --- | ---: | --- |
| Training | 7 | abstract_reason; adaptivestep + critic_discernment_game (one GSM8K/MATH family); benger; bird_sql; cellverse; clasheval; fewshot_ttt_bbh |
| Development | 2 | ceval; erbench |
| Calibration | 2 | causal_axioms; afrieval (all MasakhaNER variants together) |
| Confirmation | 2 | evouna; bertaqa (English/Basque and shot variants together) |

Families mean shared source tasks or benchmark variants, not simply topics or
languages. Two independent Chinese knowledge benchmarks are not automatically
one family. AdaptiveStep and Critic Discernment Game share GSM8K/MATH-500 and
therefore stay together. Alpha-SQL belongs to the BIRD source family but remains
deferred because its input completeness needs further review.

All subjects and trials from a benchmark inherit its role, including those not
selected for this pilot. Systems may occur across different benchmark roles:
this is a benchmark holdout, not an unseen-model holdout.

## Review decisions

All 61 downloaded benchmarks have an explicit decision:

- **14 eligible** for this pilot, based on recorded correctness/task-success
  semantics and available text inputs.
- **8 historical monitoring benchmarks:** BBQ, EDITBench, DevBench, CMMLU,
  AfriMed-QA, CRUXEval, AI2D, and HELM-AFR. They are excluded from every new
  fitting/selection role and are not presented as untouched confirmation data.
- **2 monitoring-related benchmarks:** AGI-Elo and Bridging Gap, grouped with
  HELM-AFR through MMLU/Winogrande sources. They are also reserved.
- **12 excluded** because their native events or subject/input types do not
  establish the universal target-correctness task for this pilot.
- **25 deferred** for missing modalities/context, insufficient 80-item pair
  coverage, extreme/context reconstruction requirements, or unresolved semantics.

This is a conservative metadata/record screen, not an independent regrading of
every released answer. Source outcomes are preserved. Benger admits only items
whose supplied feature explicitly says `input_scope=recorded_prompt`; other
Benger items stay in the same family role but are not pilot candidates.
AdaptiveStep records without a binary execution/answer verdict are not treated
as failures. See the local `benchmark_review.csv` for each decision and evidence.

## Random sampling within roles

Seed: **20261022**. Choose up to four eligible subjects per benchmark, requiring
at least 80 distinct linked binary-observed items per selected subject.
Sampling uses identifiers and input properties, not label values or model scores.

| Export | Size | Interpretation |
| --- | ---: | --- |
| `training.jsonl` | 5,000 responses | 24 system–benchmark pairs; 4,922 distinct system–item combinations |
| `development.jsonl` | 963 responses | 8 pairs; 960 distinct evaluation system–item combinations, including repeated trials |
| `calibration.jsonl` | 1,000 responses | 6 pairs; independent calibration families |
| `confirmation_inputs.jsonl` | 960 inputs | 8 pairs; one input per system/item, with **no target labels exported** |

Training and calibration are sampled without replacement, cycling through pairs
to distribute records evenly. Scarce pairs can contribute fewer records; the
sidecar supplies weights so each sampled pair has equal total weight. Do not
balance successful and unsuccessful classes artificially.

Development/confirmation use the existing 50/50 item split, with up to 40
acquisition and 120 evaluation items per pair. Pool caps are applied after the
50/50 split. Acquisition items are never exported as scored evaluation targets.
Their IDs and order remain in `evaluation_manifest.csv` for future budget-aware
comparisons. The first head pilot itself ignores revealed evidence.

Confirmation repeats are represented in trial-level provenance rather than
duplicated input-only JSONL lines. Its outcomes remain in the original local
Parquet tables, for scorer-side use after choices are frozen. This is procedural
separation, not cryptographic access control.

## Input visibility and assignment selection

The first seeded draw (`data_v1/`) had all sampled development/calibration states
truncated at 512 tokens. It passed overlap checks but was rejected as a poor
pilot design. Its artifacts and a `decision.json` supersession note were retained.

The final allocator takes the first seeded permutation satisfying input-only
constraints: training spans the reviewed broad domains, and each held-out role
contains English tasks and a short-task family. A fixed 32-item token probe with
real subject profiles determines short-task coverage; no model predictions or
outcome rates guide this choice. The fifth permutation passes these constraints.
Actual exported held-out states must also have at least 10% fitting completely.

The current inputs keep the original JSON/question, no examples, a 512-token
total sequence limit, and the native 192-token question-head budget:

| Role | Fully fitting exported states |
| --- | ---: |
| Training | 45.8% |
| Development | 98.0% |
| Calibration | 98.3% |
| Confirmation | 50.0% |

**Training truncation remains substantial: 2,711 of 5,000 records.** Long legal,
SQL, gene-list, and demonstration-heavy tasks contribute to this. They were
flagged rather than silently removed, retaining the agreed 512-token input
policy. The split is prepared, but this does not mean every training task is
fully visible. The training smoke check must inspect these retained inputs;
changing context length or filtering to short tasks would be a new, explicitly
versioned dataset/input experiment with matching baseline evaluations.

Missing `normalized_name` values remain missing: 417 training records lack that
field. Other permitted system fields can still contain information. Do not fill
the field from unavailable display names.

## Leakage and export checks

- Known shared sources are grouped before role assignment.
- All admitted and historical-monitoring item texts were screened for exact
  normalized task overlap across wrappers; no additional cross-family links
  were found. This does not detect translations or establish exhaustive source
  independence.
- Selected records across the four new roles have no shared normalized task
  hashes or effective 512-token input hashes.
- An approximate 16-minhash word-trigram LSH screen found no cross-role candidate
  pairs with verified shingle Jaccard >=0.9. This is not an exhaustive paraphrase
  detector; its scope and method are saved in `preparation.json`.
- Every exported training/development/calibration label was checked against its
  source response ID. Confirmation cases contain no `expected` field.
- Exported states contain only the competition projection, with no injected benchmark IDs, grading
  fields or joined target outcomes. Naturally occurring content inside permitted
  fields is preserved. All cases use the identical current question.
- Family/item roles are disjoint, pair weights are checked, and exported file
  hashes match the preparation manifest. Source files, code snapshot, package
  versions, row provenance and validation results are saved.

The new statistical baselines must fit **only eligible training-role data**.
Do not reuse historical caches or the ordinary runner's broad “all benchmarks
outside this evaluation suite” selection: that could include calibration or
confirmation families. Earlier public-data prior experiments used a broader
inventory; these are new head-specific holdouts, not a claim that all public
data were previously unknown to this project or Laya's upstream pretraining.

## Files and next step

`provenance.parquet` maps exported JSONL lines to role, source family, subject,
item, response ID, trial, weights, input hashes, truncation and pool membership.
Rows with `exported=false` are provenance/pool records, not extra JSONL examples;
`jsonl_line` is zero-based for exported cases. Gold labels are separate from the
state and are not duplicated into the provenance sidecar.

Reproduce into a **new** directory:

```bash
.venv-laya-mlx/bin/python scripts/prepare_head_data.py --out runs/head_finetuning/NEW
```

The script rejects an existing output directory, dataset/config mismatches,
unreviewed benchmarks, varying per-trial tool inputs within a subject/item,
cross-role exact/near-duplicate candidates, and inadequate held-out visibility.
No trace table or model weights are loaded. Confirmation model scoring is not
part of preparation.

The preparation is logged in the separate **PAIEC Head Fine-tuning** MLflow
experiment as a data-preparation run, with no Brier score or training claim:
[run 84ecbaa2](http://127.0.0.1:5001/#/experiments/3/runs/84ecbaa26110444ba917e9ee7fff5744).
Validation passed: 154 unit tests in MLX; the reference environment ran 154 with
one native-MLX audit test skipped. Logs: `runs/head_data_tests.log` and
`runs/head_data_torch_tests.log`. `git diff --check` passed.

Next: implement explicit local trained-checkpoint loading and a head-only trainer
that consumes the saved pair weights and explicit roles. The stock upstream
random calibration split must not replace these assignments. No gradient
updates, encoder changes, evaluation runs, commits or pushes were performed.
