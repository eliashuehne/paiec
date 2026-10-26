# Configuration index

For the main shared collection, see [model/harness information](../docs/shared-data-start.md).
`data/enrichment_handoff_v2.json` exports the full versioned collection, supplemental
research and reviewed preparation views. Its inventory pins every source file.
The dataset-only and Laya recipes below describe separate experiments.

`head_finetuning/grouped_enriched_v1.json` binds the enriched-only multilingual
grouped training recipe to its exact input export and passed qualification.
`pipelines/grouped_enriched_preparation_v1.json` prepares its tokens through
ZenML; `pipelines/grouped_enriched_training_v1.json` plans its single training
arm. Training and evaluation completed with Brier ALC 0.282439 versus constant
0.25; Laya work is now on hold. These recipes remain historical references. See
[grouped enriched training](../docs/grouped-enriched-training.md).

`head_finetuning/multilingual_qualification_v1.json` freezes the original
multilingual parent, 8,192-token grouped-input policy and bounded learning gates.
`pipelines/multilingual_qualification_workflow_v1.json` dispatches that probe
through ZenML/Modal. It never starts a full training pilot. See
[multilingual qualification](../docs/multilingual-qualification.md).

`pipelines/laya_collected_inputs_v3.json` selects the latest collected-data
input export and native tokenizer audit. Earlier v1/v2 input audits remain
reproducible. See [Laya input contract](../docs/laya-input-contract.md) for actual
field delivery, context overflow and the multilingual 8k audit.

Canonical workflow recipes: `pipelines/laya_training_workflow_v1.json`,
`pipelines/evidence_validation_workflow_v1.json` and
`pipelines/evidence_release_workflow_v1.json`, alongside preparation/selection
recipes below. Plans do not execute; training and release are explicit.
`data/data_versions_v1.json` records data roles and retirement eligibility.
Use [the workflow guide](../docs/workflows.md) rather than comparing version
numbers across unrelated data/input contracts.

`pipelines/competition_evidence_workflow_v1.json` selects training-only fixed
controls and whole-family evidence-policy selection through ZenML. It cannot
launch Laya, access development outcomes or approve a release. See
[workflow organization](../docs/workflows.md).

Competition dataset preparation: `pipelines/competition_data_protocol_v2.json` selects
only supplied public-data fields and separate training targets. Research model
profiles and manual annotations are not used. See
[the competition route](../docs/competition-pipeline.md).

`pipelines/competition_pairs_v1.json` declares the whole-public-corpus pair
matrix: admission rules (binary, item granularity, text-only, 80 items), the
corpus revision and the recorded label direction. `pipelines/competition_controls_v1.json`
declares the fixed-control draws and their evidence weights;
`pipelines/rate_model_v1.json` declares the benchmark folds, shrinkage, difficulty
penalty, evidence weights and the optional CatBoost item term. See
[pair matrix and rate model](../docs/competition-pairs.md). None of the three
authorizes a GPU job, a sealed role or an upload.

`pipelines/competition_panel_v1.json` fixes the outcome-blind source augmentation.
All original rows remain, with 4,718 additional training records to give all
170 visible scopes full-budget coverage. Historical data recipes stay unchanged.

`pipelines/conditional_control_v1.json` fixes the initial CPU-only conditional
predictor comparison. Its fitted variants were rejected by the first training-
family quality checks; see [the measured control result](../docs/conditional-control.md).

`pipelines/training_data_expansion_v1.json` fixes the unused-benchmark expansion
A/B: seven predeclared text-only `higher_is_better` benchmarks, per-benchmark row
caps, repeat count and the fixed ridge L2. It admits nothing and promotes nothing;
see [the measured study](../docs/training-data-expansion.md).

`pipelines/catboost_base_v1.json` and `pipelines/catboost_base_v2.json` fix the
CPU CatBoost base experiment: six declared depth/learning-rate/L2 configurations,
maximum tree count, early-stopping patience, seed, thread count and evidence
controls. v2 changes only the iteration floor after v1 showed it binding every
fold. `pipelines/catboost_base_v3.json` adds the declared
`feature_groups: ["task_structure"]` recipe, which fits only those columns and
records `ablations_run: false`. All panel runs are reported in
[the CatBoost result](../docs/catboost-base.md).

`pipelines/catboost_base_v4.json` adds the research-only `system_profile` group
with its documented catalog and enrichment policy; `pipelines/catboost_base_v5.json`
is the same recipe with the system-to-fact link scrambled, so the profile block's
effect can be separated from the extra columns alone. Neither is a release
candidate: external profile sources remain unreviewed for competition
eligibility.

`pipelines/catboost_unseen_benchmark_v1.json` fixes the competition-shaped
evaluation: which panel run is scored, the hashed row sampling limits, the
minimum items per subject and the zero-tolerance rule for shared task content
with the panel. It fits no new model. `pipelines/catboost_unseen_benchmark_v2.json`
and `_v3.json` score the profile and scrambled-control panel runs on the same
rows. `pipelines/additive_logit_v1.json` fixes
the additive two-block model's feature groups, penalty grid, convergence limits
and the same unseen-benchmark settings. See
[the CatBoost document](../docs/catboost-base.md) for all results.

`pipelines/catboost_permutation_control_v1.json` declares the permutation study
that turns the single scrambled control into a distribution: the unpermuted
recipe, the evaluation recipe, 19 fresh seeds, the already-run seed-1 replicate
and the observed and task-structure reference arms. It fits exactly those
declared recipes and adds no feature, row or role.

Local feature preparation: `pipelines/feature_preparation_offline_v10.json` selects the
existing expanded bundle and unsealed roles for the preparation-only ZenML
pipeline. See [ZenML](../docs/zenml.md). It does not dispatch training.

Expanded enriched training: `head_finetuning/enriched_30k_data_v1.json` fixes
the 30k training sample inside existing family roles;
`head_finetuning/enriched_30k_encoder_v1.json` fixes the one-epoch frozen versus
upper-two-layer recipe. CUDA operational settings are selected by a separate
saved qualification. See [the guide](../docs/enriched-30k-encoder.md).

Keep scientific configurations immutable once prepared or used. A changed recipe
gets a new descriptive version; do not rename historical artifacts merely to
make their number match the latest input format. None of the files below starts
a job by itself.

## Retained shared settings

| Path | Purpose |
| --- | --- |
| `laya_question.json` | Shared target-system correctness question (`task_correctness_v1`) |
| `suites.json` | Public-data evaluation suites, caps and seeds |
| `execution/profiles.json` | Placement and resources; separate from the model runtime |
| `head_finetuning/frozen_split_v1.json` | Frozen family roles, dataset/export hashes and original input contract |
| `schemas/system-profile-v1.schema.json` | Version 1.0 model/harness/run profile schema |
| `schemas/system-profile-source-v1.schema.json` | Version 1.0 provenance-source schema |

## Head recipes and enrichment policies

| Path within `head_finetuning/` | Status and role |
| --- | --- |
| `compact_pilot_v3.json` | Current 2,048-token recipe; enriched completed one epoch, control deferred |
| `compact_profiles_v3.json` | Compact facts and scoped alias policy for collection-v3 |
| `task_features_pilot_v2.json` | Prepared task-v2 predecessor; not trained |
| `enrichment_catalog_v2.json` | Expanded collection-v2 research policy |
| `enrichment_pilot_v1.json` | Historical enriched pilot recipe, 1,024 tokens; completed cached enriched run uses it |
| `enrichment_v1.json` | Collection-v1 policy used by that earlier head |
| `smoke_v1.json` | Historical bounded 256-row, ten-update training check |
| `data_split_v1.json` | Reviewed source specification for the frozen family allocation |

The frozen split's original `status` and input settings describe its preparation
at that time; later experiments have separate records and recipes. Preserve
its contents and hashes. Sealed calibration/confirmation roles stay sealed.
See [the data split](../docs/head-data-split.md) and
[compact preparation](../docs/compact-inputs-and-development.md).

`experiments/input_design.json` records the earlier prompt/format experiment.
`benchmarks/*.json` are historical public-data research notes, retained for
provenance and tests. They are **not** an active benchmark-specific question
registry; runtime prediction uses the shared question and anonymous IDs.

## Version meanings

- Profile schema `1.0`: homogeneous record structure, independently versioned.
- `collection_v1`, `collection_v2`, `collection_v3`: research-data snapshots.
- `enriched_v1`, `task_v2`, `compact_v3`: distinct input contracts, with matching enriched variants where defined.
- Pilot recipe versions: fixed scientific settings, artifact paths and policies.

A training checkpoint bundles its input policy/catalog and renderer hashes.
Use those bundled files at inference; a newer policy must not silently replace
what an older head saw during training.
