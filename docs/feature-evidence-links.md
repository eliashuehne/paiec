# Training-only evidence collection links

`prepare_training_evidence_links` is an offline ZenML preparation stage selected
by `configs/pipelines/feature_preparation_offline_v10.json`. It writes collection
indices, without reading outcome values, fitting a model, running inference or
changing historical source files. These indices are not capability estimates,
retrieved semantic neighbors or inputs usable by an existing Laya checkpoint.

The first preparation has 30,000 training source rows, 15,383 normalized task
groups and 170 visible system/anonymous-benchmark scopes. The frozen export
already chose one recorded trial per original system/item; this stage does not
recover unsampled trials or treat repeated records as independent questions.

## Two separate uses

For later capability features, every normalized training task is assigned to
one of five folds using a fixed outcome-blind seed. All input variants and
systems for that normalized task share its fold. The stage writes 850 support
pools: one per visible scope and query fold, excluding every task in that fold.
The row index records which pool a query could use. Pools reference one
outcome-blind representative row per normalized support task. This grouping
prevents direct query-label leakage; it does not establish paraphrase or
translation independence, or generalization to unseen families.
These are support-pool exclusions only. They do not make an existing model's
predictions out of fold: any later fitted capability model or query prediction
must separately respect the same exclusions.

For later budgeted evidence episodes, each training scope separately splits
normalized task groups into 50/50 acquisition and query sides. Odd remainders
go to queries. All available rows of a group remain together. A fixed support
ordering supplies nested prefixes at budgets 0/1/3/7/15/31. Actual available
counts are recorded when a pool is too small. Acquisition groups do not enter
the query side. This is a new training-only collection plan, separate from
the frozen evaluation/acquisition manifests, which remain unchanged.

These two plans serve different purposes: the five folds are for future
out-of-fold capability features; the 50/50 splits are for future evidence
episodes. They must not be combined to let a query's label enter its features.

## Boundaries and files

Scope matching uses the complete permitted system description's hash and the
supplied anonymous benchmark ID. It never recovers private subject identities
or joins across models from similar names. Equal visible descriptions stay
indistinguishable. Monitor and original development links are inspected only
to reject normalized task overlap, and contribute no rows to these plans.
Sealed calibration/confirmation roles are rejected.

`evidence_plan_v1/` contains:

- `training_row_index.jsonl`: source-row pointers, visible scope, normalized
  task group and capability fold.
- `out_of_fold_support_pools.jsonl`: eligible same-scope support pointers for
  each capability query fold.
- `evidence_episode_links.jsonl`: acquisition/query groups, query pointers,
  ordered support pointers and nested budget counts.
- `summary.json` and `manifest.json`: counts, seed, source-table hashes and
  artifact hashes. ZenML publication also logs these artifacts to MLflow.

No file contains query outcomes or support outcomes. Source-line pointers are
audit/join metadata, not predictor features. A future outcome-materialization
stage must bind the exact frozen source manifest, project only permitted
support outcomes into inputs, keep query targets separate, retain raw label
semantics and recheck fold/scope/budget constraints. Fitting remains held under
the existing stop instruction. No empirical capability or evidence correction
has been fitted by this stage.

The current full feature dataset is still incomplete. Task annotations require
coverage and quality work; system/task compatibility and a new input contract
remain unfinished. Green preparation steps do not establish training readiness.

Verified execution: `runs/feature_pipeline/offline_evidence_preparation_v1/`,
ZenML `e9240899-a17c-4051-83c4-5956628e6a6d`, MLflow
`4a5d5a4cefdd4dfa8da46d5648070485` (FINISHED). A full-data audit checked all
30,000 row links, 850 fold pools, all 170 episode splits and nested budgets,
53 root-manifest artifacts and 18 child artifacts logged to MLflow. It found
15,051 query-side rows in these training-only episodes. Evidence:
`runs/task_annotation_review/offline_evidence_verification_v1.json`.
The complete suite ran 285 tests successfully with 12 environment skips;
the ten annotation tests also pass in the pipeline environment. Four profile
catalogs and whitespace checks passed. No model calls or fitting occurred.
