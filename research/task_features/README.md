# Task-feature collection

Current contract: `configs/pipelines/task_annotation_v2.json`, taxonomy
`configs/schemas/task-taxonomy-v2.json`, prompt
`configs/prompts/task_annotation_v2.txt`. `candidates_v2.jsonl` contains 14 manual
agent candidates with exact evidence, nullable difficulty and no answers.
None has an independent quality review. The pipeline prepares full-input
requests for all 16,909 selected task inputs and an unsent 14-task pilot.
Offline export/import is implemented; no teacher calls have been made.
The user chose offline preparation. See [the guide](../../docs/task-annotation-collection.md).

The v2 candidates supersede the seven v1 seeds for the new annotation stage.
SQL schemas do not establish whether execution is required; absent context
does not prove an impossible target task. Historical v1 outputs remain
unchanged and are not a reviewed training dataset.

## Historical v1 starter

`seed_v1.json` contains seven agent-authored candidate assessments of real
training inputs, selected without outcomes: the shortest task input in each of
the seven training families. These examples are not independent human reviews
or a representative annotation-quality sample. No answers, target responses,
labels, benchmark identity features or fitted difficulty estimates are included.

The ZenML collection stage binds each seed to a full visible-input hash and
converts supporting quotes to exact character spans. Schema:
`configs/schemas/task-assessment-v1.schema.json`; validation model:
`paiec.feature_collection.TaskAssessment`. Quotes substantiate estimates; they
do not establish empirical difficulty or system proficiency. Missing claims
are unknown, not false. Family names are sampling metadata only.

The pipeline also exports the median-length task per family, for 14 starter
review tasks. The seven longer examples were unannotated in v1; v2 now has
candidates for them as well. Source documents/model facts remain in the original immutable profile
bundle. `collection_queue.jsonl` distinguishes missing static facts, existing
facts requiring applicability review, unresolved visible identities, and raw
recorded settings requiring typing. Defaults must never become verified run
settings. Sources need version/date applicability and competition eligibility
review; new external-source collection has not run in this starter.

Next: independently review these candidates, refine the taxonomy, annotate
the longer examples, then measure agreement on a larger label-blind sample
before bulk annotation. A teacher/API extractor was not implemented or launched in v1.
Capability/difficulty fitting and compatibility/evidence features remain
separate training-only stages. Existing Laya renderers are unchanged.
