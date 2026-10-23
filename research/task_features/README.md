# Task-feature collection starter

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
review tasks. The seven longer examples are intentionally unannotated pending
review. Source documents/model facts remain in the original immutable profile
bundle. `collection_queue.jsonl` distinguishes missing static facts, existing
facts requiring applicability review, unresolved visible identities, and raw
recorded settings requiring typing. Defaults must never become verified run
settings. Sources need version/date applicability and competition eligibility
review; new external-source collection has not run in this starter.

Next: independently review these candidates, refine the taxonomy, annotate
the longer examples, then measure agreement on a larger label-blind sample
before bulk annotation. A teacher/API extractor is not implemented or launched.
Capability/difficulty fitting and compatibility/evidence features remain
separate training-only stages. Existing Laya renderers are unchanged.
