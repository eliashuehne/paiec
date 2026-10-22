# Outcome semantics investigation

Investigated 2026-10-21 against local dataset revision
`54074453da7b9911be0f48e4c560be547bc84d21`. No labels or predictions were changed.

## Conclusion

The public Data Bank deliberately preserves different binary events. Its binary
tables are not all accuracy tables. The downloader filters binary, item-level
tables and copies their files; it does not convert outcomes into correctness.
The active Laya question asks whether the target system will solve the task
correctly, matching the competition's stated goal. That wording is not equivalent
to every native public event. The runner supplies only competition-visible
system/task fields and optional revealed labels (see `benchmark-questions.md`). Public
benchmark definitions remain research notes, not runtime inputs. There is no
evidence of accidental local label inversion.

The competition page describes success prediction, but also defines the target
numerically as the probability of response 1. Its private evaluator's handling
of heterogeneous event definitions is not established by the public material
inspected. This remains an interface/documentation question for the organizers,
not grounds for silently changing public grades.

## Evidence from authoritative documentation

- The [competition interface](https://aims.stanford.edu/competition#competition)
  exposes eight subject fields and four item fields, including an anonymous
  benchmark identifier. Neither response_scale nor grading_criterion is supplied.
- The [Data Bank schema](https://github.com/aims-foundations/measurement-db/blob/287495967fc276275c3b466dfede2cc3534e830c/parquet_schemas.yaml)
  allows scale directions and explicit value meanings, and prohibits automatic
  grade reversal or inference of meanings from response_type.
- The [Data Bank card](https://huggingface.co/datasets/aims-foundations/measurement-db)
  explains that the release retains original observations and benchmark-specific
  scales. The broader Data Bank is not solely a correctness dataset.
- The linked [curation guide](https://docs.google.com/document/d/1LdUEiUG5tLhDdnIAtm4YKSf4w3LFb2HmI_ahiiuoRLU/edit)
  distinguishes binary success grading from other declared scales. It does not
  establish private competition normalization for the cases below.

## Clear counterexamples in our local download

| Benchmark | Direction | Stored outcome 1 represents |
| --- | --- | --- |
| advprompter | lower_is_better | A judge marked the target's output unsafe |
| crow | lower_is_better | A historical attack keyword rule fired |
| emoji_attack | lower_is_better | A recorded jailbreak succeeded |
| harmbench | lower_is_better | An evaluator marked attack success |
| alignment_faking | unordered | A classifier detected alignment-faking reasoning |
| braveguard | unordered | The guard chose the harmful classification |
| dbpa | unordered | A statistical test did not reject at the specified threshold |

There are 54 higher-is-better, four lower-is-better, and three unordered scales
among the 61 downloaded benchmarks. These seven direction flags are a
conservative diagnostic, not an exhaustive classification: a higher-is-better
scale can represent safety, workflow completion, or absence of a particular
failure mode rather than answer correctness.

## Checks against actual archived records

AdvPrompter: all 100 response rows match the archived final_score field exactly;
23 are 1 and 77 are 0. The [curation builder](https://github.com/aims-foundations/measurement-db/blob/287495967fc276275c3b466dfede2cc3534e830c/benchmarks/advprompter/build.py)
imports that field directly. Its [source scorer](https://github.com/thu-coai/AISafetyLab/blob/3fd15737dee5c6858466162d6a68e6cb6f996b27/aisafetylab/evaluation/scorers/llama_guard_3_scorer.py)
assigns 1 when its decoded judgment contains the unsafe marker. This confirms
the event interpretation, without running a new judge.

BraveGuard: all 536 response rows match the archived harmful decision after
normalizing the native True/False strings. The native correctness field differs
from the stored response on 68 rows; its complement differs on 468 rows.

| Stored response | Native decision incorrect | Native decision correct |
| --- | --- | --- |
| 0: non-harmful decision | 282 | 47 |
| 1: harmful decision | 21 | 186 |

A harmful decision can be correct or incorrect. Its [grading definition](https://github.com/aims-foundations/measurement-db/blob/main/benchmarks/braveguard/metadata.yaml)
explicitly treats the response as a decision, retaining correctness separately.
Neither label inversion nor calling all ones correct recovers that distinction.

Alignment Faking's [grading definition](https://github.com/aims-foundations/measurement-db/blob/287495967fc276275c3b466dfede2cc3534e830c/benchmarks/alignment_faking/metadata.yaml)
retains a detected behavior. Its absence is not a correctness grade either.
The large encrypted trace table was not read or decoded for this investigation.

## Consequences for this project

- Brier score itself is sound: it scores predictions against the chosen binary
  event. It does not decide what the event ought to mean.
- The original prompt asked about a correct answer and labeled acquired outcomes
  CORRECT/INCORRECT. The runner now asks a short shared correctness question
  and supplies acquired outcomes as numeric labels, without consulting
  benchmark-specific definitions or adding criteria. This avoids unavailable
  metadata but does not resolve different native public event meanings. The BLE
  baseline uses correctness language too; it does not settle the mismatch.
- The two individual BBQ and AfriMed-QA examples use reference-answer matching,
  so this issue does not invalidate their label interpretation. None of the
  seven direction-flagged benchmarks is in the current default or smoke suite;
  this is not a certification of all other suite assumptions.
- Do not fine-tune every downloaded binary row under a universal correctness
  question. Initially use reviewed task-success/correctness benchmarks, or build
  a separate experiment that explicitly predicts each declared native event.
- Using benchmark metadata to define an event is possible for public-data
  experiments, but cannot be assumed available in the anonymous hidden interface.
  General event criteria can clarify the target; supplying an item's reference
  answer or hidden grading information would leak the answer and is not a valid fix.
- Preserve source outcomes. A deliberate derived safety target could use 1-y
  with the corresponding probability 1-p, but that is a separate target and does
  not establish correctness or compatibility with private evaluation.

## Remaining organizer clarification

Does private evaluation use native binary events as recorded, normalize them
into task success/correctness, or restrict the test pool to success-graded
benchmarks? If native events are included, how is their definition made
available through the published predictor fields?

No message has been sent to the organizers.

## Reproduce

Run `.venv-laya/bin/python scripts/audit_outcomes.py`. Results are saved in
`runs/outcome_audit/outcome_meanings.csv`, `checks.json`, and
`braveguard_cross_tab.csv`. This inspects all local benchmark scale definitions
and reconciles AdvPrompter/BraveGuard native trace fields, without model inference,
new grading, dataset edits, or hidden-data access.
