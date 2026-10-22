# First head-only development scores

Measured 2026-10-22 after the 256-row, 10-update smoke training. This is an
exploratory development result, not confirmation or a complete pilot.

| Predictor | Brier ALC |
| --- | ---: |
| Always predict 50% | 0.250000 |
| Original Laya with shipped calibration | 0.379140 |
| Head after 10 updates, no calibration | 0.394842 |
| Original Laya, no calibration | 0.459007 |

Lower is better. With temperature 1 for both, the trained head improves by
**0.064165**, winning **8/8 pairs**. Both development benchmark averages improve.
This is evidence that the short training changed the head in a useful direction
on this sample. The head's current uncalibrated pipeline remains worse than the
original calibrated pipeline by **0.015702**, losing **7/8 pairs**. Both are worse
than always returning 50%.

The initial three-way comparison revealed the calibration difference, so we
added the fourth, matched raw control. Only that control required additional
inference; the other three completed runs were reused without changing their
metrics or historical comparison artifacts. No temperature was fitted or chosen
using these results. Original temperatures are inherited from upstream; the
raw control explicitly loads temperature 1 with an empty per-option map in
memory. Base checkpoint files and weights are unchanged.

## Sample and scoring

All variants use the frozen development export: **960 subject–item inputs**,
**963 observed response outcomes**, **8 subject–benchmark pairs**, and **2
benchmark families**. Each Laya model processes the same 930 distinct rendered
states, using MLX FP32, batch size 8, the original correctness question, JSON
state, and a 512-token sequence limit. Token and truncation counters match:
18 of 930 unique states truncate. No prompt changes or new training occurred.

The preparation's 50/50 acquisition/evaluation split and trial grouping remain
unchanged. Only exported evaluation-pool responses are scored. These predictors
ignore revealed examples, so their scores are flat at budgets 0/1/3/7/15/31:
**Brier ALC equals budget-zero Brier**. All recorded response trials contribute
within a pair; the final score gives every pair equal weight. Independent
response-level calculations reproduce the scores. Squaring error against an
item-average label would produce the wrong competition score.

| Benchmark | 50% | Original calibrated | Tuned raw | Original raw |
| --- | ---: | ---: | ---: | ---: |
| C-Eval | 0.250000 | 0.405387 | 0.419799 | 0.487730 |
| ERBench | 0.250000 | 0.352893 | 0.369884 | 0.430283 |

The tuned probabilities average 17.6% while observed success is 50.6% across
these response rows. The model is substantially too pessimistic on this sample.
This is a descriptive diagnostic, not a fitted calibration or permission to
calibrate on development labels.

Pair-bootstrap intervals for tuned minus original are [-0.090910, -0.038074]
against raw probabilities and [+0.007991, +0.024668] against shipped calibration.
Pairs share only two families; those intervals do not establish generalization
across unseen benchmark families. Seven of eight pairs have missing normalized
model names; other supplied system fields are preserved. Future work must not
fill missing model names with unavailable metadata.

## Artifacts and tracking

- Matched initial comparison: `runs/head_finetuning/development_smoke_v1/`.
- Raw control and combined report:
  `runs/head_finetuning/development_smoke_raw_control_v1/report.html`.
- Per-input predictions, pair/budget scores, benchmark breakdowns, exact runtime
  and checkpoint provenance, source snapshots, and validation are saved.
- Timings in these runs include model initialization; inference-only timing is
  recorded separately in the details' runtime metadata. Do not compare those
  end-to-end timings with prior inference-only measurements.

| Run | MLflow ID |
| --- | --- |
| 50% control | `a0fcc34c33704a209c6ae8193ed2fcdd` |
| Original calibrated | `444613648b3a47bbb0ea3f695a773ceb` |
| Tuned raw | `fe61cdca394d408089ef6530a26b70a2` |
| Original raw control | `bc5a041e84324e64b8c800385cd5e690` |

The confirmation split remains untouched and no calibration-role outcomes were
read. The next proposed experiment is a properly monitored head-only pilot,
with development checkpoint selection and later calibration on its dedicated
role. The full training set's truncation still needs inspection before choosing
that pilot's input handling. These scores do not justify unfreezing the encoder.
