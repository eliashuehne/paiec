# Training-data expansion: measured transfer

2026-10-24. Diagnostic only: this study admits no data, fits no candidate
predictor and changes no frozen role. It exists so a later admission decision
has a rerunnable experiment and a recorded baseline.

## Question

Would adding the 47 eligible public benchmarks that no current role uses
materially improve prediction on a benchmark the model has never trained on?
The competition evaluates a benchmark-level cold start on a private pool, so
cross-benchmark transfer is the property that matters, not in-benchmark fit.

## Design

```bash
.venv-pipeline/bin/python scripts/run_training_data_expansion.py \
  --out runs/competition_audit/training_data_expansion_v1
```

Recipe: `configs/pipelines/training_data_expansion_v1.json`. It needs the
prepared protocol panel (`runs/competition_pipeline/protocol_panel_v2`) and the
public data root (`payload/data`); both are local, generated artifacts.

Instrument: the existing portable lexical/system ridge control
(`conditional_features` + `fit_conditional_ridge`), L2 fixed at 100 with
interactions, unchanged from the rejected conditional experiment. It is reused
only as a *relative* instrument: every other choice is held fixed, and its
absolute error is not a quality claim. No CatBoost or other new model is fitted.

Protocol: leave-one-training-family-out over the panel's seven families.
Repeated query task groups are removed from the fitting side. Development,
monitor, calibration and confirmation outcomes are never opened; the frozen
split's non-training benchmark names are refused outright.

Two metrics are reported:

- **equal scope**: per-scope mean squared error over every panel scope, then an
  equal average across scopes. This has the official shape and is dominated by
  `bird`, which owns 120 of 170 scopes.
- **equal family**: per-family means averaged equally. This is a
  composition-neutral diagnostic.

Part A fits every subset of the non-held families to measure how much *family
diversity* is worth. Part B adds row draws from seven predeclared unused
benchmarks and compares them with the unmodified panel fitting side and with an
equal-row control sampled from the panel's own families. That control separates
"more rows" from "more benchmarks". Each repeat re-draws both added rows and
control rows by hash, so draw-to-draw spread is visible.

Admission filters are enforced before any table is read: text-only modality,
`higher_is_better` direction, binary item-level granularity, at least two
eligible pairs, and no non-training role in the frozen split. Selection uses
hashed identifiers only; labels never choose rows.

## Result

Panel: 34,718 training rows, seven families, 170 scopes, positive rate 0.4255.

### Family diversity saturates early

Equal-family cold-start Brier, by number of non-held families used:

| Training families | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---:|---:|---:|---:|---:|---:|
| Equal family | 0.3158 | 0.3051 | 0.2997 | 0.2975 | 0.2969 | 0.2971 |
| Equal scope | 0.2882 | 0.3072 | 0.3260 | 0.3417 | 0.3527 | 0.3612 |

Moving from one to three families is worth 0.016; three to six is worth 0.003.
The panel already has seven, so more *families* have almost nothing left to
give at this representation. The equal-scope column moves the other way because
more families shift a single global intercept away from `bird`'s very low
success rate; that is a calibration effect, not a task-understanding effect.

### Row count versus benchmark diversity

Three hashed draws per arm; `base` and `base+panel_rows` share the same panel fit
side, so the base row is exact. Added rows are 12,700 per repeat from 254 scopes
across seven benchmarks, a 43% increase over the 29,836-row fitting side.

| Arm | Rows | Equal scope | Equal family |
|---|---:|---:|---:|
| base (panel only) | 29,836 | 0.3612 | 0.2971 |
| base + equal panel rows | 42,536 | 0.3535 | 0.2974 |
| base + unused benchmarks | 42,536 | 0.3404 | **0.2698** |
| unused benchmarks only | 12,700 | 0.3340 | 0.2772 |

Draw ranges:

| Arm | Equal-scope range | Equal-family range |
|---|---:|---:|
| base + equal panel rows | 0.3502–0.3584 | 0.2969–0.2979 |
| base + unused benchmarks | 0.3079–0.3572 | 0.2656–0.2745 |

On the composition-neutral metric, new benchmarks are worth about 0.027 Brier
(9% relative) while an equal number of panel rows is worth 0.000. The
equal-family spread across draws is 0.0089, roughly a third of that gain. On the
official-shaped equal-scope metric the mean gain is 0.021 over base and 0.013
over the equal-row control, but the draw spread is 0.049 — larger than the
effect, so **the equal-scope improvement is not established**.

### The gain is concentrated

Per-family cold-start Brier, mean over draws:

| Held family | base | +panel rows | +unused | Δ |
|---|---:|---:|---:|---:|
| benger | 0.4115 | 0.4093 | 0.2569 | **−0.1546** |
| gsm8k_math | 0.2519 | 0.2557 | 0.2392 | −0.0127 |
| fewshot_ttt_bbh | 0.2563 | 0.2597 | 0.2442 | −0.0121 |
| clasheval | 0.2677 | 0.2710 | 0.2571 | −0.0106 |
| cellverse | 0.2532 | 0.2568 | 0.2466 | −0.0066 |
| abstract_reason | 0.2672 | 0.2684 | 0.2672 | +0.0001 |
| bird | 0.3721 | 0.3611 | 0.3773 | +0.0051 |

`benger`'s positive rate is about 0.80 while the panel's fitting-side mean is
about 0.43, so most of its gain is the global intercept moving toward its base
rate. Excluding `benger`, the other six families improve by 0.006 on average,
with two of six worse. Every arm remains far worse than the constant-0.25
baseline, which is the stronger reason not to read this as a usable predictor.

## Interpretation

- Benchmark diversity is worth *something* on a composition-neutral metric, and
  is not explained by row count alone. The measured effect is modest and
  concentrated in one family with an extreme success rate.
- On the official equal-pair weighting, draw-to-draw variation is larger than
  the mean gain. A single draw can show anything from 0.308 to 0.357; an earlier
  one-draw pilot of the same design landed at 0.355 and looked like a null or
  negative result. Report ranges, not one draw.
- The binding constraint is representation, not corpus size: a global linear
  lexical model cannot express task difficulty separately from system strength,
  and it cannot clear 0.25 with either fitting set.
- A wider pool does exist, but it is smaller than the 47/6,680 headline. Only
  18 unused benchmarks are simultaneously text-only, `higher_is_better` and
  eligible for at least two pairs (764 pairs). `alignment_faking` alone is 4,278
  pairs and 2.1M rows, and its `unordered` event is not a correctness target;
  seven unused benchmarks are wrong-target and 15 are multimodal.

## Decision guidance

Do not admit unused benchmarks to the modeling set on this evidence. Re-run this
study as the admission gate after a base predictor can beat the constant-0.25
baseline cold start; at that point the comparison is meaningful and the same
row-matched control applies. If data are admitted later, keep the enforced
filters, cap rows per benchmark so one benchmark cannot dominate the equal-scope
weighting, sample one response per subject-item outcome-blind, preserve native
labels, and leave sealed roles untouched.

## Artifacts

Local, Git-ignored: `runs/competition_audit/training_data_expansion_v1/`
(`summary.json`, `family_diversity_curve.csv`, `expansion_arms.csv`,
`expansion_per_family.csv`, `added_rows_index.jsonl`, `feature_contract.json`).
The run records a local MLflow entry in **PAIEC Competition Analysis** unless
`--no-tracking` is passed. A repeated run reproduces every reported score to
floating-point precision; only last-bit BLAS differences appear.

## Limitations

- Lexical ridge control only; CatBoost or a text encoder may use the same rows
  differently.
- One dataset revision, seven training families, one model configuration, fixed
  L2, no tuning.
- Repeats vary the hashed row draw; they do not resample systems, tasks,
  acquisition trajectories or families.
- `higher_is_better` is a coarse success proxy. External-source competition
  eligibility remains unreviewed.
- A fresh checkout cannot run this without the prepared panel and public data;
  neither is tracked in Git.
- The official metric is equal-pair, so the equal-family result is a diagnostic,
  not a score.
