# Complete protocol panel and portable evidence-policy candidate

2026-10-24. The original 30k sample's protocol-coverage gap is repaired in a
separate dataset-only source version. Earlier preparations, trained checkpoints,
scores and frozen roles are unchanged. The full pipeline goal is not yet done:
system stress tests and the final
reproducibility report remain outstanding.

## Source augmentation

```bash
.venv-pipeline/bin/python scripts/prepare_competition_panel.py \
  --out runs/competition_sources/NEW_SOURCE
```

Recipe: `configs/pipelines/competition_panel_v1.json`. The loader reuses the
original reviewed training admission and trial draw, verifies all minimal
source-table hashes, and excludes both reserved-family task groups and the
base training-family monitor assignment. It reads no trace table, profile
catalog, model weights or external data. Private source pointers are used only
for admission/provenance, never predictor inputs.

All 30,000 original training records and their order, state strings, questions,
targets and source trial choices are retained. Added pointers are selected by
seeded hashes of source identifiers and normalized task groups, without outcome
values. Duplicate question variants do not count toward the minimum. Source
availability, unchanged scope membership and the row bound are checked before
selection is published.

The actual source is `runs/competition_sources/protocol_panel_v1/`:

- 4,718 added records; **34,718 total training records**.
- All **170 visible system/benchmark scopes** have at least **80 independent
  normalized task groups** and therefore cover all label budgets.
- Monitor/development input states and questions are unchanged. Their expected
  outcomes are omitted from the new source export.
- The selected extra observations connect existing questions with additional
  systems/configurations. Unique task input count and retained text remain
  16,909 and 63,300,770 characters; these are not 4,718 new questions.

The new source follows the same supplied-field projection as the base public
preparation. It does not rewrite raw table cells, invent missing identities or
change any historical renderer/token contract. Its separate family provenance
file contains scorer metadata only; obsolete token/render hashes are not reused.

## Dataset-only preparation and comparisons

```bash
.venv-pipeline/bin/python scripts/run_competition_pipeline.py \
  --config configs/pipelines/competition_data_protocol_v2.json \
  --out runs/competition_pipeline/NEW_DATA
```

The CLI default now selects that recipe. The first completed dataset bundle is
`runs/competition_pipeline/protocol_panel_v2/`; ZenML
`3c796d21-3296-4798-b4fc-5aee5307ab64`, MLflow
`8e6b6b6e3ad04afb8f57b3714894c6d1`. Training targets remain separate from inputs.
All original training row keys remain unchanged. Calibration/confirmation and
supplemental development outcomes remain unused.

The seven-family control was rerun with identical L2 and evidence settings via
`configs/pipelines/conditional_protocol_control_v2.json`. All 170 scopes now
contribute to full-budget comparison; none is excluded. These results use a
different cohort and acquisition splits from the earlier 50-scope diagnostic.
Do not attribute the metric change to improved generalization.

| Method | Training-family Brier ALC, all 170 scopes |
| --- | ---: |
| Constant 0.5 | 0.250000 |
| Acquired-label mean | 0.149105 |
| Fixed beta mean, strength 5 | 0.160365 |
| Conditional ridge, beta mean | 0.199843 |
| Conditional ridge, log-odds offset | 0.195096 |

Cold-start conditional Brier remains poor (0.361193). All six fitted control/
update candidates pass the repaired coverage check but fail the predictive
quality checks. They remain rejected, with results preserved in
`runs/competition_analysis/conditional_protocol_family_v2/`, MLflow
`b04e7ce0ff1546f9bcbb86f02139fa4e`.

## Training-only smoothing selection

```bash
.venv-pipeline/bin/python scripts/select_evidence_policy.py \
  --prepared runs/competition_pipeline/protocol_panel_v2 \
  --out runs/competition_analysis/NEW_SELECTION
```

The predeclared prior-strength grid is 0, 0.5, 1, 2, 5, 10, 20, with prior mean
fixed at 0.5. For each held training family, strength is selected using only the
other families' acquisition/query diagnostics. The held family's query outcomes
cannot choose its own strength. Two inner weighting strategies are compared:
equal visible scope and equal family. They share the same full-protocol splits.
The final scalar is fitted on all training families after this comparison.
No original development or monitor outcomes enter selection.

The selected strategy is equal scope; final strength is **0.5**. Its fixed-grid
score on all selection families is **0.139463**, which is a fitted selection
diagnostic. The more conservative outer-family estimate is **0.148013**; equal-
family inner selection scores 0.161048. Choosing between those strategies also
uses training-family validation, so neither is an untouched final estimate.
Shared questions/systems make scopes dependent. Hidden-test quality is unknown.

Selection: `runs/competition_analysis/evidence_policy_selection_v1/`, MLflow
`1c1598a14c7c427ca7d863cb5c3d6035`. The frozen `evidence_policy.json` contains its
source/selection provenance. The method estimates a shared native-event mean
within a visible scope; it does not model within-pair question difficulty.

## Submission candidate and local qualification

```bash
.venv-pipeline/bin/python scripts/build_competition_submission.py \
  --selection runs/competition_analysis/evidence_policy_selection_v1 \
  --out runs/submissions/NEW_PACKAGE
```

The allowlisted ZIP contains only the entry point, JSON policy, README and two
standard-library runtime modules with a package initializer. No raw training
data, research facts, fitted conditional control, API key, third-party runtime
dependency or model repository is included. It omits `labeling.py` and therefore
uses the platform's default acquisition. File hashes and ZIP bytes are
deterministic for unchanged inputs. Missing known input fields are filled with
empty strings for older local smoke compatibility; supplied strings are kept
literal and private/unknown fields are rejected.

The candidate is `runs/submissions/evidence_policy_candidate_v1/submission.zip`,
**3,393 bytes**, SHA-256
`edde62ca279157ad417439c98e2573c3c2a28d258a241b7a96593d0423ba3df9`.
It passed `baseline/check_submission_zip.py`, including prediction smoke.
This uses the repository's pinned starting-kit version; it is not a hosted
Codabench execution or evidence of compatibility with an uninspected kit update.

```bash
.venv-pipeline/bin/python scripts/qualify_evidence_submission.py \
  --prepared runs/competition_pipeline/protocol_panel_v2 \
  --package runs/submissions/evidence_policy_candidate_v1 \
  --out runs/submissions/NEW_QUALIFICATION
```

The exact extracted package ran in an isolated `-I -S` Python process without
site packages. It made **1,190 calls** on all 170 scopes with global revealed
evidence at budgets 31/0/7/1/15/3/0. Query outcomes were not supplied. Every
probability matched the independent direct calculation exactly; unrelated
evidence and previous budget calls had no effect. Prediction time was 1.725
seconds, peak process RSS 97,550,336 bytes including fixture input data. These
measurements qualify this local workload, not an arbitrary hosted workload.

No upload, paid request, GPU job, native Laya inference or canceled-job restart
was performed. Source/record preservation and 43 MLflow artifact hashes were
checked in `runs/competition_sources/protocol_panel_verification_v1.json`.

Validation: both native environments ran **314 tests** successfully (12 MLX
environment skips, one Torch skip); all four profile catalogs and whitespace
checks passed. The shared field/evidence boundary was extracted into a small
standard-library module. Both conditional fold-model variants were requalified
after this refactor: all 14 models / 42 real fixtures retain 1.11e-16 maximum
drift. The earlier staged commit checkpoint still has its original 75 files;
these continued-work additions have not been committed or uploaded.

## Next gates

The frozen-development assessment is complete in
`runs/competition_analysis/evidence_policy_development_v1/` (MLflow
`f9640ab271e942e4b1358081c137dab2`). On the original eight pairs / 963
responses, the selected strength-0.5 policy scores **0.264185** Brier ALC,
constant 0.5 scores **0.250000**, and the predeclared fixed-strength-5 control
scores **0.239531**. Lower is better. The selected policy **fails** its
development gate; the existing candidate ZIP is not an approved release.
The predeclared recommendation is the fixed-strength-5 control, without
refitting. No replacement ZIP has been built. Development was used for this
choice and has been inspected historically; these are not fresh test results.
Calibration, confirmation and supplemental development outcomes remain unused.

At the user's request to conserve usage, expansion stops at this checkpoint.
The separate 75-file offline research-feature change remains staged; this
dataset-only experiment and the native-training work remain outside that
staged commit. No additional fit, evaluation, API call or cloud job is queued.

The candidate is ready for local review, not a declaration that the entire
project is complete. The orchestration migration is now implemented; see
[workflow organization](workflows.md). Remaining scientific work includes
an unseen-system stress analysis,
and the final selection/fallback and reproducibility report. Task-dependent
adaptation remains a useful experimental direction; the current lexical control
has not demonstrated enough transfer to justify deployment.
