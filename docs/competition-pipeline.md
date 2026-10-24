# Competition pipeline: dataset-only route

Latest continuation: [complete protocol panel and evidence policy](protocol-panel-and-policy.md).
The sampling gap is repaired: 34,718 training records, all 170 scopes covering
the full protocol. The original 30k/50-scope results below remain historical.
Training-family smoothing selection and a small portable candidate ZIP are
implemented. Frozen-development assessment is complete and rejected the tuned
candidate. Unified ZenML orchestration is now available; use
[workflow organization](workflows.md) for the current entry points and boundaries.

Resumed 2026-10-24 from the offline feature-preparation handoff. The full goal
remains unfinished. This route makes the provided dataset usable without web
metadata or a prerequisite teacher-annotation pass. It is separate from the
research feature-collection graph; existing catalogs, renderers, checkpoints,
recipes and staged commit checkpoint are preserved.

## Source policy

The current [competition rules](https://aims.stanford.edu/competition) restrict
competition-specific training, development and curation to inventory sources.
They permit pretrained models, but prohibit runtime browsing/grounding and
restrict runtime network calls to approved inference providers. The predictor
receives eight subject strings, four item strings and budget-limited outcomes.
These requirements were checked on 2026-10-24.

Consequently, the collected external model/harness metadata is research-only;
this route never opens the profile bundle, source registry or manual task
annotation catalog. It uses only original `source/*.jsonl` records from the
pinned public-data preparation and the frozen split. Dataset revision, selected
source bytes and split are checked. Sealed calibration/confirmation are not
accepted roles. No identity is recovered from a private display name or alias.

The copied supplied strings retain all characters, including leading/trailing
whitespace and literal `None`/`null` task text. Missing keys in older exports
are restored as empty strings. Anonymous benchmark IDs remain join metadata,
outside learned model features. No hidden benchmark identity is inferred.

## Implemented preparation

```bash
.venv-pipeline/bin/python scripts/run_competition_pipeline.py \
  --out runs/competition_pipeline/NEW
```

The separate three-step ZenML graph verifies sources, exports dataset tables
and training links, then validates/publishes to the local MLflow tracker.
Configuration: `configs/pipelines/competition_data_v1.json`.

The first execution is `runs/competition_pipeline/dataset_only_v1/`:
30,000 training rows, 1,000 training-family monitor rows, 963 original development
rows; 176 visible systems and 16,909 distinct full task inputs. All 63,300,770
supplied task characters are retained. Training outcomes are stored separately
in `training_targets.jsonl`, with native binary semantics preserved. Outcomes
for other roles are not exported. The three-step graph does not fit a model,
run native model inference, call a teacher API or allocate a GPU.

The output includes `systems.jsonl`, `tasks.jsonl`, `examples.jsonl`,
`training_targets.jsonl`, the training-only `evidence_plan_v1/`, source contract,
summary, report and hash manifest. Nominal capacity text such as `8x7B` is
preserved as a supplied string; no exact or active parameter count is invented.
Recorded settings remain recorded settings, with no harness defaults supplied.

The first run is FINISHED: ZenML `0357b19a-dd58-4376-854b-ce10e8ce5528`, MLflow
`64cfdbd5781c40ceb1bafc4e441a4158`. Actual runtime packaging/hosted execution and
predictor quality have not been verified by preparation.

## First training-only evidence analysis

```bash
.venv-pipeline/bin/python scripts/analyze_competition_data.py \
  --prepared runs/competition_pipeline/dataset_only_v1 \
  --out runs/competition_analysis/NEW
```

This command verifies the dataset-only artifact manifest, reads training
targets only, and scores fixed numerical controls on the predeclared 50/50
training acquisition/query splits. All budgets use nested support prefixes.
The constant is 0.5, the empirical control is the acquired-label mean, and the
fixed smoothing control is `(sum(labels) + 2.5) / (len(labels) + 5)`.
No parameters are optimized. Query targets enter scoring, never predictions.

Only 50 of the 170 retained visible system/benchmark scopes contain at least
80 normalized question groups and enough supports for all budgets. The other
120 retained scopes have too few sampled questions for this full diagnostic.
They are explicitly excluded rather than showing a capped pool as 31 labels.
These are counts in the existing 30k sample, not claims about original source
availability. Evaluation equally weights the same 50 scopes at every budget.

| Fixed control | Training-only Brier ALC |
| --- | ---: |
| Constant 0.5 | 0.250000 |
| Acquired-label mean | 0.243859 |
| Smoothed acquired-label mean, k=5 | 0.213164 |

The unsmoothed mean is especially poor at budget one (0.385400); the smoothed
control scores 0.237844 there. This is one acquisition trajectory on training
families, not new-model or unseen-family generalization. Shared questions and
systems make the scopes dependent. The frozen source selected one trial per
original system/item; these scores do not include all native trials.
Outputs retain per-scope/per-budget scores, predictions, exclusions and code
hashes. No original development, calibration or confirmation outcomes are used.
The checked analysis is `runs/competition_analysis/training_evidence_v2/`,
MLflow `62b8a24985f040308a5fc5422db6aa05` (FINISHED), under
**PAIEC Competition Analysis**. Logging is local and can be disabled with
`--no-track`. The fixed-control scores exactly reproduce the earlier untracked
diagnostic. Bundle/source checks and 17 saved MLflow artifact hashes were
verified in `runs/competition_pipeline/verification_v1.json`.

The response graph also remains sparse: 11,818 of 15,383 training task groups
occur in only one visible scope. Text-conditioned system/task representations
are needed to generalize; item-ID difficulty and model-ID lookup tables would
not solve the unseen-task problem.

Subsequent implemented experiment: [conditional text/system control](conditional-control.md).
The portable predictor, interaction ablation and item-dependent evidence update
are implemented, but both fitted variants fail the initial quality/release gate.
Their seven-family comparison uses training outcomes only; none has been
selected or deployed. The fixed smoothed evidence control remains stronger.

## Completion requirements and remaining work

| Requirement | Current evidence | Remaining deliverable |
| --- | --- | --- |
| Eligible sources and exact input boundary | Dataset-only preparation, strict policy and source/role checks | Revalidate release/inventory and source disclosures at submission |
| Model and harness information | Supplied-field encoder, nominal size/date proxies and explicit missingness | Better transferable representation; no inferred defaults |
| Task demands and difficulty | Full-text lexical encoder and structured observations | Useful transferable difficulty representation; no named-skill claims |
| Demonstrated abilities | Seven whole-family fits and query-excluding fold links | Accepted attribute-conditioned estimates; current control fails quality checks |
| System/task compatibility | Implemented conditional interactions and matched additive ablation | Evidence of useful transfer beyond simpler controls |
| Revealed evidence | Fixed control and item-dependent offset; scope/budget isolation tested | Sampling qualification and complete selection/fallback policy |
| Scientific model selection | Whole-training-family comparison and fail-closed initial quality gate | Unseen-system stress tests, uncertainty and full-protocol coverage |
| Executable predictor | JSON fitted state and portable exact-interface predictor; 42 real fixtures qualified | Final accepted model and runtime configuration; current candidates rejected |
| Packaging and reproducibility | Source/artifact hashes and ZenML/MLflow tracking | ZIP validation, isolated local smoke, runtime/resource audit and technical report |
| Laya integration | Existing Laya routes unchanged | Versioned optional encoder/input contract, with paired qualification before use |

The conditional control is now implemented and rejected by the initial quality
checks. The next substantive step is an outcome-blind data-selection audit
that gives every included scope enough questions for the full acquisition
protocol, followed by complete selection and runtime qualification. Model selection must keep
whole training families out and exclude repeated query tasks across systems.
The independent monitor and original development roles must not become fitting
data. No empirical feature estimate may contain its query's own outcome.

The earlier user stop still prevents automatically restarting the canceled
head/upper-encoder jobs. This resumed goal has used offline source preparation,
fixed statistical diagnostics and small CPU-only conditional control fits. Paid APIs and GPU launches
remain separate from this work. A passing data pipeline or a useful numerical
control is not completion of the requested predictor pipeline.

Validation: 294 unit tests ran successfully with 12 native-environment skips,
all four existing profile catalogs validated and whitespace checks passed.
The new tests cover exact string preservation, rejection of external registries
and sealed roles, source-hash failure, training-target isolation, missing target
hashes, exclusion of undersized evidence scopes and query-label perturbation.
The earlier staged commit checkpoint was not changed or committed by this work.
