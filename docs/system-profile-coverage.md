# Expanded system-knowledge coverage

Subsequent work fixes delivery and adds reviewed harness-scoped ERBench family/
base-family context. See [compact inputs and development](compact-inputs-and-development.md).
The collection-v2 counts below remain a historical comparison.

The user selected all **178 training/development candidates**: 170 training
configurations plus eight development configurations. This collection uses
only their published subject fields. Calibration and confirmation remain sealed;
the frozen training sample has not been expanded automatically.

The new catalog is
[`research/system_profiles/collection_v2`](../research/system_profiles/collection_v2/).
Its fact structure is identical to collection_v1. There are 44 model/artifact/
family profiles, eight harness profiles and 178 recorded-run profiles. The
collection contains 240 documented specification facts, six explicitly inferred
facts and 273 recorded settings. Inferences are excluded from rendered input.
Every known fact includes source evidence; missing specifications remain null.

## What has actually reached 100%

All 178 systems have **some sourced model or harness information available**.
Full model-and-harness coverage is still incomplete:

| Measure, across the same 178 systems | Previous catalog and policy | New catalog and policy |
| --- | ---: | ---: |
| Model facts available | 44 (24.7%) | 169 (94.9%) |
| Harness facts available | 171 (96.1%) | 171 (96.1%) |
| Both available | 41 (23.0%) | 162 (91.0%) |
| At least one available | 174 (97.8%) | 178 (100%) |

The remaining gaps are three missing development model identities, four generic
study aliases/private deployments, two unverified private CDG checkpoints and
seven missing training harness names. There is a complete
[resolution queue](../research/system_profiles/collection_v2/unresolved.jsonl)
with supplied fields, investigation notes and the evidence needed for each gap.
Inventory entries and unresolved candidates do not count as model/harness facts.

Some family information can be used safely even when a variant is ambiguous.
MetaMath family profiles describe mathematical fine-tuning without assigning a
guessed size or candidate configuration. Llama 3 8B family facts do not assert
base versus instruction tuning. Bedrock family facts do not claim service/HF
checkpoint equivalence. The 169 matches include 34 family-only matches and 135
reported artifact/family matches. **None independently verifies evaluated weights.**

Generic ERBench aliases remain unbound by the global resolver. The pinned study
code maps `gemini` to `gemini-pro`, so a future harness-scoped resolver could use
that family evidence. Declaring a global `gemini` alias would incorrectly assign
the same version to other contexts. Other study aliases still hide private
deployment or fine-tuning details. Missing C-Eval names cannot be recovered by
promoting `display_name` into the published predictor fields.

## Collected information and identity controls

New primary sources cover the additional Qwen, DeepSeek, Claude, Gemini, DBRX,
SQLCoder, Mistral/Mixtral, WizardCoder, Code Llama, Llama 2 and historical OpenAI
families present in this scope. Official provider sources replace unavailable
configuration downloads where possible. Two fetched Code Llama mirror snapshots
are explicitly `not_permitted` and contribute no facts; the official Meta card
is used instead. No subjective capability rankings or benchmark scores were
added as features.

The 86 source records comprise 74 external sources marked `unreviewed`, ten
hashed original subject tables and the two excluded mirror snapshots. Seventy-six
external snapshots are content-hashed. Published parameter counts are rounded;
proprietary undisclosed sizes remain unknown. Configuration position limits are
not treated as documented runtime capacities. Preview/stable versions and
Mistral v0.1/v0.2 stay separate. Literal run settings are stored separately from
model capacities and harness defaults.

The new policy is
[`enrichment_catalog_v2.json`](../configs/head_finetuning/enrichment_catalog_v2.json).
It works with the existing `enriched_v1` renderer and `task_enriched_v2` adapter;
the filename versions the catalog selection, not the renderer implementation.
It prioritizes published size, parameterization, model type and modalities before
deeper dimensions. It keeps the 96-token ceiling and complete-fact/source guards.
No default policy, original renderer, old catalog, prepared bundle or trained
checkpoint was changed. This new policy requires a new preparation bundle and
a checkpoint trained under that bundle before using it as a trained-head input.

## Available facts versus facts Laya sees

A tokenizer-only audit inspected the existing 5,000 training and 963 development
input states with `TaskStateBuilder`, the pinned English tokenizer, a 1,024-token
limit and this new policy. It did not run Laya or train weights.

| Role | Records | Model facts available / rendered | Harness facts available / rendered | Any fact rendered |
| --- | ---: | ---: | ---: | ---: |
| Training | 5,000 | 4,584 / 3,185 | 3,958 / 2,304 | 3,599 |
| Development | 963 | 120 / 120 | 963 / 963 | 963 |

Thus broader research coverage alone does not produce 100% model context in
training. Long inputs leave too little room for a complete metadata heading and
fact line. The existing renderer preserves target content and cannot force facts
into that space. The existing task-v2 input design also still omits task content
in 2,040 training records. Changing those budgets would be a separate input
experiment requiring new preparation and matched controls.

## Reproduction and checks

The collection's [coverage report](../research/system_profiles/collection_v2/coverage.md)
has an architecture summary, source links and per-gap details. The local
interactive-browser report is
`runs/head_finetuning/system_profile_coverage_v2_final/report.html`; durable
metrics are in
[`coverage_summary.json`](../research/system_profiles/collection_v2/coverage_summary.json).

```bash
.venv-laya-mlx/bin/python scripts/build_system_profile_collection.py --catalog research/system_profiles/collection_v2
.venv-laya-mlx/bin/python scripts/validate_system_profiles.py --catalog research/system_profiles/collection_v2
.venv-laya-mlx/bin/python scripts/audit_system_profile_coverage.py --pilot --out runs/NEW_COVERAGE_AUDIT
```

The audit verifies exact scope, frozen roles, inventory hashes and all eight
subject fields against original hashed subject tables. `--require-full` produces
the report and then exits with status 2 while any model/harness gap remains.
It reads no response or trace tables. Training-role candidates remain candidates;
the scope request did not admit them to a new training sample.

Offline rebuild/schema validation pass. Both lightweight suites pass: 225 tests,
with nine expected skips in MLX and one in PyTorch. The earlier enriched and
task-v2 prepared bundles and completed best-head input contract still validate.
No new training, model inference, scoring, paid API calls or GPU work occurred.
Competition eligibility of external research remains unreviewed.
