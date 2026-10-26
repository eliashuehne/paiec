# Data and feature design before the next predictor

**Scope correction approved by the user:** the population is the full downloaded public release (61 benchmarks, 1,356,713 item rows), with explicit training/validation/test assignments and protected-role dispositions. The earlier bounded 6,252-cell/15,436-task first-pass scope below is superseded. See [the current PR target](../../feature-foundation-pr-target.md) for binding implementation scope. Legacy counts remain historical audit evidence.

2026-10-25. Design and source inventory, incorporating an upstream branch audit.
The current checkout is `catboost` at `6f78ac6`. This document proposes reusable
data contracts and gap-closing work. It does not select or fit the next model.
The existing uncommitted facility-text work is preserved.

## Intent and approach

The user requested understanding the available data, including collected
enrichment, brainstorming useful features, and closing important information
gaps before concentrating on models. They also requested checking other
branches for useful upstream work. The proposed deliverables are:

1. A source inventory with coverage and uncertainty stated at the correct unit.
2. A feature dictionary whose definitions do not depend on a particular predictor.
3. A prioritized gap register with observable completion criteria.
4. A reuse map for existing local and upstream implementations.

Three approaches were considered:

| Approach | What it gives | Decision |
| --- | --- | --- |
| Inventory and mechanism-based feature design first | Preserves facts, exposes missing information and distinguishes absent features from failed models | The approach accepted in the conversation; recommended here |
| Start with the upstream feature-lift loop | Fast feedback about candidate columns under one fixed model | Useful later; its budget-zero proxy cannot define the information we should collect |
| Expand every catalog and annotate every task first | Potentially broad coverage | Defer exhaustive collection; establish semantics, applicability and priority before paying for breadth |

Both interpretable features and learned representations belong in the candidate
inventory. Including embeddings is the author's proposed default; no encoder,
provider, dimensions or compute budget has been selected. Creating a feature
dictionary does not establish predictive usefulness. Later controlled models
must measure that separately.

## Boundaries

- Preserve the eight published subject fields and four item fields, their empty
  strings, and native outcome meanings. Never repair model identity with display
  names or private/scorer IDs. IDs and hashes may join audit tables and define
  exclusions; they are not predictive columns. The anonymous benchmark ID scopes
  permitted evidence.
- Keep research-source eligibility separate from source quality. All external
  enrichment remains research-only until its competition eligibility is resolved.
  Retaining a fact in the inventory does not authorize it in a submitted model.
- Distinguish exact target identities, reported aliases, base-family matches and
  unresolved matches. A matching profile does not verify the evaluated checkpoint.
  A supported tool or documented default does not establish an enabled run setting.
- Use historical versions appropriate to the recorded run. Missing access dates
  remain missing; today's date must not become the run's access date.
- Preserve all existing frozen roles, manifests, renderer sources and recipes.
  Calibration/confirmation and supplemental-development outcomes remain unopened.
  Previous development results are already exposed and cannot become pristine by
  renaming a split or importing another branch's holdout configuration.
- This stage performs no model fitting, embedding inference, annotation API calls,
  new web collection, target-system execution or GPU jobs. Existing Laya stop
  instructions remain in force. Future outcome-derived features must be fitted
  within separately authorized training folds.

## First implementation scope

The following scope is the author's proposed default, chosen to inventory the
broad collection while preserving existing task-role boundaries:

- **System and source accounting:** all 6,252 admitted cells across 41 benchmarks,
  with the 1,936 higher-is-better cells reported as a separate primary stratum.
  Account for every visible subject field and encountered extras key/container
  as typed, retained literal or explicitly unresolved. Do not drop the other
  native directions or combine their target meanings.
- **Enrichment accounting:** all 46 model/base/family profiles, eight harness
  profiles and 189 catalog configuration mappings in shared v2. Keep the
  176-description preparation view separate. Measure matches and per-fact reach
  across all admitted cells, with direction-specific and unique-description
  denominators. Do not expand aliases without a recorded review decision.
- **Initial task feature accounting and any later annotation-quality pilot:**
  the 15,436 distinct training-role visible task inputs referenced by the existing
  offline v2 annotation plan, linked to its 30,000 training source rows. Monitor
  and original-development task inputs are represented here only by existing
  manifest counts; they are excluded from new task inspection/annotation under
  this default. Calibration/confirmation and supplemental-development inputs
  and outcomes remain unopened.
- **Coverage outside that task reference view:** report it as not yet measured.
  The broader pair matrix is not claimed to have complete task features because
  its system metadata was inventoried. A subsequent expansion must declare a
  new role-aware task manifest before loading additional task content.

The first pass is complete when every source/field in these populations is
accounted for, each candidate has its contract and coverage or explicit
unmeasured status, and each unresolved gap has a disposition and next action.
Actual feature builders, annotation production, collection expansion and fitted
features remain proposed work; this document and audit do not implement them.

## Upstream audit and reuse map

`git fetch origin --no-recurse-submodules` completed. No checkout, branch merge,
cherry-pick, dependency installation or source replacement was performed.

| Ref at audit | Relationship to current HEAD | Useful work and proposed reuse |
| --- | --- | --- |
| `origin/main` `b202afd` | Eight commits absent from current HEAD; current HEAD has 21 commits absent from main | Feature-table workflow, grouped folds, source-family evidence, paired reports and join checks |
| `origin/features/lift-model` `e267eda` | Seven absent commits; already merged into upstream main by PR #5 | Same feature-lift work; do not import it twice |
| `origin/logreg` `65a359d` | Six absent commits; not merged into main | Portable sparse encoders, explicit predictor contracts, revealed-tag grouping, separation of item priors and evidence updates |
| `origin/laya` `6043600` | Already an ancestor of current HEAD | Full research collection, typed facts, parsing, annotation/import machinery and training-only support links already available here |
| `origin/baseline` `963f1b9` | Already contained | Existing baseline and benchmark-filter work; no additional import needed |
| `origin/data/loader` `abe7bab` | Already contained | Existing downloader; no additional import needed |
| `origin/eval/harness` `fe90756` | Already contained | Existing local evaluator; no additional import needed |

Local `main` is 13 commits behind fetched `origin/main`. Local `catboost` and
`laya` match their upstream tips. These are Git history counts, not file counts.

Specific reusable components:

| Source | Adopt or adapt later | Qualification before integration |
| --- | --- | --- |
| main: `src/paiec/subject_fields.py` | Strict visible-subject projection and conflicting-record detection | Verify ID uniqueness scope; private subject IDs stay in the audit join, never the predictor |
| main: `lift.join_features` | Independent item/system tables; duplicate-key and column-collision checks | Default to composite `(benchmark, item_id)` source keys and visible-subject hashes; require evidence before treating source IDs as global |
| main: `configs/lift/benchmark_groups.json`, `folds.py`, family scanner | Eleven evidence-described source/item-overlap groups and group-preserving split construction | Reconcile with frozen roles and actual admitted scope; do not replace historical folds or read holdout outcomes |
| main: `lift.in_fold_ability` | Leave-own-family-out construction and support counts | Outcome-derived feature, not static metadata; fit only inside permitted training partitions |
| main: `lift_report.py` | Paired comparisons, feature coverage, fingerprint warnings, source/run hashes | Fingerprint is a diagnostic, not a ban; shuffled-label invariance is not proof of absence of leakage |
| main: `scripts/lift/`, `tests/lift/` | Fixed experiment interface and synthetic guard tests | The loop is budget zero, row-weighted with a benchmark-macro companion; these are not pair-equal Brier ALC. New tests use pytest and need explicit inclusion |
| logreg: `features.py`, `contracts.py` | Shared offline/runtime encoding, missingness flags, separate item/system vectors and an explicit callable contract | Review truncation, hash collisions, unknown-effort encoding, option heuristics, date interpretation and parameter parsing; the first size-like token must not certify MoE or checkpoint size |
| logreg: `groups.py` | Candidate within-benchmark tag groups and support counts derived from visible evidence | Its semicolon parser loses quoted/nested cases; reconcile with existing robust parsing. Prefixes/hash filters do not prove a tag is a transferable skill |
| logreg: `prior_offset.py`, `update.py` | Design reference for separating an item prior from item-conditioned revealed-label residuals and uncertainty | This is fitted model machinery; defer adoption until feature contracts and evidence semantics are settled |

Avoid a wholesale source-tree copy. Upstream main's README and project metadata
describe a smaller project; this checkout has additional dependencies, frozen
contracts and reproducibility documentation. A future normal merge should
resolve those explicitly and preserve the richer local contracts. Import
logreg components through reviewed adapters rather than replacing the current
evaluator, predictor registry or episode builder: that branch's episodes reveal
the first recorded trial per item and its existing evidence handling needs
reconciliation with the published shared-evidence protocol.

Source links: [main feature guide](https://github.com/eliashuehne/paiec/blob/b202afd/docs/lift.md),
[family config](https://github.com/eliashuehne/paiec/blob/b202afd/configs/lift/benchmark_groups.json),
[logreg feature encoder](https://github.com/eliashuehne/paiec/blob/65a359d/src/paiec/features.py),
[logreg evidence update](https://github.com/eliashuehne/paiec/blob/65a359d/src/paiec/update.py).

## Source inventory and measured gaps

The scope of a coverage number is part of its definition. Distinguish response
trials, source system/configuration records, distinct visible descriptions,
subject–benchmark cells, distinct model/family profiles and unique tasks.

| Asset | What exists | Limit |
| --- | --- | --- |
| Public pair matrix | 41 admitted benchmarks, 6,252 cells; 1,936 higher-is-better cells across 37 benchmarks | Other native target directions remain retained with their meaning; success-oriented feature work must declare its outcome scope |
| Shared enrichment v2 | 46 model/base/family profiles, eight harness profiles, 189 catalog configurations, 176 preparation descriptions | Profile/configuration counts are not broad-corpus coverage or exact-checkpoint verification |
| Scoped source review | 617 scoped fields accounted for, 205 confirmed known values and eight reviewed conflicts | Zero independent reviews; reviewed unknowns are still unknown. Other documented/catalog-only fields remain separately qualified |
| Recorded-run preparation | 149 populated run-setting facts across the 176 descriptions | Does not establish actual enabled tools, temperatures or limits for every run |
| Full-input annotation plan | 16,909 visible task inputs: 15,436 training, 583 monitor, 890 original development | Fourteen candidate annotations; zero independently reviewed; no teacher calls. This is a plan/import pipeline, not a completed semantic feature dataset |
| Training-only support/query links | 30,000 training rows, 15,383 normalized task groups, 170 visible scopes, 850 support pools | Indices only; no empirical abilities, retrieved neighbors or evidence correction fitted |

Existing guides: [shared collection](../../shared-data-start.md),
[task annotation](../../task-annotation-collection.md),
[evidence links](../../feature-evidence-links.md).

### Coverage in the broader matrix

A source-only audit used existing pair-matrix subject dictionaries, the exact
`compact_profiles_v3.json` resolver and the shared export. It did not use outcome
values or create new aliases. Counts below cover **all 1,936 higher-is-better
cells**, rather than the sampled 1,916 evaluation cells:

| Supplied or matched information | Cells | Interpretation |
| --- | ---: | --- |
| Nonempty canonical name | 423 | 21.8%; extras may contain other reported identifiers |
| Nonempty harness | 1,908 | 98.6%; name alone does not establish configuration |
| Nonempty subject extras | 1,903 | 98.3%; a primary parsing opportunity |
| Nonempty harness version | 418 | 21.6%; unversioned behavior stays uncertain |
| Nonempty dedicated reasoning effort | 9 | 0.5%; extras may record other reasoning settings |
| Model profile matched | 225 | 11.6%; mixes reviewed identity bases, not exact checkpoints |
| Harness profile matched | 204 | 10.5% |
| Both profiles matched | 170 | 8.8% |
| Exact preparation-description match | 187 | Existing preparation can be reused directly for this subset |

There are 1,918 distinct visible descriptions in this scope. Across all 6,252
admitted cells, the same policy matches 226 model profiles and 204 harness
profiles. The large unordered block makes all-cell percentages especially
misleading for the success-oriented scope.

The shared export contains useful supplemented facts absent from the older
catalog view under that same policy:

| Candidate fact | Raw catalog cells with known facts | Shared-export cells with known facts |
| --- | ---: | ---: |
| Documented model language coverage | 0 | 88 |
| Model specialization/tuning domains | 22 | 56 |
| Harness context management | 0 | 128 |
| Documented total parameters | 74 | 96 |

These are presence counts of recorded/documented values, with base-only matches
restricted to architecture. They are not automatic approvals: version,
evidence, review disposition, matching basis and competition eligibility must
still be checked. Counts do not establish that existing CatBoost profile
features expose these fields.

Exact shared-run matches provide known shot counts for 131 cells, output limits
for 25 and selection methods for 11. They provide no known enabled-tool list,
temperature, separate reasoning budget or turn/tool-call limit in this scope.
That is a gap in the matched export, not proof that no such settings exist in
the original extras.

Frequent raw keys include `source_model_label` (970 cells),
`source_model_identifier` (418), `fine_tuned` (417), and training language,
partition, size and quality fields. Their semantics need review: a reported
source/base name must not silently become the exact fine-tuned target model.

Audit receipt:
`.planning/2026-10-25-feature-inventory-and-upstream-audit/source_inventory.json`.
It records cell, policy and shared-export file hashes and the fetched Git refs.
It is a local research receipt, not a new frozen dataset or experiment recipe.

## Feature contracts

Keep raw facts reusable and independent of encoding choices. Five related
tables are proposed:

1. **Visible systems:** exact permitted field strings and an audit hash, with
   separate identity/matching records. Preserve indistinguishable descriptions.
2. **Model/harness facts:** typed value plus status, source, applicable version,
   evidence, review disposition and identity relation. Preserve conflicts.
3. **Tasks:** complete visible fields, source keys, normalized-content grouping,
   observable structure and separately versioned semantic annotations.
4. **Derived compatibility:** deterministic relationships between known system/run
   facts and reviewed task requirements, retaining both operands and uncertainty.
5. **Evidence-derived state:** budget-bound observed counts, similarities and
   residual summaries, recomputed from the supplied evidence at runtime.

Every feature dictionary entry must declare: stable name/version, unit and
type, level (model/run/task/interaction/evidence), definition, proposed
mechanism, source fields, availability at inference, unknown/conflict behavior,
applicability, source eligibility, fit scope if learned, coverage denominators,
extractor quality checks and an implementation status. A source ID, rationale
or content hash is audit metadata unless explicitly approved as a legitimate
input-derived feature.

Unknown scalar values remain null with a reason until a particular model's
adapter chooses imputation. Empty documented lists differ from unknown lists.
Keep per-fact missingness, not just a single profile-found flag. Learned
vocabularies, normalization, projection and label-based estimates belong inside
the appropriate training fold; deterministic raw extraction can be shared.

## Candidate feature dictionary

The following are candidates, not claimed gains. Existing observations should
be extracted before adding speculative judgments. Names are proposed and may
be versioned during implementation; an implementer can choose storage formats.

### Model and identity

| Candidate | Why it might matter | Existing source and remaining work |
| --- | --- | --- |
| `identity_relation`, `identity_conflict`, per-field missingness | Controls which facts apply and whether transfer is justified | Conservative resolver and review ledgers exist; unify their output instead of one found/not-found bit |
| `model_family`, `base_relation`, `adaptation_type` | Supports related-system transfer without pretending two checkpoints are identical | Catalog lineage plus supplied fine-tuning metadata; separate reported target from reported base |
| `total_parameters`, `active_parameters`, `parameterization` | Coarse capacity/resource descriptors | Typed facts available for a subset; name-derived sizes remain separate proxies with parse status |
| `documented_specializations`, `documented_languages` | Task-specific compatibility hypotheses | Full shared export has more than the renderer catalog; absence of a language/domain claim means unknown coverage |
| `input_modalities`, `output_modalities` | Potential feasibility mismatch | Model facts plus task requirements; text-only admission does not prove every task is self-contained |
| `input_limit`, `output_limit`, `combined_context_limit` | Potential resource constraints | Keep documented capacity separate from actual run limits and from any task-token approximation |
| `release_date`, `recorded_access_date`, validity status | Historical applicability and possible distribution shift | Dates are sparse; release date does not prove a knowledge cutoff or current model revision |

Architecture depth, attention/KV heads, experts and hidden dimensions remain
available secondary candidates. Collecting every architecture field is a lower
priority than exposing recorded settings and functional compatibility.

### Harness and actual run

| Candidate | Why it might matter | Existing source and remaining work |
| --- | --- | --- |
| `harness_identity`, `version_known`, `documented_workflow` | Distinguishes fixed configurations and plausible execution strategies | Supplied fields and harness profiles; version applicability often unknown |
| `supported_tools` and separate `enabled_tools` | Execution, search or retrieval can change task feasibility | Never fill enabled tools from support; exact run facts are sparse |
| `few_shot_count`, `prompting_method`, `selection_method` | Demonstrations and prompting alter performance | Reuse typed run review and settings parsing; retain raw values and parse provenance |
| `output_token_budget`, `reasoning_budget`, `max_turns`, `max_tool_calls`, `max_retries`, `candidate_count` | Resources and verification opportunities differ | Keep units and scope; unknown is not zero or a provider default |
| `temperature`, `top_p`, `serving_configuration`, `quantization` | Sampling and implementation can affect response probabilities | Mostly unresolved; prefer explicit supplied values over inferred defaults |
| `context_management`, `verification_available` | Long-task retention and error checking mechanisms | Documented capability versus verified run usage must stay distinct |
| `extra_parse_status`, `unknown_setting_keys`, `unparsed_fraction` | Measures information loss and parser reach | Current structured CatBoost extras expose counts/length rather than many values; retain unfamiliar containers |

Reuse quote-aware settings parsing and the existing URL/JSON decoding rather
than the simpler upstream semicolon splitter. Keep method labels literal unless
their meaning is established; digits in a method name are not a shot count,
sample count or reasoning budget.

### Task requirements and observables

| Candidate | Why it might matter | Existing source and remaining work |
| --- | --- | --- |
| Full/section character lengths, token estimate and estimation method | Context/output pressure and content visibility | Existing parsers/counts; compute full lengths and record every truncation separately |
| Language/script, mixed-language status, translation demand | Language coverage and task interpretation | Script heuristics exist; language identity and translation requirements need reviewed annotations |
| Domain and required skills | Supports conditional ability transfer | Existing eight-facet taxonomy and quote-grounded import contract; quality/coverage incomplete |
| Answer format, options, output constraints, format conflicts | Different failure modes for MCQ, code, structured output and free text | Parse visible options and constraints; `1 / option_count` is only a heuristic, never a guaranteed success floor |
| Code/SQL/math structure, schema/table/operand counts | Describes concrete task demands | Surface features exist; stronger parsers need fallback and errors reported, not guessed syntax validity |
| Context structure, message roles, evidence locations, cross-reference cues | Long-context and multi-document demands | Existing task parser; missing supplied context does not prove the target lacked it |
| Information needs, temporal dependence, interaction/tool requirements | Identifies dependencies on knowledge, retrieval or execution | Use supplied spans; distinguish required, helpful and unknown tools |
| Ambiguity, conflicting instructions, visible missing assets, malformed input | Captures uncertainty about what the task asks | Existing input-issues facet; do not generate corrected questions or answers |
| Constraint count and stated reasoning structure | Potential demand on sequential reasoning and instruction following | Count observable requirements; avoid solving the task to invent a true step count or difficulty rating |

Task annotations must not contain reference answers, hidden grading rules,
generated solutions, numeric difficulty ratings or assessments of the target's
strength. Exact quotation verifies grounding location, not annotation accuracy.
Retain unknown, abstention and disagreement explicitly.

### System–task compatibility

| Candidate | Definition and mechanism | Conditions |
| --- | --- | --- |
| `visible_context_to_capacity` | Visible input tokens divided by known applicable capacity | Need the target tokenizer or explicitly labeled approximation; hidden prompts and retrieved context make this a partial request measure, not a certified overflow |
| `output_budget_vs_format_demand` | Recorded output budget related to observable requested output requirements | Do not invent exact answer length or chain-of-thought requirements |
| `required_tool_enabled`, `required_tool_supported` | Separate task/tool relationships | Tri-state known yes/no/unknown; supported does not imply enabled and unknown does not imply unavailable |
| `language_documented_match`, `domain_specialization_match` | Reviewed task facet related to applicable documented coverage | Keep inferred relation separate from fact; an undocumented language is not a known mismatch |
| `interaction_budget_match`, `verification_opportunity` | Required interaction/validation related to known run resources | Most operands are sparse; feature remains unknown when a necessary operand is unknown |
| `training_language_or_domain_relation` | Supplied adaptation language/domain related to task requirements | Source training descriptions are reported properties, not measured ability or permission to use external scores |

Start with explicit compatibility relationships rather than every possible
pairwise feature cross. Nothing in a failed architecture-heavy profile model
settles these unmeasured interactions.

### Text representations and behavioral evidence

| Candidate | Role | Required safeguards |
| --- | --- | --- |
| Word/character counts, hashes and TF-IDF | Lexical structure and content reference representations | Use separate item/system namespaces; fit vocabularies/IDF in training only; keep Unicode and full-length/truncation metadata |
| Frozen semantic embeddings of task sections | Candidate transfer representation for meaning and requirements | Pin encoder/version/tokenizer/pooling, section coverage and runtime availability; separately authorize model download/inference and verify source eligibility |
| Embeddings of model/harness descriptions | Candidate representation of sourced functional facts | Preserve explicit numerics outside the embedding; encode uncertainty and exclude source URLs, marketing strength ratings and unsupported descriptions |
| Training-only demonstrated capability by skill/domain | Behavioral evidence beyond descriptions | Separate feature family requiring fitting; exclude the query family/tasks, preserve native target meaning and attach support/system/task counts |
| Own/sibling revealed successes and counts | Runtime level adaptation | Only supplied budget-permitted examples; retain trial semantics, distinguish exact same visible system from siblings |
| Target-support text/facet/tag similarity and coverage | Allows nearby tasks to inform unobserved items | No evaluation labels; similarity extractor fixed or trained without target outcomes; report support count and similarity spread |
| Revealed residual summaries | Compare acquired outcomes to the prior's predictions on those acquired items | Outcome-derived state; requires a specified, separately validated prior and update. A raw mean alone can confound acquisition difficulty with ability |
| Evidence uncertainty and provenance | Controls trust in sparse or heterogeneous observations | Record budget, native target compatibility, matching ambiguity, effective support and conflicting evidence |

These families stay distinct. Static sourced facts, deterministic input
measurements, learned representations and outcome-derived estimates must not
all be described as label-free feature engineering.

## Gap register and order of work

Priorities below are design judgments, not feature-importance results. Completion
means a usable, audited contract; it does not require making every unknown known.

| Priority | Gap | Reuse | Completion criterion |
| --- | --- | --- | --- |
| P0 | Multiple source scopes and feature contracts | Pair matrix, full export, upstream feature-table joins | One inventory binds revisions/hashes and distinguishes cells, visible systems, profiles, tasks and trials; key conflicts fail explicitly |
| P0 | Broad-corpus enrichment reach and applicability | Existing resolver, scoped alias/review ledgers | Coverage reported by cell, unique description, benchmark and feature; unmatched reasons and exact/base/family relation retained. Counts above are the starting audit, not coverage expansion |
| P0 | Settings present as opaque extras | Existing decoder, typed run records, logreg key/value feature idea | All encountered keys/containers accounted for as typed, literal or unresolved; examples reviewed for quotes, nested JSON, URL encoding, conflicts and units |
| P0 | Full collection not exposed by older feature projections | Shared v2 combined facts and provenance | An export manifest lists selected facts, review disposition and applicability; reviewed unknowns take precedence and source IDs resolve |
| P0 | Evaluation and export parity uncertainty | Current scorer, upstream grouped folds and shared encoding | Reconcile sampling cap, nested response budgets, visible-system ambiguity, common evidence and offline/runtime feature parity before any predictive claim; preserve historical results |
| P1 | Task taxonomy quality and coverage | Eight-facet annotation contract and 14 candidates | Before bulk production, define and independently review an outcome-blind training-only sample across relevant languages, formats, lengths and domains; record errors, abstention and facet agreement. Set acceptance thresholds before scaling |
| P1 | Missing compatibility features | Typed system/run facts plus reviewed task requirements | Each interaction has explicit operands, tri-state/unknown behavior, applicability checks and reviewed examples |
| P1 | Cross-benchmark task/source overlap | Upstream 11-group evidence config, scanner and local task grouping | Versioned group/overlap manifest audited against frozen roles; no reopening, moving or relabeling existing sealed data |
| P1 | Collection expansion lacks a value-of-information order | Coverage/unresolved reports | Rank missing facts by affected systems/tasks, plausible mechanism, resolvability, source eligibility and collection cost; close high-value gaps or document justified abstention |
| P2 | Learned representation choice | Existing TF-IDF and upstream portable hashing | Design a matched comparison with identical visible inputs and documented truncation; encoder inference remains a later authorized step |
| P2 | Learned abilities/evidence-state features | Local support links and upstream tag/offset separation | Specify exclusions, runtime evidence and counts without materializing or fitting under this design pass |

Work should initially concentrate on information already present: parsing actual
settings, exposing supplemented functional facts, aligning keys and applicability,
and validating task requirements. New architecture research is lower priority
while these extraction gaps remain. Some missing identities and run settings
will remain irreducible at the supplied interface.

## Validation and the later handoff to models

Before evaluating a predictor, verify:

- Same visible input and feature version produce the same raw features offline
  and at runtime. Parsing never introduces unavailable fields or hidden answers.
- Feature builders preserve raw values and report malformed input, conflicts,
  unknowns and truncation. Synthetic cases cover these behaviors; source review
  checks meaningful real examples without reading target outcomes.
- Joins do not multiply records or equate different identity scopes. All trials
  remain accounted for as trials, not additional tasks or independent systems.
- Model/harness facts carry review/source applicability; base properties do not
  become exact fine-tuned-target properties. Enriched and supplied-only views
  have independently auditable eligibility policies.
- Static output is invariant to changes in outcome columns, supplemented by
  dependency/provenance review; a permutation test alone cannot prove safety.
- Estimated features have explicit outer/inner fit boundaries and task/family
  exclusions. Runtime state uses only supplied, nested budget-permitted evidence.
- Upstream holdouts are reconciled against this project's exposure history. A
  holdout sealed on another branch may already have been analyzed here.

Only after the above design is approved and the necessary implementation is
authorized should simple reference models assess feature blocks. Later tests
should use paired comparisons, uncertainty at benchmark/family/system levels,
scope-appropriate weighting, all relevant budgets and coverage-stratified
results. Weak results under one model should update a feature hypothesis rather
than erase the raw collection or establish an interface-wide limit.

The builder may choose table serialization, module names and parser adapter
details while preserving these contracts. Annotation provider, encoder model,
bulk annotation volume, external collection and predictive experiments remain
future decisions. No new scientific recipe or training authorization is created
by this document.
