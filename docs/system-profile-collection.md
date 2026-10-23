# Primary-source system profiles: first collection

Subsequent work: the user authorized preparation of an opt-in enriched head
experiment. See [enriched-head-pilot.md](enriched-head-pilot.md). No new training
has started; the collection-stage statements below describe the earlier stage.

Collected 2026-10-22 at the user's request. This is actual web research, following
the earlier schema-only seed work. It adds research data and collection tooling;
Laya inputs, frozen training data, model weights and scores are unchanged.

## Scope and files

The first pass covers **all 32 exported training/development systems** in
`runs/head_finetuning/data_v2/provenance.parquet`: 24 training systems across
eight benchmarks, and eight development systems across two benchmarks. It does
not cover the entire public corpus. No sealed validation/test subjects were
used to choose profiles, and no item outcomes or response traces were read.

The saved [catalog](../research/system_profiles/collection_v1/) contains:

| File | Contents |
| --- | --- |
| `profiles.jsonl` | 18 documented model/artifact profiles, 8 harness profiles, 32 recorded systems; identical fact/evidence structure |
| `sources.jsonl` | 52 sources: 42 external metadata snapshots and 10 pinned original subject tables |
| `subjects.jsonl` | Each selected subject's eight published fields, split role, model/harness links, match status, candidate links and reasons |
| `assertions.jsonl` | Reviewed document assertions and recorded subject settings, each with a citation and excerpt |
| `recipe.json` | Profile declarations and configuration sources for reproducible extraction |
| `snapshots.json` | Snapshot filenames and content hashes |
| `fetch_log.json` | Collection attempts, including gated/missing sources and superseded preliminary fetches |
| `coverage.md` | Readable model architecture summary and per-subject identity decisions |

There are **121 documented specification facts**, five explicitly inferred
specification facts, and 20 recorded run settings. Identity facts are counted
separately. Every unestablished field remains `null` with `status: unknown`.
These are counts of metadata fields, not evidence of improved prediction.

## What the collection adds

Published configurations provide layer counts, hidden dimensions, attention/KV
heads and routed-expert counts. Qwen's configuration nests these in
`text_config`; the extractor uses that component instead of confusing the
vision and text networks. Its model card describes an alternating Gated DeltaNet
and Gated Attention layout; the `hybrid` backbone classification is explicitly
marked as an inference.

Examples of the collected architecture information include Mistral's grouped
query/sliding window attention, DeepSeek's MoE sizes and active parameter
counts, Llama 4 Maverick's multimodal MoE architecture, and Qwen's mixture of
linear and full attention layers. Published parameter counts are rounded; the
Qwen 35B/3B numbers are scoped to the card's language-model component. Configured
expert counts exclude shared experts where the config counts routed experts.

Provider documentation supplies modalities, output limits and reasoning
settings where published. Proprietary parameter counts and internal architecture
remain unknown. Harness documentation supplies supported workflows; the catalog
does not claim those workflows or tools were enabled in a particular run.

Original subject metadata separately records settings such as an 8,000-token
output budget, zero/five-shot prompting, minimum-perplexity selection, and
serving provider. Unrecognized extra settings remain in the raw supplied fields;
they are not discarded or guessed into a different meaning.

## Important identity and version findings

**19 of 32 subjects have a reported identifier matching a collected model or
family; none has an independently verified evaluated weight/API snapshot.**
Thirteen remain unresolved: seven missing/insufficient model identities, three
ambiguous variants, two private CDG checkpoints, and one historical API version.
Five subjects have no published harness name. Candidate links are separate from
resolved links, and no lookup runs automatically during prediction.

- MetaMath-Mistral and MetaMath-Llama lack size/checkpoint identifiers. The
  `meta-math/MetaMath-7B-V1.0` card currently describes Llemma despite its repository
  name and Llama config. It remains an unresolved candidate, with no run binding.
- CDG's public model card identifies a local base model and a fine-tuned model,
  but equivalence to the recorded `llama-CDG2` checkpoint is unverified. Base and
  adapted models are kept separate.
- Three C-Eval subjects have empty published names. Their audit display names
  are not used to manufacture model identity.
- ERBench's original code maps `gemini` to `gemini-pro`, not Gemini 1.5 Flash.
  Other aliases use private deployments or placeholders. Exact snapshot and
  fine-tuned checkpoint identities remain unresolved.
- The historical `gpt-3.5-turbo-16k` subject is not assigned today's GPT-3.5
  default specs. Current GPT-4o/mini documentation defaults are stored with notes
  that they may differ from the evaluated API version.
- Some official Meta configuration downloads require authorization (HTTP 401).
  Those layer/head counts remain unknown. A CodeLlama mirror explicitly says it
  is unofficial; its config is excluded from catalog assertions. The original
  Meta model card is used instead.

## Context windows and workflow limits

`max_position_embeddings` is deliberately **not** copied into supported context
limits. For example, DeepSeek-R1's config specifies 163,840 positions, while its
card advertises 128K context. The latter shorthand is retained in cited source
material, rather than turned into a guessed exact token ceiling. Similarly,
advertised 1M/200K labels are not silently interpreted as exact token integers.
Qwen's explicitly stated native 262,144-token context is stored; the optional
1,010,000-token extension is not assumed enabled.

Current OpenAI model pages state exact numerical limits and default snapshots;
the facts are qualified as current documentation, not verified historical run
limits. Recorded per-run `max_tokens` values stay separate from model output
capacity. Harness examples/defaults do not establish actual tool availability,
retry budgets, candidate counts or reasoning settings.

## Evidence and reproduction

Each fact references a source URL plus a document passage or JSON key. GitHub
and Hugging Face sources are pinned to commits; mutable provider pages retain
retrieval date and SHA-256. The full downloaded evidence archive is ignored by
Git at `runs/system_profile_sources_20261022/`. It can contain unrelated benchmark
tables; those are neither extracted as facts nor included in model inputs.

Validate the saved catalog without network access or source downloads:

```bash
.venv-laya-mlx/bin/python scripts/validate_system_profiles.py --catalog research/system_profiles/collection_v1
```

Rebuild from the existing local source archive:

```bash
.venv-laya-mlx/bin/python scripts/build_system_profile_collection.py
```

To restore missing snapshots, run `scripts/fetch_system_profile_sources.py`
with the same Python environment. It downloads metadata only and checks saved
hashes before writing. If a mutable provider page has changed, it reports failure
instead of silently replacing evidence; the original local snapshot is needed
to reproduce that extraction exactly. The builder checks all snapshot hashes,
quoted passages, JSON values, recorded-subject excerpts, schema types and links.
It never imports or executes downloaded source code.

All external sources remain `eligibility: unreviewed`; every profile is
`research_only`. Research collection is authorized, but competition eligibility
and usefulness as training features have not been established. No training,
inference, cloud GPU work or Brier evaluation was run for this collection.

Verification: the offline rebuild and catalog validation pass. The MLX
environment's full lightweight suite passes: 182 tests, three expected skips.
See local `runs/profile_collection_tests.log`. The original three seed profiles
remain preserved and independently valid.
