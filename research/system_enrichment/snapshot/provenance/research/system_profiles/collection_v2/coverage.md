# Expanded collection: 178 systems

Collected 2026-10-22 for all 170 training candidates and eight selected development systems. Calibration and confirmation remain sealed. These are system configurations, not 178 independent models.

**Full model-and-harness coverage is not achieved.** Every system has at least one sourced model or harness fact available. Availability does not mean every field is known, the evaluated checkpoint is verified, or a fact fits in the input.

| Coverage over the same 178 systems | Previous catalog/policy | Expanded catalog/policy |
| --- | ---: | ---: |
| Systems with model facts | 44 | 169 |
| Systems with harness facts | 171 | 171 |
| Systems with both | 41 | 162 |
| Systems with at least one | 174 | 178 |
| Independently verified evaluated checkpoints | 0 | 0 |

169/178 (94.9%) have model facts; 171/178 (96.1%) have harness facts. Only 162/178 (91.0%) have both. The 169 model matches include 34 deliberately limited family matches and 135 reported artifact/family matches. None proves exact evaluated weights.

## What blocks the requested 100%

- Three development systems have no model identity in the permitted subject fields. Audit display names cannot replace those fields.
- Four development systems use generic study aliases or private deployments. The pinned study code maps `gemini` to `gemini-pro`, but the current global alias resolver cannot safely assign that meaning outside its harness. Other aliases do not establish the deployed snapshot or fine-tuning.
- Two training systems reference private CDG checkpoints whose equivalence to the published model is unverified.
- Seven training systems have no supplied harness name. A benchmark-to-harness guess is not a permitted join.

The complete gap list is [unresolved.jsonl](unresolved.jsonl). It records the supplied fields, the investigation and the evidence needed to close each gap. This queue is not counted as researched specification coverage.

## Actual input visibility

A separate tokenizer-only check uses the existing frozen 5,000 training / 963 development records with `TaskStateBuilder` at 1,024 tokens and this new 96-token enrichment policy. No inference, fitting, scored outcomes or new training-data admission.

| Role | Records | Model facts available | Model facts rendered | Harness facts available | Harness facts rendered | Any fact rendered |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| training | 5000 | 4584 | 3185 | 3958 | 2304 | 3599 |
| development | 963 | 120 | 120 | 963 | 963 | 963 |

Long-input visibility still limits enrichment: complete fact lines need room for their section heading and contents. No target text is displaced to force metadata into the input. There are still 2,040 task omissions in the existing task-v2 training inputs. Zero native overflow does not mean complete task retention.

## Published model facts

The table describes documented artifacts or shared family properties. Unknowns are preserved as null. Rounded size labels are not exact tensor counts; Qwen3.5/3.6 sizes refer to the language component. Candidate-only profiles are stored separately from actual joins.

| Profile | Published size / active | Layers | Family/type or modality fact | Source |
| --- | --- | ---: | --- | --- |
| model.metamath-mistral | unknown / unknown | 32 | fine-tuned | [primary source](https://huggingface.co/meta-math/MetaMath-Mistral-7B/resolve/c4a2bb751735e8fb18a20e0c18619db7fe614e7b/README.md) |
| model.metamath-llama-candidate | unknown / unknown | 32 | llama | [primary source](https://huggingface.co/meta-math/MetaMath-7B-V1.0/resolve/e634cdc03e2e32e1ad35ea1aa2418090221a0951/config.json) |
| model.mistral-7b-instruct-v0.1 | unknown / unknown | 32 | mistral | [primary source](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1/resolve/ec5deb64f2c6e6fa90c1abf74a91d5c93a9669ca/config.json) |
| model.sqlcoder-70b-alpha | unknown / unknown | 80 | llama | [primary source](https://huggingface.co/defog/sqlcoder-70b-alpha/resolve/8095a5ded3d3ed70fe27e9ed46b91f3e29a1b51a/config.json) |
| model.deepseek-r1 | 671B / 37B | 61 | deepseek_v3 | [primary source](https://huggingface.co/deepseek-ai/DeepSeek-R1/resolve/56d4cbbb4d29f4355bab4b9a39ccb717a14ad5ad/config.json) |
| model.deepseek-v4-flash | 284B / 13B | 43 | deepseek_v4 | [primary source](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/resolve/60d8d70770c6776ff598c94bb586a859a38244f1/config.json) |
| model.qwen3.6-35b-a3b | 35B / 3B | 40 | qwen3_5_moe_text | [primary source](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/resolve/995ad96eacd98c81ed38be0c5b274b04031597b0/config.json) |
| model.qwen-7b | 7B / unknown | 32 | base pretrained | [primary source](https://huggingface.co/Qwen/Qwen-7B/resolve/ef3c5c9c57b252f3149c1408daf4d649ec8b6c85/README.md) |
| model.llama3.1-8b-instruct | 8B / unknown | unknown | instruction-tuned | [primary source](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct/resolve/0e9e39f249a16976918f6564b8830bc894c89659/README.md) |
| model.llama3-8b-instruct | 8B / unknown | unknown | instruction-tuned | [primary source](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct/resolve/8afb486c1db24fe5011ec46dfbe5b5dccdb575c2/README.md) |
| model.llama4-maverick-fp8 | 400B / 17B | unknown | text, image | [primary source](https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8/resolve/94125d2bd83076b21eed33119525e29eaf3894f4/README.md) |
| model.llama-cdg | unknown / unknown | 32 | llama | [primary source](https://huggingface.co/PinzhengWang/Llama_CDG/resolve/fd13812158755bedc5daadec66850c6faa3abd39/config.json) |
| model.codellama-34b-instruct | 34B / unknown | unknown | text | [primary source](https://raw.githubusercontent.com/meta-llama/codellama/e81b597e44dbecc2a0dedb9949fdf84adfc22395/MODEL_CARD.md) |
| model.gpt-4o-mini | unknown / unknown | unknown | text, image | [primary source](https://developers.openai.com/api/docs/models/gpt-4o-mini.md) |
| model.gpt-4o | unknown / unknown | unknown | text, image | [primary source](https://developers.openai.com/api/docs/models/gpt-4o.md) |
| model.gpt-5.4-mini | unknown / unknown | unknown | text, image | [primary source](https://developers.openai.com/api/docs/models/gpt-5.4-mini.md) |
| model.claude3-sonnet | unknown / unknown | unknown | vision-capable language model | [primary source](https://www.anthropic.com/news/claude-3-family) |
| model.gemini1.5-flash | unknown / unknown | unknown | distilled | [primary source](https://blog.google/innovation-and-ai/products/google-gemini-update-flash-ai-assistant-io-2024/) |
| model.qwen3-235b-thinking-2507 | 235B / 22B | 94 | thinking-only | [primary source](https://huggingface.co/Qwen/Qwen3-235B-A22B-Thinking-2507/resolve/6cbffae6d8e28b986a6b17bd36f42f9fa0f1f0a5/README.md) |
| model.qwen3.5-122b-a10b | 122B / 10B | 48 | qwen3_5_moe_text | [primary source](https://huggingface.co/Qwen/Qwen3.5-122B-A10B/resolve/dc4d348443bc740c68e2d77492492c11606384d5/config.json) |
| model.deepseek-v4-pro | 1600B / 49B | 61 | deepseek_v4 | [primary source](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/resolve/b5968e9190ef611bbf34a7229255be88a0e937c1/config.json) |
| model.sqlcoder-7b-2 | unknown / unknown | 32 | SQL generation model | [primary source](https://huggingface.co/defog/sqlcoder-7b-2/resolve/7e5b6f7981c0aa7d143f6bec6fa26625bdfcbe66/README.md) |
| model.mistral-7b-instruct-v0.2 | unknown / unknown | 32 | mistral | [primary source](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/resolve/63a8b081895390a26e140280378bc85ec8bce07a/config.json) |
| model.mixtral-8x7b-instruct-v0.1 | unknown / unknown | 32 | mixtral | [primary source](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1/resolve/eba92302a2861cdc0098cc54bc9f17cb2c47eb61/config.json) |
| model.wizardcoder-33b-v1.1 | unknown / unknown | 62 | llama | [primary source](https://huggingface.co/WizardLM/WizardCoder-33B-V1.1/resolve/22d03e16bd052cec88228e17e832f851391e3565/config.json) |
| model.dbrx-instruct | 132B / 36B | unknown | unknown | [primary source](https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm) |
| model.codellama-70b-instruct | 70B / unknown | unknown | instruction-tuned code model | [primary source](https://huggingface.co/meta-llama/CodeLlama-70b-Instruct-hf/resolve/2826050dce6c8271bb9f667083b64e8254bd374f/README.md) |
| model.llama2-70b-chat | 70B / unknown | unknown | dialogue fine-tuned | [primary source](https://raw.githubusercontent.com/meta-llama/llama/llama_v2/MODEL_CARD.md) |
| model.llama3-8b-family | 8B / unknown | unknown | text | [primary source](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct/resolve/8afb486c1db24fe5011ec46dfbe5b5dccdb575c2/README.md) |
| model.metamath-llama-family | unknown / unknown | unknown | mathematical reasoning fine-tuning family; size/version unknown | [primary source](https://meta-math.github.io/) |
| model.metamath-mistral-family | unknown / unknown | unknown | mathematical reasoning fine-tuning family; size/version unknown | [primary source](https://meta-math.github.io/) |
| model.claude3-haiku | unknown / unknown | unknown | vision-capable language model | [primary source](https://www.anthropic.com/news/claude-3-family) |
| model.claude3-opus | unknown / unknown | unknown | vision-capable language model | [primary source](https://www.anthropic.com/news/claude-3-family) |
| model.claude-sonnet4.6 | unknown / unknown | unknown | language model with computer-use capabilities | [primary source](https://www.anthropic.com/news/claude-sonnet-4-6) |
| model.claude-opus4.7 | unknown / unknown | unknown | vision-capable language model | [primary source](https://www.anthropic.com/news/claude-opus-4-7) |
| model.gemini3.1-pro-preview | unknown / unknown | unknown | text, image, video, audio, PDF | [primary source](https://ai.google.dev/gemini-api/docs/models/gemini-3.1-pro-preview) |
| model.gemini3.1-flashlite-preview | unknown / unknown | unknown | text, image, video, audio, PDF | [primary source](https://ai.google.dev/gemini-api/docs/models/gemini-3.1-flash-lite-preview) |
| model.gemini1.0-pro-family | unknown / unknown | unknown | text | [primary source](https://blog.google/technology/ai/gemini-api-developers-cloud/) |
| model.gpt5.4 | unknown / unknown | unknown | text, image | [primary source](https://developers.openai.com/api/docs/models/gpt-5.4.md) |
| model.gpt4-turbo-preview | unknown / unknown | unknown | text | [primary source](https://developers.openai.com/api/docs/models/gpt-4-turbo-preview.md) |
| model.gpt3.5-turbo-family | unknown / unknown | unknown | text | [primary source](https://developers.openai.com/api/docs/models/gpt-3.5-turbo.md) |
| model.gpt3.5-turbo16k-family | unknown / unknown | unknown | text | [primary source](https://developers.openai.com/api/docs/models/gpt-3.5-turbo-16k-0613.md) |
| model.bedrock-mistral7b-family | unknown / unknown | unknown | mistral | [primary source](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/resolve/63a8b081895390a26e140280378bc85ec8bce07a/config.json) |
| model.bedrock-mixtral-family | unknown / unknown | unknown | unknown | [primary source](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1/resolve/eba92302a2861cdc0098cc54bc9f17cb2c47eb61/README.md) |

## Files and use

The schema stays version 1.0. `profiles.jsonl` contains 44 model/artifact/family profiles, eight harness profiles and 178 recorded-run profiles. `sources.jsonl` contains 86 source records: 74 unreviewed external research sources, ten original subject tables, and two explicitly excluded unofficial-mirror snapshots. There are 76 hashed evidence snapshots. No fact references an excluded source.

Specification facts: 240 documented, six explicitly inferred, 273 recorded settings. Unknown fields remain unknown. Inferred facts are not rendered by the enrichment policy. Extra run settings are preserved verbatim, with literal output budgets, shot counts, serving environment and reasoning effort additionally typed where supplied. Documentation defaults are not recorded run settings.

The new opt-in policy is `configs/head_finetuning/enrichment_catalog_v2.json`. It uses the existing renderer version and source guards, but a different catalog and field order. Rounded size, parameterization, model type and modalities come before deeper configuration dimensions. Old policies, prepared bundles and checkpoints remain unchanged and valid. A new preparation/output directory is required before training with this policy. Do not apply it to the old trained head as though its contract matched.

```bash
.venv-laya-mlx/bin/python scripts/build_system_profile_collection.py --catalog research/system_profiles/collection_v2
.venv-laya-mlx/bin/python scripts/validate_system_profiles.py --catalog research/system_profiles/collection_v2
.venv-laya-mlx/bin/python scripts/audit_system_profile_coverage.py --pilot --out runs/NEW_COVERAGE_AUDIT
```

`--require-full` on the audit writes the report then exits with status 2 while any model/harness gap remains. It cannot silently turn inventory placeholders into coverage. The audit verifies exact subject fields against hashed original subject tables and frozen roles. Source restoration uses the existing fetch script with `--catalog research/system_profiles/collection_v2`; mutable pages must still match saved hashes.

Full retained-input report: `runs/head_finetuning/system_profile_coverage_v2_final/report.html`. Durable metrics: [coverage_summary.json](coverage_summary.json). External sources remain research-only and unreviewed for competition eligibility. This work did not change scores or launch a training/GPU job.
