# Collection v3: scoped family context and actual delivery

This preserves v1/v2 and adds all 19 subjects in the already frozen development
families to the prior 170 training candidates: 189 systems, 46 model/artifact/
family profiles, eight harness profiles, 88 sources and 78 hashed snapshots.

Model/base-family context is available for 177 systems, harness context for 182,
both for 170, and some context for all 189. These counts do not establish complete
model identities or exact evaluated weights. Ten C-Eval identities are missing;
two private CDG targets remain unverified; seven training harness names are absent.
Private fine-tuned GPT targets remain unbound with explicit separate base links.

See [compact delivery and development review](../../../docs/compact-inputs-and-development.md)
for source-scoped rules, the unchanged original evaluation, actual token delivery,
remaining task omissions and the question-only supplement. No training or quality
scores were produced. `compact_delivery_summary.json` and `development_review.jsonl`
are durable measured/review records; profile facts retain the same strict schema.

The new paper source is https://arxiv.org/html/2403.05266v3. Only model/base-family
identity passages were extracted; study outcomes and grading definitions were
not model inputs. Sources remain research-only and unreviewed for competition
eligibility. Rebuild/validate with the existing scripts and `--catalog
research/system_profiles/collection_v3`. Native-token audits are reproduced using
`configs/head_finetuning/compact_pilot_v3.json` and the development-review script.
