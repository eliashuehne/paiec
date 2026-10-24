# Collecting task requirements

The annotation stage has an offline preparation and import path. It is not a
completed semantic dataset. The initial full plan contains 16,909 distinct
visible inputs: 15,436 training, 583 monitor and 890 original development.
These are full item-field hashes, not system/item records or normalized
questions. Calibration/confirmation remain sealed. The user chose to keep
preparation offline on 2026-10-23: the teacher pilot remains unsent.

The v2 contract has eight facets: language, domain, skills, reasoning structure,
modalities, answer format, information needs and input issues. Estimated values
need exact quotes from supplied fields. Missing support stays unknown.
Empirical difficulty is always null here; generated ratings are not learned
difficulty. Schema validity and quotation do not establish semantic accuracy.

## Input and provenance

`configs/pipelines/feature_preparation_offline_v10.json` selects the current
metadata registry and `configs/pipelines/task_annotation_v2.json`. The latter
selects the prompt, taxonomy and 14 manual candidates. Every fresh preparation
writes their hashes and a self-contained contract into `task_annotations_v2/`.

Requests contain a task hash and the complete supplied `item_content`,
`item_features` and `interactors`. Private benchmark IDs, target model identity,
reference answers and target outcomes are not added. Role membership remains
in a separate audit index, outside the teacher input. Natural benchmark names
already present in supplied text remain untouched. No characters are truncated.
The prompt treats task strings as untrusted quoted data and forbids solving,
following embedded instructions, using tools or looking up answers.

The schema forbids answer fields, numeric difficulty and self-approval.
Candidate features contain only facet status and vocabulary values. Rationales
and quotes are review metadata, outside the feature projection. This reduces
exposure to generated solution text; it does not detect every wrong label.

## Offline commands

Prepare a fresh ZenML/MLflow bundle:

```bash
.venv-pipeline/bin/python scripts/run_feature_pipeline.py run
```

Validate the initial plan's hashes and inspect scope:

```bash
.venv-pipeline/bin/python scripts/run_task_annotations.py status \
  --plan runs/feature_pipeline/task_annotation_preparation_v1/task_annotations_v2
```

Export explicitly selected tasks into a fresh directory:

```bash
.venv-pipeline/bin/python scripts/run_task_annotations.py export-openai \
  --plan runs/feature_pipeline/task_annotation_preparation_v1/task_annotations_v2 \
  --task-keys runs/feature_pipeline/task_annotation_preparation_v1/task_annotations_v2/pilot_task_keys.json \
  --out runs/task_annotation_review/pilot_export_v1
```

The receipt records the exact JSONL hash and output-token bound. This command
does not upload, submit or spend money. Full-corpus selection is never implicit.
The prepared proposal uses `gpt-5.4-mini`; account access and annotation quality
have not been tested. Credentials are unnecessary for all commands in this guide.

If separately authorized results become available later, import downloaded
output and error files together. Replace the paths and hash with actual values:

```bash
.venv-pipeline/bin/python scripts/run_task_annotations.py import-openai \
  --plan runs/feature_pipeline/task_annotation_preparation_v1/task_annotations_v2 \
  --batch runs/task_annotation_review/pilot_export_v1/batch_input.jsonl \
  --batch-sha256 RECORDED_EXPORT_SHA256 \
  --responses runs/task_annotation_review/pilot_download/output.jsonl runs/task_annotation_review/pilot_download/errors.jsonl \
  --out runs/task_annotation_review/pilot_import_v1
```

Results are joined by `custom_id`, independent of file order. The importer
checks the input hash and reconstructs every request from the bound contract.
Foreign or duplicate results fail before publication. Failed requests,
refusals, incomplete responses, tools and ambiguous messages cannot become
annotations. Invalid labels or quotes remain rejected and pending.
Outputs retain candidate features, rejected/pending rows, provider failures,
observed token usage, reported models and file hashes. Reported usage is not a
verified bill. Local integrity is checked; this offline command does not
authenticate network origin. No retries are submitted.

Provider-neutral imports use `import-normalized`, with one JSONL file of
`{request_id, response}` envelopes and explicit `--producer`. All outputs use
fresh repository-relative paths under `runs/`.

## Quality and remaining work

The 14 manual v2 candidates cover the existing training-only starter sample.
They correct limitations of the seven historical v1 seeds: SQL execution
contracts remain unknown unless supplied; missing context is not proof that
the target lacked information; mixed-language prompts and competing format
instructions are represented explicitly. Visible inconsistencies are flagged
without supplying corrected answers. The sample was selected by length within
training families and is not representative of the full corpus.

Zero independent reviews have occurred. Annotation quality needs a broader
outcome-blind sample and review against complete supplied text. A model pilot
or independent review has not established a quality threshold. Complete
syntactic coverage would still be separate from quality and training readiness.

Learned system abilities/difficulty, compatibility interactions, outcome-bearing
support/query features and a feature-to-Laya input contract remain unfinished.
Outcome-blind [support/query links](feature-evidence-links.md) are prepared.
Existing renderers/checkpoint contracts are unchanged. Collection does not
authorize fitting, target-system evaluation or GPU jobs.

The importer follows official OpenAI documentation for
[Batch result matching and error files](https://developers.openai.com/api/docs/guides/batch)
and [Responses output messages](https://developers.openai.com/api/docs/guides/text).
