#!/usr/bin/env python3
"""Prepare matched task-v2 inputs/features and retained-input checks. No weights."""
import argparse
from html import escape
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import pandas as pd
from paiec.head_pilot import load_config, read_rows, write_json, snapshot_profiles, input_contract, digest_state
from paiec.head_training import file_hash
from paiec.laya_predictor import load_questions
from paiec.profile_enrichment import ProfileEnricher, native_budget
from paiec.task_inputs import TaskStateBuilder
from measure_tokens import load_tokenizer


def prepare(config):
    if config.input_pipeline != "task_v2":
        raise ValueError("Select an explicit task_v2 recipe")
    out = ROOT / config.prepared
    out.mkdir(parents=True, exist_ok=False)
    split_path = ROOT / config.split
    split = json.loads(split_path.read_text())
    source = ROOT / split["artifact_directory"]
    for name in ("training.jsonl", "development.jsonl", "provenance.parquet"):
        if file_hash(source/name) != split["artifact_fingerprints"][name]:
            raise ValueError("Frozen source changed: " + name)
    question = load_questions()
    if question != split["input"]["question_definition"]:
        raise ValueError("Question differs from frozen split")
    tok = load_tokenizer("english")
    count, room = native_budget(tok, question["correct"], config.max_len, "mlx", config.head_max_len)
    builder = TaskStateBuilder(tok, room)
    policy = snapshot_profiles(ROOT / config.policy, out, ROOT)
    enricher = ProfileEnricher(policy, root=out, token_count=count, state_room=room)
    original = pd.read_parquet(source/"provenance.parquet")
    manifests = {"baseline": [], "enriched": []}
    features, examples = [], []
    for variant in manifests:
        (out/variant).mkdir()
    for role in ("training", "development"):
        rows = read_rows(source/(role+".jsonl"))
        selected = original[original.role.eq(role) & original.exported].sort_values("jsonl_line")
        if selected.jsonl_line.tolist() != list(range(len(rows))):
            raise ValueError("Frozen source ordering differs")
        if role == "training" and len(rows) != config.samples:
            raise ValueError("Training size differs")
        outputs = {variant: [] for variant in manifests}
        for row, provenance in zip(rows, selected.to_dict("records")):
            if digest_state(row["state"]) != provenance["state_sha256"] or row["questions"] != question:
                raise ValueError("Frozen state/question changed")
            parsed = json.loads(row["state"])
            built = builder.build(parsed["target_system"], parsed["target_item"])
            enriched, enrichment = enricher.enrich(built.state, parsed["target_system"])
            # Features and renderer never receive labels; preserve outcomes only
            # in the training row supplied to the supervised trainer/scorer.
            features.append(dict(role=role, jsonl_line=provenance["jsonl_line"],
                row_id=provenance["row_id"], features=built.features, visibility=built.visibility))
            for variant, state in (("baseline", built.state), ("enriched", enriched)):
                outputs[variant].append(json.dumps(dict(row, state=state), ensure_ascii=False))
                ids = builder.ids(state)
                if len(ids) > room:
                    raise ValueError("Native input overflow")
                manifests[variant].append(dict(provenance, source_state_sha256=provenance["state_sha256"],
                    state_sha256=digest_state(state), state_tokens=len(ids), state_tokens_used=len(ids),
                    truncated=False, effective_input_sha256=digest_state(json.dumps(ids)),
                    task_omitted=built.visibility["task_omitted"], system_omitted=built.visibility["system_omitted"],
                    metadata_omitted=built.visibility["metadata_omitted"],
                    retained_task_sha256=built.visibility["retained_task_sha256"],
                    source_task_content_sha256=built.visibility["source_task_content_sha256"],
                    variant=variant, enriched=bool(enrichment["selected_facts"]) if variant=="enriched" else False,
                    added_tokens=enrichment["added_tokens"] if variant=="enriched" else 0))
            if len(examples) < 3 or (role == "training" and provenance["jsonl_line"] in (14,41)):
                examples.append(dict(row_id=provenance["row_id"], role=role,
                    state=built.state, features=built.features, visibility=built.visibility))
        for variant, lines in outputs.items():
            (out/variant/(role+".jsonl")).write_text("\n".join(lines)+"\n")
        print(f"Prepared {role}: {len(rows)} records", flush=True)
    audits = []
    for variant, records in manifests.items():
        frame = pd.DataFrame(records)
        if (frame.groupby("effective_input_sha256").role.nunique()>1).any():
            raise ValueError("Retained inputs overlap frozen roles")
        frame.to_parquet(out/variant/"provenance.parquet", index=False)
        write_json(out/variant/"input_contract.json", input_contract(out, variant, config))
        for (role, benchmark), group in frame.groupby(["role","benchmark"]):
            hashes = group.groupby("effective_input_sha256").agg(tasks=("task_sha256","nunique"), rows=("row_id","size"))
            content_hashes = group.groupby(["subject_id","retained_task_sha256"]).agg(
                tasks=("source_task_content_sha256","nunique"),rows=("row_id","size"))
            audits.append(dict(variant=variant, role=role, benchmark=benchmark, records=len(group),
                task_omitted=int(group.task_omitted.sum()),system_omitted=int(group.system_omitted.sum()),
                metadata_omitted=int(group.metadata_omitted.sum()), native_truncated=int(group.truncated.sum()),
                multitask_collision_rows=int(hashes[hashes.tasks.gt(1)].rows.sum()),
                content_collision_rows=int(content_hashes[content_hashes.tasks.gt(1)].rows.sum()),
                effective_inputs=group.effective_input_sha256.nunique()))
    audit = pd.DataFrame(audits)
    audit.to_csv(out/"visibility.csv", index=False)
    (out/"features.jsonl").write_text("\n".join(json.dumps(r,ensure_ascii=False) for r in features)+"\n")
    write_json(out/"examples.json", examples)
    source_hashes = input_contract(out,"baseline",config)["renderer_sources"]
    manifest = dict(config=config.model_dump(), split_sha256=file_hash(split_path),
        dataset_revision=split["dataset_revision"], checkpoint_revision=split["input"]["checkpoint_revision"],
        question=question, state_room=room, enrichment=enricher.metadata,
        renderer_sources=source_hashes, training_performed=False, model_inference=False,
        calibration_opened=False, confirmation_opened=False, planned_updates_per_variant=config.updates,
        artifacts={str(p.relative_to(out)):file_hash(p) for p in sorted(out.rglob("*")) if p.is_file()})
    write_json(out/"manifest.json", manifest)
    html = '<!doctype html><meta charset="utf-8"><title>Task-v2 input preparation</title><style>body{font:16px system-ui;max-width:1200px;margin:40px auto;padding:20px}pre{white-space:pre-wrap;background:#f3f4f6;padding:16px}td,th{padding:7px}</style>'
    html += '<h1>Task-v2 input preparation</h1><p>No training, inference or quality scores. Explicit field budgets replace silent native truncation. Long tasks still lose content: inspect the omission flags below. Frozen records, labels and role assignments are preserved.</p><p>content_collision_rows compares retained task text within each system, ignoring item metadata and IDs. Zero whole-input collisions does not prove complete task visibility.</p>'
    html += '<p><a href="../../../docs/task-feature-pipeline.md">Design, sources and validation</a> · <a href="../evidence_control_v1/report.html">Separate saved-head evidence control</a></p>'
    html += audit.to_html(index=False)
    for example in examples:
        html += '<h2>Real prepared input</h2><pre>'+escape(example["state"])+ '</pre><pre>'+escape(json.dumps(example["visibility"],indent=2))+'</pre>'
    (out/"report.html").write_text(html)
    print(audit.to_string(index=False))
    return out


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default="configs/head_finetuning/task_features_pilot_v2.json")
    args=parser.parse_args()
    prepare(load_config(ROOT/args.config))
