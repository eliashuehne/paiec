#!/usr/bin/env python3
"""Prepare shared-task compact inputs and real-token visibility. No weights or scoring."""
import argparse
from collections import Counter
from html import escape
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import pandas as pd
from paiec.compact_inputs import CompactStateBuilder
from paiec.compact_profiles import CompactProfiles
from paiec.head_pilot import load_config, read_rows, write_json, snapshot_profiles, input_contract, digest_state
from paiec.head_training import file_hash
from paiec.laya_predictor import load_questions
from paiec.profile_enrichment import native_budget
from measure_tokens import load_tokenizer


def prepare(config):
    if config.input_pipeline != "compact_v3":
        raise ValueError("Select an explicit compact-v3 recipe")
    out = ROOT / config.prepared
    out.mkdir(parents=True, exist_ok=False)
    split_path = ROOT / config.split
    split = json.loads(split_path.read_text())
    source = ROOT / split["artifact_directory"]
    for name in ["training.jsonl", "development.jsonl", "provenance.parquet", "evaluation_manifest.csv"]:
        if file_hash(source / name) != split["artifact_fingerprints"][name]:
            raise ValueError("Frozen artifact changed: " + name)
    question = load_questions()
    if question != split["input"]["question_definition"]:
        raise ValueError("Frozen question differs")
    tokenizer = load_tokenizer("english")
    _, room = native_budget(tokenizer, question["correct"], config.max_len, "mlx", config.head_max_len)
    policy = snapshot_profiles(ROOT / config.policy, out, ROOT)
    profiles = CompactProfiles(policy, root=out)
    builder = CompactStateBuilder(tokenizer, room, profiles)
    provenance = pd.read_parquet(source / "provenance.parquet")
    manifests = {variant: [] for variant in ["baseline", "enriched"]}
    features, audits, examples = [], [], []
    summary = []
    for variant in manifests:
        (out / variant).mkdir()
    for role in ["training", "development"]:
        rows = read_rows(source / (role + ".jsonl"))
        selected = provenance[provenance.role.eq(role) & provenance.exported].sort_values("jsonl_line")
        if selected.jsonl_line.tolist() != list(range(len(rows))):
            raise ValueError("Frozen provenance ordering differs")
        if role == "training" and len(rows) != config.samples:
            raise ValueError("Frozen training count differs")
        outputs = {variant: [] for variant in manifests}
        counts = Counter(records=len(rows))
        for row, original in zip(rows, selected.to_dict("records")):
            if digest_state(row["state"]) != original["state_sha256"] or row["questions"] != question:
                raise ValueError("Frozen input state/question changed")
            parsed = json.loads(row["state"])
            pair = builder.build(parsed["target_system"], parsed["target_item"])
            v, e = pair.visibility, pair.enrichment
            if v["options_omitted"] or not v["matched_task_payload"]:
                raise ValueError("Task/options preservation gate failed")
            for kind in ["model", "harness"]:
                counts[kind + "_available"] += e["available"][kind]
                counts[kind + "_delivered"] += e["delivered"][kind]
            counts["any_facts_delivered"] += any(e["delivered"].values())
            for key in ["task_omitted", "question_or_user_omitted", "options_omitted", "system_omitted", "metadata_omitted"]:
                counts[key] += v[key]
            counts["rows_with_detected_options"] += v["options_detected"] > 0
            audits.append(dict(role=role, row_id=original["row_id"], jsonl_line=original["jsonl_line"],
                               benchmark=original["benchmark"], subject_id=original["subject_id"],
                               visibility=v, enrichment=e))
            features.append(dict(role=role, row_id=original["row_id"], features=pair.features, visibility=v))
            for variant, state in [("baseline", pair.baseline), ("enriched", pair.enriched)]:
                outputs[variant].append(dict(row, state=state))
                ids = builder.ids(state)
                if len(ids) > room:
                    raise ValueError("Native token overflow")
                manifests[variant].append(dict(original, source_state_sha256=original["state_sha256"],
                    state_sha256=digest_state(state), state_tokens=len(ids), state_tokens_used=len(ids),
                    truncated=False, effective_input_sha256=digest_state(json.dumps(ids)), variant=variant,
                    enriched=variant == "enriched" and bool(e["selected_facts"]),
                    added_tokens=len(ids)-builder.count(pair.baseline) if variant == "enriched" else 0,
                    task_omitted=v["task_omitted"], system_omitted=v["system_omitted"], metadata_omitted=v["metadata_omitted"],
                    retained_task_sha256=v["retained_task_sha256"], source_task_content_sha256=v["source_task_content_sha256"]))
            if len(examples) < 2 or (role == "training" and original["jsonl_line"] in [62,1229]) or (role == "development" and e["model_identity_basis"].startswith("documented_") and len(examples) < 7):
                examples.append(dict(role=role, row_id=original["row_id"], baseline=pair.baseline,
                                     enriched=pair.enriched, visibility=v, enrichment=e))
        for variant in manifests:
            (out / variant / (role + ".jsonl")).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in outputs[variant]))
        summary.append(dict(role=role, **counts))
        print(role, dict(counts), flush=True)
    for variant, records in manifests.items():
        frame = pd.DataFrame(records)
        if (frame.groupby("effective_input_sha256").role.nunique() > 1).any():
            raise ValueError("Effective inputs overlap frozen roles")
        frame.to_parquet(out / variant / "provenance.parquet", index=False)
        write_json(out / variant / "input_contract.json", input_contract(out, variant, config))
    for name, records in [("features.jsonl", features), ("input_audit.jsonl", audits)]:
        (out / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    write_json(out / "summary.json", {"rows": summary, "native_truncation": 0,
        "control_enriched_task_payloads_identical": True, "full_task_retention": False,
        "training_performed": False, "model_inference": False, "scores_computed": False,
        "context_limit": config.max_len, "context_limit_comparison": "Task-v2 reference used 1024 tokens; new compact recipe uses 2048. No quality gain is established."})
    write_json(out / "examples.json", examples)
    source_hashes = input_contract(out, "baseline", config)["renderer_sources"]
    manifest = dict(config=config.model_dump(), split_sha256=file_hash(split_path),
        dataset_revision=split["dataset_revision"], checkpoint_revision=split["input"]["checkpoint_revision"],
        question=question, state_room=room, enrichment=profiles.metadata,
        renderer_sources=source_hashes, training_performed=False, model_inference=False,
        calibration_opened=False, confirmation_opened=False, planned_updates_per_variant=config.updates,
        original_evaluation_sha256=file_hash(source / "evaluation_manifest.csv"),
        artifacts={str(p.relative_to(out)): file_hash(p) for p in sorted(out.rglob("*")) if p.is_file()})
    write_json(out / "manifest.json", manifest)
    html = '<!doctype html><meta charset="utf-8"><title>Compact Laya inputs</title><style>body{font:16px system-ui;max-width:1200px;margin:32px auto;padding:16px}td,th{padding:8px}pre{white-space:pre-wrap;background:#f3f4f6;padding:16px}</style>'
    html += '<h1>Compact Laya inputs: preparation only</h1><p>The control and enriched versions have identical retained task text. Every safely available model/harness gets a documented fact. Detected answer options remain complete. Long questions and context can still have explicit omissions, counted below.</p><p>The new recipe uses 2,048 tokens; the old task-v2 reference used 1,024. This is an input check, not a model-quality result. Original evaluation records and labels remain unchanged.</p>'
    html += pd.DataFrame(summary).to_html(index=False)
    for example in examples:
        html += '<h2>Real input: ' + escape(example["row_id"]) + '</h2><pre>' + escape(example["enriched"]) + '</pre><details><summary>Visibility and sources</summary><pre>' + escape(json.dumps({"visibility":example["visibility"],"enrichment":example["enrichment"]},indent=2)) + '</pre></details>'
    (out / "report.html").write_text(html)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/head_finetuning/compact_pilot_v3.json")
    args = parser.parse_args()
    prepare(load_config(ROOT / args.config))
