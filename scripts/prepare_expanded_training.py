#!/usr/bin/env python3
"""Prepare 30k enriched records within frozen roles; no model weights or scoring."""
import argparse
from collections import Counter
from html import escape
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import pandas as pd
from paiec.expanded_data import ExpandedDataConfig, balanced_unique_sample, monitor_key, load_candidates
from paiec.compact_inputs import CompactStateBuilder
from paiec.compact_profiles import CompactProfiles
from paiec.competition import competition_input
from paiec.head_pilot import snapshot_profiles, read_rows, input_contract, write_json, digest_state
from paiec.head_training import file_hash
from paiec.laya_predictor import load_questions
from paiec.profile_enrichment import native_budget
from paiec.schema import subject_record, item_record
from measure_tokens import load_tokenizer


def prepare(root, config):
    out = root / config.prepared
    out.mkdir(parents=True, exist_ok=False)
    candidates, tables, split, sources = load_candidates(root, config, out)
    monitor = candidates.task_sha256.map(lambda key: monitor_key(key, config.seed))
    selected = {"training": balanced_unique_sample(candidates[~monitor], config.samples, config.seed),
                "monitor": balanced_unique_sample(candidates[monitor], config.monitor_samples, config.seed)}
    if set(selected["training"].task_sha256) & set(selected["monitor"].task_sha256):
        raise ValueError("Training monitor contains training tasks")
    policy = snapshot_profiles(root / config.policy, out, root)
    tok = load_tokenizer("english")
    question = load_questions()
    _, room = native_budget(tok, question["correct"], config.max_len, "mlx", config.head_max_len)
    builder = CompactStateBuilder(tok, room, CompactProfiles(policy, root=out))
    (out / "enriched").mkdir()
    (out / "source").mkdir()
    provenance, summaries, audit = [], [], []
    original = root / split["artifact_directory"]
    original_frame = pd.read_parquet(original / "provenance.parquet")
    for role in ["training", "monitor", "development"]:
        if role == "development":
            rows = read_rows(original / "development.jsonl")
            records = original_frame[original_frame.role.eq(role) & original_frame.exported].sort_values("jsonl_line").to_dict("records")
        else:
            records, rows = selected[role].to_dict("records"), []
            for record in records:
                subjects, items = tables[record["benchmark"]]
                item = item_record(items.loc[record["item_id"]], record["benchmark"])
                item["interactors"] = record["interactors"]
                subject, item = competition_input(subject_record(subjects.loc[record["subject_id"]]), item)
                rows.append(dict(state=json.dumps(dict(target_system=subject, target_item=item), ensure_ascii=False),
                                 questions=question, expected={"correct": bool(record["response"])}))
        counts = Counter(records=len(rows))
        with (out / "enriched" / (role + ".jsonl")).open("w") as target, (out / "source" / (role + ".jsonl")).open("w") as source:
            for line, (row, record) in enumerate(zip(rows, records)):
                parsed = json.loads(row["state"])
                pair = builder.build(parsed["target_system"], parsed["target_item"])
                if pair.visibility["options_omitted"] or not pair.visibility["matched_task_payload"]:
                    raise ValueError("Task/options preservation failed")
                ids = builder.ids(pair.enriched)
                if len(ids) > room:
                    raise ValueError("Native token overflow")
                row_id = str(record.get("row_id", f"{role}:{record['benchmark']}:{record['response_id']}"))
                base = {key: record[key] for key in ["benchmark", "subject_id", "item_id", "response_id", "trial", "family", "task_sha256"]}
                provenance.append(dict(base, role=role, row_id=row_id, jsonl_line=line,
                    exported=True, pool="evaluation" if role == "development" else "",
                    sample_weight=float(record.get("sample_weight", 1.)),
                    state_sha256=digest_state(pair.enriched), source_state_sha256=digest_state(row["state"]),
                    state_tokens=len(ids), effective_input_sha256=digest_state(json.dumps(ids)),
                    retained_task_sha256=pair.visibility["retained_task_sha256"]))
                target.write(json.dumps(dict(row, state=pair.enriched), ensure_ascii=False) + "\n")
                source.write(json.dumps(row, ensure_ascii=False) + "\n")
                for kind in ["model", "harness"]:
                    counts[kind + "_facts"] += pair.enrichment["delivered"][kind]
                for key in ["task_omitted", "question_or_user_omitted", "options_omitted"]:
                    counts[key] += pair.visibility[key]
                audit.append(dict(role=role, row_id=row_id, visibility=pair.visibility, enrichment=pair.enrichment))
                if (line + 1) % 1000 == 0:
                    print(role, line + 1, flush=True)
        summaries.append(dict(role=role, **counts))
    frame = pd.DataFrame(provenance)
    if (frame.groupby("effective_input_sha256").role.nunique() > 1).any():
        raise ValueError("Identical retained input spans folds")
    frame.to_parquet(out / "enriched/provenance.parquet", index=False)
    frame.groupby(["role", "family", "benchmark"]).agg(records=("row_id", "size"), pairs=("subject_id", "nunique")).to_csv(out / "sample_counts.csv")
    write_json(out / "enriched/input_contract.json", input_contract(out, "enriched", config))
    with (out / "input_audit.jsonl").open("w") as handle:
        for row in audit:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_json(out / "summary.json", dict(rows=summaries, candidate_unique_system_items=len(candidates),
        candidate_pairs=candidates[["benchmark", "subject_id"]].drop_duplicates().shape[0],
        training_unique_system_items=config.samples, monitor_question_disjoint=True,
        original_development_unchanged=True, calibration_opened=False, confirmation_opened=False,
        training_performed=False, scores_computed=False))
    renderer = input_contract(out, "enriched", config)["renderer_sources"]
    write_json(out / "manifest.json", dict(version="enriched_expanded_v1", data_config=config.model_dump(),
        split_sha256=file_hash(root / config.split), dataset_revision=split["dataset_revision"],
        checkpoint_revision=split["input"]["checkpoint_revision"], question=question,
        state_room=room, renderer_sources=renderer, source_sha256=sources,
        original_development_sha256=file_hash(original / "development.jsonl"),
        original_evaluation_sha256=file_hash(original / "evaluation_manifest.csv"),
        selection="family then pair balanced; unique system/item; uniformly random recorded trial; outcome blind",
        monitor="5% normalized task-hash bucket within training families; all systems/trials inherit fold",
        artifacts={str(p.relative_to(out)): file_hash(p) for p in sorted(out.rglob("*")) if p.is_file()}))
    html = '<!doctype html><meta charset="utf-8"><title>Expanded enriched training data</title><style>body{font:16px system-ui;margin:40px;line-height:1.5}td,th{padding:8px}</style>'
    html += '<h1>Expanded enriched training data</h1><p>Preparation only. Frozen family roles and original development unchanged. One uniformly sampled recorded trial per distinct system/question. No outcome-based class balancing. Internal monitor tasks are excluded across every training system.</p>'
    html += pd.DataFrame(summaries).to_html(index=False)
    html += frame.groupby(["role", "family"]).agg(records=("row_id", "size"), pairs=("subject_id", "nunique")).to_html()
    html += '<pre>' + escape((out / "summary.json").read_text()) + '</pre>'
    (out / "report.html").write_text(html)
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/head_finetuning/enriched_30k_data_v1.json")
    args = parser.parse_args()
    prepare(ROOT, ExpandedDataConfig.model_validate_json((ROOT / args.config).read_text()))
