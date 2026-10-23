#!/usr/bin/env python3
"""Export paired baseline/enriched inputs and audit native tokens; no inference."""
import argparse
from html import escape
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
from paiec.head_pilot import (load_config, read_rows, write_json, digest_state,
    snapshot_profiles, input_contract, VARIANTS)
from paiec.head_training import file_hash
from paiec.laya_predictor import build_state, load_questions
from paiec.profile_enrichment import ProfileEnricher, native_budget
from paiec.tracking import ExperimentTracker
from measure_tokens import load_tokenizer


def prepare(config):
    if config.input_pipeline == "compact_v3":
        from prepare_compact_pilot import prepare as prepare_compact
        return prepare_compact(config)
    if config.input_pipeline == "task_v2":
        from prepare_task_pilot import prepare as prepare_task
        return prepare_task(config)
    out = ROOT / config.prepared
    out.mkdir(parents=True, exist_ok=False)
    frozen = ROOT / config.split
    split = json.loads(frozen.read_text())
    data = ROOT / split["artifact_directory"]
    for name in ["training.jsonl", "development.jsonl", "provenance.parquet"]:
        if file_hash(data / name) != split["artifact_fingerprints"][name]:
            raise ValueError(f"Frozen artifact changed: {name}")
    question = load_questions()
    if question != split["input"]["question_definition"]:
        raise ValueError("Frozen question differs")
    policy = snapshot_profiles(ROOT / config.policy, out, ROOT)
    tokenizer = load_tokenizer("english")
    count, room = native_budget(tokenizer, question["correct"], config.max_len, "mlx", config.head_max_len)
    enricher = ProfileEnricher(policy, root=out, token_count=count, state_room=room)
    provenance = pd.read_parquet(data / "provenance.parquet")
    audits, manifests = [], {variant: [] for variant in VARIANTS}
    examples = []
    for variant in VARIANTS:
        (out / variant).mkdir()
    for role in ["training", "development"]:
        source = read_rows(data / (role + ".jsonl"))
        selected = provenance[provenance.role.eq(role) & provenance.exported].sort_values("jsonl_line")
        if selected.jsonl_line.tolist() != list(range(len(source))):
            raise ValueError("Frozen response manifest does not align")
        if role == "training" and len(source) != config.samples:
            raise ValueError("Pilot must use exactly the configured frozen training records")
        handles = {v: (out / v / (role + ".jsonl")).open("w") for v in VARIANTS}
        try:
            for row, original in zip(source, selected.to_dict("records")):
                if digest_state(row["state"]) != original["state_sha256"] or row["questions"] != question:
                    raise ValueError("Frozen state/question changed")
                parsed = json.loads(row["state"])
                if set(parsed) != {"target_system", "target_item"}:
                    raise ValueError("Only original zero-shot states may be enriched")
                base = build_state(parsed["target_system"], parsed["target_item"], [], state_format="context_v1")
                enriched, details = enricher.enrich(base, parsed["target_system"])
                for variant, state in [("baseline", base), ("enriched", enriched)]:
                    # Label is cloned, never passed into the enricher or renderer.
                    handles[variant].write(json.dumps(dict(row, state=state), ensure_ascii=False) + "\n")
                    ids = tokenizer(state.replace(tokenizer.mask_token, " "), add_special_tokens=False)["input_ids"]
                    info = dict(original, source_state_sha256=original["state_sha256"],
                        state_sha256=digest_state(state), state_tokens=len(ids),
                        state_tokens_used=min(len(ids), room), truncated=len(ids) > room,
                        effective_input_sha256=digest_state(json.dumps(ids[:room])),
                        variant=variant, enriched=bool(details["selected_facts"]) if variant == "enriched" else False,
                        added_tokens=details["added_tokens"] if variant == "enriched" else 0)
                    manifests[variant].append(info)
                    audits.append({k: info[k] for k in ["role", "benchmark", "subject_id", "row_id", "variant", "state_tokens", "truncated", "enriched", "added_tokens"]})
                if len(examples) < 5 and details["selected_facts"]:
                    examples.append(dict(row_id=original["row_id"], role=role, baseline=base,
                                         enriched=enriched, evidence=details))
        finally:
            for handle in handles.values():
                handle.close()
        print(f"Prepared {role}: {len(source)} matched response records", flush=True)
    for variant in VARIANTS:
        pd.DataFrame(manifests[variant]).to_parquet(out / variant / "provenance.parquet", index=False)
        write_json(out / variant / "input_contract.json", input_contract(out, variant, config))
    frame = pd.DataFrame(audits)
    frame.to_csv(out / "token_audit.csv", index=False)
    summary = frame.groupby(["role", "variant"]).agg(rows=("row_id", "size"),
        truncated=("truncated", "sum"), enriched=("enriched", "sum"),
        median_state_tokens=("state_tokens", "median"), mean_added_tokens=("added_tokens", "mean"))
    summary.to_csv(out / "summary.csv")
    by_pair = frame.groupby(["role", "benchmark", "subject_id", "variant"]).agg(
        rows=("row_id", "size"), enriched=("enriched", "sum"), truncated=("truncated", "sum"))
    by_pair.to_csv(out / "coverage_by_pair.csv")
    write_json(out / "examples.json", examples)
    sources = ["src/paiec/laya_predictor.py", "src/paiec/state_rendering.py",
               "src/paiec/profile_enrichment.py", "src/paiec/input_contract.py"]
    manifest = dict(config=config.model_dump(), split_sha256=file_hash(frozen),
        dataset_revision=split["dataset_revision"], checkpoint_revision=split["input"]["checkpoint_revision"],
        question=question, state_room=room, enrichment=enricher.metadata,
        original_policy_sha256=file_hash(ROOT / config.policy),
        renderer_sources={name: file_hash(ROOT / name) for name in sources},
        training_performed=False, model_inference=False, confirmation_opened=False,
        calibration_opened=False, planned_updates_per_variant=config.updates,
        artifacts={str(p.relative_to(out)): file_hash(p) for p in sorted(out.rglob("*")) if p.is_file()})
    write_json(out / "manifest.json", manifest)
    report = '<!doctype html><meta charset="utf-8"><title>Enriched head pilot preparation</title><style>body{font:16px system-ui;max-width:1150px;margin:40px auto;padding:0 20px}td,th{padding:8px;border-bottom:1px solid #ddd}table{border-collapse:collapse}pre{white-space:pre-wrap;background:#f4f4f4;padding:16px}</style>'
    report += f'<h1>Prepared baseline vs enriched head experiment</h1><p>Input preparation only. No training, inference or Brier scores. Same frozen records, question and {config.max_len:,}-token limit. Enrichment uses at most {enricher.policy.max_added_tokens} spare tokens and never removes original text. External metadata is research-only.</p>'
    report += '<h2>Token audit</h2>' + summary.to_html() + '<h2>Coverage by system</h2>' + by_pair.to_html()
    for example in examples:
        report += '<h2>Real prepared example</h2><pre>' + escape(example["enriched"]) + '</pre>'
    (out / "report.html").write_text(report)
    tracker = ExperimentTracker(ROOT, experiment="PAIEC Head Fine-tuning")
    with tracker.run("enriched_v1 / input preparation / no training", manifest,
                     {"run_type": "input_preparation", "training_performed": False,
                      "max_len": config.max_len, "catalog_sha256": enricher.metadata["catalog_sha256"]},
                     {"run_type": "input_preparation", "quality_evaluation": "false"}) as tracked:
        if tracked:
            for name in ["manifest.json", "summary.csv", "coverage_by_pair.csv", "examples.json", "report.html"]:
                tracked.client.log_artifact(tracked.run_id, str(out / name))
            write_json(out / "mlflow_receipt.json", {"run_id": tracked.run_id})
    print(summary.to_string())
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/head_finetuning/enrichment_pilot_v1.json")
    args = parser.parse_args()
    prepare(load_config(args.config))
