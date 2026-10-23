#!/usr/bin/env python3
"""Measure sourced enrichment coverage; inventory rows never count as known facts.

Reads subject metadata, catalog facts and optionally the existing untrained pilot
inputs. No weights, outcomes, responses, fitting or model inference are needed.
"""
import argparse
from collections import Counter
import csv
from html import escape
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from paiec.head_training import file_hash
from paiec.profile_enrichment import ProfileEnricher, native_budget
from paiec.schema import SUBJECT_FIELDS, subject_record


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def fact_available(enricher, profile_id, kind):
    if not profile_id:
        return False
    fields = enricher.policy.selected_model_fields if kind == "model" else ["workflow"]
    return any(enricher._fact_line(profile_id, field, field) for field in fields)


def subject_coverage(rows, enricher):
    """Separate available sourced facts from actual 96-token probe rendering."""
    output = []
    for row in rows:
        match = enricher.resolve(row["subject"])
        _, audit = enricher.enrich("Catalog coverage probe task.", row["subject"])
        output.append({
            "benchmark": row["benchmark"], "subject_id": row["subject_id"],
            "role": row["role"], "model_name": row["subject"]["normalized_name"],
            "harness_name": row["subject"]["harness"],
            "model_profile_id": match["model_profile_id"],
            "harness_profile_id": match["harness_profile_id"],
            "model_fact_available": fact_available(enricher, match["model_profile_id"], "model"),
            "harness_fact_available": fact_available(enricher, match["harness_profile_id"], "harness"),
            "probe_model_fact_rendered": any(f["profile_id"].startswith("model.") for f in audit["selected_facts"]),
            "probe_harness_fact_rendered": any(f["profile_id"].startswith("harness.") for f in audit["selected_facts"]),
            "probe_added_tokens": audit["added_tokens"],
            "exact_evaluated_checkpoint_verified": match["exact_evaluated_checkpoint_verified"],
        })
    return output


def totals(rows):
    return {
        "systems": len(rows),
        "model_facts": sum(r["model_fact_available"] for r in rows),
        "harness_facts": sum(r["harness_fact_available"] for r in rows),
        "both_model_and_harness_facts": sum(r["model_fact_available"] and r["harness_fact_available"] for r in rows),
        "any_sourced_facts": sum(r["model_fact_available"] or r["harness_fact_available"] for r in rows),
        "probe_model_facts_rendered": sum(r["probe_model_fact_rendered"] for r in rows),
        "probe_harness_facts_rendered": sum(r["probe_harness_fact_rendered"] for r in rows),
        "exact_checkpoints_verified": sum(r["exact_evaluated_checkpoint_verified"] for r in rows),
    }


def validate_scope(catalog, rows):
    """Do not expand training roles or manufacture subject identifiers."""
    import pandas as pd
    recipe = json.loads((catalog / "recipe.json").read_text())
    selection = recipe["selection"]
    hash_keys = {"candidate_inventory": "candidate_inventory_sha256",
                 "development_inventory": "development_inventory_sha256",
                 "frozen_split_file": "frozen_split_sha256"}
    for key, hash_key in hash_keys.items():
        if file_hash(ROOT / selection[key]) != selection[hash_key]:
            raise ValueError("Changed collection inventory: " + key)
    candidates = read_jsonl(ROOT / selection["candidate_inventory"])
    expected = {(r["benchmark"], r["subject_id"], "training") for r in candidates}
    expected |= {(r["benchmark"], r["subject_id"], r["role"])
                 for r in read_jsonl(ROOT / selection["development_inventory"])
                 if r["role"] == "development"}
    keys = [(r["benchmark"], r["subject_id"], r["role"]) for r in rows]
    if set(keys) != expected or len(keys) != len(set(keys)):
        raise ValueError("Collection scope differs from approved inventories")
    split = json.loads((ROOT / selection["frozen_split_file"]).read_text())
    sources = {s["source_id"]: s for s in read_jsonl(catalog / "sources.jsonl")}
    # Only the ten approved subject tables, never response/item/trace tables.
    for benchmark in sorted({r["benchmark"] for r in rows}):
        path = ROOT / "payload/data" / benchmark / "formatted_tables/subjects.parquet"
        if file_hash(path) != sources["measurement-db." + benchmark + ".subjects"]["sha256"]:
            raise ValueError("Source subject table changed: " + benchmark)
        table = pd.read_parquet(path).set_index("subject_id", drop=False)
        for row in (r for r in rows if r["benchmark"] == benchmark):
            if benchmark not in split["roles"][row["role"]]:
                raise ValueError("Subject escaped its frozen role")
            if set(row["subject"]) != set(SUBJECT_FIELDS):
                raise ValueError("Subject contains forbidden fields")
            if subject_record(table.loc[row["subject_id"]]) != row["subject"]:
                raise ValueError("Published subject fields changed")


def missing_facts(rows, coverage):
    lookup = {(r["benchmark"], r["subject_id"]): r for r in rows}
    issues = []
    for record in coverage:
        row = lookup[(record["benchmark"], record["subject_id"])]
        for kind in ["model", "harness"]:
            if record[kind + "_fact_available"]:
                continue
            if kind == "harness":
                reason = "missing_supplied_harness"
                needed = "A harness identifier supplied through the published subject interface."
            elif row["benchmark"] == "ceval":
                reason = "missing_supplied_model"
                needed = "A model identifier supplied through the published subject interface; display_name cannot replace it."
            elif row["benchmark"] == "critic_discernment_game":
                reason = "private_checkpoint_unverified"
                needed = "Primary evidence linking the recorded private checkpoint to a public artifact/base model."
            else:
                reason = "ambiguous_study_alias"
                needed = "An unambiguous model/base identifier, or an explicitly scoped and reviewed harness alias resolver."
            issues.append({"benchmark": row["benchmark"], "subject_id": row["subject_id"],
                           "role": row["role"], "kind": kind, "reason": reason,
                           "supplied_subject": row["subject"],
                           "investigation": row["match_note"] if kind == "model" else row["harness_match_note"],
                           "needed_to_resolve": needed})
    return issues


def pilot_visibility(enricher, tokenizer, room):
    from paiec.task_inputs import TaskStateBuilder
    builder = TaskStateBuilder(tokenizer, room)
    split = json.loads((ROOT / "configs/head_finetuning/frozen_split_v1.json").read_text())
    output = []
    for role in ["training", "development"]:
        path = ROOT / split["artifact_directory"] / (role + ".jsonl")
        if file_hash(path) != split["artifact_fingerprints"][path.name]:
            raise ValueError("Frozen pilot inputs changed")
        counts = Counter()
        for row in read_jsonl(path):
            parsed = json.loads(row["state"])
            built = builder.build(parsed["target_system"], parsed["target_item"])
            _, audit = enricher.enrich(built.state, parsed["target_system"])
            counts["records"] += 1
            for kind in ["model", "harness"]:
                counts[kind + "_facts_available"] += fact_available(enricher, audit[kind + "_profile_id"], kind)
                counts[kind + "_facts_rendered"] += any(f["profile_id"].startswith(kind + ".") for f in audit["selected_facts"])
            counts["any_facts_rendered"] += bool(audit["selected_facts"])
            counts["no_spare_room"] += audit["skip_reason"] == "original_state_has_no_spare_room"
            counts["task_omitted"] += built.visibility["task_omitted"]
        output.append({"role": role, **counts})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="research/system_profiles/collection_v2")
    parser.add_argument("--policy", default="configs/head_finetuning/enrichment_catalog_v2.json")
    parser.add_argument("--out", default="runs/head_finetuning/system_profile_coverage_v2")
    parser.add_argument("--pilot", action="store_true", help="Inspect existing task-v2 token visibility, without inference")
    parser.add_argument("--require-full", action="store_true", help="Fail after reporting if model or harness facts are missing")
    args = parser.parse_args()
    from measure_tokens import load_tokenizer
    from paiec.laya_predictor import load_questions
    catalog, output = ROOT / args.catalog, ROOT / args.out
    if output.exists():
        raise ValueError("Choose a new output directory; past audits are preserved")
    rows = read_jsonl(catalog / "subjects.jsonl")
    validate_scope(catalog, rows)
    tokenizer = load_tokenizer("english")
    count, room = native_budget(tokenizer, load_questions()["correct"], 1024, "mlx")
    current = ProfileEnricher(ROOT / args.policy, token_count=count, state_room=room)
    old = ProfileEnricher(ROOT / "configs/head_finetuning/enrichment_v1.json", token_count=count, state_room=room)
    coverage = subject_coverage(rows, current)
    previous = subject_coverage(rows, old)
    missing = missing_facts(rows, coverage)
    result = {
        "scope": "170 training candidates plus 8 selected development systems",
        "requested_model_and_harness_coverage": 1.0,
        "full_model_and_harness_coverage_achieved": not missing,
        "previous": totals(previous), "current": totals(coverage),
        "by_role": {role: totals([r for r in coverage if r["role"] == role]) for role in ["training", "development"]},
        "missing_by_reason": dict(Counter(r["reason"] for r in missing)),
        "subject_match_scopes": dict(Counter(r["model_match_status"] for r in rows)),
        "pilot_visibility": pilot_visibility(current, tokenizer, room) if args.pilot else None,
        "catalog_sha256": current.metadata["catalog_sha256"],
        "policy_sha256": current.metadata["policy_sha256"],
        "scope_verified_against_original_subject_tables": True,
        "training_or_model_inference": False, "sealed_sets_opened": False,
        "limitations": ["A probe is a short synthetic task; it does not establish visibility on real long inputs.",
                        "All subjects having some facts is different from all model and harness identities being known.",
                        "Published capacities/capabilities do not establish enabled run settings or exact checkpoint identity.",
                        "Research eligibility remains separate from competition eligibility."],
    }
    output.mkdir(parents=True)
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    for name, records in [("coverage.jsonl", coverage), ("unresolved.jsonl", missing)]:
        (output / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    with (output / "coverage.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(coverage[0]))
        writer.writeheader()
        writer.writerows(coverage)
    html = '<!doctype html><meta charset="utf-8"><title>System profile coverage</title><style>body{font:16px system-ui;max-width:1300px;margin:32px auto;padding:16px}td,th{padding:7px;text-align:left}pre{white-space:pre-wrap;background:#f3f4f6;padding:16px}</style>'
    status = "Full model-and-harness coverage is still incomplete." if missing else "Every approved system has model and harness facts available."
    html += '<h1>System profile coverage: 178 systems</h1><p>' + status + ' Availability and actual rendering are measured separately below. Unknown identities and private variants were not guessed. No training or model inference.</p>'
    html += '<h2>Measured coverage</h2><pre>' + escape(json.dumps(result, indent=2)) + '</pre><h2>Unresolved gaps</h2>'
    html += '<table><tr><th>Role</th><th>Table</th><th>Subject</th><th>Missing</th><th>Reason</th></tr>'
    for issue in missing:
        html += '<tr>' + ''.join('<td>' + escape(str(issue[k])) + '</td>' for k in ['role', 'benchmark', 'subject_id', 'kind', 'reason']) + '</tr>'
    html += '</table><p><a href="coverage.csv">All 178 lookups</a> · <a href="unresolved.jsonl">Unresolved queue</a></p>'
    (output / "report.html").write_text(html)
    print(json.dumps(result, indent=2))
    if args.require_full and missing:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
