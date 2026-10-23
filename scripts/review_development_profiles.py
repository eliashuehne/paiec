#!/usr/bin/env python3
"""Review all development-family identities and export a separate question-only supplement.

Selection uses published subjects, item text and response linkage IDs. Response
values, reference answers and grading definitions are never loaded.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import pandas as pd
import pyarrow.parquet as pq
from paiec.compact_profiles import CompactProfiles
from paiec.competition import competition_input
from paiec.head_training import file_hash
from paiec.schema import subject_record, item_record
from paiec.split import make_split


def review(out, policy, *, render_inputs=False):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    profiles = CompactProfiles(policy)
    split_path = ROOT / "configs/head_finetuning/frozen_split_v1.json"
    split = json.loads(split_path.read_text())
    original = [json.loads(s) for s in (ROOT / "research/system_profiles/collection_v2/subjects.jsonl").read_text().splitlines()]
    original_ids = {(r["benchmark"], r["subject_id"]) for r in original if r["role"] == "development"}
    rows, manifest, inputs = [], [], []
    hashes = {str(split_path.relative_to(ROOT)): file_hash(split_path)}
    for benchmark in split["roles"]["development"]:
        folder = ROOT / "payload/data" / benchmark / "formatted_tables"
        subjects = pd.read_parquet(folder / "subjects.parquet")
        # This projection intentionally cannot load response values or traces.
        fields = pq.read_schema(folder / "response.parquet").names
        columns = ["subject_id", "item_id"] + (["interactors"] if "interactors" in fields else [])
        links = pd.read_parquet(folder / "response.parquet", columns=columns)
        items = pd.read_parquet(folder / "items.parquet", columns=["item_id", "content", "item_features"])
        items = items[items.content.fillna("").str.strip().ne("")].set_index("item_id")
        for name in ["subjects.parquet", "response.parquet", "items.parquet"]:
            path = folder / name
            hashes[str(path.relative_to(ROOT))] = file_hash(path)
        for _, source in subjects.iterrows():
            subject = subject_record(source)
            groups, match = profiles.candidates(subject)
            recorded = (benchmark, source.subject_id) in original_ids
            ids = sorted(set(links[links.subject_id.eq(source.subject_id)].item_id).intersection(items.index))
            eligible = bool(groups["model"]) and not recorded and len(ids) >= 80
            rows.append(dict(benchmark=benchmark, subject_id=source.subject_id, subject=subject,
                original_evaluation=recorded, model_facts_available=bool(groups["model"]),
                harness_facts_available=bool(groups["harness"]), distinct_linked_text_items=len(ids),
                supplement_selected=eligible, **match))
            if not eligible:
                continue
            pools = make_split(ids, benchmark=benchmark, subject_id=source.subject_id,
                               seed=split["seed"], acquisition_items=40, evaluation_items=120)
            recorded_links = links[links.subject_id.eq(source.subject_id)].drop_duplicates("item_id").set_index("item_id")
            for role, selected in [("acquisition", pools.acquisition), ("evaluation", pools.evaluation)]:
                for item_id in selected:
                    manifest.append(dict(benchmark=benchmark, subject_id=source.subject_id, item_id=item_id,
                                         role=role, seed=split["seed"]))
                    # This is a review input, not a supervised row: no expected label.
                    item = item_record(items.loc[item_id], benchmark)
                    if "interactors" in recorded_links:
                        item["interactors"] = recorded_links.loc[item_id, "interactors"]
                    projected = competition_input(subject, item)
                    inputs.append(dict(benchmark=benchmark, subject_id=source.subject_id, item_id=item_id,
                        role=role, subject=projected[0], item=projected[1]))
    for name, values in [("subjects.jsonl", rows), ("supplement_inputs.jsonl", inputs), ("supplement_manifest.jsonl", manifest)]:
        (out / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in values))
    rendered_counts = Counter()
    if render_inputs:
        from measure_tokens import load_tokenizer
        from paiec.compact_inputs import CompactStateBuilder
        from paiec.laya_predictor import load_questions
        from paiec.profile_enrichment import native_budget
        tokenizer = load_tokenizer("english")
        question = load_questions()
        _, room = native_budget(tokenizer, question["correct"], 2048, "mlx")
        builder = CompactStateBuilder(tokenizer, room, profiles)
        with (out / "compact_inputs.jsonl").open("w") as stream:
            for record in inputs:
                pair = builder.build(record["subject"], record["item"])
                if pair.visibility["options_omitted"]:
                    raise ValueError("Supplement option-preservation check failed")
                rendered_counts[record["role"] + "_inputs"] += 1
                for kind in ["model", "harness"]:
                    rendered_counts[record["role"] + "_" + kind + "_facts_delivered"] += pair.enrichment["delivered"][kind]
                stream.write(json.dumps({**record, "questions": question, "baseline": pair.baseline,
                    "enriched": pair.enriched, "visibility": pair.visibility, "enrichment": pair.enrichment},
                    ensure_ascii=False) + "\n")
    summary = dict(development_subjects=len(rows), model_fact_subjects=sum(r["model_facts_available"] for r in rows),
        missing_model_subjects=sum(not r["model_facts_available"] for r in rows),
        extra_identifiable_subjects=sum(r["supplement_selected"] for r in rows),
        original_subjects_preserved=len(original_ids),
        supplement_inputs_by_pool=dict(Counter(r["role"] for r in manifest)),
        selection="All additional identifiable systems in frozen development families with >=80 linked text items; metadata only.",
        source_sha256=hashes, profile_policy=profiles.metadata,
        response_values_loaded=False, reference_answers_loaded=False, new_scores=False,
        original_evaluation_modified=False, sealed_sets_opened=False,
        compact_input_visibility=dict(rendered_counts) if render_inputs else None,
        binary_outcome_eligibility_checked=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="runs/head_finetuning/development_identity_review_v3")
    parser.add_argument("--policy", default="configs/head_finetuning/compact_profiles_v3.json")
    parser.add_argument("--render-inputs", action="store_true", help="Tokenize the matched supplement inputs without inference")
    args = parser.parse_args()
    review(ROOT / args.out, ROOT / args.policy, render_inputs=args.render_inputs)
