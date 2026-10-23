#!/usr/bin/env python3
"""Audit training-role breadth and crossed system/task observations. No fitting."""
import argparse
from html import escape
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
import pandas as pd
from paiec.head_training import file_hash
from paiec.schema import benchmark_dir, subject_record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",default="runs/head_finetuning/capability_data_audit_v1")
    args=parser.parse_args()
    out=ROOT/args.out
    out.mkdir(parents=True,exist_ok=False)
    split_path=ROOT/"configs/head_finetuning/frozen_split_v1.json"
    split=json.loads(split_path.read_text())
    config_path=ROOT/split["source_config"]
    if file_hash(config_path)!=split["source_config_sha256"]:
        raise ValueError("Frozen allocation source changed")
    reviews=json.loads(config_path.read_text())["reviews"]
    provenance_path=ROOT/split["artifact_directory"]/"provenance.parquet"
    selected=pd.read_parquet(provenance_path)
    selected=selected[selected.role.eq("training") & selected.exported]
    training=selected.groupby(["benchmark","item_id"]).subject_id.nunique()
    summary=[]; candidates=[]; sources=[split_path,config_path,provenance_path,Path(__file__).resolve()]
    for benchmark in split["roles"]["training"]:
        folder=benchmark_dir(ROOT/"payload/data",benchmark)
        sources.extend(folder/name for name in ("subjects.parquet","items.parquet","response.parquet"))
        subjects=pd.read_parquet(folder/"subjects.parquet").set_index("subject_id",drop=False)
        items=pd.read_parquet(folder/"items.parquet",columns=["item_id","content","item_features"])
        items=items[items.content.fillna("").astype(str).str.strip().ne("")]
        required=reviews[benchmark].get("required_feature")
        if required:
            items=items[items.item_features.fillna("").astype(str).str.contains(required,regex=False)]
        responses=pd.read_parquet(folder/"response.parquet",columns=["subject_id","item_id","response"])
        observed=responses[responses.response.isin([0,1]) & responses.item_id.isin(items.item_id)
                           & responses.subject_id.isin(subjects.index)]
        counts=observed.groupby("subject_id").item_id.nunique()
        eligible=counts[counts.ge(80)].index
        paired=observed[observed.subject_id.isin(eligible)]
        per_item=paired.groupby("item_id").subject_id.nunique()
        signatures=[]
        for subject_id in eligible:
            subject=subject_record(subjects.loc[subject_id])
            signature=json.dumps(subject,sort_keys=True,ensure_ascii=False)
            signatures.append(signature)
            candidates.append(dict(benchmark=benchmark,subject_id=subject_id,
                distinct_items=int(counts[subject_id]),normalized_name_present=bool(subject["normalized_name"]),
                supplied_profile=signature,selection="candidate_not_automatically_admitted"))
        old=selected[selected.benchmark.eq(benchmark)]
        summary.append(dict(benchmark=benchmark,training_family=reviews[benchmark]["family"],
            current_sample_pairs=old.subject_id.nunique(),source_subject_rows=len(subjects),
            candidate_pairs=len(eligible),candidate_visible_profiles=len(set(signatures)),
            candidate_canonical_names=subjects.loc[eligible].normalized_name.dropna().nunique(),
            binary_response_rows=len(paired),distinct_system_items=len(paired.drop_duplicates(["subject_id","item_id"])),
            distinct_items=per_item.size,multi_system_items=int(per_item.ge(2).sum()),
            current_sample_rows=len(old),review_required=required or "benchmark-level review only"))
    table=pd.DataFrame(summary)
    table.to_csv(out/"training_breadth.csv",index=False)
    pd.DataFrame(candidates).to_json(out/"candidate_profiles.jsonl",orient="records",lines=True,force_ascii=False)
    names={}
    for candidate in candidates:
        name=json.loads(candidate["supplied_profile"])["normalized_name"]
        if name:names.setdefault(name,set()).add(candidate["benchmark"])
    manifest=dict(training=False,model_inference=False,
        calibration_or_confirmation_outcomes_opened=False,new_roles_assigned=False,
        current_sample_pairs=int(table.current_sample_pairs.sum()),
        candidate_pairs=int(table.candidate_pairs.sum()),
        candidate_distinct_canonical_names=len(names),
        canonical_names_in_multiple_training_benchmarks={n:sorted(b) for n,b in names.items() if len(b)>1},
        current_items_by_number_of_sampled_systems={int(k):int(v) for k,v in training.value_counts().sort_index().items()},
        caution="Candidates require per-subject/run/input review. Configurations are not independent LLMs; trials are not new tasks. No expansion is admitted by this audit.",
        source_sha256={str(p.relative_to(ROOT)):file_hash(p) for p in sources})
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    html='<!doctype html><meta charset="utf-8"><title>Capability-training data audit</title><style>body{font:16px system-ui;max-width:1400px;margin:30px auto;padding:20px}td,th{padding:7px}pre{white-space:pre-wrap}</style>'
    html+='<h1>Capability-training data audit</h1><p>Training-role tables only. No fitting, embedding inference, GPU work, new role assignment or automatic subject admission. Candidate breadth is not a count of independent model families.</p>'
    html+=table.to_html(index=False)+'<pre>'+escape(json.dumps(manifest,indent=2))+'</pre>'
    (out/"report.html").write_text(html)
    print(table.to_string(index=False))


if __name__=="__main__":main()
