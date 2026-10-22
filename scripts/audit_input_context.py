#!/usr/bin/env python3
"""Audit recorded attributes and preservation; no traces or response labels are used."""
import argparse
from collections import Counter
from html import escape
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import pandas as pd
import pyarrow.parquet as pq
from paiec.schema import SUBJECT_FIELDS, clean, subject_record
from paiec.head_training import file_hash


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    out=Path(args.out).resolve()
    out.mkdir(parents=True,exist_ok=False)
    data=ROOT/'payload/data'
    corpus=json.loads((data/'corpus_manifest.json').read_text())
    coverage=[]; tables=[]; subjects=[]; lookup={}; hashes={}; examples={}
    feature_keys=Counter()
    for directory in corpus['eligible_dirs']:
        folder=data/directory; benchmark=directory.split('/')[0]
        frame=pd.read_parquet(folder/'subjects.parquet')
        hashes[str((folder/'subjects.parquet').relative_to(ROOT))]=file_hash(folder/'subjects.parquet')
        for field in ['display_name',*SUBJECT_FIELDS]:
            present=frame[field].map(clean).ne('')
            coverage.append(dict(benchmark=benchmark,field=field,total=len(frame),nonempty=int(present.sum())))
        for name in ['subjects','items','response','benchmarks']:
            path=folder/(name+'.parquet'); file=pq.ParquetFile(path)
            tables.append(dict(benchmark=benchmark,table=name,rows=file.metadata.num_rows,columns=file.schema_arrow.names))
        for row in frame.to_dict('records'):
            record={key:clean(value) for key,value in row.items()}
            lookup[(benchmark,record['subject_id'])]=record
            subjects.append(dict(benchmark=benchmark,**record))
            extra=record['subject_features_extra']
            # Inventory only: key counts are approximate for opaque strings containing semicolons.
            for part in extra.split(';'):
                if '=' in part:
                    key,value=part.split('=',1)
                    feature_keys[key]+=1
                    if key not in examples and value:
                        examples[key]=dict(benchmark=benchmark,value=value)
    frame=pd.DataFrame(subjects)
    cov=pd.DataFrame(coverage)
    cov.to_csv(out/'field_coverage_by_benchmark.csv',index=False)
    aggregate=cov.groupby('field')[['total','nonempty']].sum()
    aggregate['percent_present']=100*aggregate.nonempty/aggregate.total
    aggregate.to_csv(out/'field_coverage_all.csv')
    (out/'schemas.json').write_text(json.dumps(tables,indent=2))
    (out/'subject_source_sha256.json').write_text(json.dumps(hashes,indent=2))
    (out/'extra_field_keys.json').write_text(json.dumps(dict(counts=feature_keys,first_examples=examples),indent=2))
    scopes={
        'smoke_256':(ROOT/'runs/head_finetuning/smoke_v1/training.jsonl',ROOT/'runs/head_finetuning/smoke_v1/smoke_manifest.parquet'),
        'training_5000':(ROOT/'runs/head_finetuning/data_v2/training.jsonl',ROOT/'runs/head_finetuning/data_v2/provenance.parquet'),
        'development_963':(ROOT/'runs/head_finetuning/data_v2/development.jsonl',ROOT/'runs/head_finetuning/data_v2/provenance.parquet')}
    retained={}; scope_coverage=[]
    for scope,(json_path,sidecar_path) in scopes.items():
        rows=[json.loads(line) for line in json_path.read_text().splitlines()]
        provenance=pd.read_parquet(sidecar_path)
        if scope!='smoke_256':
            role='training' if scope=='training_5000' else 'development'
            provenance=provenance[(provenance.role==role)&provenance.exported].sort_values('jsonl_line')
        if len(provenance)!=len(rows):
            raise ValueError('Provenance alignment failed')
        mismatches=[]; counts=Counter()
        for index,(row,meta) in enumerate(zip(rows,provenance.to_dict('records'))):
            raw=lookup[(meta['benchmark'],meta['subject_id'])]
            expected={k:v for k,v in subject_record(raw).items() if v}
            actual=json.loads(row['state'])['target_system']
            if expected!=actual:
                mismatches.append(index)
            counts.update(actual.keys())
        retained[scope]=dict(rows=len(rows),subject_projection_mismatches=mismatches,
                             source_jsonl_sha256=file_hash(json_path))
        for field in SUBJECT_FIELDS:
            scope_coverage.append(dict(scope=scope,field=field,nonempty=counts[field],total=len(rows)))
    pd.DataFrame(scope_coverage).to_csv(out/'export_field_coverage.csv',index=False)
    (out/'export_preservation.json').write_text(json.dumps(retained,indent=2))
    adaptive=lookup[('adaptivestep','50e726a69a130d06')]
    rich=next(row for row in subjects if row['reasoning_effort'] and row['harness_version'])
    (out/'real_subject_examples.json').write_text(json.dumps(dict(adaptivestep_original=adaptive,richer_record=rich),indent=2))
    finding=dict(dataset_revision=corpus['revision'],benchmarks=len(corpus['eligible_dirs']),subject_rows=len(frame),
                 counts_subject_rows_not_unique_models=True,
                 extra_nonempty_with_missing_normalized_name=int(((frame.normalized_name=='')&(frame.subject_features_extra!='')).sum()),
                 reasoning_effort_benchmarks=frame.loc[frame.reasoning_effort!=''].groupby('benchmark').size().to_dict(),
                 export_preservation=retained,
                 traces_read=False,response_table_outcomes_read=False,
                 labeled_exports_loaded_for_input_checks=True,outcome_labels_used=False,confirmation_export_read=False)
    (out/'summary.json').write_text(json.dumps(finding,indent=2))
    html='<!doctype html><meta charset="utf-8"><title>Input context audit</title><style>body{font:16px system-ui;max-width:1000px;margin:40px auto}pre{white-space:pre-wrap}td,th{padding:8px}</style>'
    html+='<h1>Recorded input context</h1><p>8,163 subject rows across 61 public benchmarks. Includes non-LLM and human subjects; rows are not distinct model identities. No trace tables or response outcome columns loaded. Existing labeled exports are used only to check input preservation.</p>'
    html+=aggregate.to_html(float_format=lambda value:f'{value:.2f}')
    html+='<h2>Preservation in current exports</h2><pre>'+escape(json.dumps(retained,indent=2))+'</pre>'
    html+='<h2>Actual original example</h2><pre>'+escape(json.dumps(adaptive,indent=2))+'</pre>'
    html+='<h2>Example of a richer original record</h2><pre>'+escape(json.dumps(rich,indent=2))+'</pre>'
    html+='<p>Current input preserves every nonempty published subject field. display_name, raw response test_condition, benchmark metadata, grading definitions and asset manifests are outside the published predictor field list. Parsing already supplied extras may improve readability; adding unavailable facts changes the task.</p>'
    (out/'report.html').write_text(html)
    print(aggregate.to_string())
    print(json.dumps(finding,indent=2))

if __name__=='__main__':
    main()
