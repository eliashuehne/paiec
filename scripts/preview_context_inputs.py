#!/usr/bin/env python3
"""Versioned input export and native token audit; no model weights or inference."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import pandas as pd
from paiec.laya_predictor import build_state
from paiec.input_audit import audit_state
from paiec.head_training import file_hash
from measure_tokens import load_tokenizer


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',default='runs/head_finetuning/smoke_v1/training.jsonl')
    ap.add_argument('--out',required=True)
    args=ap.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    source=Path(args.source)
    rows=[json.loads(line) for line in source.read_text().splitlines()]
    tok=load_tokenizer('english')
    audits=[]
    with (out/'inputs_context_v1.jsonl').open('w') as export:
        for index,row in enumerate(rows):
            original=json.loads(row['state'])
            if set(original)-{'target_system','target_item'}:
                raise ValueError('This export supports zero-shot states only')
            new=build_state(original['target_system'],original['target_item'],[],state_format='context_v1')
            # Clone labels/schema unchanged, never render expected values into the state.
            export.write(json.dumps(dict(row,state=new),ensure_ascii=False)+'\n')
            for fmt,state in [('json',row['state']),('context_v1',new)]:
                audit=audit_state(tok,state,row['questions']['correct'],state_format=fmt)
                audits.append(dict(line=index+1,state_format=fmt,state_sha256=hashlib.sha256(state.encode()).hexdigest(),
                                   state_tokens=audit['state_tokens'],state_tokens_used=audit['state_tokens_used'],
                                   truncated=audit['truncated'],task_complete=audit['target_task']['complete'],
                                   system_complete=audit['target_system']['complete']))
            if index<3:
                (out/f'example_{index+1}.txt').write_text(new)
    frame=pd.DataFrame(audits);frame.to_csv(out/'token_audit.csv',index=False)
    summary=frame.groupby('state_format').agg(rows=('line','size'),truncated=('truncated','sum'),
                   complete_tasks=('task_complete','sum'),complete_systems=('system_complete','sum'),
                   median_state_tokens=('state_tokens','median'))
    summary.to_csv(out/'summary.csv')
    meta=dict(source=str(source.resolve()),source_sha256=file_hash(source),state_format='context_v1',
              export_sha256=file_hash(out/'inputs_context_v1.jsonl'),max_len=512,head_max_len=192,
              rows=len(rows),model_inference=False,training_performed=False,
              note='New input export only. Original provenance token hashes apply to original states; do not reuse them for these states.')
    from importlib.metadata import version
    from paiec.laya_runtime import DEFAULT_CHECKPOINT_REVISION
    meta.update(tokenizer_revision=DEFAULT_CHECKPOINT_REVISION,
                packages={key:version(key) for key in ['laya-mlx','tokenizers']},
                renderer_source_sha256=file_hash(ROOT/'src/paiec/state_rendering.py'))
    (out/'manifest.json').write_text(json.dumps(meta,indent=2))
    print(summary.to_string())

if __name__=='__main__':
    main()
