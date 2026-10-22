#!/usr/bin/env python3
"""Matched development-only scoring of original Laya and the smoke-tuned head."""
import argparse
from dataclasses import asdict
from html import escape
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
import pandas as pd
from paiec.compare import RunScores, compare_to, by_benchmark, alc_from_curves, pair_curves
from paiec.head_evaluation import score_frozen_development
from paiec.head_training import file_hash
from paiec.laya_runtime import create_runtime
from paiec.tracking import ExperimentTracker


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', required=True)
    ap.add_argument('--checkpoint', default='runs/head_finetuning/smoke_v1/checkpoint')
    ap.add_argument('--raw-control-reference', default=None, help='reuse completed matched comparison and add original temperature-1 control')
    args = ap.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    frozen_path = ROOT/'configs/head_finetuning/frozen_split_v1.json'
    split = json.loads(frozen_path.read_text())
    data = ROOT/split['artifact_directory']
    for name in ['development.jsonl','provenance.parquet','evaluation_manifest.csv']:
        if file_hash(data/name) != split['artifact_fingerprints'][name]:
            raise ValueError(f'Frozen artifact changed: {name}')
    manifest = pd.read_parquet(data/'provenance.parquet')
    manifest = manifest[(manifest.role=='development') & manifest.exported].sort_values('jsonl_line').reset_index(drop=True)
    rows = [json.loads(line) for line in (data/'development.jsonl').read_text().splitlines()]
    if manifest.jsonl_line.tolist() != list(range(len(rows))):
        raise ValueError('Response provenance does not align with export')
    question = split['input']['question_definition']
    if any(row['questions'] != question for row in rows):
        raise ValueError('Frozen questions differ')
    import hashlib
    if any(hashlib.sha256(row['state'].encode()).hexdigest() != expected
           for row,expected in zip(rows,manifest.state_sha256)):
        raise ValueError('State hashes differ from provenance')
    labels = [int(row['expected']['correct']) for row in rows]
    states = list(dict.fromkeys(row['state'] for row in rows))
    state_index = {state:i for i,state in enumerate(states)}
    manifest.to_parquet(out/'response_manifest.parquet', index=False)
    config = dict(split_sha256=file_hash(frozen_path), dataset_revision=split['dataset_revision'],
                  development_sha256=file_hash(data/'development.jsonl'),
                  question=question, max_len=512, head_max_len=192, backend='mlx', dtype='float32',
                  batch_size=8, use_labels=False, checkpoint_path=str(Path(args.checkpoint).resolve()),
                  n_eval_inputs=len(manifest[['benchmark','subject_id','item_id']].drop_duplicates()),
                  n_responses=len(rows), n_unique_states=len(states), n_pairs=8,
                  confirmation_scored=False, calibration_fitted=False,
                  note='Flat budget curves; pretrained uses shipped calibration, tuned uses raw temperature 1')
    (out/'config.json').write_text(json.dumps(config,indent=2))
    tracker = ExperimentTracker(ROOT,experiment='PAIEC Head Fine-tuning')
    results, receipts, summary = {}, {}, []
    variants = ['constant','pretrained','head_smoke']
    if args.raw_control_reference:
        reference = Path(args.raw_control_reference).resolve()
        prior_config = json.loads((reference/'config.json').read_text())
        for key in ['split_sha256','development_sha256','question','dtype','max_len','checkpoint_path']:
            if prior_config[key] != config[key]:
                raise ValueError(f'Raw control reference differs: {key}')
        config['reference_directory'] = str(reference)
        config['note'] += '; additional original temperature-1 control; no calibration fitting'
        for name in variants:
            original = reference/name
            details = json.loads((original/'details.json').read_text())
            predictions = pd.read_csv(original/'predictions.csv',dtype={'benchmark':str,'subject_id':str,'item_id':str})
            results[name] = RunScores(name,str(original),alc_from_curves(pair_curves(predictions)),predictions)
            summary.append({k:details[k] for k in ['name','brier_alc','forward_items','truncated_items','mean_probability','run_id']})
            receipts[name] = details['run_id']
        variants = ['pretrained_raw']
    (out/'config.json').write_text(json.dumps(config,indent=2))
    for name in variants:
        folder = out/name
        folder.mkdir()
        with tracker.run('development / '+name+' / FP32',config,
                         dict(predictor=name,n_eval_inputs=config['n_eval_inputs'],n_pairs=8,
                              dataset_revision=config['dataset_revision'],development_sha256=config['development_sha256'],
                              backend='none' if name=='constant' else 'mlx', dtype='float32',
                              role='development',run_type='head_development_evaluation'),
                         dict(run_type='head_development_evaluation',confirmation_scored='false')) as tracked:
            start = time.perf_counter()
            runtime = None
            if name == 'constant':
                probabilities = [.5]*len(rows)
                forwards = truncated = 0
                runtime_metadata = {'backend':'none','predictor':'constant'}
            else:
                runtime = create_runtime(backend='mlx',model='english',device='auto',dtype='float32',
                    revision=split['input']['checkpoint_revision'],batch_size=8,
                    checkpoint_path=str(Path(args.checkpoint).resolve()) if name=='head_smoke' else None)
                runtime_metadata = runtime.metadata
                if name == 'pretrained_raw':
                    override = {'temperature':[1.,1.,1.], 'temperature_by_options':{}}
                    override_path = folder/'raw_temperature.json'
                    override_path.write_text(json.dumps(override,indent=2))
                    runtime.agent.load_calibration(str(override_path))
                    runtime_metadata['calibration_override'] = override
                    runtime_metadata['calibration_override_sha256'] = file_hash(override_path)
                if tracked:
                    tracked.json(runtime_metadata,'runtime.json')
                (folder/'runtime.json').write_text(json.dumps(runtime_metadata,indent=2))
                outputs = []
                inference_start = time.perf_counter()
                # Persist each batch as audit output. This is not automatic resume support.
                with (folder/'forward_outputs.jsonl').open('w') as handle:
                    for first in range(0,len(states),8):
                        requests = [dict(state=state,questions=question,model='english',max_len=512)
                                    for state in states[first:first+8]]
                        batch = runtime.predict_batch(requests,8)
                        if len(batch) != len(requests):
                            raise ValueError('Wrong inference batch length')
                        for offset, value in enumerate(batch):
                            handle.write(json.dumps(dict(state_index=first+offset,output=value))+'\n')
                        handle.flush()
                        outputs.extend(batch)
                        if (first//8+1)%15==0 or len(outputs)==len(states):
                            print(f'{name}: {len(outputs)}/{len(states)} inputs',flush=True)
                        runtime.trim_cache()
                runtime.synchronize()
                runtime_metadata['inference_seconds'] = time.perf_counter()-inference_start
                values = [float(output['answers']['correct']['noul']) for output in outputs]
                probabilities = [values[state_index[row['state']]] for row in rows]
                forwards = len(outputs)
                truncated = sum(bool(output['usage']['truncated']) for output in outputs)
                runtime.close()
                runtime = None
            seconds = time.perf_counter()-start
            result,predictions = score_frozen_development(manifest,labels,probabilities)
            predictions.to_csv(folder/'predictions.csv',index=False)
            result.frame.to_csv(folder/'scores.csv',index=False)
            details = dict(name=name,brier_alc=result.brier_alc,seconds_including_initialization=seconds,
                           forward_items=forwards,truncated_items=truncated,
                           run_id=tracked.run_id if tracked else None,
                           mean_probability=float(np.mean(probabilities)),
                           mean_observed_outcome=float(np.mean(labels)),
                           calibration='shipped upstream' if name=='pretrained' else 'none',
                           runtime=runtime_metadata)
            (folder/'details.json').write_text(json.dumps(details,indent=2))
            if tracked:
                tracked.result(result,seconds,predictions)
                tracked.json(details,'details.json')
                tracked.json(split,'frozen_split.json')
                tracked.json(question,'question_definition.json')
                tracked.frame(manifest,'response_manifest.csv')
                tracked.dataset(manifest[['benchmark','subject_id','item_id','response_id','row_id']],
                                'head-frozen-development',context='development')
                for key,value in dict(forward_items=forwards,truncated_items=truncated,
                                      n_eval_inputs=config['n_eval_inputs'],n_eval_responses=len(rows)).items():
                    tracked.client.log_metric(tracked.run_id,key,value)
                if name=='head_smoke':
                    tracked.json(json.loads((Path(args.checkpoint)/'paiec_training.json').read_text()),'training_provenance.json')
            summary.append({k:details[k] for k in ['name','brier_alc','forward_items','truncated_items','mean_probability','run_id']})
            receipts[name] = details['run_id']
            results[name] = RunScores(name,str(folder),result.pair_alc,predictions)
            print(name,result.brier_alc,flush=True)
    comparison = asdict(compare_to(results['pretrained'],results['head_smoke']))
    raw_comparison = (asdict(compare_to(results['pretrained_raw'],results['head_smoke']))
                      if 'pretrained_raw' in results else None)
    benchmarks = by_benchmark(results)
    benchmarks.to_csv(out/'by_benchmark.csv')
    pairs = pd.DataFrame({name:result.curves for name,result in results.items()})
    pairs.to_csv(out/'by_pair.csv')
    summary_frame = pd.DataFrame(summary)
    summary_frame.to_csv(out/'summary.csv',index=False)
    report = dict(comparison=comparison, raw_head_comparison=raw_comparison, config=config, run_ids=receipts,
                  by_benchmark=benchmarks.reset_index().to_dict('records'),
                  caveat='Eight pairs but only two benchmark families; pair bootstrap does not establish cross-family generalization')
    (out/'comparison.json').write_text(json.dumps(report,indent=2))
    html = '<!doctype html><meta charset="utf-8"><title>Head development check</title><style>body{font:16px system-ui;max-width:1100px;margin:40px auto}td,th{padding:9px;border-bottom:1px solid #ddd}table{border-collapse:collapse}</style>'
    html += '<h1>Head-only development check</h1><p>Lower Brier ALC is better. 960 system–item inputs, 963 response outcomes, 8 pairs and 2 families. Same frozen development sample; confirmation untouched. No revealed examples: every budget has the same score.</p>'
    html += summary_frame.to_html(index=False,float_format=lambda x:f'{x:.6f}')
    html += '<h2>By benchmark</h2>'+benchmarks.to_html(float_format=lambda x:f'{x:.6f}')
    html += '<h2>By system–benchmark pair</h2>'+pairs.to_html(float_format=lambda x:f'{x:.6f}')
    html += '<p>Original Laya uses its shipped temperature; tuned head uses temperature 1. This is an end-to-end pipeline comparison and does not isolate weight changes from calibration. Eight pairs cover only two benchmark families. No broad generalization claim follows.</p>'
    html += '<pre>'+escape(json.dumps(dict(shipped_comparison=comparison,raw_head_comparison=raw_comparison),indent=2))+'</pre>'
    (out/'report.html').write_text(html)
    # Reference runs retain their historical comparison artifacts unchanged.
    for variant in variants:
        run_id = receipts[variant]
        if run_id:
            for name in ['comparison.json','by_benchmark.csv','by_pair.csv','summary.csv','report.html']:
                tracker.client.log_artifact(run_id,str(out/name),'comparison')
    print(summary_frame.to_string(index=False),flush=True)
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':
    main()
