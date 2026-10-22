#!/usr/bin/env python3
"""Qualify explicit trained-checkpoint inference on saved training-only fixtures."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from paiec.laya_runtime import create_runtime

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--run', required=True)
ap.add_argument('--backend', choices=['torch', 'mlx'], required=True)
ap.add_argument('--dtype', default='auto')
args = ap.parse_args()
folder = Path(args.run).resolve()
report = json.loads((folder/'report.json').read_text())
fixtures = json.loads((folder/'reload_fixtures.json').read_text())
# Add a truncation fixture without opening any held-out data.
long = dict(fixtures[0], state=fixtures[0]['state'] + '\n' + 'extra context ' * 900)
fixtures.append(long)
runtime = create_runtime(backend=args.backend, model='english', device='auto', dtype=args.dtype,
                         revision=None, batch_size=1, checkpoint_path=str(folder/'checkpoint'))
outputs = [runtime.predict_batch([dict(state=row['state'], questions=row['questions'],
                                       model='english', max_len=512)], 1)[0] for row in fixtures]
probabilities = [output['answers']['correct']['noul'] for output in outputs]
errors = [abs(value - expected[1]) for value, expected in zip(probabilities, report['fixture_probabilities'])]
result = dict(runtime=runtime.metadata, probabilities=probabilities, outputs=outputs,
              max_native_vs_direct_difference=max(errors), scope='two training rows and synthetic long input')
if args.backend == 'torch':
    assert max(errors) <= .000051, errors  # Native API rounds to four decimal places.
else:
    reference_path = folder/'runtime_torch_auto.json'
    reference = json.loads(reference_path.read_text())
    result['max_vs_torch_difference'] = max(abs(a-b) for a,b in zip(probabilities, reference['probabilities']))
    if args.dtype == 'float32':
        assert result['max_vs_torch_difference'] <= .0002, result
path = folder/f'runtime_{args.backend}_{args.dtype}.json'
if path.exists():
    raise FileExistsError(path)
path.write_text(json.dumps(result, indent=2))
runtime.close()
print(json.dumps({k:v for k,v in result.items() if k not in ['outputs','runtime']}, indent=2))
