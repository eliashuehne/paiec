#!/usr/bin/env python3
"""Export supplied-only foundation features for an existing lift sample."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--sample', type=Path, default=ROOT / 'runs/lift/sample.parquet')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({'execution_started': False, 'policy': 'supplied_only',
                          'bundle': str(args.bundle.resolve()), 'sample': str(args.sample.resolve()),
                          'out': str(args.out.resolve())}, indent=2))
        return
    from paiec.feature_foundation.lift import export_lift_features
    print(json.dumps(export_lift_features(args.bundle, args.sample, args.out), indent=2))


if __name__ == '__main__':
    main()
