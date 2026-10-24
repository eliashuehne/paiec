#!/usr/bin/env python3
"""Check the exact packaged predictor on every training scope in isolation."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


from paiec.feature_preparation import local_path
from paiec.submission_qualification import qualify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--package", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    print(json.dumps(qualify(local_path(ROOT, args.prepared), local_path(ROOT, args.package), local_path(ROOT, args.out)), indent=2))


if __name__ == "__main__":
    main()
