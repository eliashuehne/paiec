#!/usr/bin/env python3
"""Build a reproducible evidence-policy ZIP for local review; never uploads."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from paiec.feature_preparation import local_path
from paiec.submission_package import build_evidence_submission


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    print(json.dumps(build_evidence_submission(local_path(ROOT, args.selection), local_path(ROOT, args.out)), indent=2))


if __name__ == "__main__":
    main()
