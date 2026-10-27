#!/usr/bin/env python3
"""Validate a local feature bundle without the original corpus or model packages."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from paiec.feature_foundation.build import validate_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--require-review", action="store_true")
    args = parser.parse_args()
    print(json.dumps(validate_bundle(args.bundle, require_review=args.require_review), indent=2))


if __name__ == "__main__":
    main()
