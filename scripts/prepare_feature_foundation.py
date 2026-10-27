#!/usr/bin/env python3
"""Plan or build outcome-independent features for the full downloaded release."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from paiec.feature_foundation.contracts import FeatureFoundationConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/features/foundation_v1.json")
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config = FeatureFoundationConfig.model_validate_json(args.config.read_text())
    if not args.execute:
        print(json.dumps({"execution_started": False, "source_root": str(args.source_root.resolve()),
                          "out": str(args.out.resolve()), "config": config.model_dump()}, indent=2))
        return
    from paiec.feature_foundation.build import build_bundle
    bundle = build_bundle(config, source_root=args.source_root, out=args.out,
                          progress=lambda row: print(json.dumps(row), file=sys.stderr, flush=True))
    print(json.dumps({"directory": str(bundle.directory), "population": bundle.manifest["population"],
                      "review_complete": bundle.manifest["review_complete"]}, indent=2))


if __name__ == "__main__":
    main()
