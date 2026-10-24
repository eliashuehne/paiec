#!/usr/bin/env python3
"""Prepare or run one bounded multilingual qualification, never a full pilot."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from paiec.multilingual_recipe import MultilingualRecipe, prepare_qualification
from paiec.feature_preparation import local_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/head_finetuning/multilingual_qualification_v1.json")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--out")
    parser.add_argument("--download-parent", action="store_true")
    args = parser.parse_args()
    config = MultilingualRecipe.model_validate_json((ROOT / args.config).read_text())
    if args.prepare:
        report = prepare_qualification(ROOT, config.model_dump())
    else:
        if not args.out:
            parser.error("Qualification requires a fresh --out")
        from paiec.multilingual_qualification import qualify_multilingual
        report = qualify_multilingual(ROOT, local_path(ROOT, args.out), config.model_dump(), download=args.download_parent)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
