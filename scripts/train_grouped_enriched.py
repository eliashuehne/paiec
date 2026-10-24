#!/usr/bin/env python3
"""Prepare enriched grouped inputs or explicitly train one qualified CUDA arm."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from paiec.grouped_data import GroupedTrainingConfig, prepare_grouped
from paiec.feature_preparation import local_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/head_finetuning/grouped_enriched_v1.json")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true", help="Tokenization only; no model inference or fitting")
    action.add_argument("--train", action="store_true")
    parser.add_argument("--out")
    parser.add_argument("--download-parent", action="store_true")
    args = parser.parse_args()
    config = GroupedTrainingConfig.model_validate_json((ROOT / args.config).read_text())
    if args.prepare:
        if args.out or args.download_parent:
            parser.error("Preparation uses the recipe's fresh prepared directory and locally cached tokenizer")
        report = prepare_grouped(ROOT, config.model_dump())
    else:
        if not args.out:
            parser.error("Training requires a fresh --out")
        from paiec.grouped_training import train_grouped
        report = train_grouped(ROOT, config.model_dump(), local_path(ROOT, args.out), download=args.download_parent)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
