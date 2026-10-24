#!/usr/bin/env python3
"""Separate original-development preparation, inference and saved-score reporting."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from paiec.grouped_evaluation import GroupedEvaluationConfig, prepare_development, predict_development, score_development
from paiec.feature_preparation import local_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["prepare", "predict", "score"], required=True)
    parser.add_argument("--config")
    parser.add_argument("--out", required=True)
    parser.add_argument("--predictions")
    args = parser.parse_args()
    out = local_path(ROOT, args.out)
    if args.stage == "prepare":
        report = prepare_development(ROOT, out)
    else:
        if not args.config:
            parser.error("Prediction/scoring require a checkpoint-bound --config")
        config = GroupedEvaluationConfig.model_validate_json((ROOT / args.config).read_text())
        if args.stage == "predict":
            report = predict_development(ROOT, config.model_dump(), out)
        else:
            if not args.predictions:
                parser.error("Scoring requires --predictions")
            report = score_development(ROOT / config.prepared, local_path(ROOT, args.predictions), out,
                                       checkpoint_manifest_sha256=config.checkpoint_manifest_sha256)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
