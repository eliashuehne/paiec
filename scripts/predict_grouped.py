#!/usr/bin/env python3
"""Explicit local-checkpoint grouped prediction, without revealing any outcomes.

Each JSONL request has exactly subject and item dictionaries containing the
published string fields. Each result contains a probability, in request order.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from paiec.grouped_predictor import make_grouped_laya_predictor
from paiec.feature_preparation import read_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    args = parser.parse_args()
    output = Path(args.out)
    if output.exists():
        parser.error("Use a fresh output path")
    predictor = make_grouped_laya_predictor(checkpoint_path=args.checkpoint, device=args.device, dtype=args.dtype)
    try:
        with output.open("x") as stream:
            for row in read_jsonl(args.input):
                if set(row) != {"subject", "item"}:
                    raise ValueError("Grouped requests contain only subject and item, with no labels")
                stream.write(json.dumps({"probability": predictor(row["subject"], row["item"])}) + "\n")
    finally:
        predictor.close()


if __name__ == "__main__":
    main()
