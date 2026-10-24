#!/usr/bin/env python3
"""Validate prepared Laya inputs with cached native tokenizer; no model inference."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from paiec.collected_input_audit import audit_native_inputs
from paiec.feature_preparation import local_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", choices=["english", "multilingual"], default="english")
    parser.add_argument("--max-len", type=int)
    parser.add_argument("--head-max-len", type=int)
    args = parser.parse_args()
    print(json.dumps(audit_native_inputs(local_path(ROOT, args.prepared), local_path(ROOT, args.out),
        model=args.model, max_len=args.max_len, head_max_len=args.head_max_len), indent=2))
