#!/usr/bin/env python3
"""Export annotation requests or import local responses; no network calls."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from paiec.feature_preparation import local_path, read_jsonl, write_json
from paiec.task_annotation import export_openai_batch, import_responses, validate_plan
from paiec.task_annotation_batch import import_openai_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["status", "export-openai", "import-openai", "import-normalized"])
    parser.add_argument("--plan", required=True, help="Repository-relative prepared annotation plan")
    parser.add_argument("--out", help="Fresh output under runs/")
    parser.add_argument("--task-keys", help="JSON list of explicitly selected task hashes")
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--responses", nargs="+", help="Downloaded output/error files, or normalized responses")
    parser.add_argument("--batch", help="Exact exported input JSONL")
    parser.add_argument("--batch-sha256", help="Hash recorded before submission")
    parser.add_argument("--producer", help="Provenance for a normalized response import")
    args = parser.parse_args()
    plan = local_path(ROOT, args.plan)
    if args.action == "status":
        report, _ = validate_plan(plan)
    else:
        if not args.out:
            parser.error("--out is required")
        out = local_path(ROOT, args.out)
        if args.action == "export-openai":
            if not args.task_keys:
                parser.error("--task-keys is required; full-corpus submission is never implicit")
            keys = json.loads(local_path(ROOT, args.task_keys).read_text())
            if not isinstance(keys, list) or any(not isinstance(k, str) for k in keys):
                parser.error("--task-keys must contain a JSON list of task hashes")
            out.mkdir(parents=True, exist_ok=False)
            report = export_openai_batch(plan, out / "batch_input.jsonl", task_keys=keys,
                                        model=args.model, max_output_tokens=args.max_output_tokens)
            write_json(out / "export_receipt.json", report)
        else:
            if not args.responses:
                parser.error("--responses is required")
            paths = [local_path(ROOT, p) for p in args.responses]
            if args.action == "import-openai":
                if not args.batch or not args.batch_sha256:
                    parser.error("--batch and --batch-sha256 are required")
                report = import_openai_batch(plan, local_path(ROOT, args.batch), paths, out, batch_sha256=args.batch_sha256)
            else:
                if not args.producer or len(paths) != 1:
                    parser.error("Normalized import needs one response file and --producer")
                report = import_responses(plan, read_jsonl(paths[0]), out, producer=args.producer)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
