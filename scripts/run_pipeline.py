#!/usr/bin/env python3
"""Canonical ZenML workflow command. Planning is the default; execution is explicit."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from paiec.pipeline_workflows import WORKFLOWS, workflow_plan, execute_workflow


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workflow", choices=sorted(WORKFLOWS))
    parser.add_argument("--config")
    parser.add_argument("--out")
    parser.add_argument("--source-root", help="Physical root of hash-pinned feature inputs; features workflow only")
    parser.add_argument("--execute", action="store_true", help="Execute the selected workflow, including its declared resource/data access")
    parser.add_argument("--allow-development-evaluation", action="store_true", help="Allow a newly requested assessment; not needed for pinned saved-result review")
    args = parser.parse_args()
    out = args.out or "runs/workflows/" + args.workflow + "/" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    plan = workflow_plan(ROOT, args.workflow, args.config, out, source_root=args.source_root)
    print(json.dumps(plan, indent=2), flush=True)
    if args.execute:
        execute_workflow(plan, allow_development=args.allow_development_evaluation)


if __name__ == "__main__":
    main()
