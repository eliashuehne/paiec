#!/usr/bin/env python3
"""Run dataset-only preparation or training-only evidence selection in ZenML."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow", choices=["data", "evidence"], default="data")
    parser.add_argument("--config", help="Versioned recipe under configs/pipelines/")
    parser.add_argument("--out", help="Fresh output under runs/")
    parser.add_argument("--plan", action="store_true", help="Print the plan without reading outcomes, creating files or initializing ZenML")
    args = parser.parse_args()
    from paiec.pipeline_workflows import workflow_plan, execute_workflow
    out = args.out or ('runs/workflows/' + args.workflow + '/' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    plan = workflow_plan(ROOT, args.workflow, args.config, out)
    print(json.dumps(plan, indent=2), flush=True)
    if not args.plan:
        execute_workflow(plan)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
