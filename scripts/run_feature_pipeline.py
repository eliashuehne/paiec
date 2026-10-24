#!/usr/bin/env python3
"""Set up or execute local, preparation-only ZenML pipelines with MLflow."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from paiec.zenml_pipeline import configure_environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["setup", "run", "ui", "status"])
    parser.add_argument("--config", default="configs/pipelines/feature_preparation_offline_v10.json")
    parser.add_argument("--out", help="Fresh repository-relative output under runs/")
    parser.add_argument("--port", type=int, default=8237)
    args = parser.parse_args()
    configure_environment(ROOT)
    os.chdir(ROOT)
    if args.action == "ui":
        return subprocess.call([str(Path(sys.executable).parent / "zenml"), "login", "--local",
                                "--port", str(args.port), "--ip-address", "127.0.0.1"])
    from paiec.zenml_pipeline import setup_stack
    if args.action == "setup":
        print(json.dumps(setup_stack(ROOT), indent=2))
        return 0
    if args.action == "status":
        from zenml.client import Client
        client = Client(root=ROOT)
        print(json.dumps({"stack": client.active_stack_model.name, "store_type": client.zen_store.type.value,
                          "setup": json.loads((ROOT / "runs/zenml/setup.json").read_text())}, indent=2))
        return 0
    from paiec.pipeline_workflows import workflow_plan, execute_workflow
    out = args.out or ('runs/workflows/research/' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    plan = workflow_plan(ROOT, 'research', args.config, out)
    execute_workflow(plan)
    print(f"Feature preparation: {Path(plan['out']) / 'report.html'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
