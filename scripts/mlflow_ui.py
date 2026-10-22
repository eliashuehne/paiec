#!/usr/bin/env python
"""Serve the local experiment dashboard (run logging works without this)."""

import argparse
import os
from pathlib import Path
import sys

from paiec.tracking import local_tracking_uri

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=5001)
    args = parser.parse_args()
    storage = ROOT / "runs/mlflow"
    storage.mkdir(parents=True, exist_ok=True)
    os.chdir(ROOT)
    os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "true")
    os.execv(sys.executable, [
        sys.executable, "-m", "mlflow", "server",
        "--host", "127.0.0.1", "--port", str(args.port),
        "--backend-store-uri", local_tracking_uri(ROOT),
        "--default-artifact-root", (storage / "artifacts").as_uri(),
        "--no-serve-artifacts", "--workers", "1",
    ])
