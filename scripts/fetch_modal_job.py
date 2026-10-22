#!/usr/bin/env python3
"""Recover a completed Modal job and import its runs into local MLflow. No GPU."""

import argparse
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from paiec.execution.modal_adapter import fetch, VOLUME_NAME
from paiec.execution.transport import import_tracking

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"job_[0-9a-f]{32}", args.job_id):
        parser.error("Invalid job ID")
    import modal

    volume = modal.Volume.from_name(VOLUME_NAME)
    destination = Path(args.out).resolve()
    fetch(volume, f"{args.job_id}/results.zip", destination)
    print(import_tracking(ROOT, destination))
