"""Fit the submission model on all public training benchmarks and save its parameters.

    python scripts/train_submission.py

Model: logistic prior + Bayesian pair offset with tag shifts (``prior_offset``,
5-fold ALC 0.2096). Writes ``artifacts/prior_offset.npz`` and a JSON record.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from paiec.prior_offset import PriorOffset  # noqa: E402
from paiec.schema import training_benchmarks  # noqa: E402
from paiec.submission import save_artifacts  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=ROOT / "payload" / "data")
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts" / "prior_offset.npz")
    args = parser.parse_args()

    started = time.time()
    benchmarks = training_benchmarks(args.data)
    model = PriorOffset().fit(args.data, benchmarks)
    path = save_artifacts(model, args.out)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    record = {
        "model": "prior_offset",
        "benchmarks": len(benchmarks),
        "data_revision": json.loads((args.data / "manifest.json").read_text())["revision"],
        "git_commit": commit,
        "parameters": {name: getattr(model, name) for name in ("level", "shift", "scale", "center", "tau", "sigma")},
        "prior_regularization": model.prior.regularization,
        "seconds": round(time.time() - started),
    }
    path.with_suffix(".json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
