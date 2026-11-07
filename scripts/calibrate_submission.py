"""Write a copy of the trained pooled-offset artifact with a different starting point.

    python scripts/calibrate_submission.py --start 0.35 --no-text-prior
    python scripts/build_submission.py --artifacts artifacts/pooled_offset_start035.npz --zip dist/pooled_offset_start035.zip

Every prediction starts from ``level + shift + scale * (prior logit - center)``.
``--start`` sets that starting point to a fixed probability for every pair, and
``--no-text-prior`` sets ``scale = 0`` so the logistic prior no longer moves it.
The budget-0 output is pulled towards 0.5 by the shift's prior uncertainty
(start 0.35 with tau 1.58 predicts 0.392).
All fitted update parameters (tau, sigma, share, item_sigma) stay as trained.

This builds the variant described, and not selected, in Section 5 of the report. The
public benchmarks prefer a start of 0.5, so a lower start is not justified by training data.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifacts", type=Path, default=ROOT / "artifacts" / "pooled_offset.npz")
    parser.add_argument("--start", type=float, required=True, help="starting probability, e.g. 0.35")
    parser.add_argument("--no-text-prior", action="store_true", help="set the prior's scale to 0")
    parser.add_argument("--out", type=Path, help="defaults to artifacts/pooled_offset_start<start>.npz")
    args = parser.parse_args()
    if not 0 < args.start < 1:
        parser.error("--start must be a probability in (0, 1)")

    with np.load(args.artifacts) as stored:
        parameters = {name: stored[name] for name in stored.files}
    trained = {name: float(parameters[name]) for name in ("level", "shift", "scale", "center")}
    parameters["level"] = np.float64(math.log(args.start / (1 - args.start)))
    parameters["shift"] = np.float64(0.0)
    if args.no_text_prior:
        parameters["scale"] = np.float64(0.0)

    out = args.out or args.artifacts.with_name(f"pooled_offset_start{round(args.start * 100):03d}.npz")
    np.savez_compressed(out, **parameters)
    record = {
        "source": args.artifacts.name,
        "start_probability": args.start,
        "text_prior": not args.no_text_prior,
        "trained": trained,
        "calibrated": {name: float(parameters[name]) for name in ("level", "shift", "scale", "center")},
        "unchanged": {name: float(parameters[name]) for name in ("tau", "sigma", "share", "item_sigma")},
    }
    out.with_suffix(".json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
