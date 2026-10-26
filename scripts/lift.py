"""Score feature columns against a baseline on held-out benchmarks.

    python scripts/lift.py --base A --abilities normalized_name --name b_vs_a
    python scripts/lift.py --features runs/features/items.parquet --columns n_chars has_code

The candidate is the base plus the given columns and abilities. Prints the
comparison and saves it to runs/lift/<time>_<name>/; every run is appended to
runs/lift/history.jsonl.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from paiec.folds import load_benchmark_groups  # noqa: E402
from paiec.lift import BASELINES, PARAMS, ROUNDS, load_features, run_lift  # noqa: E402
from paiec.lift_report import compare, file_sha256, fingerprint_scores, format_report, save_run  # noqa: E402

SUBJECT_FIELDS = ROOT / "runs" / "features" / "subject_fields.parquet"


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sample", type=Path, default=ROOT / "runs" / "lift" / "sample.parquet")
    parser.add_argument("--groups", type=Path, default=ROOT / "configs" / "lift" / "benchmark_groups.json")
    parser.add_argument("--features", type=Path, nargs="*", default=[], help="feature tables to join")
    parser.add_argument("--columns", nargs="*", default=[], help="feature columns added to the base")
    parser.add_argument("--abilities", nargs="*", default=[], help="in-fold ability keys added to the base")
    parser.add_argument("--base", choices=sorted(BASELINES), default="A")
    parser.add_argument("--name", default="run", help="short label for the run directory")
    parser.add_argument("--out", type=Path, default=ROOT / "runs" / "lift")
    return parser.parse_args()


def git_state() -> dict:
    def git(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()

    return {"commit": git("rev-parse", "--short", "HEAD"), "dirty": bool(git("status", "--porcelain", "--", "src"))}


def main():
    args = parse_arguments()
    if not args.columns and not args.abilities:
        sys.exit("nothing to compare: pass --columns and/or --abilities")
    base = BASELINES[args.base]
    tables = list(args.features)
    if (base.abilities or args.abilities) and SUBJECT_FIELDS not in tables:
        tables.append(SUBJECT_FIELDS)  # abilities are keyed by visible subject fields
    sample = load_features(pd.read_parquet(args.sample), tables)
    groups = load_benchmark_groups(args.groups)

    started = time.time()
    base_result = run_lift(sample, base.columns, base.abilities, groups)
    candidate_result = run_lift(
        sample,
        [*base.columns, *args.columns],
        [*base.abilities, *args.abilities],
        groups,
    )
    comparison = compare(base_result, candidate_result)
    fingerprints = fingerprint_scores(sample, args.columns)
    title = f"{args.name}: base {args.base} + columns {args.columns} + abilities {args.abilities}"
    report = format_report(comparison, fingerprints, title)
    print(report)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    record = {
        "name": args.name,
        "created": stamp,
        "seconds": round(time.time() - started, 1),
        "base": args.base,
        "columns": args.columns,
        "abilities": args.abilities,
        "features": {str(path): file_sha256(path) for path in tables},
        "sample_sha256": file_sha256(args.sample),
        "groups_sha256": file_sha256(args.groups),
        "params": {**PARAMS, "rounds": ROUNDS},
        "git": git_state(),
    }
    out = args.out / f"{stamp}_{args.name}"
    save_run(out, comparison, fingerprints, report, record)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
