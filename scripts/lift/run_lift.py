"""Score feature columns against a baseline on held-out benchmarks.

    python scripts/lift/run_lift.py --base KM --features runs/features/subjects.parquet --columns release_days
    python scripts/lift/run_lift.py --base A --candidate K --name k_vs_a
    python scripts/lift/run_lift.py --budgets --base A --abilities normalized_name --name b_vs_a_budget0

The candidate is the base plus the given columns and abilities, or another
named baseline (--candidate). By default every official budget is scored on
the same held-out rows and combined with the official weights; --budgets with
values narrows the list, and --budgets with none runs the original budget-0
loop on all rows. Prints the comparison and saves it to
runs/lift/<time>_<name>/; every run is appended to runs/lift/history.jsonl.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from paiec.folds import load_benchmark_groups  # noqa: E402
from paiec.lift import BASELINES, PARAMS, ROUNDS, load_features, run_lift  # noqa: E402
from paiec.lift_budget import BUDGET_BASELINES, RULES, BudgetBaseline, combine_budgets, run_budgets  # noqa: E402
from paiec.lift_report import (  # noqa: E402
    budget_table,
    compare,
    file_sha256,
    fingerprint_scores,
    format_budget_report,
    format_report,
    save_run,
)
from paiec.metrics import BUDGETS  # noqa: E402

SUBJECT_FIELDS = ROOT / "runs" / "features" / "subject_fields.parquet"


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sample", type=Path, default=ROOT / "runs" / "lift" / "sample.parquet")
    parser.add_argument("--groups", type=Path, default=ROOT / "configs" / "lift" / "benchmark_groups.json")
    parser.add_argument("--features", type=Path, nargs="*", default=[], help="feature tables to join")
    parser.add_argument("--columns", nargs="*", default=[], help="feature columns added to the base")
    parser.add_argument("--abilities", nargs="*", default=[], help="in-fold ability keys added to the base")
    parser.add_argument("--increasing", nargs="*", default=[],
                        help="candidate columns that may only raise P(correct) (budget mode)")
    parser.add_argument("--base", choices=sorted(BUDGET_BASELINES), default="A")
    parser.add_argument("--candidate", choices=sorted(BUDGET_BASELINES), help="compare against this baseline")
    parser.add_argument("--budgets", type=int, nargs="*", default=list(BUDGETS),
                        help="budgets to score (default: all official); none = budget-0 loop on all rows")
    parser.add_argument("--name", default="run", help="short label for the run directory")
    parser.add_argument("--out", type=Path, default=ROOT / "runs" / "lift")
    return parser.parse_args()


def git_state() -> dict:
    def git(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()

    return {"commit": git("rev-parse", "--short", "HEAD"), "dirty": bool(git("status", "--porcelain", "--", "src"))}


def candidate_of(args, base: BudgetBaseline) -> BudgetBaseline:
    if args.candidate:
        if args.columns or args.abilities:
            sys.exit("--candidate compares two baselines; do not add --columns or --abilities")
        return BUDGET_BASELINES[args.candidate]
    if not args.columns and not args.abilities:
        sys.exit("nothing to compare: pass --columns, --abilities or --candidate")
    if base.predictor in RULES:
        sys.exit(f"base {args.base} is a fixed rule; add features to KM instead")
    return BudgetBaseline((*base.columns, *args.columns), (*base.abilities, *args.abilities), base.predictor)


def main():
    args = parse_arguments()
    base = BUDGET_BASELINES[args.base]
    candidate = candidate_of(args, base)
    if not set(args.increasing) <= set(args.columns):
        sys.exit("--increasing must name columns passed with --columns")
    if args.increasing and not args.budgets:
        sys.exit("--increasing works in budget mode only")
    if not args.budgets and (args.base not in BASELINES or (args.candidate and args.candidate not in BASELINES)):
        sys.exit("the budget-0 loop on all rows supports only baselines A and B")
    tables = list(args.features)
    if (base.abilities or candidate.abilities) and SUBJECT_FIELDS not in tables:
        tables.append(SUBJECT_FIELDS)  # abilities are keyed by visible subject fields
    sample = load_features(pd.read_parquet(args.sample), tables)
    groups = load_benchmark_groups(args.groups)
    fingerprints = fingerprint_scores(sample, args.columns)
    title = (f"{args.name}: base {args.base} vs "
             + (f"baseline {args.candidate}" if args.candidate else
                f"base + columns {args.columns} + abilities {args.abilities}"))

    started = time.time()
    tables_out = {}
    if args.budgets:
        base_runs = run_budgets(sample, args.budgets, **vars(base), groups=groups)
        candidate_runs = run_budgets(sample, args.budgets, **vars(candidate), groups=groups,
                                     increasing=args.increasing)
        per_budget = {b: compare(base_runs[b], candidate_runs[b]) for b in args.budgets}
        comparison = compare(combine_budgets(base_runs), combine_budgets(candidate_runs))
        report = format_budget_report(per_budget, comparison, fingerprints, title)
        tables_out["budgets"] = budget_table(per_budget)
        budget_summaries = {str(b): c.summary for b, c in per_budget.items()}
    else:
        comparison = compare(
            run_lift(sample, base.columns, base.abilities, groups),
            run_lift(sample, candidate.columns, candidate.abilities, groups),
        )
        report = format_report(comparison, fingerprints, title + " (budget-0 loop, all rows)")
        budget_summaries = {}
    print(report)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    record = {
        "name": args.name,
        "created": stamp,
        "seconds": round(time.time() - started, 1),
        "base": args.base,
        "candidate": args.candidate,
        "columns": args.columns,
        "abilities": args.abilities,
        "increasing": args.increasing,
        "budgets": args.budgets,
        "budget_summaries": budget_summaries,
        "features": {str(path): file_sha256(path) for path in tables},
        "sample_sha256": file_sha256(args.sample),
        "groups_sha256": file_sha256(args.groups),
        "params": {**PARAMS, "rounds": ROUNDS},
        "git": git_state(),
    }
    out = args.out / f"{stamp}_{args.name}"
    save_run(out, comparison, fingerprints, report, record, tables_out)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
