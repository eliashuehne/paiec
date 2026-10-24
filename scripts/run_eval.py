"""Score predictors with benchmark-level cross-validation on the public training data.

    python scripts/run_eval.py --predictors constant empirical_mean

Each fold holds out ~20% of benchmarks; predictors are scored only on held-out
benchmarks, so every benchmark is scored exactly once across the folds. Per-pair
results go to runs/<timestamp>/ for later comparison.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from paiec.evaluate import evaluate  # noqa: E402
from paiec.folds import benchmark_folds  # noqa: E402
from paiec.metrics import brier_alc  # noqa: E402
from paiec.pairs import choose_pairs  # noqa: E402
from paiec.predictors import REGISTRY  # noqa: E402
from paiec.schema import load_benchmark, training_benchmarks  # noqa: E402


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--predictors", nargs="+", default=list(REGISTRY), choices=list(REGISTRY))
    parser.add_argument("--data", type=Path, default=ROOT / "payload" / "data")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-pairs", type=int, default=20, help="pairs per benchmark")
    parser.add_argument("--eval-items", type=int, default=200, help="evaluation items per pair")
    parser.add_argument("--limit-benchmarks", type=int, default=None, help="quick smoke runs")
    return parser.parse_args()


def summarise(frame: pd.DataFrame) -> pd.Series:
    """Equal-weight mean over pairs of the per-pair Brier curve and ALC."""
    curves = frame.pivot_table(index=["benchmark", "subject_id"], columns="budget", values="brier")
    alc = curves.apply(lambda row: brier_alc(row.to_dict()), axis=1)
    summary = curves.mean()
    summary.index = [f"B{b}" for b in summary.index]
    return pd.concat([summary, pd.Series({"ALC": alc.mean(), "pairs": len(alc)})])


def main():
    args = parse_arguments()
    benchmarks = training_benchmarks(args.data)
    if args.limit_benchmarks:
        benchmarks = benchmarks[: args.limit_benchmarks]
    folds = benchmark_folds(benchmarks, k=args.folds, seed=args.seed)
    print(f"{len(benchmarks)} public eligible benchmarks, {args.folds} folds", flush=True)

    predictors = {name: REGISTRY[name]() for name in args.predictors}
    frames = []
    started = time.time()
    for fold, (_train, test) in enumerate(folds):
        for benchmark in test:
            tables = load_benchmark(args.data, benchmark)
            pairs = choose_pairs(tables, max_subjects=args.max_pairs)
            if not pairs:
                continue
            for name, predict in predictors.items():
                result = evaluate(pairs, predict, seed=args.seed, evaluation_items=args.eval_items)
                frames.append(result.frame.assign(predictor=name, fold=fold))
            print(f"  fold {fold} {benchmark}: {len(pairs)} pairs ({time.time() - started:.0f}s)",
                  flush=True)

    frame = pd.concat(frames, ignore_index=True)
    out = ROOT / "runs" / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True)
    frame.to_csv(out / "pairs.csv", index=False)

    columns = ["benchmark", "subject_id", "budget", "brier"]
    overall = frame.groupby("predictor")[columns].apply(summarise).round(4)
    by_fold = (
        frame.groupby(["predictor", "fold"])[columns]
        .apply(lambda f: summarise(f)["ALC"])
        .unstack()
        .round(4)
    )
    print("\n", overall.to_string(), "\n\nALC by fold:\n", by_fold.to_string(), sep="")
    overall.to_csv(out / "summary.csv")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
