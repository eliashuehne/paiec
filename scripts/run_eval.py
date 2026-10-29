"""Score predictors with benchmark-level cross-validation on the public training data.

    python scripts/run_eval.py --predictors constant empirical_mean

Each fold holds out ~20% of benchmarks. Learned predictors are fitted fresh on
the other ~80% and scored only on the held-out benchmarks, so every benchmark
is scored exactly once across the folds. Per-pair results go to runs/<timestamp>/.
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
from paiec.predictors import REGISTRY, build  # noqa: E402
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


def fitted_parameters(predictor) -> dict[str, float]:
    """Return displayable numeric parameters from a learned predictor."""
    return {
        name: round(value, 4)
        for name, value in vars(predictor).items()
        if isinstance(value, float)
    }


def evaluate_benchmark(benchmark, predictors, args, fold: int) -> list[pd.DataFrame]:
    """Evaluate every requested predictor on one held-out benchmark."""
    tables = load_benchmark(args.data, benchmark)
    pairs = choose_pairs(tables, max_subjects=args.max_pairs)
    if not pairs:
        return []

    frames = []
    for name, predict in predictors.items():
        result = evaluate(
            pairs,
            predict,
            seed=args.seed,
            evaluation_items=args.eval_items,
        )
        frames.append(result.frame.assign(predictor=name, fold=fold))
    return frames


def main():
    args = parse_arguments()
    benchmarks = training_benchmarks(args.data)
    if args.limit_benchmarks:
        benchmarks = benchmarks[: args.limit_benchmarks]
    folds = benchmark_folds(benchmarks, k=args.folds, seed=args.seed)
    print(f"{len(benchmarks)} public eligible benchmarks, {args.folds} folds", flush=True)

    frames = []
    started = time.time()
    for fold, (train, test) in enumerate(folds):
        predictors = {name: build(name, args.data, train) for name in args.predictors}
        elapsed = time.time() - started
        print(f"fold {fold}: fitted on {len(train)} benchmarks ({elapsed:.0f}s)", flush=True)
        for name, predictor in predictors.items():
            if hasattr(predictor, "fit"):
                print(f"  {name}: {fitted_parameters(predictor)}", flush=True)
        for benchmark in test:
            benchmark_frames = evaluate_benchmark(benchmark, predictors, args, fold)
            frames.extend(benchmark_frames)
            if benchmark_frames:
                pairs_scored = benchmark_frames[0][["benchmark", "subject_id"]].drop_duplicates()
                elapsed = time.time() - started
                print(
                    f"  fold {fold} {benchmark}: {len(pairs_scored)} pairs ({elapsed:.0f}s)",
                    flush=True,
                )

    if not frames:
        raise ValueError("no scoreable benchmark/subject pairs were found")
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
