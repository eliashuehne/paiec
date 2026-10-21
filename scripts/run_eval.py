#!/usr/bin/env python
"""Score one or more predictors over a named evaluation suite.

    PYTHONPATH=src .venv/bin/python scripts/run_eval.py \
        --predictor constant empirical_mean subject_prior shrunk_mean

The suite is defined in configs/suites.json. All pairs come from the public
training release, so this measures predictors on benchmarks they can see; the
honest cold-start check is the leave-benchmarks-out suite used for the learned
models.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from paiec.evaluate import evaluate, format_summary
from paiec.pairs import choose_pairs
from paiec.predictors import Constant, EmpiricalMean, ShrunkMean, SubjectPrior
from paiec.schema import load_benchmark
from paiec.stats import load_or_build_subject_stats
from paiec.textmodel import TextRidge, load_or_build_text_training

ROOT = Path(__file__).resolve().parents[1]


def manifest_benchmarks(data_root: Path) -> list[str]:
    path = data_root / "corpus_manifest.json"
    manifest = json.loads(path.read_text())
    return [entry.split("/")[0] for entry in manifest["eligible_dirs"]]


def load_suite(config: Path, name: str) -> dict:
    suites = json.loads(config.read_text())
    if name not in suites:
        raise SystemExit(f"unknown suite {name!r}; have {sorted(suites)}")
    return suites[name]


def build_pairs(data_root: Path, suite: dict):
    pairs = []
    for benchmark in suite["benchmarks"]:
        tables = load_benchmark(data_root, benchmark)
        chosen = choose_pairs(
            tables,
            max_subjects=suite["pairs_per_benchmark"],
            min_items=suite.get("min_items", 80),
        )
        if not chosen:
            print(f"  {benchmark}: no pair with {suite.get('min_items', 80)}+ items")
        else:
            print(f"  {benchmark}: {len(chosen)} pair(s)")
        pairs.extend(chosen)
    if not pairs:
        raise SystemExit("suite produced no pairs")
    return pairs


def make_predictor(name: str, data_root: Path, suite: dict, cache: Path, k: float):
    if name == "constant":
        return Constant(0.5)
    if name == "empirical_mean":
        return EmpiricalMean()
    if name in {"subject_prior", "shrunk_mean"}:
        all_benchmarks = manifest_benchmarks(data_root)
        stats = load_or_build_subject_stats(
            data_root, all_benchmarks, cache / "subject_stats.parquet"
        )
        prior = SubjectPrior(stats)
        return prior if name == "subject_prior" else ShrunkMean(prior, k=k)
    if name == "text_ridge":
        train = [
            b for b in manifest_benchmarks(data_root) if b not in suite["benchmarks"]
        ]
        if not train:
            raise SystemExit("no training benchmarks left outside the suite")
        frame = load_or_build_text_training(
            data_root, train, cache / f"text_train_{suite['seed']}.parquet",
            per_benchmark=100, seed=suite["seed"],
        )
        print(f"  ridge trained on {len(frame)} items from {len(train)} benchmarks")
        return TextRidge().fit(frame["text"], frame["label"])
    raise SystemExit(f"unknown predictor {name!r}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", default=str(ROOT / "payload" / "data"))
    ap.add_argument("--config", default=str(ROOT / "configs" / "suites.json"))
    ap.add_argument("--suite", default="default")
    ap.add_argument("--predictor", nargs="+", default=["constant"])
    ap.add_argument("--k", type=float, default=5.0, help="prior weight for shrunk_mean")
    ap.add_argument("--cache", default=str(ROOT / "runs" / "cache"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data_root = Path(args.data_root)
    suite = load_suite(Path(args.config), args.suite)
    cache = Path(args.cache)

    print(f"suite {args.suite}: {', '.join(suite['benchmarks'])}")
    pairs = build_pairs(data_root, suite)
    print(f"  {len(pairs)} pairs, {suite['evaluation_items']} eval items "
          f"and {suite['acquisition_items']} acquisition items per pair\n")

    results = {}
    for name in args.predictor:
        predictor = make_predictor(name, data_root, suite, cache, args.k)
        result = evaluate(
            pairs,
            predictor,
            seed=suite["seed"],
            acquisition_items=suite["acquisition_items"],
            evaluation_items=suite["evaluation_items"],
        )
        results[name] = result
        print(format_summary(result, label=name))
        print()

    summary = pd.DataFrame(
        {
            name: {
                "brier_alc": result.brier_alc,
                "mean_ece": result.ece,
                **{f"brier_{b}": v for b, v in result.per_budget.items()},
            }
            for name, result in results.items()
        }
    ).T.sort_values("brier_alc")

    print("comparison (lower is better)")
    print(summary.to_string(float_format=lambda v: f"{v:.4f}"))

    out = Path(args.out) if args.out else ROOT / "runs" / f"eval_{args.suite}"
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "summary.csv"
    if summary_path.is_file():
        # Keep earlier predictors so a suite can be filled in across runs.
        previous = pd.read_csv(summary_path, index_col=0)
        previous = previous.drop(index=summary.index, errors="ignore")
        summary = pd.concat([previous, summary]).sort_values("brier_alc")
    summary.to_csv(summary_path)
    for name, result in results.items():
        result.frame.to_csv(out / f"scores_{name}.csv", index=False)
    (out / "suite.json").write_text(json.dumps(suite, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
