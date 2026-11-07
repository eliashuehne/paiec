"""Which shared part of LfHL produces the gain from other systems' labels?

    python scripts/run_pooling_ablation.py --folds 0 1 2 3 4   # one process per fold also works
    python scripts/run_pooling_ablation.py --merge             # writes runs/ablation/pairs.csv

Each fold fits ``pooled_offset`` as ``scripts/run_eval.py`` does, then scores the same
fitted model with shared terms switched off (nothing is refitted):

    ablate_tags            tag shifts shared; no shared level, no question shifts
    ablate_tags_level      tags and the benchmark level
    ablate_tags_questions  tags and the per-question shifts
    ablate_all             the full model, at pool size 2 only, as a check on the run above
    calibrated             the full model with start 0.35 and no text prior (LfHL-cal)

Pairs, splits and pools are those of ``run_eval.py``, so results pair up with its runs.
Finished benchmarks are kept, so a stopped run resumes where it was.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from paiec.evaluate import evaluate  # noqa: E402
from paiec.folds import benchmark_folds  # noqa: E402
from paiec.pairs import choose_pairs  # noqa: E402
from paiec.predictors import build  # noqa: E402
from paiec.schema import load_benchmark, training_benchmarks  # noqa: E402

OUT = ROOT / "runs" / "ablation"
POOLS = ("2", "5", "all")
START = 0.35


def set_terms(predictor, share: float, item_sigma: float, start: tuple) -> None:
    """Switch shared terms on or off and drop every posterior cached under the old ones."""
    predictor.share, predictor.item_sigma = share, item_sigma
    predictor.level, predictor.shift, predictor.scale = start
    predictor._posteriors.clear()
    predictor._labeled, predictor._labeled_size, predictor._pools = None, -1, {}


def score_fold(fold: int, data: Path, seed: int = 0) -> None:
    benchmarks = training_benchmarks(data)
    train, test = benchmark_folds(benchmarks, k=5, seed=seed)[fold]
    todo = [b for b in test if not (OUT / f"{b}.csv").exists()]
    started = time.time()
    if not todo:
        return
    predictor = build("pooled_offset", data, train)
    share, item_sigma = predictor.share, predictor.item_sigma
    trained = (predictor.level, predictor.shift, predictor.scale)
    calibrated = (math.log(START / (1 - START)), 0.0, 0.0)
    print(f"fold {fold}: fitted, share {share}, item_sigma {item_sigma} ({time.time() - started:.0f}s)", flush=True)
    variants = [
        ("ablate_all", share, item_sigma, trained, ("2",)),
        ("ablate_tags", 0.0, 0.0, trained, POOLS),
        ("ablate_tags_level", share, 0.0, trained, POOLS),
        ("ablate_tags_questions", 0.0, item_sigma, trained, POOLS),
        ("calibrated", share, item_sigma, calibrated, ("1", *POOLS)),
    ]
    for benchmark in todo:
        pairs = choose_pairs(load_benchmark(data, benchmark), max_subjects=20)
        frames = []
        for name, variant_share, variant_sigma, start, pools in variants if pairs else ():
            set_terms(predictor, variant_share, variant_sigma, start)
            for pool in pools:
                result = evaluate(pairs, predictor, seed=seed, evaluation_items=200,
                                  pool_size=None if pool == "all" else int(pool))
                frames.append(result.frame.assign(predictor=name, pool=pool, fold=fold))
        table = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        target = OUT / f"{benchmark}.csv"
        table.to_csv(target.with_suffix(".tmp"), index=False)
        target.with_suffix(".tmp").replace(target)
        print(f"fold {fold} {benchmark}: {len(pairs)} pairs ({time.time() - started:.0f}s)", flush=True)


def merge() -> None:
    frames = []
    for path in sorted(OUT.glob("*.csv")):
        if path.name != "pairs.csv" and path.stat().st_size > 2:
            frames.append(pd.read_csv(path, dtype={"pool": str}))
    table = pd.concat(frames, ignore_index=True)
    table.to_csv(OUT / "pairs.csv", index=False)
    print(f"{table.benchmark.nunique()} benchmarks -> {OUT / 'pairs.csv'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--folds", nargs="+", type=int, default=[])
    parser.add_argument("--data", type=Path, default=ROOT / "payload" / "data")
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for fold in args.folds:
        score_fold(fold, args.data)
    if args.merge:
        merge()


if __name__ == "__main__":
    main()
