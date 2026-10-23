"""Create reproducible benchmark-level cross-validation folds."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def benchmark_folds(
    benchmarks: Iterable[str], k: int = 5, seed: int = 0
) -> list[tuple[list[str], list[str]]]:
    """Return ``k`` train/test splits, holding out every benchmark once.

    Sorting before shuffling means callers get the same folds regardless of
    the order in which they collected benchmark names.
    """
    benchmark_names = sorted(set(benchmarks))
    if k < 2:
        raise ValueError("k must be at least 2")
    if len(benchmark_names) < k:
        raise ValueError(f"need at least {k} benchmarks for {k} folds")

    shuffled_indexes = np.random.default_rng(seed).permutation(len(benchmark_names))
    fold_by_benchmark = {
        benchmark_names[index]: position % k
        for position, index in enumerate(shuffled_indexes)
    }
    return [
        (
            [name for name in benchmark_names if fold_by_benchmark[name] != fold_number],
            [name for name in benchmark_names if fold_by_benchmark[name] == fold_number],
        )
        for fold_number in range(k)
    ]


__all__ = ["benchmark_folds"]
