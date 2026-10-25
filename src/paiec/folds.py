"""Create reproducible benchmark-level cross-validation folds."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

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


def load_benchmark_groups(path: str | Path) -> list[list[str]]:
    """Read the benchmark groups that must share a fold from a JSON config."""
    return [sorted(members) for members in json.loads(Path(path).read_text())["groups"].values()]


def grouped_benchmark_folds(
    benchmarks: Iterable[str],
    groups: Iterable[Iterable[str]] = (),
    k: int = 5,
    seed: int = 0,
    sealed_fraction: float = 0.2,
) -> tuple[list[str], list[tuple[list[str], list[str]]]]:
    """Return a sealed holdout plus ``k`` folds that never split a group.

    Benchmarks that share items would leak across folds, so each group moves
    as one unit. The sealed set is drawn first from the shuffled units, so it
    stays the same when ``k`` changes. Group members absent from
    ``benchmarks`` are ignored.
    """
    benchmark_names = sorted(set(benchmarks))
    if k < 2:
        raise ValueError("k must be at least 2")
    if not 0 <= sealed_fraction < 1:
        raise ValueError("sealed_fraction must be in [0, 1)")

    group_of = {}
    for members in map(list, groups):
        for name in members:
            if name in group_of:
                raise ValueError(f"{name} is in more than one group")
            group_of[name] = min(members)
    units: dict[str, list[str]] = {}
    for name in benchmark_names:
        units.setdefault(group_of.get(name, name), []).append(name)
    unit_list = [units[key] for key in sorted(units)]

    shuffled_indexes = np.random.default_rng(seed).permutation(len(unit_list))
    shuffled = [unit_list[index] for index in shuffled_indexes]

    sealed_target = round(sealed_fraction * len(benchmark_names))
    sealed: list[str] = []
    while shuffled and len(sealed) < sealed_target:
        sealed.extend(shuffled.pop(0))
    if len(shuffled) < k:
        raise ValueError(f"need at least {k} unsealed benchmark units for {k} folds")

    # Largest units first, each to the emptiest fold, keeps fold sizes even.
    fold_members: list[list[str]] = [[] for _ in range(k)]
    for unit in sorted(shuffled, key=len, reverse=True):
        min(fold_members, key=len).extend(unit)

    unsealed = sorted(name for members in fold_members for name in members)
    folds = [
        ([name for name in unsealed if name not in test], sorted(test))
        for test in (set(members) for members in fold_members)
    ]
    return sorted(sealed), folds


__all__ = ["benchmark_folds", "grouped_benchmark_folds", "load_benchmark_groups"]
