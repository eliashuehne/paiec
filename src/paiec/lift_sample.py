"""Build the fixed, balanced response sample that every lift run trains on.

Each benchmark contributes at most ``cap`` responses, drawn with a seed derived
from the benchmark name so the draw does not depend on loading order. Folds are
assigned after sampling, so benchmarks with no labelled responses never take a
fold slot. Every row carries its benchmark's fold, so later code never
re-splits the data.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import pandas as pd

from .split import stable_seed

SEALED = "sealed"
COLUMNS = ["benchmark", "subject_id", "item_id", "trial", "label", "fold"]


def fold_labels(
    sealed: Sequence[str], folds: Sequence[tuple[Sequence[str], Sequence[str]]]
) -> dict[str, str]:
    """Map each benchmark to its fold number as text, or ``"sealed"``."""
    labels = {name: SEALED for name in sealed}
    for number, (_train, test) in enumerate(folds):
        for name in test:
            if name in labels:
                raise ValueError(f"{name} is assigned to more than one fold")
            labels[name] = str(number)
    return labels


def sample_benchmark(
    responses: pd.DataFrame, benchmark: str, cap: int, seed: int = 0
) -> pd.DataFrame:
    """Draw at most ``cap`` labelled responses from one benchmark."""
    if cap < 1:
        raise ValueError("cap must be at least 1")
    labelled = responses[responses["label"].isin([0, 1])]
    if len(labelled) > cap:
        labelled = labelled.sample(n=cap, random_state=stable_seed(seed, benchmark) % 2**32)
    sample = labelled.assign(benchmark=benchmark)
    sample["label"] = sample["label"].astype(int)
    return sample[COLUMNS[:-1]].reset_index(drop=True)


def build_sample(
    tables: Iterable[tuple[str, pd.DataFrame]], cap: int, seed: int = 0
) -> pd.DataFrame:
    """Concatenate per-benchmark samples, without folds yet."""
    parts = [sample_benchmark(responses, benchmark, cap, seed) for benchmark, responses in tables]
    parts = [part for part in parts if len(part)]
    if not parts:
        return pd.DataFrame(columns=COLUMNS[:-1])
    return pd.concat(parts, ignore_index=True)


def assign_folds(sample: pd.DataFrame, fold_of: dict[str, str]) -> pd.DataFrame:
    """Add the fold column; every sampled benchmark must have exactly one fold."""
    missing = sorted(set(sample["benchmark"]) - set(fold_of))
    if missing:
        raise ValueError(f"no fold for {missing}")
    return sample.assign(fold=sample["benchmark"].map(fold_of))[COLUMNS]


def summarise_sample(sample: pd.DataFrame) -> dict:
    """Row and benchmark counts per fold, for the sample manifest."""
    by_fold = sample.groupby("fold")
    return {
        "rows": int(len(sample)),
        "benchmarks": int(sample["benchmark"].nunique()),
        "positive_rate": round(float(sample["label"].mean()), 4) if len(sample) else None,
        "rows_by_fold": {fold: int(n) for fold, n in by_fold.size().items()},
        "benchmarks_by_fold": {
            fold: int(n) for fold, n in by_fold["benchmark"].nunique().items()
        },
    }


__all__ = ["COLUMNS", "SEALED", "assign_folds", "build_sample", "fold_labels", "sample_benchmark", "summarise_sample"]
