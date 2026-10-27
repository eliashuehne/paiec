"""Build the fixed, balanced response sample that every lift run trains on.

Legacy sampling contributes at most ``cap`` responses per benchmark, drawn with a seed derived
from the benchmark name so the draw does not depend on loading order. Folds are
assigned after sampling, so benchmarks with no labelled responses never take a
fold slot. Every row carries its benchmark's fold, so later code never
re-splits the data. Budget sampling splits complete eligible pairs first,
retains acquisition observations, and applies the cap only to evaluation rows.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import pandas as pd

from .split import MAX_BUDGET, make_split, stable_seed

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


BUDGET_COLUMNS = ["pool", "order", "n_pair_items"]
MIN_PAIR_ITEMS = 80


def sample_budget_benchmark(
    responses: pd.DataFrame, benchmark: str, cap: int, seed: int = 0
) -> pd.DataFrame:
    """Split eligible complete pairs, retain 31 reveals, then cap evaluation rows.

    Every eligible pair keeps at least one scored response. The cap applies to
    evaluation rows only; acquisition observations are retained separately.
    """
    if cap < 1:
        raise ValueError("cap must be at least 1")
    labelled = responses.loc[responses["label"].isin([0, 1])].copy()
    labelled["label"] = labelled["label"].astype(int)
    parts = []
    for subject, group in labelled.groupby("subject_id", sort=True):
        ids = group["item_id"].unique().tolist()
        if len(ids) < MIN_PAIR_ITEMS:
            continue
        split = make_split(ids, benchmark=benchmark, subject_id=subject, seed=seed)
        order = {item: i for i, item in enumerate(split.acquisition[:MAX_BUDGET])}
        reveal = group.loc[group["item_id"].isin(order)].sort_values("trial", kind="stable")
        reveal = reveal.drop_duplicates("item_id").assign(pool="reveal")
        reveal["order"] = reveal["item_id"].map(order)
        evaluate = group.loc[group["item_id"].isin(split.evaluation)].assign(pool="evaluate", order=-1)
        parts.append(pd.concat([reveal, evaluate]).assign(n_pair_items=len(ids)))
    columns = [*COLUMNS[:-1], *BUDGET_COLUMNS]
    if not parts:
        return pd.DataFrame(columns=columns)
    complete = pd.concat(parts, ignore_index=True)
    evaluate = complete[complete["pool"] == "evaluate"]
    if len(evaluate) > cap:
        # Reserve one random response per pair before filling the remaining cap.
        shuffled = evaluate.sample(frac=1, random_state=stable_seed(seed, benchmark) % 2**32)
        reserved = shuffled.drop_duplicates("subject_id")
        if len(reserved) > cap:
            raise ValueError(f"cap {cap} must cover all {len(reserved)} eligible pairs in {benchmark}")
        rest = shuffled.drop(index=reserved.index).iloc[:cap - len(reserved)]
        evaluate = pd.concat([reserved, rest])
    sample = pd.concat([complete[complete["pool"] == "reveal"], evaluate])
    sample = sample.assign(benchmark=benchmark)
    return sample[columns].sort_values(["subject_id", "pool", "order", "item_id", "trial"]).reset_index(drop=True)


def build_sample(
    tables: Iterable[tuple[str, pd.DataFrame]], cap: int, seed: int = 0, budget_mode: bool = False
) -> pd.DataFrame:
    """Concatenate per-benchmark samples, without folds yet."""
    draw = sample_budget_benchmark if budget_mode else sample_benchmark
    parts = [draw(responses, benchmark, cap, seed) for benchmark, responses in tables]
    parts = [part for part in parts if len(part)]
    if not parts:
        return pd.DataFrame(columns=[*COLUMNS[:-1], *(BUDGET_COLUMNS if budget_mode else [])])
    return pd.concat(parts, ignore_index=True)


def assign_folds(sample: pd.DataFrame, fold_of: dict[str, str]) -> pd.DataFrame:
    """Add the fold column; every sampled benchmark must have exactly one fold."""
    missing = sorted(set(sample["benchmark"]) - set(fold_of))
    if missing:
        raise ValueError(f"no fold for {missing}")
    extra = [column for column in BUDGET_COLUMNS if column in sample]
    return sample.assign(fold=sample["benchmark"].map(fold_of))[[*COLUMNS, *extra]]


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
