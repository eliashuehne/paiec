"""Subject-benchmark pairs, the unit the competition scores.

A pair is one subject's responses on one benchmark. The platform scores only
pairs with at least 80 distinct items, split 50/50 into acquisition and
evaluation pools, so the harness builds the same unit here.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterator

import pandas as pd
import numpy as np

from .schema import BenchTables, clean, item_record, subject_record
from .split import stable_seed


@dataclass(frozen=True)
class Pair:
    """One subject's responses on one benchmark."""

    benchmark: str
    subject_id: str
    subject: dict
    items: pd.DataFrame
    responses: pd.DataFrame
    item_ids: tuple[str, ...]

    def item(self, item_id: str) -> dict:
        """Return the four-field item dictionary for one target."""
        return item_record(self.items.loc[item_id], self.benchmark)


def iter_pairs(
    tables: BenchTables,
    *,
    min_items: int = 80,
    max_subjects: int | None = None,
) -> Iterator[Pair]:
    """Yield one Pair per subject with enough distinct items to be scored.

    ``min_items`` mirrors the competition's 80-item rule, which decides both
    which pairs exist and which items are eligible for the 50/50 split.
    """
    yielded = 0
    for subject_id, group in tables.responses.groupby("subject_id", sort=True):
        if subject_id not in tables.subjects.index:
            continue
        distinct = group["item_id"].drop_duplicates()
        if distinct.size < min_items:
            continue

        rows = group.drop_duplicates("item_id").set_index("item_id")
        items = tables.items.reindex(rows.index)[["content", "item_features"]].copy()
        items["interactors"] = rows["interactors"]

        yield Pair(
            benchmark=tables.benchmark,
            subject_id=str(subject_id),
            subject=subject_record(tables.subjects.loc[subject_id]),
            items=items,
            responses=group[["item_id", "label"]].reset_index(drop=True),
            item_ids=tuple(str(x) for x in distinct),
        )
        yielded += 1
        if max_subjects is not None and yielded >= max_subjects:
            return


def choose_pairs(
    tables: BenchTables,
    *,
    max_subjects: int,
    min_items: int = 80,
    randomize: bool = False,
    seed: int = 0,
) -> list[Pair]:
    """Pick a deterministic, reproducible subset of a benchmark's pairs.

    Subjects are sorted by id so the same benchmark always yields the same
    pairs regardless of table ordering.
    """
    counts = tables.responses.groupby("subject_id")["item_id"].nunique()
    eligible = sorted(counts[(counts >= min_items) & counts.index.isin(tables.subjects.index)].index, key=str)
    if randomize:
        rng = np.random.default_rng(stable_seed(seed, tables.benchmark, "subject-selection"))
        rng.shuffle(eligible)
    selected = eligible[:max_subjects]
    # Select IDs before constructing Pair objects, avoiding copies for every
    # eligible subject when a benchmark contains hundreds of systems.
    subset = replace(tables, responses=tables.responses[tables.responses.subject_id.isin(selected)])
    pairs = {pair.subject_id: pair for pair in iter_pairs(subset, min_items=min_items)}
    return [pairs[str(subject_id)] for subject_id in selected]


def balanced_pairs(pairs: list[Pair], total: int, *, seed: int) -> list[Pair]:
    """Sample without replacement across benchmarks, without using outcomes.

    Cycle through benchmarks in shuffled order. When a benchmark has fewer
    subjects, redistribute its remaining allocation across other benchmarks.
    """
    if len(pairs) < total:
        raise ValueError(f"requested {total} pairs but only {len(pairs)} eligible candidates were loaded")
    groups = {}
    for pair in pairs:
        groups.setdefault(pair.benchmark, []).append(pair)
    rng = np.random.default_rng(stable_seed(seed, "benchmark-balanced-pairs"))
    for group in groups.values():
        rng.shuffle(group)
    selected = []
    while len(selected) < total:
        active = sorted(name for name, group in groups.items() if group)
        rng.shuffle(active)
        for name in active:
            selected.append(groups[name].pop())
            if len(selected) == total:
                break
    return selected


__all__ = ["Pair", "iter_pairs", "choose_pairs", "balanced_pairs", "clean"]
