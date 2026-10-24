"""The competition's item split: 50/50, seeded, with nested label budgets.

Items, not responses, are split. Acquisition and evaluation pools are disjoint,
and the acquisition order is shuffled once so the labels at budget 3 are a
prefix of those at budget 7. Only distinct items are ever revealed.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

FIRST_BUDGET = 0
MAX_BUDGET = 31


@dataclass(frozen=True)
class Split:
    """One pair's item pools, in acquisition order."""

    acquisition: tuple[str, ...]
    evaluation: tuple[str, ...]


def stable_seed(*parts: object) -> int:
    """Derive a seed from text so splits survive process restarts and reordering."""
    digest = hashlib.blake2b(
        "|".join(str(part) for part in parts).encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(digest, "little")


def make_split(
    item_ids: tuple[str, ...] | list[str],
    *,
    benchmark: str,
    subject_id: str,
    seed: int = 0,
    acquisition_items: int | None = None,
    evaluation_items: int | None = None,
) -> Split:
    """Split a pair's items and cap both pools for tractable local runs.

    Caps are applied after the split so the 50/50 structure and the acquisition
    order are unchanged; a capped run evaluates fewer items, not different ones.
    """
    ids = np.array(sorted(set(item_ids)), dtype=object)
    if ids.size < 2:
        raise ValueError("a pair needs at least two distinct items")

    if acquisition_items is not None and acquisition_items < MAX_BUDGET:
        raise ValueError(f"acquisition cap must cover the {MAX_BUDGET}-label budget")

    rng = np.random.default_rng(stable_seed(seed, benchmark, subject_id))
    shuffled = ids.copy()
    rng.shuffle(shuffled)

    n_acquisition = shuffled.size // 2
    acquisition = [str(x) for x in shuffled[:n_acquisition]]
    evaluation = [str(x) for x in shuffled[n_acquisition:]]

    if acquisition_items is not None:
        acquisition = acquisition[:acquisition_items]
    if evaluation_items is not None:
        evaluation = evaluation[:evaluation_items]
    if not acquisition or not evaluation:
        raise ValueError("split produced an empty pool; lower the caps")

    return Split(tuple(acquisition), tuple(evaluation))


def labelled_prefix(pair, split: Split, budget: int, first_label: dict) -> list:
    """Return the ``labeled`` evidence available at one budget.

    Each entry is ``[[subject, item], label]`` with the full subject dictionary,
    matching the evaluator. Multi-trial items reveal their first recorded
    response, since a candidate is one item.
    """
    if budget > len(split.acquisition):
        raise ValueError("budget exceeds the acquisition pool")
    entries = []
    for item_id in split.acquisition[:budget]:
        entries.append([[pair.subject, pair.item(item_id)], first_label[item_id]])
    return entries


def first_labels(responses: pd.DataFrame) -> dict[str, int]:
    """Map item id to the first recorded response, in table order."""
    ordered = responses.drop_duplicates("item_id", keep="first")
    return dict(zip(ordered["item_id"], ordered["label"]))


__all__ = ["Split", "make_split", "labelled_prefix", "first_labels", "stable_seed", "MAX_BUDGET"]
