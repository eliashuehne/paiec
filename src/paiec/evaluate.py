"""Run the competition's scoring loop over local pairs.

For each pair: split the items, reveal labels one budget at a time, predict the
evaluation items, score Brier at every budget, and summarise the curve. This is
the same call pattern the platform uses, minus the hidden data.

The platform sends every target the labels revealed so far for all pairs in
the run, all at the same budget. ``pool_size`` reproduces that: pairs of one
benchmark are cut into pools, and each pair sees its whole pool's labels.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .contracts import Predictor
from .metrics import BUDGETS, brier, brier_alc, ece
from .pairs import Pair
from .split import first_labels, labelled_prefix, make_split, stable_seed


@dataclass(frozen=True)
class EvalResult:
    """Per-pair scores plus the suite summary."""

    frame: pd.DataFrame
    per_budget: pd.Series
    pair_alc: pd.Series
    brier_alc: float
    ece: float

    def by_benchmark(self) -> pd.Series:
        """Mean pair ALC per benchmark, for spotting one domain carrying the score."""
        return self.pair_alc.groupby(level="benchmark").mean()


def valid_probability(value, context: str) -> float:
    """Convert a prediction to a probability and reject invalid values."""
    probability = float(value)
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError(f"{context} returned an invalid probability: {value!r}")
    return probability


def pair_split(
    pair: Pair,
    *,
    seed: int = 0,
    acquisition_items: int | None = None,
    evaluation_items: int | None = None,
):
    """The pair's own split, identical however the pair is pooled."""
    return make_split(
        pair.item_ids,
        benchmark=pair.benchmark,
        subject_id=pair.subject_id,
        seed=seed,
        acquisition_items=acquisition_items,
        evaluation_items=evaluation_items,
    )


def make_pools(
    pairs: Sequence[Pair], pool_size: int | None, seed: int = 0
) -> list[list[Pair]]:
    """Shuffle each benchmark's pairs and cut them into pools of ``pool_size``.

    ``None`` puts all pairs of a benchmark in one pool. Pools never mix
    benchmarks; the last pool of a benchmark may be smaller.
    """
    if pool_size is not None and pool_size < 1:
        raise ValueError("pool_size must be at least 1, or None for whole benchmarks")
    by_benchmark: dict[str, list[Pair]] = {}
    for pair in pairs:
        by_benchmark.setdefault(pair.benchmark, []).append(pair)

    pools = []
    for benchmark, members in by_benchmark.items():
        order = np.random.default_rng(stable_seed(seed, benchmark, "pools")).permutation(
            len(members)
        )
        members = [members[index] for index in order]
        size = len(members) if pool_size is None else pool_size
        pools.extend(members[start : start + size] for start in range(0, len(members), size))
    return pools


def pooled_evidence(
    pool: Sequence[Pair],
    *,
    seed: int = 0,
    budgets: Sequence[int] = BUDGETS,
    acquisition_items: int | None = None,
    evaluation_items: int | None = None,
) -> dict[int, list]:
    """Return the ``labeled`` list the whole pool shares at each budget.

    Every member reveals the first ``budget`` items of its own acquisition
    order, so a member's evaluation items are never revealed as its own labels.
    """
    prefixes = []
    for pair in pool:
        split = pair_split(
            pair,
            seed=seed,
            acquisition_items=acquisition_items,
            evaluation_items=evaluation_items,
        )
        prefixes.append(
            labelled_prefix(pair, split, max(budgets), first_labels(pair.responses))
        )
    return {
        budget: [entry for prefix in prefixes for entry in prefix[:budget]]
        for budget in budgets
    }


def score_pair(
    pair: Pair,
    predict: Predictor,
    *,
    seed: int = 0,
    budgets: Sequence[int] = BUDGETS,
    acquisition_items: int | None = None,
    evaluation_items: int | None = None,
    evidence: Mapping[int, list] | None = None,
) -> list[dict]:
    """Score one pair at every budget.

    ``evidence`` maps each budget to a shared ``labeled`` list (see
    ``pooled_evidence``); without it the pair sees only its own labels.
    """
    split = pair_split(
        pair,
        seed=seed,
        acquisition_items=acquisition_items,
        evaluation_items=evaluation_items,
    )
    first_response_by_item = first_labels(pair.responses)
    evaluation_item_ids = set(split.evaluation)
    evaluation_responses = pair.responses[
        pair.responses["item_id"].isin(evaluation_item_ids)
    ]

    labels = evaluation_responses["label"].to_numpy()
    results = []
    for budget in budgets:
        if evidence is None:
            labeled = labelled_prefix(pair, split, budget, first_response_by_item)
        else:
            labeled = evidence[budget]
        predictions = {}
        for item_id in split.evaluation:
            item = pair.item(item_id)
            predictions[item_id] = valid_probability(
                predict(pair.subject, item, labeled),
                f"{pair.benchmark}/{pair.subject_id} at budget {budget}",
            )
        # One item prediction can score several response rows (repeated trials).
        probabilities = evaluation_responses["item_id"].map(predictions).to_numpy()
        results.append(
            {
                "benchmark": pair.benchmark,
                "subject_id": pair.subject_id,
                "budget": int(budget),
                "brier": brier(labels, probabilities),
                "ece": ece(labels, probabilities),
                "n_responses": int(labels.size),
                "n_items": len(split.evaluation),
            }
        )
    return results


def _pair_brier_alc(scores: pd.Series) -> float:
    budgets = scores.index.get_level_values("budget")
    return brier_alc(dict(zip(budgets, scores)))


def evaluate(
    pairs: Iterable[Pair],
    predict: Predictor,
    *,
    seed: int = 0,
    budgets: Sequence[int] = BUDGETS,
    acquisition_items: int | None = None,
    evaluation_items: int | None = None,
    pool_size: int | None = 1,
) -> EvalResult:
    """Score a suite of pairs and average the curves with equal pair weight.

    ``pool_size`` is how many pairs of a benchmark share revealed labels
    (1 = own labels only, ``None`` = all pairs of the benchmark).
    """
    pairs = list(pairs)
    caps = dict(seed=seed, acquisition_items=acquisition_items, evaluation_items=evaluation_items)
    evidence_by_pair = {}
    if pool_size != 1:
        for pool in make_pools(pairs, pool_size, seed):
            evidence = pooled_evidence(pool, budgets=budgets, **caps)
            for pair in pool:
                evidence_by_pair[pair.benchmark, pair.subject_id] = evidence

    records = []
    for pair in pairs:
        records.extend(
            score_pair(
                pair,
                predict,
                budgets=budgets,
                evidence=evidence_by_pair.get((pair.benchmark, pair.subject_id)),
                **caps,
            )
        )
    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        raise ValueError("no pairs were scored")

    scores = frame.set_index(["benchmark", "subject_id", "budget"])["brier"]
    pair_alc = scores.groupby(level=["benchmark", "subject_id"]).agg(
        _pair_brier_alc
    )
    per_budget = frame.groupby("budget")["brier"].mean()
    return EvalResult(
        frame=frame,
        per_budget=per_budget,
        pair_alc=pair_alc,
        brier_alc=float(pair_alc.mean()),
        ece=float(frame["ece"].mean()),
    )


def format_summary(result: EvalResult, label: str = "") -> str:
    """Render one run as a compact table for the terminal."""
    lines = []
    if label:
        lines.append(label)
    lines.append(
        "  budget: " + "  ".join(f"{b:>6}" for b in result.per_budget.index)
    )
    lines.append(
        "  brier : " + "  ".join(f"{v:6.4f}" for v in result.per_budget.to_numpy())
    )
    lines.append(f"  Brier ALC: {result.brier_alc:.4f}   mean ECE: {result.ece:.4f}")
    lines.append(f"  pairs: {len(result.pair_alc)}")
    return "\n".join(lines)


__all__ = [
    "EvalResult",
    "evaluate",
    "format_summary",
    "make_pools",
    "pooled_evidence",
    "score_pair",
]
