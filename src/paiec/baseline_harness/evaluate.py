"""Run the competition's scoring loop over local pairs.

For each pair: split the items, reveal labels one budget at a time, predict the
evaluation items, score Brier at every budget, and summarise the curve. This is
the same call pattern the platform uses, minus the hidden data.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from paiec.metrics import BUDGETS, brier, brier_alc, ece
from .pairs import Pair
from paiec.split import first_labels, labelled_prefix, make_split

Predict = Callable[[dict, dict, list], float]


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


def score_pair(
    pair: Pair,
    predict: Predict,
    *,
    seed: int = 0,
    budgets: Sequence[int] = BUDGETS,
    acquisition_items: int | None = None,
    evaluation_items: int | None = None,
) -> list[dict]:
    """Score one pair at every budget."""
    split = make_split(
        pair.item_ids,
        benchmark=pair.benchmark,
        subject_id=pair.subject_id,
        seed=seed,
        acquisition_items=acquisition_items,
        evaluation_items=evaluation_items,
    )
    first_response_by_item = first_labels(pair.responses)
    evaluation_item_ids = set(split.evaluation)
    evaluation_responses = pair.responses[
        pair.responses["item_id"].isin(evaluation_item_ids)
    ]

    results = []
    for budget in budgets:
        labeled = labelled_prefix(pair, split, budget, first_response_by_item)
        predictions = {}
        for item_id in split.evaluation:
            item = pair.item(item_id)
            predictions[item_id] = valid_probability(
                predict(pair.subject, item, labeled),
                f"{pair.benchmark}/{pair.subject_id} at budget {budget}",
            )
        labels = evaluation_responses["label"].to_numpy()
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


def evaluate(
    pairs: Iterable[Pair],
    predict: Predict,
    *,
    seed: int = 0,
    budgets: Sequence[int] = BUDGETS,
    acquisition_items: int | None = None,
    evaluation_items: int | None = None,
) -> EvalResult:
    """Score a suite of pairs and average the curves with equal pair weight."""
    records = []
    for pair in pairs:
        records.extend(
            score_pair(
                pair,
                predict,
                seed=seed,
                budgets=budgets,
                acquisition_items=acquisition_items,
                evaluation_items=evaluation_items,
            )
        )
    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        raise ValueError("no pairs were scored")

    keyed = frame.set_index(["benchmark", "subject_id", "budget"])
    pair_alc = (
        keyed["brier"].groupby(level=["benchmark", "subject_id"]).agg(
            lambda s: brier_alc(dict(zip(s.index.get_level_values("budget"), s)))
        )
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


__all__ = ["EvalResult", "Predict", "evaluate", "format_summary", "score_pair"]
