"""Brier score, expected calibration error, and Brier ALC.

The competition scores each subject-benchmark pair at label budgets
0, 1, 3, 7, 15, 31, then averages the weighted curve over pairs. On
x = log2(n + 1) those budgets sit at 0, 1, 2, 3, 4, 5, so a trapezoid gives
the published 0.1 / 0.2 / 0.2 / 0.2 / 0.2 / 0.1 weights exactly.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

BUDGETS: tuple[int, ...] = (0, 1, 3, 7, 15, 31)


def brier(labels: Sequence[float], probabilities: Sequence[float]) -> float:
    """Mean squared error between predicted probability and observed 0/1."""
    y = np.asarray(labels, dtype=float)
    p = np.asarray(probabilities, dtype=float)
    if y.size == 0:
        raise ValueError("brier needs at least one response")
    if y.shape != p.shape:
        raise ValueError("labels and probabilities must align")
    return float(np.mean((p - y) ** 2))


def ece(labels: Sequence[float], probabilities: Sequence[float], bins: int = 10) -> float:
    """Expected calibration error over equal-width probability bins.

    Diagnostic only: the competition reports it but does not rank on it. Empty
    bins contribute nothing.
    """
    y = np.asarray(labels, dtype=float)
    p = np.asarray(probabilities, dtype=float)
    if y.size == 0:
        raise ValueError("ece needs at least one response")
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(p, edges[1:-1], right=True), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        mask = index == b
        if not mask.any():
            continue
        total += mask.sum() / y.size * abs(float(p[mask].mean() - y[mask].mean()))
    return float(total)


def alc_weights(budgets: Sequence[int] = BUDGETS) -> np.ndarray:
    """Trapezoid weights on x = log2(n + 1), normalised to sum to one."""
    x = np.log2(np.asarray(budgets, dtype=float) + 1.0)
    if x.size < 2:
        raise ValueError("ALC needs at least two budgets")
    weights = np.empty_like(x)
    weights[0] = (x[1] - x[0]) / 2.0
    weights[-1] = (x[-1] - x[-2]) / 2.0
    weights[1:-1] = (x[2:] - x[:-2]) / 2.0
    return weights / weights.sum()


def brier_alc(scores: Mapping[int, float]) -> float:
    """Weighted area under one pair's Brier-vs-budget curve."""
    budgets = tuple(sorted(scores))
    weights = alc_weights(budgets)
    values = np.array([scores[b] for b in budgets], dtype=float)
    return float(np.dot(weights, values))


def summarise(scores: Mapping[int, float]) -> dict:
    """Curve summary for one pair or an averaged suite."""
    return {
        "brier_alc": brier_alc(scores),
        **{f"brier_{b}": float(scores[b]) for b in sorted(scores)},
    }


__all__ = ["BUDGETS", "brier", "ece", "alc_weights", "brier_alc", "summarise"]
