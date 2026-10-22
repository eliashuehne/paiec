"""Compare completed evaluation runs pair by pair.

Per-item predictions are sufficient statistics for Brier. For an item with ``w``
recorded responses of which ``k`` are one, its contribution is
``k(1-p)^2 + (w-k)p^2``; averaging that over responses reproduces the harness's
per-pair score exactly. Comparing runs therefore needs no new inference and no
access to the 23 GB trace payload.

Differences are paired by subject-benchmark pair, because the competition
weights pairs equally, and are reported with a paired t-test plus a seeded
bootstrap interval over pairs. The t-test assumes roughly normal pair
differences and is unreliable with very few pairs; the bootstrap interval is
reported alongside it rather than instead of it.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .metrics import brier_alc

PASS_COLUMNS = ("benchmark", "subject_id", "item_id", "budget", "probability",
                "n_responses", "n_outcomes_one")
SCORE_COLUMNS = ("benchmark", "subject_id", "budget", "brier")
NEGLIGIBLE = 1e-12
BOOTSTRAP_SAMPLES = 10000


# --- Brier reconstruction -------------------------------------------------

def pair_curves(items: pd.DataFrame) -> dict[tuple[str, str], dict[int, float]]:
    """Rebuild per-pair, per-budget Brier from logged item predictions."""
    missing = [column for column in PASS_COLUMNS if column not in items.columns]
    if missing:
        raise ValueError(f"prediction rows lack {missing}")
    if items.empty:
        raise ValueError("no prediction rows")
    curves: dict[tuple[str, str], dict[int, float]] = {}
    for (benchmark, subject), group in items.groupby(["benchmark", "subject_id"], sort=True):
        per_budget: dict[int, float] = {}
        for budget, rows in group.groupby("budget", sort=True):
            probability = rows["probability"].to_numpy(dtype=float)
            responses = rows["n_responses"].to_numpy(dtype=float)
            ones = rows["n_outcomes_one"].to_numpy(dtype=float)
            total = responses.sum()
            if total <= 0:
                raise ValueError(f"{benchmark}/{subject} budget {budget} has no responses")
            squared = ones * (1.0 - probability) ** 2 + (responses - ones) * probability ** 2
            per_budget[int(budget)] = float(squared.sum() / total)
        curves[(str(benchmark), str(subject))] = per_budget
    return curves


def curves_from_pairs(scores: pd.DataFrame) -> dict[tuple[str, str], dict[int, float]]:
    """Read per-pair, per-budget Brier from a harness score table."""
    missing = [column for column in SCORE_COLUMNS if column not in scores.columns]
    if missing:
        raise ValueError(f"score rows lack {missing}")
    curves: dict[tuple[str, str], dict[int, float]] = {}
    for (benchmark, subject), group in scores.groupby(["benchmark", "subject_id"], sort=True):
        curves[(str(benchmark), str(subject))] = {
            int(budget): float(value) for budget, value in zip(group["budget"], group["brier"])
        }
    return curves


def alc_from_curves(curves: Mapping[tuple[str, str], Mapping[int, float]]) -> pd.Series:
    """One Brier ALC per pair, sorted by benchmark then subject."""
    if not curves:
        raise ValueError("no pairs to score")
    values = {key: brier_alc(dict(value)) for key, value in curves.items()}
    return pd.Series(values, name="brier_alc").sort_index()


# --- Run inputs -----------------------------------------------------------

@dataclass
class RunScores:
    """One completed run, either item-level or pair-level."""

    label: str
    source: str
    curves: pd.Series  # per-pair ALC
    items: pd.DataFrame | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def n_pairs(self) -> int:
        return len(self.curves)

    @property
    def brier_alc(self) -> float:
        return float(self.curves.mean())

    def fingerprint(self) -> set[tuple[str, str, str]] | None:
        """Evaluation item IDs, if the run kept item-level predictions."""
        if self.items is None:
            return None
        rows = self.items[self.items["budget"].eq(self.items["budget"].min())]
        return {(str(b), str(s), str(i)) for b, s, i, _ in
                rows[["benchmark", "subject_id", "item_id", "budget"]].itertuples(index=False, name=None)}

    def restricted(self, items: set[tuple[str, str, str]] | None,
                   pairs: set[tuple[str, str]] | None) -> "RunScores":
        """Re-score on a shared subset of pairs and items."""
        if self.items is not None and items is not None:
            mask = [
                (str(b), str(s), str(i)) in items
                for b, s, i in zip(self.items["benchmark"], self.items["subject_id"], self.items["item_id"])
            ]
            subset = self.items[mask]
            if subset.empty:
                raise ValueError(f"{self.label} shares no scored items with the reference")
            curves = alc_from_curves(pair_curves(subset))
            return RunScores(self.label, self.source, curves, subset, self.metadata)
        if pairs is not None:
            keep = self.curves.index.intersection(pd.Index(sorted(pairs)))
            if keep.empty:
                raise ValueError(f"{self.label} shares no scored pairs with the reference")
            return RunScores(self.label, self.source, self.curves.loc[keep], self.items, self.metadata)
        return self


def shared_subset(reference: RunScores, other: RunScores) -> tuple[set | None, set]:
    """Keys the two runs have in common, and the pairs they cover."""
    left, right = reference.fingerprint(), other.fingerprint()
    if left is not None and right is not None:
        shared_items = left & right
        shared_pairs = {(benchmark, subject) for benchmark, subject, _ in shared_items}
        return shared_items, shared_pairs
    shared_pairs = set(reference.curves.index) & set(other.curves.index)
    return None, shared_pairs


def covers_everything(reference: RunScores, other: RunScores, shared_items, shared_pairs) -> bool:
    """Whether the shared keys are all of both runs' scored inputs.

    Item-level runs can be checked exactly. Pair-level runs can only be checked
    by pair, and a mixed pair is not verifiable, so it counts as partial.
    """
    left, right = reference.fingerprint(), other.fingerprint()
    if left is not None and right is not None:
        return shared_items is not None and len(shared_items) == len(left) == len(right)
    if left is None and right is None:
        return set(reference.curves.index) == set(other.curves.index)
    return False


# --- Paired statistics ----------------------------------------------------

def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (Lentz's method)."""
    tiny, epsilon = 1e-300, 3e-16
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        coefficient = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + coefficient * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + coefficient / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        coefficient = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + coefficient * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + coefficient / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < epsilon:
            break
    return h


def regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    """I_x(a, b), the CDF of a beta distribution."""
    if not 0.0 <= x <= 1.0:
        raise ValueError("x must lie in [0, 1]")
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0
    log_beta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(log_beta + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def paired_t_test(differences: Sequence[float]) -> tuple[float, float]:
    """Two-sided paired t-test; returns (t, p). Perfectly flat input gives p=1."""
    values = np.asarray(differences, dtype=float)
    n = values.size
    if n < 2:
        raise ValueError("a paired test needs at least two pairs")
    mean = float(values.mean())
    deviation = float(values.std(ddof=1))
    if deviation == 0.0:
        return (0.0, 1.0) if mean == 0.0 else (math.copysign(math.inf, mean), 0.0)
    statistic = mean / (deviation / math.sqrt(n))
    p = regularized_incomplete_beta((n - 1) / 2.0, 0.5, (n - 1) / ((n - 1) + statistic ** 2))
    return statistic, float(min(max(p, 0.0), 1.0))


def bootstrap_interval(differences: Sequence[float], *, level: float = 0.95,
                       samples: int = BOOTSTRAP_SAMPLES, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap interval for the mean paired difference."""
    values = np.asarray(differences, dtype=float)
    if values.size < 2:
        raise ValueError("a bootstrap interval needs at least two pairs")
    generator = np.random.default_rng(seed)
    draws = generator.choice(values, size=(samples, values.size), replace=True).mean(axis=1)
    tail = (1.0 - level) / 2.0
    return float(np.quantile(draws, tail)), float(np.quantile(draws, 1.0 - tail))


@dataclass(frozen=True)
class PairedResult:
    """One run measured against the reference on shared pairs."""

    label: str
    n_pairs: int
    mean_delta: float
    median_delta: float
    wins: int
    losses: int
    ties: int
    t: float
    p: float
    ci_low: float
    ci_high: float
    partial: bool


def compare_to(reference: RunScores, other: RunScores, *, seed: int = 0,
               strict: bool = True, level: float = 0.95) -> PairedResult:
    """Paired comparison of one run against a reference run.

    A positive delta means the run is worse than the reference, since Brier is
    lower-is-better.
    """
    shared_items, shared_pairs = shared_subset(reference, other)
    identical = covers_everything(reference, other, shared_items, shared_pairs)
    if strict and not identical:
        raise ValueError(
            f"{other.label} and {reference.label} do not score the same items; "
            "pass strict=False (--allow-partial) to compare their shared subset"
        )
    left = reference.restricted(shared_items, shared_pairs).curves
    right = other.restricted(shared_items, shared_pairs).curves
    keys = left.index.intersection(right.index)
    if len(keys) < 2:
        raise ValueError(
            f"{other.label} shares {len(keys)} pair(s) with {reference.label}; "
            "a paired comparison needs at least two"
        )
    differences = (right.loc[keys] - left.loc[keys]).to_numpy(dtype=float)
    statistic, p = paired_t_test(differences)
    low, high = bootstrap_interval(differences, level=level, seed=seed)
    return PairedResult(
        label=other.label,
        n_pairs=len(keys),
        mean_delta=float(differences.mean()),
        median_delta=float(np.median(differences)),
        wins=int((differences < -NEGLIGIBLE).sum()),
        losses=int((differences > NEGLIGIBLE).sum()),
        ties=int((np.abs(differences) <= NEGLIGIBLE).sum()),
        t=float(statistic),
        p=float(p),
        ci_low=low,
        ci_high=high,
        partial=not identical,
    )


def summary_table(runs: Iterable[RunScores]) -> pd.DataFrame:
    """One row per run with its mean score and spread across pairs."""
    rows = []
    for run in runs:
        rows.append({
            "run": run.label,
            "pairs": run.n_pairs,
            "brier_alc": run.brier_alc,
            "sd": float(run.curves.std(ddof=1)) if run.n_pairs > 1 else float("nan"),
            "source": run.source,
        })
    return pd.DataFrame(rows).sort_values("brier_alc").reset_index(drop=True)


def by_benchmark(runs: Mapping[str, RunScores]) -> pd.DataFrame:
    """Mean pair ALC per benchmark for each run."""
    frame = pd.DataFrame({
        label: run.curves.groupby(level=0).mean() for label, run in runs.items()
    })
    return frame.sort_index()


__all__ = [
    "PairedResult", "RunScores", "alc_from_curves", "bootstrap_interval", "by_benchmark",
    "compare_to", "covers_everything", "curves_from_pairs", "pair_curves", "paired_t_test",
    "regularized_incomplete_beta", "shared_subset", "summary_table",
]
