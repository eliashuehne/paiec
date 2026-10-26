"""Competition-faithful Brier-ALC scoring over subject-benchmark pairs.

The platform splits each pair's items 50/50 into acquisition and evaluation
pools, scores the same evaluation items at budgets 0/1/3/7/15/31, reveals
labels only from acquisition items, and reports
``0.1 B0 + 0.2 B1 + 0.2 B3 + 0.2 B7 + 0.2 B15 + 0.1 B31`` averaged equally over
pairs. ``labeled`` entries come from *all* sampled pairs, so a prediction for one
subject may use another subject's revealed outcomes on the same benchmark. That
channel is what makes the new benchmark's difficulty estimable at all, and the
earlier pipeline discarded it.

Every Brier value here is weighted by recorded responses: an item attempted
three times contributes three targets, exactly as the platform counts them.
"""
from dataclasses import dataclass, field
import math
from typing import Callable, Iterable, Literal, Sequence

import numpy as np

from .feature_preparation import stable_hash
from .metrics import BUDGETS, alc_weights
from .schema import canonical_identity, clean

REGIMES = ("all", "shared_subject", "cold_subject")


@dataclass(frozen=True)
class PairEpisode:
    """One sampled subject-benchmark pair with its fixed 50/50 item split."""

    cell_index: int
    cell_key: str
    benchmark: str
    benchmark_id: str
    subject_key: str
    acquisition: tuple[str, ...]
    evaluation: tuple[str, ...]
    n_responses: dict[str, int]
    label_sum: dict[str, int]

    @property
    def n_evaluation_responses(self):
        return sum(self.n_responses[item] for item in self.evaluation)

    @property
    def rate(self):
        total = sum(self.n_responses.values())
        return sum(self.label_sum.values()) / total


@dataclass
class Draw:
    """One seeded evaluation draw, fixed across budgets and predictors."""

    seed: int
    regime: str
    pairs: list[PairEpisode]
    excluded_cells: int = 0

    def by_benchmark(self, benchmark_id):
        return [pair for pair in self.pairs if pair.benchmark_id == benchmark_id]


def split_pair(cell, labels, seed, fraction=0.5):
    """Deterministic 50/50 item split; all responses of an item stay together."""
    items = [str(value) for value in labels["item_id"]]
    if len(items) < 2:
        return None
    ordered = sorted(items, key=lambda item: stable_hash({"seed": seed, "cell": cell["cell_key"],
                                                          "item": item}))
    half = int(len(ordered) * fraction)
    half = min(max(half, 1), len(ordered) - 1)
    acquisition, evaluation = ordered[:half], ordered[half:]
    arrival = sorted(acquisition, key=lambda item: stable_hash({"seed": seed, "cell": cell["cell_key"],
                                                               "arrival": item}))
    counts = {str(row.item_id): int(row.n_responses) for row in labels.itertuples(index=False)}
    sums = {str(row.item_id): int(row.label_sum) for row in labels.itertuples(index=False)}
    return PairEpisode(cell_index=int(cell["cell_index"]), cell_key=cell["cell_key"],
                       benchmark=cell["benchmark"], benchmark_id=cell["benchmark_id"],
                       subject_key=canonical_identity(clean(cell["subject"].get("normalized_name"))),
                       acquisition=tuple(arrival),
                       evaluation=tuple(evaluation), n_responses=counts, label_sum=sums)


def benchmark_presence(cells):
    """Map each named model to the benchmarks it appears in."""
    presence = {}
    for cell in cells:
        name = cell["subject"]["normalized_name"].strip().lower()
        if name:
            presence.setdefault(name, set()).add(cell["benchmark"])
    return presence


def select_draw(cells, labels, *, seed, max_pairs=1000, regime="all", min_items=80,
                benchmarks=None, directions=None):
    """Sample pairs by hash and split them; the regimes come from name overlap."""
    if regime not in REGIMES:
        raise ValueError("Unknown subject-coverage regime: " + str(regime))
    eligible = [cell for cell in cells if cell["n_items"] >= min_items]
    if benchmarks is not None:
        allowed = set(benchmarks)
        eligible = [cell for cell in eligible if cell["benchmark"] in allowed]
    if directions is not None:
        allowed_directions = set(directions)
        eligible = [cell for cell in eligible if cell["direction"] in allowed_directions]
    if regime != "all":
        presence = benchmark_presence(cells)
        keep = []
        for cell in eligible:
            name = cell["subject"]["normalized_name"].strip().lower()
            spanning = len(presence.get(name, ())) if name else 0
            if (regime == "shared_subject" and spanning >= 2) or (regime == "cold_subject" and spanning == 1):
                keep.append(cell)
        eligible = keep
    order = sorted(eligible, key=lambda cell: stable_hash({"draw": seed, "cell": cell["cell_key"]}))
    chosen = order[:max_pairs]
    pairs = []
    for cell in chosen:
        pair = split_pair(cell, labels[cell["cell_index"]], seed)
        if pair is not None:
            pairs.append(pair)
    if not pairs:
        raise ValueError("Evaluation draw selected no pairs")
    return Draw(seed=seed, regime=regime, pairs=pairs, excluded_cells=len(eligible) - len(chosen))


def revealed_units(pair, budget):
    """Item arrival prefix whose response count fits inside the budget.

    Candidates are items in a fixed order; revealing an item reveals all of its
    recorded responses. An item that would overshoot the budget is skipped
    rather than partially revealed, which only matters where trials exist.
    """
    if budget <= 0:
        return []
    revealed, used = [], 0
    for item in pair.acquisition:
        count = pair.n_responses[item]
        if used + count <= budget:
            revealed.append(item)
            used += count
    return revealed


def label_list(pair, item):
    """The recorded binary responses of one item, in a deterministic order."""
    count, successes = pair.n_responses[item], pair.label_sum[item]
    values = [1] * successes + [0] * (count - successes)
    return sorted(values, key=lambda value: stable_hash({"value": value, "item": item, "n": count}))


def evidence_for(draw, pair, budget):
    """Own-pair and same-benchmark labels available at this budget.

    Returns ``(own, other)`` label lists. ``other`` excludes the scored pair's
    own labels and never contains an evaluation item's outcome.
    """
    own = [value for item in revealed_units(pair, budget) for value in label_list(pair, item)]
    other = []
    for sibling in draw.pairs:
        if sibling.cell_key == pair.cell_key or sibling.benchmark_id != pair.benchmark_id:
            continue
        for item in revealed_units(sibling, budget):
            other.extend(label_list(sibling, item))
    return own, other


class Predictor:
    """Base probability per item plus an evidence update."""

    name = "base"
    uses_pair_labels = False
    uses_benchmark_labels = False

    def base(self, pair: PairEpisode) -> np.ndarray:
        raise NotImplementedError

    def update(self, base: np.ndarray, own: Sequence[int], other: Sequence[int]) -> np.ndarray:
        return base


class ConstantPredictor(Predictor):
    def __init__(self, name, value=0.5):
        if not 0 <= value <= 1:
            raise ValueError("Constant predictions must be probabilities")
        self.name, self.value = name, float(value)

    def base(self, pair):
        return np.full(len(pair.evaluation), self.value, dtype=float)


class OracleRatePredictor(Predictor):
    """Diagnostic only: uses the evaluation items' own outcomes."""

    name = "oracle_cell_rate"

    def base(self, pair):
        return np.full(len(pair.evaluation), pair.rate, dtype=float)


class EvidencePredictor(Predictor):
    """Shrinkage toward a base prior, optionally pooling the same benchmark.

    ``own_weight`` scales the scored pair's revealed labels, ``other_weight``
    the same-benchmark labels of other subjects, and ``prior_strength`` acts as
    pseudo-counts at the prior probability.
    """

    def __init__(self, name, *, prior_strength=5.0, own_weight=1.0, other_weight=0.0,
                 prior: Callable[[PairEpisode], float] | Literal["base", "half"] = "base",
                 base_predictor: Predictor | None = None):
        self.name = name
        self.prior_strength = float(prior_strength)
        self.own_weight = float(own_weight)
        self.other_weight = float(other_weight)
        if prior_strength < 0 or own_weight < 0 or other_weight < 0 or min(own_weight, other_weight) < 0:
            raise ValueError("Evidence weights cannot be negative")
        if not own_weight and not other_weight and not prior_strength:
            raise ValueError("A predictor that ignores all evidence must declare a prior strength")
        self.prior = prior
        self.base_predictor = base_predictor

    def base(self, pair):
        if self.base_predictor is not None:
            return self.base_predictor.base(pair)
        if callable(self.prior):
            value = float(self.prior(pair))
            return np.full(len(pair.evaluation), value, dtype=float)
        return np.full(len(pair.evaluation), 0.5, dtype=float)

    def update(self, base, own, other):
        base = np.asarray(base, dtype=float)
        if not own and not other:
            return base
        prior = float(np.clip(base.mean(), 1e-6, 1 - 1e-6))
        successes = self.own_weight * np.sum(own, dtype=float) + self.other_weight * np.sum(other, dtype=float)
        count = self.own_weight * len(own) + self.other_weight * len(other)
        denominator = self.prior_strength + count
        if denominator <= 0:
            # Only evidence this arm weights at zero is available; keep the base.
            return base
        blended = (self.prior_strength * prior + successes) / denominator
        return np.full_like(base, float(np.clip(blended, 1e-6, 1 - 1e-6)))


def own_label_mean():
    return EvidencePredictor("acquired_label_mean", prior_strength=0.0, prior="half")


def fixed_beta(prior_strength=5.0):
    return EvidencePredictor(f"fixed_beta_k{prior_strength:g}", prior_strength=prior_strength, prior="half")


def constant_baseline():
    return ConstantPredictor("constant_0.5", 0.5)


def item_brier(pair, item_ids, probabilities, probability_weights=None):
    """Response-weighted Brier for one pair's evaluation items.

    ``n p^2 - 2 p S + S`` is exact for any mix of responses behind an item,
    because each response contributes ``(p - y)^2`` with ``y`` in {0, 1}.
    """
    total, score = 0, 0.0
    for index, item in enumerate(item_ids):
        count = pair.n_responses[item]
        successes = pair.label_sum[item]
        p = float(probabilities[index])
        if probability_weights is not None:
            p = float(np.clip(p * probability_weights[index], 1e-6, 1 - 1e-6))
        score += count * p * p - 2 * p * successes + successes
        total += count
    if not total:
        raise ValueError("Pair has no evaluation responses")
    return score / total


def pair_auroc(pair, item_ids, probabilities):
    """Within-pair ordering, response weighted; ties count as half.

    Counted over response pairs rather than items: an item attempted three times
    with two successes contributes two positive and one negative observation.
    A constant predictor therefore returns exactly 0.5.
    """
    scores = np.asarray([float(probabilities[index]) for index in range(len(item_ids))], dtype=float)
    targets = np.array([pair.label_sum[item] / pair.n_responses[item] for item in item_ids], dtype=float)
    weights = np.array([pair.n_responses[item] for item in item_ids], dtype=float)
    positive = weights * targets
    negative = weights * (1 - targets)
    total_positive, total_negative = positive.sum(), negative.sum()
    if total_positive <= 0 or total_negative <= 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    numerator = 0.0
    seen_negative = 0.0
    position = 0
    while position < len(sorted_scores):
        end = position
        while end + 1 < len(sorted_scores) and sorted_scores[end + 1] == sorted_scores[position]:
            end += 1
        group = order[position:end + 1]
        group_positive = positive[group].sum()
        group_negative = negative[group].sum()
        numerator += group_positive * (seen_negative + 0.5 * group_negative)
        seen_negative += group_negative
        position = end + 1
    return float(numerator / (total_positive * total_negative))


def draw_evidence(draw, *, budgets=BUDGETS):
    """Cache revealed own/same-benchmark labels once per draw and budget."""
    by_benchmark = {}
    for pair in draw.pairs:
        by_benchmark.setdefault(pair.benchmark_id, []).append(pair)
    cache = {}
    for budget in budgets:
        revealed = {pair.cell_key: [value for item in revealed_units(pair, budget)
                                    for value in label_list(pair, item)] for pair in draw.pairs}
        cache[int(budget)] = {pair.cell_key: (
            revealed[pair.cell_key],
            [value for sibling in by_benchmark[pair.benchmark_id]
             if sibling.cell_key != pair.cell_key
             for value in revealed[sibling.cell_key]]) for pair in draw.pairs}
    return cache


def score_draw(draw, predictor, *, budgets=BUDGETS, with_auroc=True, evidence_cache=None):
    """Score one predictor on one draw at every budget."""
    evidence_cache = draw_evidence(draw, budgets=budgets) if evidence_cache is None else evidence_cache
    rows = []
    for budget in budgets:
        for pair in draw.pairs:
            own, other = evidence_cache[int(budget)][pair.cell_key]
            base = predictor.base(pair)
            values = predictor.update(base, own, other)
            record = {"cell_key": pair.cell_key, "benchmark": pair.benchmark,
                      "benchmark_id": pair.benchmark_id, "budget": int(budget),
                      "brier": item_brier(pair, pair.evaluation, values),
                      "own_labels": len(own), "other_labels": len(other),
                      "observed_rate": pair.rate}
            if with_auroc:
                record["auroc"] = pair_auroc(pair, pair.evaluation, values)
            rows.append(record)
    per_pair = {}
    for row in rows:
        per_pair.setdefault(row["cell_key"], {})[row["budget"]] = row["brier"]
    weights = dict(zip(budgets, alc_weights(budgets)))
    pair_alc = {key: float(sum(weights[budget] * value for budget, value in curve.items()))
                for key, curve in per_pair.items()}
    curve = {int(budget): float(np.mean([row["brier"] for row in rows if row["budget"] == budget]))
             for budget in budgets}
    auroc = [row["auroc"] for row in rows if with_auroc and row.get("auroc") is not None]
    return {
        "name": predictor.name,
        "pairs": len(draw.pairs),
        "curve": curve,
        "brier_alc": float(np.mean(list(pair_alc.values()))),
        "brier_alc_curve": float(sum(weights[budget] * curve[budget] for budget in budgets)),
        "auroc": float(np.mean(auroc)) if auroc else None,
        "pairs_with_evidence": sum(1 for row in rows if row["budget"] and row["own_labels"]),
        "pairs_with_benchmark_evidence": sum(1 for row in rows
                                             if row["budget"] and row["other_labels"]),
        "mean_other_labels": float(np.mean([row["other_labels"] for row in rows
                                            if row["other_labels"]])) if any(
            row["other_labels"] for row in rows) else 0.0,
        "rows": rows,
    }


def oracle_curve(draw, budgets=BUDGETS):
    """Diagnostic bound: the pair's own evaluation rate, known exactly."""
    curve = {budget: float(np.mean([item_brier(pair, pair.evaluation,
                                               np.full(len(pair.evaluation), pair.rate))
                                    for pair in draw.pairs])) for budget in budgets}
    weights = dict(zip(budgets, alc_weights(budgets)))
    return {"name": "oracle_cell_rate", "curve": curve,
            "brier_alc": float(sum(weights[budget] * curve[budget] for budget in budgets)),
            "pairs": len(draw.pairs)}


def compare(arms, reference):
    """Paired difference against one reference arm on identical pairs."""
    if reference not in arms:
        raise ValueError("Reference arm is missing")
    baseline = {row["cell_key"]: {} for row in arms[reference]["rows"]}
    for row in arms[reference]["rows"]:
        baseline[row["cell_key"]][row["budget"]] = row["brier"]
    out = {}
    for name, arm in arms.items():
        if name == reference:
            continue
        deltas = []
        for row in arm["rows"]:
            deltas.append({row["budget"]: row["brier"] - baseline[row["cell_key"]][row["budget"]]})
        curve = {budget: float(np.mean([delta[budget] for delta in deltas])) for budget in deltas[0]}
        weights = dict(zip(sorted(curve), alc_weights(sorted(curve))))
        out[name] = {"curve": curve,
                     "brier_alc": float(sum(weights[budget] * curve[budget] for budget in curve))}
    return out


def response_scale_conflict(cells, reference="higher_is_better"):
    """Report direction mix instead of silently pooling or flipping it."""
    counts = {}
    for cell in cells:
        counts[cell["direction"]] = counts.get(cell["direction"], 0) + 1
    return {"reference": reference, "cells_by_direction": counts,
            "pooled_across_directions": any(key != reference for key in counts)}


__all__ = ["BUDGETS", "Draw", "PairEpisode", "Predictor", "ConstantPredictor", "EvidencePredictor",
           "OracleRatePredictor", "benchmark_presence", "compare", "constant_baseline", "draw_evidence",
           "evidence_for", "fixed_beta", "item_brier", "label_list", "oracle_curve", "own_label_mean",
           "pair_auroc", "response_scale_conflict", "revealed_units", "score_draw", "select_draw",
           "split_pair"]
