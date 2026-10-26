"""Does a model's success belong to the item, or to the model-item pair?

The rate model estimated a pair's overall success rate and reached within-pair
AUROC 0.5027 with fifteen surface counts, so it told us nothing about which
questions a system gets wrong. That leaves two very different worlds:

* item difficulty is shared — a hard question is hard for everyone, so other
  systems' revealed outcomes on that question predict this system's outcome;
* item difficulty is model specific — one system fails code, another fails
  arithmetic — and no static description of a question can rank it for a system.

Which one holds decides whether item modelling is worth funding at all. This
module measures it with no GPU and no new data:

* a Rasch fit per benchmark (``logit p = theta(subject) - b(item)``) gives the
  spread of item difficulty, of model ability, and of what neither explains;
* split-half reliability of item facility says whether the same questions look
  hard to disjoint pools of systems;
* held-out transfer says whether one pool's item difficulty ranks another pool's
  individual outcomes, which is exactly what an item model would exploit;
* the oracle ceilings price the ordering prize on the same draws the scorer
  uses, and report how much of it revealed labels could actually reach.

Every ceiling uses evaluation outcomes and exists only to bound a score. None of
it is a predictor, and nothing here writes back into the pair matrix.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .competition_scorer import BUDGETS, item_brier, revealed_units
from .feature_preparation import stable_hash
from .metrics import alc_weights
from .rate_model import sigmoid


def item_frames(cells, labels, *, direction="higher_is_better", min_items=80):
    """Per benchmark, subject-by-item counts and successes as flat arrays.

    Only cells of the judged direction enter: pooling a lower-is-better
    benchmark would invert its labels. Cells and items keep a deterministic
    order, so every run addresses the same entry with the same index.
    """
    admitted = [cell for cell in cells
                if cell["direction"] == direction and cell["n_items"] >= min_items]
    if not admitted:
        return {}
    keys = {int(cell["cell_index"]): cell["cell_key"] for cell in cells}
    pieces = []
    for cell in admitted:
        table = labels[cell["cell_index"]]
        pieces.append(pd.DataFrame({"benchmark": cell["benchmark"],
                                    "cell_index": int(cell["cell_index"]),
                                    "item_id": pd.Series(list(table["item_id"])).astype(str).to_numpy(),
                                    "n": np.asarray(table["n_responses"], dtype=float),
                                    "s": np.asarray(table["label_sum"], dtype=float)}))
    combined = pd.concat(pieces, ignore_index=True)
    frames = {}
    for benchmark, group in combined.groupby("benchmark", sort=True):
        order = group["cell_index"].unique()
        position = {int(value): index for index, value in enumerate(order)}
        codes, item_ids = pd.factorize(group["item_id"])
        frames[benchmark] = {
            "cells": [{"cell_index": int(value), "cell_key": keys[int(value)]} for value in order],
            "items": {item: index for index, item in enumerate(item_ids)},
            "item_ids": [str(item) for item in item_ids],
            "rows": group["cell_index"].map(position).to_numpy(dtype=np.int64),
            "cols": codes.astype(np.int64),
            "n": group["n"].to_numpy(dtype=float),
            "s": group["s"].to_numpy(dtype=float)}
    return frames


def item_observations(frame):
    """One weighted observation per subject-item, however many trials it had.

    A cell records every recorded trial; the fit wants one success rate per
    subject-item, weighted by how many responses stand behind it.
    """
    width = max(1, len(frame["item_ids"]))
    keys = frame["rows"] * width + frame["cols"]
    order = np.argsort(keys, kind="mergesort")
    keys, rows, cols = keys[order], frame["rows"][order], frame["cols"][order]
    counts, sums = frame["n"][order], frame["s"][order]
    unique, start = np.unique(keys, return_index=True)
    del unique
    return rows[start], cols[start], np.add.reduceat(counts, start), np.add.reduceat(sums, start)


def rasch_fit(frame, *, subject_penalty=1.0, item_penalty=1.0, iterations=60, tolerance=1e-6):
    """Alternating MAP fit of ``logit p = theta(subject) - b(item)``.

    Both blocks are penalised, so a subject with one attempt and an item with
    one attempt cannot run to infinity. Difficulty is centred: ``b`` is a
    deviation from the benchmark's average item, because the pair's overall
    level is the evidence rule's job rather than this fit's.
    """
    if subject_penalty < 0 or item_penalty < 0:
        raise ValueError("Penalties cannot be negative")
    rows, cols, counts, sums = item_observations(frame)
    if not len(rows):
        raise ValueError("Benchmark has no item observations")
    n_subjects, n_items = len(frame["cells"]), len(frame["item_ids"])
    theta, difficulty = np.zeros(n_subjects), np.zeros(n_items)
    for _ in range(int(iterations)):
        previous = difficulty.copy()
        probability = sigmoid(theta[rows] - difficulty[cols])
        weight = counts * probability * (1 - probability)
        theta = theta + np.bincount(rows, weights=sums - counts * probability,
                                    minlength=n_subjects) / (
            np.bincount(rows, weights=weight, minlength=n_subjects) + subject_penalty)
        probability = sigmoid(theta[rows] - difficulty[cols])
        weight = counts * probability * (1 - probability)
        difficulty = difficulty - np.bincount(cols, weights=sums - counts * probability,
                                              minlength=n_items) / (
            np.bincount(cols, weights=weight, minlength=n_items) + item_penalty)
        difficulty = difficulty - difficulty.mean()
        if np.max(np.abs(difficulty - previous)) < tolerance:
            break
    return {"theta": theta, "difficulty": difficulty, "rows": rows, "cols": cols,
            "counts": counts, "sums": sums}


def _halves(frame, seed):
    """Split subjects deterministically in two by hashed cell key."""
    return np.array([stable_hash({"seed": seed, "cell": cell["cell_key"]})[0] in "01234567"
                     for cell in frame["cells"]], dtype=bool)


def split_half_reliability(frame, *, seed=0, minimum=2):
    """Correlate item facility between two disjoint pools of systems.

    A hard question looks hard to both pools when difficulty belongs to the
    question. When the pools disagree, difficulty belongs to the model-item
    interaction and no description of the question can recover it.
    """
    first = _halves(frame, seed)
    rows, cols = frame["rows"], frame["cols"]
    sides = []
    for mask in (first, ~first):
        selected = mask[rows]
        total = np.bincount(cols[selected], weights=frame["n"][selected],
                            minlength=len(frame["item_ids"]))
        successes = np.bincount(cols[selected], weights=frame["s"][selected],
                                minlength=len(frame["item_ids"]))
        rate = (successes + 0.5) / (total + 1.0)
        sides.append((np.log(rate / (1 - rate)), total >= minimum))
    shared = sides[0][1] & sides[1][1]
    if shared.sum() < 3:
        return {"items": int(shared.sum()), "pearson": None}
    left, right = sides[0][0][shared], sides[1][0][shared]
    if left.std() == 0 or right.std() == 0:
        return {"items": int(shared.sum()), "pearson": None}
    return {"items": int(shared.sum()), "pearson": float(np.corrcoef(left, right)[0, 1])}


def _fit_subset(rows, cols, counts, sums, n_cells, n_items, keep, *, item_penalty=1.0,
                iterations=60):
    """Rasch fit on a subset of observations, keeping the full index space."""
    rows, cols, counts, sums = rows[keep], cols[keep], counts[keep], sums[keep]
    theta, difficulty = np.zeros(n_cells), np.zeros(n_items)
    for _ in range(int(iterations)):
        probability = sigmoid(theta[rows] - difficulty[cols])
        weight = counts * probability * (1 - probability)
        theta = theta + np.bincount(rows, weights=sums - counts * probability,
                                    minlength=n_cells) / (
            np.bincount(rows, weights=weight, minlength=n_cells) + 1.0)
        probability = sigmoid(theta[rows] - difficulty[cols])
        weight = counts * probability * (1 - probability)
        difficulty = difficulty - np.bincount(cols, weights=sums - counts * probability,
                                              minlength=n_items) / (
            np.bincount(cols, weights=weight, minlength=n_items) + item_penalty)
        difficulty = difficulty - difficulty.mean()
    return {"theta": theta, "difficulty": difficulty}


def _auc(scores, positive, negative):
    """Response-weighted AUC with ties counted as half."""
    order = np.argsort(scores, kind="mergesort")
    scores, positive, negative = scores[order], positive[order], negative[order]
    numerator, seen_negative, position = 0.0, 0.0, 0
    while position < len(scores):
        end = position
        while end + 1 < len(scores) and scores[end + 1] == scores[position]:
            end += 1
        group_positive = positive[position:end + 1].sum()
        group_negative = negative[position:end + 1].sum()
        numerator += group_positive * (seen_negative + 0.5 * group_negative)
        seen_negative += group_negative
        position = end + 1
    total_positive, total_negative = positive.sum(), negative.sum()
    if total_positive <= 0 or total_negative <= 0:
        return None
    return float(numerator / (total_positive * total_negative))


def transfer_auroc(frame, *, seed=0, minimum=2, item_penalty=1.0):
    """Rank one pool's items by difficulty learned from the other pool.

    This is an item model's mechanism in isolation: fit item difficulty on the
    systems in one half, then ask whether it ranks *individual* outcomes of the
    systems in the other half. Scored within each subject and averaged by
    responses, so the benchmark's overall level cannot flatter the result.
    """
    rows, cols, counts, sums = item_observations(frame)
    side = _halves(frame, seed)[rows]
    if side.all() or not side.any():
        return {"auc": None, "observations": 0, "items": 0}
    model = _fit_subset(rows, cols, counts, sums, len(frame["cells"]), len(frame["item_ids"]),
                        side, item_penalty=item_penalty)
    covered = np.bincount(cols[side], minlength=len(frame["item_ids"])) >= minimum
    selected = (~side) & covered[cols]
    if selected.sum() < 3:
        return {"auc": None, "observations": int(selected.sum()), "items": int(covered.sum())}
    scores = -model["difficulty"][cols][selected]
    subjects = rows[selected]
    weights = counts[selected]
    rate = np.clip(sums[selected] / np.where(weights > 0, weights, 1), 0, 1)
    positive, negative = weights * rate, weights * (1 - rate)
    numerator = denominator = 0.0
    for subject in np.unique(subjects):
        mask = subjects == subject
        pos, neg = positive[mask].sum(), negative[mask].sum()
        if pos <= 0 or neg <= 0:
            continue
        numerator += pos * neg * _auc(scores[mask], positive[mask], negative[mask])
        denominator += pos * neg
    if denominator <= 0:
        return {"auc": None, "observations": int(selected.sum()), "items": int(covered.sum())}
    return {"auc": float(numerator / denominator), "observations": int(selected.sum()),
            "items": int(covered.sum())}


def variance_components(frame, *, item_penalty=1.0, subject_penalty=1.0):
    """Spread of ability, of difficulty, and the residual neither explains."""
    fit = rasch_fit(frame, item_penalty=item_penalty, subject_penalty=subject_penalty)
    rows, cols, counts, sums = fit["rows"], fit["cols"], fit["counts"], fit["sums"]
    probability = sigmoid(fit["theta"][rows] - fit["difficulty"][cols])
    residual = ((sums - counts * probability) ** 2).sum() / max(1.0, counts.sum())
    return {"subjects": int(len(fit["theta"])), "items": int(len(fit["difficulty"])),
            "observations": int(len(rows)), "responses": float(counts.sum()),
            "sd_ability": float(fit["theta"].std()), "sd_difficulty": float(fit["difficulty"].std()),
            "residual_brier": float(residual)}


def oracle_ceilings(draw, *, budgets=BUDGETS, prior_strength=5.0):
    """Price the ordering prize and how much of it revealed labels could reach.

    ``own_item`` knows every evaluation item's own recorded success rate; no
    predictor may use it. ``sibling_facility`` knows the same quantity computed
    only from *other* subjects' revealed items on the benchmark, falling back to
    the pair's rate where nothing was revealed, so it bounds what item evidence
    can buy at that budget. ``covered_item_share`` says how often that fallback
    was unnecessary.
    """
    by_benchmark = {}
    for pair in draw.pairs:
        by_benchmark.setdefault(pair.benchmark_id, []).append(pair)
    rows = []
    for budget in budgets:
        revealed = {pair.cell_key: {item: (pair.n_responses[item], pair.label_sum[item])
                                    for item in revealed_units(pair, budget)} for pair in draw.pairs}
        own_scores, facility_scores, covered, total = [], [], 0, 0
        for pair in draw.pairs:
            own = [(pair.label_sum[item] + 0.5) / (pair.n_responses[item] + 1.0)
                   for item in pair.evaluation]
            counts, sums = {}, {}
            for sibling in by_benchmark[pair.benchmark_id]:
                if sibling.cell_key == pair.cell_key:
                    continue
                for item, (count, successes) in revealed[sibling.cell_key].items():
                    counts[item] = counts.get(item, 0) + count
                    sums[item] = sums.get(item, 0) + successes
            prior = sum(pair.label_sum.values()) / max(1, sum(pair.n_responses.values()))
            facility = []
            for item in pair.evaluation:
                total += 1
                if item in counts:
                    covered += 1
                    facility.append((sums[item] + prior_strength * prior) /
                                    (counts[item] + prior_strength))
                else:
                    facility.append(prior)
            own_scores.append(item_brier(pair, pair.evaluation, np.asarray(own)))
            facility_scores.append(item_brier(pair, pair.evaluation, np.asarray(facility)))
        rows.append({"budget": int(budget),
                     "own_item_oracle": float(np.mean(own_scores)),
                     "sibling_facility_oracle": float(np.mean(facility_scores)),
                     "covered_item_share": covered / max(1, total)})
    weights = dict(zip(budgets, alc_weights(budgets)))
    return {"rows": rows,
            "own_item_alc": float(sum(weights[row["budget"]] * row["own_item_oracle"] for row in rows)),
            "sibling_facility_alc": float(sum(weights[row["budget"]] * row["sibling_facility_oracle"]
                                              for row in rows))}


def summarise(frames, *, seed=0, minimum=2):
    """Aggregate the per-benchmark diagnostics into one report."""
    reliability, transfer, components = [], [], []
    for benchmark, frame in sorted(frames.items()):
        split = split_half_reliability(frame, seed=seed, minimum=minimum)
        if split["pearson"] is not None:
            reliability.append({"benchmark": benchmark, **split})
        moved = transfer_auroc(frame, seed=seed, minimum=minimum)
        if moved["auc"] is not None:
            transfer.append({"benchmark": benchmark, **moved})
        components.append({"benchmark": benchmark, **variance_components(frame)})

    def pooled(rows, key):
        total = sum(row["items"] for row in rows)
        return float(sum(row[key] * row["items"] for row in rows) / total) if total else None

    return {
        "benchmarks": len(frames),
        "reliability": {"per_benchmark": reliability, "pooled_pearson": pooled(reliability, "pearson")},
        "transfer": {"per_benchmark": transfer, "pooled_auc": pooled(transfer, "auc"),
                     "positive_benchmarks": sum(1 for row in transfer if row["auc"] > 0.5),
                     "benchmarks": len(transfer)},
        "components": components,
        "median_sd_ability": float(np.median([row["sd_ability"] for row in components])),
        "median_sd_difficulty": float(np.median([row["sd_difficulty"] for row in components])),
        "median_residual_brier": float(np.median([row["residual_brier"] for row in components])),
    }


__all__ = ["item_frames", "item_observations", "oracle_ceilings", "rasch_fit",
           "split_half_reliability", "summarise", "transfer_auroc", "variance_components"]
