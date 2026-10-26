"""Two-way ability x difficulty rate model with test-time evidence pooling.

Brier is dominated by a pair's overall success rate, and that rate is a sum of a
system's ability and a benchmark's difficulty. The earlier panel could not
separate them because no system appeared in two benchmarks; the pair matrix can.

Ability is estimated with shrinkage over canonical model names. A new
benchmark's difficulty is the unknown term, so it is estimated from task-text
statistics of its own items through a small ridge meta-regression fitted across
training benchmarks, then refined at test time from revealed labels of any
subject on the same benchmark. The item-level block is optional and is the
existing CatBoost task-structure model refit as a within-pair term.
"""
from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np

from .feature_preparation import stable_hash
from .schema import canonical_identity, clean

try:  # CatBoost is present only in the CPU fitting environment.
    from catboost import CatBoostClassifier
except ImportError:  # pragma: no cover - absent from the model environments
    CatBoostClassifier = None

ITEM_FEATURES = (
    "log_characters",
    "log_words",
    "mean_word_length",
    "digit_fraction",
    "upper_fraction",
    "log_newlines",
    "has_code_fence",
    "has_pipe",
    "log_options",
    "cjk_fraction",
    "cyrillic_fraction",
    "arabic_fraction",
    "json_like",
    "log_numeric_spans",
    "log_question_marks",
)

EPS = 1e-6

# Built from code points rather than ``\uXXXX`` escapes: pandas may delegate
# ``str.count`` to a pyarrow/RE2 engine, which rejects that escape form.
SCRIPT_CLASSES = {
    "cjk": "[" + chr(0x3400) + "-" + chr(0x9FFF) + "]",
    "cyrillic": "[" + chr(0x0400) + "-" + chr(0x04FF) + "]",
    "arabic": "[" + chr(0x0600) + "-" + chr(0x06FF) + "]",
}
OPTION_PATTERN = r"(?:^|\n)\s*[A-D][\).:]\s"


def logit(value):
    p = np.clip(np.asarray(value, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def empirical_logit(successes, trials):
    """Haldane-Anscombe logit: a cell with 0 or n successes is not infinite.

    Without the half-count correction, an 80-item cell at rate 0 contributes
    -6.9 and dominates every effect estimate; the fitted residual spread then
    measures those extremes rather than transfer.
    """
    successes = float(successes)
    trials = float(trials)
    if trials <= 0:
        raise ValueError("Empirical logits need a positive trial count")
    return math.log((successes + 0.5) / (trials - successes + 0.5))


def sigmoid(value):
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(value, dtype=float), -40.0, 40.0)))


def ability_key(subject):
    """Canonical name is the capability key; a display-only subject has none."""
    return canonical_identity(clean(subject.get("normalized_name")))


def cell_prior_weight(cell):
    """Fisher-information weight for one cell's logit rate."""
    rate = min(max(float(cell["rate"]), 1e-4), 1 - 1e-4)
    return max(float(cell["n_responses"]) * rate * (1 - rate), 1e-6)


@dataclass
class TwoWayFit:
    mu: float
    ability: dict
    difficulty: dict
    residual_sd: float
    raw_sd: float
    ability_sd: float
    difficulty_sd: float
    cells: int

    def as_report(self):
        return {"mu": self.mu, "cells": self.cells, "ability_sd": self.ability_sd,
                "difficulty_sd": self.difficulty_sd, "residual_sd": self.residual_sd,
                "raw_sd": self.raw_sd,
                "variance_share_explained": (1 - (self.residual_sd / self.raw_sd) ** 2)
                if self.raw_sd else None,
                "names": len(self.ability), "benchmarks": len(self.difficulty)}


def fit_two_way(cells, *, name_shrinkage=2.0, benchmark_shrinkage=2.0, iterations=40,
                direction_scope="higher_is_better"):
    """Alternating penalised fit of ``logit(rate) = mu + ability + difficulty``.

    Only cells in ``direction_scope`` contribute: pooling a lower-is-better
    benchmark's native labels into an ability estimate would invert them.
    """
    if name_shrinkage < 0 or benchmark_shrinkage < 0:
        raise ValueError("Shrinkage cannot be negative")
    observed = []
    for cell in cells:
        if cell["direction"] != direction_scope:
            continue
        name = ability_key(cell["subject"])
        if not name:
            continue
        observed.append((name, cell["benchmark"],
                         empirical_logit(cell["label_sum"], cell["n_responses"]),
                         cell_prior_weight(cell)))
    if len(observed) < 3:
        raise ValueError("The two-way fit needs at least three observed cells")
    weights = np.array([row[3] for row in observed], dtype=float)
    mu = float(np.average([row[2] for row in observed], weights=weights))
    ability = {row[0]: 0.0 for row in observed}
    difficulty = {row[1]: 0.0 for row in observed}
    for _ in range(iterations):
        buckets = {}
        for name, benchmark, value, weight in observed:
            buckets.setdefault(name, []).append((value - mu - difficulty[benchmark], weight))
        for name, rows in buckets.items():
            total = sum(weight for _, weight in rows)
            ability[name] = sum((value * weight for value, weight in rows), 0.0) / (total + name_shrinkage)
        buckets = {}
        for name, benchmark, value, weight in observed:
            buckets.setdefault(benchmark, []).append((value - mu - ability[name], weight))
        for benchmark, rows in buckets.items():
            total = sum(weight for _, weight in rows)
            difficulty[benchmark] = sum((value * weight for value, weight in rows), 0.0) / (
                total + benchmark_shrinkage)
    residuals = np.array([value - mu - ability[name] - difficulty[benchmark]
                          for name, benchmark, value, _ in observed])
    raw = np.array([row[2] for row in observed])
    return TwoWayFit(mu=mu, ability=ability, difficulty=difficulty,
                     residual_sd=float(residuals.std()), raw_sd=float(raw.std()),
                     ability_sd=float(np.std(list(ability.values()))),
                     difficulty_sd=float(np.std(list(difficulty.values()))),
                     cells=len(observed))


def item_feature_frame(items):
    """Vectorised task-text statistics for a table of supplied item strings."""
    import pandas as pd

    text = items["content"].astype(str)
    characters = text.str.len().clip(lower=1)
    words = (text.str.count(r"\s+") + 1).clip(lower=1)
    digits = text.str.count(r"\d")
    upper = text.str.count(r"[A-Z]")
    numeric_spans = text.str.count(r"\d+(?:\.\d+)?")
    frame = pd.DataFrame({
        "log_characters": np.log1p(characters),
        "log_words": np.log1p(words),
        "mean_word_length": characters / words,
        "digit_fraction": digits / characters,
        "upper_fraction": upper / characters,
        "log_newlines": np.log1p(text.str.count("\n")),
        "has_code_fence": text.str.contains("```", regex=False).astype(float),
        "has_pipe": text.str.contains("|", regex=False).astype(float),
        "log_options": np.log1p(text.str.count(OPTION_PATTERN)),
        "cjk_fraction": text.str.count(SCRIPT_CLASSES["cjk"]) / characters,
        "cyrillic_fraction": text.str.count(SCRIPT_CLASSES["cyrillic"]) / characters,
        "arabic_fraction": text.str.count(SCRIPT_CLASSES["arabic"]) / characters,
        "json_like": text.str.startswith("{").astype(float),
        "log_numeric_spans": np.log1p(numeric_spans),
        "log_question_marks": np.log1p(text.str.count(r"\?")),
    }).astype("float32")
    frame.insert(0, "item_id", items["item_id"].astype(str).values)
    frame.insert(0, "benchmark", items["benchmark"].astype(str).values)
    return frame


class ItemFeatureStore:
    """Per-benchmark item features with id lookup."""

    def __init__(self, frame):
        self.by_benchmark = {}
        for benchmark, group in frame.groupby("benchmark", sort=False):
            ids = pd_index(group["item_id"])
            self.by_benchmark[str(benchmark)] = (ids, group[list(ITEM_FEATURES)].to_numpy(dtype=float))

    def matrix(self, benchmark, item_ids):
        ids, values = self.by_benchmark[str(benchmark)]
        positions = ids.get_indexer([str(item) for item in item_ids])
        if (positions < 0).any():
            raise ValueError("Evaluation item is missing from the feature table")
        return values[positions]

    def mean(self, benchmark, item_ids):
        matrix = self.matrix(benchmark, item_ids)
        return matrix.mean(axis=0) if len(matrix) else np.zeros(len(ITEM_FEATURES))


def pd_index(values):
    import pandas as pd
    return pd.Index([str(value) for value in values])


def cell_feature_matrix(cells, labels, store):
    """Mean task-text features of each cell's own items."""
    out = {}
    for cell in cells:
        item_ids = labels[cell["cell_index"]]["item_id"]
        out[cell["cell_key"]] = store.mean(cell["benchmark"], item_ids)
    return out


def fit_difficulty(cells, difficulties, cell_features, *, alpha=3.0):
    """Ridge meta-regression from task-text statistics to benchmark difficulty."""
    rows, targets = [], []
    for cell in cells:
        if cell["benchmark"] not in difficulties:
            continue
        features = cell_features.get(cell["cell_key"])
        if features is None:
            continue
        rows.append(features)
        targets.append(difficulties[cell["benchmark"]])
    if len(rows) < 3:
        return {"alpha": float(alpha), "mean": np.zeros(len(ITEM_FEATURES)),
                "scale": np.ones(len(ITEM_FEATURES)), "coefficients": np.zeros(len(ITEM_FEATURES)),
                "intercept": float(np.mean(targets)) if targets else 0.0, "rows": len(rows)}
    matrix = np.asarray(rows, dtype=float)
    targets = np.clip(np.asarray(targets, dtype=float), -6.0, 6.0)
    mean, scale = matrix.mean(axis=0), matrix.std(axis=0)
    # A near-constant aggregate is a unit-scale feature, never a huge lever.
    scale[scale < 1e-6] = 1.0
    standard = (matrix - mean) / scale
    design = np.column_stack([np.ones(len(standard)), standard])
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ targets)
    return {"alpha": float(alpha), "mean": mean, "scale": scale,
            "coefficients": np.clip(coefficients[1:], -4.0, 4.0),
            "intercept": float(np.clip(coefficients[0], -4.0, 4.0)), "rows": len(rows)}


def predict_difficulty(state, features, *, bound=4.0):
    """Bounded logit offset: an unseen item mix cannot extrapolate without limit."""
    if features is None:
        return float(np.clip(state["intercept"], -bound, bound))
    standard = (np.asarray(features, dtype=float) - state["mean"]) / state["scale"]
    return float(np.clip(state["intercept"] + standard @ state["coefficients"], -bound, bound))


def difficulty_cv(cells, difficulties, cell_features, *, alpha, seed):
    """Leave-one-benchmark-out error of the difficulty meta-regression."""
    errors, baseline = [], []
    benchmarks = sorted({cell["benchmark"] for cell in cells if cell["benchmark"] in difficulties})
    for held in benchmarks:
        train = [cell for cell in cells if cell["benchmark"] != held]
        state = fit_difficulty(train, {b: d for b, d in difficulties.items() if b != held},
                               cell_features, alpha=alpha)
        for cell in cells:
            if cell["benchmark"] != held or cell["cell_key"] not in cell_features:
                continue
            predicted = predict_difficulty(state, cell_features[cell["cell_key"]])
            target = difficulties[held]
            errors.append((predicted - target) ** 2)
            baseline.append((state["intercept"] - target) ** 2)
    if not errors:
        return None
    return {"benchmarks": len(benchmarks), "mse": float(np.mean(errors)),
            "baseline_mse": float(np.mean(baseline))}


def thin_label(cell_key, item, n_responses, label_sum, seed):
    """One recorded response per item, chosen by hash, for CatBoost targets."""
    if label_sum <= 0:
        return 0
    if label_sum >= n_responses:
        return 1
    draw = int(stable_hash({"seed": seed, "cell": cell_key, "item": str(item)})[:8], 16)
    return 1 if draw / 0xFFFFFFFF < label_sum / n_responses else 0


def item_block_rows(cells, labels, cell_difficulty, store, *, items_per_cell, seed):
    """Build the CatBoost design matrix: prior column plus task features."""
    rows, targets, weights = [], [], []
    for cell in cells:
        if cell["cell_key"] not in cell_difficulty:
            continue
        prior = cell_difficulty[cell["cell_key"]]
        table = labels[cell["cell_index"]]
        order = sorted(range(len(table)), key=lambda index: stable_hash(
            {"seed": seed, "cell": cell["cell_key"], "item": str(table.iloc[index]["item_id"])}))
        chosen = order[:items_per_cell]
        matrix = store.matrix(cell["benchmark"], [table.iloc[index]["item_id"] for index in chosen])
        for position, index in enumerate(chosen):
            row = table.iloc[index]
            rows.append(np.concatenate([[prior], matrix[position]]))
            targets.append(thin_label(cell["cell_key"], row["item_id"], int(row["n_responses"]),
                                      int(row["label_sum"]), seed))
            weights.append(1.0 / max(1, len(chosen)))
    if not rows:
        raise ValueError("The item block needs at least one cell")
    return np.asarray(rows, dtype=float), np.asarray(targets, dtype=int), np.asarray(weights, dtype=float)


def fit_item_block(cells, labels, cell_prior_logit, store, *, items_per_cell, seed, threads,
                   iterations, depth, learning_rate, l2_leaf_reg=3.0):
    """Fit the within-pair item term around the supplied pair prior."""
    if CatBoostClassifier is None:
        raise RuntimeError("CatBoost is not installed in this environment")
    design, targets, weights = item_block_rows(cells, labels, cell_prior_logit, store,
                                               items_per_cell=items_per_cell, seed=seed)
    model = CatBoostClassifier(iterations=int(iterations), depth=int(depth),
                               learning_rate=float(learning_rate), l2_leaf_reg=float(l2_leaf_reg),
                               loss_function="Logloss", random_seed=int(seed),
                               thread_count=int(threads), verbose=False, allow_writing_files=False)
    model.fit(design, targets, sample_weight=weights)
    return model


class HierarchicalPredictor:
    """Ability + difficulty prior, optional item term, evidence posterior."""

    name = "hierarchical_rate"
    uses_pair_labels = True
    uses_benchmark_labels = True

    def __init__(self, two_way, difficulty_state, cell_difficulty, *, store=None, item_model=None,
                 prior_strength=5.0, benchmark_weight=0.5, use_difficulty=True):
        self.two_way = two_way
        self.difficulty_state = difficulty_state
        self.cell_difficulty = cell_difficulty
        self.store = store
        self.item_model = item_model
        self.prior_strength = float(prior_strength)
        self.benchmark_weight = float(benchmark_weight)
        self.use_difficulty = bool(use_difficulty)

    def pair_prior(self, pair):
        ability = self.two_way.ability.get(pair.subject_key, 0.0) if pair.subject_key else 0.0
        difficulty = self.cell_difficulty.get(pair.cell_key, self.difficulty_state["intercept"]) \
            if self.use_difficulty else 0.0
        return float(sigmoid(self.two_way.mu + ability + difficulty))

    def base(self, pair):
        prior = self.pair_prior(pair)
        if self.item_model is None or self.store is None:
            return np.full(len(pair.evaluation), prior, dtype=float)
        matrix = self.store.matrix(pair.benchmark, pair.evaluation)
        design = np.column_stack([np.full(len(matrix), float(logit(prior))), matrix])
        probabilities = np.asarray(self.item_model.predict_proba(design), dtype=float)[:, 1]
        return np.clip(probabilities, EPS, 1 - EPS)

    def update(self, base, own, other):
        base = np.asarray(base, dtype=float)
        if not own and not other:
            return base
        observed_mean = float(base.mean())
        successes = float(np.sum(own)) + self.benchmark_weight * float(np.sum(other))
        count = len(own) + self.benchmark_weight * len(other)
        posterior = (self.prior_strength * observed_mean + successes) / (self.prior_strength + count)
        shift = float(logit(posterior) - logit(observed_mean))
        return np.clip(sigmoid(logit(base) + shift), EPS, 1 - EPS)


class AbilityOnlyPredictor(HierarchicalPredictor):
    """Ability prior with no new-benchmark difficulty estimate."""

    name = "ability_only"

    def __init__(self, two_way, *, prior_strength=5.0, benchmark_weight=0.5):
        super().__init__(two_way, {"intercept": 0.0}, {}, prior_strength=prior_strength,
                         benchmark_weight=benchmark_weight, use_difficulty=False)


def assign_folds(cells, count, seed):
    if count < 2:
        raise ValueError("Benchmark-level cross-validation needs at least two folds")
    return {cell["benchmark"]: int(stable_hash({"folds": count, "seed": seed,
                                                "benchmark": cell["benchmark"]})[:8], 16) % count
            for cell in cells}


__all__ = ["AbilityOnlyPredictor", "HierarchicalPredictor", "ITEM_FEATURES", "ItemFeatureStore",
           "TwoWayFit", "ability_key", "assign_folds", "cell_feature_matrix", "difficulty_cv",
           "empirical_logit", "fit_difficulty", "fit_item_block", "fit_two_way", "item_block_rows",
           "item_feature_frame", "logit", "predict_difficulty", "sigmoid", "thin_label"]
