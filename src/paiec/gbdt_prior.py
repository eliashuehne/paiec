"""Gradient-boosted trees as the per-item prior, so subject x item interactions can be learned.

Same feature rows as ``LogisticPrior`` (hashed item and subject vectors, see
``features.py``); trees can combine them, e.g. model size with math content.
The number of trees is chosen by early stopping on held-out *training
benchmarks*, so the model cannot win by memorising benchmarks.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

from .features import DIM, item_vector, subject_vector
from .folds import benchmark_folds
from .prior import training_matrix

PARAMS = {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.5,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "max_bin": 63,
    "verbose": -1,
    "seed": 0,
    "deterministic": True,
    "num_threads": 4,
}
MAX_TREES = 2000
EARLY_STOPPING = 50


def feature_row(subject_vec, item_vec) -> csr_matrix:
    """One CSR row from subject and item vectors; collisions summed like in training."""
    indices = np.concatenate([item_vec[0], subject_vec[0]])
    values = np.concatenate([item_vec[1], subject_vec[1]])
    row = csr_matrix((values, indices, [0, len(indices)]), shape=(1, DIM))
    row.sum_duplicates()
    return row


class GBDTPrior:
    """LightGBM on the prior's feature rows; ``logit`` returns the raw boosted score."""

    def __init__(self):
        self.booster = None
        self.trees = 0
        self._cache: dict = {}

    def fit(self, data_root: str | Path, train_benchmarks: Sequence[str]) -> GBDTPrior:
        import lightgbm as lgb

        matrix, labels, weights, groups = training_matrix(data_root, train_benchmarks)
        matrix.sum_duplicates()
        weights = weights / weights.mean()
        fit_names, val_names = benchmark_folds(sorted(set(groups)), k=5, seed=1)[0]
        fit_rows, val_rows = np.isin(groups, fit_names), np.isin(groups, val_names)
        train_set = lgb.Dataset(matrix[fit_rows], labels[fit_rows], weight=weights[fit_rows], free_raw_data=False)
        val_set = lgb.Dataset(matrix[val_rows], labels[val_rows], weight=weights[val_rows], reference=train_set)
        probe = lgb.train(PARAMS, train_set, num_boost_round=MAX_TREES, valid_sets=[val_set],
                          callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False)])
        self.trees = max(int(probe.best_iteration), 1)
        full_set = lgb.Dataset(matrix, labels, weight=weights)
        self.booster = lgb.train(PARAMS, full_set, num_boost_round=self.trees)
        self._cache.clear()
        return self

    def row_logits(self, matrix) -> np.ndarray:
        matrix = csr_matrix(matrix)
        matrix.sum_duplicates()
        return np.asarray(self.booster.predict(matrix, raw_score=True), dtype=float)

    def logit(self, subject: dict, item: dict) -> float:
        key = (tuple(sorted(subject.items())),
               item.get("item_content"), item.get("item_features"), item.get("interactors"))
        if key not in self._cache:
            row = feature_row(subject_vector(subject), item_vector(item))
            self._cache[key] = float(self.booster.predict(row, raw_score=True)[0])
        return self._cache[key]

    def __call__(self, subject: dict, item: dict, labeled) -> float:
        return float(1.0 / (1.0 + np.exp(-self.logit(subject, item))))


__all__ = ["GBDTPrior", "feature_row"]
