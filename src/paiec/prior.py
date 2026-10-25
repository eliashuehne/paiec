"""Logistic-regression prior: P(correct) for a subject and item before any labels.

Training rows come from training benchmarks only: up to ``max_pairs`` scoreable
pairs per benchmark and ``items_per_pair`` items per pair, each pair weighted
equally (rows within a pair share its weight). Rows are cached per benchmark
under ``<data_root>/../cache`` because building them needs the item text.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .features import DIM, FEATURE_VERSION, item_vector, subject_vector
from .folds import benchmark_folds
from .schema import item_record, load_benchmark, subject_record
from .split import stable_seed


def cache_dir(data_root) -> Path:
    return Path(data_root).resolve().parent / "cache" / f"prior_v{FEATURE_VERSION}"


def benchmark_rows(data_root, benchmark: str, max_pairs: int = 20, items_per_pair: int = 50,
                   seed: int = 0, min_items: int = 80) -> dict[str, np.ndarray]:
    """Sparse training rows (CSR parts), labels and row weights for one benchmark."""
    name = benchmark.split("/")[0]
    path = cache_dir(data_root) / f"{name}_p{max_pairs}_i{items_per_pair}_s{seed}.npz"
    if path.is_file():
        with np.load(path) as stored:
            return dict(stored)

    tables = load_benchmark(data_root, name)
    responses = tables.responses.drop_duplicates(["subject_id", "item_id"])
    responses = responses[responses["subject_id"].isin(tables.subjects.index)]
    sizes = responses.groupby("subject_id").size()
    qualifying = sorted(sizes[sizes >= min_items].index)
    rng = np.random.default_rng(stable_seed(seed, name, "prior"))
    chosen = rng.choice(qualifying, size=min(max_pairs, len(qualifying)), replace=False) if qualifying else []

    indptr, indices, values, labels, weights = [0], [], [], [], []
    for subject_id in sorted(chosen):
        subject_idx, subject_val = subject_vector(subject_record(tables.subjects.loc[subject_id]))
        group = responses[responses["subject_id"] == subject_id]
        group = group.sample(n=min(items_per_pair, len(group)), random_state=int(rng.integers(2**31)))
        for row in group.itertuples(index=False):
            stored = tables.items.loc[row.item_id]
            item = item_record({"content": stored["content"], "item_features": stored["item_features"],
                                "interactors": row.interactors}, name)
            item_idx, item_val = item_vector(item)
            indices.append(np.concatenate([item_idx, subject_idx]))
            values.append(np.concatenate([item_val, subject_val]))
            indptr.append(indptr[-1] + len(indices[-1]))
            labels.append(row.label)
            weights.append(1.0 / len(group))

    rows = {
        "indptr": np.asarray(indptr, dtype=np.int64),
        "indices": np.concatenate(indices) if indices else np.zeros(0, dtype=np.int64),
        "values": np.concatenate(values) if values else np.zeros(0),
        "labels": np.asarray(labels, dtype=np.float64),
        "weights": np.asarray(weights, dtype=np.float64),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **rows)
    return rows


def training_matrix(data_root, benchmarks):
    """Stack cached rows: (CSR matrix, labels, weights, benchmark of each row)."""
    from scipy.sparse import csr_matrix

    parts = [(b, benchmark_rows(data_root, b)) for b in benchmarks]
    parts = [(b, rows) for b, rows in parts if rows["labels"].size]
    matrices = [csr_matrix((rows["values"], rows["indices"], rows["indptr"]), shape=(rows["labels"].size, DIM))
                for _, rows in parts]
    from scipy.sparse import vstack

    matrix = vstack(matrices).tocsr()
    labels = np.concatenate([rows["labels"] for _, rows in parts])
    weights = np.concatenate([rows["weights"] for _, rows in parts])
    groups = np.concatenate([[b] * rows["labels"].size for b, rows in parts])
    return matrix, labels, weights, groups


def _fit_logistic(matrix, labels, weights, regularization: float):
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(C=regularization, max_iter=300)
    model.fit(matrix, labels, sample_weight=weights / weights.mean())
    return model.coef_.ravel().copy(), float(model.intercept_[0])


def _weighted_brier(matrix, labels, weights, coef, intercept) -> float:
    probabilities = 1.0 / (1.0 + np.exp(-(matrix @ coef + intercept)))
    return float(np.average((probabilities - labels) ** 2, weights=weights))


class LogisticPrior:
    """Budget-0 prediction from item and subject features; ignores revealed labels."""

    REGULARIZATION_GRID = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)

    def __init__(self):
        self.coef = np.zeros(DIM)
        self.intercept = 0.0
        self.regularization = 0.0
        self._item_cache: dict = {}
        self._subject_cache: dict = {}

    def fit(self, data_root, train_benchmarks):
        matrix, labels, weights, groups = training_matrix(data_root, train_benchmarks)
        inner_train, inner_val = benchmark_folds(sorted(set(groups)), k=5, seed=1)[0]
        fit_rows, val_rows = np.isin(groups, inner_train), np.isin(groups, inner_val)
        scores = []
        for regularization in self.REGULARIZATION_GRID:
            coef, intercept = _fit_logistic(matrix[fit_rows], labels[fit_rows], weights[fit_rows], regularization)
            scores.append(_weighted_brier(matrix[val_rows], labels[val_rows], weights[val_rows], coef, intercept))
        self.regularization = float(self.REGULARIZATION_GRID[int(np.argmin(scores))])
        self.coef, self.intercept = _fit_logistic(matrix, labels, weights, self.regularization)
        self._item_cache.clear()
        self._subject_cache.clear()
        return self

    def logit(self, subject: dict, item: dict) -> float:
        item_key = (item.get("item_content"), item.get("item_features"), item.get("interactors"))
        if item_key not in self._item_cache:
            indices, values = item_vector(item)
            self._item_cache[item_key] = float(self.coef[indices] @ values)
        subject_key = tuple(sorted(subject.items()))
        if subject_key not in self._subject_cache:
            indices, values = subject_vector(subject)
            self._subject_cache[subject_key] = float(self.coef[indices] @ values)
        return self.intercept + self._item_cache[item_key] + self._subject_cache[subject_key]

    def __call__(self, subject: dict, item: dict, labeled) -> float:
        return float(1.0 / (1.0 + np.exp(-self.logit(subject, item))))


__all__ = ["LogisticPrior", "benchmark_rows", "cache_dir", "training_matrix"]
