"""Logistic-regression prior: P(correct) for a subject and item before any labels.

Training rows come from training benchmarks only: up to ``max_pairs`` scoreable
pairs per benchmark and ``items_per_pair`` items per pair, each pair weighted
equally (rows within a pair share its weight). Rows are cached per benchmark
under ``<data_root>/../cache`` because building them needs the item text.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .features import DIM, FEATURE_VERSION, item_vector, subject_vector
from .folds import benchmark_folds
from .schema import item_record, load_benchmark, subject_record
from .split import stable_seed

if TYPE_CHECKING:
    from scipy.sparse import csr_matrix


@dataclass(frozen=True)
class _TrainingData:
    """Keep feature rows, labels, weights and benchmark names aligned."""

    matrix: csr_matrix
    labels: np.ndarray
    weights: np.ndarray
    benchmarks: np.ndarray

    def for_benchmarks(self, names: Sequence[str]) -> _TrainingData:
        selected = np.isin(self.benchmarks, names)
        return _TrainingData(
            self.matrix[selected],
            self.labels[selected],
            self.weights[selected],
            self.benchmarks[selected],
        )


def cache_dir(data_root) -> Path:
    return Path(data_root).resolve().parent / "cache" / f"prior_v{FEATURE_VERSION}"


def benchmark_rows(
    data_root,
    benchmark: str,
    max_pairs: int = 20,
    items_per_pair: int = 50,
    seed: int = 0,
    min_items: int = 80,
) -> dict[str, np.ndarray]:
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
    chosen = []
    if qualifying:
        chosen = rng.choice(qualifying, size=min(max_pairs, len(qualifying)), replace=False)

    # CSR stores the start of each sparse row, followed by its columns and values.
    indptr = [0]
    indices, values = [], []
    labels, weights = [], []
    for subject_id in sorted(chosen):
        subject = subject_record(tables.subjects.loc[subject_id])
        subject_indices, subject_values = subject_vector(subject)
        subject_responses = responses[responses["subject_id"] == subject_id]
        sampled = subject_responses.sample(
            n=min(items_per_pair, len(subject_responses)),
            random_state=int(rng.integers(2**31)),
        )
        for row in sampled.itertuples(index=False):
            stored_item = tables.items.loc[row.item_id]
            item = item_record(
                {
                    "content": stored_item["content"],
                    "item_features": stored_item["item_features"],
                    "interactors": row.interactors,
                },
                name,
            )
            item_indices, item_values = item_vector(item)
            indices.append(np.concatenate([item_indices, subject_indices]))
            values.append(np.concatenate([item_values, subject_values]))
            indptr.append(indptr[-1] + len(indices[-1]))
            labels.append(row.label)
            weights.append(1.0 / len(sampled))

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
    from scipy.sparse import csr_matrix, vstack

    parts = []
    matrices = []
    for benchmark in benchmarks:
        rows = benchmark_rows(data_root, benchmark)
        if rows["labels"].size == 0:
            continue
        parts.append((benchmark, rows))
        matrix = csr_matrix(
            (rows["values"], rows["indices"], rows["indptr"]),
            shape=(rows["labels"].size, DIM),
        )
        matrices.append(matrix)

    matrix = vstack(matrices).tocsr()
    labels = np.concatenate([rows["labels"] for _, rows in parts])
    weights = np.concatenate([rows["weights"] for _, rows in parts])
    groups = np.concatenate([
        [benchmark] * rows["labels"].size for benchmark, rows in parts
    ])
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

    def fit(self, data_root: str | Path, train_benchmarks: Sequence[str]) -> LogisticPrior:
        data = _TrainingData(*training_matrix(data_root, train_benchmarks))
        self.regularization = self._choose_regularization(data)
        self.coef, self.intercept = _fit_logistic(
            data.matrix, data.labels, data.weights, self.regularization
        )
        self._item_cache.clear()
        self._subject_cache.clear()
        return self

    def _choose_regularization(self, data: _TrainingData) -> float:
        # Tune on a held-out subset of the outer fold's training benchmarks.
        train_names, validation_names = benchmark_folds(
            sorted(set(data.benchmarks)), k=5, seed=1
        )[0]
        train = data.for_benchmarks(train_names)
        validation = data.for_benchmarks(validation_names)
        scores = []
        for regularization in self.REGULARIZATION_GRID:
            coef, intercept = _fit_logistic(
                train.matrix, train.labels, train.weights, regularization
            )
            score = _weighted_brier(
                validation.matrix, validation.labels, validation.weights, coef, intercept
            )
            scores.append(score)
        return float(self.REGULARIZATION_GRID[int(np.argmin(scores))])

    def logit(self, subject: dict, item: dict) -> float:
        item_key = (
            item.get("item_content"), item.get("item_features"), item.get("interactors")
        )
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
