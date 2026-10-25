"""Score feature columns on held-out benchmarks with one fixed model.

The lift sample already carries a fold per benchmark. ``run_lift`` trains
LightGBM on four folds and scores the fifth, for each fold in turn; sealed
benchmarks are never read. Only the feature columns change between runs: the
sample, the folds and the model settings stay fixed, so two runs differ only by
what the features add.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .lift_sample import SEALED
from .metrics import brier

KEYS = ("benchmark", "subject_id", "item_id")
EPSILON = 1e-15

# Fixed on purpose: changing any of these makes earlier runs incomparable.
PARAMS = {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 200,
    "lambda_l2": 1.0,
    "seed": 0,
    "deterministic": True,
    "force_row_wise": True,
    "num_threads": 4,
    "verbose": -1,
}
ROUNDS = 300


@dataclass(frozen=True)
class LiftResult:
    """Scores per held-out fold and per benchmark."""

    columns: tuple[str, ...]
    folds: pd.DataFrame
    benchmarks: pd.DataFrame


def join_features(sample: pd.DataFrame, tables: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Left-join feature tables onto the sample by whichever keys each one has.

    A table keyed by ``item_id`` alone adds item features; one keyed by
    ``subject_id`` adds subject features. Keys must be unique in each table and
    feature names must not clash, so a join never duplicates or overwrites rows.
    """
    joined = sample
    for table in tables:
        keys = [key for key in KEYS if key in table.columns]
        if not keys:
            raise ValueError(f"feature table has none of the keys {KEYS}")
        if table.duplicated(keys).any():
            raise ValueError(f"feature table has duplicate keys on {keys}")
        clash = sorted((set(table.columns) - set(keys)) & set(joined.columns))
        if clash:
            raise ValueError(f"feature columns already present: {clash}")
        joined = joined.merge(table, on=keys, how="left", validate="many_to_one")
    return joined


def load_features(sample: pd.DataFrame, paths: Sequence[Path]) -> pd.DataFrame:
    """Read feature tables from parquet and join them onto the sample."""
    return join_features(sample, (pd.read_parquet(path) for path in paths))


def fold_splits(sample: pd.DataFrame) -> Iterator[tuple[str, pd.DataFrame, pd.DataFrame]]:
    """Yield ``(fold, train, test)`` for each numbered fold, never touching sealed rows."""
    open_rows = sample[sample["fold"] != SEALED]
    for fold in sorted(open_rows["fold"].unique(), key=int):
        in_fold = open_rows["fold"] == fold
        train, test = open_rows[~in_fold], open_rows[in_fold]
        if set(train["benchmark"]) & set(test["benchmark"]):
            raise ValueError(f"fold {fold}: a benchmark is in both train and test")
        yield fold, train, test


def _model_frame(train: pd.DataFrame, test: pd.DataFrame, columns: Sequence[str]):
    """Columns as LightGBM inputs; text becomes categories learned from train."""
    x_train, x_test = train[list(columns)].copy(), test[list(columns)].copy()
    for column in columns:
        if x_train[column].dtype.kind in "biufc":
            continue
        categories = pd.Index(x_train[column].dropna().unique()).sort_values()
        x_train[column] = pd.Categorical(x_train[column], categories=categories)
        # Values never seen in training become missing, as they would at evaluation.
        seen = x_test[column].where(x_test[column].isin(categories))
        x_test[column] = pd.Categorical(seen, categories=categories)
    return x_train, x_test


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    """Train on ``train`` and return P(correct) for each ``test`` row.

    With no columns the prediction is the training correct rate, which is the
    floor every feature set is compared against.
    """
    if not columns:
        return np.full(len(test), train["label"].mean())
    x_train, x_test = _model_frame(train, test, columns)
    model = lgb.train(PARAMS, lgb.Dataset(x_train, label=train["label"]), num_boost_round=ROUNDS)
    return model.predict(x_test)


def log_loss(labels: Sequence[float], probabilities: Sequence[float]) -> float:
    """Mean negative log-likelihood, with probabilities clipped away from 0 and 1."""
    y = np.asarray(labels, dtype=float)
    p = np.clip(np.asarray(probabilities, dtype=float), EPSILON, 1 - EPSILON)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _scores(frame: pd.DataFrame) -> dict:
    return {
        "rows": len(frame),
        "brier": brier(frame["label"], frame["p"]),
        "log_loss": log_loss(frame["label"], frame["p"]),
    }


def run_lift(sample: pd.DataFrame, columns: Sequence[str] = ()) -> LiftResult:
    """Score ``columns`` on every numbered fold; sealed benchmarks are skipped.

    Fold rows report row-weighted Brier and log loss plus ``brier_macro``, the
    mean of per-benchmark Brier, so small benchmarks count as much as large ones.
    """
    missing = sorted(set(columns) - set(sample.columns))
    if missing:
        raise ValueError(f"columns not in sample: {missing}")
    folds, benchmarks = [], []
    for fold, train, test in fold_splits(sample):
        scored = test.assign(p=fit_predict(train, test, columns))
        per_benchmark = [
            {"fold": fold, "benchmark": name, **_scores(group)}
            for name, group in scored.groupby("benchmark", sort=True)
        ]
        benchmarks.extend(per_benchmark)
        folds.append({
            "fold": fold,
            "benchmarks": len(per_benchmark),
            **_scores(scored),
            "brier_macro": float(np.mean([row["brier"] for row in per_benchmark])),
        })
    return LiftResult(tuple(columns), pd.DataFrame(folds), pd.DataFrame(benchmarks))


__all__ = [
    "KEYS",
    "PARAMS",
    "ROUNDS",
    "LiftResult",
    "fit_predict",
    "fold_splits",
    "join_features",
    "load_features",
    "log_loss",
    "run_lift",
]
