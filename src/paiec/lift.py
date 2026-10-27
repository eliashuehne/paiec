"""Score feature columns on held-out benchmarks with one fixed model.

The lift sample already carries a fold per benchmark. ``run_lift`` trains
LightGBM on four folds and scores the fifth, for each fold in turn; sealed
benchmarks are never read. Only the feature columns change between runs: the
sample, the folds and the model settings stay fixed, so two runs differ only by
what the features add.

Features learned from labels (a system's ability) are rebuilt inside each fold
from training benchmarks only; see ``in_fold_ability``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .lift_sample import SEALED
from .metrics import brier
from .subject_dates import fill_release_dates, subject_folds

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
# Pseudo-responses at zero ability added to every in-fold ability.
ABILITY_STRENGTH = 20


@dataclass(frozen=True)
class LiftResult:
    """Scores per held-out fold and benchmark, plus pairs in budget mode."""

    columns: tuple[str, ...]
    abilities: tuple[str, ...]
    folds: pd.DataFrame
    benchmarks: pd.DataFrame
    pairs: pd.DataFrame | None = None


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


@dataclass(frozen=True)
class Baseline:
    """A named reference model: plain columns plus in-fold abilities."""

    columns: tuple[str, ...] = ()
    abilities: tuple[str, ...] = ()


# A: training correct rate only. B: A plus the system's ability on other
# benchmarks, keyed by the visible ``normalized_name`` (``subject_id`` is not a
# predictor field). B needs the subject fields joined onto the sample.
# Abilities enter the model alone and may only raise P(correct); see run_lift.
BASELINES = {
    "A": Baseline(),
    "B": Baseline(abilities=("normalized_name",)),
}


def group_units(groups: Iterable[Sequence[str]]) -> dict[str, str]:
    """Map each grouped benchmark to one shared unit name; ungrouped ones map to themselves."""
    units = {}
    for group in groups:
        unit = "+".join(sorted(group))
        for name in group:
            if name in units:
                raise ValueError(f"{name} is in more than one group")
            units[name] = unit
    return units


def in_fold_ability(
    train: pd.DataFrame,
    rows: pd.DataFrame,
    key: str,
    strength: float = ABILITY_STRENGTH,
    units: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """How much better than average ``key`` does, for ``rows``, learned from ``train`` only.

    Each training response counts as its label minus its benchmark's training
    correct rate, so a strong system tested only on hard benchmarks is not
    mistaken for a weak one; a raw correct rate is confounded that way and
    scores worse than no subject feature at all. The mean is smoothed toward
    zero with ``strength`` pseudo-responses.

    Each row's ability leaves out its own unit: its benchmark, or the whole
    group from ``units`` (see ``group_units``) when its benchmark shares items
    with others. Folds hold groups out together, so a test row's unit is never
    in ``train`` and nothing is left out; a training row then sees exactly what
    a test row sees, and the model cannot learn to trust a value built from its
    own label or a sibling's. Rows whose key is missing or unseen get zero with
    zero count.

    Returns ``<key>_ability`` and ``<key>_n`` (responses behind it, for
    diagnostics), indexed like ``rows``.
    """
    if not rows.index.is_unique:
        raise ValueError("rows must have a unique index")
    units = units or {}

    def unit(frame):
        return frame["benchmark"].map(lambda name: units.get(name, name))

    benchmark_rate = train.groupby("benchmark")["label"].mean()
    known = train.dropna(subset=[key])
    known = known.assign(
        excess=known["label"] - known["benchmark"].map(benchmark_rate),
        unit=unit(known),
    )
    total = known.groupby(key)["excess"].agg(["sum", "count"])
    own = known.groupby([key, "unit"])["excess"].agg(["sum", "count"])
    own_index = pd.MultiIndex.from_arrays([rows[key], unit(rows)])
    excess, count = (
        total.reindex(rows[key]).fillna(0).to_numpy()
        - own.reindex(own_index).fillna(0).to_numpy()
    ).T
    denominator = count + strength
    ability = np.divide(excess, denominator, out=np.zeros(len(rows)), where=denominator > 0)
    return pd.DataFrame({f"{key}_ability": ability, f"{key}_n": count}, index=rows.index)


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


def fit_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: Sequence[str],
    increasing: Sequence[str] = (),
    offset: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Train on ``train`` and return P(correct) for each ``test`` row.

    Columns in ``increasing`` may only raise the prediction as they grow.
    ``offset`` gives starting log-odds for the train and test rows; the trees
    then learn corrections to it. With no columns the prediction is the offset,
    or else the training correct rate, the floor every feature set is compared
    against.
    """
    if not columns:
        if offset is not None:
            return 1 / (1 + np.exp(-offset[1]))
        return np.full(len(test), train["label"].mean())
    x_train, x_test = _model_frame(train, test, columns)
    params = PARAMS
    if increasing:
        params = {**PARAMS, "monotone_constraints": [int(c in increasing) for c in columns]}
    init = None if offset is None else offset[0]
    model = lgb.train(params, lgb.Dataset(x_train, label=train["label"], init_score=init), num_boost_round=ROUNDS)
    if offset is None:
        return model.predict(x_test)
    return 1 / (1 + np.exp(-(model.predict(x_test, raw_score=True) + offset[1])))


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


def run_lift(
    sample: pd.DataFrame,
    columns: Sequence[str] = (),
    abilities: Sequence[str] = (),
    groups: Iterable[Sequence[str]] = (),
    prepare: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    prepared_columns: Sequence[str] = (),
    increasing: Sequence[str] = (),
    predict: Callable[..., np.ndarray] | None = None,
    pair_weighted: bool = False,
) -> LiftResult:
    """Score ``columns`` on every numbered fold; sealed benchmarks are skipped.

    Each key in ``abilities`` adds a ``<key>_ability`` column, rebuilt in every
    fold from that fold's training benchmarks, leaving out each row's benchmark
    group (``groups``, as in ``configs/lift/benchmark_groups.json``). The model
    sees the ability alone and may only raise P(correct) with it: an exact value
    or count differs per training benchmark, and trees would use it to memorise
    that benchmark's difficulty, which never transfers.

    ``prepare`` turns each fold's train and test rows into the rows to fit and
    score, after abilities are built from the full training rows; budget-k mode
    (``lift_budget``) uses it to reveal labels and add the ``prepared_columns``.
    Columns in ``increasing`` may only raise P(correct). ``predict`` replaces
    ``fit_predict`` (same arguments), for rule or offset baselines.

    Fold rows report row-weighted Brier and log loss plus ``brier_macro``, the
    mean of per-benchmark Brier, so small benchmarks count as much as large ones.
    With ``pair_weighted=True`` these scores instead average pair scores and
    retain the pair table for the global equal-pair mean. Derived release-date
    columns are always rebuilt without the outer validation fold.
    """
    missing = sorted((set(columns) | set(abilities)) - set(sample.columns) - set(prepared_columns))
    if missing:
        raise ValueError(f"columns not in sample: {missing}")
    if not sample.index.is_unique:
        raise ValueError("sample must have a unique index")
    units = group_units(groups)
    ability_columns = [f"{key}_ability" for key in abilities]
    model_columns = [*columns, *ability_columns]
    monotone = [*ability_columns, *increasing]
    predict = predict or fit_predict
    folds, benchmarks, pairs = [], [], []
    derived_dates = {"release_days_filled", "release_year_filled", "release_lo", "release_hi",
                     "release_source", "release_filled"} & set(columns)
    date_inputs = ["subject_id", "identity_key", "family", "line", "version", "log_params_b",
                   "provider_filled", "release_days", "name_date_days"]
    if derived_dates:
        missing_dates = set(date_inputs) - set(sample.columns)
        if missing_dates:
            raise ValueError(f"date features require raw subject columns: {sorted(missing_dates)}")
        date_subjects = sample[date_inputs].drop_duplicates()
        if date_subjects["subject_id"].duplicated().any():
            raise ValueError("a subject has inconsistent raw date features")
        memberships = subject_folds(sample)
    for fold, train, test in fold_splits(sample):
        if derived_dates:
            dates = fill_release_dates(date_subjects, memberships, accepted=None, excluded_folds={fold})
            dates = dates.set_index("subject_id")
            train, test = (
                rows.assign(**{c: rows["subject_id"].map(dates[c]) for c in derived_dates})
                for rows in (train, test)
            )
        for key, column in zip(abilities, ability_columns):
            train, test = (
                train.assign(**{column: in_fold_ability(train, train, key, units=units)[column]}),
                test.assign(**{column: in_fold_ability(train, test, key, units=units)[column]}),
            )
        if prepare is not None:
            train, test = prepare(train), prepare(test)
        scored = test.assign(p=predict(train, test, model_columns, increasing=monotone))
        per_pair = pd.DataFrame([
            {"fold": fold, "benchmark": benchmark, "subject_id": subject, **_scores(group)}
            for (benchmark, subject), group in scored.groupby(["benchmark", "subject_id"], sort=True)
        ]) if pair_weighted else None
        if per_pair is not None:
            pairs.append(per_pair)
            per_benchmark = [
                {"fold": fold, "benchmark": name, "rows": int(group["rows"].sum()),
                 "brier": float(group["brier"].mean()), "log_loss": float(group["log_loss"].mean())}
                for name, group in per_pair.groupby("benchmark", sort=True)
            ]
            scores = {"rows": len(scored), "brier": float(per_pair["brier"].mean()),
                      "log_loss": float(per_pair["log_loss"].mean())}
        else:
            per_benchmark = [
                {"fold": fold, "benchmark": name, **_scores(group)}
                for name, group in scored.groupby("benchmark", sort=True)
            ]
            scores = _scores(scored)
        benchmarks.extend(per_benchmark)
        folds.append({
            "fold": fold,
            "benchmarks": len(per_benchmark),
            **scores,
            "brier_macro": float(np.mean([row["brier"] for row in per_benchmark])),
        })
    return LiftResult(
        tuple(columns), tuple(abilities), pd.DataFrame(folds), pd.DataFrame(benchmarks),
        pd.concat(pairs, ignore_index=True) if pair_weighted else None,
    )


__all__ = [
    "ABILITY_STRENGTH",
    "BASELINES",
    "KEYS",
    "PARAMS",
    "ROUNDS",
    "Baseline",
    "LiftResult",
    "fit_predict",
    "fold_splits",
    "group_units",
    "in_fold_ability",
    "join_features",
    "load_features",
    "log_loss",
    "run_lift",
]
