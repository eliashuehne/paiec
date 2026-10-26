"""Budget-k mode: score held-out rows after revealing k labels per system.

The competition scores each (system, benchmark) pair at budgets 0, 1, 3, 7, 15
and 31. A pair's items are split 50/50 with ``split.make_split``: the first k
items of the acquisition pool are revealed, and the evaluation pool is scored
at every budget, so per-budget scores compare like for like. The revealed
labels include other systems on the same benchmark, so from budget 1 the
predictor can estimate the benchmark's level, which budget 0 cannot.

Training benchmarks are prepared the same way, so the model learns how
revealed labels relate to held-out ones. Sealed benchmarks are never read.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial

import numpy as np
import pandas as pd

from .lift import LiftResult, fit_predict, run_lift
from .metrics import BUDGETS, alc_weights
from .split import make_split

PAIR_KEYS = ["benchmark", "subject_id"]
ITEM_KEYS = ["benchmark", "subject_id", "item_id"]
REVEALED = ("bench_revealed_rate", "bench_revealed_n", "pair_revealed_rate", "pair_revealed_n")
# Pseudo-labels for the K baseline: the benchmark rate is pulled toward the
# training rate, and the pair rate toward the benchmark rate. Fixed, not tuned.
BENCH_STRENGTH = 20
PAIR_STRENGTH = 5


def reveal_order(sample: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Each pair's items with their pool and acquisition position, as ``make_split`` draws them.

    A pair with a single item has nothing to reveal; its item is evaluated.
    """
    columns = {key: [] for key in (*ITEM_KEYS, "pool", "order")}
    items = sample[ITEM_KEYS].drop_duplicates()
    for (benchmark, subject_id), group in items.groupby(PAIR_KEYS, sort=True):
        ids = group["item_id"].tolist()
        if len(ids) < 2:
            acquisition, evaluation = (), tuple(ids)
        else:
            split = make_split(ids, benchmark=benchmark, subject_id=subject_id, seed=seed)
            acquisition, evaluation = split.acquisition, split.evaluation
        pools = [(item, "reveal", order) for order, item in enumerate(acquisition)]
        pools += [(item, "evaluate", -1) for item in evaluation]
        for item, pool, order in pools:
            columns["benchmark"].append(benchmark)
            columns["subject_id"].append(subject_id)
            columns["item_id"].append(item)
            columns["pool"].append(pool)
            columns["order"].append(order)
    return pd.DataFrame(columns)


def attach_reveal_order(sample: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Add ``pool`` and ``order`` to every row; the index is kept."""
    order = reveal_order(sample[sample["fold"] != "sealed"], seed)
    keyed = sample.merge(order, on=ITEM_KEYS, how="left", validate="many_to_one")
    keyed.index = sample.index
    return keyed


def budget_frame(frame: pd.DataFrame, budget: int) -> pd.DataFrame:
    """Evaluation-pool rows with the labels revealed at ``budget`` summarised.

    A revealed item counts once, with its first trial, as the evaluator reveals
    one response per item. An item in the acquisition pool is never scored,
    even at budgets that do not reveal it, so the scored rows are the same at
    every budget.
    """
    if budget < 0:
        raise ValueError("budget must be at least 0")
    revealed = frame[(frame["pool"] == "reveal") & (frame["order"] < budget)]
    revealed = revealed.sort_values("trial", kind="stable").drop_duplicates(ITEM_KEYS)
    bench = revealed.groupby("benchmark")["label"].agg(["sum", "count"])
    pair = revealed.groupby(PAIR_KEYS)["label"].agg(["sum", "count"])

    scored = frame[frame["pool"] == "evaluate"].drop(columns=["pool", "order"])
    bench_hits, bench_n = bench.reindex(scored["benchmark"]).fillna(0).to_numpy().T
    pair_index = pd.MultiIndex.from_frame(scored[PAIR_KEYS])
    pair_hits, pair_n = pair.reindex(pair_index).fillna(0).to_numpy().T
    with np.errstate(invalid="ignore", divide="ignore"):
        return scored.assign(
            bench_revealed_rate=np.where(bench_n > 0, bench_hits / bench_n, np.nan),
            bench_revealed_n=bench_n,
            pair_revealed_rate=np.where(pair_n > 0, pair_hits / pair_n, np.nan),
            pair_revealed_n=pair_n,
        )


def empirical_mean(_train: pd.DataFrame, test: pd.DataFrame, _columns: Sequence[str] = (), **_) -> np.ndarray:
    """The organizers' baseline: the pair's revealed rate, or 0.5 with none."""
    return np.where(test["pair_revealed_n"] > 0, test["pair_revealed_rate"], 0.5)


def shrunk_rate(train: pd.DataFrame, test: pd.DataFrame, _columns: Sequence[str] = (), **_) -> np.ndarray:
    """The pair's revealed rate pulled toward the benchmark's, itself pulled toward the training rate.

    At budget 0 this is the training rate, so K equals A there.
    """
    prior = float(train["label"].mean())
    bench_n = test["bench_revealed_n"].to_numpy()
    bench_hits = np.nan_to_num(test["bench_revealed_rate"].to_numpy()) * bench_n
    bench = (bench_hits + BENCH_STRENGTH * prior) / (bench_n + BENCH_STRENGTH)
    pair_n = test["pair_revealed_n"].to_numpy()
    pair_hits = np.nan_to_num(test["pair_revealed_rate"].to_numpy()) * pair_n
    return (pair_hits + PAIR_STRENGTH * bench) / (pair_n + PAIR_STRENGTH)


def shrunk_offset(
    train: pd.DataFrame, test: pd.DataFrame, columns: Sequence[str] = (), increasing: Sequence[str] = ()
) -> np.ndarray:
    """LightGBM that starts from the shrunk rate's log-odds and learns corrections from ``columns``.

    The revealed rates are not tree inputs: each is one value per training
    benchmark, and trees on them memorise training benchmarks (on real data
    they matched the shrunk rate with more fold-to-fold noise). With no columns
    this is the shrunk rate exactly; features can only adjust it, which is the
    within-benchmark effect subject features carry.
    """
    def logit(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    offset = (logit(shrunk_rate(train, train)), logit(shrunk_rate(train, test)))
    return fit_predict(train, test, columns, increasing=increasing, offset=offset)


PREDICTORS = {"empirical_mean": empirical_mean, "shrunk_rate": shrunk_rate, "shrunk_offset": shrunk_offset}
# Fixed rules ignore feature columns; features are added to KM instead.
RULES = ("empirical_mean", "shrunk_rate")


@dataclass(frozen=True)
class BudgetBaseline:
    """A reference model for budget-k mode; ``predictor`` names one in PREDICTORS, or None for LightGBM."""

    columns: tuple[str, ...] = ()
    abilities: tuple[str, ...] = ()
    predictor: str | None = None


# A and B as in lift.BASELINES; E is the organizers' empirical mean; K is the
# model-free shrunk rate; KM starts from K and learns corrections from feature
# columns. KM with no columns is K; add subject and item features to KM.
BUDGET_BASELINES = {
    "A": BudgetBaseline(),
    "B": BudgetBaseline(abilities=("normalized_name",)),
    "E": BudgetBaseline(predictor="empirical_mean"),
    "K": BudgetBaseline(predictor="shrunk_rate"),
    "KM": BudgetBaseline(predictor="shrunk_offset"),
}


def run_budgets(
    sample: pd.DataFrame,
    budgets: Sequence[int] = BUDGETS,
    columns: Sequence[str] = (),
    abilities: Sequence[str] = (),
    predictor: str | None = None,
    groups: Iterable[Sequence[str]] = (),
    seed: int = 0,
) -> dict[int, LiftResult]:
    """``run_lift`` at each budget on the same scored rows."""
    keyed = attach_reveal_order(sample, seed)
    predict = PREDICTORS[predictor] if predictor else None
    groups = [list(group) for group in groups]
    return {
        budget: run_lift(
            keyed,
            columns,
            abilities,
            groups,
            prepare=partial(budget_frame, budget=budget),
            prepared_columns=REVEALED,
            predict=predict,
        )
        for budget in budgets
    }


def combine_budgets(results: Mapping[int, LiftResult]) -> LiftResult:
    """Weight per-budget results with the official ALC weights into one result.

    Every metric here is a mean over the same scored rows, so the weighted
    result is the ALC of each fold and benchmark. With one budget it is that
    budget's result.
    """
    budgets = sorted(results)
    first = results[budgets[0]]
    if len(budgets) == 1:
        return first
    weights = alc_weights(budgets)
    for budget in budgets[1:]:
        other = results[budget]
        same = first.folds[["fold", "rows"]].equals(other.folds[["fold", "rows"]]) and first.benchmarks[
            ["fold", "benchmark", "rows"]
        ].equals(other.benchmarks[["fold", "benchmark", "rows"]])
        if not same:
            raise ValueError(f"budget {budget} scored different rows")

    def weighted(table: str, metrics: Sequence[str]) -> pd.DataFrame:
        combined = getattr(first, table).copy()
        for metric in metrics:
            combined[metric] = sum(w * getattr(results[b], table)[metric] for w, b in zip(weights, budgets))
        return combined

    return LiftResult(
        first.columns,
        first.abilities,
        weighted("folds", ["brier", "log_loss", "brier_macro"]),
        weighted("benchmarks", ["brier", "log_loss"]),
    )


__all__ = [
    "BENCH_STRENGTH",
    "BUDGET_BASELINES",
    "PAIR_STRENGTH",
    "PREDICTORS",
    "REVEALED",
    "RULES",
    "BudgetBaseline",
    "attach_reveal_order",
    "budget_frame",
    "combine_budgets",
    "empirical_mean",
    "reveal_order",
    "run_budgets",
    "shrunk_offset",
    "shrunk_rate",
]
