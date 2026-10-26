"""Shared full-protocol scoring over training-only acquisition episodes.

One implementation of the nested-budget scoring used by the dataset-only
controls, so a new base predictor is compared on exactly the same episodes,
query rows and fixed controls. Query outcomes are scoring targets only; the
support prefix is the only revealed evidence.
"""
import numpy as np

from .competition_analysis import numeric_evidence_predictions
from .conditional_predictor import evidence_logit_offset, logistic, logit
from .feature_preparation import read_jsonl
from .metrics import BUDGETS, brier_alc

MIN_TASK_GROUPS = 80
MIN_SUPPORTS = max(BUDGETS)


def full_protocol_episodes(prepared, rows, scopes, families):
    """Validated episodes covering every label budget, plus excluded scopes.

    Each returned episode carries ``scope_key``, ``validation_family``,
    ``support`` and ``query`` row indices. Structural failures raise; a scope
    that is merely too small is excluded with its reason.
    """
    row_index = {row["row_key"]: i for i, row in enumerate(rows)}
    if len(row_index) != len(rows):
        raise ValueError("Duplicate training row key")
    episodes, excluded = [], []
    for episode in read_jsonl(prepared / "evidence_plan_v1/evidence_episode_links.jsonl"):
        retained = len(episode["acquisition_task_groups"]) + len(episode["query_task_groups"])
        if retained < MIN_TASK_GROUPS or len(episode["support_order_row_keys"]) < MIN_SUPPORTS:
            excluded.append({"scope_key": episode["scope_key"], "retained_task_groups": retained,
                             "available_supports": len(episode["support_order_row_keys"]),
                             "reason": "insufficient_retained_items_for_full_protocol"})
            continue
        if any(key not in row_index for key in episode["support_order_row_keys"] + episode["query_row_keys"]):
            raise ValueError("Episode pointer is not a training row")
        support = [row_index[key] for key in episode["support_order_row_keys"]]
        query = [row_index[key] for key in episode["query_row_keys"]]
        if (set(support) & set(query) or len({families[i] for i in support + query}) != 1 or
                any(scopes[i] != episode["scope_key"] for i in support + query)):
            raise ValueError("Episode violates held-family/scope isolation")
        episodes.append({"scope_key": episode["scope_key"],
                         "validation_family": str(families[query[0]]),
                         "support": support, "query": query})
    if not episodes:
        raise ValueError("No training scope covers the complete protocol")
    return episodes, excluded


def curve_scores(episodes, labels, bases, *, prior_strength=5., offset_regularization=1.):
    """Score fixed controls and named base predictors at every budget.

    ``bases`` maps a candidate name to a full-length out-of-family prediction
    array. Fixed controls (constant, empirical mean, beta-mean with the declared
    prior strength) are computed once per episode from the revealed prefix.
    Returns ``(scored_rows, curves, alc)`` keyed like the existing controls.
    """
    labels = np.asarray(labels, dtype=float)
    for name, values in bases.items():
        values = np.asarray(values, dtype=float)
        if values.shape != labels.shape or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError("Base predictions must be finite probabilities for every training row: " + name)
    if not episodes:
        raise ValueError("Scoring needs at least one episode")
    scored = []
    for episode in episodes:
        support, query = episode["support"], episode["query"]
        for budget in BUDGETS:
            revealed = support[:budget]
            observed = [int(labels[i]) for i in revealed]
            methods = {name: np.full(len(query), value)
                       for name, value in numeric_evidence_predictions(observed).items()}
            for name, values in bases.items():
                base = np.asarray(values, dtype=float)[query]
                methods[name + "/none"] = base
                methods[name + "/beta_mean"] = (prior_strength * base + sum(observed)) / (prior_strength + budget)
                if observed:
                    # One scalar offset per scope; relative item difficulty is preserved.
                    shift = evidence_logit_offset(np.asarray(values, dtype=float)[revealed].tolist(),
                                                  observed, offset_regularization)
                    methods[name + "/logit_shift"] = np.asarray(
                        [logistic(logit(float(p)) + shift) for p in base])
                else:
                    methods[name + "/logit_shift"] = base
            for name, values in methods.items():
                scored.append({"scope_key": episode["scope_key"],
                               "validation_family": episode["validation_family"],
                               "method": name, "budget": budget,
                               "brier": float(np.mean((values - labels[query]) ** 2)),
                               "query_rows": len(query)})
    curves, alc = {}, {}
    for method in sorted({row["method"] for row in scored}):
        curves[method] = {budget: float(np.mean([row["brier"] for row in scored
                                                 if row["method"] == method and row["budget"] == budget]))
                          for budget in BUDGETS}
        alc[method] = brier_alc(curves[method])
    return scored, curves, alc
