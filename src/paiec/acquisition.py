"""Choosing which labels to reveal, as the platform's streaming coordinator does.

Between two budget checkpoints the coordinator streams a pair's not-yet-revealed
acquisition candidates one at a time and asks ``acquisition_function(input,
prediction, labeled, context)`` whether to buy each one, until the checkpoint's
budget is reached (``baseline/tools/streaming_ingestion.py``). Without a
``labeling.py`` it buys with probability labels_remaining / items_remaining,
a uniform sample without replacement.

The coordinator's candidate order is not published; here every checkpoint
streams the remaining candidates in a fresh seeded order. Policies have the
``labeling.py`` signature, so a policy that wins here can be shipped as is.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence

import numpy as np

from .metrics import BUDGETS
from .split import Split, stable_seed

Policy = Callable[[list, float | None, list, dict], bool]


def random_policy(input: list, prediction: float | None = None,
                  labeled: list | None = None, context: dict | None = None) -> bool:
    """The platform's default: buy with probability labels_remaining / items_remaining."""
    key = json.dumps([context, input], sort_keys=True, separators=(",", ":"), default=str)
    uniform = stable_seed(key) / 2**64
    return uniform < min(1.0, context["labels_remaining"] / context["items_remaining"])


random_policy.needs_prediction = False


def uncertainty_policy(width: float) -> Policy:
    """Buy candidates predicted within ``width`` of 0.5, as the official BLE baseline does.

    Falls back to buying everything once the stream would otherwise run out.
    """
    def policy(input, prediction=None, labeled=None, context=None) -> bool:
        if context["labels_remaining"] <= 0:
            return False
        must_query = context["items_remaining"] <= context["labels_remaining"]
        return bool(must_query or abs(prediction - 0.5) <= width)

    policy.needs_prediction = True
    return policy


POLICIES: dict[str, Callable[[], Policy]] = {
    "random": lambda: random_policy,
    "uncertainty10": lambda: uncertainty_policy(0.10),
    "uncertainty15": lambda: uncertainty_policy(0.15),
}


def streamed_evidence(
    pair,
    split: Split,
    predict: Callable,
    policy: Policy,
    first_label: dict,
    *,
    seed: int = 0,
    budgets: Sequence[int] = BUDGETS,
) -> dict[int, list]:
    """Return the pair's own ``labeled`` list at each budget, bought by ``policy``.

    A policy that does not fill a checkpoint's budget before its stream runs
    out simply has fewer labels, as on the platform.
    """
    labeled, acquired, evidence = [], set(), {}
    needs_prediction = getattr(policy, "needs_prediction", True)
    for budget in sorted(budgets):
        candidates = [item_id for item_id in split.acquisition if item_id not in acquired]
        order = np.random.default_rng(
            stable_seed(seed, pair.benchmark, pair.subject_id, "stream", budget)
        ).permutation(len(candidates))
        for position, index in enumerate(order):
            labels_remaining = budget - len(acquired)
            if labels_remaining <= 0:
                break
            item_id = candidates[index]
            item = pair.item(item_id)
            context = {
                "labels_remaining": labels_remaining,
                "labels_acquired": len(acquired),
                "max_labels": max(budgets),
                "items_remaining": len(candidates) - position,
                "subject_id": pair.subject_id,
                "benchmark_id": pair.benchmark,
            }
            prediction = predict(pair.subject, item, labeled) if needs_prediction else None
            if policy([pair.subject, item], prediction, labeled, context):
                acquired.add(item_id)
                labeled = labeled + [[[pair.subject, item], first_label[item_id]]]
        evidence[budget] = labeled
    return evidence


__all__ = ["POLICIES", "random_policy", "streamed_evidence", "uncertainty_policy"]
