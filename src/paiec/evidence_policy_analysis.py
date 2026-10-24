"""Nested training-family selection of evidence smoothing; no hidden-role labels."""
from collections import defaultdict
from pathlib import Path
import json

import numpy as np

from .conditional_analysis import training_panel
from .evidence_policy import posterior_mean
from .feature_preparation import read_jsonl, write_json, write_jsonl, stable_hash
from .head_training import file_hash
from .metrics import BUDGETS, brier_alc

STRENGTHS = (0., .5, 1., 2., 5., 10., 20.)


def select_strength(scores, *, excluded_family=None, weighting="equal_scope"):
    eligible = [r for r in scores if r["family"] != excluded_family]
    if not eligible or weighting not in {"equal_scope", "equal_family"}:
        raise ValueError("No eligible inner-training evidence or unknown weighting")
    values = {}
    for strength in STRENGTHS:
        subset = [r for r in eligible if r["strength"] == strength]
        if not subset:
            raise ValueError("Incomplete predeclared strength grid")
        if weighting == "equal_scope":
            values[strength] = float(np.mean([r["brier_alc"] for r in subset]))
        else:
            families = sorted({r["family"] for r in subset})
            values[strength] = float(np.mean([np.mean([r["brier_alc"] for r in subset if r["family"] == family]) for family in families]))
    # Ties prefer more smoothing; outcome order never chooses the tie-break.
    chosen = min(values, key=lambda k: (values[k], -k))
    return chosen, values


def analyze_evidence_policies(prepared, out):
    prepared, out = Path(prepared), Path(out)
    _, rows, targets, provenance = training_panel(prepared)
    index = {row["row_key"]: row for row in rows}
    episodes = list(read_jsonl(prepared / "evidence_plan_v1/evidence_episode_links.jsonl"))
    if len({e["scope_key"] for e in episodes}) != len(episodes):
        raise ValueError("Duplicate evidence scope")
    scores, budget_scores = [], []
    for episode in episodes:
        acquisition, query_groups = set(episode["acquisition_task_groups"]), set(episode["query_task_groups"])
        if acquisition & query_groups or len(acquisition) + len(query_groups) < 80 or len(episode["support_order_row_keys"]) < 31:
            raise ValueError("Smoothing selection requires full protocol coverage for every scope")
        supports = [index[key] for key in episode["support_order_row_keys"]]
        queries = [index[key] for key in episode["query_row_keys"]]
        if not queries or len({r["validation_family"] for r in supports + queries}) != 1:
            raise ValueError("Evidence scope crosses training families")
        if any(r["scope_key"] != episode["scope_key"] or r["task_group"] not in acquisition for r in supports):
            raise ValueError("Foreign or query-side support")
        if any(r["scope_key"] != episode["scope_key"] or r["task_group"] not in query_groups for r in queries):
            raise ValueError("Foreign or acquisition-side query")
        family = queries[0]["validation_family"]
        for strength in STRENGTHS:
            curve = {}
            for budget in BUDGETS:
                allowed = [targets[r["row_key"]] for r in supports[:budget]]
                probability = posterior_mean(allowed, strength)
                curve[budget] = float(np.mean([(probability - targets[r["row_key"]]) ** 2 for r in queries]))
                budget_scores.append({"scope_key": episode["scope_key"], "family": family, "strength": strength,
                                      "budget": budget, "probability": probability, "brier": curve[budget]})
            scores.append({"scope_key": episode["scope_key"], "family": family, "strength": strength,
                           "brier_alc": brier_alc(curve)})
    families = sorted({r["family"] for r in scores})
    if len(families) < 3:
        raise ValueError("Nested selection requires at least three training families")
    selections, nested, final = [], {}, {}
    for weighting in ["equal_scope", "equal_family"]:
        held_scores = []
        for family in families:
            strength, inner = select_strength(scores, excluded_family=family, weighting=weighting)
            selected = [r for r in scores if r["family"] == family and r["strength"] == strength]
            held_scores.extend(r["brier_alc"] for r in selected)
            selections.append({"strategy": weighting, "held_family": family, "selected_strength": strength,
                               "inner_scores": inner, "held_scopes": len(selected),
                               "held_brier_alc": float(np.mean([r["brier_alc"] for r in selected])),
                               "held_family_query_outcomes_used_for_selection": False})
        nested[weighting] = float(np.mean(held_scores))
        chosen, inner = select_strength(scores, weighting=weighting)
        final[weighting] = {"strength": chosen, "all_training_selection_scores": inner}
    fixed = {str(k): float(np.mean([r["brier_alc"] for r in scores if r["strength"] == k])) for k in STRENGTHS}
    # Select the strategy using training-family validation, then fit its one
    # hyperparameter on all training families. Original development stays unused.
    chosen_strategy = min(nested, key=lambda name: (nested[name], name))
    strength = final[chosen_strategy]["strength"]
    out.mkdir(parents=True, exist_ok=False)
    state = {"version": "evidence_policy_v1", "source_policy": "provided_dataset_only", "prior": .5,
        "strength": strength, "training_provenance": {"prepared_manifest_sha256": file_hash(prepared / "manifest.json"),
        "selection": "whole_training_family_outer_validation", "strategy": chosen_strategy,
        "training_scopes": len(episodes), "training_families": len(families),
        "predeclared_strengths": list(STRENGTHS), "original_development_outcomes_used": False}}
    write_json(out / "evidence_policy.json", state)
    report = {"version": "evidence_policy_selection_v1", "source_policy": "provided_dataset_only",
        "prepared_manifest_sha256": file_hash(prepared / "manifest.json"), "family_provenance": provenance,
        "training_rows": len(rows), "training_scopes": len(episodes), "training_families": len(families),
        "full_budget_coverage_complete": True, "fixed_strength_alc_equal_scope": fixed,
        "nested_strategy_alc_equal_scope": nested, "outer_folds": selections,
        "selected_strategy": chosen_strategy, "selected_strength": strength,
        "policy_sha256": file_hash(out / "evidence_policy.json"),
        "monitor_outcomes_used": False, "original_development_outcomes_used": False, "sealed_roles_opened": False,
        "native_laya_inference": False, "gpu_jobs": 0, "paid_calls": 0,
        "limitations": ["One predeclared acquisition trajectory; scope outcomes are dependent.",
            "Strength choices are evaluated outside each whole held training family.",
            "Strategy selection uses these training-family validation results; this is not a fresh final estimate.",
            "Fixed-grid scores fitted/compared on all training families are selection diagnostics, not unbiased generalization.",
            "The policy estimates the same mean for all tasks within a visible system/benchmark scope."]}
    write_jsonl(out / "strength_scope_scores.jsonl", scores)
    write_jsonl(out / "strength_budget_scores.jsonl", budget_scores)
    write_json(out / "summary.json", report)
    write_json(out / "manifest.json", {"artifacts": {p.name: file_hash(p) for p in sorted(out.iterdir()) if p.is_file()}})
    return report
