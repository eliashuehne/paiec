"""Uncertainty for whole-family CatBoost ablation margins.

The ablation table reports equal-scope Brier, so a margin is an average over 170
visible scopes that share systems and questions. A row-level interval would
overstate precision. The primary interval here resamples whole scopes as
clusters and is reported as a lower bound on uncertainty; a leave-one-family-out
jackknife supplies the coarse family-level view, which has only seven clusters
and is therefore nearly powerless.

Everything is computed from saved out-of-family predictions and recorded
targets. Nothing is refitted and no candidate is promoted.
"""
import json
import math
from pathlib import Path

import numpy as np

from .conditional_analysis import training_panel
from .feature_preparation import read_jsonl, write_json
from .grouped_data import checked_artifacts
from .head_training import file_hash


def _checked(values, name):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError(f"{name} must be a nonempty finite vector")
    return values


def paired_scope_errors(candidate, reference, labels, scopes):
    """Per-row paired squared-error difference, candidate minus reference.

    A positive value means the candidate is worse than the reference on that
    row. ``scopes`` supplies the equally weighted evaluation units.
    """
    candidate, reference, labels = (_checked(candidate, "candidate"), _checked(reference, "reference"),
                                    _checked(labels, "labels"))
    scopes = np.asarray(scopes, dtype=object)
    if not (candidate.shape == reference.shape == labels.shape == scopes.shape):
        raise ValueError("Paired comparison needs aligned predictions, labels and scopes")
    if not np.isin(labels, [0., 1.]).all():
        raise ValueError("Paired comparison needs binary targets")
    return (candidate - labels) ** 2 - (reference - labels) ** 2


def scope_means(values, scopes):
    """Mean per scope in sorted scope order, with the matching scope names."""
    values = _checked(values, "values")
    scopes = np.asarray(scopes, dtype=object)
    if values.shape != scopes.shape:
        raise ValueError("Scope means need aligned values and scopes")
    names = np.asarray(sorted(set(scopes.tolist())), dtype=object)
    if not len(names):
        raise ValueError("Scope means need at least one scope")
    means = np.asarray([float(values[scopes == name].mean()) for name in names], dtype=float)
    return names, means


def equal_scope(values, scopes):
    """Mean of per-scope means: every visible scope carries equal weight."""
    return float(scope_means(values, scopes)[1].mean())


def clustered_bootstrap(differences, scopes, *, draws=2000, seed=0, alpha=.05):
    """Percentile interval from resampling whole scopes with replacement.

    Returns the point estimate, the interval, the share of draws in which the
    candidate is worse, and the number of scopes contributing.
    """
    differences = _checked(differences, "differences")
    if type(draws) is not int or draws < 1:
        raise ValueError("Bootstrap draws must be a positive integer")
    if type(seed) is not int:
        raise ValueError("Bootstrap seed must be an integer")
    if not 0 < alpha < 1:
        raise ValueError("Alpha must lie strictly between zero and one")
    names, means = scope_means(differences, scopes)
    if len(names) < 2:
        raise ValueError("A clustered bootstrap needs at least two scopes")
    rng = np.random.default_rng(seed)
    sampled = means[rng.integers(0, len(means), size=(draws, len(means)))].mean(axis=1)
    lower, upper = (float(value) for value in
                    np.percentile(sampled, [100 * alpha / 2, 100 * (1 - alpha / 2)]))
    return {"point": float(means.mean()), "lower": lower, "upper": upper,
            "share_candidate_worse": float(np.mean(sampled > 0)),
            "scopes": int(len(names)), "draws": int(draws), "alpha": alpha}


def family_jackknife(differences, scopes, families):
    """Equal-scope difference with each family's scopes removed in turn."""
    differences = _checked(differences, "differences")
    scopes, families = np.asarray(scopes, dtype=object), np.asarray(families, dtype=object)
    if differences.shape != scopes.shape or differences.shape != families.shape:
        raise ValueError("Jackknife needs aligned values, scopes and families")
    names = sorted(set(families.tolist()))
    if len(names) < 2:
        raise ValueError("A family jackknife needs at least two families")
    rows = {}
    for name in names:
        retained = families != name
        if not retained.any():
            raise ValueError("Removing a family left no scopes")
        rows[str(name)] = equal_scope(differences[retained], scopes[retained])
    return rows


def _scope_consistency(differences, scopes):
    """How many scopes favour the candidate, ties included as neither."""
    names, means = scope_means(differences, scopes)
    return {"candidate_worse_scopes": int((means > 0).sum()),
            "candidate_better_scopes": int((means < 0).sum()),
            "tied_scopes": int((means == 0).sum()),
            "scopes": int(len(names))}


def analyze_ablation_margins(run, out, *, draws=2000, seed=20261024, alpha=.05):
    """Score every recorded ablation margin against the full CatBoost model.

    The run's artifact manifest is re-hashed, the saved predictions must cover
    exactly the recorded training rows, and every recomputed equal-scope Brier
    must reproduce the recorded value before any interval is reported.
    """
    run, out = Path(run).resolve(), Path(out).resolve()
    if out.exists():
        raise ValueError("Use a fresh uncertainty output directory")
    manifest = json.loads((run / "manifest.json").read_text())
    checked_artifacts(run, manifest["artifacts"])
    recipe = json.loads((run / "resolved_config.json").read_text())
    summary = json.loads((run / "summary.json").read_text())
    if summary.get("source_policy") != "provided_dataset_only":
        raise ValueError("Uncertainty requires a dataset-only run")
    _, rows, targets, _ = training_panel(recipe["prepared"])
    scope = {row["row_key"]: row["scope_key"] for row in rows}
    family = {row["row_key"]: row["validation_family"] for row in rows}
    saved = {row["row_key"]: row for row in
             read_jsonl(run / "out_of_family_predictions.jsonl")}
    order = [row["row_key"] for row in rows]
    if set(saved) != set(order) or set(targets) != set(order):
        raise ValueError("Saved predictions do not cover exactly the recorded training rows")
    labels = np.asarray([targets[key] for key in order], dtype=float)
    scopes = np.asarray([scope[key] for key in order], dtype=object)
    families = np.asarray([family[key] for key in order], dtype=object)
    names = sorted(set(saved[order[0]]) - {"row_key", "validation_family", "training_base_rate"})
    if "catboost" not in names or len(names) < 2:
        raise ValueError("Uncertainty needs the full model and at least one ablation")
    prediction = {name: np.asarray([saved[key][name] for key in order], dtype=float) for name in names}
    recorded = summary["zero_budget_brier_all_held_rows_equal_scope"]
    reproduced = {}
    for name, values in prediction.items():
        recomputed = equal_scope((values - labels) ** 2, scopes)
        if name not in recorded or abs(recomputed - float(recorded[name])) > 1e-9:
            raise ValueError("Saved predictions do not reproduce the recorded score: " + name)
        reproduced[name] = recomputed
    out.mkdir(parents=True, exist_ok=False)
    reference = prediction["catboost"]
    comparisons = {}
    for name in names:
        if name == "catboost":
            continue
        differences = paired_scope_errors(prediction[name], reference, labels, scopes)
        interval = clustered_bootstrap(differences, scopes, draws=draws, seed=seed, alpha=alpha)
        jackknife = family_jackknife(differences, scopes, families)
        comparisons[name] = {"equal_scope_brier": reproduced[name],
                             "difference_vs_full_model": interval["point"],
                             "difference_sign": "worse_than_full" if interval["point"] > 0 else "better_than_full",
                             "interval": interval, "scope_consistency": _scope_consistency(differences, scopes),
                             **margin_assessment(interval["point"], interval, jackknife)}
    report = {
        "version": "catboost_base_uncertainty_v1",
        "source_policy": "provided_dataset_only",
        "source_run": {"path": str(run), "recipe_version": recipe["version"],
                       "manifest_sha256": file_hash(run / "manifest.json"),
                       "summary_sha256": file_hash(run / "summary.json")},
        "reference_model": "catboost",
        "training_rows": len(order), "visible_scopes": int(len(set(scopes.tolist()))),
        "validation_families": int(len(set(families.tolist()))),
        "reproduced_equal_scope_brier": reproduced,
        "reproduction_tolerance": 1e-9,
        "design": {"primary_interval": "percentile bootstrap resampling whole visible scopes",
                   "draws": int(draws), "seed": int(seed), "alpha": float(alpha),
                   "secondary_view": "leave-one-training-family-out jackknife",
                   "decision_rule": "distinguishable only if the interval excludes zero and every family jackknife keeps the same sign"},
        "comparisons": comparisons,
        "model_fitted": False, "candidate_promoted": False, "outcomes_opened_outside_training": False,
        "limitations": [
            "Scopes share systems and questions, so the scope bootstrap is a lower bound on uncertainty rather than a calibrated interval.",
            "Seven training families give the jackknife almost no power; it is a coarse sign check, not an interval.",
            "Ablations are evidence about feature groups, not release candidates.",
            "No multiplicity correction across the six reported margins.",
        ],
    }
    write_json(out / "summary.json", report)
    write_json(out / "manifest.json", {"artifacts": {
        str(path.relative_to(out)): file_hash(path)
        for path in sorted(out.rglob("*")) if path.is_file() and path.name != "manifest.json"}})
    return report


def margin_assessment(point, interval, jackknife):
    """Predeclared reading of one ablation margin.

    A margin is distinguishable from zero only when the scope-bootstrap interval
    excludes zero **and** every leave-one-family-out estimate keeps its sign.
    The two conditions answer different questions and are never merged into a
    significance claim.
    """
    if not (math.isfinite(point) and math.isfinite(interval["lower"])
            and math.isfinite(interval["upper"])):
        raise ValueError("Margin assessment needs finite values")
    if not jackknife:
        raise ValueError("Margin assessment needs a family jackknife")
    signs = {value > 0 for value in jackknife.values()}
    same_sign = len(signs) == 1 and (point > 0) == next(iter(signs))
    excludes_zero = interval["lower"] > 0 or interval["upper"] < 0
    return {"point": point, "interval_lower": interval["lower"],
            "interval_upper": interval["upper"],
            "interval_excludes_zero": bool(excludes_zero),
            "family_jackknife_same_sign": bool(same_sign),
            "family_jackknife": {name: value for name, value in jackknife.items()},
            "distinguishable_from_zero": bool(excludes_zero and same_sign),
            "reading": ("distinguishable" if excludes_zero and same_sign else
                        "not_distinguishable_at_this_design")}
