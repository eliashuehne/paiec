"""Additive two-block logistic base predictor: system strength plus task difficulty.

``logit p = b0 + f_system(x_system) + f_task(x_task)`` with no cross-block term.
The model has no per-system offset at all, so an unseen system's contribution
comes only from its supplied attributes. That is the hypothesis under test: can
supplied system attributes carry strength, and task text carry difficulty,
without learning interactions between them?

The columns, weighting, folds and controls are the CatBoost base's, so the two
models are directly comparable. Low-cardinality categoricals are one-hot
encoded from the fitting fold only, unseen levels fall back to the intercept,
numeric columns are standardized with fitting-fold statistics, and the two block
penalties are chosen from a predeclared grid by inner training-family log loss.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .budget_evaluation import curve_scores, full_protocol_episodes
from .catboost_analysis import (build_feature_frames, checked_variants, cold_start_metrics,
                                feature_diagnostics, system_strata)
from .catboost_features import CATALOG, categorical_names, feature_names
from .catboost_gate import assess_catboost_base
from .catboost_training import log_loss, scope_weights
from .catboost_unseen_benchmark import (UnseenBenchmarkConfig, _fold_identity, build_external_frame,
                                        load_external_evaluation, normalize_name, subject_identity)
from .conditional_analysis import family_training_mask, training_panel
from .feature_preparation import stable_hash, write_json, write_jsonl
from .grouped_data import checked_artifacts
from .head_training import file_hash

PENALTY_KEYS = ("l2_system", "l2_task")
SYSTEM_GROUPS = ("system_attributes", "system_missingness")
TASK_GROUPS = ("task_structure", "item_supplied")
DEFAULT_GROUPS = (*SYSTEM_GROUPS, *TASK_GROUPS)
DEFAULT_PENALTIES = [{"l2_system": 10.0, "l2_task": 10.0},
                     {"l2_system": 100.0, "l2_task": 10.0},
                     {"l2_system": 10.0, "l2_task": 100.0},
                     {"l2_system": 100.0, "l2_task": 100.0}]


def checked_penalties(grid):
    """Validate the predeclared two-block penalty grid."""
    if not isinstance(grid, list) or not grid:
        raise ValueError("A nonempty predeclared penalty grid is required")
    checked = []
    for config in grid:
        if not isinstance(config, dict) or set(config) != set(PENALTY_KEYS):
            raise ValueError("Each penalty configuration needs l2_system and l2_task")
        values = {}
        for key in PENALTY_KEYS:
            value = config[key]
            if type(value) not in {int, float} or not math.isfinite(value) or value <= 0:
                raise ValueError("Penalties must be positive and finite")
            values[key] = float(value)
        checked.append(values)
    if len({tuple(sorted(config.items())) for config in checked}) != len(checked):
        raise ValueError("Duplicate penalty configuration")
    return checked


def block_columns(groups):
    """Numeric and categorical columns for each block, in catalog order."""
    numeric = [name for name in feature_names(groups) if CATALOG[name][1] != "categorical"]
    categorical = categorical_names(groups)
    system = [name for name in numeric + categorical if CATALOG[name][0] in SYSTEM_GROUPS]
    task = [name for name in numeric + categorical if CATALOG[name][0] in TASK_GROUPS]
    if set(system) & set(task) or sorted(system + task) != sorted(numeric + categorical):
        raise ValueError("Blocks must partition the declared columns")
    return {"numeric": numeric, "categorical": categorical, "system": system, "task": task}


def _design(frame, columns, categories, means, scale):
    """Intercept, one-hot block indices and standardized numeric columns."""
    rows = len(frame)
    system_columns = set(columns["system"])
    pieces, names, blocks = [np.ones((rows, 1))], ["intercept"], [0]
    for column in columns["categorical"]:
        block = 1 if column in system_columns else 2
        values = frame[column].astype(str).to_numpy()
        for level in categories[column][1:]:
            pieces.append((values == level).astype(float).reshape(-1, 1))
            names.append(f"{column}={level}")
            blocks.append(block)
    for column in columns["numeric"]:
        block = 1 if column in system_columns else 2
        values = np.asarray(frame[column], dtype=float)
        pieces.append(((values - means[column]) / scale[column]).reshape(-1, 1))
        names.append(column)
        blocks.append(block)
    return np.hstack(pieces), names, np.asarray(blocks)


def _penalty_vector(blocks, l2_system, l2_task):
    return np.where(blocks == 1, l2_system, np.where(blocks == 2, l2_task, 0.))


def fit_additive(frame, labels, sample_weight, columns, *, l2_system, l2_task,
                 max_iterations=100, tolerance=1e-8):
    """Weighted L2 logistic regression with block-wise penalties.

    Newton/IRLS on the standardized design. Every coefficient has a finite
    penalty, so a constant column simply keeps a zero contribution. The fit
    raises rather than returning an unconverged state.
    """
    labels = np.asarray(labels, dtype=float)
    sample_weight = np.asarray(sample_weight, dtype=float)
    if (frame.shape[0] < 2 or labels.shape != (frame.shape[0],) or
            sample_weight.shape != labels.shape or not np.isin(labels, [0, 1]).all() or
            not np.isfinite(sample_weight).all() or (sample_weight <= 0).any()):
        raise ValueError("Fitting requires aligned binary targets and positive weights")
    categories = {column: sorted(frame[column].astype(str).unique())
                  for column in columns["categorical"]}
    numeric = np.asarray(frame[columns["numeric"]], dtype=float)
    means = {column: float(numeric[:, index].mean())
             for index, column in enumerate(columns["numeric"])}
    scale = {column: float(numeric[:, index].std()) or 1.
             for index, column in enumerate(columns["numeric"])}
    design, names, blocks = _design(frame, columns, categories, means, scale)
    penalty = _penalty_vector(blocks, float(l2_system), float(l2_task))

    def state(coefficients, iterations, objective):
        return {"version": "additive_logit_v1", "columns": columns, "categories": categories,
                "means": means, "scale": scale, "names": names, "blocks": blocks.tolist(),
                "coefficients": np.asarray(coefficients).tolist(), "l2_system": float(l2_system),
                "l2_task": float(l2_task), "iterations": int(iterations),
                "objective": float(objective)}

    def objective(coefficients):
        linear = np.clip(design @ coefficients, -30, 30)
        probability = 1. / (1. + np.exp(-linear))
        return float(-np.sum(sample_weight * (labels * np.log(np.clip(probability, 1e-12, 1))
                                             + (1 - labels) * np.log(np.clip(1 - probability, 1e-12, 1)))))

    beta = np.zeros(design.shape[1])
    previous = objective(beta)
    for iteration in range(1, int(max_iterations) + 1):
        linear = np.clip(design @ beta, -30, 30)
        probability = 1. / (1. + np.exp(-linear))
        curvature = np.clip(probability * (1. - probability), 1e-9, None)
        working = linear + (labels - probability) / curvature
        weights = sample_weight * curvature
        hessian = design.T @ (weights[:, None] * design) + np.diag(penalty)
        # Penalized Newton step: the ridge term belongs in the score as well as
        # the curvature, otherwise the penalty never shrinks the current state.
        gradient = design.T @ (weights * (working - linear)) - penalty * beta
        step = np.linalg.solve(hessian, gradient)
        beta = beta + step
        current = objective(beta)
        # Newton converges quadratically; a coefficient step below the declared
        # tolerance is the scale-free stopping rule. Non-convergence raises.
        if np.max(np.abs(step)) <= tolerance:
            return state(beta, iteration, current)
        previous = current
    raise ValueError("Additive logistic fit did not converge")


def predict_additive(state, frame):
    """Probabilities from a fitted additive state; unseen categories drop out."""
    design, _, _ = _design(frame, state["columns"], state["categories"], state["means"], state["scale"])
    if design.shape[1] != len(state["coefficients"]):
        raise ValueError("Design matrix does not match the fitted state")
    linear = design @ np.asarray(state["coefficients"], dtype=float)
    return np.clip(1. / (1. + np.exp(-np.clip(linear, -30, 30))), 1e-6, 1 - 1e-6)


def select_penalties(frame, labels, scopes, families, groups, outer_train, grid, columns, *,
                     max_iterations, tolerance):
    """Choose block penalties from inner training-family folds only."""
    outer_train = np.asarray(outer_train, dtype=bool)
    scopes, families, groups = np.asarray(scopes), np.asarray(families), np.asarray(groups)
    inner_families = sorted(set(families[outer_train]))
    if len(inner_families) < 2:
        raise ValueError("Inner selection needs at least two training families")
    rows = []
    for config in grid:
        folds = []
        for inner_family in inner_families:
            inner_train, inner_held = family_training_mask(families, groups, inner_family)
            inner_train &= outer_train
            inner_held &= outer_train
            if not inner_train.any() or not inner_held.any():
                raise ValueError("Inner fold has an empty fitting or validation side")
            state = fit_additive(frame.loc[inner_train], labels[inner_train],
                                 scope_weights(scopes[inner_train]), columns,
                                 l2_system=config["l2_system"], l2_task=config["l2_task"],
                                 max_iterations=max_iterations, tolerance=tolerance)
            held = predict_additive(state, frame.loc[inner_held])
            folds.append({"inner_held_family": str(inner_family),
                          "logloss": log_loss(labels[inner_held], held),
                          "brier": float(np.mean((held - labels[inner_held]) ** 2)),
                          "held_rows": int(inner_held.sum())})
        total = sum(row["held_rows"] for row in folds)
        rows.append({"config": dict(config),
                     "mean_inner_logloss": float(sum(row["logloss"] * row["held_rows"]
                                                     for row in folds) / total),
                     "inner_folds": folds})
    chosen = min(range(len(rows)), key=lambda index: rows[index]["mean_inner_logloss"])
    return {"selected_config": rows[chosen]["config"], "selected_index": chosen, "candidates": rows}


def evaluate_additive(root, prepared, out, *, recipe):
    """Panel cold start, budget curves and optionally the competition-shaped set."""
    root, prepared, out = Path(root).resolve(), Path(prepared), Path(out).resolve()
    variants, ablations_run = checked_variants(recipe)
    if ablations_run:
        raise ValueError("The additive model needs an explicit declared feature-group set")
    groups = variants["catboost"]
    columns_all = feature_names(groups)
    columns = block_columns(groups)
    grid = checked_penalties(recipe["penalty_grid"])
    max_iterations, tolerance = int(recipe["max_iterations"]), float(recipe["tolerance"])
    prior_strength, offset_regularization = float(recipe["prior_strength"]), float(recipe["offset_regularization"])
    _, rows, targets, provenance = training_panel(prepared)
    out.mkdir(parents=True, exist_ok=False)
    frames, systems = build_feature_frames(prepared, rows, variants)
    frame = frames["catboost"]
    labels = np.asarray([targets[row["row_key"]] for row in rows], dtype=float)
    scopes = np.asarray([row["scope_key"] for row in rows])
    family = np.asarray([row["validation_family"] for row in rows])
    task_groups = np.asarray([row["task_group"] for row in rows])
    predictions = np.empty(len(rows))
    training_prior = np.empty(len(rows))
    fold_reports, strata_rows = [], {}
    external = None
    if recipe.get("unseen_benchmark"):
        settings = UnseenBenchmarkConfig.model_validate(
            {"panel_run": recipe["prepared"], **recipe["unseen_benchmark"]})
        panel_names = {normalize_name(systems[row["system_key"]]["normalized_name"]) for row in rows}
        panel_names.discard("")
        external = load_external_evaluation(root, settings.data_root, set(task_groups.tolist()),
                                            panel_names, settings)
        external_rows = external["rows"]
        external_frame = build_external_frame(external_rows, columns_all)
        external_labels = np.asarray([row["label"] for row in external_rows], dtype=float)
        external_scopes = np.asarray([row["scope_key"] for row in external_rows], dtype=object)
        external_names = np.asarray([normalize_name(row["subject"]["normalized_name"])
                                     for row in external_rows], dtype=object)
        external_identity = [subject_identity(row["subject"]) for row in external_rows]
        external_predictions = {}
        external_exact = {}
    for held_family in sorted(set(family.tolist())):
        train, held = family_training_mask(family, task_groups, held_family)
        selection = select_penalties(frame, labels, scopes, family, task_groups, train, grid, columns,
                                     max_iterations=max_iterations, tolerance=tolerance)
        weights = scope_weights(scopes[train])
        state = fit_additive(frame.loc[train], labels[train], weights, columns,
                             l2_system=selection["selected_config"]["l2_system"],
                             l2_task=selection["selected_config"]["l2_task"],
                             max_iterations=max_iterations, tolerance=tolerance)
        held_values = predict_additive(state, frame.loc[held])
        predictions[held] = held_values
        training_prior[held] = float(np.average(labels[train], weights=weights))
        metrics = {key: cold_start_metrics(labels[held], held_values, scopes[held])[key]
                   for key in ["equal_scope_brier", "pooled_brier", "logloss", "ece",
                               "mean_within_scope_auroc"]}
        fold_report = {"held_training_family": str(held_family),
                       "training_rows": int(train.sum()), "held_rows": int(held.sum()),
                       "task_overlap_rows_excluded": int((~held & ~train).sum()),
                       "selected_penalties": selection["selected_config"],
                       "inner_candidates": [{"config": row["config"],
                                             "mean_inner_logloss": row["mean_inner_logloss"]}
                                            for row in selection["candidates"]],
                       "training_base_rate": training_prior[held][0],
                       "observed_rate": float(labels[held].mean()),
                       "mean_probability": float(held_values.mean()),
                       "held_metrics": metrics,
                       "training_row_keys_sha256": stable_hash([row["row_key"] for row, chosen
                                                                in zip(rows, train) if chosen]),
                       "held_row_keys_sha256": stable_hash([row["row_key"] for row, chosen
                                                            in zip(rows, held) if chosen])}
        for name, positions in system_strata(rows, systems, np.flatnonzero(held),
                                             np.flatnonzero(train)).items():
            strata_rows.setdefault(name, []).extend(
                (float(held_values[position]), float(labels[held][position])) for position in positions)
        if external is not None:
            fit_names, fit_identity = _fold_identity(rows, systems, train)
            eligible = [index for index in range(len(external_rows))
                        if external_names[index]
                        and (external_names[index] in fit_names or external_identity[index] in fit_identity)]
            if eligible:
                values = predict_additive(state, external_frame.iloc[eligible])
                for index, value in zip(eligible, values):
                    external_predictions.setdefault(index, []).append(float(value))
                    if external_identity[index] in fit_identity:
                        external_exact[index] = True
                fold_report["external_metrics"] = {
                    key: cold_start_metrics(external_labels[eligible], values,
                                            external_scopes[eligible])[key]
                    for key in ["equal_scope_brier", "pooled_brier", "logloss", "ece",
                                "mean_within_scope_auroc"]}
                fold_report["external_eligible_rows"] = len(eligible)
        fold_reports.append(fold_report)
    cold_metrics = {"constant": cold_start_metrics(labels, np.full(len(rows), .5), scopes),
                    "training_base_rate": cold_start_metrics(labels, training_prior, scopes),
                    "additive_logit": cold_start_metrics(labels, predictions, scopes)}
    episodes, excluded = full_protocol_episodes(prepared, rows, scopes, family)
    scored, curves, alc = curve_scores(episodes, labels, {"additive_logit": predictions},
                                       prior_strength=prior_strength,
                                       offset_regularization=offset_regularization)
    strata = {name: {"rows": len(values),
                     "brier": float(np.mean([(p - y) ** 2 for p, y in values])) if values else None}
              for name, values in sorted(strata_rows.items())}
    report = {
        "version": "additive_logit_analysis_v1", "source_policy": "provided_dataset_only",
        "model": "additive_logit_v1", "feature_groups": list(groups), "recipe": recipe,
        "fitting_role": "training_only",
        "validation": "leave_training_family_out_and_exclude_repeated_query_tasks",
        "family_provenance": provenance, "training_rows": len(rows),
        "training_families": len(fold_reports), "visible_scopes": len(set(scopes)),
        "additivity": {"cross_block_terms": 0, "per_system_offsets": 0,
                       "unseen_system_contribution": "supplied attributes only"},
        "folds": fold_reports,
        "cold_start_metrics": cold_metrics,
        "zero_budget_brier_all_held_rows_equal_scope":
            {name: metrics["equal_scope_brier"] for name, metrics in cold_metrics.items()},
        "full_budget_eligible_scopes": len(episodes), "excluded_small_scopes": len(excluded),
        "per_budget_on_common_eligible_scopes": curves,
        "brier_alc_on_common_eligible_scopes": alc,
        "unfamiliar_system_strata": strata,
        "feature_diagnostics_reference": "feature_diagnostics.json",
        "model_selection_performed": True,
        "selection_scope": "inner training-family folds only; outer held outcomes never select settings",
        "final_all_training_model_fitted": False,
        "original_development_outcomes_used": False, "monitor_outcomes_used": False,
        "sealed_roles_opened": False, "paid_calls": 0, "gpu_jobs": 0, "native_laya_inference": False,
        "limitations": [
            "Training-family validation; not a hidden-competition or development result.",
            "No cross-block interaction, so a task whose difficulty depends on the system is unrepresentable.",
            "No per-system offset, so within-benchmark system strength is not modelled separately.",
        ],
    }
    if external is not None:
        scored_rows = sorted(external_predictions)
        averaged = np.asarray([float(np.mean(external_predictions[index])) for index in scored_rows])
        exact_mask = np.asarray([bool(external_exact.get(index)) for index in scored_rows])
        report["unseen_benchmark"] = {
            "admitted_benchmarks": [benchmark.name for benchmark in external["admitted"]],
            "rows": len(scored_rows),
            "primary_equal_scope_brier": float(np.mean(
                [row["external_metrics"]["equal_scope_brier"] for row in fold_reports
                 if "external_metrics" in row])),
            "pooled_metrics": {key: cold_start_metrics(external_labels[scored_rows], averaged,
                                                       external_scopes[scored_rows])[key]
                               for key in ["equal_scope_brier", "pooled_brier", "logloss", "ece",
                                           "mean_within_scope_auroc"]},
            "exact_subject_rows": int(exact_mask.sum()),
            "constant_0.5": .25,
        }
        write_jsonl(out / "unseen_row_predictions.jsonl", (
            {"benchmark": external["rows"][index]["benchmark"],
             "scope_key": external["rows"][index]["scope_key"],
             "label": int(external_labels[index]),
             "exact_subject_identity": bool(external_exact.get(index)),
             "prediction": float(np.mean(external_predictions[index]))} for index in scored_rows))
    write_json(out / "feature_diagnostics.json", feature_diagnostics({"additive_logit": frame}))
    write_jsonl(out / "out_of_family_predictions.jsonl", (
        {"row_key": row["row_key"], "validation_family": row["validation_family"],
         "training_base_rate": float(training_prior[index]),
         "additive_logit": float(predictions[index])} for index, row in enumerate(rows)))
    write_jsonl(out / "pair_budget_scores.jsonl", scored)
    write_json(out / "candidate_assessment.json",
               assess_catboost_base(report, deployable=("additive_logit",), base="additive_logit"))
    write_json(out / "summary.json", report)
    write_json(out / "manifest.json", {"artifacts": {
        str(path.relative_to(out)): file_hash(path)
        for path in sorted(out.rglob("*")) if path.is_file() and path.name != "manifest.json"}})
    return report
