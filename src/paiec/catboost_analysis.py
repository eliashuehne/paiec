"""Whole-training-family CatBoost base-predictor experiment.

Every training row receives a prediction from a model fitted outside its whole
training family, and every occurrence of the held query tasks in other families
is excluded from fitting. Fixed controls share the same folds, rows and
acquisition episodes. Feature-group ablations and missingness/coverage
diagnostics are reported, but no candidate is promoted and no final model is
fitted on all training families.
"""
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .budget_evaluation import curve_scores, full_protocol_episodes
from .catboost_features import (CATALOG, DEFAULT_GROUPS, IDENTITY_GROUP, PROFILE_GROUP, VERSION,
                                categorical_names, feature_names, system_feature_record,
                                task_feature_record)
from .catboost_gate import assess_catboost_base, assess_research_only_catboost_base
from .system_profile_features import ProfileFacts
from .catboost_training import (checked_configs, fit_model, log_loss, probabilities, scope_weights,
                                select_configuration)
from .conditional_analysis import family_training_mask, training_panel
from .feature_preparation import read_jsonl, stable_hash, write_json, write_jsonl
from .head_training import file_hash
from .metrics import ece

VARIANTS = {
    "catboost": DEFAULT_GROUPS,
    "catboost_no_system_attributes": ("system_missingness", "task_structure", "item_supplied"),
    "catboost_no_system_missingness": ("system_attributes", "task_structure", "item_supplied"),
    "catboost_no_task_structure": ("system_attributes", "system_missingness", "item_supplied"),
    "catboost_no_item_supplied": ("system_attributes", "system_missingness", "task_structure"),
    "catboost_with_system_identity": (*DEFAULT_GROUPS, IDENTITY_GROUP),
}
ABLATIONS_NOT_DEPLOYABLE = ("catboost_with_system_identity",)


def feature_diagnostics(frames):
    """Declared columns, constant columns and missing categories per variant."""
    report = {}
    for name, frame in frames.items():
        categorical = [column for column in frame.columns if CATALOG[column][1] == "categorical"]
        report[name] = {
            "columns": int(frame.shape[1]),
            "categorical_columns": categorical,
            "numeric_columns": int(frame.shape[1]) - len(categorical),
            "constant_columns": [column for column in frame.columns if frame[column].nunique(dropna=False) <= 1],
            "missing_categories": {column: int((frame[column] == "__missing__").sum()) for column in categorical},
            "rows_with_any_missing_system_field": int((frame[[column for column in frame.columns
                                                             if column.startswith("missing_")]].sum(axis=1) > 0).sum()),
        }
    return report


def build_feature_frames(prepared, rows, variants=None, profile_facts=None, profile_permutation_seed=None):
    """Build one frame per declared variant.

    ``profile_facts`` is required only when a variant selects the declared
    ``system_profile`` group; without it those values would have to be invented.
    ``profile_permutation_seed`` reassigns profile records between systems. The
    column distribution and every within-scope constant stay identical while the
    system-to-fact link is destroyed, which is the control separating genuine
    capability information from a changed boosting path.
    """
    variants = VARIANTS if variants is None else variants
    prepared = Path(prepared)
    systems = {record["system_key"]: record["recorded"] for record in read_jsonl(prepared / "systems.jsonl")}
    tasks = {record["task_key"]: record["recorded"] for record in read_jsonl(prepared / "tasks.jsonl")}
    system_records = {key: system_feature_record(value) for key, value in systems.items()}
    task_records = {key: task_feature_record(value) for key, value in tasks.items()}
    needs_profiles = any(PROFILE_GROUP in groups for groups in variants.values())
    if needs_profiles and profile_facts is None:
        raise ValueError("The system_profile group needs a documented profile catalog")
    if needs_profiles:
        source = systems
        if profile_permutation_seed is not None:
            keys = list(systems)
            shuffled = np.random.default_rng(int(profile_permutation_seed)).permutation(len(keys))
            source = {key: systems[keys[index]] for key, index in zip(keys, shuffled)}
        profile_records = {key: profile_facts.record(value) for key, value in source.items()}
    else:
        profile_records = {}
    merged = [{**system_records[row["system_key"]], **task_records[row["task_key"]],
               **profile_records.get(row["system_key"], {})} for row in rows]
    frames = {}
    for name, groups in variants.items():
        columns = feature_names(groups)
        frames[name] = pd.DataFrame([{column: record[column] for column in columns} for record in merged],
                                    columns=columns)
    return frames, systems


def within_scope_auroc(labels, probabilities_values, scopes):
    """Mean Mann-Whitney AUROC within eligible visible scopes."""
    values = []
    for scope in sorted(set(scopes)):
        selected = scopes == scope
        y, p = labels[selected], probabilities_values[selected]
        positive, negative = int(y.sum()), int(len(y) - y.sum())
        if not positive or not negative:
            continue
        ranks = pd.Series(p).rank(method="average").to_numpy()
        values.append(float((ranks[y == 1].sum() - positive * (positive + 1) / 2) / (positive * negative)))
    return {"mean_within_scope_auroc": float(np.mean(values)) if values else None,
            "eligible_scopes": len(values)}


def cold_start_metrics(labels, probabilities_values, scopes):
    errors = []
    for scope in sorted(set(scopes)):
        selected = scopes == scope
        errors.append(float(np.mean((probabilities_values[selected] - labels[selected]) ** 2)))
    return {"equal_scope_brier": float(np.mean(errors)),
            "pooled_brier": float(np.mean((probabilities_values - labels) ** 2)),
            "logloss": log_loss(labels, probabilities_values),
            "ece": ece(labels, probabilities_values),
            "scopes": len(errors),
            **within_scope_auroc(labels, probabilities_values, scopes)}


def system_strata(rows, systems, held_index, fitted_index):
    """Positions of held rows whose exact system/harness/provider was unseen.

    The returned values are positions inside ``held_index``, so predictions can
    be gathered without a global row lookup.
    """
    fitted_systems = {rows[i]["system_key"] for i in fitted_index}
    fitted_harnesses = {systems[rows[i]["system_key"]]["harness"] for i in fitted_index}
    fitted_providers = {systems[rows[i]["system_key"]]["provider"] for i in fitted_index}
    strata = defaultdict(list)
    for position, i in enumerate(held_index):
        subject = systems[rows[i]["system_key"]]
        strata["exact_system_seen" if rows[i]["system_key"] in fitted_systems else "exact_system_unseen"].append(position)
        strata["harness_seen" if subject["harness"] in fitted_harnesses else "harness_unseen"].append(position)
        strata["provider_seen" if subject["provider"] in fitted_providers else "provider_unseen"].append(position)
    return strata


def checked_variants(recipe):
    """Declared variants for one recipe.

    A recipe that declares ``feature_groups`` fits exactly those columns as its
    primary model and runs no ablations: re-running the default ablation suite
    relative to a reduced base would not describe the same experiment.
    """
    declared = recipe.get("feature_groups")
    if declared is None:
        return dict(VARIANTS), False
    if (not isinstance(declared, list) or not declared or len(set(declared)) != len(declared)
            or any(not isinstance(group, str) for group in declared)):
        raise ValueError("Declared feature groups must be a nonempty list of unique names")
    # feature_names raises on unknown groups, so the catalog stays the contract.
    return {"catboost": tuple(declared)}, False


def fit_fold_models(frames, labels, scopes, family, groups, train, *, recipe, variants, seed, threads):
    """Select settings on inner families and fit every declared variant once.

    Shared by the panel experiment and the competition-shaped evaluation so the
    two cannot drift apart in weighting, grid or stopping policy.
    """
    primary_groups = variants["catboost"]
    selection = select_configuration(frames["catboost"], labels, scopes, family, groups, train,
                                     checked_configs(recipe["fit_configs"]),
                                     cat_features=categorical_names(primary_groups), seed=seed,
                                     threads=threads, max_iterations=int(recipe["max_iterations"]),
                                     early_stopping_rounds=int(recipe["early_stopping_rounds"]),
                                     min_iterations=int(recipe["min_iterations"]))
    weights = scope_weights(scopes[train])
    models = {name: fit_model(frame.loc[train], labels[train], weights,
                              categorical_names(variants[name]), selection["selected_config"],
                              iterations=selection["iterations"], seed=seed, threads=threads)
              for name, frame in frames.items()}
    return selection, models, weights


def profile_source(recipe):
    """Declared external profile catalog and policy, checked against each other."""
    catalog, policy = recipe.get("profile_catalog"), recipe.get("profile_policy")
    if catalog is None and policy is None:
        return None
    if not isinstance(catalog, str) or not isinstance(policy, str):
        raise ValueError("Declare both profile_catalog and profile_policy")
    from .profile_enrichment import ProfileEnricher
    root = Path(__file__).resolve().parents[2]
    if not (root / catalog).is_dir() or not (root / policy).is_file():
        raise ValueError("Declared profile catalog or policy is missing")
    declared = json.loads((root / policy).read_text()).get("catalog")
    if declared != catalog:
        raise ValueError("Profile policy points at a different catalog than the recipe")
    return ProfileFacts(root / policy, root=root)


def evaluate_catboost_base(prepared, out, *, recipe, profile_facts=None):
    prepared, out = Path(prepared), Path(out)
    if profile_facts is None:
        profile_facts = profile_source(recipe)
    source_policy = "provided_plus_research_profiles" if profile_facts else "provided_dataset_only"
    configs = checked_configs(recipe["fit_configs"])
    variants, ablations_run = checked_variants(recipe)
    primary_groups = variants["catboost"]
    threads, seed = int(recipe["threads"]), int(recipe["seed"])
    prior_strength, offset_regularization = float(recipe["prior_strength"]), float(recipe["offset_regularization"])
    summary, rows, targets, provenance = training_panel(prepared)
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    frames, systems = build_feature_frames(prepared, rows, variants, profile_facts,
                                           recipe.get("profile_permutation_seed"))
    labels = np.asarray([targets[row["row_key"]] for row in rows], dtype=float)
    scopes = np.asarray([row["scope_key"] for row in rows])
    family = np.asarray([row["validation_family"] for row in rows])
    groups = np.asarray([row["task_group"] for row in rows])
    diagnostics = feature_diagnostics(frames)
    write_json(out / "feature_diagnostics.json", diagnostics)
    predictions = {name: np.empty(len(rows)) for name in variants}
    training_prior = np.empty(len(rows))
    fold_reports, importance_rows, strata_rows = [], defaultdict(list), defaultdict(list)
    for held_family in sorted(set(family)):
        train, held = family_training_mask(family, groups, held_family)
        fold_started = time.monotonic()
        selection, models, weights = fit_fold_models(frames, labels, scopes, family, groups, train,
                                                     recipe=recipe, variants=variants, seed=seed, threads=threads)
        training_prior[held] = float(np.average(labels[train], weights=weights))
        fold_predictions = {}
        for name, frame in frames.items():
            model = models[name]
            fold_predictions[name] = probabilities(model, frame.loc[held])
            predictions[name][held] = fold_predictions[name]
            if name == "catboost":
                for column, value in zip(frame.columns, model.get_feature_importance()):
                    importance_rows[column].append(float(value))
        held_index = np.flatnonzero(held)
        strata = system_strata(rows, systems, held_index, np.flatnonzero(train))
        for stratum, positions in strata.items():
            strata_rows[stratum].extend((float(fold_predictions["catboost"][position]), float(labels[held_index[position]]))
                                        for position in positions)
        fold_reports.append({
            "held_training_family": str(held_family),
            "training_rows": int(train.sum()), "held_rows": int(held.sum()),
            "task_overlap_rows_excluded": int((~held & ~train).sum()),
            "training_row_keys_sha256": stable_hash([row["row_key"] for row, selected in zip(rows, train) if selected]),
            "held_row_keys_sha256": stable_hash([row["row_key"] for row, selected in zip(rows, held) if selected]),
            "selected_config": selection["selected_config"], "selected_index": selection["selected_index"],
            "iterations": selection["iterations"],
            "inner_candidates": [{"config": row["config"], "mean_inner_logloss": row["mean_inner_logloss"]}
                                 for row in selection["candidates"]],
            "trained_scope_weight_mean": float(np.average(labels[train], weights=weights)),
            "fit_seconds": time.monotonic() - fold_started,
        })
        fold_report = fold_reports[-1]
        held_report = {}
        for name, values in fold_predictions.items():
            metrics = cold_start_metrics(labels[held], values, scopes[held])
            held_report[name] = {key: metrics[key] for key in
                                 ["equal_scope_brier", "pooled_brier", "logloss", "ece", "mean_within_scope_auroc"]}
        held_report["training_base_rate"] = cold_start_metrics(labels[held], np.full(int(held.sum()), training_prior[held]),
                                                               scopes[held])["equal_scope_brier"]
        held_report["observed_rate"] = float(labels[held].mean())
        held_report["mean_probability"] = float(fold_predictions["catboost"].mean())
        fold_report["held_metrics"] = held_report
        write_json(out / "progress.json", {"phase": "family_cross_validation",
                                           "completed_families": len(fold_reports),
                                           "total_families": len(set(family)),
                                           "latest_family": str(held_family),
                                           "elapsed_seconds": time.monotonic() - started})
    write_jsonl(out / "out_of_family_predictions.jsonl", ({
        "row_key": row["row_key"], "validation_family": row["validation_family"],
        "training_base_rate": float(training_prior[i]),
        **{name: float(values[i]) for name, values in predictions.items()},
    } for i, row in enumerate(rows)))
    cold_metrics = {"constant": cold_start_metrics(labels, np.full(len(rows), .5), scopes),
                    "training_base_rate": cold_start_metrics(labels, training_prior, scopes),
                    **{name: cold_start_metrics(labels, values, scopes) for name, values in predictions.items()}}
    episodes, excluded = full_protocol_episodes(prepared, rows, scopes, family)
    scored, curves, alc = curve_scores(episodes, labels, predictions, prior_strength=prior_strength,
                                       offset_regularization=offset_regularization)
    strata = {name: {"rows": len(values),
                     "brier": float(np.mean([(p - y) ** 2 for p, y in values])) if values else None}
              for name, values in sorted(strata_rows.items())}
    importance = sorted(((float(np.mean(values)), column) for column, values in importance_rows.items()), reverse=True)
    report = {
        "version": "catboost_base_analysis_v1", "source_policy": source_policy,
        "feature_version": VERSION, "recipe": recipe,
        "profile_source": ({**profile_facts.metadata,
                            "usage": "research_only",
                            "eligibility": "external_profile_sources_remain_unreviewed",
                            "permutation_seed": recipe.get("profile_permutation_seed")}
                           if profile_facts else None),
        "feature_groups": list(primary_groups), "ablations_run": ablations_run,
        "fitting_role": "training_only",
        "validation": "leave_training_family_out_and_exclude_repeated_query_tasks",
        "family_provenance": provenance, "training_rows": len(rows),
        "training_families": len(fold_reports), "visible_scopes": len(set(scopes)),
        "folds": fold_reports,
        "cold_start_metrics": cold_metrics,
        "zero_budget_brier_all_held_rows_equal_scope": {name: metrics["equal_scope_brier"]
                                                        for name, metrics in cold_metrics.items()},
        "full_budget_eligible_scopes": len(episodes), "excluded_small_scopes": len(excluded),
        "per_budget_on_common_eligible_scopes": curves,
        "brier_alc_on_common_eligible_scopes": alc,
        "feature_diagnostics_reference": "feature_diagnostics.json",
        "feature_importance_mean_across_folds_top15": [
            {"feature": column, "group": CATALOG[column][0], "importance": value} for value, column in importance[:15]],
        "feature_importance_is_causal_evidence": False,
        "unfamiliar_system_strata": strata,
        "ablation_models_not_deployable": list(ABLATIONS_NOT_DEPLOYABLE),
        "model_selection_performed": True,
        "selection_scope": "inner training-family folds only; outer held outcomes never select settings",
        "final_all_training_model_fitted": False,
        "original_development_outcomes_used": False, "monitor_outcomes_used": False,
        "sealed_roles_opened": False, "paid_calls": 0, "gpu_jobs": 0, "native_laya_inference": False,
        "runtime_seconds": time.monotonic() - started,
        "limitations": [
            "This is a training-family validation study, not a hidden-competition or original-development result.",
            "Held-out families contribute unseen harnesses and very different outcome rates; cold-start transfer is the binding difficulty.",
            "The system-identity ablation can memorize training systems and is reported as a diagnostic only.",
            "The 30k+ export contains one selected trial per original system/item.",
            "Feature importance is predictive split usage, not causal evidence.",
            *(["The system_profile group uses external research catalogs, so this model is research-only and not a release candidate.",
               "Profile facts describe documented configurations; no evaluated checkpoint or enabled setting is verified."]
              if profile_facts else []),
        ],
    }
    write_jsonl(out / "pair_budget_scores.jsonl", scored)
    assessment = (assess_catboost_base(report) if profile_facts is None
                  else assess_research_only_catboost_base(report))
    write_json(out / "candidate_assessment.json", assessment)
    write_json(out / "summary.json", report)
    write_json(out / "progress.json", {"phase": "finished", "training_rows": len(rows), "families": len(fold_reports)})
    write_json(out / "manifest.json", {"artifacts": {str(path.relative_to(out)): file_hash(path)
                                                     for path in sorted(out.rglob("*")) if path.is_file()}})
    return report
