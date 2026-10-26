"""Whole-training-family analysis of dataset-only conditional predictors."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .budget_evaluation import curve_scores, full_protocol_episodes
from .competition_analysis import verify_competition_bundle
from .conditional_features import VERSION, DIMS, feature_vector, task_features, system_features
from .conditional_training import fit_conditional_ridge, predict_matrix
from .conditional_gate import assess_conditional_controls
from .feature_preparation import read_jsonl, stable_hash, write_json, write_jsonl
from .head_training import file_hash


def family_training_mask(families, task_groups, held_family):
    families, task_groups = np.asarray(families), np.asarray(task_groups)
    if families.shape != task_groups.shape or families.ndim != 1 or held_family not in families:
        raise ValueError("Invalid family fold")
    held = families == held_family
    query_groups = set(task_groups[held])
    train = ~held & np.asarray([group not in query_groups for group in task_groups])
    if not train.any():
        raise ValueError("No independent training rows remain")
    return train, held


def training_panel(prepared):
    summary, contract = verify_competition_bundle(prepared)
    prepared = Path(prepared)
    source_prepared = Path(contract["prepared"])
    manifest_path = source_prepared / "manifest.json"
    if file_hash(manifest_path) != contract["source_manifest_sha256"]:
        raise ValueError("Frozen source manifest changed")
    source_manifest = json.loads(manifest_path.read_text())
    # Family attribution is scorer-side metadata only; no other columns are read.
    provenance_name = source_manifest.get("family_provenance_path", "enriched/provenance.parquet")
    provenance_path = (source_prepared / provenance_name).resolve()
    if not provenance_path.is_relative_to(source_prepared.resolve()):
        raise ValueError("Training provenance path escaped the source bundle")
    expected = source_manifest["artifacts"][provenance_name]
    if file_hash(provenance_path) != expected:
        raise ValueError("Training family provenance changed")
    provenance = pd.read_parquet(provenance_path, columns=["role", "family", "jsonl_line"])
    provenance = provenance.loc[provenance.role.eq("training")]
    if provenance.jsonl_line.duplicated().any():
        raise ValueError("Ambiguous family attribution")
    families = dict(zip(provenance.jsonl_line, provenance.family))
    split_path = Path(contract["root"]) / source_manifest["data_config"]["split"]
    if file_hash(split_path) != contract["split_sha256"]:
        raise ValueError("Frozen role split changed")
    split = json.loads(split_path.read_text())
    if any(split["family_roles"].get(family) != "training" for family in families.values()):
        raise ValueError("Family attribution escaped training roles")
    rows = [r for r in read_jsonl(prepared / "examples.jsonl") if r["role"] == "training"]
    target_rows = list(read_jsonl(prepared / "training_targets.jsonl"))
    if any(type(r["native_binary_value"]) not in {bool, int} or r["native_binary_value"] not in (0, 1) for r in target_rows):
        raise ValueError("Native training targets must be binary")
    targets = {r["row_key"]: int(r["native_binary_value"]) for r in target_rows}
    if len(targets) != len(target_rows):
        raise ValueError("Duplicate native training target")
    if len(rows) != len(targets) or {r["row_key"] for r in rows} != set(targets) or {r["source_line"] for r in rows} != set(families):
        raise ValueError("Training inputs, targets and family attribution do not align")
    for row in rows:
        row["validation_family"] = families[row["source_line"]]
    return summary, rows, targets, {"path": str(provenance_path), "sha256": expected,
                                   "fields_read": ["role", "family", "jsonl_line"]}


def prepare_feature_matrix(prepared, rows, out):
    prepared, out = Path(prepared), Path(out)
    systems = {r["system_key"]: r["recorded"] for r in read_jsonl(prepared / "systems.jsonl")}
    tasks = {r["task_key"]: r["recorded"] for r in read_jsonl(prepared / "tasks.jsonl")}
    task_cache, system_cache = {}, {}
    matrix = np.empty((len(rows), DIMS), dtype=np.float64)
    started = time.monotonic()
    for i, row in enumerate(rows):
        subject = systems[row["system_key"]]
        item = dict(tasks[row["task_key"]], benchmark_id=row["benchmark_id"])
        if row["task_key"] not in task_cache:
            task_cache[row["task_key"]] = task_features(item)
        if row["system_key"] not in system_cache:
            system_cache[row["system_key"]] = system_features(subject)
        matrix[i] = feature_vector(subject, item, task_vector=task_cache[row["task_key"]], system_vector=system_cache[row["system_key"]])
        if (i + 1) % 2000 == 0 or i + 1 == len(rows):
            write_json(out / "progress.json", {"phase": "feature_preparation", "rows_complete": i + 1,
                "rows_total": len(rows), "elapsed_seconds": time.monotonic() - started})
    np.save(out / "feature_matrix.npy", matrix, allow_pickle=False)
    write_json(out / "feature_contract.json", {"version": VERSION, "dimensions": DIMS, "dtype": "float64",
        "training_rows": len(rows), "row_order_sha256": stable_hash([r["row_key"] for r in rows]),
        "truncation_performed": False, "anonymous_ids_are_features": False,
        "encoder_sha256": file_hash(Path(__file__).with_name("conditional_features.py")),
        "matrix_sha256": file_hash(out / "feature_matrix.npy")})
    return matrix


def evaluate_conditional_controls(prepared, out, *, l2=100., prior_strength=5., offset_regularization=1.):
    prepared, out = Path(prepared), Path(out)
    summary, rows, targets, provenance = training_panel(prepared)
    out.mkdir(parents=True, exist_ok=False)
    matrix = prepare_feature_matrix(prepared, rows, out)
    labels = np.asarray([targets[r["row_key"]] for r in rows], dtype=float)
    scopes = np.asarray([r["scope_key"] for r in rows])
    family = np.asarray([r["validation_family"] for r in rows])
    groups = np.asarray([r["task_group"] for r in rows])
    predictions = {"additive": np.empty(len(rows)), "conditional": np.empty(len(rows))}
    fold_reports = []
    for held_family in sorted(set(family)):
        train, held = family_training_mask(family, groups, held_family)
        counts = Counter(scopes[train])
        weights = np.asarray([1. / counts[scope] for scope in scopes[train]])
        report = {"held_training_family": str(held_family), "training_rows": int(train.sum()),
                  "held_rows": int(held.sum()), "task_overlap_rows_excluded": int((~held & ~train).sum()),
                  "training_row_keys_sha256": stable_hash([r["row_key"] for r, selected in zip(rows, train) if selected]),
                  "held_row_keys_sha256": stable_hash([r["row_key"] for r, selected in zip(rows, held) if selected])}
        started = time.monotonic()
        for variant in predictions:
            state = fit_conditional_ridge(matrix[train], labels[train], weights, interactions=variant == "conditional", l2=l2)
            state["training_provenance"] = {**report, "prepared_manifest_sha256": file_hash(prepared / "manifest.json")}
            predictions[variant][held] = predict_matrix(state, matrix[held])
            destination = out / "fold_models" / str(held_family)
            destination.mkdir(parents=True, exist_ok=True)
            write_json(destination / (variant + ".json"), state)
        report["fit_seconds"] = time.monotonic() - started
        fold_reports.append(report)
        write_json(out / "progress.json", {"phase": "family_cross_validation", "completed_families": len(fold_reports),
                    "total_families": len(set(family)), "latest": report})
    write_jsonl(out / "out_of_family_predictions.jsonl", ({"row_key": r["row_key"], "validation_family": r["validation_family"],
        **{name: float(values[i]) for name, values in predictions.items()}} for i, r in enumerate(rows)))
    all_scope_errors = defaultdict(list)
    for name, values in {"constant": np.full(len(rows), .5), **predictions}.items():
        for scope in sorted(set(scopes)):
            selected = scopes == scope
            all_scope_errors[name].append(float(np.mean((values[selected] - labels[selected]) ** 2)))
    # Shared full-protocol scorer: identical episodes, query rows and fixed controls.
    episodes, excluded = full_protocol_episodes(prepared, rows, scopes, family)
    scored, curves, alc = curve_scores(episodes, labels, predictions, prior_strength=prior_strength,
                                       offset_regularization=offset_regularization)
    report = {"version": "conditional_training_family_analysis_v1", "source_policy": "provided_dataset_only",
        "feature_version": VERSION, "dimensions": DIMS, "l2_fixed_before_analysis": l2,
        "evidence_prior_strength": prior_strength, "offset_regularization": offset_regularization,
        "fitting_role": "training_only", "validation": "leave_training_family_out_and_exclude_repeated_query_tasks",
        "family_provenance": provenance, "training_rows": len(rows), "training_families": len(fold_reports),
        "visible_scopes": len(set(scopes)), "folds": fold_reports,
        "zero_budget_brier_all_held_rows_equal_scope": {k: float(np.mean(v)) for k, v in all_scope_errors.items()},
        "full_budget_eligible_scopes": len(episodes), "excluded_small_scopes": len(excluded),
        "per_budget_on_common_eligible_queries": curves, "brier_alc_on_common_eligible_scopes": alc,
        "model_selection_performed": False, "final_all_training_model_fitted": False,
        "original_development_outcomes_used": False, "monitor_outcomes_used": False, "sealed_roles_opened": False,
        "paid_calls": 0, "gpu_jobs": 0, "native_laya_inference": False,
        "limitations": ["Selection has not been frozen; these are fixed initial controls.",
            "Small retained scopes are excluded from full-budget ALC; zero-budget and ALC denominators may differ.",
            "Hashed text features do not establish named skills or pretrained semantic understanding.",
            "The 30k export contains one selected trial per original system/item."]}
    write_jsonl(out / "pair_budget_scores.jsonl", scored)
    write_json(out / "candidate_assessment.json", assess_conditional_controls(report))
    write_json(out / "summary.json", report)
    write_json(out / "progress.json", {"phase": "finished", "training_rows": len(rows), "families": len(fold_reports)})
    write_json(out / "manifest.json", {"artifacts": {str(p.relative_to(out)): file_hash(p) for p in sorted(out.rglob("*")) if p.is_file()}})
    return report
