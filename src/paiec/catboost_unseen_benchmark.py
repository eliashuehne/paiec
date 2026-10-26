"""Competition-shaped evaluation: new benchmark, model name already seen.

The panel's leave-one-family-out protocol also removes every system of the held
family, because no system in the panel appears under more than one benchmark.
The competition instead evaluates a benchmark-level cold start whose subject
coverage may differ between pools. This module measures the base predictor in
that shape:

* fitting stays on the panel only, with exactly the panel run's recipe;
* evaluation rows come from unused public benchmarks, one hashed response per
  subject-item, and are never fitted on;
* an evaluation row is eligible for a fold only when its supplied model name
  (or its full supplied identity) appears in that fold's training systems.

Identity is eligibility metadata only. It never enters a feature, and the
anonymous benchmark identifier never enters the matrix either.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, model_validator

from .catboost_analysis import (build_feature_frames, checked_variants, cold_start_metrics, profile_source,
                                fit_fold_models, probabilities)
from .catboost_features import feature_names, feature_record
from .conditional_analysis import family_training_mask, training_panel
from .competition_fields import SUBJECT_FIELDS
from .feature_preparation import local_path, stable_hash, write_json, write_jsonl
from .grouped_data import checked_artifacts
from .head_data import task_key
from .head_training import file_hash
from .training_data_expansion import (AddedBenchmark, frozen_roles, resolve_data_root,
                                      sample_added_rows)

_NAME = re.compile(r"[^a-z0-9]+")
METRIC_KEYS = ("equal_scope_brier", "pooled_brier", "logloss", "ece", "mean_within_scope_auroc")


def normalize_name(value):
    """Case- and punctuation-insensitive form of a supplied model name."""
    return _NAME.sub(" ", str(value).strip().casefold()).strip()


def subject_identity(subject):
    """The exact supplied identity: model, harness, version and effort."""
    return tuple(subject[field] for field in SUBJECT_FIELDS)


class UnseenBenchmarkConfig(BaseModel):
    """Predeclared recipe for the competition-shaped evaluation."""

    model_config = ConfigDict(extra="forbid", strict=True)
    version: Literal["catboost_unseen_benchmark_v1"] = "catboost_unseen_benchmark_v1"
    panel_run: str
    data_root: str
    seed: int = 20261024
    min_pair_items: int = 20
    rows_per_system: int = 100
    benchmark_cap: int = 6000
    max_panel_group_overlap: float = 0.0

    @model_validator(mode="after")
    def coherent(self):
        if self.min_pair_items < 2 or self.rows_per_system <= 0 or self.benchmark_cap <= 0:
            raise ValueError("Use positive row and item limits")
        if type(self.seed) is not int:
            raise ValueError("Seed must be an integer")
        if not 0 <= self.max_panel_group_overlap <= 1:
            raise ValueError("Overlap tolerance must lie in [0, 1]")
        return self


def _metadata(directory):
    """Benchmark metadata that decides whether a directory can be evaluated."""
    metadata = pd.read_parquet(directory / "benchmarks.parquet",
                               columns=["modality", "response_scale", "response_type", "granularity"])
    if len(metadata) != 1:
        raise ValueError("expected exactly one benchmark record")
    row = metadata.iloc[0]
    scale = json.loads(row["response_scale"])
    return {"text_only": list(row["modality"]) == ["text"],
            "higher_is_better": scale.get("direction") == "higher_is_better",
            "binary_item": row["response_type"] == "binary" and row["granularity"] == "item"}


def select_benchmarks(data_root, roles, panel_model_names, panel_task_groups, config):
    """Apply the predeclared admission filters to every unused public benchmark.

    Nothing here reads an outcome. A directory is eligible when it is unused,
    outside every sealed role, text-only, success-as-higher, binary item-level,
    shares no more than a declared fraction of its task content with the panel,
    and contains at least one subject whose supplied model name the panel also
    contains.
    """
    data_root = Path(data_root)
    manifest = json.loads((data_root / "corpus_manifest.json").read_text())
    admitted, decisions, seen = [], [], set()
    for entry in manifest["eligible_dirs"]:
        name = entry.split("/")[0]
        if name in seen:
            continue
        seen.add(name)
        directory = data_root / name / "formatted_tables"
        decision = {"benchmark": name}
        role = roles.get(name)
        if role is not None and role != "training":
            decisions.append({**decision, "eligible": False, "reason": f"sealed_role:{role}"})
            continue
        if not (directory / "benchmarks.parquet").is_file():
            decisions.append({**decision, "eligible": False, "reason": "missing_tables"})
            continue
        try:
            metadata = _metadata(directory)
        except (ValueError, KeyError) as error:
            decisions.append({**decision, "eligible": False, "reason": f"metadata:{error}"})
            continue
        rejection = next((reason for key, reason in
                          [("text_only", "not_text_only"), ("higher_is_better", "direction_not_success"),
                           ("binary_item", "not_binary_item")] if not metadata[key]), None)
        if rejection:
            decisions.append({**decision, "eligible": False, "reason": rejection})
            continue
        subjects = pd.read_parquet(directory / "subjects.parquet", columns=["normalized_name"])
        names = [normalize_name(value) for value in subjects.normalized_name.fillna("")]
        names = [value for value in names if value]
        decision["subjects"] = int(len(subjects))
        decision["matched_subjects"] = int(sum(1 for value in names if value in panel_model_names))
        if not decision["matched_subjects"]:
            decisions.append({**decision, "eligible": False, "reason": "no_shared_model_name"})
            continue
        items = pd.read_parquet(directory / "items.parquet", columns=["content"])
        # The panel keys task content by a stable hash of a normalized text slug.
        contents = {stable_hash(task_key(content)) for content in items.content.fillna("")}
        contents.discard(stable_hash(task_key("")))
        overlap = len(contents & panel_task_groups) / len(contents) if contents else 1.
        decision["task_group_overlap"] = float(overlap)
        if overlap > config.max_panel_group_overlap:
            decisions.append({**decision, "eligible": False, "reason": "panel_content_overlap"})
            continue
        decisions.append({**decision, "eligible": True, "reason": "admitted"})
        admitted.append(AddedBenchmark(name=name, cap=config.benchmark_cap,
                                       domain="unused_public_benchmark"))
    return admitted, decisions


def build_external_rows(data_root, benchmarks, panel_task_groups, config):
    """One hashed response per subject-item, capped, with repeated content dropped."""
    rows = sample_added_rows(data_root, benchmarks, panel_task_groups,
                             rows_per_system=config.rows_per_system,
                             min_pair_items=config.min_pair_items, seed=config.seed)
    if not rows:
        raise ValueError("No eligible rows were sampled from the unused benchmarks")
    repeated = [row for row in rows if row["task_group"] in panel_task_groups]
    if repeated:
        raise ValueError("Sampled row repeats panel task content")
    return rows


def build_external_frame(rows, columns, profile_facts=None, permutation_seed=None):
    """Feature frame for evaluation rows, asserted onto the fitted contract.

    Under the permutation control the evaluation facts are scrambled between
    subjects the same way the fitting facts are, so a surviving difference comes
    from the extra columns rather than from any system-to-fact information.
    """
    records = {}
    if profile_facts is not None:
        for row in rows:
            records.setdefault(row["subject_id"], profile_facts.record(row["subject"]))
        if permutation_seed is not None:
            subjects = sorted(records)
            shuffled = np.random.default_rng(int(permutation_seed)).permutation(len(subjects))
            records = {key: records[subjects[index]] for key, index in zip(subjects, shuffled)}
    records = [feature_record(row["subject"], row["item"], records.get(row.get("subject_id")))
               for row in rows]
    frame = pd.DataFrame([{column: record[column] for column in columns} for record in records],
                         columns=columns)
    if list(frame.columns) != list(columns):
        raise ValueError("External feature frame violates the fitted column contract")
    return frame


def load_external_evaluation(root, data_root, panel_task_groups, panel_model_names, settings):
    """Select, admit and sample the unseen-benchmark evaluation rows once."""
    data_root = resolve_data_root(root, data_root)
    admitted, decisions = select_benchmarks(data_root, frozen_roles(root),
                                            panel_model_names, panel_task_groups, settings)
    if not admitted:
        raise ValueError("No unused benchmark passed the admission filters")
    rows = build_external_rows(data_root, admitted, panel_task_groups, settings)
    return {"admitted": admitted, "decisions": decisions, "rows": rows}


def _fold_identity(rows, systems, train):
    names, identities = set(), set()
    for index in np.flatnonzero(train):
        subject = systems[rows[index]["system_key"]]
        name = normalize_name(subject["normalized_name"])
        if name:
            names.add(name)
        identities.add(subject_identity(subject))
    return names, identities


def _subset_metrics(labels, values, scopes):
    return {key: cold_start_metrics(labels, values, scopes)[key] for key in METRIC_KEYS}


def _oracle_equal_scope(labels, scopes, mask):
    """Diagnostic bound: predict each scope's own observed rate.

    This uses the evaluation labels themselves, so it is an upper reference for
    how much a per-scope level correction could be worth. It is never scored as
    a predictor.
    """
    values = []
    for scope in sorted(set(scopes[mask].tolist())):
        selected = mask & (scopes == scope)
        rate = float(labels[selected].mean())
        values.append(float(np.mean((rate - labels[selected]) ** 2)))
    return float(np.mean(values)) if values else None


def evaluate_unseen_benchmarks(root, config, out):
    """Score each panel fold model on unseen benchmarks with seen model names."""
    root, out = Path(root).resolve(), Path(out).resolve()
    config = UnseenBenchmarkConfig.model_validate(config.model_dump())
    panel_run = local_path(root, config.panel_run)
    data_root = resolve_data_root(root, config.data_root)
    if out.exists():
        raise ValueError("Use a fresh evaluation output directory")
    checked_artifacts(panel_run, json.loads((panel_run / "manifest.json").read_text())["artifacts"])
    recipe = json.loads((panel_run / "resolved_config.json").read_text())
    summary = json.loads((panel_run / "summary.json").read_text())
    policy = summary.get("source_policy")
    if policy not in {"provided_dataset_only", "provided_plus_research_profiles"}:
        raise ValueError("Evaluation requires a declared panel source policy")
    profile_facts = profile_source(recipe) if policy != "provided_dataset_only" else None
    if (profile_facts is None) != (policy == "provided_dataset_only"):
        raise ValueError("Panel source policy does not match its declared profile catalog")
    variants, ablations_run = checked_variants(recipe)
    primary = variants["catboost"]
    columns = feature_names(primary)
    prepared = local_path(root, recipe["prepared"])
    _, rows, targets, _ = training_panel(prepared)
    frames, systems = build_feature_frames(prepared, rows, variants, profile_facts,
                                           recipe.get("profile_permutation_seed"))
    labels = np.asarray([targets[row["row_key"]] for row in rows], dtype=float)
    scopes = np.asarray([row["scope_key"] for row in rows])
    family = np.asarray([row["validation_family"] for row in rows])
    groups = np.asarray([row["task_group"] for row in rows])
    panel_task_groups = set(groups.tolist())
    panel_model_names = {normalize_name(systems[row["system_key"]]["normalized_name"]) for row in rows}
    panel_model_names.discard("")
    external_set = load_external_evaluation(root, data_root, panel_task_groups,
                                            panel_model_names, config)
    admitted, decisions, external = (external_set["admitted"], external_set["decisions"],
                                     external_set["rows"])
    frame = build_external_frame(external, columns, profile_facts, recipe.get("profile_permutation_seed"))
    if list(frame.columns) != list(frames["catboost"].columns):
        raise ValueError("External frame does not match the fitted feature contract")
    external_labels = np.asarray([row["label"] for row in external], dtype=float)
    external_scopes = np.asarray([row["scope_key"] for row in external], dtype=object)
    external_benchmarks = np.asarray([row["benchmark"] for row in external], dtype=object)
    external_names = np.asarray([normalize_name(row["subject"]["normalized_name"]) for row in external],
                                dtype=object)
    external_identity = [subject_identity(row["subject"]) for row in external]
    out.mkdir(parents=True, exist_ok=False)
    predictions, exact_flags, fold_reports = defaultdict(list), defaultdict(bool), []
    panel_predictions = np.empty(len(rows), dtype=float)
    for held_family in sorted(set(family.tolist())):
        train, held = family_training_mask(family, groups, held_family)
        selection, models, weights = fit_fold_models(frames, labels, scopes, family, groups, train,
                                                     recipe=recipe, variants=variants,
                                                     seed=int(recipe["seed"]), threads=int(recipe["threads"]))
        held_prediction = probabilities(models["catboost"], frames["catboost"].loc[held])
        panel_predictions[held] = held_prediction
        panel_cold = cold_start_metrics(labels[held], held_prediction, scopes[held])["equal_scope_brier"]
        names, identities = _fold_identity(rows, systems, train)
        eligible = [index for index in range(len(external))
                    if external_names[index]
                    and (external_names[index] in names or external_identity[index] in identities)]
        exact = [index for index in eligible if external_identity[index] in identities]
        for index in exact:
            exact_flags[index] = True
        if eligible:
            values = probabilities(models["catboost"], frame.iloc[eligible])
            for index, value in zip(eligible, values):
                predictions[index].append(float(value))
            metrics = _subset_metrics(external_labels[eligible], values, external_scopes[eligible])
        else:
            metrics = None
        prior = float(np.average(labels[train], weights=weights))
        fold_reports.append({
            "held_training_family": str(held_family),
            "selected_config": selection["selected_config"], "iterations": selection["iterations"],
            "panel_reproduced_equal_scope_brier": panel_cold,
            "panel_training_base_rate": prior,
            "eligible_rows": len(eligible), "exact_subject_rows": len(exact),
            "eligible_subjects": int(len(set(external_scopes[eligible].tolist()))),
            "external_metrics": metrics,
            "panel_base_rate_equal_scope_brier": (
                cold_start_metrics(external_labels[eligible], np.full(len(eligible), prior),
                                   external_scopes[eligible])["equal_scope_brier"] if eligible else None),
        })
    recorded = float(summary["zero_budget_brier_all_held_rows_equal_scope"]["catboost"])
    # The recorded score averages every scope once; averaging the seven fold
    # scores would weight a 120-scope family like a 1-scope family.
    recomputed_panel = cold_start_metrics(labels, panel_predictions, scopes)["equal_scope_brier"]
    if abs(recomputed_panel - recorded) > 1e-9:
        raise ValueError(f"Recomputed panel score {recomputed_panel} does not reproduce {recorded}")
    scored = sorted(predictions)
    if not scored:
        raise ValueError("No evaluation row matched any fold's training systems")
    averaged = np.asarray([float(np.mean(predictions[index])) for index in scored])
    scored_labels, scored_scopes = external_labels[scored], external_scopes[scored]
    scored_benchmarks = external_benchmarks[scored]
    exact_mask = np.asarray([exact_flags[index] for index in scored])
    strata = {}
    for name, mask in [("exact_subject", exact_mask), ("model_only", ~exact_mask)]:
        strata[name] = ({"rows": int(mask.sum()), "scopes": int(len(set(scored_scopes[mask].tolist()))),
                         **_subset_metrics(scored_labels[mask], averaged[mask], scored_scopes[mask])}
                        if mask.any() else {"rows": 0, "scopes": 0})
    per_benchmark = []
    for name in sorted(set(scored_benchmarks.tolist())):
        mask = scored_benchmarks == name
        metrics = _subset_metrics(scored_labels[mask], averaged[mask], scored_scopes[mask])
        per_benchmark.append({"benchmark": name, "rows": int(mask.sum()),
                              "scopes": int(len(set(scored_scopes[mask].tolist()))),
                              "observed_rate": float(scored_labels[mask].mean()),
                              "mean_prediction": float(averaged[mask].mean()),
                              "equal_scope_brier": metrics["equal_scope_brier"],
                              "observed_rate_oracle_equal_scope_brier": _oracle_equal_scope(
                                  scored_labels, scored_scopes, mask)})
    folds_with_external = [row for row in fold_reports if row["external_metrics"] is not None]
    report = {
        "version": "catboost_unseen_benchmark_v1",
        "source_policy": policy,
        "profile_source": ({**profile_facts.metadata, "usage": "research_only"} if profile_facts else None),
        "design": {"fitting": "panel only, exactly the panel run's recipe",
                   "evaluation": "unused public benchmarks, one hashed response per subject-item",
                   "eligibility": "fold training systems must contain the row's supplied model name or full supplied identity",
                   "identity_role": "eligibility metadata only; never a feature",
                   "benchmark_id_role": "never a feature"},
        "panel_run": {"path": str(panel_run), "recipe_version": recipe["version"],
                      "feature_groups": list(primary), "ablations_run": ablations_run,
                      "manifest_sha256": file_hash(panel_run / "manifest.json"),
                      "recorded_cold_start_equal_scope_brier": recorded,
                      "recomputed_cold_start_equal_scope_brier": recomputed_panel,
                      "reproduction_tolerance": 1e-9},
        "corpus_manifest_sha256": file_hash(data_root / "corpus_manifest.json"),
        "selection": {"criteria": {"text_only": True, "higher_is_better": True, "binary_item": True,
                                   "min_pair_items": config.min_pair_items,
                                   "rows_per_system": config.rows_per_system,
                                   "benchmark_cap": config.benchmark_cap,
                                   "max_panel_group_overlap": config.max_panel_group_overlap,
                                   "seed": config.seed},
                      "admitted_benchmarks": [benchmark.name for benchmark in admitted],
                      "decisions_recorded_in": "benchmark_decisions.jsonl"},
        "rows": {"sampled": len(external), "scored": len(scored),
                 "subjects": int(len(set(external_scopes.tolist()))),
                 "benchmarks": int(len(set(external_benchmarks.tolist()))),
                 "label_mean": float(external_labels.mean()),
                 "exact_subject_evaluations": int(exact_mask.sum())},
        "folds": fold_reports,
        "pooled_row_metrics": _subset_metrics(scored_labels, averaged, scored_scopes),
        "primary_equal_scope_brier": float(np.mean([row["external_metrics"]["equal_scope_brier"]
                                                    for row in folds_with_external])),
        "baselines": {"constant_0.5": .25,
                      "mean_panel_base_rate_equal_scope_brier":
                          float(np.mean([row["panel_base_rate_equal_scope_brier"]
                                         for row in folds_with_external])),
                      "external_observed_rate": "diagnostic oracle only, never a predictor"},
        "strata": strata,
        "per_benchmark": per_benchmark,
        "integrity": {"external_profiles_used": profile_facts is not None,
                      "external_rows_used_for_fitting": 0,
                      "external_rows_repeating_panel_content": 0,
                      "sealed_roles_opened": False,
                      "original_development_outcomes_used": False,
                      "monitor_outcomes_used": False,
                      "calibration_fitted": False,
                      "panel_score_reproduced": True,
                      "hyperparameters_reselected": False},
        "limitations": [
            "Rows come from unused public benchmarks and become development evidence once scored.",
            "Model-name matching is not subject identity: harness and version may differ.",
            "Only benchmarks that passed the admission filters are represented.",
            "Cold-start Brier only; these rows have no acquisition plan, so Brier ALC is not reported.",
            "One hashed response per subject-item and a per-subject cap keep the set bounded.",
            "Multi-fold rows average their fold predictions equally; per-fold rows are also reported.",
            *(["Panel models use external research profile facts, so this evaluation is a research diagnostic."]
              if profile_facts else []),
        ],
    }
    write_jsonl(out / "row_predictions.jsonl", (
        {"benchmark": external[index]["benchmark"], "scope_key": external[index]["scope_key"],
         "subject_id": external[index]["subject_id"], "item_id": external[index]["item_id"],
         "label": int(external_labels[index]), "qualifying_folds": len(predictions[index]),
         "exact_subject_identity": bool(exact_flags[index]),
         "prediction": float(np.mean(predictions[index]))} for index in scored))
    write_jsonl(out / "benchmark_decisions.jsonl", decisions)
    write_json(out / "summary.json", report)
    write_json(out / "manifest.json", {"artifacts": {
        str(path.relative_to(out)): file_hash(path)
        for path in sorted(out.rglob("*")) if path.is_file() and path.name != "manifest.json"}})
    return report
