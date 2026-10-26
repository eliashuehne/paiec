#!/usr/bin/env python3
"""Fit and score the two-way ability x difficulty rate model.

Benchmarks are held out whole, which is the competition's cold-start condition.
Everything fitted for a fold comes from the other benchmarks; the held
benchmark's difficulty is predicted from its own item texts and then refined
from revealed labels of any subject on it.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from paiec.competition_gate import assess
from paiec.competition_pairs import load_pair_matrix, refresh_manifest
from paiec.competition_scorer import (BUDGETS, EvidencePredictor, constant_baseline, draw_evidence,
                                      fixed_beta, oracle_curve, own_label_mean, score_draw, select_draw)
from paiec.competition_tracking import record
from paiec.feature_preparation import local_path, write_json
from paiec.rate_model import (ITEM_FEATURES, AbilityOnlyPredictor, HierarchicalPredictor,
                              ItemFeatureStore, ability_key, assign_folds, cell_feature_matrix,
                              difficulty_cv, fit_difficulty, fit_item_block, fit_two_way,
                              item_feature_frame, predict_difficulty)

KEYS = {"version", "pair_matrix", "folds", "seed", "max_pairs_per_draw", "min_items", "name_shrinkage",
        "benchmark_shrinkage", "ability_direction", "difficulty_alpha", "prior_strength",
        "benchmark_evidence_weight", "item_block", "items_per_cell", "threads", "item_block_iterations",
        "item_block_depth", "item_block_learning_rate", "regime"}
DEVICES = ("constant_0.5", "acquired_label_mean", "fixed_beta_k5", "pooled_plus_own",
           "ability_only", "ability_plus_difficulty", "hierarchical_rate")


def load_recipe(path):
    config = json.loads(Path(path).read_text())
    if set(config) != KEYS or config["version"] != "rate_model_v1":
        raise ValueError("Unknown rate-model recipe")
    if config["item_block"] not in {"catboost", "none"}:
        raise ValueError("The item block is either catboost or none")
    if config["ability_direction"] not in {"higher_is_better", "lower_is_better", "unordered"}:
        raise ValueError("Unknown label direction")
    if config["regime"] not in {"all", "shared_subject", "cold_subject"}:
        raise ValueError("Unknown subject-coverage regime")
    if not 2 <= config["folds"] <= 20:
        raise ValueError("Use between two and twenty benchmark folds")
    return config


def feature_store(matrix_directory):
    table = pq.read_table(Path(matrix_directory) / "items.parquet",
                          columns=["benchmark", "item_id", "content"]).to_pandas()
    if not len(table):
        raise ValueError("The pair matrix has no item texts")
    return ItemFeatureStore(item_feature_frame(table))


def restore_row(row):
    """JSON round-trips budget keys to strings; the summarizer needs integers."""
    for arm in row["arms"].values():
        arm["curve"] = {int(budget): value for budget, value in arm["curve"].items()}
    return row


def summarize_arms(fold_rows):
    """Average each arm over folds, equally weighted by fold and by pair."""
    arms = {}
    for name in DEVICES:
        values = [row["arms"][name]["brier_alc"] for row in fold_rows]
        curves = {budget: float(np.mean([row["arms"][name]["curve"][budget] for row in fold_rows]))
                  for budget in BUDGETS}
        aurocs = [row["arms"][name]["auroc"] for row in fold_rows
                  if row["arms"][name]["auroc"] is not None]
        arms[name] = {"brier_alc_mean": float(np.mean(values)),
                      "brier_alc_sd": float(np.std(values)),
                      "curve_mean": {str(budget): value for budget, value in curves.items()},
                      "auroc_mean": float(np.mean(aurocs)) if aurocs else None,
                      "pairs": sum(row["arms"][name]["pairs"] for row in fold_rows)}
    base = arms["acquired_label_mean"]["brier_alc_mean"]
    for values in arms.values():
        values["difference_from_acquired_mean"] = values["brier_alc_mean"] - base
    return arms


def cell_priors(train, two_way):
    return {cell["cell_key"]: float(two_way.mu
                                    + two_way.ability.get(ability_key(cell["subject"]), 0.0)
                                    + two_way.difficulty.get(cell["benchmark"], 0.0))
            for cell in train}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pipelines/rate_model_v1.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--no-tracking", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="Reuse folds already written to the output directory")
    args = parser.parse_args()
    config_path = (ROOT / args.config).resolve()
    if not config_path.is_relative_to(ROOT / "configs/pipelines"):
        raise ValueError("Use a repository pipeline configuration")
    recipe = load_recipe(config_path)
    out = local_path(ROOT, args.out)
    fold_log = out / "folds.jsonl"
    if fold_log.exists() and not args.resume:
        raise ValueError("Use a fresh experiment output directory")
    if out.exists() and not args.resume:
        raise ValueError("Use a fresh experiment output directory")
    matrix_directory = local_path(ROOT, recipe["pair_matrix"])
    matrix = load_pair_matrix(matrix_directory)
    cells, labels = matrix["cells"], matrix["labels"]
    out.mkdir(parents=True, exist_ok=True)
    # One fold at a time is written to disk, so an interruption costs the fold in
    # flight rather than every fold already scored. --resume skips those folds and
    # rebuilds the summary from the persisted rows, using the same recipe.
    done = {}
    if fold_log.is_file():
        for line in fold_log.read_text().splitlines():
            if line.strip():
                row = restore_row(json.loads(line))
                done[row["fold"]] = row
    store = feature_store(matrix_directory)
    features = cell_feature_matrix(cells, labels, store)

    folds = assign_folds(cells, recipe["folds"], recipe["seed"])
    for fold in range(recipe["folds"]):
        if fold in done:
            print(f"fold={fold} already scored, resuming", flush=True)
            continue
        started = time.monotonic()
        held_benchmarks = sorted({benchmark for benchmark, value in folds.items() if value == fold})
        train = [cell for cell in cells if cell["benchmark"] not in held_benchmarks
                 and cell["direction"] == recipe["ability_direction"]]
        if len({cell["benchmark"] for cell in train}) < 3:
            raise ValueError("A fold needs at least three training benchmarks")
        two_way = fit_two_way(train, name_shrinkage=recipe["name_shrinkage"],
                              benchmark_shrinkage=recipe["benchmark_shrinkage"],
                              direction_scope=recipe["ability_direction"])
        difficulty_state = fit_difficulty(train, two_way.difficulty, features,
                                          alpha=recipe["difficulty_alpha"])
        held = [cell for cell in cells if cell["benchmark"] in set(held_benchmarks)
                and cell["direction"] == recipe["ability_direction"]]
        cell_difficulty = {cell["cell_key"]: predict_difficulty(difficulty_state,
                                                               features[cell["cell_key"]])
                           for cell in held}
        priors = cell_priors(train, two_way)
        item_model = None
        if recipe["item_block"] == "catboost":
            item_model = fit_item_block(train, labels, priors, store,
                                        items_per_cell=recipe["items_per_cell"], seed=recipe["seed"] + fold,
                                        threads=recipe["threads"],
                                        iterations=recipe["item_block_iterations"],
                                        depth=recipe["item_block_depth"],
                                        learning_rate=recipe["item_block_learning_rate"])
        ability = AbilityOnlyPredictor(two_way, prior_strength=recipe["prior_strength"],
                                       benchmark_weight=recipe["benchmark_evidence_weight"])
        ability_difficulty = HierarchicalPredictor(
            two_way, difficulty_state, cell_difficulty, prior_strength=recipe["prior_strength"],
            benchmark_weight=recipe["benchmark_evidence_weight"])
        ability_difficulty.name = "ability_plus_difficulty"
        hierarchical = HierarchicalPredictor(
            two_way, difficulty_state, cell_difficulty, store=store, item_model=item_model,
            prior_strength=recipe["prior_strength"],
            benchmark_weight=recipe["benchmark_evidence_weight"])
        devices = {"constant_0.5": constant_baseline(),
                   "acquired_label_mean": own_label_mean(),
                   "fixed_beta_k5": fixed_beta(5.0),
                   "pooled_plus_own": EvidencePredictor(
                       "pooled_plus_own", prior_strength=recipe["prior_strength"], own_weight=1.0,
                       other_weight=recipe["benchmark_evidence_weight"], prior="half"),
                   "ability_only": ability, "ability_plus_difficulty": ability_difficulty,
                   "hierarchical_rate": hierarchical}
        draw = select_draw(cells, labels, seed=recipe["seed"] + fold,
                           max_pairs=recipe["max_pairs_per_draw"], regime=recipe["regime"],
                           min_items=recipe["min_items"], benchmarks=held_benchmarks,
                           directions=[recipe["ability_direction"]])
        cache = draw_evidence(draw)
        scored = {name: score_draw(draw, device, evidence_cache=cache)
                  for name, device in devices.items()}
        oracle = oracle_curve(draw)
        known = sum(1 for pair in draw.pairs if pair.subject_key in two_way.ability)
        row = {"fold": fold, "regime": recipe["regime"], "held_benchmarks": held_benchmarks,
               "pairs": len(draw.pairs),
               "training_cells": len(train), "training_benchmarks": len({c["benchmark"] for c in train}),
               "oracle_brier_alc": oracle["brier_alc"],
               "two_way": two_way.as_report(),
               "difficulty_regression": {"rows": difficulty_state["rows"],
                                         "alpha": difficulty_state["alpha"],
                                         "coefficients": {name: float(value) for name, value in
                                                          zip(ITEM_FEATURES,
                                                              difficulty_state["coefficients"])}},
               "difficulty_cv": difficulty_cv(train, two_way.difficulty, features,
                                              alpha=recipe["difficulty_alpha"], seed=recipe["seed"]),
               "pairs_with_known_ability": known,
               "pairs_with_known_ability_share": known / len(draw.pairs),
               "arms": {name: {"brier_alc": arm["brier_alc"], "auroc": arm["auroc"],
                               "pairs": arm["pairs"],
                               "curve": {str(budget): value for budget, value in arm["curve"].items()}}
                        for name, arm in scored.items()}}
        with fold_log.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        done[fold] = restore_row(row)
        print(f"fold={fold} pairs={len(draw.pairs)} known_ability={known}/{len(draw.pairs)} "
              + " ".join(f"{name}={arm['brier_alc']:.4f}" for name, arm in scored.items())
              + f" {time.monotonic() - started:.0f}s", flush=True)

    fold_rows = [done[fold] for fold in sorted(done)]
    if [row["fold"] for row in fold_rows] != list(range(recipe["folds"])):
        raise ValueError("Every fold must be scored before the summary is written")
    arms = summarize_arms(fold_rows)
    summary = {"version": "rate_model_v1", "pair_matrix_sha256": matrix["manifest_sha256"],
               "recipe": recipe, "folds": fold_rows, "arms": {recipe["regime"]: arms},
               "budgets": list(BUDGETS), "labels_reinterpreted": False,
               "sealed_roles_opened": False, "gpu_jobs": 0}
    summary["gate"] = assess(summary, regime=recipe["regime"])
    write_json(out / "resolved_config.json", recipe)
    write_json(out / "summary.json", summary)
    refresh_manifest(out)
    if not args.no_tracking:
        record(ROOT, out, run_name="rate_model_v1",
               tags={"source_policy": "provided_public_corpus_only", "fitting_role": "benchmark_folds",
                     "gpu_jobs": "0", "sealed_roles_opened": "false",
                     "verdict": summary["gate"]["verdict"]},
               params={key: recipe[key] for key in KEYS - {"version"}},
               metrics={f"alc_{name}": values["brier_alc_mean"] for name, values in arms.items()},
               artifacts=[ROOT / "src/paiec/rate_model.py", ROOT / "src/paiec/competition_scorer.py",
                          ROOT / "src/paiec/competition_gate.py", Path(__file__).resolve()])
        refresh_manifest(out)
    print(json.dumps({"arms": {name: round(values["brier_alc_mean"], 4)
                               for name, values in arms.items()},
                      "gate": summary["gate"]["verdict"],
                      "failed": summary["gate"]["failed"]}, indent=2))


if __name__ == "__main__":
    main()
