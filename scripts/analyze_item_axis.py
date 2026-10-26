#!/usr/bin/env python3
"""Measure whether item difficulty is shared across systems or model specific.

This decides where item-level modelling should invest. It fits nothing that can
be shipped and opens no sealed role: the Rasch fits describe the public corpus,
and every oracle values a diagnostic bound rather than a predictor.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from paiec.competition_pairs import load_pair_matrix, refresh_manifest
from paiec.competition_scorer import draw_evidence, oracle_curve, select_draw
from paiec.competition_tracking import record
from paiec.feature_preparation import local_path, write_json
from paiec.item_axis import item_frames, oracle_ceilings, summarise
from paiec.rate_model import assign_folds

KEYS = {"version", "pair_matrix", "direction", "min_items", "folds", "seed", "max_pairs_per_draw",
        "minimum_item_observations", "prior_strength", "transfer_auc_threshold",
        "reliability_threshold"}


def load_recipe(path):
    config = json.loads(Path(path).read_text())
    if set(config) != KEYS or config["version"] != "item_axis_v1":
        raise ValueError("Unknown item-axis recipe")
    if config["direction"] not in {"higher_is_better", "lower_is_better", "unordered"}:
        raise ValueError("Unknown label direction")
    if not 2 <= config["folds"] <= 20:
        raise ValueError("Use between two and twenty benchmark folds")
    if not 0 < config["transfer_auc_threshold"] < 1 or not 0 < config["reliability_threshold"] < 1:
        raise ValueError("Thresholds are correlations or AUCs in (0, 1)")
    return config


def decide(axis, recipe):
    """Turn the two measured statistics into the funding decision they imply."""
    pooled_auc = axis["transfer"]["pooled_auc"]
    reliability = axis["reliability"]["pooled_pearson"]
    shared = (pooled_auc is not None and pooled_auc >= recipe["transfer_auc_threshold"]
              and reliability is not None and reliability >= recipe["reliability_threshold"])
    if shared:
        stance = "item_difficulty_is_shared"
    elif pooled_auc is not None and pooled_auc >= recipe["transfer_auc_threshold"]:
        stance = "transfer_without_stable_facility"
    else:
        stance = "item_difficulty_is_model_specific"
    return {"stance": stance, "pooled_transfer_auc": pooled_auc, "pooled_reliability": reliability,
            "transfer_auc_threshold": recipe["transfer_auc_threshold"],
            "reliability_threshold": recipe["reliability_threshold"],
            "fund_rasch_on_labels": stance != "item_difficulty_is_model_specific",
            "fund_similarity_retrieval_first": True,
            "note": "A model-specific verdict still leaves retrieval of revealed labels open; "
                    "it removes static item descriptions as a route to ordering."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pipelines/item_axis_v1.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--no-tracking", action="store_true")
    args = parser.parse_args()
    config_path = (ROOT / args.config).resolve()
    if not config_path.is_relative_to(ROOT / "configs/pipelines"):
        raise ValueError("Use a repository pipeline configuration")
    recipe = load_recipe(config_path)
    out = local_path(ROOT, args.out)
    if out.exists():
        raise ValueError("Use a fresh experiment output directory")
    matrix = load_pair_matrix(local_path(ROOT, recipe["pair_matrix"]))
    cells, labels = matrix["cells"], matrix["labels"]

    frames = item_frames(cells, labels, direction=recipe["direction"], min_items=recipe["min_items"])
    if not frames:
        raise ValueError("No benchmark cleared the item-level admission rule")
    axis = summarise(frames, seed=recipe["seed"], minimum=recipe["minimum_item_observations"])

    folds = assign_folds(cells, recipe["folds"], recipe["seed"])
    ceiling_rows, rate_alc = [], []
    for fold in range(recipe["folds"]):
        held = sorted({benchmark for benchmark, value in folds.items() if value == fold})
        draw = select_draw(cells, labels, seed=recipe["seed"] + fold,
                           max_pairs=recipe["max_pairs_per_draw"], regime="all",
                           min_items=recipe["min_items"], benchmarks=held,
                           directions=[recipe["direction"]])
        draw_evidence(draw)
        ceiling = oracle_ceilings(draw, prior_strength=recipe["prior_strength"])
        rate = oracle_curve(draw)
        ceiling_rows.append({"fold": fold, "held_benchmarks": held, "pairs": len(draw.pairs),
                             "own_item_alc": ceiling["own_item_alc"],
                             "sibling_facility_alc": ceiling["sibling_facility_alc"],
                             "pair_rate_alc": rate["brier_alc"],
                             "covered_item_share": {str(row["budget"]): row["covered_item_share"]
                                                    for row in ceiling["rows"]},
                             "budgets": ceiling["rows"]})
        rate_alc.append(rate["brier_alc"])
        print(f"fold={fold} pairs={len(draw.pairs)} own_item_alc={ceiling['own_item_alc']:.4f} "
              f"sibling_facility_alc={ceiling['sibling_facility_alc']:.4f} "
              f"pair_rate_alc={rate['brier_alc']:.4f} "
              f"coverage@31={ceiling['rows'][-1]['covered_item_share']:.2f}", flush=True)

    report = {"version": "item_axis_v1", "pair_matrix_sha256": matrix["manifest_sha256"],
              "recipe": recipe, "benchmarks": sorted(frames),
              "axis": axis,
              "decision": decide(axis, recipe),
              "ceilings": {"folds": ceiling_rows,
                           "own_item_alc_mean": float(np.mean([row["own_item_alc"] for row in ceiling_rows])),
                           "sibling_facility_alc_mean": float(np.mean([row["sibling_facility_alc"]
                                                                       for row in ceiling_rows])),
                           "pair_rate_alc_mean": float(np.mean(rate_alc))},
              "labels_reinterpreted": False, "sealed_roles_opened": False, "gpu_jobs": 0}
    out.mkdir(parents=True)
    write_json(out / "resolved_config.json", recipe)
    write_json(out / "report.json", report)
    refresh_manifest(out)
    if not args.no_tracking:
        record(ROOT, out, run_name="item_axis_v1",
               tags={"source_policy": "provided_public_corpus_only", "gpu_jobs": "0",
                     "sealed_roles_opened": "false", "stance": report["decision"]["stance"]},
               params={key: recipe[key] for key in KEYS - {"version"}},
               metrics={"pooled_transfer_auc": report["decision"]["pooled_transfer_auc"],
                        "pooled_reliability": report["decision"]["pooled_reliability"],
                        "own_item_alc": report["ceilings"]["own_item_alc_mean"],
                        "sibling_facility_alc": report["ceilings"]["sibling_facility_alc_mean"],
                        "pair_rate_alc": report["ceilings"]["pair_rate_alc_mean"]},
               artifacts=[ROOT / "src/paiec/item_axis.py", Path(__file__).resolve()])
        refresh_manifest(out)
    print(json.dumps({"stance": report["decision"]["stance"],
                      "pooled_transfer_auc": report["decision"]["pooled_transfer_auc"],
                      "pooled_reliability": report["decision"]["pooled_reliability"],
                      "own_item_alc": report["ceilings"]["own_item_alc_mean"],
                      "sibling_facility_alc": report["ceilings"]["sibling_facility_alc_mean"],
                      "pair_rate_alc": report["ceilings"]["pair_rate_alc_mean"]}, indent=2))


if __name__ == "__main__":
    main()
