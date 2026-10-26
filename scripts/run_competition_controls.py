#!/usr/bin/env python3
"""Score fixed controls on the competition-shaped pair matrix.

No model is fitted here. The run establishes how much of the metric a constant,
the revealed-label mean, the published fixed beta prior, and same-benchmark
evidence already explain before any learner is graded.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from paiec.competition_pairs import load_pair_matrix, refresh_manifest
from paiec.competition_scorer import (BUDGETS, EvidencePredictor, constant_baseline, draw_evidence,
                                      fixed_beta, oracle_curve, own_label_mean, score_draw, select_draw)
from paiec.competition_tracking import record
from paiec.feature_preparation import local_path, write_json

KEYS = {"version", "pair_matrix", "seed", "draws", "max_pairs_per_draw", "min_items",
        "prior_strength", "benchmark_evidence_weight", "regimes", "directions"}
DEVICES = ("constant_0.5", "acquired_label_mean", "fixed_beta_k5", "benchmark_pooled_rate",
           "pooled_plus_own")


def load_recipe(path):
    config = json.loads(Path(path).read_text())
    if set(config) != KEYS or config["version"] != "competition_controls_v1":
        raise ValueError("Unknown competition control recipe")
    if not 1 <= config["draws"] <= 20:
        raise ValueError("Use between one and twenty draws")
    if not 0 <= config["prior_strength"] <= 50 or not 0 <= config["benchmark_evidence_weight"] <= 1:
        raise ValueError("Control strengths are out of range")
    if not config["regimes"]:
        raise ValueError("Declare at least one subject-coverage regime")
    if (not config["directions"]
            or set(config["directions"]) - {"higher_is_better", "lower_is_better", "unordered"}):
        raise ValueError("Declare which recorded label directions are scored")
    return config


def arms(prior_strength, benchmark_weight):
    return {
        "constant_0.5": constant_baseline(),
        "acquired_label_mean": own_label_mean(),
        "fixed_beta_k5": fixed_beta(5.0),
        "benchmark_pooled_rate": EvidencePredictor(
            "benchmark_pooled_rate", prior_strength=0.0, own_weight=0.0, other_weight=1.0, prior="half"),
        "pooled_plus_own": EvidencePredictor(
            "pooled_plus_own", prior_strength=prior_strength, own_weight=1.0,
            other_weight=benchmark_weight, prior="half"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pipelines/competition_controls_v1.json")
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
    out.mkdir(parents=True)

    runs, per_draw = [], []
    for regime in recipe["regimes"]:
        for index in range(recipe["draws"]):
            seed = recipe["seed"] + index
            draw = select_draw(cells, labels, seed=seed, max_pairs=recipe["max_pairs_per_draw"],
                               regime=regime, min_items=recipe["min_items"],
                               directions=recipe["directions"])
            cache = draw_evidence(draw)
            scored = {name: score_draw(draw, predictor, evidence_cache=cache)
                      for name, predictor in arms(recipe["prior_strength"],
                                                  recipe["benchmark_evidence_weight"]).items()}
            oracle = oracle_curve(draw)
            benchmarks = sorted({pair.benchmark for pair in draw.pairs})
            per_draw.append({
                "regime": regime, "seed": seed, "pairs": len(draw.pairs),
                "benchmarks": benchmarks,
                "mean_pair_rate": float(np.mean([pair.rate for pair in draw.pairs])),
                "mean_pair_items": float(np.mean([len(pair.acquisition) + len(pair.evaluation)
                                                  for pair in draw.pairs])),
                "oracle_brier_alc": oracle["brier_alc"],
                "oracle_curve": {str(budget): value for budget, value in oracle["curve"].items()},
                "arms": {name: {"brier_alc": arm["brier_alc"], "auroc": arm["auroc"],
                                "curve": {str(budget): value for budget, value in arm["curve"].items()},
                                "pairs_with_benchmark_evidence": arm["pairs_with_benchmark_evidence"],
                                "mean_other_labels": arm["mean_other_labels"]}
                         for name, arm in scored.items()},
            })
            runs.append((regime, seed, scored))
            print(f"{regime} seed={seed} pairs={len(draw.pairs)} "
                  + " ".join(f"{name}={arm['brier_alc']:.4f}" for name, arm in scored.items()),
                  flush=True)

    summary = {"version": "competition_controls_v1", "pair_matrix_sha256": matrix["manifest_sha256"],
               "budgets": list(BUDGETS), "draws": recipe["draws"],
               "regimes": recipe["regimes"], "directions": recipe["directions"],
               "max_pairs_per_draw": recipe["max_pairs_per_draw"],
               "prior_strength": recipe["prior_strength"],
               "benchmark_evidence_weight": recipe["benchmark_evidence_weight"],
               "arms": {}, "per_draw": per_draw, "labels_reinterpreted": False,
               "sealed_roles_opened": False, "fitting_performed": False}
    pooled = {}
    for regime in recipe["regimes"]:
        selected = [(seed, scored) for name, seed, scored in runs if name == regime]
        pooled[regime] = {}
        for device in DEVICES:
            values = [scored[device]["brier_alc"] for _, scored in selected]
            curves = {budget: float(np.mean([scored[device]["curve"][budget] for _, scored in selected]))
                      for budget in BUDGETS}
            aurocs = [scored[device]["auroc"] for _, scored in selected if scored[device]["auroc"] is not None]
            pooled[regime][device] = {"brier_alc_mean": float(np.mean(values)),
                                      "brier_alc_sd": float(np.std(values)),
                                      "curve_mean": {str(budget): value for budget, value in curves.items()},
                                      "auroc_mean": float(np.mean(aurocs)) if aurocs else None}
        oracle_values = [row["oracle_brier_alc"] for row in per_draw if row["regime"] == regime]
        pooled[regime]["oracle_cell_rate"] = {"brier_alc_mean": float(np.mean(oracle_values))}
        base = pooled[regime]["acquired_label_mean"]["brier_alc_mean"]
        for device, values in pooled[regime].items():
            if "brier_alc_mean" in values:
                values["difference_from_acquired_mean"] = values["brier_alc_mean"] - base
        summary["arms"][regime] = pooled[regime]
    write_json(out / "resolved_config.json", recipe)
    write_json(out / "summary.json", summary)
    refresh_manifest(out)
    if not args.no_tracking:
        record(ROOT, out, run_name="competition_controls_v1",
               tags={"source_policy": "provided_public_corpus_only", "fitting_role": "none",
                     "gpu_jobs": "0", "sealed_roles_opened": "false"},
               params={key: recipe[key] for key in KEYS - {"version"}},
               metrics={f"{regime}_{device}_alc": values["brier_alc_mean"]
                        for regime, values in pooled.items() for device, values in values.items()
                        if isinstance(values, dict) and "brier_alc_mean" in values},
               artifacts=[ROOT / "src/paiec/competition_scorer.py",
                          ROOT / "src/paiec/competition_pairs.py", Path(__file__).resolve()])
        refresh_manifest(out)
    print(json.dumps({regime: {device: round(values["brier_alc_mean"], 4)
                               for device, values in arms_by_regime.items()
                               if "brier_alc_mean" in values}
                      for regime, arms_by_regime in pooled.items()}, indent=2))


if __name__ == "__main__":
    main()
