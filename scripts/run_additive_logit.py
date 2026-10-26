#!/usr/bin/env python3
"""Additive two-block logistic base: system strength plus task difficulty."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from paiec.additive_logit import checked_penalties, evaluate_additive
from paiec.catboost_features import CATALOG
from paiec.catboost_unseen_benchmark import UnseenBenchmarkConfig
from paiec.feature_preparation import local_path, write_json
from paiec.head_training import file_hash

RECIPE_KEYS = {"version", "prepared", "feature_groups", "penalty_grid", "max_iterations",
               "tolerance", "prior_strength", "offset_regularization", "unseen_benchmark"}


def load_recipe(path):
    config = json.loads(path.read_text())
    if set(config) != RECIPE_KEYS or config["version"] != "additive_logit_v1":
        raise ValueError("Unknown additive-logit recipe")
    if config["prepared"] != "runs/competition_pipeline/protocol_panel_v2":
        raise ValueError("This recipe version is bound to the complete protocol panel")
    groups, known = config["feature_groups"], {spec[0] for spec in CATALOG.values()}
    if (not isinstance(groups, list) or not groups or len(set(groups)) != len(groups)
            or any(not isinstance(group, str) or group not in known for group in groups)):
        raise ValueError("Declared feature groups must be known catalog groups")
    checked_penalties(config["penalty_grid"])
    for key in ["max_iterations", "prior_strength", "offset_regularization", "tolerance"]:
        if type(config[key]) not in {int, float} or not config[key] > 0:
            raise ValueError("Invalid positive recipe setting: " + key)
    UnseenBenchmarkConfig.model_validate({"panel_run": config["prepared"], **config["unseen_benchmark"]})
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pipelines/additive_logit_v1.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--no-tracking", action="store_true")
    args = parser.parse_args()
    path = (ROOT / args.config).resolve()
    if not path.is_relative_to(ROOT / "configs/pipelines"):
        raise ValueError("Use a repository pipeline configuration")
    config = load_recipe(path)
    out = local_path(ROOT, args.out)
    prepared = local_path(ROOT, config["prepared"])
    if out.exists():
        raise ValueError("Use a fresh experiment output directory")
    if out == prepared or out.is_relative_to(prepared) or prepared.is_relative_to(out):
        raise ValueError("Keep experiment output separate from prepared data")
    result = evaluate_additive(ROOT, prepared, out, recipe=config)
    write_json(out / "resolved_config.json", config)
    write_json(out / "manifest.json", {"artifacts": {
        str(item.relative_to(out)): file_hash(item)
        for item in sorted(out.rglob("*")) if item.is_file() and item.name != "manifest.json"}})
    if not args.no_tracking:
        import os
        os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
        import mlflow
        from paiec.tracking import local_tracking_uri
        mlflow.set_tracking_uri(local_tracking_uri(ROOT))
        mlflow.set_experiment("PAIEC Competition Analysis")
        with mlflow.start_run(run_name="additive_logit_v1") as run:
            mlflow.set_tags({"source_policy": "provided_dataset_only", "fitting_role": "training_only",
                             "model": "additive_logit_v1", "native_laya_inference": "false"})
            mlflow.log_params({key: config[key] for key in
                               ["version", "max_iterations", "tolerance", "prior_strength",
                                "offset_regularization"]})
            mlflow.log_param("feature_groups", json.dumps(config["feature_groups"]))
            mlflow.log_param("penalty_grid", json.dumps(config["penalty_grid"], sort_keys=True))
            mlflow.log_metrics({"cold_brier_" + name: value for name, value in
                                result["zero_budget_brier_all_held_rows_equal_scope"].items()})
            mlflow.log_metrics({"alc_" + name.replace("/", "_"): value for name, value in
                                result["brier_alc_on_common_eligible_scopes"].items()})
            if "unseen_benchmark" in result:
                mlflow.log_metrics({"unseen_primary_equal_scope_brier":
                                    result["unseen_benchmark"]["primary_equal_scope_brier"]})
            mlflow.log_artifacts(str(out), "analysis")
            for name in ["additive_logit.py", "catboost_features.py", "budget_evaluation.py",
                         "catboost_unseen_benchmark.py"]:
                mlflow.log_artifact(str(ROOT / "src/paiec" / name), "source")
            mlflow.log_artifact(str(Path(__file__).resolve()), "source")
            write_json(out / "tracking_receipt.json", {"mlflow_run_id": run.info.run_id,
                                                      "experiment_id": run.info.experiment_id})
            write_json(out / "manifest.json", {"artifacts": {
                str(item.relative_to(out)): file_hash(item)
                for item in sorted(out.rglob("*")) if item.is_file() and item.name != "manifest.json"}})
    print(json.dumps({key: result[key] for key in
                      ["zero_budget_brier_all_held_rows_equal_scope",
                       "brier_alc_on_common_eligible_scopes"]} |
                     ({"unseen": result["unseen_benchmark"]} if "unseen_benchmark" in result else {}),
                     indent=2))


if __name__ == "__main__":
    main()
