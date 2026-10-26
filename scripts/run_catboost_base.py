#!/usr/bin/env python3
"""CPU-only, whole-training-family CatBoost base-predictor experiment."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from paiec.catboost_analysis import evaluate_catboost_base
from paiec.catboost_features import CATALOG
from paiec.catboost_training import checked_configs
from paiec.feature_preparation import local_path, write_json
from paiec.head_training import file_hash

RECIPE_KEYS = {"version", "prepared", "threads", "seed", "max_iterations", "early_stopping_rounds",
               "min_iterations", "prior_strength", "offset_regularization", "fit_configs"}


PROFILE_KEYS = {"profile_catalog", "profile_policy"}
CONTROL_KEYS = {"profile_permutation_seed"}
GROUPS_VERSION = "catboost_base_v3"
PROFILE_VERSION = "catboost_base_v4"
CONTROL_VERSION = "catboost_base_v5"


def load_recipe(path):
    config = json.loads(path.read_text())
    declared_groups = RECIPE_KEYS | {"feature_groups"}
    if (set(config) not in (RECIPE_KEYS, declared_groups, declared_groups | PROFILE_KEYS,
                            declared_groups | PROFILE_KEYS | CONTROL_KEYS)
            or not str(config["version"]).startswith("catboost_base_v")):
        raise ValueError("Unknown CatBoost base recipe")
    version = str(config["version"])
    if ("feature_groups" in config) != (version in {GROUPS_VERSION, PROFILE_VERSION, CONTROL_VERSION}):
        raise ValueError("Only the declared-group recipe versions declare feature groups")
    if version in {PROFILE_VERSION, CONTROL_VERSION}:
        if not PROFILE_KEYS <= set(config) or "system_profile" not in config["feature_groups"]:
            raise ValueError("The profile recipe must declare its catalog and select system_profile")
        for key in PROFILE_KEYS:
            if type(config[key]) is not str or not (ROOT / config[key]).exists():
                raise ValueError("Declared profile catalog or policy is missing: " + key)
    elif PROFILE_KEYS & set(config):
        raise ValueError("Only the profile recipe version declares a profile catalog")
    if (version == CONTROL_VERSION) != ("profile_permutation_seed" in config):
        raise ValueError("Only the control recipe version declares a permutation seed")
    if "profile_permutation_seed" in config and type(config["profile_permutation_seed"]) is not int:
        raise ValueError("The permutation seed must be an integer")
    if "feature_groups" in config:
        groups = config["feature_groups"]
        known = {spec[0] for spec in CATALOG.values()}
        if (not isinstance(groups, list) or not groups or len(set(groups)) != len(groups)
                or any(not isinstance(group, str) or group not in known for group in groups)):
            raise ValueError("Declared feature groups must be known catalog groups")
    if config["prepared"] != "runs/competition_pipeline/protocol_panel_v2":
        raise ValueError("This recipe version is bound to the complete protocol panel")
    for key in ["threads", "seed", "max_iterations", "early_stopping_rounds", "min_iterations"]:
        if type(config[key]) is not int or config[key] < (0 if key == "seed" else 1):
            raise ValueError("Invalid integer recipe setting: " + key)
    for key in ["prior_strength", "offset_regularization"]:
        if type(config[key]) not in {int, float} or not config[key] > 0:
            raise ValueError("Invalid positive recipe setting: " + key)
    checked_configs(config["fit_configs"])
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pipelines/catboost_base_v1.json")
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
    result = evaluate_catboost_base(prepared, out, recipe=config)
    write_json(out / "resolved_config.json", config)
    write_json(out / "manifest.json", {"artifacts": {str(p.relative_to(out)): file_hash(p)
                                                     for p in sorted(out.rglob("*")) if p.is_file()
                                                     and p.name != "manifest.json"}})
    if not args.no_tracking:
        os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
        import mlflow
        from paiec.tracking import local_tracking_uri
        mlflow.set_tracking_uri(local_tracking_uri(ROOT))
        mlflow.set_experiment("PAIEC Competition Analysis")
        with mlflow.start_run(run_name="catboost_base_family_" + config["version"].rsplit("_", 1)[-1]) as run:
            mlflow.set_tags({"source_policy": "provided_dataset_only", "fitting_role": "training_only",
                             "validation": result["validation"], "native_laya_inference": "false",
                             "gpu_jobs": "0"})
            mlflow.log_params({key: config[key] for key in RECIPE_KEYS - {"prepared", "fit_configs"}})
            mlflow.log_param("fit_configs", json.dumps(config["fit_configs"], sort_keys=True))
            mlflow.log_metrics({"cold_brier_" + name: value for name, value in
                                result["zero_budget_brier_all_held_rows_equal_scope"].items()})
            mlflow.log_metrics({"alc_" + name.replace("/", "_"): value for name, value in
                                result["brier_alc_on_common_eligible_scopes"].items()})
            mlflow.log_artifacts(str(out), "analysis")
            for name in ["catboost_analysis.py", "catboost_features.py", "catboost_training.py",
                         "catboost_gate.py", "budget_evaluation.py"]:
                mlflow.log_artifact(str(ROOT / "src/paiec" / name), "source")
            mlflow.log_artifact(str(Path(__file__).resolve()), "source")
            write_json(out / "tracking_receipt.json", {"mlflow_run_id": run.info.run_id,
                                                       "experiment_id": run.info.experiment_id})
            write_json(out / "manifest.json", {"artifacts": {str(p.relative_to(out)): file_hash(p)
                                                             for p in sorted(out.rglob("*")) if p.is_file()
                                                             and p.name != "manifest.json"}})
    print(json.dumps({key: result[key] for key in ["training_families", "full_budget_eligible_scopes",
                                                   "zero_budget_brier_all_held_rows_equal_scope",
                                                   "brier_alc_on_common_eligible_scopes",
                                                   "runtime_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
