#!/usr/bin/env python3
"""CPU-only, whole-training-family conditional predictor experiment."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from paiec.conditional_analysis import evaluate_conditional_controls
from paiec.feature_preparation import local_path, write_json
from paiec.head_training import file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pipelines/conditional_control_v1.json")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    path = (ROOT / args.config).resolve()
    if not path.is_relative_to(ROOT / "configs/pipelines"):
        raise ValueError("Use a repository pipeline configuration")
    config = json.loads(path.read_text())
    if set(config) != {"version", "prepared", "l2", "prior_strength", "offset_regularization"} or config["version"] != "conditional_control_v1":
        raise ValueError("Unknown conditional-control recipe")
    out = local_path(ROOT, args.out)
    prepared = local_path(ROOT, config["prepared"])
    if out == prepared or out.is_relative_to(prepared) or prepared.is_relative_to(out):
        raise ValueError("Keep experiment output separate from prepared data")
    result = evaluate_conditional_controls(prepared, out, l2=config["l2"], prior_strength=config["prior_strength"],
                                          offset_regularization=config["offset_regularization"])
    write_json(out / "resolved_config.json", config)
    write_json(out / "manifest.json", {"artifacts": {str(p.relative_to(out)): file_hash(p) for p in sorted(out.rglob("*")) if p.is_file() and p.name != "manifest.json"}})
    os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
    import mlflow
    from paiec.tracking import local_tracking_uri
    mlflow.set_tracking_uri(local_tracking_uri(ROOT))
    mlflow.set_experiment("PAIEC Competition Analysis")
    with mlflow.start_run(run_name="conditional_training_family_v1") as run:
        mlflow.set_tags({"source_policy": "provided_dataset_only", "fitting_role": "training_only",
                         "validation": result["validation"], "native_laya_inference": "false"})
        mlflow.log_params({k: v for k, v in config.items() if k != "prepared"})
        mlflow.log_metrics({"alc_" + k.replace("/", "_"): v for k, v in result["brier_alc_on_common_eligible_scopes"].items()})
        mlflow.log_metrics({"cold_brier_" + k: v for k, v in result["zero_budget_brier_all_held_rows_equal_scope"].items()})
        mlflow.log_artifacts(str(out), "analysis")
        for name in ["conditional_analysis.py", "conditional_training.py", "conditional_features.py", "conditional_predictor.py"]:
            mlflow.log_artifact(str(ROOT / "src/paiec" / name), "source")
        mlflow.log_artifact(str(Path(__file__).resolve()), "source")
        write_json(out / "tracking_receipt.json", {"mlflow_run_id": run.info.run_id, "experiment_id": run.info.experiment_id})
    print(json.dumps({k: result[k] for k in ["training_families", "full_budget_eligible_scopes", "zero_budget_brier_all_held_rows_equal_scope", "brier_alc_on_common_eligible_scopes"]}, indent=2))


if __name__ == "__main__":
    main()
