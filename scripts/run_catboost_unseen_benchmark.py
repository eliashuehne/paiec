#!/usr/bin/env python3
"""Score panel fold models on unseen benchmarks with already-seen model names."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from paiec.catboost_unseen_benchmark import UnseenBenchmarkConfig, evaluate_unseen_benchmarks
from paiec.feature_preparation import local_path, write_json
from paiec.head_training import file_hash


def load_config(path):
    config = UnseenBenchmarkConfig.model_validate(json.loads(path.read_text()))
    if config.data_root != "payload/data":
        raise ValueError("This recipe version is bound to the public corpus root")
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pipelines/catboost_unseen_benchmark_v1.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--no-tracking", action="store_true")
    args = parser.parse_args()
    path = (ROOT / args.config).resolve()
    if not path.is_relative_to(ROOT / "configs/pipelines"):
        raise ValueError("Use a repository pipeline configuration")
    config = load_config(path)
    out = local_path(ROOT, args.out)
    panel_run = local_path(ROOT, config.panel_run)
    if out == panel_run or out.is_relative_to(panel_run) or panel_run.is_relative_to(out):
        raise ValueError("Keep evaluation output separate from the panel run")
    result = evaluate_unseen_benchmarks(ROOT, config, out)
    write_json(out / "resolved_config.json", config.model_dump())
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
        with mlflow.start_run(run_name="catboost_unseen_benchmark_v1") as run:
            mlflow.set_tags({"source_policy": "provided_dataset_only",
                             "evaluation": "unseen_benchmark_known_model",
                             "external_rows_used_for_fitting": "false",
                             "sealed_roles_opened": "false"})
            mlflow.log_params({key: value for key, value in config.model_dump().items()
                               if not isinstance(value, dict)})
            mlflow.log_metrics({"primary_equal_scope_brier": result["primary_equal_scope_brier"],
                                "pooled_equal_scope_brier": result["pooled_row_metrics"]["equal_scope_brier"],
                                "pooled_pooled_brier": result["pooled_row_metrics"]["pooled_brier"],
                                "scored_rows": result["rows"]["scored"]})
            mlflow.log_artifacts(str(out), "analysis")
            for name in ["catboost_unseen_benchmark.py", "catboost_analysis.py", "catboost_features.py",
                         "catboost_training.py"]:
                mlflow.log_artifact(str(ROOT / "src/paiec" / name), "source")
            mlflow.log_artifact(str(Path(__file__).resolve()), "source")
            write_json(out / "tracking_receipt.json", {"mlflow_run_id": run.info.run_id,
                                                       "experiment_id": run.info.experiment_id})
            write_json(out / "manifest.json", {"artifacts": {
                str(item.relative_to(out)): file_hash(item)
                for item in sorted(out.rglob("*")) if item.is_file() and item.name != "manifest.json"}})
    print(json.dumps({"admitted_benchmarks": result["selection"]["admitted_benchmarks"],
                      "scored_rows": result["rows"]["scored"],
                      "primary_equal_scope_brier": result["primary_equal_scope_brier"],
                      "pooled_row_metrics": result["pooled_row_metrics"],
                      "strata": {name: {"rows": row["rows"],
                                        "equal_scope_brier": row.get("equal_scope_brier")}
                                 for name, row in result["strata"].items()},
                      "folds": [{"held": row["held_training_family"], "rows": row["eligible_rows"],
                                 "exact": row["exact_subject_rows"],
                                 "brier": (row["external_metrics"] or {}).get("equal_scope_brier")}
                                for row in result["folds"]]}, indent=2))


if __name__ == "__main__":
    main()
