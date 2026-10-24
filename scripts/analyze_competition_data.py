#!/usr/bin/env python3
"""Analyze training-only evidence controls; no model fitting, API or GPU calls."""
import argparse
import json
from pathlib import Path
import sys
import platform

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from paiec.competition_analysis import analyze_training_evidence
from paiec.feature_preparation import local_path, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--no-track", action="store_true", help="Skip local MLflow logging")
    args = parser.parse_args()
    result = analyze_training_evidence(local_path(ROOT, args.prepared), local_path(ROOT, args.out))
    if not args.no_track:
        import mlflow
        import numpy as np
        from paiec.tracking import local_tracking_uri
        mlflow.set_tracking_uri(local_tracking_uri(ROOT))
        mlflow.set_experiment("PAIEC Competition Analysis")
        with mlflow.start_run(run_name="training_evidence_diagnostic_v1") as run:
            mlflow.set_tags({"source_policy": "provided_dataset_only", "training_performed": "false",
                             "generalization_result": "false", "native_inference_performed": "false"})
            mlflow.log_params({"prepared_manifest_sha256": result["prepared_manifest_sha256"],
                               "python_version": platform.python_version(), "numpy_version": np.__version__,
                               "implementation_sha256": result["implementation_sha256"]})
            mlflow.log_metrics({"alc_" + name: value for name, value in result["brier_alc_equal_visible_scope"].items()})
            mlflow.log_metrics({"eligible_scopes": result["fully_eligible_retained_scopes"],
                                "excluded_small_scopes": result["excluded_small_retained_scopes"]})
            mlflow.log_artifacts(str(local_path(ROOT, args.out)), "analysis")
            for name in ["competition_analysis.py", "competition_data.py", "feature_evidence_plan.py", "metrics.py"]:
                mlflow.log_artifact(str(ROOT / "src/paiec" / name), "source")
            mlflow.log_artifact(str(Path(__file__).resolve()), "source")
            write_json(local_path(ROOT, args.out) / "tracking_receipt.json", {"mlflow_run_id": run.info.run_id,
                "experiment_id": run.info.experiment_id, "generalization_result": False})
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
