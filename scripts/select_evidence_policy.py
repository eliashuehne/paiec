#!/usr/bin/env python3
"""Select a portable evidence policy using training-family outcomes only."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from paiec.evidence_policy_analysis import analyze_evidence_policies
from paiec.feature_preparation import local_path, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = local_path(ROOT, args.out)
    report = analyze_evidence_policies(local_path(ROOT, args.prepared), out)
    os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
    import mlflow
    from paiec.tracking import local_tracking_uri
    mlflow.set_tracking_uri(local_tracking_uri(ROOT))
    mlflow.set_experiment("PAIEC Competition Analysis")
    with mlflow.start_run(run_name="evidence_policy_training_family_selection_v1") as run:
        mlflow.set_tags({"source_policy": "provided_dataset_only", "fitting_role": "training_only",
                         "validation": "whole_training_family_outer_validation"})
        mlflow.log_params({"selected_strength": report["selected_strength"], "selected_strategy": report["selected_strategy"],
                           "prepared_manifest_sha256": report["prepared_manifest_sha256"]})
        mlflow.log_metrics({"nested_alc_" + k: v for k, v in report["nested_strategy_alc_equal_scope"].items()})
        mlflow.log_artifacts(str(out), "analysis")
        for name in ["evidence_policy.py", "evidence_policy_analysis.py"]:
            mlflow.log_artifact(str(ROOT / "src/paiec" / name), "source")
        write_json(out / "tracking_receipt.json", {"mlflow_run_id": run.info.run_id, "experiment_id": run.info.experiment_id})
    print(json.dumps({key: report[key] for key in ["selected_strategy", "selected_strength", "fixed_strength_alc_equal_scope", "nested_strategy_alc_equal_scope"]}, indent=2))


if __name__ == "__main__":
    main()
