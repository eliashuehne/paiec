#!/usr/bin/env python3
"""Assess a frozen evidence policy on the original development split; no fitting."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from paiec.feature_preparation import local_path,write_json
from paiec.policy_validation import assess_frozen_policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--policy-sha256", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = local_path(ROOT,args.out)
    report = assess_frozen_policy(ROOT,local_path(ROOT,args.prepared),local_path(ROOT,args.policy),args.policy_sha256,out)
    os.environ["MLFLOW_DISABLE_AGENT_HINT"]="1"
    import mlflow
    from paiec.tracking import local_tracking_uri
    mlflow.set_tracking_uri(local_tracking_uri(ROOT));mlflow.set_experiment("PAIEC Competition Analysis")
    with mlflow.start_run(run_name="frozen_evidence_policy_assessment_v1") as run:
        mlflow.set_tags({"source_policy":"provided_dataset_only","fitting_performed":"false",
                         "evaluation_role":"original_development","sealed_roles_opened":"false"})
        mlflow.log_params({"policy_sha256":args.policy_sha256,"selected_strength":report["selected_strength"]})
        mlflow.log_metrics({"alc_"+k:v["brier_alc"] for k,v in report["results"].items()})
        mlflow.log_artifacts(str(out),"assessment")
        mlflow.log_artifact(str(ROOT/"src/paiec/policy_validation.py"),"source")
        write_json(out/"tracking_receipt.json",{"mlflow_run_id":run.info.run_id,"experiment_id":run.info.experiment_id})
    print(json.dumps({k:report[k] for k in ["selected_strength","results","candidate_passes_development_gate","recommended_strength"]},indent=2))


if __name__=="__main__":
    main()
