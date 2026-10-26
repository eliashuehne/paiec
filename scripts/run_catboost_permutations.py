#!/usr/bin/env python3
"""Run the scrambled-profile permutation distribution and report the null."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from paiec.catboost_permutation import PermutationConfig, artifact_digest, run_control
from paiec.feature_preparation import local_path, write_json
from paiec.head_training import file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pipelines/catboost_permutation_control_v1.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--aggregate-only", action="store_true",
                        help="Re-read hash-verified replicates instead of fitting missing ones")
    parser.add_argument("--allow-partial", action="store_true",
                        help="With --aggregate-only, report the declared replicates that exist")
    parser.add_argument("--no-tracking", action="store_true")
    args = parser.parse_args()
    path = (ROOT / args.config).resolve()
    if not path.is_relative_to(ROOT / "configs/pipelines"):
        raise ValueError("Use a repository pipeline configuration")
    config = PermutationConfig.model_validate(json.loads(path.read_text()))
    out = local_path(ROOT, args.out)
    protected = {local_path(ROOT, value) for value in
                 [config.observed_panel_run, config.observed_unseen_run,
                  config.baseline_panel_run, config.baseline_unseen_run]
                 + [row.panel_run for row in config.reused_replicates]
                 + [row.unseen_run for row in config.reused_replicates]}
    if any(out == folder or out.is_relative_to(folder) or folder.is_relative_to(out) for folder in protected):
        raise ValueError("Keep the permutation output separate from every declared run")
    report = run_control(ROOT, config, out, fit=not args.aggregate_only,
                         allow_partial=args.allow_partial)
    if not args.no_tracking:
        os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
        import mlflow
        from paiec.tracking import local_tracking_uri
        mlflow.set_tracking_uri(local_tracking_uri(ROOT))
        mlflow.set_experiment("PAIEC Competition Analysis")
        with mlflow.start_run(run_name="catboost_permutation_control_v1") as run:
            mlflow.set_tags({"source_policy": "provided_plus_research_profiles",
                             "usage": "research_only", "fitting_role": "training_only",
                             "evaluation": "unseen_benchmark_known_model_permutation_null",
                             "external_rows_used_for_fitting": "false",
                             "sealed_roles_opened": "false",
                             "release_eligible": "false"})
            mlflow.log_params({"permutations": report["study"]["permutations_in_distribution"],
                               "fresh_seeds": json.dumps(report["study"]["fresh_seeds"]),
                               "reused_seeds": json.dumps(report["study"]["reused_seeds"]),
                               "primary_metric": report["primary"]["metric"],
                               "alpha": config.alpha})
            mlflow.log_metrics({"primary_p_value": report["primary"]["p_value"],
                                "primary_residual_gain": report["primary"]["residual_gain"]})
            mlflow.log_metrics({f"null_mean_{metric}": row["null_mean"]
                                for metric, row in report["comparison"].items()})
            mlflow.log_artifacts(str(out), "analysis")
            for name in ["catboost_permutation.py", "catboost_analysis.py", "catboost_features.py",
                         "catboost_training.py", "catboost_unseen_benchmark.py",
                         "system_profile_features.py"]:
                mlflow.log_artifact(str(ROOT / "src/paiec" / name), "source")
            mlflow.log_artifact(str(Path(__file__).resolve()), "source")
            write_json(out / "tracking_receipt.json", {"mlflow_run_id": run.info.run_id,
                                                       "experiment_id": run.info.experiment_id,
                                                       "comparison_sha256": artifact_digest(report)})
            write_json(out / "manifest.json", {"artifacts": {
                str(item.relative_to(out)): file_hash(item)
                for item in sorted(out.rglob("*")) if item.is_file() and item.name != "manifest.json"}})
    print(json.dumps({"permutations": report["study"]["permutations_in_distribution"],
                      "complete_as_declared": report["study"]["complete_as_declared"],
                      "unfinished_declared_seeds": report["study"]["unfinished_declared_seeds"],
                      "primary": report["primary"],
                      "comparison": {metric: {"observed": row["observed"], "null_mean": row["null_mean"],
                                              "residual_gain": row["residual_gain"],
                                              "p_value": row["p_value"]}
                                     for metric, row in report["comparison"].items()}}, indent=2))


if __name__ == "__main__":
    main()
