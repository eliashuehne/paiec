#!/usr/bin/env python3
"""Uncertainty intervals for the whole-family CatBoost ablation margins."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from paiec.catboost_uncertainty import analyze_ablation_margins
from paiec.feature_preparation import local_path, write_json
from paiec.head_training import file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="runs/competition_analysis/catboost_base_family_v2")
    parser.add_argument("--out", required=True)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--alpha", type=float, default=.05)
    parser.add_argument("--no-tracking", action="store_true")
    args = parser.parse_args()
    run, out = local_path(ROOT, args.run), local_path(ROOT, args.out)
    if out == run or out.is_relative_to(run) or run.is_relative_to(out):
        raise ValueError("Keep uncertainty output separate from the analysed run")
    result = analyze_ablation_margins(run, out, draws=args.draws, seed=args.seed, alpha=args.alpha)
    if not args.no_tracking:
        import os
        os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
        import mlflow
        from paiec.tracking import local_tracking_uri
        mlflow.set_tracking_uri(local_tracking_uri(ROOT))
        mlflow.set_experiment("PAIEC Competition Analysis")
        with mlflow.start_run(run_name="catboost_base_family_v2_uncertainty") as tracked:
            mlflow.set_tags({"source_policy": "provided_dataset_only", "analysis": "ablation_uncertainty",
                             "model_fitted": "false", "source_run": result["source_run"]["recipe_version"]})
            mlflow.log_params({"draws": args.draws, "seed": args.seed, "alpha": args.alpha,
                               "source_manifest_sha256": result["source_run"]["manifest_sha256"]})
            mlflow.log_metrics({f"difference_{name}": row["difference_vs_full_model"]
                                for name, row in result["comparisons"].items()})
            mlflow.log_artifacts(str(out), "analysis")
            write_json(out / "tracking_receipt.json", {"mlflow_run_id": tracked.info.run_id,
                                                      "experiment_id": tracked.info.experiment_id})
            write_json(out / "manifest.json", {"artifacts": {
                str(path.relative_to(out)): file_hash(path)
                for path in sorted(out.rglob("*")) if path.is_file() and path.name != "manifest.json"}})
    print(json.dumps({name: {"difference": row["difference_vs_full_model"],
                            "interval": [row["interval_lower"], row["interval_upper"]],
                            "reading": row["reading"]}
                      for name, row in result["comparisons"].items()}, indent=2))


if __name__ == "__main__":
    main()
