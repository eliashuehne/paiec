#!/usr/bin/env python3
"""CPU-only sensitivity study of training-benchmark expansion.

Fits no candidate predictor and promotes nothing. See
`docs/training-data-expansion.md` for the measured result and limitations.
"""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from paiec.feature_preparation import local_path, write_json
from paiec.head_training import file_hash
from paiec.training_data_expansion import (
    TrainingExpansionConfig, resolve_data_root, run_training_data_expansion)


def refresh_manifest(out):
    write_json(out / "manifest.json", {"artifacts": {
        str(path.relative_to(out)): file_hash(path)
        for path in sorted(out.rglob("*")) if path.is_file() and path.name != "manifest.json"}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pipelines/training_data_expansion_v1.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--no-tracking", action="store_true",
                        help="Skip the local MLflow record; files are written either way")
    args = parser.parse_args()
    path = (ROOT / args.config).resolve()
    if not path.is_relative_to(ROOT / "configs/pipelines"):
        raise ValueError("Use a repository pipeline configuration")
    config = TrainingExpansionConfig.model_validate(json.loads(path.read_text()))
    out = local_path(ROOT, args.out)
    prepared = local_path(ROOT, config.prepared)
    data_root = resolve_data_root(ROOT, config.data_root)
    if out == prepared or out.is_relative_to(prepared) or prepared.is_relative_to(out):
        raise ValueError("Keep experiment output separate from prepared data")
    if out == data_root or out.is_relative_to(data_root) or data_root.is_relative_to(out):
        raise ValueError("Keep experiment output separate from the public data root")

    result = run_training_data_expansion(ROOT, config.model_dump(), out)
    write_json(out / "resolved_config.json", json.loads(path.read_text()))

    if not args.no_tracking:
        try:
            import mlflow
        except ImportError:
            print("MLflow tracking unavailable; install .[tracking] or use --no-tracking.", flush=True)
        else:
            from paiec.tracking import local_tracking_uri
            os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
            os.environ["MLFLOW_DISABLE_TELEMETRY"] = "true"
            mlflow.set_tracking_uri(local_tracking_uri(ROOT))
            mlflow.set_experiment("PAIEC Competition Analysis")
            with mlflow.start_run(run_name="training_data_expansion_v1") as run:
                mlflow.set_tags({"source_policy": "provided_dataset_only",
                                 "fitting_role": "training_only",
                                 "candidate_promoted": "false",
                                 "sealed_roles_opened": "false"})
                mlflow.log_params({"l2": config.l2, "interactions": config.interactions,
                                   "rows_per_system": config.rows_per_system,
                                   "min_pair_items": config.min_pair_items,
                                   "repeats": config.repeats,
                                   "seed": config.seed,
                                   "added_benchmarks": ",".join(b.name for b in config.added_benchmarks)})
                mlflow.log_metrics({f"equal_scope/{row['arm'].replace('+', '_')}": row["equal_scope_brier"]
                                    for row in result["expansion_arms"]})
                mlflow.log_metrics({f"diversity_gain_equal_family/{size}": value
                                    for size, value in result["diversity_gain_equal_family"].items()})
                mlflow.log_artifacts(str(out), "analysis")
                for name in ["training_data_expansion.py", "conditional_features.py",
                             "conditional_training.py", "conditional_analysis.py"]:
                    mlflow.log_artifact(str(ROOT / "src/paiec" / name), "source")
                mlflow.log_artifact(str(Path(__file__).resolve()), "source")
                write_json(out / "tracking_receipt.json",
                           {"mlflow_run_id": run.info.run_id, "experiment_id": run.info.experiment_id})
    refresh_manifest(out)

    print(json.dumps({
        "panel": result["panel"],
        "added_rows_per_repeat": result["added_rows_per_repeat"],
        "expansion_arms": result["expansion_arms"],
        "mean_delta_excluding_largest": result["mean_delta_excluding_largest"],
        "largest_per_family_abs_delta": result["largest_per_family_abs_delta"],
    }, indent=2))


if __name__ == "__main__":
    main()
