"""Thin local tracking graph; pure extraction lives in feature_foundation."""
import json
from pathlib import Path

import mlflow
from zenml import pipeline, step, get_step_context

from .feature_foundation.contracts import FeatureFoundationConfig, write_json
from .feature_foundation.sources import inventory_population
from .feature_foundation.build import build_bundle, validate_bundle
from .zenml_pipeline import TRACKER


@step(enable_cache=False)
def verify_feature_sources(config: dict, source_root: str) -> dict:
    parsed = FeatureFoundationConfig.model_validate(config)
    population = inventory_population(parsed, Path(source_root))
    return {"config": config, "source_root": source_root, "benchmarks": len(population)}


@step(enable_cache=False)
def prepare_feature_tables(inputs: dict, out: str) -> dict:
    bundle = build_bundle(FeatureFoundationConfig.model_validate(inputs["config"]),
                          source_root=Path(inputs["source_root"]), out=Path(out),
                          progress=lambda row: print(json.dumps(row), flush=True))
    return {"directory": str(bundle.directory), "population": bundle.manifest["population"],
            "review_complete": bundle.manifest["review_complete"]}


@step(enable_cache=False, experiment_tracker=TRACKER)
def publish_feature_report(bundle: dict) -> dict:
    directory = Path(bundle["directory"])
    result = validate_bundle(directory)
    mlflow.set_tags({"mode": "feature_preparation", "source_scope": "full_public_population", "model_training": "false", "outcomes_read": "false"})
    mlflow.log_metrics({"population_" + key: value for key, value in bundle["population"].items()})
    # Log compact metadata; large feature tables/task text stay in the bundle.
    for filename in ("manifest.json", "receipt.json", "coverage.json", "feature_dictionary.json", "report.html"):
        mlflow.log_artifact(str(directory / filename))
    context = get_step_context()
    tracking = {"zenml_run_id": str(context.pipeline_run.id), "mlflow_run_id": mlflow.active_run().info.run_id,
                "bundle": str(directory), "not_part_of_immutable_bundle": True}
    write_json(directory.parent / (directory.name + "_tracking_receipt.json"), tracking)
    return {**result, **tracking}


@pipeline(enable_cache=False)
def feature_foundation_preparation(root: str, config: dict, out: str, source_root: str):
    inputs = verify_feature_sources(config, source_root)
    bundle = prepare_feature_tables(inputs, out)
    publish_feature_report(bundle)
