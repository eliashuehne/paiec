"""Separate ZenML graph for competition-data-only artifacts."""
from pathlib import Path
import json

import mlflow
from zenml import pipeline, step, get_step_context

from .competition_data import verify_dataset_sources, build_dataset_tables
from .feature_preparation import write_json
from .head_training import file_hash
from .zenml_pipeline import TRACKER


@step(enable_cache=False)
def verify_competition_sources(root: str, config: dict) -> dict:
    return verify_dataset_sources(root, config)


@step(enable_cache=False)
def prepare_competition_data(inputs: dict, out: str) -> dict:
    return build_dataset_tables(inputs, out)


@step(enable_cache=False, experiment_tracker=TRACKER)
def publish_competition_data(inputs: dict, summary: dict, out: str) -> dict:
    # Recheck selected dataset bytes before publication; no source registry is read.
    fresh = verify_dataset_sources(inputs["root"], inputs["config"])
    if fresh != inputs:
        raise ValueError("Competition sources changed during preparation")
    path = Path(out)
    manifest = json.loads((path / "manifest.json").read_text())
    for name, digest in manifest["artifacts"].items():
        target = (path / name).resolve()
        if not target.is_relative_to(path.resolve()) or file_hash(target) != digest:
            raise ValueError("Competition output changed before publication")
    mlflow.set_tags({"mode": "competition_data_only", "source_policy": "provided_dataset_only",
                     "training_performed": "false", "runtime_ready": "false"})
    mlflow.log_params({"dataset_revision": inputs["dataset_revision"],
                       "source_manifest_sha256": inputs["source_manifest_sha256"],
                       "zenml_run_id": str(get_step_context().pipeline_run.id)})
    mlflow.log_metrics({"task_inputs": summary["task_inputs"], "systems": summary["systems"],
                        "training_targets": summary["training_targets"], "external_profile_facts_used": 0})
    mlflow.log_artifacts(out, "competition_data")
    run = mlflow.active_run()
    write_json(path / "tracking_receipt.json", {"mlflow_run_id": run.info.run_id,
        "zenml_run_id": str(get_step_context().pipeline_run.id), "source_policy": "provided_dataset_only"})
    return summary


@pipeline(name="competition_dataset_preparation", enable_cache=False)
def competition_dataset_preparation(root: str, config: dict, out: str):
    inputs = verify_competition_sources(root, config)
    summary = prepare_competition_data(inputs, out)
    publish_competition_data(inputs, summary, out)
