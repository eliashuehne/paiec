"""ZenML orchestration around unchanged training-only evidence algorithms."""
from pathlib import Path

import mlflow
from zenml import get_step_context, pipeline, step

from .evidence_workflow import (verify_evidence_inputs, compare_evidence_controls,
                               select_evidence_candidate, finalize_evidence_workflow)
from .feature_preparation import write_json
from .zenml_pipeline import TRACKER


@step(enable_cache=False)
def verify_evidence_sources(root: str, config: dict) -> dict:
    return verify_evidence_inputs(root, config)


@step(enable_cache=False)
def compare_training_evidence(inputs: dict, out: str) -> dict:
    return compare_evidence_controls(inputs, out)


@step(enable_cache=False)
def select_training_evidence(inputs: dict, controls: dict, out: str) -> dict:
    return select_evidence_candidate(inputs, controls, out)


@step(enable_cache=False, experiment_tracker=TRACKER)
def publish_evidence_candidate(inputs: dict, controls: dict, selection: dict, out: str) -> dict:
    report = finalize_evidence_workflow(inputs, controls, selection, out)
    mlflow.set_tags({"mode": "competition_evidence_selection", "source_policy": "provided_dataset_only",
                     "outcome_role": "training_only", "release_approved": "false", "native_laya_inference": "false"})
    mlflow.log_params({"prepared_manifest_sha256": inputs["prepared_manifest_sha256"],
                       "selected_strength": report["selected_strength"], "selected_strategy": report["selected_strategy"],
                       "zenml_run_id": str(get_step_context().pipeline_run.id)})
    mlflow.log_metrics({"fixed_alc_" + k: v for k, v in report["fixed_control_alc"].items()})
    mlflow.log_metrics({"nested_alc_" + k: v for k, v in report["nested_strategy_alc"].items()})
    mlflow.log_artifacts(out, "evidence_workflow")
    source = Path(__file__).parent
    for name in ["evidence_workflow.py", "evidence_steps.py", "competition_analysis.py",
                 "evidence_policy_analysis.py", "evidence_policy.py", "competition_fields.py"]:
        mlflow.log_artifact(str(source / name), "source")
    write_json(Path(out) / "tracking_receipt.json", {"mlflow_run_id": mlflow.active_run().info.run_id,
        "zenml_run_id": str(get_step_context().pipeline_run.id), "release_approved": False})
    return report


@pipeline(name="competition_evidence_selection", enable_cache=False)
def competition_evidence_selection(root: str, config: dict, out: str):
    inputs = verify_evidence_sources(root, config)
    controls = compare_training_evidence(inputs, out)
    selection = select_training_evidence(inputs, controls, out)
    publish_evidence_candidate(inputs, controls, selection, out)
