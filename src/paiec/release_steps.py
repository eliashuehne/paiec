"""ZenML assessment/release graph with an explicit rejection branch."""
from pathlib import Path

import mlflow
from zenml import get_step_context, pipeline, step

from .feature_preparation import write_json
from .release_workflow import verify_release_inputs, assess_or_review_candidate, package_validated_candidate
from .zenml_pipeline import TRACKER


@step(enable_cache=False)
def verify_candidate(root: str, config: dict) -> dict:
    return verify_release_inputs(root, config)


@step(enable_cache=False)
def assess_candidate(inputs: dict, out: str) -> dict:
    assessment = assess_or_review_candidate(inputs, out)
    write_json(Path(out) / "assessment_snapshot.json", assessment)
    return assessment


@step(enable_cache=False)
def apply_release_gate(inputs: dict, assessment: dict, out: str) -> dict:
    return package_validated_candidate(inputs, assessment, out)


@step(enable_cache=False, experiment_tracker=TRACKER)
def publish_release_decision(result: dict, out: str) -> dict:
    mlflow.set_tags({"mode": "validation_release", "candidate_passed": str(result["candidate_passed"]).lower(),
                     "uploaded": "false", "assessment_mode": result["assessment_mode"]})
    mlflow.log_metrics({"alc_" + name: value for name, value in result["scores"].items()})
    mlflow.log_artifacts(out, "release_review")
    write_json(Path(out) / "tracking_receipt.json", {"mlflow_run_id": mlflow.active_run().info.run_id,
        "zenml_run_id": str(get_step_context().pipeline_run.id)})
    return result


@pipeline(name="competition_validation_release", enable_cache=False)
def competition_validation_release(root: str, config: dict, out: str):
    inputs = verify_candidate(root, config)
    assessment = assess_candidate(inputs, out)
    decision = apply_release_gate(inputs, assessment, out)
    publish_release_decision(decision, out)
