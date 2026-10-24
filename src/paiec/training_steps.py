"""ZenML native execution graph; the selected worker owns native ML libraries."""
from pathlib import Path

import mlflow
from zenml import get_step_context, pipeline, step

from .feature_preparation import write_json
from .training_workflow import plan_native_job, dispatch_native_job
from .zenml_pipeline import TRACKER


@step(enable_cache=False)
def verify_native_job(root: str, config: dict, out: str) -> dict:
    plan = plan_native_job(root, config, out)
    write_json(Path(out) / "native_plan.json", plan)
    return plan


@step(enable_cache=False)
def execute_native_job(plan: dict, out: str) -> dict:
    return dispatch_native_job(plan, out)


@step(enable_cache=False, experiment_tracker=TRACKER)
def publish_native_job(plan: dict, result: dict, out: str) -> dict:
    mlflow.set_tags({"mode": "laya_native_execution", "task": plan["job"]["task"],
                     "executor": plan["profile"]["executor"], "release_approved": "false"})
    mlflow.log_dict(plan, "native_plan.json")
    mlflow.log_dict(result, "summary.json")
    if plan["job"]["task"] == "predict-grouped":
        from .grouped_evaluation import GroupedEvaluationConfig, score_development
        from .head_training import file_hash
        root = Path(plan["root"])
        for name, digest in plan["local_scorer_sha256"].items():
            if file_hash(root / name) != digest:
                raise ValueError("Local scoring artifact changed during inference: " + name)
        config = GroupedEvaluationConfig.model_validate_json((root / plan["job"]["grouped_prediction_config"]).read_text())
        report = score_development(root / config.prepared, root / result["job_output"] / "output", Path(out) / "scores",
                                   checkpoint_manifest_sha256=config.checkpoint_manifest_sha256)
        mlflow.log_dict(report, "development/report.json")
        for key in ["brier_alc", "constant_brier_alc", "brier_alc_delta_vs_constant", "pair_weighted_ece", "mean_within_pair_auroc"]:
            if report[key] is not None:
                mlflow.log_metric(key, report[key])
        mlflow.log_artifacts(str(Path(out) / "scores"), "development")
        mlflow.set_tags({"calibration_opened": "false", "confirmation_opened": "false", "revealed_evidence_used": "false"})
    # Native trainers retain their own detailed MLflow/checkpoint artifacts.
    write_json(Path(out) / "tracking_receipt.json", {"mlflow_run_id": mlflow.active_run().info.run_id,
        "zenml_run_id": str(get_step_context().pipeline_run.id), "native_job_output": result["job_output"]})
    return result


@pipeline(name="laya_native_execution", enable_cache=False)
def laya_native_execution(root: str, config: dict, out: str):
    plan = verify_native_job(root, config, out)
    result = execute_native_job(plan, out)
    publish_native_job(plan, result, out)
