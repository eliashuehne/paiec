"""ZenML steps for local feature preparation; no training dispatch."""
import json
import mlflow
import pandas as pd
from zenml import pipeline, step
from zenml.types import HTMLString
from pathlib import Path
from typing import Annotated
from . import feature_preparation as features
from .zenml_pipeline import TRACKER
from .tracking import git_context, log_source


def profile_table(path):
    rows = []
    def visit(profile_id, field, value):
        if isinstance(value, dict) and {"value", "status", "evidence"} <= value.keys():
            rows.append({"profile_id": profile_id, "field": field, "value": json.dumps(value["value"], ensure_ascii=False),
                         "status": value["status"], "note": value["note"]})
        elif isinstance(value, dict):
            for key, child in value.items():
                visit(profile_id, (field + "." + key).strip("."), child)
    for row in features.read_jsonl(path):
        for field in ["identity", "specifications"]:
            visit(row["profile_id"], field, row[field])
    return pd.DataFrame(rows, columns=["profile_id", "field", "value", "status", "note"])

# Disabling caching is deliberate: the source contents can change at the
# same path. Every execution verifies hashes and publishes to a fresh folder.
@step(enable_cache=False)
def verify_frozen_inputs(root: str, config: dict) -> dict:
    return features.verify_inputs(root, features.FeaturePreparationConfig.model_validate(config))

@step(enable_cache=False)
def prepare_system_identities(inputs: dict, out: str) -> tuple[Annotated[dict, "system_manifest"], Annotated[pd.DataFrame, "system_profiles"]]:
    result = features.build_systems(inputs, out)
    table = pd.DataFrame([{ "system_key": row["system_key"], **row["recorded"], **row["resolution"]}
                          for row in features.read_jsonl(result["path"])])
    return result, table

@step(enable_cache=False)
def prepare_model_profiles(inputs: dict, systems: dict, out: str) -> tuple[Annotated[dict, "model_manifest"], Annotated[pd.DataFrame, "model_facts"]]:
    from .feature_sources import supplement_models
    result = features.build_profiles(inputs, systems, out, "model")
    result = supplement_models(inputs["root"], result, out)
    return result, profile_table(result["path"])

@step(enable_cache=False)
def prepare_harness_and_run_profiles(inputs: dict, systems: dict, out: str) -> tuple[Annotated[dict, "harness_manifest"], Annotated[pd.DataFrame, "harness_facts"]]:
    result = features.build_profiles(inputs, systems, out, "harness")
    return result, profile_table(result["path"])

@step(enable_cache=False)
def prepare_task_profiles(inputs: dict, systems: dict, out: str) -> tuple[Annotated[dict, "task_manifest"], Annotated[pd.DataFrame, "task_preview"]]:
    from itertools import islice
    result = features.build_tasks(inputs, systems, out)
    table = pd.DataFrame([{ "task_key": row["task_key"], "text_preview": row["recorded"]["item_content"][:240],
                          **row["observed"], "assessment_status": row["assessment"]["status"]}
                         for row in islice(features.read_jsonl(result["path"]), 100)])
    return result, table

@step(enable_cache=False, experiment_tracker=TRACKER)
def validate_and_publish(inputs: dict, systems: dict, models: dict, harnesses: dict,
                         tasks: dict, collection: dict, out: str) -> tuple[Annotated[dict, "feature_summary"], Annotated[HTMLString, "coverage_report"]]:
    from zenml import get_step_context
    summary = features.finalize_bundle(inputs, systems, models, harnesses, tasks, out, collection=collection)
    mlflow.log_params({"dataset_revision": inputs["dataset_revision"], "input_manifest_sha256": inputs["manifest_sha256"],
                       "feature_pipeline_version": "feature_preparation_v1", "mode": "preparation_only",
                       "zenml_run_id": str(get_step_context().pipeline_run.id)})
    mlflow.set_tags({"training_performed": "false", "new_feature_training_ready": "false",
                     "source_eligibility": "research_only"})
    mlflow.log_metrics({"systems": systems["systems"], "model_profiles": models["profiles"],
                        "harness_profiles": harnesses["profiles"], "unique_task_inputs": tasks["tasks"],
                        **{f"records_{role}": count for role, count in tasks["records_by_role"].items()}})
    mlflow.log_metrics({"collection_requests": collection["collection_requests"], "annotation_candidates": collection["candidate_annotations"]})
    for name in ["summary.json", "feature_plan.json", "manifest.json", "report.html", "resolved_config.json",
                 "collection_summary.json", "collection_plan.json", "task_assessment.schema.json", "task_annotation_candidates.jsonl", "collection_review.html",
                 "source_collection_audit.json", "coverage_audit.json"]:
        mlflow.log_artifact(str(Path(out) / name), "preparation")
    active = mlflow.active_run()
    log_source(mlflow.MlflowClient(), active.info.run_id, Path(inputs["root"]))
    mlflow.log_dict(git_context(Path(inputs["root"])), "reproducibility/git.json")
    features.write_json(Path(out) / "tracking_receipt.json", {"mlflow_run_id": active.info.run_id,
                        "experiment_id": active.info.experiment_id,
                        "zenml_run_id": str(get_step_context().pipeline_run.id)})
    return summary, HTMLString((Path(out) / "report.html").read_text())

@step(enable_cache=False)
def prepare_collection_review(inputs: dict, systems: dict, models: dict, harnesses: dict,
                              tasks: dict, out: str) -> tuple[Annotated[dict, "collection_summary"], Annotated[pd.DataFrame, "collection_requests"], Annotated[pd.DataFrame, "annotation_candidates"], Annotated[HTMLString, "collection_review"]]:
    from .feature_collection import build_collection_bundle
    summary = build_collection_bundle(inputs, systems, models, harnesses, tasks, out)
    requests = pd.DataFrame([{k: r.get(k, "") for k in ["entity_type", "profile_id", "system_key", "field", "collection_status"]}
                             for r in features.read_jsonl(Path(out) / "collection_queue.jsonl")])
    annotations = pd.DataFrame([{"task_key": r["task_key"], "review_status": r["review_status"], "claims": json.dumps(r["claims"], ensure_ascii=False)}
                                for r in features.read_jsonl(Path(out) / "task_annotation_candidates.jsonl")])
    return summary, requests, annotations, HTMLString((Path(out) / "collection_review.html").read_text())


@pipeline(name="laya_feature_preparation", enable_cache=False)
def laya_feature_preparation(root: str, config: dict, out: str):
    inputs = verify_frozen_inputs(root, config)
    systems, _ = prepare_system_identities(inputs, out)
    models, _ = prepare_model_profiles(inputs, systems, out)
    harnesses, _ = prepare_harness_and_run_profiles(inputs, systems, out)
    tasks, _ = prepare_task_profiles(inputs, systems, out)
    collection, _, _, _ = prepare_collection_review(inputs, systems, models, harnesses, tasks, out)
    validate_and_publish(inputs, systems, models, harnesses, tasks, collection, out)
