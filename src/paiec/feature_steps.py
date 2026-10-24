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
    result = supplement_models(inputs["root"], result, out,
                               source_registry=inputs["config"].get("source_registry", "configs/pipelines/feature_sources_v1.json"))
    table = profile_table(result["path"])
    qualified = [{"profile_id": r["profile_id"], "field": "qualifications." + r["field"],
                  "value": json.dumps(r["value"]), "status": r["status"], "note": r["note"]}
                 for r in features.read_jsonl(Path(out) / "qualified_model_observations.jsonl")]
    if qualified:
        table = pd.concat([table, pd.DataFrame(qualified)], ignore_index=True)
    return result, table

@step(enable_cache=False)
def prepare_harness_and_run_profiles(inputs: dict, systems: dict, out: str) -> tuple[Annotated[dict, "harness_manifest"], Annotated[pd.DataFrame, "harness_facts"]]:
    from .feature_sources import supplement_harnesses
    result = features.build_profiles(inputs, systems, out, "harness")
    result = supplement_harnesses(inputs["root"], result, out,
        source_registry=inputs["config"].get("source_registry", "configs/pipelines/feature_sources_v1.json"))
    table = profile_table(result["path"])
    qualified = [{"profile_id": r["profile_id"], "field": "qualifications." + r["field"],
                  "value": json.dumps(r["value"]), "status": r["status"], "note": r["note"]}
                 for r in features.read_jsonl(Path(out) / "qualified_harness_observations.jsonl")]
    if qualified:
        table = pd.concat([table, pd.DataFrame(qualified)], ignore_index=True)
    return result, table

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
                         tasks: dict, collection: dict, annotations: dict, evidence_plan: dict, out: str) -> tuple[Annotated[dict, "feature_summary"], Annotated[HTMLString, "coverage_report"]]:
    from zenml import get_step_context
    if annotations.get("prepared"):
        coverage_path = Path(out) / "coverage_audit.json"
        coverage = json.loads(coverage_path.read_text())
        coverage["task_annotations"].update(legacy_candidates=coverage["task_annotations"]["candidates"],
            candidates=annotations["candidate_annotations"], prepared_requests=annotations["tasks"],
            response_coverage_complete=annotations["response_coverage_complete"], semantic_quality_established=False)
        features.write_json(coverage_path, coverage)
    summary = features.finalize_bundle(inputs, systems, models, harnesses, tasks, out, collection=collection, annotations=annotations, evidence_plan=evidence_plan)
    mlflow.log_params({"dataset_revision": inputs["dataset_revision"], "input_manifest_sha256": inputs["manifest_sha256"],
                       "feature_pipeline_version": "feature_preparation_v1", "mode": "preparation_only",
                       "zenml_run_id": str(get_step_context().pipeline_run.id)})
    mlflow.set_tags({"training_performed": "false", "new_feature_training_ready": "false",
                     "source_eligibility": "research_only"})
    mlflow.log_metrics({"systems": systems["systems"], "model_profiles": models["profiles"],
                        "harness_profiles": harnesses["profiles"], "unique_task_inputs": tasks["tasks"],
                        **{f"records_{role}": count for role, count in tasks["records_by_role"].items()}})
    mlflow.log_metrics({"collection_requests": collection["collection_requests"], "annotation_candidates": collection["candidate_annotations"]})
    if annotations.get("prepared"):
        mlflow.log_metrics({"annotation_requests_v2": annotations["tasks"], "annotation_candidates_v2": annotations["candidate_annotations"]})
        mlflow.log_artifacts(str(Path(out) / "task_annotations_v2"), "task_annotations_v2")
    if evidence_plan.get("prepared"):
        mlflow.log_metrics({"evidence_link_training_rows": evidence_plan["training_rows"],
                            "evidence_link_scopes": evidence_plan["visible_system_benchmark_scopes"]})
        mlflow.log_artifacts(str(Path(out) / "evidence_plan_v1"), "evidence_plan_v1")
    for name in ["summary.json", "feature_plan.json", "manifest.json", "report.html", "resolved_config.json",
                 "collection_summary.json", "collection_plan.json", "task_assessment.schema.json", "task_annotation_candidates.jsonl", "collection_review.html",
                 "source_collection_audit.json", "coverage_audit.json", "source_review_audit.json", "run_setting_review_audit.json",
                 "qualified_model_observations.jsonl", "source_review_status.jsonl", "run_setting_reviews.jsonl",
                 "supplemental_sources.jsonl", "supplemented_model_profiles.jsonl",
                 "harness_source_collection_audit.json", "qualified_harness_observations.jsonl",
                 "harness_supplemental_sources.jsonl", "supplemented_harness_profiles.jsonl",
                 "identity_review_audit.json", "identity_review_status.jsonl", "reviewed_static_features.jsonl"]:
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


@step(enable_cache=False)
def prepare_task_annotation_requests(inputs: dict, tasks: dict, collection: dict, out: str) -> tuple[Annotated[dict, "annotation_plan"], Annotated[pd.DataFrame, "task_candidates_v2"], Annotated[HTMLString, "task_annotation_review"]]:
    config = inputs["config"].get("task_annotation_config")
    if not config:
        return {"prepared": False}, pd.DataFrame(columns=["task_key", "review_status"]), HTMLString("<p>Task annotation v2 not selected.</p>")
    from .task_annotation import prepare_pipeline_annotations
    result = prepare_pipeline_annotations(inputs["root"], tasks, out, config_path=config)
    table = pd.DataFrame([{"task_key": r["task_key"], "review_status": r["review_status"],
                           "features": json.dumps(r["feature_projection"], ensure_ascii=False)}
                          for r in features.read_jsonl(Path(out) / "task_annotations_v2/manual/candidates.jsonl")])
    return result, table, HTMLString((Path(out) / "task_annotations_v2/review.html").read_text())


@step(enable_cache=False)
def prepare_training_evidence_links(inputs: dict, tasks: dict, out: str) -> Annotated[dict, "training_evidence_plan"]:
    if not inputs["config"].get("prepare_evidence_links"):
        return {"prepared": False}
    from .feature_evidence_plan import prepare_evidence_plan
    from .head_training import file_hash
    task_path, example_path = Path(tasks["path"]), Path(out) / "examples.jsonl"
    result = prepare_evidence_plan(features.read_jsonl(task_path), features.read_jsonl(example_path),
                                   Path(out) / "evidence_plan_v1")
    result.update(prepared=True, task_table_sha256=file_hash(task_path),
                  example_links_sha256=file_hash(example_path), input_manifest_sha256=inputs["manifest_sha256"])
    # Keep hashes inside the child manifest as well as the final pipeline summary.
    child = Path(out) / "evidence_plan_v1"
    features.write_json(child / "summary.json", result)
    features.write_json(child / "manifest.json", {"artifacts": {p.name: file_hash(p) for p in sorted(child.iterdir()) if p.is_file() and p.name != "manifest.json"}})
    return result


@pipeline(name="laya_feature_preparation", enable_cache=False)
def laya_feature_preparation(root: str, config: dict, out: str):
    inputs = verify_frozen_inputs(root, config)
    systems, _ = prepare_system_identities(inputs, out)
    models, _ = prepare_model_profiles(inputs, systems, out)
    harnesses, _ = prepare_harness_and_run_profiles(inputs, systems, out)
    tasks, _ = prepare_task_profiles(inputs, systems, out)
    collection, _, _, _ = prepare_collection_review(inputs, systems, models, harnesses, tasks, out)
    annotations, _, _ = prepare_task_annotation_requests(inputs, tasks, collection, out)
    evidence_plan = prepare_training_evidence_links(inputs, tasks, out)
    validate_and_publish(inputs, systems, models, harnesses, tasks, collection, annotations, evidence_plan, out)
