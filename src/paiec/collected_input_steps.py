"""Input preparation and native-token audit within the ZenML workflow family."""
import json
from pathlib import Path
import subprocess

import mlflow
from zenml import pipeline, step

from .collected_inputs import prepare_collected_inputs
from .feature_preparation import write_json
from .head_training import file_hash
from .zenml_pipeline import TRACKER


@step(enable_cache=False)
def compile_collected_inputs(root: str, config: dict, out: str) -> dict:
    if config["version"] == "collected_inputs_v3":
        from .collected_inputs_v3 import prepare_collected_inputs_v3
        return prepare_collected_inputs_v3(root, config, out)
    if config["version"] == "collected_inputs_v2":
        from .collected_inputs_v2 import prepare_collected_inputs_v2
        return prepare_collected_inputs_v2(root, config, out)
    return prepare_collected_inputs(root, config, out)


@step(enable_cache=False)
def check_native_input_visibility(root: str, config: dict, prepared: dict, out: str) -> dict:
    root, folder = Path(root), Path(out)
    with (folder / "native_audit.log").open("w") as log:
        subprocess.run([str(root / config["native_python"]), str(root / "scripts/check_laya_inputs.py"),
            "--prepared", str(folder.relative_to(root)), "--out", str((folder / "native_audit").relative_to(root))],
            cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
    return json.loads((folder / "native_audit/summary.json").read_text())


@step(enable_cache=False, experiment_tracker=TRACKER)
def publish_laya_input_audit(prepared: dict, audit: dict, out: str) -> dict:
    folder = Path(out)
    if file_hash(folder / "input_manifest.json") != audit["input_manifest_sha256"]:
        raise ValueError("Input contract changed during native audit")
    summary = {**prepared, "native_token_audit_completed": True, "native_schema_valid": audit["native_schema_valid"],
               "full_visibility_at_requested_limit": audit["full_visibility_at_requested_limit"], "training_ready": False}
    write_json(folder / "summary.json", summary)
    write_json(folder / "manifest.json", {"artifacts": {str(p.relative_to(folder)): file_hash(p)
        for p in sorted(folder.rglob("*")) if p.is_file() and p != folder / "manifest.json"}})
    mlflow.set_tags({"mode": "laya_input_preparation", "source_policy": "research_only",
                     "training_ready": "false", "model_inference": "false", "revealed_examples": "false"})
    mlflow.log_metrics({"training_rows": prepared["records_by_role"]["training"],
        "collected_training_overflow_rows": audit["results"]["collected/training"]["overflow_rows"]})
    mlflow.log_artifacts(out, "laya_inputs")
    write_json(folder / "tracking_receipt.json", {"mlflow_run_id": mlflow.active_run().info.run_id})
    return summary


@pipeline(name="laya_collected_input_preparation", enable_cache=False)
def laya_collected_input_preparation(root: str, config: dict, out: str):
    prepared = compile_collected_inputs(root, config, out)
    audit = check_native_input_visibility(root, config, prepared, out)
    publish_laya_input_audit(prepared, audit, out)
