"""CPU token preparation through ZenML; the native environment owns tokenization."""
import json
from pathlib import Path
import subprocess

import mlflow
from zenml import get_step_context, pipeline, step

from .feature_preparation import write_json
from .grouped_data import GroupedTrainingConfig, verify_prepared
from .zenml_pipeline import TRACKER


@step(enable_cache=False)
def prepare_grouped_inputs(root: str, config: dict, out: str) -> dict:
    root, out = Path(root), Path(out)
    recipe = GroupedTrainingConfig.model_validate_json((root / config["recipe"]).read_text())
    if (root / recipe.prepared).resolve() != (out / "tokens").resolve():
        raise ValueError("Grouped token output must be the recipe's workflow-output/tokens directory")
    with (out / "preparation.log").open("w") as log:
        subprocess.run([str(root / config["native_python"]), "-u", "scripts/train_grouped_enriched.py",
                        "--config", config["recipe"], "--prepare"], cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
    prepared, _ = verify_prepared(root, recipe.model_dump())
    return json.loads((prepared / "summary.json").read_text())


@step(enable_cache=False, experiment_tracker=TRACKER)
def publish_grouped_inputs(summary: dict, out: str) -> dict:
    mlflow.set_tags({"mode": "grouped_enriched_preparation", "training_launched": "false", "source_policy": "research_only"})
    mlflow.log_dict(summary, "summary.json")
    write_json(Path(out) / "summary.json", summary)
    write_json(Path(out) / "tracking_receipt.json", {"mlflow_run_id": mlflow.active_run().info.run_id,
        "zenml_run_id": str(get_step_context().pipeline_run.id)})
    return summary


@pipeline(name="laya_grouped_input_preparation", enable_cache=False)
def laya_grouped_input_preparation(root: str, config: dict, out: str):
    summary = prepare_grouped_inputs(root, config, out)
    publish_grouped_inputs(summary, out)
