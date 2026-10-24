#!/usr/bin/env python3
"""Finish one already-authorized grouped pilot; never start/restart training.

Wait for the declared existing training job, recover its exact checkpoint, then
execute exactly one checkpoint-bound development inference/scoring workflow.
STOP_AFTER_TRAINING in the training workflow folder cancels deferred stages.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from paiec.feature_preparation import write_json
from paiec.grouped_evaluation import GroupedEvaluationConfig
from paiec.head_training import file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-launch", required=True)
    parser.add_argument("--training-job-id", required=True)
    parser.add_argument("--out", default="runs/workflows/grouped_pilot_completion_v1")
    args = parser.parse_args()
    launch = json.loads((ROOT / args.training_launch).read_text())
    training_out = ROOT / launch["out"]
    job_out = training_out / "job"
    # Validation rejects non-job IDs and traversal before any cloud operation.
    pending = GroupedEvaluationConfig(training_job_id=args.training_job_id, checkpoint_manifest_sha256="0" * 64,
                                     training_output=str(job_out.relative_to(ROOT)))
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    env = dict(os.environ, MLFLOW_DISABLE_AGENT_HINT="1", PYTHONUNBUFFERED="1")
    stopped_apps = set()
    evaluation_out = ROOT / "runs/workflows/grouped_development_v1"
    evaluation_config = ROOT / "configs/head_finetuning/grouped_development_v1.json"
    workflow_config = ROOT / "configs/pipelines/grouped_development_workflow_v1.json"

    def status(phase, **values):
        record = {"phase": phase, "updated_at": datetime.now(timezone.utc).isoformat(),
                  "training_job_id": pending.training_job_id, **values}
        write_json(out / "status.json", record)
        print(json.dumps(record), flush=True)

    def cancelled():
        if (training_out / "STOP_AFTER_TRAINING").exists() or (out / "CANCEL").exists():
            raise InterruptedError("Deferred stages canceled; completed training artifacts remain unchanged")

    def stop_app(receipt_path):
        if not receipt_path.is_file():
            return
        receipt = json.loads(receipt_path.read_text())
        app = receipt.get("app_id")
        if app and app not in stopped_apps:
            result = subprocess.run([str(ROOT / ".venv-modal/bin/python"), "-m", "modal", "app", "stop", app],
                                    cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)
            message = (result.stdout or "") + (result.stderr or "")
            # Modal returns 1 for this already-satisfied cleanup operation.
            # Keep genuine failures visible; do not block a successful fit on it.
            if result.returncode and "app is already stopped" not in message.casefold():
                raise subprocess.CalledProcessError(result.returncode, result.args, output=result.stdout, stderr=result.stderr)
            stopped_apps.add(app)
            write_json(out / "gpu_cleanup.json", {"stopped_apps": sorted(stopped_apps)})

    def evaluation_receipts():
        return [p for p in (ROOT / "runs/modal_jobs").glob("job_*.json")
                if json.loads(p.read_text()).get("job", {}).get("out") == str((evaluation_out / "job").relative_to(ROOT))]

    try:
        status("waiting_for_existing_training", launches_training=False)
        while not (training_out / "tracking_receipt.json").is_file():
            cancelled()
            if time.monotonic() - started > 15300:
                raise TimeoutError("Training did not finish within its declared execution bound")
            result = subprocess.run(["ps", "-p", str(launch["pid"]), "-o", "state="], capture_output=True, text=True)
            if result.returncode or not result.stdout.strip() or result.stdout.strip().startswith("Z"):
                raise RuntimeError("Training workflow ended before publishing completion; no follow-on GPU job will start")
            time.sleep(15)
        cancelled()
        receipt = json.loads((job_out / "modal_receipt.json").read_text())
        report = json.loads((job_out / "output/report.json").read_text())
        if receipt.get("job_id") != pending.training_job_id or receipt.get("return_code") != 0 or report.get("status") != "finished":
            raise ValueError("Existing training job did not complete successfully")
        checkpoint_manifest = job_out / "output/final/paiec_training.json"
        config = pending.model_copy(update={"checkpoint_manifest_sha256": file_hash(checkpoint_manifest)})
        if evaluation_out.exists() or evaluation_config.exists() or workflow_config.exists() or evaluation_receipts():
            raise FileExistsError("An evaluation declaration/output already exists; never launch a duplicate")
        write_json(evaluation_config, config.model_dump())
        write_json(workflow_config, {"version": "laya_execution_workflow_v1", "profile": "modal-l4-grouped-evaluation",
            "job": {"task": "predict-grouped", "out": "runs/workflows/grouped_development_v1/job", "seed": 0,
                    "grouped_prediction_config": str(evaluation_config.relative_to(ROOT))}})
        status("recovering_verified_checkpoint", checkpoint_manifest_sha256=config.checkpoint_manifest_sha256)
        subprocess.run([str(ROOT / ".venv-modal/bin/python"), "-u", "scripts/fetch_grouped_checkpoint.py", "--config",
                        str(evaluation_config.relative_to(ROOT))], cwd=ROOT, env=env, check=True, timeout=1200)
        stop_app(ROOT / "runs/modal_jobs" / (pending.training_job_id + ".json"))
        cancelled()
        status("original_development_inference_and_scoring", calibration_opened=False, confirmation_opened=False)
        with (out / "development_workflow.log").open("x") as stream:
            subprocess.run([str(ROOT / ".venv-pipeline/bin/python"), "-u", "scripts/run_pipeline.py", "training", "--config",
                str(workflow_config.relative_to(ROOT)), "--out", str(evaluation_out.relative_to(ROOT)), "--execute"],
                cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=1800)
        result = json.loads((evaluation_out / "scores/report.json").read_text())
        status("finished", development_report=str((evaluation_out / "scores/report.json").relative_to(ROOT)),
               brier_alc=result["brier_alc"], constant_brier_alc=result["constant_brier_alc"],
               mean_within_pair_auroc=result["mean_within_pair_auroc"], checkpoint_verified_locally=True)
    except BaseException as exc:
        status("canceled" if isinstance(exc, InterruptedError) else "failed", error=f"{type(exc).__name__}: {exc}",
               training_restarted=False, deferred_job_retried=False)
        raise
    finally:
        # Training is stopped only after successful completion, never during fit.
        if (job_out / "modal_receipt.json").is_file():
            stop_app(ROOT / "runs/modal_jobs" / (pending.training_job_id + ".json"))
        for receipt_path in evaluation_receipts():
            stop_app(receipt_path)


if __name__ == "__main__":
    main()
