"""Verified native-job orchestration without importing a model framework."""
import json
from pathlib import Path
import subprocess
from typing import Literal

from pydantic import BaseModel, ConfigDict

from .execution.jobs import ExecutionProfile, JobSpec, command, input_files
from .feature_preparation import local_path, write_json
from .head_training import file_hash


class TrainingWorkflowConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    version: Literal["laya_execution_workflow_v1"] = "laya_execution_workflow_v1"
    profile: str
    job: JobSpec


def plan_native_job(root, config, out):
    root, out = Path(root).resolve(), Path(out).resolve()
    config = TrainingWorkflowConfig.model_validate(config)
    profiles = json.loads((root / "configs/execution/profiles.json").read_text())
    profile = ExecutionProfile.model_validate(profiles[config.profile])
    job = JobSpec.model_validate(dict(config.job.model_dump(), out=str((out / "job").relative_to(root))))
    local_path(root, job.out)
    if (root / job.out).exists():
        raise FileExistsError("Native job output already exists; stopped jobs are never resumed")
    files = input_files(root, job)
    python = str(root / profile.python) if profile.executor == "local" else "python"
    inputs = {str(path): file_hash(root / path) for path in files}
    # Bind worker source as well as data: native environments remain separate.
    sources = {str(p.relative_to(root)): file_hash(p) for folder in ["src", "scripts", "configs", "requirements"]
               for p in sorted((root / folder).rglob("*")) if p.is_file() and p.suffix in {".py", ".json", ".lock"}}
    scorer = {}
    if job.task == "predict-grouped":
        from .grouped_evaluation import GroupedEvaluationConfig
        from .grouped_data import checked_artifacts
        c = GroupedEvaluationConfig.model_validate_json((root / job.grouped_prediction_config).read_text())
        path = root / c.prepared
        manifest = json.loads((path / "scoring_manifest.json").read_text())
        checked_artifacts(path, manifest["artifacts"])
        scorer = {str(p.relative_to(root)): file_hash(p) for p in
                  [path / "scoring_manifest.json", *[path / n for n in manifest["artifacts"]]]}
    return {"root": str(root), "profile": profile.model_dump(), "job": job.model_dump(),
            "command": command(job, profile, python), "input_sha256": inputs, "source_sha256": sources,
            "local_scorer_sha256": scorer, "execution_started": False, "resumes_existing_job": False}


def dispatch_native_job(plan, out):
    root = Path(plan["root"])
    for name, digest in {**plan["input_sha256"], **plan["source_sha256"], **plan.get("local_scorer_sha256", {})}.items():
        if file_hash(root / name) != digest:
            raise ValueError("Native job input/source changed after planning: " + name)
    profile, job = ExecutionProfile.model_validate(plan["profile"]), JobSpec.model_validate(plan["job"])
    if (root / job.out).exists():
        raise FileExistsError("Native job output already exists")
    if profile.executor == "modal":
        # Keep the SDK in its authenticated client environment rather than adding
        # cloud dependencies to the ZenML or native model environments.
        client = root / ".venv-modal/bin/python"
        if not client.is_file():
            raise RuntimeError("Install the separate Modal client environment")
        plan_path = Path(out) / "modal_client_plan.json"
        write_json(plan_path, plan)
        with (Path(out) / "native_job.log").open("w") as log:
            subprocess.run([str(client), "-u", "-m", "paiec.execution.modal_client", "--plan", str(plan_path)],
                           cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
        result = json.loads((root / job.out / "modal_receipt.json").read_text())
    else:
        with (Path(out) / "native_job.log").open("w") as log:
            subprocess.run(plan["command"], cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
        result = {"return_code": 0}
    if not (root / job.out).is_dir():
        raise ValueError("Native job returned without its expected output directory")
    report = {"version": "laya_execution_result_v1", "task": job.task, "job_output": job.out,
              "executor": profile.executor, "result": result, "execution_completed": True,
              "release_approved": False, "hosted_competition_compatibility_verified": False}
    write_json(Path(out) / "summary.json", report)
    return report
