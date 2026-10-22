"""Container-side whole-job execution. Called only by the Modal adapter."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile
from .jobs import ExecutionProfile, JobSpec, command
from .transport import unpack, sha256


def execute(
    job_id,
    job_dict,
    profile_dict,
    provenance,
    *,
    storage=Path("/jobs"),
    source=Path("/opt/project"),
):
    import re

    if not re.fullmatch(r"job_[0-9a-f]{32}", job_id):
        raise ValueError("Invalid job ID")
    profile = ExecutionProfile.model_validate(profile_dict)
    job = JobSpec.model_validate(job_dict)
    if profile.executor != "modal":
        raise ValueError("Modal worker requires a Modal profile")
    base = storage / job_id
    work = base / "work"
    shutil.copytree(source, work)
    unpack(base / "inputs.zip", work)
    for path, expected in json.loads(
        (work / "input_manifest.json").read_text()
    ).items():
        if sha256(work / path) != expected:
            raise ValueError(f"Input hash mismatch: {path}")
    env = dict(
        os.environ,
        PYTHONPATH=str(work / "src"),
        PAIEC_EXECUTION_METADATA=json.dumps(
            dict(executor="modal", profile=profile.model_dump(), source=provenance)
        ),
    )
    cmd = command(job, profile, sys.executable)
    metadata = dict(
        job=job.model_dump(),
        profile=profile.model_dump(),
        provenance=provenance,
        command=cmd,
    )
    (base / "execution.json").write_text(json.dumps(metadata, indent=2))
    # A new root isolates data, MLflow SQLite and artifacts from other jobs.
    with (base / "console.log").open("w") as log:
        try:
            result = subprocess.run(
                cmd,
                cwd=work,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=profile.timeout - 30,
            )
            return_code = result.returncode
        except subprocess.TimeoutExpired:
            log.write("\nJob timed out; partial output retained.\n")
            return_code = 124
    records = []
    store = work / "runs/mlflow/mlflow.db"
    if store.exists():
        from mlflow import MlflowClient

        client = MlflowClient(tracking_uri="sqlite:///" + str(store))
        for experiment in client.search_experiments():
            for run in client.search_runs([experiment.experiment_id]):
                from urllib.parse import urlparse, unquote

                artifact_root = Path(unquote(urlparse(run.info.artifact_uri).path))
                records.append(
                    dict(
                        run_id=run.info.run_id,
                        name=run.data.tags.get("mlflow.runName", "run"),
                        params=run.data.params,
                        metrics=run.data.metrics,
                        status=run.info.status,
                    )
                )
                if artifact_root.exists():
                    shutil.copytree(artifact_root, base / "tracking" / run.info.run_id)
    (base / "remote_tracking.json").write_text(json.dumps(records, indent=2))
    (base / "status.json").write_text(json.dumps({"return_code": return_code}))
    with zipfile.ZipFile(
        base / "results.zip", "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for name in [
            "execution.json",
            "console.log",
            "status.json",
            "remote_tracking.json",
        ]:
            archive.write(base / name, name)
        for origin, prefix in [
            (base / "tracking", "tracking"),
            (work / job.out, "output"),
        ]:
            if origin.exists():
                for path in origin.rglob("*"):
                    if (
                        path.is_file()
                        and "checkpoint" not in path.relative_to(origin).parts
                    ):
                        archive.write(
                            path, str(Path(prefix) / path.relative_to(origin))
                        )
    return {
        "job_id": job_id,
        "return_code": return_code,
        "results_path": f"{job_id}/results.zip",
        "checkpoint_directory": f"{job_id}/work/{job.out}/checkpoint",
    }
