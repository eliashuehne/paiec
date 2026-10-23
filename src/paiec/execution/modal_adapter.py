"""Optional Modal SDK boundary. Importing normal predictors never imports it."""

import json
from pathlib import Path
import tempfile
import uuid
from .transport import pack_inputs, unpack, import_tracking, sha256

VOLUME_NAME = "paiec-jobs"
CACHE_NAME = "paiec-model-cache"


def create_app(root, profile):
    import modal

    image = (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("git")
        .pip_install_from_requirements(str(root / "requirements/modal-worker.lock"))
        .env(
            {
                "HF_HOME": "/cache/huggingface",
                "MLFLOW_DISABLE_TELEMETRY": "true",
                "PYTHONPATH": "/opt/project/src",
            }
        )
    )
    for folder in ["src", "scripts", "configs"]:
        image = image.add_local_dir(
            root / folder,
            f"/opt/project/{folder}",
            copy=True,
            ignore=lambda path: "__pycache__" in path.parts or path.suffix == ".pyc",
        )
    image = image.add_local_file(
        root / "pyproject.toml", "/opt/project/pyproject.toml", copy=True
    )
    image = image.add_local_file(
        root / "requirements/modal-worker.lock",
        "/opt/project/requirements/modal-worker.lock",
        copy=True,
    )
    app = modal.App("paiec-laya")
    volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
    cache = modal.Volume.from_name(CACHE_NAME, create_if_missing=True)

    @app.function(
        image=image,
        gpu=profile.gpu,
        cpu=2,
        memory=16384,
        timeout=profile.timeout,
        volumes={"/jobs": volume, "/cache": cache},
        retries=0,
        max_containers=1,
        serialized=True,
    )
    def worker(job_id, job, execution_profile, provenance):
        from paiec.execution.modal_worker import execute

        try:
            return execute(job_id, job, execution_profile, provenance)
        finally:
            # Preserve partial logs/files even when training or packaging fails.
            volume.commit()
            cache.commit()

    return app, worker, volume


def launch(root, job, profile, files):
    import modal
    from paiec.tracking import git_context

    job_id = "job_" + uuid.uuid4().hex
    app, worker, volume = create_app(root, profile)
    with tempfile.TemporaryDirectory() as directory:
        bundle = Path(directory) / "inputs.zip"
        manifest = pack_inputs(root, files, bundle)
        sources = [
            p
            for folder in ["src", "scripts", "configs"]
            for p in (root / folder).rglob("*")
            if p.is_file() and p.suffix in {".py", ".json"}
        ]
        sources += [root / "pyproject.toml", root / "requirements/modal-worker.lock"]
        provenance = {
            "git": git_context(root),
            "input_sha256": manifest,
            "bundle_sha256": sha256(bundle),
            "source_sha256": {str(p.relative_to(root)): sha256(p) for p in sources},
        }
        receipt = root / "runs/modal_jobs" / f"{job_id}.json"
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text(
            json.dumps(
                dict(
                    job_id=job_id,
                    job=job.model_dump(),
                    profile=profile.model_dump(),
                    volume=VOLUME_NAME,
                    status="prepared",
                ),
                indent=2,
            )
        )
        with modal.enable_output(), app.run(detach=True):
            with volume.batch_upload() as upload:
                upload.put_file(bundle, f"/{job_id}/inputs.zip")
            print(f"Modal job ID: {job_id}", flush=True)
            call = worker.spawn(
                job_id, job.model_dump(), profile.model_dump(), provenance
            )
            submitted = json.loads(receipt.read_text())
            submitted.update(status="submitted", app_id=app.app_id,
                             function_call_id=call.object_id, detached=True)
            receipt.write_text(json.dumps(submitted, indent=2))
            print(f"Detached function call: {call.object_id}", flush=True)
            result = call.get()
        destination = root / job.out
        fetch(volume, result["results_path"], destination)
        import_tracking(root, destination)
        (destination / "modal_receipt.json").write_text(json.dumps(result, indent=2))
        if result["return_code"]:
            raise RuntimeError(f"Modal job failed; see {destination}/console.log")
        return result


def fetch(volume, path, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory() as directory:
        bundle = Path(directory) / "results.zip"
        with bundle.open("wb") as file:
            for chunk in volume.read_file(path):
                file.write(chunk)
        unpack(bundle, destination)
