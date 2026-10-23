"""File transport and result import. No Modal or model framework dependency."""

import hashlib
import json
from pathlib import Path
import zipfile
from .jobs import relative_path


def sha256(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def pack_inputs(root, files, destination):
    manifest = {str(path): sha256(root / path) for path in files}
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in files:
            bundle.write(root / path, str(path))
        bundle.writestr("input_manifest.json", json.dumps(manifest, indent=2))
    return manifest


def unpack(bundle, destination):
    """Reject traversal and symlink zip members before extracting anything."""
    with zipfile.ZipFile(bundle) as archive:
        for info in archive.infolist():
            relative_path(info.filename)
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Symlinks are not supported in job bundles")
        archive.extractall(destination)


def import_tracking(root, directory):
    """Re-log cloud runs into the local store; never copy cloud SQLite paths."""
    from paiec.tracking import ExperimentTracker

    receipt = directory / "local_tracking_receipt.json"
    if receipt.exists():
        return json.loads(receipt.read_text())
    summary = directory / "remote_tracking.json"
    if not summary.exists():
        return {}
    tracker = ExperimentTracker(root, experiment="PAIEC Modal")
    mapping = {}
    for record in json.loads(summary.read_text()):
        with tracker.run(
            "Modal / " + record["name"],
            record["params"],
            record["params"],
            {"executor": "modal", "remote_run_id": record["run_id"]},
        ) as run:
            if run:
                for key, value in record["metrics"].items():
                    history = record.get("metric_history", {}).get(key, [])
                    if history:
                        for point in history:
                            run.client.log_metric(run.run_id, key, point["value"],
                                                  timestamp=point["timestamp"], step=point["step"])
                    else:
                        run.client.log_metric(run.run_id, key, value)
                artifacts = directory / "tracking" / record["run_id"]
                if artifacts.exists():
                    run.client.log_artifacts(run.run_id, str(artifacts))
                mapping[record["run_id"]] = run.run_id
        if run and record["status"] != "FINISHED":
            run.client.set_terminated(run.run_id, "KILLED" if record["status"] == "KILLED" else "FAILED")
    receipt.write_text(json.dumps(mapping, indent=2))
    return mapping
