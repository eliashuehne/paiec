#!/usr/bin/env python3
"""Recover missing files for an exact grouped checkpoint, without allocating a GPU."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from paiec.execution.jobs import relative_path
from paiec.grouped_checkpoint import verify_grouped_checkpoint
from paiec.head_training import file_hash
from paiec.grouped_evaluation import GroupedEvaluationConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = GroupedEvaluationConfig.model_validate_json((ROOT / args.config).read_text())
    folder = ROOT / config.training_output / "output/final"
    path = folder / "paiec_training.json"
    if file_hash(path) != config.checkpoint_manifest_sha256:
        raise ValueError("Recovered checkpoint manifest differs from the declared fitted model")
    manifest = json.loads(path.read_text())
    import modal
    volume = modal.Volume.from_name("paiec-jobs")
    fetched = []
    for name, digest in manifest["artifact_sha256"].items():
        relative_path(name)
        destination = (folder / name).resolve()
        if not destination.is_relative_to(folder.resolve()):
            raise ValueError("Checkpoint artifact escaped its folder")
        if destination.is_file():
            if file_hash(destination) != digest:
                raise ValueError("Existing checkpoint file changed: " + name)
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".download")
        with temporary.open("xb") as stream:
            remote = config.training_job_id + "/work/" + config.training_output + "/final/" + name
            for chunk in volume.read_file(remote):
                stream.write(chunk)
        if file_hash(temporary) != digest:
            raise ValueError("Downloaded checkpoint hash differs: " + name)
        temporary.rename(destination)
        fetched.append(name)
    verify_grouped_checkpoint(folder)
    print(json.dumps({"checkpoint": str(folder), "fetched": fetched, "all_hashes_verified": True, "gpu_allocated": False}), flush=True)


if __name__ == "__main__":
    main()
