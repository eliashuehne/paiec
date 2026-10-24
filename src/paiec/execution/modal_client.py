"""Separate SDK-process bridge for the framework-free ZenML environment."""
import argparse
import json
from pathlib import Path

from .jobs import JobSpec, ExecutionProfile
from .transport import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    root = Path(plan["root"])
    for name, digest in {**plan["input_sha256"], **plan["source_sha256"]}.items():
        if sha256(root / name) != digest:
            raise ValueError("Native plan changed before Modal dispatch")
    from .modal_adapter import launch
    launch(root, JobSpec.model_validate(plan["job"]), ExecutionProfile.model_validate(plan["profile"]),
           [Path(name) for name in plan["input_sha256"]])


if __name__ == "__main__":
    main()
