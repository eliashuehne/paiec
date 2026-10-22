#!/usr/bin/env python3
"""Choose job placement without changing model, data or scoring implementation."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from paiec.execution.jobs import ExecutionProfile, JobSpec, command, input_files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="local-mlx")
    parser.add_argument(
        "--task", choices=["evaluate", "train-head-smoke"], required=True
    )
    parser.add_argument("--suite", default="smoke")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--state-format", choices=["json", "text_v1", "context_v1"], default="json"
    )
    parser.add_argument("--checkpoint")
    parser.add_argument("--out", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    profiles = json.loads((ROOT / "configs/execution/profiles.json").read_text())
    if args.profile not in profiles:
        parser.error("Unknown execution profile")
    profile = ExecutionProfile.model_validate(profiles[args.profile])
    job = JobSpec(
        task=args.task,
        out=args.out,
        suite=args.suite,
        seed=args.seed,
        state_format=args.state_format,
        checkpoint=args.checkpoint,
    )
    python = str(ROOT / profile.python) if profile.executor == "local" else "python"
    cmd = command(job, profile, python)
    if (ROOT / job.out).exists():
        raise FileExistsError("Use a new output directory")
    files = input_files(ROOT, job)
    plan = dict(
        profile=profile.model_dump(),
        job=job.model_dump(),
        command=cmd,
        input_files=[str(path) for path in files],
        input_bytes=sum((ROOT / path).stat().st_size for path in files),
        creates_cloud_resources=profile.executor == "modal" and not args.dry_run,
    )
    print(json.dumps(plan, indent=2), flush=True)
    if args.dry_run:
        return
    if profile.executor == "modal":
        from paiec.execution.modal_adapter import launch

        launch(ROOT, job, profile, files)
    else:
        subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
