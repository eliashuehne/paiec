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
        "--task", choices=["evaluate", "train-head-smoke", "train-head-pilot"], required=True
    )
    parser.add_argument("--suite", default="smoke")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--state-format", choices=["json", "text_v1", "context_v1", "enriched_v1", "task_v2", "task_enriched_v2", "compact_v3", "compact_enriched_v3"], default=None
    )
    parser.add_argument("--checkpoint")
    parser.add_argument("--evidence-mode", choices=["none", "beta_mean"], default="none")
    parser.add_argument("--evidence-strength", type=float, default=5.)
    parser.add_argument("--encoder-cache-gib", type=int, default=0)
    parser.add_argument("--pilot-config", default="configs/head_finetuning/enrichment_pilot_v1.json")
    parser.add_argument("--variant", choices=["enriched", "baseline", "both"], default="enriched",
                        help="head pilot selection; enriched runs alone by default")
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
        evidence_mode=args.evidence_mode, evidence_strength=args.evidence_strength,
        pilot_config=args.pilot_config,
        variant=args.variant,
        encoder_cache_gib=args.encoder_cache_gib,
    )
    python = str(ROOT / profile.python) if profile.executor == "local" else "python"
    cmd = command(job, profile, python)
    if (ROOT / job.out).exists():
        raise FileExistsError("Use a new output directory")
    files = input_files(ROOT, job)
    pilot = None
    if job.task == "train-head-pilot":
        from paiec.head_pilot import load_config
        config = load_config(ROOT / job.pilot_config)
        pilot = dict(config=config.model_dump(), variants=["baseline", "enriched"] if job.variant == "both" else [job.variant],
                     optimizer_updates_per_variant=config.updates,
                     device_override=profile.device, preparation_verified=True,
                     calibration_opened=False, confirmation_opened=False,
                     resumes_local_training=False)
    plan = dict(
        profile=profile.model_dump(),
        job=job.model_dump(),
        command=cmd,
        input_files=[str(path) for path in files],
        input_bytes=sum((ROOT / path).stat().st_size for path in files),
        creates_cloud_resources=profile.executor == "modal" and not args.dry_run,
        pilot=pilot,
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
