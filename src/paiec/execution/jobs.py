"""Validated job plans shared by local and Modal launchers. No framework imports."""

from pathlib import Path
from typing import Literal
import json
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ExecutionProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    executor: Literal["local", "modal"]
    backend: Literal["torch", "mlx"]
    device: Literal["mps", "cuda", "cpu"]
    python: str = "python"
    gpu: Literal["L4", "A10G", "T4", "A100-40GB"] = "L4"
    timeout: int = 900

    @model_validator(mode="after")
    def compatible(self):
        if self.executor == "modal" and (
            self.backend != "torch" or self.device != "cuda"
        ):
            raise ValueError("Modal uses PyTorch/CUDA; MLX requires Apple Silicon")
        if self.backend == "mlx" and self.device == "cuda":
            raise ValueError("MLX cannot use CUDA")
        if not 60 <= self.timeout <= 86400:
            raise ValueError("Job timeout must be between 60 seconds and 24 hours")
        return self


class JobSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task: Literal["evaluate", "train-head-smoke", "train-head-pilot"]
    out: str
    suite: str = "smoke"
    seed: int = 0
    state_format: Literal["json", "text_v1", "context_v1", "enriched_v1", "task_v2", "task_enriched_v2", "compact_v3", "compact_enriched_v3"] | None = None
    evidence_mode: Literal["none", "beta_mean"] = "none"
    evidence_strength: float = Field(default=5., gt=0, allow_inf_nan=False)
    checkpoint: str | None = None
    pilot_config: str = "configs/head_finetuning/enrichment_pilot_v1.json"
    variant: Literal["enriched", "baseline", "both"] = "enriched"
    encoder_cache_gib: int = Field(default=0, ge=0, le=16)

    @model_validator(mode="after")
    def paths(self):
        relative_path(self.out)
        if not self.out.startswith("runs/"):
            raise ValueError("Output must be inside runs/")
        if self.checkpoint:
            relative_path(self.checkpoint)
        relative_path(self.pilot_config)
        if self.encoder_cache_gib and self.task != "train-head-pilot":
            raise ValueError("Encoder cache is a head-pilot option")
        if self.evidence_mode != "none" and self.task != "evaluate":
            raise ValueError("Evidence updating is an evaluation option")
        if self.task == "train-head-pilot" and (self.state_format or self.checkpoint):
            raise ValueError("Pilot inputs and original parent come from its verified preparation")
        if self.task == "train-head-smoke" and (
            self.state_format not in {None, "json"} or self.checkpoint
        ):
            raise ValueError(
                "Smoke trainer continues the pinned JSON recipe; use a new pilot for changed inputs/parents"
            )
        if self.state_format in {"enriched_v1", "task_enriched_v2", "compact_v3", "compact_enriched_v3"} and not self.checkpoint:
            raise ValueError("Execution jobs use the trained checkpoint's bundled research profiles")
        return self


def relative_path(value):
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not path.parts or str(path) == ".":
        raise ValueError("Expected a repository-relative path without traversal")
    return path


def command(job, profile, python):
    if job.task == "train-head-pilot":
        if profile.backend != "torch" or profile.device not in {"mps", "cuda"}:
            raise ValueError("Head pilot requires PyTorch MPS or CUDA")
        args = [python, "scripts/run_head_pilot.py", "--config", job.pilot_config,
                "--variant", job.variant, "--device", profile.device,
                "--out", job.out, "--train"]
        if profile.executor == "modal":
            args.append("--download-parent")
        if job.encoder_cache_gib:
            if profile.device != "cuda":
                raise ValueError("Pilot encoder cache requires CUDA")
            args.extend(["--encoder-cache-gib", str(job.encoder_cache_gib)])
        return args
    if job.task == "train-head-smoke":
        if profile.backend != "torch":
            raise ValueError(
                "Training currently requires PyTorch; choose local-torch or modal-l4"
            )
        return [
            python,
            "scripts/train_head_smoke.py",
            "--out",
            job.out,
            "--device",
            profile.device,
        ]
    args = [
        python,
        "scripts/run_eval.py",
        "--suite",
        job.suite,
        "--predictor",
        "laya",
        "--laya-backend",
        profile.backend,
        "--laya-device",
        profile.device,
        "--laya-dtype",
        "float32" if profile.backend == "mlx" else "auto",
        "--seed",
        str(job.seed),
        "--out",
        job.out,
    ]
    if job.state_format:
        args.extend(["--laya-state-format", job.state_format])
    if job.checkpoint:
        args.extend(["--laya-checkpoint", job.checkpoint])
    if job.evidence_mode != "none":
        args.extend(["--laya-evidence", job.evidence_mode, "--laya-evidence-strength", str(job.evidence_strength)])
    return args


def input_files(root, job):
    """Explicit file inventory; never upload trace tables, credentials or whole runs."""
    files = []
    if job.task == "train-head-pilot":
        from paiec.head_pilot import load_config, verify_prepared
        from paiec.head_training import file_hash

        config = load_config(root / job.pilot_config)
        if config.seed != job.seed:
            raise ValueError("Pilot seed is frozen in the prepared configuration")
        prepared, manifest = verify_prepared(root, config)
        files += [Path(job.pilot_config), relative_path(config.split),
                  relative_path(config.prepared) / "manifest.json"]
        # Include every manifest-pinned artifact so remote verification is the
        # same as local verification, even when training just one variant.
        files += [relative_path(config.prepared) / relative_path(name)
                  for name in manifest["artifacts"]]
        split = json.loads((root / config.split).read_text())
        original = relative_path(split["artifact_directory"])
        for role in ["training", "development"]:
            path = original / (role + ".jsonl")
            if file_hash(root / path) != split["artifact_fingerprints"][role + ".jsonl"]:
                raise ValueError(f"Frozen {role} source changed")
            files.append(path)
    elif job.task == "train-head-smoke":
        config = json.loads(
            (root / "configs/head_finetuning/frozen_split_v1.json").read_text()
        )
        directory = relative_path(config["artifact_directory"])
        files += [directory / name for name in ["training.jsonl", "provenance.parquet"]]
    else:
        suites = json.loads((root / "configs/suites.json").read_text())
        if job.suite not in suites:
            raise ValueError("Unknown evaluation suite")
        files.append(Path("payload/data/corpus_manifest.json"))
        for benchmark in suites[job.suite]["benchmarks"]:
            relative_path(benchmark)
            files += [
                Path("payload/data") / benchmark / "formatted_tables" / name
                for name in ["subjects.parquet", "items.parquet", "response.parquet"]
            ]
    if job.checkpoint:
        from paiec.laya_runtime import local_checkpoint_files

        folder, _ = local_checkpoint_files(str(root / job.checkpoint), "english")
        files += [p.relative_to(root) for p in folder.rglob("*") if p.is_file()]
    for path in files:
        actual = (root / path).resolve()
        if not actual.is_relative_to(root.resolve()) or not actual.is_file():
            raise ValueError(f"Missing or external input: {path}")
    return sorted(set(files))
