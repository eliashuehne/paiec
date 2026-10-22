"""Validated job plans shared by local and Modal launchers. No framework imports."""

from pathlib import Path
from typing import Literal
import json
from pydantic import BaseModel, ConfigDict, model_validator


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
        if not 60 <= self.timeout <= 3600:
            raise ValueError("Job timeout must be between 60 and 3600 seconds")
        return self


class JobSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task: Literal["evaluate", "train-head-smoke"]
    out: str
    suite: str = "smoke"
    seed: int = 0
    state_format: Literal["json", "text_v1", "context_v1"] = "json"
    checkpoint: str | None = None

    @model_validator(mode="after")
    def paths(self):
        relative_path(self.out)
        if not self.out.startswith("runs/"):
            raise ValueError("Output must be inside runs/")
        if self.checkpoint:
            relative_path(self.checkpoint)
        if self.task == "train-head-smoke" and (
            self.state_format != "json" or self.checkpoint
        ):
            raise ValueError(
                "Smoke trainer continues the pinned JSON recipe; use a new pilot for changed inputs/parents"
            )
        return self


def relative_path(value):
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not path.parts or str(path) == ".":
        raise ValueError("Expected a repository-relative path without traversal")
    return path


def command(job, profile, python):
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
        "--laya-state-format",
        job.state_format,
        "--seed",
        str(job.seed),
        "--out",
        job.out,
    ]
    if job.checkpoint:
        args.extend(["--laya-checkpoint", job.checkpoint])
    return args


def input_files(root, job):
    """Explicit file inventory; never upload trace tables, credentials or whole runs."""
    files = []
    if job.task == "train-head-smoke":
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
