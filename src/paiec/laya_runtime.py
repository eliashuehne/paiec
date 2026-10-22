"""Interchangeable inference engines behind the shared Laya predictor.

Importing this module does not import torch or MLX. Both engines load the same
pinned upstream files by default. Local training artifacts require an explicit path.
"""

from __future__ import annotations

import gc
import hashlib
import json
from importlib.metadata import version
import os
from pathlib import Path
import platform
import time
from typing import Protocol

DEFAULT_CHECKPOINT_REVISION = "7b928d828b7b0e022f929d9bd2e44165aa270148"
MODEL_REPOSITORY = "convaiinnovations/laya"

# MLX keeps freed Metal buffers in a size-classed cache instead of returning them to
# the OS, and each new batch shape gets its own blocks. Its defaults (a 16.32 GB cache
# on a 16 GB Mac, above the 12.71 GB recommended working set) let a long run climb into
# swap. Cap both: the cache so eviction starts early, the memory limit so an oversized
# allocation raises instead of pushing the machine into swap death.
MLX_CACHE_LIMIT_BYTES = 2 * 1024**3
MLX_MEMORY_LIMIT_BYTES = 6 * 1024**3
MLX_CACHE_LIMIT_ENV = "PAIEC_MLX_CACHE_LIMIT_GB"
MLX_MEMORY_LIMIT_ENV = "PAIEC_MLX_MEMORY_LIMIT_GB"


def _env_gib(name: str, default: int) -> int:
    """Read a gibibyte override, refusing values that would disable the guard."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number of gibibytes, got {raw!r}") from exc
    if not value > 0:
        raise ValueError(f"{name} must be positive, got {raw!r}")
    return int(value * 1024**3)


class LayaRuntime(Protocol):
    metadata: dict

    def predict_batch(self, requests: list[dict], batch_size: int) -> list[dict]: ...

    def synchronize(self) -> None: ...

    def close(self) -> None: ...


def checkpoint_files(model: str, revision: str) -> tuple[Path, dict]:
    from huggingface_hub import snapshot_download

    prefix = "multilingual/" if model == "multilingual" else ""
    snapshot = Path(snapshot_download(
        MODEL_REPOSITORY, revision=revision,
        allow_patterns=[prefix + filename for filename in (
            "rl_agent_config.json", "model.safetensors", "encoder/config.json", "tokenizer/*",
        )],
    ))
    checkpoint = snapshot / prefix
    hashes = {}
    for name in ("model.safetensors", "rl_agent_config.json", "encoder/config.json", "tokenizer/tokenizer.json"):
        with (checkpoint / name).open("rb") as handle:
            hashes[name] = hashlib.file_digest(handle, "sha256").hexdigest()
    return checkpoint, {"checkpoint_repository": MODEL_REPOSITORY,
                        "checkpoint_revision": snapshot.name, "checkpoint_subfolder": prefix.rstrip("/"),
                        "checkpoint_sha256": hashes}


def local_checkpoint_files(path: str, model: str) -> tuple[Path, dict]:
    """Explicit project training artifacts; never replace the pinned default."""
    checkpoint = Path(path).expanduser().resolve()
    manifest = json.loads((checkpoint / "paiec_training.json").read_text())
    if manifest.get("model") != model:
        raise ValueError("Local checkpoint model differs from requested model")
    hashes = {}
    for name, expected in manifest["artifact_sha256"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid checkpoint artifact path")
        with (checkpoint / relative).open("rb") as handle:
            hashes[name] = hashlib.file_digest(handle, "sha256").hexdigest()
        if hashes[name] != expected:
            raise ValueError(f"Local checkpoint hash mismatch: {name}")
    required = {"model.safetensors", "rl_agent_config.json", "encoder/config.json", "tokenizer/tokenizer.json"}
    if not required <= hashes.keys():
        raise ValueError("Incomplete local checkpoint manifest")
    identity = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return checkpoint, {"checkpoint_repository": "local", "checkpoint_revision": "local-" + identity,
                        "checkpoint_path": str(checkpoint), "checkpoint_sha256": hashes,
                        "training_provenance": manifest}


class AgentRuntime:
    """Normalize the common Agent API to the request-list interface we use."""

    def __init__(self, agent, metadata: dict):
        self.agent = agent
        self.metadata = metadata

    def predict_batch(self, requests: list[dict], batch_size: int) -> list[dict]:
        if not requests:
            return []
        first = requests[0]
        if any(any(request[key] != first[key] for key in ("questions", "model", "max_len"))
               for request in requests):
            raise ValueError("A Laya runtime batch must share its model, questions, and context limit")
        if first["model"] != self.metadata["model"]:
            raise ValueError("Requested model differs from the loaded checkpoint")
        if self.agent is None:
            raise RuntimeError("Laya runtime has been closed")
        return self.agent.predict_batch(
            [request["state"] for request in requests], first["questions"],
            batch_size=batch_size, max_len=first["max_len"],
        )

    def synchronize(self) -> None:
        backend, device = self.metadata["backend"], self.metadata["device"]
        if backend == "mlx":
            import mlx.core as mx
            mx.synchronize()
        elif device == "mps":
            import torch
            torch.mps.synchronize()
        elif device == "cuda":
            import torch
            torch.cuda.synchronize()

    def trim_cache(self) -> None:
        """Release cached buffers between scoring pairs, keeping resident memory flat."""
        if self.agent is None:
            return
        self.synchronize()
        if self.metadata["backend"] == "mlx":
            import mlx.core as mx
            mx.clear_cache()
        elif self.metadata["device"] == "mps":
            import torch
            torch.mps.empty_cache()

    def close(self) -> None:
        self.synchronize()
        self.agent = None
        gc.collect()
        if self.metadata["backend"] == "mlx":
            import mlx.core as mx
            mx.clear_cache()
        elif self.metadata["device"] == "mps":
            import torch
            torch.mps.empty_cache()
        elif self.metadata["device"] == "cuda":
            import torch
            torch.cuda.empty_cache()


def create_runtime(*, backend: str, model: str, device: str, dtype: str,
                   revision: str | None, batch_size: int, checkpoint_path: str | None = None) -> LayaRuntime:
    if backend not in {"torch", "mlx"}:
        raise ValueError("backend must be torch or mlx")
    if dtype not in {"auto", "float16", "float32"}:
        raise ValueError("dtype must be auto, float16, or float32")
    if backend == "torch" and dtype != "auto":
        raise ValueError("Normal Laya manages precision automatically; use dtype='auto' with torch")
    if backend == "mlx" and device == "cuda":
        raise ValueError("MLX does not support CUDA; use auto, mps, or cpu")
    started = time.perf_counter()
    if backend == "mlx":
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            raise RuntimeError("The MLX backend requires an Apple Silicon Mac; select --laya-backend torch on other hosts")
        try:
            import laya_mlx
        except ImportError as exc:
            raise RuntimeError("MLX runtime is unavailable; use .venv-laya-mlx or install .[mlx]. See docs/laya.md") from exc
        import mlx.core as mx
        cache_limit = _env_gib(MLX_CACHE_LIMIT_ENV, MLX_CACHE_LIMIT_BYTES)
        memory_limit = _env_gib(MLX_MEMORY_LIMIT_ENV, MLX_MEMORY_LIMIT_BYTES)
        mx.set_cache_limit(cache_limit)
        mx.set_memory_limit(memory_limit)
        resolved_device = "gpu" if device in {"auto", "mps"} else "cpu"
        resolved_dtype = "float16" if dtype == "auto" else dtype
        checkpoint, source = (local_checkpoint_files(checkpoint_path, model) if checkpoint_path else
                              checkpoint_files(model, revision or DEFAULT_CHECKPOINT_REVISION))
        agent = laya_mlx.load(str(checkpoint), device=resolved_device, dtype=resolved_dtype,
                              batch_size=batch_size, local_files_only=True)
        precision = {"precision_policy": "explicit", "weight_dtype": resolved_dtype,
                     "compute_dtype": resolved_dtype}
        packages = {name: version(name) for name in ("laya-mlx", "mlx", "mlx-metal", "tokenizers")}
        source = {**source, "mlx_cache_limit_bytes": cache_limit, "mlx_memory_limit_bytes": memory_limit}
    else:
        try:
            import torch
            import laya
        except ImportError as exc:
            raise RuntimeError("PyTorch Laya is unavailable; use .venv-laya or install .[laya]. See docs/laya.md") from exc
        resolved_device = device
        if device == "auto":
            resolved_device = "mps" if torch.backends.mps.is_available() else (
                "cuda" if torch.cuda.is_available() else "cpu")
        checkpoint, source = (local_checkpoint_files(checkpoint_path, model) if checkpoint_path else
                              checkpoint_files(model, revision or DEFAULT_CHECKPOINT_REVISION))
        agent = laya.load(str(checkpoint), device=resolved_device)
        precision = {"precision_policy": "upstream_auto", "compute_dtype": "upstream_auto",
                     "autocast_dtype": str(agent.dtype),
                     "weight_dtype": str(next(agent.model.parameters()).dtype),
                     "amp_enabled": agent.amp_enabled,
                     "mps_amp_min_rows": getattr(agent, "mps_amp_min_rows", None)}
        packages = {name: version(name) for name in ("laya", "torch", "transformers", "tokenizers")}
    metadata = {"backend": backend, "model": model, "device": resolved_device,
                **source, **precision, "packages": packages,
                "initialization_seconds": time.perf_counter() - started}
    return AgentRuntime(agent, metadata)
