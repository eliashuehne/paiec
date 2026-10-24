"""Predictor factories and adapters used by the evaluation harness."""

from __future__ import annotations

from collections.abc import Sequence
import importlib.util
from pathlib import Path

from .shrinkage import ShrunkMean


BASELINE_ROOT = Path(__file__).resolve().parents[2] / "baseline"


def constant(value: float = 0.5):
    """Return a predictor that always returns the same probability."""
    return lambda subject, item, labeled: value


def empirical_mean():
    """Load the organizers' empirical-mean baseline without changing it."""
    model_path = BASELINE_ROOT / "empirical_mean" / "model.py"
    if not model_path.is_file():
        raise FileNotFoundError(
            f"{model_path} missing; run `git submodule update --init`"
        )

    specification = importlib.util.spec_from_file_location(
        "baseline_empirical_mean", model_path
    )
    if specification is None or specification.loader is None:
        raise ImportError(f"could not load baseline model from {model_path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return lambda subject, item, labeled: module.predict([subject, item], labeled)


REGISTRY = {
    "constant": constant,
    "empirical_mean": empirical_mean,
    "shrunk_mean": ShrunkMean,
}


def build(name: str, data_root: str | Path, train_benchmarks: Sequence[str]):
    """Create a predictor and fit it only on this fold's training benchmarks."""
    predictor = REGISTRY[name]()
    fit = getattr(predictor, "fit", None)
    if fit is not None:
        fit(data_root, list(train_benchmarks))
    return predictor


__all__ = ["REGISTRY", "build", "constant", "empirical_mean"]
