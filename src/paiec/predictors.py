"""Predictors with the harness signature ``(subject, item, labeled) -> float``.

Learned predictors also define ``fit(data_root, train_benchmarks)``; ``build``
creates a fresh instance per fold and fits it on that fold's training benchmarks
only, so nothing learned can come from the benchmarks it is scored on.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Sequence
from pathlib import Path

BASELINE_ROOT = Path(__file__).resolve().parents[2] / "baseline"


def constant(value: float = 0.5):
    """Always predict ``value``; 0.5 gives Brier 0.25 at every budget."""
    return lambda subject, item, labeled: value


def empirical_mean():
    """The organizers' empirical mean, loaded from the baseline submodule unchanged."""
    path = BASELINE_ROOT / "empirical_mean" / "model.py"
    if not path.is_file():
        raise FileNotFoundError(f"{path} missing; run `git submodule update --init`")
    spec = importlib.util.spec_from_file_location("baseline_empirical_mean", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return lambda subject, item, labeled: module.predict([subject, item], labeled)


REGISTRY = {"constant": constant, "empirical_mean": empirical_mean}


def build(name: str, data_root: str | Path, train_benchmarks: Sequence[str]):
    """Create a fresh predictor and fit it on the training benchmarks if it learns."""
    predictor = REGISTRY[name]()
    if hasattr(predictor, "fit"):
        predictor.fit(data_root, list(train_benchmarks))
    return predictor


__all__ = ["REGISTRY", "build", "constant", "empirical_mean"]
