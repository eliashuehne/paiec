"""Predictor factories and adapters used by the evaluation harness."""

from __future__ import annotations

import importlib.util
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from .contracts import Predictor
from .offset import BayesianOffset, TagOffset
from .prior import LogisticPrior
from .prior_offset import GBDTOffset, PriorOffset
from .shrinkage import ShrunkMean


BASELINE_ROOT = Path(__file__).resolve().parents[2] / "baseline"


@dataclass
class ConstantPredictor:
    """A fixed probability, useful as a reference score."""

    value: float = 0.5

    def __call__(self, subject: dict, item: dict, labeled: Sequence | None) -> float:
        return self.value


def constant(value: float = 0.5) -> ConstantPredictor:
    """Return a predictor that always returns the same probability."""
    return ConstantPredictor(value)


def empirical_mean() -> Predictor:
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

    def predict(subject, item, labeled):
        return module.predict([subject, item], labeled)

    return predict


REGISTRY: dict[str, Callable[[], Predictor]] = {
    "constant": constant,
    "empirical_mean": empirical_mean,
    "shrunk_mean": ShrunkMean,
    "logreg_prior": LogisticPrior,
    "offset": BayesianOffset,
    "tag_offset": TagOffset,
    "prior_offset": PriorOffset,
    "gbdt_offset": GBDTOffset,
}


def build(
    name: str, data_root: str | Path, train_benchmarks: Sequence[str]
) -> Predictor:
    """Create a predictor and fit it only on this fold's training benchmarks."""
    predictor = REGISTRY[name]()
    fit = getattr(predictor, "fit", None)
    if fit is not None:
        fit(data_root, list(train_benchmarks))
    return predictor


__all__ = ["REGISTRY", "ConstantPredictor", "build", "constant", "empirical_mean"]
