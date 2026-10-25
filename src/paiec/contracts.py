"""The interfaces shared by predictors and the evaluation harness."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol


class Predictor(Protocol):
    """A callable returning P(correct) from a subject, item and revealed evidence.

    Evidence entries have the evaluator's shape: [[subject, item], label].
    Functions and callable classes can both implement this interface.
    """

    def __call__(self, subject: dict, item: dict, labeled: Sequence | None) -> float:
        ...


class TrainablePredictor(Predictor, Protocol):
    """A predictor that can be fitted on a fold's training benchmarks."""

    def fit(
        self, data_root: str | Path, train_benchmarks: Sequence[str]
    ) -> TrainablePredictor:
        ...
