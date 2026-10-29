"""A prior-smoothed predictor fitted from simulated training episodes."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd

from .episodes import training_episodes
from .metrics import BUDGETS, alc_weights


def matching_examples(
    subject: dict, item: dict, labeled: Sequence | None
) -> list[tuple[dict, int]]:
    """Return (revealed item, label) for this subject and benchmark from evaluator evidence."""
    benchmark = item.get("benchmark_id")
    examples = []
    for (observed_subject, observed_item), label in labeled or []:
        if observed_subject == subject and observed_item.get("benchmark_id") == benchmark:
            examples.append((observed_item, int(label)))
    return examples


def matching_labels(subject: dict, item: dict, labeled: Sequence | None) -> list[int]:
    """Return labels for this subject and benchmark from evaluator evidence."""
    return [label for _, label in matching_examples(subject, item, labeled)]


def shrunk_brier_alc(
    episodes: pd.DataFrame, prior_mean: float, prior_strength: float
) -> float:
    """Score one prior setting against weighted training episodes."""
    episode_weights = episodes["weight"].to_numpy()
    evaluation_means = episodes["eval_mean"].to_numpy()
    scores = np.zeros(len(episodes))

    for alc_weight, budget in zip(alc_weights(BUDGETS), BUDGETS):
        correct_labels = episodes[f"k{budget}"].to_numpy()
        probabilities = (prior_strength * prior_mean + correct_labels) / (
            prior_strength + budget
        )
        # For binary labels, mean((p - y)**2) = p**2 - 2*p*mean(y) + mean(y).
        scores += alc_weight * (
            probabilities**2 - 2 * probabilities * evaluation_means + evaluation_means
        )

    return float(np.average(scores, weights=episode_weights))


@dataclass
class ShrunkMean:
    """Average revealed labels, with a Beta prior learned from training data."""

    PRIOR_STRENGTHS: ClassVar[np.ndarray] = np.geomspace(0.05, 200, 81)

    prior_mean: float = 0.5
    prior_strength: float = 0.0

    def fit(self, data_root: str | Path, train_benchmarks: Sequence[str]) -> ShrunkMean:
        """Fit from training benchmarks in one cross-validation fold."""
        return self.fit_episodes(training_episodes(data_root, train_benchmarks))

    def fit_episodes(self, episodes: pd.DataFrame) -> ShrunkMean:
        """Choose a prior mean and strength from already-built episodes."""
        if episodes.empty:
            raise ValueError("cannot fit ShrunkMean without training episodes")

        weights = episodes["weight"].to_numpy()
        self.prior_mean = float(np.average(episodes["accuracy"], weights=weights))
        candidate_scores = [
            shrunk_brier_alc(episodes, self.prior_mean, strength)
            for strength in self.PRIOR_STRENGTHS
        ]
        best_index = int(np.argmin(candidate_scores))
        self.prior_strength = float(self.PRIOR_STRENGTHS[best_index])
        return self

    def __call__(self, subject: dict, item: dict, labeled: Sequence | None) -> float:
        labels = matching_labels(subject, item, labeled)
        return (self.prior_strength * self.prior_mean + sum(labels)) / (
            self.prior_strength + len(labels)
        )


__all__ = ["ShrunkMean", "matching_examples", "matching_labels", "shrunk_brier_alc"]
