"""Bayesian pair offset on a constant prior, fitted on simulated training episodes.

Every item of a new pair starts at the same prior logit ``level``; revealed
labels move the pair's shift (see ``update.py``). ``level`` and the prior spread
``tau`` are chosen to minimise the ALC of the training episodes, the same way
the shrunk mean chooses its prior mean and strength.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .episodes import training_episodes
from .metrics import BUDGETS, alc_weights
from .shrinkage import matching_labels
from .update import constant_prior_posterior, offset_posterior, predictive


def offset_brier_alc(episodes: pd.DataFrame, level: float, tau: float) -> float:
    """Weighted mean ALC of the constant-prior offset over training episodes."""
    eval_mean = episodes["eval_mean"].to_numpy()
    scores = np.zeros(len(episodes))
    for alc_weight, budget in zip(alc_weights(BUDGETS), BUDGETS):
        mean, variance = constant_prior_posterior(level, episodes[f"k{budget}"].to_numpy(), budget, tau)
        p = predictive(level, mean, variance)
        scores += alc_weight * (p * p - 2 * p * eval_mean + eval_mean)
    return float(np.average(scores, weights=episodes["weight"].to_numpy()))


class BayesianOffset:
    """Constant prior logit plus a Normal(0, tau^2) pair shift updated from revealed labels."""

    LEVELS = np.linspace(-1.5, 1.5, 31)
    TAUS = np.geomspace(0.1, 10.0, 41)

    def __init__(self, level: float = 0.0, tau: float = 1.0):
        self.level = level
        self.tau = tau

    def fit(self, data_root, train_benchmarks: Sequence[str]):
        return self.fit_episodes(training_episodes(data_root, train_benchmarks))

    def fit_episodes(self, episodes: pd.DataFrame):
        if episodes.empty:
            raise ValueError("cannot fit BayesianOffset without training episodes")
        scores = np.array([[offset_brier_alc(episodes, level, tau) for tau in self.TAUS] for level in self.LEVELS])
        level_index, tau_index = np.unravel_index(int(np.argmin(scores)), scores.shape)
        self.level = float(self.LEVELS[level_index])
        self.tau = float(self.TAUS[tau_index])
        return self

    def __call__(self, subject: dict, item: dict, labeled) -> float:
        labels = matching_labels(subject, item, labeled)
        mean, variance = offset_posterior(np.full(len(labels), self.level), labels, self.tau)
        return float(predictive(self.level, mean, variance))


__all__ = ["BayesianOffset", "offset_brier_alc"]
