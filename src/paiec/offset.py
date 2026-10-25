"""Bayesian pair offset on a constant prior, fitted on simulated training episodes.

Every item of a new pair starts at the same prior logit ``level``; revealed
labels move the pair's shift (see ``update.py``). ``level`` and the prior spread
``tau`` are chosen to minimise the ALC of the training episodes, the same way
the shrunk mean chooses its prior mean and strength.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
import pandas as pd

from .episodes import tag_episodes, training_episodes
from .groups import TagDesign, parse_tags
from .metrics import BUDGETS, alc_weights
from .shrinkage import matching_examples, matching_labels
from .update import (
    constant_prior_posterior,
    offset_posterior,
    predictive,
    tag_offset_posterior,
    tag_predictive,
)


class TagEpisodeState(NamedTuple):
    """Inputs at one reveal budget, reusable for every parameter candidate.

    Each pattern maps (known tag columns, unseen tag count) to the number of
    evaluation rows and correct labels. Grouping equivalent items avoids
    predicting the same probability repeatedly during fitting.
    """

    design: np.ndarray
    labels: np.ndarray
    patterns: dict[tuple[tuple[int, ...], int], tuple[float, float]]


@dataclass
class TagPosterior:
    """A fitted pair's tag mapping and posterior, shared by its target items."""

    design: TagDesign
    mean: np.ndarray
    covariance: np.ndarray

    def predict(self, tags: dict[str, str], level: float, sigma: float) -> float:
        columns, unseen = self.design.target_columns(tags)
        return tag_predictive(level, self.mean, self.covariance, columns, unseen, sigma)


def offset_brier_alc(episodes: pd.DataFrame, level: float, tau: float) -> float:
    """Weighted mean ALC of the constant-prior offset over training episodes."""
    evaluation_mean = episodes["eval_mean"].to_numpy()
    scores = np.zeros(len(episodes))
    for alc_weight, budget in zip(alc_weights(BUDGETS), BUDGETS):
        correct = episodes[f"k{budget}"].to_numpy()
        mean, variance = constant_prior_posterior(level, correct, budget, tau)
        probability = predictive(level, mean, variance)
        brier = probability * probability - 2 * probability * evaluation_mean + evaluation_mean
        scores += alc_weight * brier
    return float(np.average(scores, weights=episodes["weight"].to_numpy()))


@dataclass
class BayesianOffset:
    """Constant prior logit plus a Normal(0, tau^2) pair shift updated from revealed labels."""

    LEVELS = np.linspace(-1.5, 1.5, 31)
    TAUS = np.geomspace(0.1, 10.0, 41)

    level: float = 0.0
    tau: float = 1.0

    def fit(self, data_root, train_benchmarks: Sequence[str]):
        return self.fit_episodes(training_episodes(data_root, train_benchmarks))

    def fit_episodes(self, episodes: pd.DataFrame):
        if episodes.empty:
            raise ValueError("cannot fit BayesianOffset without training episodes")
        scores = np.array([
            [offset_brier_alc(episodes, level, tau) for tau in self.TAUS]
            for level in self.LEVELS
        ])
        level_index, tau_index = np.unravel_index(int(np.argmin(scores)), scores.shape)
        self.level = float(self.LEVELS[level_index])
        self.tau = float(self.TAUS[tau_index])
        return self

    def __call__(self, subject: dict, item: dict, labeled) -> float:
        labels = matching_labels(subject, item, labeled)
        mean, variance = offset_posterior(np.full(len(labels), self.level), labels, self.tau)
        return float(predictive(self.level, mean, variance))


def episode_states(episode: dict) -> list:
    """Per budget: revealed design, labels and evaluation items grouped by tag pattern.

    None of this depends on the model parameters, so the grid search reuses it.
    """
    states = []
    for budget in BUDGETS:
        design = TagDesign.from_revealed(episode["revealed_tags"][:budget])
        patterns = {}
        evaluation_items = zip(episode["eval_tags"], episode["eval_rows"], episode["eval_correct"])
        for tags, rows, correct in evaluation_items:
            columns, unseen = design.target_columns(tags)
            pattern = (tuple(columns), unseen)
            seen_rows, seen_correct = patterns.get(pattern, (0.0, 0.0))
            patterns[pattern] = (seen_rows + rows, seen_correct + correct)
        states.append(TagEpisodeState(design.matrix, episode["revealed_labels"][:budget], patterns))
    return states


def _state_error(state: TagEpisodeState, level: float, tau: float, sigma: float) -> tuple[float, float]:
    """Return summed squared error and row count for a single reveal budget."""
    mean, covariance = tag_offset_posterior(level, state.design, state.labels, tau, sigma)
    squared_error, total_rows = 0.0, 0.0
    for (columns, unseen), (rows, correct) in state.patterns.items():
        probability = tag_predictive(level, mean, covariance, columns, unseen, sigma)
        squared_error += rows * probability * probability - 2 * probability * correct + correct
        total_rows += rows
    return squared_error, total_rows


def tag_brier_alc(states: list, weights, level: float, tau: float, sigma: float) -> float:
    """Weighted mean ALC of the tag offset over precomputed episode states."""
    scores = []
    for pair_states in states:
        alc = 0.0
        for alc_weight, state in zip(alc_weights(BUDGETS), pair_states):
            squared_error, rows = _state_error(state, level, tau, sigma)
            alc += alc_weight * squared_error / rows
        scores.append(alc)
    return float(np.average(scores, weights=weights))


@dataclass
class TagOffset:
    """Bayesian offset plus one shift per revealed tag value (e.g. ``task=qa3``).

    ``logit = level + pair_shift + sum(tag_shifts)``; the pair shift has prior
    spread ``tau``, each tag shift ``sigma``. Tags only count when their values
    repeat among the revealed items, so pairs without useful tags get exactly
    the plain offset. ``level`` and a starting ``tau`` come from the plain
    offset; ``tau`` and ``sigma`` are then chosen on item-level training episodes.
    """

    SIGMAS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5)
    TAU_SCALES = (0.85, 1.0)
    CACHE_LIMIT = 4096

    level: float = 0.0
    tau: float = 1.0
    sigma: float = 0.0
    _posteriors: dict[tuple, TagPosterior] = field(default_factory=dict, init=False, repr=False, compare=False)

    def fit(self, data_root, train_benchmarks: Sequence[str]):
        base = BayesianOffset().fit(data_root, train_benchmarks)
        episodes = [
            episode
            for benchmark in train_benchmarks
            for episode in tag_episodes(str(data_root), benchmark)
        ]
        return self.fit_episodes(episodes, base.level, base.tau)

    def fit_episodes(self, episodes: Sequence[dict], level: float, tau: float):
        states = [episode_states(episode) for episode in episodes]
        weights = [episode["weight"] for episode in episodes]
        grid = [(tau * scale, sigma) for scale in self.TAU_SCALES for sigma in self.SIGMAS]
        scores = [
            tag_brier_alc(states, weights, level, candidate_tau, candidate_sigma)
            for candidate_tau, candidate_sigma in grid
        ]
        self.level = float(level)
        best_tau, best_sigma = grid[int(np.argmin(scores))]
        self.tau, self.sigma = float(best_tau), float(best_sigma)
        self._posteriors.clear()
        return self

    def __call__(self, subject: dict, item: dict, labeled) -> float:
        examples = matching_examples(subject, item, labeled)
        posterior = self._posterior_for(subject, item, examples)
        tags = parse_tags(item.get("item_features", ""))
        return posterior.predict(tags, self.level, self.sigma)

    def _posterior_for(self, subject: dict, item: dict, examples) -> TagPosterior:
        # Targets from the same pair and reveal history share one posterior.
        key = (
            tuple(sorted(subject.items())),
            item.get("benchmark_id"),
            tuple((example.get("item_features"), label) for example, label in examples),
        )
        if key not in self._posteriors:
            if len(self._posteriors) >= self.CACHE_LIMIT:
                self._posteriors.clear()
            tags = [parse_tags(example.get("item_features", "")) for example, _ in examples]
            labels = [label for _, label in examples]
            design = TagDesign.from_revealed(tags)
            mean, covariance = tag_offset_posterior(
                self.level, design.matrix, labels, self.tau, self.sigma
            )
            self._posteriors[key] = TagPosterior(design, mean, covariance)
        return self._posteriors[key]


__all__ = ["BayesianOffset", "TagOffset", "episode_states", "offset_brier_alc", "tag_brier_alc"]
