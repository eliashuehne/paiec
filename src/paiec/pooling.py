"""Pooled posterior for one benchmark, from every system's revealed labels.

The platform's ``labeled`` holds the labels revealed so far for every pair in
the run. For system ``s`` answering question ``q`` of one benchmark:

    logit = start(s, q) + level + own[s] + tag shifts of q + item[q]

``start`` is the prior offset's per-item logit. ``level`` and the tag shifts
are shared by every system on the benchmark; ``own[s]`` belongs to one system.
Priors: ``level ~ Normal(0, share * tau^2)``, ``own ~ Normal(0, (1 - share) *
tau^2)``, tag shifts ``~ Normal(0, sigma^2)``. With a single system,
``level + own ~ Normal(0, tau^2)``, so the model is then exactly the prior
offset. ``item[q] ~ Normal(0, item_sigma^2)`` is fitted afterwards, with the
other coefficients fixed, for questions some system revealed; it only moves
predictions on those questions.

Pure numpy, so the same code runs in the submission.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass

import numpy as np

from .groups import TagDesign
from .update import laplace_posterior, predictive, sigmoid

ITEM_STEPS = 10


@dataclass(frozen=True)
class PoolPrior:
    """Prior scales: total shift ``tau`` split by ``share``, tags ``sigma``, questions ``item_sigma``."""

    tau: float
    share: float
    sigma: float
    item_sigma: float

    def __post_init__(self):
        if not 0.0 <= self.share < 1.0:
            raise ValueError("share must be in [0, 1)")

    @property
    def level_variance(self) -> float:
        return self.share * self.tau * self.tau

    @property
    def own_variance(self) -> float:
        return (1.0 - self.share) * self.tau * self.tau


@dataclass(frozen=True)
class Layout:
    """Coefficient columns: optional shared level, one own shift per system, then tags."""

    level: int | None
    own: int
    tags: int
    precision: np.ndarray

    @classmethod
    def build(cls, n_subjects: int, n_tags: int, prior: PoolPrior) -> Layout:
        precision = []
        level = None
        if prior.share > 0:
            level = 0
            precision.append(1.0 / prior.level_variance)
        own = len(precision)
        precision += [1.0 / prior.own_variance] * n_subjects
        tags = len(precision)
        if prior.sigma > 0:
            precision += [1.0 / (prior.sigma * prior.sigma)] * n_tags
        return cls(level, own, tags, np.asarray(precision, dtype=float))

    def design(self, subject_index: np.ndarray, tag_matrix: np.ndarray) -> np.ndarray:
        """Rows for revealed labels; ``tag_matrix`` is ``TagDesign.matrix`` (column 0 unused)."""
        design = np.zeros((len(subject_index), len(self.precision)))
        if self.level is not None:
            design[:, self.level] = 1.0
        design[np.arange(len(subject_index)), self.own + subject_index] = 1.0
        design[:, self.tags :] = tag_matrix[:, 1 : 1 + len(self.precision) - self.tags]
        return design


def item_shifts(fitted_logits, item_index, labels, n_items: int, item_sigma: float):
    """Posterior mean and variance of each revealed question's shift, the rest held fixed."""
    if item_sigma <= 0 or n_items == 0:
        return np.zeros(n_items), np.zeros(n_items)
    labels = np.asarray(labels, dtype=float)
    precision = 1.0 / (item_sigma * item_sigma)
    shift = np.zeros(n_items)
    for _ in range(ITEM_STEPS):
        probabilities = sigmoid(fitted_logits + shift[item_index])
        gradient = np.bincount(item_index, labels - probabilities, n_items) - precision * shift
        curvature = np.bincount(item_index, probabilities * (1 - probabilities), n_items) + precision
        shift += gradient / curvature
    probabilities = sigmoid(fitted_logits + shift[item_index])
    curvature = np.bincount(item_index, probabilities * (1 - probabilities), n_items) + precision
    return shift, 1.0 / curvature


def index_of(keys: Sequence[Hashable]) -> tuple[dict, np.ndarray]:
    """Number distinct keys in order of appearance; return the mapping and each key's number."""
    column: dict = {}
    for key in keys:
        column.setdefault(key, len(column))
    return column, np.asarray([column[key] for key in keys], dtype=int)


@dataclass
class PoolPosterior:
    """The fitted shared and own shifts of one benchmark, shared by all its targets."""

    prior: PoolPrior
    tags: TagDesign
    layout: Layout
    subject_column: dict
    mean: np.ndarray
    covariance: np.ndarray
    item_column: dict
    item_mean: np.ndarray
    item_variance: np.ndarray

    @classmethod
    def fit(cls, starts, subjects, tags, items, labels, prior: PoolPrior) -> PoolPosterior:
        """One revealed label per position: start logit, system key, tag dict, question key, label."""
        starts = np.asarray(starts, dtype=float)
        labels = np.asarray(labels, dtype=float)
        subject_column, subject_index = index_of(subjects)
        design_tags = TagDesign.from_revealed(list(tags))
        n_tags = design_tags.matrix.shape[1] - 1 if prior.sigma > 0 else 0
        layout = Layout.build(len(subject_column), n_tags, prior)
        design = layout.design(subject_index, design_tags.matrix)
        mean, covariance = laplace_posterior(starts, design, labels, layout.precision)
        item_column, item_index = index_of(items)
        item_mean, item_variance = item_shifts(
            starts + design @ mean, item_index, labels, len(item_column), prior.item_sigma
        )
        return cls(prior, design_tags, layout, subject_column, mean, covariance,
                   item_column, item_mean, item_variance)

    def columns(self, subject: Hashable, tags: dict) -> tuple[list[int], float]:
        """A target's coefficient columns and the prior variance of its parts not in the fit."""
        columns, missing = [], 0.0
        if self.layout.level is not None:
            columns.append(self.layout.level)
        if subject in self.subject_column:
            columns.append(self.layout.own + self.subject_column[subject])
        else:
            missing += self.prior.own_variance
        if self.prior.sigma > 0:
            seen, unseen = self.tags.target_columns(tags)
            columns += [self.layout.tags + column - 1 for column in seen]
            missing += unseen * self.prior.sigma * self.prior.sigma
        return columns, missing

    def predict(self, start: float, subject: Hashable, tags: dict, item: Hashable) -> float:
        columns, variance = self.columns(subject, tags)
        mean = start + float(self.mean[columns].sum())
        variance += float(self.covariance[np.ix_(columns, columns)].sum())
        if item in self.item_column:
            number = self.item_column[item]
            mean += float(self.item_mean[number])
            variance += float(self.item_variance[number])
        return float(predictive(mean, 0.0, variance))


__all__ = ["Layout", "PoolPosterior", "PoolPrior", "index_of", "item_shifts"]
