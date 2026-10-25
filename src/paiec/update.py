"""Bayesian pair offset: how revealed labels shift a pair's predictions.

Model: ``logit P(correct) = prior_logit(item) + delta`` with one shift ``delta``
per subject-benchmark pair and ``delta ~ Normal(0, tau^2)`` before any labels.
After revealed labels the posterior of ``delta`` is approximated by a Normal
(Laplace: Newton steps to the mode, curvature for the variance), and the
prediction averages over that remaining uncertainty with the probit
approximation ``sigmoid(z / sqrt(1 + pi * var / 8))``.

Pure numpy, so the same code runs in the submission.
"""

from __future__ import annotations

import numpy as np

NEWTON_STEPS = 25


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def offset_posterior(prior_logits, labels, tau: float) -> tuple[float, float]:
    """Posterior mean and variance of the pair shift given revealed labels."""
    prior_logits = np.asarray(prior_logits, dtype=float)
    labels = np.asarray(labels, dtype=float)
    precision = 1.0 / (tau * tau)
    shift = 0.0
    for _ in range(NEWTON_STEPS):
        p = sigmoid(prior_logits + shift)
        gradient = np.sum(labels - p) - shift * precision
        curvature = np.sum(p * (1.0 - p)) + precision
        shift += gradient / curvature
    p = sigmoid(prior_logits + shift)
    return float(shift), float(1.0 / (np.sum(p * (1.0 - p)) + precision))


def constant_prior_posterior(level: float, correct, revealed: int, tau: float):
    """Vectorized posterior when every item shares the prior logit ``level``."""
    correct = np.asarray(correct, dtype=float)
    precision = 1.0 / (tau * tau)
    shift = np.zeros_like(correct)
    for _ in range(NEWTON_STEPS):
        p = sigmoid(level + shift)
        shift += (correct - revealed * p - shift * precision) / (revealed * p * (1.0 - p) + precision)
    p = sigmoid(level + shift)
    return shift, 1.0 / (revealed * p * (1.0 - p) + precision)


def predictive(prior_logit, shift_mean, shift_variance):
    """P(correct) averaged over the remaining uncertainty about the shift."""
    return sigmoid((prior_logit + shift_mean) / np.sqrt(1.0 + np.pi * shift_variance / 8.0))


__all__ = ["constant_prior_posterior", "offset_posterior", "predictive", "sigmoid"]
