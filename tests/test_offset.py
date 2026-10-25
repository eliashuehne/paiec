import unittest

import numpy as np
import pandas as pd

from paiec.metrics import BUDGETS
from paiec.offset import BayesianOffset, offset_brier_alc
from paiec.update import constant_prior_posterior, offset_posterior, predictive, sigmoid

SUBJECT = {"normalized_name": "model-a"}
ITEM = {"item_content": "q", "benchmark_id": "bench"}


def revealed(labels):
    return [[[SUBJECT, {"item_content": "x", "benchmark_id": "bench"}], y] for y in labels]


def logit_normal_episodes(level, tau, pairs=4000, eval_items=400, seed=0):
    rng = np.random.default_rng(seed)
    theta = sigmoid(level + rng.normal(0.0, tau, size=pairs))
    reveals = rng.random((pairs, max(BUDGETS))) < theta[:, None]
    correct = np.concatenate([np.zeros((pairs, 1)), np.cumsum(reveals, axis=1)], axis=1)
    frame = pd.DataFrame({"accuracy": theta, "eval_mean": rng.binomial(eval_items, theta) / eval_items,
                          "weight": 1.0})
    for b in BUDGETS:
        frame[f"k{b}"] = correct[:, b]
    return frame


class PosteriorTests(unittest.TestCase):
    def test_no_labels_leaves_the_prior(self):
        mean, variance = offset_posterior([], [], tau=1.5)
        self.assertEqual(mean, 0.0)
        self.assertAlmostEqual(variance, 1.5 ** 2)

    def test_balanced_labels_do_not_shift(self):
        mean, _ = offset_posterior([0.0] * 4, [1, 0, 1, 0], tau=2.0)
        self.assertAlmostEqual(mean, 0.0, places=6)

    def test_successes_shift_up_and_shrink_uncertainty(self):
        mean, variance = offset_posterior([0.0] * 3, [1, 1, 1], tau=1.0)
        self.assertGreater(mean, 0.0)
        self.assertLess(variance, 1.0)

    def test_many_labels_approach_the_observed_rate(self):
        labels = [1] * 160 + [0] * 40
        mean, variance = offset_posterior([0.0] * 200, labels, tau=3.0)
        self.assertAlmostEqual(float(predictive(0.0, mean, variance)), 0.8, delta=0.01)

    def test_tiny_tau_ignores_labels(self):
        mean, variance = offset_posterior([0.4] * 5, [0] * 5, tau=1e-4)
        self.assertAlmostEqual(float(predictive(0.4, mean, variance)), float(sigmoid(0.4)), places=4)

    def test_vectorized_constant_prior_matches_the_general_posterior(self):
        means, variances = constant_prior_posterior(0.3, [0, 2, 5, 7], 7, tau=1.2)
        for correct, mean, variance in zip([0, 2, 5, 7], means, variances):
            expected = offset_posterior([0.3] * 7, [1] * correct + [0] * (7 - correct), tau=1.2)
            self.assertAlmostEqual(mean, expected[0], places=8)
            self.assertAlmostEqual(variance, expected[1], places=8)


class BayesianOffsetTests(unittest.TestCase):
    def test_predictions_match_the_closed_form_used_for_fitting(self):
        model = BayesianOffset(level=0.2, tau=1.4)
        for correct in range(4):
            labels = [1] * correct + [0] * (3 - correct)
            means, variances = constant_prior_posterior(0.2, [correct], 3, tau=1.4)
            expected = float(predictive(0.2, means[0], variances[0]))
            self.assertAlmostEqual(model(SUBJECT, ITEM, revealed(labels)), expected, places=10)

    def test_fit_recovers_level_and_spread(self):
        episodes = logit_normal_episodes(level=0.5, tau=1.5)
        model = BayesianOffset().fit_episodes(episodes)
        self.assertAlmostEqual(model.level, 0.5, delta=0.2)
        self.assertGreater(model.tau, 1.1)
        self.assertLess(model.tau, 2.0)

    def test_fit_beats_extreme_spreads(self):
        episodes = logit_normal_episodes(level=0.0, tau=1.5)
        model = BayesianOffset().fit_episodes(episodes)
        fitted = offset_brier_alc(episodes, model.level, model.tau)
        self.assertLess(fitted, offset_brier_alc(episodes, model.level, 0.05))
        self.assertLess(fitted, offset_brier_alc(episodes, model.level, 50.0))


if __name__ == "__main__":
    unittest.main()
