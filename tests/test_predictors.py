import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from paiec import predictors
from paiec.folds import benchmark_folds
from paiec.metrics import BUDGETS
from paiec.shrinkage import ShrunkMean, matching_labels, shrunk_brier_alc

NAMES = [f"bench{i:02d}" for i in range(12)]


class Recorder:
    """A learned predictor that remembers which benchmarks it was fitted on."""

    def __init__(self):
        self.seen = None

    def fit(self, data_root, train_benchmarks):
        self.seen = list(train_benchmarks)

    def __call__(self, subject, item, labeled):
        return 0.5


class BuildTests(unittest.TestCase):
    def test_learned_predictor_sees_only_training_benchmarks(self):
        with patch.dict(predictors.REGISTRY, {"recorder": Recorder}):
            for train, test in benchmark_folds(NAMES, k=4):
                fitted = predictors.build("recorder", "unused", train)
                self.assertEqual(fitted.seen, train)
                self.assertFalse(set(fitted.seen) & set(test))

    def test_each_build_is_a_fresh_instance(self):
        with patch.dict(predictors.REGISTRY, {"recorder": Recorder}):
            first = predictors.build("recorder", "unused", ["a"])
            second = predictors.build("recorder", "unused", ["b"])
            self.assertIsNot(first, second)
            self.assertEqual(first.seen, ["a"])

    def test_stateless_predictors_need_no_fit(self):
        predict = predictors.build("constant", "unused", NAMES)
        self.assertEqual(predict({}, {}, []), 0.5)


SUBJECT = {"normalized_name": "model-a"}
OTHER = {"normalized_name": "model-b"}
ITEM = {"item_content": "q", "benchmark_id": "bench"}


def revealed(subject, benchmark, label):
    return [[subject, {"item_content": "x", "benchmark_id": benchmark}], label]


class PairLabelTests(unittest.TestCase):
    def test_only_same_subject_and_benchmark_count(self):
        labeled = [
            revealed(SUBJECT, "bench", 1),
            revealed(SUBJECT, "bench", 0),
            revealed(OTHER, "bench", 1),
            revealed(SUBJECT, "elsewhere", 1),
        ]
        self.assertEqual(matching_labels(SUBJECT, ITEM, labeled), [1, 0])


def beta_binomial_episodes(alpha, p0, pairs=4000, eval_items=400, seed=0):
    rng = np.random.default_rng(seed)
    theta = rng.beta(alpha * p0, alpha * (1 - p0), size=pairs)
    reveals = rng.random((pairs, max(BUDGETS))) < theta[:, None]
    correct = np.concatenate([np.zeros((pairs, 1)), np.cumsum(reveals, axis=1)], axis=1)
    frame = pd.DataFrame({
        "accuracy": theta,
        "eval_mean": rng.binomial(eval_items, theta) / eval_items,
        "weight": 1.0,
    })
    for b in BUDGETS:
        frame[f"k{b}"] = correct[:, b]
    return frame


class ShrunkMeanTests(unittest.TestCase):
    def test_no_labels_predicts_the_prior(self):
        model = ShrunkMean(prior_mean=0.6, prior_strength=3.0)
        self.assertAlmostEqual(model(SUBJECT, ITEM, []), 0.6)

    def test_formula_after_labels(self):
        model = ShrunkMean(prior_mean=0.6, prior_strength=3.0)
        labeled = [revealed(SUBJECT, "bench", 1), revealed(SUBJECT, "bench", 1)]
        self.assertAlmostEqual(model(SUBJECT, ITEM, labeled), (3.0 * 0.6 + 2) / 5)

    def test_tiny_strength_is_the_empirical_mean(self):
        model = ShrunkMean(prior_mean=0.6, prior_strength=1e-9)
        labeled = [revealed(SUBJECT, "bench", 1), revealed(SUBJECT, "bench", 0)]
        self.assertAlmostEqual(model(SUBJECT, ITEM, labeled), 0.5)

    def test_fit_recovers_the_true_prior_strength(self):
        model = ShrunkMean().fit_episodes(beta_binomial_episodes(alpha=3.0, p0=0.6))
        self.assertAlmostEqual(model.prior_mean, 0.6, delta=0.02)
        self.assertGreater(model.prior_strength, 2.0)
        self.assertLess(model.prior_strength, 4.5)

    def test_fit_beats_the_extremes(self):
        episodes = beta_binomial_episodes(alpha=3.0, p0=0.6)
        model = ShrunkMean().fit_episodes(episodes)
        fitted = shrunk_brier_alc(episodes, model.prior_mean, model.prior_strength)
        self.assertLess(fitted, shrunk_brier_alc(episodes, model.prior_mean, 1e-6))
        self.assertLess(fitted, shrunk_brier_alc(episodes, model.prior_mean, 1e6))


if __name__ == "__main__":
    unittest.main()
