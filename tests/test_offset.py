import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from paiec.metrics import BUDGETS, alc_weights
from paiec.offset import BayesianOffset, TagOffset, episode_states, offset_brier_alc, tag_brier_alc
from paiec.update import (constant_prior_posterior, offset_posterior, predictive, sigmoid,
                          tag_offset_posterior)

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


TASKS = ["a", "b", "c", "d"]


def tagged(task, label=None):
    item = {"item_content": "x", "item_features": f"task={task};id={np.random.random()}", "benchmark_id": "bench"}
    return item if label is None else [[SUBJECT, item], label]


def tag_episode_set(sigma, pairs=300, level=0.0, tau=1.2, seed=0):
    """Synthetic pairs whose items differ by task with per-pair task effects of spread sigma."""
    rng = np.random.default_rng(seed)
    episodes = []
    for _ in range(pairs):
        shift = rng.normal(0, tau)
        effects = dict(zip(TASKS, rng.normal(0, sigma, size=len(TASKS)) if sigma else np.zeros(len(TASKS))))
        def draw(n):
            tasks = rng.choice(TASKS, size=n)
            p = sigmoid(level + shift + np.array([effects[t] for t in tasks]))
            return [{"task": t} for t in tasks], (rng.random(n) < p).astype(float)
        revealed_tags, revealed_labels = draw(max(BUDGETS))
        eval_tags, eval_labels = draw(120)
        episodes.append({"weight": 1.0, "revealed_tags": revealed_tags, "revealed_labels": revealed_labels,
                         "eval_tags": eval_tags, "eval_rows": np.ones(120), "eval_correct": eval_labels})
    return episodes


class TagOffsetTests(unittest.TestCase):
    def test_targets_share_a_posterior_until_revealed_labels_change(self):
        model = TagOffset(level=0.2, tau=1.3, sigma=0.8)
        labeled = [tagged("a", 1), tagged("b", 0)] * 2
        with patch("paiec.offset.tag_offset_posterior", wraps=tag_offset_posterior) as fit_posterior:
            model(SUBJECT, tagged("a"), labeled)
            model(SUBJECT, tagged("b"), labeled)
            self.assertEqual(fit_posterior.call_count, 1)
            model(SUBJECT, tagged("a"), labeled + [tagged("a", 1)])
            self.assertEqual(fit_posterior.call_count, 2)

    def test_cache_eviction_preserves_predictions(self):
        model = TagOffset(level=0.2, tau=1.3, sigma=0.8)
        model.CACHE_LIMIT = 2
        histories = [
            [tagged("a", 1), tagged("b", 0)] * 2,
            [tagged("a", 0), tagged("b", 1)] * 2,
            [],
        ]
        for labeled in histories + histories:
            fresh_model = TagOffset(level=model.level, tau=model.tau, sigma=model.sigma)
            expected = fresh_model(SUBJECT, tagged("a"), labeled)
            self.assertEqual(model(SUBJECT, tagged("a"), labeled), expected)
            self.assertLessEqual(len(model._posteriors), model.CACHE_LIMIT)

    def test_refitting_refreshes_cached_predictions(self):
        model = TagOffset(level=-1.0, tau=2.0, sigma=0.8)
        labeled = [tagged("a", 1), tagged("b", 0)] * 2
        target = tagged("a")
        model(SUBJECT, target, labeled)
        model.fit_episodes(tag_episode_set(sigma=1.0, pairs=1), level=0.5, tau=1.2)
        fresh_model = TagOffset(level=model.level, tau=model.tau, sigma=model.sigma)
        self.assertEqual(model(SUBJECT, target, labeled), fresh_model(SUBJECT, target, labeled))

    def test_zero_sigma_is_the_plain_offset(self):
        labels = [1, 0, 1, 1, 0, 1, 1]
        theta, covariance = tag_offset_posterior(0.2, np.ones((7, 3)), labels, tau=1.3, sigma=0.0)
        mean, variance = offset_posterior([0.2] * 7, labels, tau=1.3)
        self.assertAlmostEqual(theta[0], mean, places=8)
        self.assertAlmostEqual(covariance[0, 0], variance, places=8)
        tag_model = TagOffset(level=0.2, tau=1.3, sigma=0.0)
        plain = BayesianOffset(level=0.2, tau=1.3)
        labeled = [tagged(TASKS[k % 2], y) for k, y in enumerate(labels)]
        self.assertAlmostEqual(tag_model(SUBJECT, tagged("a"), labeled), plain(SUBJECT, ITEM, labeled), places=8)

    def test_tag_groups_separate_after_revealed_labels(self):
        model = TagOffset(level=0.0, tau=1.5, sigma=1.0)
        labeled = [tagged("a", 1) for _ in range(5)] + [tagged("b", 0) for _ in range(5)]
        p_a, p_b, p_new = (model(SUBJECT, tagged(t), labeled) for t in ("a", "b", "z"))
        self.assertGreater(p_a, 0.7)
        self.assertLess(p_b, 0.3)
        self.assertGreater(p_new, p_b)
        self.assertLess(p_new, p_a)

    def test_fast_fitting_path_matches_per_call_predictions(self):
        episode = tag_episode_set(sigma=1.0, pairs=1, seed=3)[0]
        model = TagOffset(level=0.1, tau=1.2, sigma=0.8)
        fast = tag_brier_alc([episode_states(episode)], [1.0], 0.1, 1.2, 0.8)
        alc = 0.0
        for weight, budget in zip(alc_weights(BUDGETS), BUDGETS):
            labeled = [[[SUBJECT, {"item_features": f"task={t['task']}", "benchmark_id": "bench"}], y]
                       for t, y in zip(episode["revealed_tags"][:budget], episode["revealed_labels"][:budget])]
            p = np.array([model(SUBJECT, {"item_features": f"task={t['task']}", "benchmark_id": "bench"}, labeled)
                          for t in episode["eval_tags"]])
            alc += weight * np.mean((p - episode["eval_correct"]) ** 2)
        self.assertAlmostEqual(fast, alc, places=10)

    def test_fit_finds_tag_effects_when_present(self):
        model = TagOffset().fit_episodes(tag_episode_set(sigma=1.5), level=0.0, tau=1.2)
        self.assertGreaterEqual(model.sigma, 0.75)

    def test_fit_switches_tags_off_when_absent(self):
        model = TagOffset().fit_episodes(tag_episode_set(sigma=0.0), level=0.0, tau=1.2)
        self.assertLessEqual(model.sigma, 0.25)


if __name__ == "__main__":
    unittest.main()
