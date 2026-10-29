import unittest

import numpy as np

from paiec.metrics import BUDGETS, alc_weights
from paiec.pooled_offset import PooledOffset, pool_state, pooled_scores
from paiec.pooling import PoolPosterior, PoolPrior
from paiec.prior import LogisticPrior
from paiec.prior_offset import PriorOffset, prior_offset_alc, prior_states

FIELDS = ["normalized_name", "provider", "release_date", "access_date", "harness",
          "harness_version", "reasoning_effort", "subject_features_extra"]
PARAMETERS = dict(level=0.0, shift=0.1, scale=0.7, center=0.05, tau=1.5, sigma=0.75)


def subject(name):
    record = {field: "" for field in FIELDS}
    record.update(normalized_name=name, provider="Acme")
    return record


def item(text, task="", benchmark="bench"):
    features = f"task={task}" if task else ""
    return {"item_content": text, "item_features": features, "interactors": "",
            "benchmark_id": benchmark}


def random_prior():
    prior = LogisticPrior()
    prior.coef = np.random.default_rng(0).normal(scale=0.3, size=prior.coef.size)
    prior.intercept = -0.2
    return prior


def models(share=0.35, item_sigma=1.5, **overrides):
    parameters = dict(PARAMETERS, **overrides)
    prior = random_prior()
    return (PriorOffset(prior=prior, **parameters),
            PooledOffset(prior=prior, share=share, item_sigma=item_sigma, **parameters))


def answers(system, count, label, task=lambda k: "", prefix=""):
    return [[[system, item(f"{prefix}{system['normalized_name']} q{k}", task(k))], label(k)]
            for k in range(count)]


class OneSystemTests(unittest.TestCase):
    def test_pooled_model_with_only_own_labels_is_the_prior_offset(self):
        own_model, pooled = models()
        target = subject("Model 70B")
        labeled = answers(target, 9, lambda k: int(k % 3 != 0), task=lambda k: "ab"[k % 2])
        for budget in (0, 1, 3, 7, 9):
            for text, task in [("new", "a"), ("other", "b"), ("unseen tag", "z"), ("no tag", "")]:
                self.assertAlmostEqual(
                    pooled(target, item(text, task), labeled[:budget]),
                    own_model(target, item(text, task), labeled[:budget]), places=9)

    def test_other_benchmarks_are_ignored(self):
        _, pooled = models()
        target, other = subject("Model 70B"), subject("Other 7B")
        own = answers(target, 5, lambda k: 1)
        elsewhere = [[[other, item(f"x{k}", benchmark="elsewhere")], 0] for k in range(20)]
        self.assertAlmostEqual(pooled(target, item("t"), own),
                               pooled(target, item("t"), own + elsewhere), places=12)


class SharedEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.target = subject("Model 70B")
        self.own = answers(self.target, 5, lambda k: int(k < 3))
        self.others = []
        for name in ("A", "B", "C"):
            self.others += answers(subject(name), 15, lambda k: 0)

    def test_other_systems_move_the_benchmark_level(self):
        own_model, pooled = models(share=0.35, item_sigma=0.0)
        _, unshared = models(share=0.0, item_sigma=0.0)
        target_item = item("fresh question")
        alone = own_model(self.target, target_item, self.own)
        self.assertAlmostEqual(unshared(self.target, target_item, self.own + self.others), alone, places=9)
        self.assertLess(pooled(self.target, target_item, self.own + self.others), alone - 0.02)

    def test_a_question_others_answered_moves_only_that_question(self):
        # scale 0: every question starts equal, so only revealed answers separate them.
        _, pooled = models(share=0.0, item_sigma=1.5, scale=0.0)
        _, no_items = models(share=0.0, item_sigma=0.0, scale=0.0)
        shared_question = self.others[0][0][1]
        labeled = self.own + self.others
        fresh = pooled(self.target, item("fresh question"), labeled)
        self.assertLess(pooled(self.target, shared_question, labeled), fresh - 0.02)
        self.assertAlmostEqual(no_items(self.target, shared_question, labeled), fresh, places=12)


def synthetic_pool(members=3, seed=0):
    rng = np.random.default_rng(seed)
    questions = [f"q{k}" for k in range(70)]
    tasks = {question: {"task": "abc"[k % 3]} for k, question in enumerate(questions)}
    pool = []
    for number in range(members):
        order = rng.permutation(questions)
        revealed, evaluated = list(order[:31]), list(order[31:55])
        rows = rng.integers(1, 3, size=len(evaluated)).astype(float)
        episode = {
            "weight": 1.0 if number else 0.5, "subject_id": f"s{number}",
            "revealed_ids": revealed, "revealed_tags": [tasks[q] for q in revealed],
            "revealed_labels": rng.integers(0, 2, size=31).astype(float),
            "eval_ids": evaluated, "eval_tags": [tasks[q] for q in evaluated],
            "eval_rows": rows, "eval_correct": np.floor(rng.uniform(0, rows + 1)),
        }
        pool.append((episode, rng.normal(size=31), rng.normal(size=len(evaluated))))
    return pool


def start(logits):
    p = PARAMETERS
    return p["level"] + p["shift"] + p["scale"] * (np.asarray(logits, dtype=float) - p["center"])


class TuningTests(unittest.TestCase):
    def test_vectorised_scores_match_the_runtime_posterior(self):
        pool = synthetic_pool()
        prior = PoolPrior(PARAMETERS["tau"], 0.35, PARAMETERS["sigma"], 1.5)
        weights = np.array([e["weight"] for e, _, _ in pool])
        # Questions two apart share their visible text, as in many real benchmarks.
        twins = {f"q{k}": f"text {k // 2}" for k in range(70)}
        for keys in ({}, twins):
            expected = np.zeros(len(pool))
            for alc_weight, budget in zip(alc_weights(BUDGETS), BUDGETS):
                rows = [(start(r[k]), number, e["revealed_tags"][k], keys.get(e["revealed_ids"][k], e["revealed_ids"][k]),
                         e["revealed_labels"][k])
                        for number, (e, r, _) in enumerate(pool) for k in range(budget)]
                posterior = PoolPosterior.fit(*(list(column) for column in zip(*rows)), prior) if rows \
                    else PoolPosterior.fit([], [], [], [], [], prior)
                for number, (e, _, evaluated) in enumerate(pool):
                    p = np.array([posterior.predict(start(logit), number, tags, keys.get(question, question))
                                  for logit, tags, question in zip(evaluated, e["eval_tags"], e["eval_ids"])])
                    squared = np.sum(e["eval_rows"] * p * p - 2 * p * e["eval_correct"] + e["eval_correct"])
                    expected[number] += alc_weight * squared / e["eval_rows"].sum()
            states = [[pool_state(pool, budget, start, keys) for budget in BUDGETS]]
            actual = pooled_scores(states, PoolPrior(prior.tau, prior.share, prior.sigma, 0.0), [0.0, 1.5])
            self.assertAlmostEqual(actual[1], np.average(expected, weights=weights), places=10)
            self.assertNotAlmostEqual(actual[0], actual[1], places=6)

    def test_text_twins_of_own_revealed_questions_are_matched(self):
        pool = synthetic_pool(members=1)
        episode = pool[0][0]
        twins = {question: f"text of {question}" for question in episode["revealed_ids"]}
        twins[episode["eval_ids"][0]] = twins[episode["revealed_ids"][0]]
        state = pool_state(pool, 31, start, twins)
        self.assertEqual(state.members[0].items[0], 0)
        self.assertTrue(np.all(state.members[0].items[1:] == -1))

    def test_pools_of_one_score_like_the_prior_offset(self):
        pool = synthetic_pool()
        singles = [[[pool_state([member], budget, start) for budget in BUDGETS]] for member in pool]
        states = [prior_states(e, r, v) for e, r, v in pool]
        weights = [e["weight"] for e, _, _ in pool]
        expected = prior_offset_alc(states, weights, **PARAMETERS)
        for share in (0.0, 0.5):
            prior = PoolPrior(PARAMETERS["tau"], share, PARAMETERS["sigma"], 0.0)
            actual = pooled_scores([s[0] for s in singles], prior, [0.0, 2.0])
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
