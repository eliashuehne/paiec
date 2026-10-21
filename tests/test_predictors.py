import unittest

import pandas as pd

from paiec.predictors import Constant, EmpiricalMean, ShrunkMean, SubjectPrior
from paiec.textmodel import TextRidge

SUBJECT = {"normalized_name": "model-x", "harness": "h0"}
ITEM = {"item_content": "q", "benchmark_id": "bench-a"}

STATS = pd.DataFrame(
    {
        "benchmark": ["bench-a", "bench-b"],
        "identity": ["model-x", "model-x"],
        "n": [10, 10],
        "successes": [5, 10],
    }
)


class ConstantTests(unittest.TestCase):
    def test_returns_its_value(self):
        self.assertEqual(Constant()(SUBJECT, ITEM, []), 0.5)


class EmpiricalMeanTests(unittest.TestCase):
    def test_matches_the_baseline_predictor(self):
        mean = EmpiricalMean()
        self.assertEqual(mean(SUBJECT, ITEM, []), 0.5)
        evidence = [[[SUBJECT, dict(ITEM)], 1], [[SUBJECT, dict(ITEM)], 0]]
        self.assertEqual(mean(SUBJECT, ITEM, evidence), 0.5)
        evidence.append([[SUBJECT, dict(ITEM)], 1])
        self.assertAlmostEqual(mean(SUBJECT, ITEM, evidence), 2 / 3)


class SubjectPriorTests(unittest.TestCase):
    def test_excludes_the_current_benchmark(self):
        prior = SubjectPrior(STATS)
        # bench-a's own 5/10 must not leak; only bench-b's 10/10 counts.
        self.assertAlmostEqual(prior(SUBJECT, ITEM, []), (10 + 1) / (10 + 2))

    def test_falls_back_to_the_global_rate(self):
        prior = SubjectPrior(STATS)
        item = dict(ITEM, benchmark_id="bench-b")
        # bench-a remains once bench-b is excluded.
        self.assertAlmostEqual(prior(SUBJECT, item, []), (5 + 1) / (10 + 2))

    def test_unknown_subject_uses_global_rate(self):
        prior = SubjectPrior(STATS)
        unknown = dict(SUBJECT, normalized_name="never-seen")
        self.assertAlmostEqual(prior(unknown, ITEM, []), (15 + 1) / (20 + 2))

    def test_probability_stays_inside_the_clip(self):
        prior = SubjectPrior(STATS)
        self.assertGreaterEqual(prior(SUBJECT, ITEM, []), 0.02)
        self.assertLessEqual(prior(SUBJECT, ITEM, []), 0.98)


class ShrunkMeanTests(unittest.TestCase):
    def test_no_labels_reduces_to_the_prior(self):
        prior = SubjectPrior(STATS)
        shrunk = ShrunkMean(prior, k=5)
        self.assertAlmostEqual(shrunk(SUBJECT, ITEM, []), prior(SUBJECT, ITEM, []))

    def test_labels_override_the_prior_eventually(self):
        shrunk = ShrunkMean(SubjectPrior(STATS), k=5)
        evidence = [[[SUBJECT, dict(ITEM)], 0] for _ in range(10)]
        value = shrunk(SUBJECT, ITEM, evidence)
        base = SubjectPrior(STATS)(SUBJECT, ITEM, [])
        self.assertLess(value, base)
        self.assertGreater(value, 0.0)

    def test_ignores_labels_from_another_benchmark(self):
        shrunk = ShrunkMean(SubjectPrior(STATS), k=5)
        other = dict(ITEM, benchmark_id="bench-b")
        evidence = [[[SUBJECT, other], 0] for _ in range(10)]
        self.assertAlmostEqual(
            shrunk(SUBJECT, ITEM, evidence), SubjectPrior(STATS)(SUBJECT, ITEM, [])
        )


class TextRidgeTests(unittest.TestCase):
    def test_separates_a_trivial_corpus(self):
        texts = ["good correct answer"] * 20 + ["bad wrong answer"] * 20
        labels = [1] * 20 + [0] * 20
        ridge = TextRidge().fit(texts, labels)
        good = ridge(SUBJECT, {"item_content": "good correct answer"}, [])
        bad = ridge(SUBJECT, {"item_content": "bad wrong answer"}, [])
        self.assertGreater(good, bad)

    def test_predictions_stay_inside_the_clip(self):
        ridge = TextRidge().fit(["a"] * 5, [1] * 5)
        value = ridge(SUBJECT, {"item_content": "zzz"}, [])
        self.assertGreaterEqual(value, 0.02)
        self.assertLessEqual(value, 0.98)


if __name__ == "__main__":
    unittest.main()
