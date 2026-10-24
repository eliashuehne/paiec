import unittest
from unittest.mock import patch

from paiec import predictors
from paiec.folds import benchmark_folds

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


if __name__ == "__main__":
    unittest.main()
