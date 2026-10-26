import unittest

from paiec.folds import benchmark_folds

NAMES = [f"bench{i:03d}" for i in range(23)]


class BenchmarkFoldTests(unittest.TestCase):
    def test_each_benchmark_is_held_out_exactly_once(self):
        folds = benchmark_folds(NAMES, k=5)
        held_out = [name for _, test in folds for name in test]
        self.assertEqual(sorted(held_out), sorted(NAMES))

    def test_train_and_test_are_disjoint_and_complete(self):
        for train, test in benchmark_folds(NAMES, k=5):
            self.assertFalse(set(train) & set(test))
            self.assertEqual(sorted(train + test), sorted(NAMES))

    def test_folds_are_balanced(self):
        sizes = [len(test) for _, test in benchmark_folds(NAMES, k=5)]
        self.assertLessEqual(max(sizes) - min(sizes), 1)

    def test_input_order_does_not_matter(self):
        self.assertEqual(benchmark_folds(NAMES), benchmark_folds(list(reversed(NAMES))))

    def test_seed_changes_the_assignment(self):
        self.assertNotEqual(benchmark_folds(NAMES, seed=0), benchmark_folds(NAMES, seed=1))

    def test_too_few_benchmarks_raises(self):
        with self.assertRaises(ValueError):
            benchmark_folds(["a", "b"], k=5)

    def test_at_least_two_folds_are_required(self):
        with self.assertRaises(ValueError):
            benchmark_folds(NAMES, k=1)


if __name__ == "__main__":
    unittest.main()
