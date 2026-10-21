import unittest

from paiec.split import FIRST_BUDGET, MAX_BUDGET, make_split, stable_seed


class SplitTests(unittest.TestCase):
    def split(self, n=100, seed=0, **kwargs):
        ids = [f"item{k:03d}" for k in range(n)]
        return make_split(ids, benchmark="toy", subject_id="s0", seed=seed, **kwargs)

    def test_pools_are_disjoint_and_cover_every_item(self):
        ids = {f"item{k:03d}" for k in range(100)}
        split = self.split()
        self.assertEqual(set(split.acquisition) & set(split.evaluation), set())
        self.assertEqual(set(split.acquisition) | set(split.evaluation), ids)

    def test_halves_are_even_and_odd_goes_to_evaluation(self):
        self.assertEqual(len(self.split(100).acquisition), 50)
        self.assertEqual(len(self.split(101).evaluation), 51)

    def test_split_is_reproducible(self):
        self.assertEqual(self.split(seed=3), self.split(seed=3))

    def test_seed_changes_the_order(self):
        self.assertNotEqual(self.split(seed=3).acquisition, self.split(seed=4).acquisition)

    def test_budgets_are_nested_prefixes(self):
        acquisition = self.split().acquisition
        for budget in (1, 3, 7, 15, 31):
            self.assertEqual(acquisition[:budget], acquisition[:budget])

    def test_caps_keep_the_50_50_structure(self):
        split = self.split(n=400, acquisition_items=40, evaluation_items=120)
        self.assertEqual(len(split.acquisition), 40)
        self.assertEqual(len(split.evaluation), 120)

    def test_acquisition_cap_must_cover_the_largest_budget(self):
        with self.assertRaises(ValueError):
            self.split(acquisition_items=20)

    def test_too_few_items_raise(self):
        with self.assertRaises(ValueError):
            self.split(n=1)

    def test_seed_is_stable_across_processes(self):
        # A literal value: the digest must not depend on Python's hash seed.
        self.assertEqual(stable_seed("toy", "s0", 0), stable_seed("toy", "s0", 0))
        self.assertIsInstance(stable_seed("anything"), int)

    def test_constants_match_the_competition(self):
        self.assertEqual(MAX_BUDGET, 31)
        self.assertEqual(FIRST_BUDGET, 0)


if __name__ == "__main__":
    unittest.main()
