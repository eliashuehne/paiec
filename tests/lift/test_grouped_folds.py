import json
import tempfile
import unittest
from pathlib import Path

from paiec.folds import grouped_benchmark_folds, load_benchmark_groups

GROUPS_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "lift" / "benchmark_groups.json"


class GroupedBenchmarkFoldTests(unittest.TestCase):
    NAMES = [f"bench{i:03d}" for i in range(60)]
    GROUPS = [["bench000", "bench001", "bench002"], ["bench010", "bench011"]]

    def split(self, **kwargs):
        return grouped_benchmark_folds(self.NAMES, self.GROUPS, **kwargs)

    def test_groups_are_never_split(self):
        for seed in range(20):
            sealed, folds = self.split(seed=seed)
            for group in self.GROUPS:
                homes = [test for _, test in folds if set(group) & set(test)]
                if set(group) <= set(sealed):
                    self.assertEqual(homes, [])
                else:
                    self.assertEqual(len(homes), 1)
                    self.assertLessEqual(set(group), set(homes[0]))

    def test_sealed_benchmarks_never_appear_in_any_fold(self):
        sealed, folds = self.split()
        for train, test in folds:
            self.assertFalse(set(sealed) & (set(train) | set(test)))

    def test_every_benchmark_is_sealed_or_tested_exactly_once(self):
        sealed, folds = self.split()
        tested = [name for _, test in folds for name in test]
        self.assertEqual(sorted(sealed + tested), self.NAMES)

    def test_sealed_size_follows_fraction(self):
        sealed, _ = self.split(sealed_fraction=0.2)
        self.assertGreaterEqual(len(sealed), 12)
        self.assertLessEqual(len(sealed), 12 + max(map(len, self.GROUPS)) - 1)

    def test_sealed_set_does_not_depend_on_k(self):
        self.assertEqual(self.split(k=5)[0], self.split(k=3)[0])

    def test_same_seed_gives_same_split(self):
        self.assertEqual(self.split(seed=3), self.split(seed=3))
        self.assertNotEqual(self.split(seed=3), self.split(seed=4))

    def test_folds_are_balanced(self):
        _, folds = self.split()
        sizes = [len(test) for _, test in folds]
        self.assertLessEqual(max(sizes) - min(sizes), max(map(len, self.GROUPS)))

    def test_benchmark_in_two_groups_raises(self):
        with self.assertRaises(ValueError):
            grouped_benchmark_folds(self.NAMES, [["bench000", "bench001"], ["bench001"]])

    def test_unknown_group_members_are_ignored(self):
        sealed, folds = grouped_benchmark_folds(self.NAMES, [["bench000", "missing"]])
        everything = set(sealed) | {name for _, test in folds for name in test}
        self.assertNotIn("missing", everything)

    def test_shipped_groups_config_is_valid(self):
        groups = load_benchmark_groups(GROUPS_CONFIG)
        names = [name for group in groups for name in group]
        self.assertEqual(len(names), len(set(names)))
        self.assertIn(["adaptivestep", "felm", "ikp", "pku_saferlhf", "prox"], groups)
        self.assertEqual(json.loads(GROUPS_CONFIG.read_text())["version"], "benchmark_groups_v2")

    def test_shipped_families_land_in_one_fold_or_wholly_sealed(self):
        groups = load_benchmark_groups(GROUPS_CONFIG)
        members = {name for group in groups for name in group}
        names = sorted(members) + [f"other{i:03d}" for i in range(120)]
        for seed in range(10):
            sealed, folds = grouped_benchmark_folds(names, groups, seed=seed)
            for group in groups:
                homes = {"sealed" if name in sealed else next(
                    number for number, (_, test) in enumerate(folds) if name in test
                ) for name in group}
                self.assertEqual(len(homes), 1, f"seed {seed}: {group} split over {homes}")


class LoadBenchmarkGroupTests(unittest.TestCase):
    def load(self, groups):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "groups.json"
            path.write_text(json.dumps({"groups": groups}))
            return load_benchmark_groups(path)

    def test_plain_lists_and_objects_both_load(self):
        groups = self.load({
            "plain": ["b", "a"],
            "documented": {"members": ["d", "c"], "reasons": ["shared_source"], "evidence": "same run"},
        })
        self.assertEqual(groups, [["a", "b"], ["c", "d"]])

    def test_object_without_evidence_or_valid_reasons_raises(self):
        for group in (
            {"members": ["a"], "reasons": ["shared_source"]},
            {"members": ["a"], "reasons": [], "evidence": "x"},
            {"members": ["a"], "reasons": ["same_prefix"], "evidence": "x"},
        ):
            with self.assertRaises(ValueError):
                self.load({"g": group})


if __name__ == "__main__":
    unittest.main()
