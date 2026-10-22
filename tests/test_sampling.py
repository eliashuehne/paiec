import unittest
from unittest.mock import patch

import pandas as pd

from paiec.pairs import Pair, balanced_pairs, choose_pairs
from paiec.schema import BenchTables
from scripts.run_eval import resolve_seed, sample_manifest


def tables(n_subjects=30):
    subjects = pd.DataFrame({"subject_id": [f"s{i:02d}" for i in range(n_subjects)]})
    subjects = subjects.set_index("subject_id", drop=False)
    ids = [f"i{i}" for i in range(200)]
    items = pd.DataFrame({"item_id": ids, "content": ids, "item_features": ""}).set_index("item_id", drop=False)
    responses = pd.DataFrame([
        {"subject_id": subject, "item_id": item, "label": position % 2, "interactors": ""}
        for subject in subjects.index for position, item in enumerate(ids)
    ])
    return BenchTables("toy", subjects, items, responses)


class SamplingTests(unittest.TestCase):
    def test_random_subject_selection_repeats_with_seed_and_changes_with_new_seed(self):
        source = tables()
        def ids(seed):
            return [pair.subject_id for pair in choose_pairs(source, max_subjects=10, min_items=200, randomize=True, seed=seed)]
        self.assertEqual(ids(3), ids(3))
        self.assertNotEqual(ids(3), ids(4))
        self.assertEqual(len(set(ids(3))), 10)
        self.assertNotEqual(ids(3), list(source.subjects.index[:10]))

    def test_first_selection_preserves_existing_order(self):
        self.assertEqual([pair.subject_id for pair in choose_pairs(tables(), max_subjects=3)], ["s00", "s01", "s02"])

    def test_balancing_redistributes_sparse_benchmark_without_replacement(self):
        candidates = [Pair(name, f"s{i}", {}, pd.DataFrame(), pd.DataFrame(), ())
                      for name, count in (("a", 2), ("b", 20), ("c", 20)) for i in range(count)]
        selected = balanced_pairs(candidates, 20, seed=7)
        identities = [(pair.benchmark, pair.subject_id) for pair in selected]
        self.assertEqual(len(set(identities)), 20)
        counts = pd.Series([pair.benchmark for pair in selected]).value_counts().to_dict()
        self.assertEqual(counts, {"b": 9, "c": 9, "a": 2})
        self.assertEqual(identities, [(p.benchmark, p.subject_id) for p in balanced_pairs(candidates, 20, seed=7)])
        with self.assertRaisesRegex(ValueError, "only 42"):
            balanced_pairs(candidates, 43, seed=7)

    def test_manifest_is_reproducible_disjoint_and_has_exact_evaluation_count(self):
        pairs = choose_pairs(tables(), max_subjects=5, randomize=True, min_items=200, seed=8)
        suite = {"seed": 8, "evaluation_items": 100, "acquisition_items": 40}
        manifest = sample_manifest(pairs, suite)
        self.assertEqual(int(manifest.role.eq("evaluation").sum()), 500)
        self.assertEqual(int(manifest.role.eq("acquisition").sum()), 200)
        self.assertFalse(manifest.duplicated(["benchmark", "subject_id", "item_id"]).any())
        self.assertEqual(manifest.to_csv(index=False), sample_manifest(pairs, suite).to_csv(index=False))
        self.assertNotEqual(manifest.to_csv(index=False), sample_manifest(pairs, dict(suite, seed=9)).to_csv(index=False))

    def test_fresh_and_explicit_seed_resolution(self):
        with patch("scripts.run_eval.secrets.randbits", return_value=123):
            self.assertEqual(resolve_seed(None, None), 123)
            self.assertEqual(resolve_seed(0, "random"), 123)
        self.assertEqual(resolve_seed(0, None), 0)
        self.assertEqual(resolve_seed(None, "456"), 456)
