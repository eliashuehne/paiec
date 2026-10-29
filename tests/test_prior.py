import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from paiec.prior import LogisticPrior, benchmark_rows

SUBJECT_COLUMNS = ["normalized_name", "provider", "release_date", "access_date", "harness",
                   "harness_version", "reasoning_effort", "subject_features_extra", "display_name"]


def write_benchmark(data_root, name, n_subjects=3, n_items=100, seed=0):
    """Items mentioning 'easy' are mostly solved, items mentioning 'hard' mostly not."""
    rng = np.random.default_rng(seed)
    target = Path(data_root) / name / "formatted_tables"
    target.mkdir(parents=True)
    item_ids = [f"{name}-{k}" for k in range(n_items)]
    kinds = ["easy" if k % 2 else "hard" for k in range(n_items)]
    pd.DataFrame({"item_id": item_ids, "content": [f"{kind} question {k}" for k, kind in enumerate(kinds)],
                  "item_features": [""] * n_items}).to_parquet(target / "items.parquet")
    subjects = pd.DataFrame({"subject_id": [f"{name}-s{s}" for s in range(n_subjects)]})
    for column in SUBJECT_COLUMNS:
        subjects[column] = ""
    subjects.to_parquet(target / "subjects.parquet")
    rows = [
        {"subject_id": s, "item_id": i, "trial": 1, "interactors": "",
         "response": float(rng.random() < (0.9 if kind == "easy" else 0.1))}
        for s in subjects["subject_id"] for i, kind in zip(item_ids, kinds)
    ]
    pd.DataFrame(rows).to_parquet(target / "responses.parquet")


ITEM = {"item_features": "", "interactors": "", "benchmark_id": "new"}
SUBJECT = {field: "" for field in SUBJECT_COLUMNS[:-1]}


class BenchmarkRowTests(unittest.TestCase):
    def test_each_pair_has_total_weight_one_and_rows_are_cached(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "data"
            write_benchmark(data, "toy")
            rows = benchmark_rows(data, "toy", items_per_pair=30)
            self.assertEqual(rows["labels"].size, 3 * 30)
            self.assertAlmostEqual(rows["weights"].sum(), 3.0)
            again = benchmark_rows(data, "toy", items_per_pair=30)
            self.assertTrue(np.array_equal(rows["indices"], again["indices"]))
            self.assertTrue(any((Path(tmp) / "cache").rglob("toy_*.npz")))


class LogisticPriorTests(unittest.TestCase):
    def test_learns_that_easy_items_are_easier_and_ignores_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "data"
            names = [f"bench{k}" for k in range(6)]
            for k, name in enumerate(names):
                write_benchmark(data, name, seed=k)
            model = LogisticPrior().fit(data, names)
            easy = model(SUBJECT, dict(ITEM, item_content="easy question 999"), [])
            hard = model(SUBJECT, dict(ITEM, item_content="hard question 999"), [])
            self.assertGreater(easy, 0.7)
            self.assertLess(hard, 0.3)
            labeled = [[[SUBJECT, dict(ITEM, item_content="x")], 1]] * 5
            self.assertEqual(model(SUBJECT, dict(ITEM, item_content="hard question 999"), labeled), hard)


if __name__ == "__main__":
    unittest.main()
