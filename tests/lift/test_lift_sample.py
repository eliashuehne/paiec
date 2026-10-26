import unittest

import numpy as np
import pandas as pd

from paiec.lift_sample import (
    COLUMNS,
    SEALED,
    assign_folds,
    build_sample,
    fold_labels,
    sample_benchmark,
    summarise_sample,
)


def responses(n, prefix="s"):
    labels = np.resize([1.0, 0.0, np.nan], n)
    return pd.DataFrame({
        "subject_id": [f"{prefix}{k % 7}" for k in range(n)],
        "item_id": [f"i{k}" for k in range(n)],
        "trial": 1,
        "label": labels,
        "interactors": "",
    })


class FoldLabelTests(unittest.TestCase):
    def test_maps_folds_and_sealed(self):
        labels = fold_labels(["c"], [(["b"], ["a"]), (["a"], ["b"])])
        self.assertEqual(labels, {"a": "0", "b": "1", "c": SEALED})

    def test_benchmark_in_two_places_raises(self):
        with self.assertRaises(ValueError):
            fold_labels(["a"], [([], ["a"])])


class SampleBenchmarkTests(unittest.TestCase):
    def test_cap_holds(self):
        self.assertEqual(len(sample_benchmark(responses(300), "b", cap=50)), 50)

    def test_small_benchmark_keeps_every_labelled_row(self):
        self.assertEqual(len(sample_benchmark(responses(30), "b", cap=1000)), 20)

    def test_no_empty_labels_and_labels_are_ints(self):
        sample = sample_benchmark(responses(300), "b", cap=1000)
        self.assertTrue(sample["label"].isin([0, 1]).all())
        self.assertEqual(sample["label"].dtype.kind, "i")

    def test_columns(self):
        sample = sample_benchmark(responses(10), "b", cap=5)
        self.assertEqual(list(sample.columns), COLUMNS[:-1])
        self.assertEqual(set(sample["benchmark"]), {"b"})

    def test_same_seed_same_rows_different_seed_different_rows(self):
        first = sample_benchmark(responses(500), "b", cap=50, seed=0)
        self.assertTrue(first.equals(sample_benchmark(responses(500), "b", cap=50, seed=0)))
        other = sample_benchmark(responses(500), "b", cap=50, seed=1)
        self.assertNotEqual(set(first["item_id"]), set(other["item_id"]))

    def test_invalid_cap_raises(self):
        with self.assertRaises(ValueError):
            sample_benchmark(responses(10), "b", cap=0)


class BuildSampleTests(unittest.TestCase):
    FOLDS = {"a": "0", "b": "1", "c": SEALED}

    def tables(self):
        return [(name, responses(90, name)) for name in self.FOLDS]

    def test_each_benchmark_has_exactly_one_fold(self):
        sample = assign_folds(build_sample(self.tables(), cap=40), self.FOLDS)
        self.assertEqual(list(sample.columns), COLUMNS)
        self.assertTrue((sample.groupby("benchmark")["fold"].nunique() == 1).all())
        self.assertEqual(sample.loc[sample["benchmark"] == "c", "fold"].unique().tolist(), [SEALED])

    def test_benchmark_without_labels_is_dropped(self):
        empty = responses(30).assign(label=np.nan)
        sample = build_sample(self.tables() + [("empty", empty)], cap=40)
        self.assertNotIn("empty", set(sample["benchmark"]))

    def test_draw_does_not_depend_on_loading_order(self):
        forward = build_sample(self.tables(), cap=40)
        backward = build_sample(reversed(self.tables()), cap=40)
        key = ["benchmark", "item_id"]
        self.assertTrue(
            forward.sort_values(key).reset_index(drop=True).equals(
                backward.sort_values(key).reset_index(drop=True)
            )
        )

    def test_benchmark_without_fold_raises(self):
        with self.assertRaises(ValueError):
            assign_folds(build_sample([("unknown", responses(10))], cap=5), self.FOLDS)

    def test_summary_counts(self):
        summary = summarise_sample(assign_folds(build_sample(self.tables(), cap=40), self.FOLDS))
        self.assertEqual(summary["rows"], 120)
        self.assertEqual(summary["benchmarks"], 3)
        self.assertEqual(summary["benchmarks_by_fold"], {"0": 1, "1": 1, SEALED: 1})


if __name__ == "__main__":
    unittest.main()
