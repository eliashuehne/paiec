import unittest

import numpy as np
import pandas as pd

from paiec.lift import fit_predict, fold_splits, join_features, log_loss, run_lift
from paiec.lift_sample import SEALED

FOLDS = [str(k) for k in range(5)] * 5 + [SEALED] * 5


def synthetic_sample(rows_per_benchmark=200, seed=0):
    """30 benchmarks, 25 in folds and 5 sealed; ``signal`` drives the label, ``noise`` does not."""
    rng = np.random.default_rng(seed)
    parts = []
    for number, fold in enumerate(FOLDS):
        signal = rng.normal(size=rows_per_benchmark)
        label = (rng.random(rows_per_benchmark) < 1 / (1 + np.exp(-2 * signal))).astype(int)
        parts.append(pd.DataFrame({
            "benchmark": f"b{number:02d}",
            "subject_id": [f"s{k % 10}" for k in range(rows_per_benchmark)],
            "item_id": [f"b{number:02d}_i{k}" for k in range(rows_per_benchmark)],
            "trial": 1,
            "label": label,
            "fold": fold,
            "signal": signal,
            "noise": rng.normal(size=rows_per_benchmark),
        }))
    return pd.concat(parts, ignore_index=True)


def mean_brier(result):
    return result.folds["brier"].mean()


class FoldSplitTests(unittest.TestCase):
    def test_train_and_test_benchmarks_never_overlap(self):
        for _fold, train, test in fold_splits(synthetic_sample()):
            self.assertFalse(set(train["benchmark"]) & set(test["benchmark"]))

    def test_each_open_benchmark_is_tested_exactly_once(self):
        sample = synthetic_sample()
        tested = [name for _, _, test in fold_splits(sample) for name in test["benchmark"].unique()]
        self.assertEqual(sorted(tested), sorted(sample.loc[sample["fold"] != SEALED, "benchmark"].unique()))

    def test_sealed_rows_are_never_used(self):
        for _fold, train, test in fold_splits(synthetic_sample()):
            self.assertNotIn(SEALED, set(train["fold"]) | set(test["fold"]))

    def test_sealed_labels_do_not_change_the_scores(self):
        sample = synthetic_sample()
        flipped = sample.copy()
        sealed = flipped["fold"] == SEALED
        flipped.loc[sealed, "label"] = 1 - flipped.loc[sealed, "label"]
        pd.testing.assert_frame_equal(run_lift(sample, ["signal"]).folds, run_lift(flipped, ["signal"]).folds)


class RunLiftTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sample = synthetic_sample()
        cls.floor = run_lift(cls.sample)

    def test_signal_column_lowers_brier(self):
        self.assertLess(mean_brier(run_lift(self.sample, ["signal"])), mean_brier(self.floor) - 0.03)

    def test_noise_column_changes_brier_by_about_zero(self):
        self.assertLess(abs(mean_brier(run_lift(self.sample, ["noise"])) - mean_brier(self.floor)), 0.01)

    def test_same_inputs_give_identical_scores(self):
        first = run_lift(self.sample, ["signal", "noise"])
        second = run_lift(self.sample, ["signal", "noise"])
        pd.testing.assert_frame_equal(first.folds, second.folds)
        pd.testing.assert_frame_equal(first.benchmarks, second.benchmarks)

    def test_result_shape(self):
        result = run_lift(self.sample, ["signal"])
        self.assertEqual(result.columns, ("signal",))
        self.assertEqual(result.folds["fold"].tolist(), ["0", "1", "2", "3", "4"])
        self.assertEqual(result.folds["benchmarks"].tolist(), [5] * 5)
        self.assertEqual(len(result.benchmarks), 25)
        self.assertEqual(result.folds["rows"].sum(), result.benchmarks["rows"].sum())

    def test_no_columns_predicts_the_training_rate(self):
        train = self.sample.iloc[:100]
        p = fit_predict(train, self.sample.iloc[100:110], [])
        np.testing.assert_allclose(p, train["label"].mean())

    def test_text_column_works_with_unseen_categories(self):
        sample = self.sample.assign(kind=np.where(self.sample["signal"] > 0, "high", "low"))
        sample.loc[sample["fold"] == "0", "kind"] = "never_seen_in_training"
        result = run_lift(sample, ["kind"])
        self.assertTrue(np.isfinite(result.folds["brier"]).all())

    def test_unknown_column_raises(self):
        with self.assertRaises(ValueError):
            run_lift(self.sample, ["missing"])


class JoinFeatureTests(unittest.TestCase):
    def setUp(self):
        self.sample = synthetic_sample(rows_per_benchmark=4)

    def test_item_and_subject_tables_join_by_their_keys(self):
        items = pd.DataFrame({"item_id": self.sample["item_id"].unique()}).assign(n_chars=7)
        subjects = pd.DataFrame({"subject_id": ["s0", "s1"], "provider": ["a", "b"]})
        joined = join_features(self.sample, [items, subjects])
        self.assertEqual(len(joined), len(self.sample))
        self.assertTrue((joined["n_chars"] == 7).all())
        self.assertEqual(joined.loc[joined["subject_id"] == "s1", "provider"].unique().tolist(), ["b"])
        self.assertTrue(joined.loc[joined["subject_id"] == "s2", "provider"].isna().all())

    def test_duplicate_keys_raise(self):
        with self.assertRaises(ValueError):
            join_features(self.sample, [pd.DataFrame({"subject_id": ["s0", "s0"], "x": [1, 2]})])

    def test_clashing_column_raises(self):
        with self.assertRaises(ValueError):
            join_features(self.sample, [pd.DataFrame({"subject_id": ["s0"], "label": [1]})])

    def test_table_without_a_key_raises(self):
        with self.assertRaises(ValueError):
            join_features(self.sample, [pd.DataFrame({"x": [1]})])


class LogLossTests(unittest.TestCase):
    def test_known_value_and_clipping(self):
        self.assertAlmostEqual(log_loss([1, 0], [0.5, 0.5]), np.log(2))
        self.assertTrue(np.isfinite(log_loss([1], [0.0])))


if __name__ == "__main__":
    unittest.main()
