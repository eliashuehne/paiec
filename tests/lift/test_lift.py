import unittest

import numpy as np
import pandas as pd

from paiec.lift import (
    BASELINES,
    fit_predict,
    fold_splits,
    group_units,
    in_fold_ability,
    join_features,
    log_loss,
    run_lift,
)
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
        for _, train, test in fold_splits(synthetic_sample()):
            self.assertFalse(set(train["benchmark"]) & set(test["benchmark"]))

    def test_each_open_benchmark_is_tested_exactly_once(self):
        sample = synthetic_sample()
        tested = [name for _, _, test in fold_splits(sample) for name in test["benchmark"].unique()]
        self.assertEqual(sorted(tested), sorted(sample.loc[sample["fold"] != SEALED, "benchmark"].unique()))

    def test_sealed_rows_are_never_used(self):
        for _, train, test in fold_splits(synthetic_sample()):
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
        self.assertEqual(result.abilities, ())
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


def skill_sample(seed=0):
    """Systems whose strength carries across benchmarks; one in four has no name."""
    rng = np.random.default_rng(seed)
    skill = rng.normal(scale=1.5, size=40)
    parts = []
    for number, fold in enumerate(FOLDS):
        system = rng.integers(0, 40, size=200)
        label = (rng.random(200) < 1 / (1 + np.exp(-skill[system]))).astype(int)
        parts.append(pd.DataFrame({
            "benchmark": f"b{number:02d}",
            "subject_id": [f"b{number:02d}_m{k}" for k in system],
            "item_id": [f"b{number:02d}_i{k}" for k in range(200)],
            "trial": 1,
            "label": label,
            "fold": fold,
            "normalized_name": [f"m{k}" if k % 4 else None for k in system],
        }))
    return pd.concat(parts, ignore_index=True)


def no_skill_sample(seed=0):
    """Benchmarks differ a lot in difficulty, systems not at all; each benchmark has 4 systems."""
    rng = np.random.default_rng(seed)
    parts = []
    for number, fold in enumerate(FOLDS):
        difficulty = rng.normal(scale=2)
        system = rng.choice(40, size=4, replace=False)[rng.integers(0, 4, size=1000)]
        parts.append(pd.DataFrame({
            "benchmark": f"b{number:02d}",
            "subject_id": [f"m{k}" for k in system],
            "item_id": [f"b{number:02d}_i{k}" for k in range(1000)],
            "trial": 1,
            "label": (rng.random(1000) < 1 / (1 + np.exp(-difficulty))).astype(int),
            "fold": fold,
            "normalized_name": [f"m{k}" for k in system],
        }))
    return pd.concat(parts, ignore_index=True)


class GroupUnitTests(unittest.TestCase):
    def test_grouped_benchmarks_share_a_unit(self):
        self.assertEqual(group_units([["y", "x"], ["z"]]), {"x": "x+y", "y": "x+y", "z": "z"})

    def test_benchmark_in_two_groups_raises(self):
        with self.assertRaises(ValueError):
            group_units([["x", "y"], ["y"]])


class InFoldAbilityTests(unittest.TestCase):
    # Benchmark rates: x = 1, y = 1/3, z = 1. Excess over them:
    # x: a 0, a 0;  y: a -1/3, b +2/3, b -1/3;  z: (no name) 0.
    TRAIN = pd.DataFrame({
        "benchmark": ["x", "x", "y", "y", "y", "z"],
        "name": ["a", "a", "a", "b", "b", None],
        "label": [1, 1, 0, 1, 0, 1],
    })

    def ability(self, rows, strength=0):
        return in_fold_ability(self.TRAIN, rows, "name", strength=strength)

    def test_test_rows_use_every_training_benchmark(self):
        ability = self.ability(pd.DataFrame({"benchmark": ["new"], "name": ["a"]}))
        self.assertAlmostEqual(ability.loc[0, "name_ability"], -1 / 9)
        self.assertEqual(ability.loc[0, "name_n"], 3)

    def test_training_rows_leave_out_their_own_benchmark(self):
        ability = self.ability(self.TRAIN)
        self.assertAlmostEqual(ability["name_ability"].iloc[0], -1 / 3)  # "a" outside x: only y
        self.assertEqual(ability["name_n"].iloc[0], 1)
        self.assertAlmostEqual(ability["name_ability"].iloc[2], 0.0)  # "a" outside y: only x
        self.assertEqual(ability["name_n"].iloc[2], 2)

    def test_grouped_rows_leave_out_their_whole_group(self):
        ability = in_fold_ability(self.TRAIN, self.TRAIN, "name", strength=0, units=group_units([["x", "y"]]))
        np.testing.assert_array_equal(ability["name_n"].iloc[:5], 0)  # "a" and "b" appear only in x and y

    def test_sibling_labels_do_not_change_a_grouped_row(self):
        units = group_units([["x", "y"]])
        in_y = self.TRAIN["benchmark"] == "y"
        flipped = self.TRAIN.assign(label=np.where(in_y, 1 - self.TRAIN["label"], self.TRAIN["label"]))
        pd.testing.assert_frame_equal(
            in_fold_ability(self.TRAIN, self.TRAIN, "name", units=units).iloc[:2],
            in_fold_ability(flipped, flipped, "name", units=units).iloc[:2],
        )

    def test_duplicate_row_index_raises(self):
        with self.assertRaises(ValueError):
            self.ability(pd.DataFrame({"benchmark": ["new", "new"], "name": ["a", "b"]}, index=[0, 0]))

    def test_difficulty_is_removed(self):
        # "b" is right half the time on y, where the average is 1/3: above average.
        ability = self.ability(pd.DataFrame({"benchmark": ["new"], "name": ["b"]}))
        self.assertGreater(ability.loc[0, "name_ability"], 0)

    def test_missing_or_unseen_key_gets_zero(self):
        rows = pd.DataFrame({"benchmark": ["new", "new", "z"], "name": [None, "unseen", None]})
        for strength in (0, 20):
            ability = self.ability(rows, strength)
            np.testing.assert_array_equal(ability["name_ability"], 0)
            np.testing.assert_array_equal(ability["name_n"], 0)

    def test_smoothing_pulls_small_counts_toward_zero(self):
        ability = self.ability(pd.DataFrame({"benchmark": ["new"], "name": ["b"]}), strength=2)
        self.assertAlmostEqual(ability.loc[0, "name_ability"], (2 / 3 - 1 / 3) / (2 + 2))

    def test_changing_test_fold_labels_leaves_features_unchanged(self):
        sample = skill_sample()
        flipped = sample.assign(label=np.where(sample["fold"] == "0", 1 - sample["label"], sample["label"]))
        _, train, test = next(fold_splits(sample))
        _, flipped_train, flipped_test = next(fold_splits(flipped))
        pd.testing.assert_frame_equal(
            in_fold_ability(train, test, "normalized_name"),
            in_fold_ability(flipped_train, flipped_test, "normalized_name"),
        )

    def test_a_training_row_ignores_its_own_benchmark_labels(self):
        train = skill_sample().query("fold != 'sealed'")
        own = train["benchmark"] == "b00"
        flipped = train.assign(label=np.where(own, 1 - train["label"], train["label"]))
        pd.testing.assert_frame_equal(
            in_fold_ability(train, train, "normalized_name")[own],
            in_fold_ability(flipped, flipped, "normalized_name")[own],
        )


class BaselineTests(unittest.TestCase):
    def test_baselines_use_only_visible_keys(self):
        self.assertEqual(BASELINES["A"].columns + BASELINES["A"].abilities, ())
        self.assertEqual(BASELINES["B"].abilities, ("normalized_name",))

    def test_b_beats_a_when_system_strength_transfers(self):
        sample = skill_sample()
        a = run_lift(sample, **vars(BASELINES["A"]))
        b = run_lift(sample, **vars(BASELINES["B"]))
        self.assertEqual(b.abilities, ("normalized_name",))
        self.assertTrue((b.folds["brier"] < a.folds["brier"]).all())

    def test_b_does_not_fingerprint_benchmarks(self):
        # Each (system, benchmark) pair has ~250 rows, enough for a leaf. Exact
        # ability values and counts would let trees memorise training benchmark
        # difficulty; B must stay close to A when systems carry no signal.
        sample = no_skill_sample()
        a = mean_brier(run_lift(sample, **vars(BASELINES["A"])))
        self.assertLess(mean_brier(run_lift(sample, **vars(BASELINES["B"]))), a + 0.01)
        unconstrained = []
        for _, train, test in fold_splits(sample):
            train = train.join(in_fold_ability(train, train, "normalized_name"))
            test = test.join(in_fold_ability(train, test, "normalized_name"))
            p = fit_predict(train, test, ["normalized_name_ability", "normalized_name_n"])
            unconstrained.append(np.mean((p - test["label"]) ** 2))
        self.assertGreater(np.mean(unconstrained), a + 0.02)  # the failure the guard prevents

    def test_duplicate_sample_index_raises(self):
        sample = skill_sample()
        with self.assertRaises(ValueError):
            run_lift(pd.concat([sample, sample]), abilities=["normalized_name"])

    def test_unknown_ability_key_raises(self):
        with self.assertRaises(ValueError):
            run_lift(skill_sample(), abilities=["missing"])


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

    def test_trial_features_preserve_repeated_sample_rows_and_bookkeeping(self):
        sample = pd.DataFrame({"benchmark": ["a", "a", "a"], "subject_id": ["s"] * 3,
                               "item_id": ["i"] * 3, "trial": [1, 0, 1],
                               "label": [0, 1, 1], "fold": ["0", "sealed", "0"],
                               "pool": ["evaluate", "reveal", "evaluate"], "order": [-1, 0, -1]})
        features = pd.DataFrame({"benchmark": ["a", "a"], "subject_id": ["s"] * 2,
                                 "item_id": ["i"] * 2, "trial": [0, 1], "task_chars": [10, 20]})
        joined = join_features(sample, [features])
        pd.testing.assert_frame_equal(joined[sample.columns], sample)
        self.assertEqual(joined.task_chars.tolist(), [20, 10, 20])

    def test_null_feature_keys_and_trial_without_entity_key_raise(self):
        for table in [pd.DataFrame({"subject_id": [None], "x": [1]}),
                      pd.DataFrame({"trial": [0], "x": [1]})]:
            with self.subTest(columns=list(table)), self.assertRaises(ValueError):
                join_features(self.sample, [table])

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
